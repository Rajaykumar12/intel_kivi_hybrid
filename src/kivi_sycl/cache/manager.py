"""KiviCache: the KIVI 2-bit asymmetric KV cache state manager."""

from typing import List, Optional, Tuple

import torch
from transformers.cache_utils import DynamicCache

from ..config.model_config import detect_model_config
from .arena import LayerArena, ensure_capacity
from ..extension.loader import kivi_native

__all__ = ["KiviCache"]

# Unpack shift table for the LSB-first 2-bit packing
# (byte = q0 | q1<<2 | q2<<4 | q3<<6) — see the SYCL pack_vec kernel.
_UNPACK_SHIFTS = torch.tensor([0, 2, 4, 6], dtype=torch.uint8)


def _iter_kv_layers(past_key_values):
    """
    Yield (key, value) tensors per layer from a model's `past_key_values`,
    regardless of whether it's the legacy tuple-of-(k,v)-tuples format or a
    `transformers.Cache`/`DynamicCache` object — the concrete Cache
    implementation has changed shape across transformers versions (a
    `key_cache`/`value_cache` pair of lists in some, a `.layers` list of
    per-layer objects with `.keys`/`.values` in others), so this checks for
    each known surface in turn instead of assuming tuple-of-tuples.
    """
    if hasattr(past_key_values, "layers"):
        for layer in past_key_values.layers:
            yield layer.keys, layer.values
        return
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        for k, v in zip(past_key_values.key_cache, past_key_values.value_cache):
            yield k, v
        return
    for item in past_key_values:
        yield item[0], item[1]


def _build_cache(layers):
    """
    Wrap a list of (key, value) tensors per layer into a
    `transformers.Cache` object. Recent transformers model forward passes
    (see `modeling_gpt2.py`'s `past_key_values.get_seq_length()`) require
    `past_key_values` to be a `Cache` instance, not the legacy tuple format
    — `Cache.update()` is the stable, version-tolerant construction API
    (unlike `DynamicCache.from_legacy_cache`, which has moved/disappeared
    across transformers releases).
    """
    cache = DynamicCache()
    for i, (k, v) in enumerate(layers):
        cache.update(k, v, i)
    return cache


class KiviCache:
    """
    KIVI 2-bit asymmetric KV cache manager.

    Plug-and-play with any HuggingFace CausalLM that returns
    ``past_key_values`` as a tuple of (key, value) tensors or a
    `transformers.Cache`.

    Typical shapes: key/value = [batch, num_kv_heads, seq_len, head_dim]

    Storage model (packed-resident): flushed tokens live as their real
    2-bit packed representation (uint8 payload + per-group scales/zeros)
    on CPU. The most recent ``residual_length`` tokens stay in full
    precision. ``get_full_cache()`` reconstructs the full-precision view
    on demand; how long that reconstruction is retained is governed by
    ``serving``:

    - ``serving="rebuild"`` (default): no full-precision copy of the
      history is retained between steps — the resident cache footprint is
      packed + residual, and every ``get_full_cache()`` after a mutation
      dequantizes the history (vectorized torch ops, memory-bandwidth
      bound). This is the paper-faithful memory mode.
    - ``serving="cached"``: a dequantized FP copy of the history is kept
      on CPU and the served Cache view persists across steps (O(1) per
      non-flush step). Fastest, but the resident footprint is roughly the
      FP baseline *plus* the packed store — use only when speed matters
      more than memory.

    Flush execution: quantization kernels are submitted to the XPU
    asynchronously during ``add_tokens`` and harvested at the start of the
    *next* ``add_tokens`` call, so kernel + transfer time overlaps the
    model's forward pass. Until harvest, the affected tokens remain served
    from the FP residual, so correctness never depends on kernel timing.
    Pass ``async_flush=False`` to harvest immediately (blocking, simpler
    timing for benchmarks).

    Packed storage is a preallocated per-layer arena (``cache/arena.py``):
    a flush appends into spare capacity instead of re-concatenating the whole
    history. With ``zero_copy`` (default on XPU) the arena is USM host
    memory, which the quantize kernels write directly and the CPU dequant
    path reads in place — no host<->device copies. Call ``reserve(n)`` with
    the final sequence length when it is known (``generate()`` does) so the
    arena is sized once and carries no spare tail; otherwise it grows by
    max(4 groups, 25%). ``get_stats()`` counts the full allocation.

    Element dtype follows the model: fp32 and fp16 KV tensors are stored
    and quantized natively; bf16 is upcast to fp32 at ingestion (the
    kernels have no bf16 instantiation). The served cache always has the
    model's dtype.

    Metadata dtype (per-group scales/zeros) defaults to fp16 for every
    model. At G=32 this lifts the compression ceiling against an fp32
    baseline from 8x to 10.67x: fp32 metadata costs as much as the 2-bit
    payload itself. The narrowing happens inside the quantize kernel, so it
    never goes through torch's XPU dtype cast. Pass
    ``metadata_dtype=torch.float32`` for the previous behaviour (fp32 models
    only — fp32 metadata on fp16 data has no kernel instantiation).

    Args:
        num_layers:       Number of transformer layers.
        head_dim:         Dimension per attention head.
        num_kv_heads:     Number of KV heads (optional, auto-detected on first call).
        device:           Device for quantization kernels ("xpu").
        group_size:       Quantization group size G — an int, or a list with one G
                          per layer. Each must divide head_dim, be a multiple of 4
                          and not exceed residual_length.
        residual_length:  Number of recent full-precision tokens to keep (R in paper).
        serving:          "rebuild" (memory-honest) or "cached" (speed) — see above.
        async_flush:      Overlap flush kernels with the next forward pass.
        metadata_dtype:   "auto" / torch.float16 (default: fp16 scales/zeros)
                          or torch.float32 (fp32 models only).
        zero_copy:        "auto" (on when device is XPU and the extension
                          provides USM allocation), True, or False (stage
                          through XPU tensors and copy back at harvest).
    """

    def __init__(
        self,
        num_layers: int,
        head_dim: int,
        num_kv_heads: Optional[int] = None,
        device: str = "xpu",
        group_size=32,
        residual_length: int = 128,
        serving: str = "rebuild",
        async_flush: bool = True,
        metadata_dtype="auto",
        zero_copy="auto",
    ):
        if isinstance(group_size, int):
            group_sizes = [group_size] * num_layers
        else:
            group_sizes = [int(g) for g in group_size]
            assert len(group_sizes) == num_layers, (
                f"group_size list has {len(group_sizes)} entries for {num_layers} layers")
        for g in group_sizes:
            assert head_dim % g == 0, \
                f"head_dim ({head_dim}) must be divisible by group_size ({g})"
            assert g >= 4 and g % 4 == 0, \
                f"group_size ({g}) must be >= 4 and divisible by 4"
            assert g <= residual_length, (
                f"group_size ({g}) must not exceed residual_length "
                f"({residual_length}) — flushing requires a full group of tokens "
                f"to be available in the residual buffer, or tokens are silently dropped."
            )
        assert serving in ("rebuild", "cached"), \
            f"serving must be 'rebuild' or 'cached', got {serving!r}"
        assert zero_copy in ("auto", True, False), \
            f"zero_copy must be 'auto', True or False, got {zero_copy!r}"
        assert metadata_dtype in ("auto", torch.float16, torch.float32), (
            f"metadata_dtype must be 'auto', torch.float16 or torch.float32, "
            f"got {metadata_dtype!r}")

        self.device = device
        # Per-layer group sizes; self.G is the shared value when uniform (the
        # common case, and what older code reads), None when layers differ.
        self._G: List[int] = group_sizes
        self.G = group_sizes[0] if len(set(group_sizes)) == 1 else None
        self.R = residual_length
        self.head_dim = head_dim
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.serving = serving
        self.async_flush = async_flush
        self.metadata_dtype = metadata_dtype
        self.zero_copy = zero_copy
        self.batch_size: Optional[int] = None
        self._cache_dtype: Optional[torch.dtype] = None
        # Resolved on the first add_tokens, once the model's dtype is known.
        self._meta_dtype: Optional[torch.dtype] = None

        # FP residual buffers on CPU (model dtype)
        self.res_keys: List[Optional[torch.Tensor]] = [None] * num_layers
        self.res_values: List[Optional[torch.Tensor]] = [None] * num_layers

        # Packed-resident history: one preallocated arena per layer (see
        # cache/arena.py). Keys channel-major [B,H,D,cap/4] + [B,H,D,cap/G],
        # values token-major [B,H,cap,D/4] + [B,H,cap,D/G]. The first
        # `arena.used` tokens are valid; `used` is always a multiple of G.
        # The `_packed_k` ... `_zeros_v` properties expose the used region.
        self._arena: List[Optional[LayerArena]] = [None] * num_layers
        # Reusable USM staging buffers for flush inputs (zero_copy only),
        # one pair per layer: every layer can have a flush in flight at once.
        self._stage: List[Optional[dict]] = [None] * num_layers
        # Final sequence length hint from reserve(); 0 = unknown.
        self._reserve_tokens = 0

        # In-flight flush per layer: kernels submitted, outputs not yet
        # harvested. The record keeps the XPU *input* tensors alive too —
        # freeing a tensor a submitted kernel still reads is undefined
        # behavior. Tokens covered by a pending flush are still present in
        # the residual (they are trimmed at harvest), so serving stays
        # correct regardless of kernel timing.
        self._pending: List[Optional[dict]] = [None] * num_layers

        # Dequantized FP serving copy of the history (serving="cached" only)
        self._deq_serving: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = \
            [None] * num_layers

        self._quant_token_count: List[int] = [0] * num_layers
        self._flush_count: List[int] = [0] * num_layers

        # Persistent Cache object returned by get_full_cache(). In
        # serving="cached" mode it is reused across non-flush steps (the
        # model grows it in place, matching the O(1)-per-step cost a plain
        # HuggingFace cache pays). In serving="rebuild" mode it is dropped
        # after every add_tokens so no FP copy of the history stays
        # resident between steps.
        self._cache_view: Optional[DynamicCache] = None
        self._cache_view_valid: bool = False

    # ------------------------------------------------------------------ #
    #  Factory                                                             #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_model(
        cls,
        model,
        device: str = "xpu",
        group_size=32,
        residual_length: int = 128,
        serving: str = "rebuild",
        async_flush: bool = True,
        metadata_dtype="auto",
        zero_copy="auto",
    ) -> "KiviCache":
        """
        Create a KiviCache auto-configured for a HuggingFace model.

        >>> cache = KiviCache.from_model(model)
        """
        cfg = detect_model_config(model)
        return cls(
            num_layers=cfg["num_layers"],
            head_dim=cfg["head_dim"],
            num_kv_heads=cfg["num_kv_heads"],
            device=device,
            group_size=group_size,
            residual_length=residual_length,
            serving=serving,
            async_flush=async_flush,
            metadata_dtype=metadata_dtype,
            zero_copy=zero_copy,
        )

    # ------------------------------------------------------------------ #
    #  Core API                                                            #
    # ------------------------------------------------------------------ #

    def reserve(self, total_tokens: int):
        """Size the packed arenas for a sequence of ``total_tokens`` tokens
        (prompt + everything that will be generated). Optional: without it
        the arenas grow on demand. With it they are allocated once at the
        exact final size, so no growth copy ever runs and the resident
        footprint carries no spare capacity. Call before the first
        ``add_tokens`` of a sequence; ``reset()`` clears it."""
        assert total_tokens >= 0
        self._reserve_tokens = total_tokens

    def add_tokens(self, past_key_values):
        """
        Ingest new KV pairs from a model forward pass.

        Args:
            past_key_values: the model's past_key_values from a forward
                pass — either the legacy tuple of (key, value) per layer, or
                a transformers `Cache`/`DynamicCache` object (see
                `_iter_kv_layers`). Each tensor: [B, H, Seq, D].

        Internally appends to the FP residual. When the residual reaches
        ``R`` tokens, the excess (largest multiple of ``G`` that leaves at
        least ``R - G`` tokens behind) is quantized to the packed 2-bit
        store in a single kernel submission — one launch whether the
        excess is one group (steady-state decode) or a long prompt's
        entire grouped part (bulk prefill).
        """
        # get_full_cache() hands out `self._cache_view` and keeps it marked
        # valid on the assumption that the model grew that exact object in
        # place during the forward pass. That only holds when the model
        # returns the identical Cache instance it was given. Models that
        # rebuild their cache (legacy tuple returns, architectures not yet
        # ported to the Cache API) hand back a different object — serving
        # the untouched view again would silently drop every generated
        # token, and the per-layer desync assert below cannot catch it (the
        # new-token slice just comes out empty). Object identity is the
        # exact test for "did the model grow our view in place".
        if past_key_values is not self._cache_view:
            self._cache_view_valid = False

        # Collect any flush submitted on the previous step. The XPU had the
        # entire forward pass to finish, so the event waits below are
        # effectively free — this is where the async overlap pays off.
        self.harvest_pending()

        for i, (k, v) in enumerate(_iter_kv_layers(past_key_values)):
            self._init_session(k, i)

            # Extract only NEW tokens (model returns full cache)
            if self.res_keys[i] is not None:
                existing_len = self.res_keys[i].shape[2]
                quant_len = self._quant_token_count[i]
                new_start = quant_len + existing_len
                assert k.shape[2] >= new_start, (
                    f"KiviCache desync at layer {i}: expected past length "
                    f"{new_start} but got {k.shape[2]}. add_tokens() must be "
                    f"called exactly once per forward pass, seeded by "
                    f"get_full_cache()."
                )
                new_k = k[:, :, new_start:, :].to(self._cache_dtype)
                new_v = v[:, :, new_start:, :].to(self._cache_dtype)
            else:
                new_start = 0
                new_k = k.to(self._cache_dtype)
                new_v = v.to(self._cache_dtype)

            assert new_k.shape[2] > 0, (
                f"KiviCache desync at layer {i}: the forward pass produced "
                f"no tokens beyond the {new_start} already tracked. "
                f"add_tokens() was called twice on the same forward output, "
                f"or the model was fed a past_key_values not obtained from "
                f"get_full_cache(). Silently skipping here would drop this "
                f"step's KV and corrupt all subsequent attention."
            )

            self._append_layer(i, new_k, new_v)

            # No manual append to the cache view here: when the model grew
            # `self._cache_view` in place (the identity check at the top of
            # this method), the view already contains this token — calling
            # `.update()` again would append it a second time. When the
            # model returned a different object, the view was invalidated
            # above and the next get_full_cache() rebuilds it from the
            # residual, which does include this token.

        if self.async_flush:
            # Pending flushes are harvested at the top of the next
            # add_tokens call; until then the flushed tokens are served
            # from the FP residual.
            pass
        else:
            self.harvest_pending()

        if self.serving == "rebuild":
            # Drop the FP serving view — the resident footprint between
            # steps is packed history + residual only.
            self._cache_view = None
            self._cache_view_valid = False

    def _init_session(self, k: torch.Tensor, layer_idx: int):
        """Record batch size / KV heads / dtypes from the first tensor seen,
        and check later tensors against them."""
        if self.batch_size is None:
            self.batch_size = k.shape[0]
            self.num_kv_heads = k.shape[1]
        else:
            assert k.shape[0] == self.batch_size, (
                f"KiviCache batch size changed mid-session "
                f"({self.batch_size} -> {k.shape[0]}) at layer {layer_idx}. "
                f"Call reset() before starting a new request/batch."
            )
        if self._cache_dtype is None:
            # bf16 has no kernel instantiation — upcast to fp32.
            # fp32/fp16 are stored and quantized natively.
            self._cache_dtype = (
                k.dtype if k.dtype in (torch.float32, torch.float16)
                else torch.float32
            )
            self._meta_dtype = self._resolve_meta_dtype(self._cache_dtype)

    def _append_layer(self, i: int, new_k: torch.Tensor, new_v: torch.Tensor):
        """Append new tokens ([B,H,n,D], cache dtype) to layer i's FP residual
        and submit a flush of the excess if the residual reached R.

        The flush goes out in ONE kernel submission. The target leaves the
        residual in [R-G, R-1] tokens, but a long prompt's entire grouped
        part goes to the XPU in a single launch instead of one per G tokens.
        Flushed tokens stay in the residual (served in FP) until harvest.
        """
        self._append_residual(i, new_k, new_v)
        self._maybe_flush(i)

    def _append_residual(self, i: int, new_k: torch.Tensor, new_v: torch.Tensor):
        if self.res_keys[i] is None:
            self.res_keys[i] = new_k
            self.res_values[i] = new_v
        else:
            self.res_keys[i] = torch.cat([self.res_keys[i], new_k], dim=2)
            self.res_values[i] = torch.cat([self.res_values[i], new_v], dim=2)

    def _layer_reserve(self, i: int) -> int:
        """Final quantized-token count for layer i implied by reserve(): the
        residual keeps at least R-G tokens back, and the quantized prefix is a
        multiple of G. 0 when no reserve was given."""
        if not self._reserve_tokens:
            return 0
        G = self._G[i]
        return max(0, (self._reserve_tokens - (self.R - G)) // G * G)

    def _maybe_flush(self, i: int):
        """Submit a flush of layer i's residual excess if it reached R and no
        flush is already in flight for the layer."""
        res = self.res_keys[i]
        if res is None or self._pending[i] is not None:
            return
        res_len = res.shape[2]
        if res_len >= self.R:
            G = self._G[i]
            flush_len = ((res_len - (self.R - G)) // G) * G
            self._submit_flush(i, flush_len)

    def get_full_cache(self) -> DynamicCache:
        """
        Reconstruct the full KV cache for the next model forward pass.

        Returns:
            transformers.Cache (DynamicCache) wrapping (key, value) tensors
            per layer, on CPU. Compatible with HuggingFace
            ``past_key_values``.

        In serving="rebuild" mode this dequantizes the packed history each
        time the cache composition has changed; the returned FP tensors are
        transient (dropped on the next ``add_tokens``). In "cached" mode
        the dequantized history persists and the same view object is
        returned across non-flush steps.
        """
        if self._cache_view is not None and self._cache_view_valid:
            return self._cache_view

        full_past = []
        for i in range(self.num_layers):
            # _serve_layer writes history + residual into one buffer per
            # tensor. The residual it appends includes any tokens covered by
            # a pending (not yet harvested) flush — those are still served
            # in full precision until harvest.
            full_k, full_v = self._serve_layer(i)
            if full_k is None:
                full_k = full_v = torch.empty(0)
            full_past.append((full_k, full_v))

        self._cache_view = _build_cache(full_past)
        self._cache_view_valid = True
        return self._cache_view

    def harvest_pending(self):
        """
        Collect all in-flight flushes: wait for their kernels, move the
        packed representation to the CPU store, and trim the residual.
        Called automatically at the start of every ``add_tokens``; call it
        manually before reading ``get_stats()`` if you need the packed
        store fully up to date at an arbitrary point.
        """
        for i in range(self.num_layers):
            self._harvest_layer(i)

    def _harvest_layer(self, i: int):
        """Collect layer i's in-flight flush, if any (see harvest_pending)."""
        p = self._pending[i]
        if p is None:
            return
        p["ev_k"].wait()
        p["ev_v"].wait()
        n = p["n"]
        arena = self._arena[i]
        t0 = arena.used

        if not p["direct"]:
            # Staged path: kernels wrote XPU temporaries; copy them into
            # the arena's spare capacity (one D2H per tensor, no concat).
            for dst, src in zip(arena.key_slices(t0, t0 + n),
                                (p["packed_k"], p["scales_k"], p["zeros_k"])):
                dst.copy_(src)
            for dst, src in zip(arena.value_slices(t0, t0 + n),
                                (p["packed_v"], p["scales_v"], p["zeros_v"])):
                dst.copy_(src)
        # Direct (zero-copy) path: the kernels already wrote the arena.
        arena.used = t0 + n

        if self.serving == "cached":
            new_k = self._dequant_keys_cpu(*arena.key_slices(t0, t0 + n))
            new_v = self._dequant_values_cpu(*arena.value_slices(t0, t0 + n))
            cur = self._deq_serving[i]
            if cur is None:
                self._deq_serving[i] = (new_k, new_v)
            else:
                self._deq_serving[i] = (
                    torch.cat([cur[0], new_k], dim=2),
                    torch.cat([cur[1], new_v], dim=2),
                )

        # Trim the flushed tokens from the residual only now — until
        # this point they were still being served in full precision.
        self.res_keys[i] = self.res_keys[i][:, :, n:, :].contiguous()
        self.res_values[i] = self.res_values[i][:, :, n:, :].contiguous()

        self._quant_token_count[i] += n
        self._flush_count[i] += 1
        self._pending[i] = None
        # History composition changed (FP residual tokens replaced by
        # their quantized representation) — the persistent view no
        # longer matches what should be served.
        self._cache_view_valid = False

    def reset(self):
        """Clear all cached data. Reuse the same KiviCache for a new sequence."""
        for i in range(self.num_layers):
            p = self._pending[i]
            if p is not None:
                # Never free tensors a submitted kernel may still touch.
                p["ev_k"].wait()
                p["ev_v"].wait()
            self._pending[i] = None
            self._quant_token_count[i] = 0
            self._flush_count[i] = 0
            self.res_keys[i] = None
            self.res_values[i] = None
            self._arena[i] = None
            self._stage[i] = None
            self._deq_serving[i] = None
        self.batch_size = None
        self._cache_dtype = None
        self._meta_dtype = None
        self._reserve_tokens = 0
        self._cache_view = None
        self._cache_view_valid = False

    # ------------------------------------------------------------------ #
    #  Internals                                                           #
    # ------------------------------------------------------------------ #

    def _submit_flush(self, layer_idx: int, flush_len: int):
        """Quantize the oldest ``flush_len`` residual tokens to the packed
        2-bit representation on XPU. Kernels are submitted without waiting;
        the outputs are collected by ``harvest_pending``. The tokens stay
        in the residual (still served in FP) until harvest."""
        i = layer_idx
        G = self._G[i]
        assert self._pending[i] is None, (
            f"Layer {i} already has an in-flight flush — harvest_pending() "
            f"must run before a new flush is submitted."
        )
        assert flush_len >= G and flush_len % G == 0, (
            f"flush_len ({flush_len}) must be a positive multiple of G ({G})."
        )
        assert self.res_keys[i].shape[2] >= flush_len, (
            f"Cannot flush {flush_len} tokens: residual only has "
            f"{self.res_keys[i].shape[2]} at layer {i}."
        )

        k_chunk = self.res_keys[i][:, :, :flush_len, :]
        v_chunk = self.res_values[i][:, :, :flush_len, :]
        B, H, L, D = k_chunk.shape

        # Room for this chunk in the layer's arena. Safe to grow (reallocate)
        # here: the assert above guarantees no kernel is writing into it.
        self._arena[i] = arena = ensure_capacity(
            self._arena[i], self._quant_token_count[i] + flush_len,
            B=B, H=H, D=D, G=G, meta_dtype=self._meta_dtype,
            reserve=self._layer_reserve(i), alloc=self._host_alloc)
        t0 = arena.used

        if self._zero_copy_enabled():
            # Inputs go into reusable USM staging (keys transposed to the
            # channel-major [B,H,D,L] the per-channel kernel reads); outputs
            # are views straight into the arena. No device copies at all.
            k_in = self._staging(i, "k", (B, H, D, L), k_chunk.dtype)
            k_in.copy_(k_chunk.transpose(2, 3))
            v_in = self._staging(i, "v", (B, H, L, D), v_chunk.dtype)
            v_in.copy_(v_chunk)
            pk, sk, zk = arena.key_slices(t0, t0 + flush_len)
            pv, sv, zv = arena.value_slices(t0, t0 + flush_len)
            ev_k = kivi_native.quantize_keys_async(k_in, pk, sk, zk, G)
            ev_v = kivi_native.quantize_values_async(v_in, pv, sv, zv, D, G)
            self._pending[i] = {"n": flush_len, "direct": True,
                                "ev_k": ev_k, "ev_v": ev_v}
            return

        # Staged path: device temporaries, copied into the arena at harvest.
        k_trans = k_chunk.transpose(2, 3).contiguous().to(self.device)
        packed_k = torch.empty((B, H, D, L // 4), dtype=torch.uint8, device=self.device)
        scales_k = torch.empty((B, H, D, L // G), dtype=self._meta_dtype, device=self.device)
        zeros_k = torch.empty_like(scales_k)
        ev_k = kivi_native.quantize_keys_async(k_trans, packed_k, scales_k, zeros_k, G)

        v_dev = v_chunk.contiguous().to(self.device)
        packed_v = torch.empty((B, H, L, D // 4), dtype=torch.uint8, device=self.device)
        scales_v = torch.empty((B, H, L, D // G), dtype=self._meta_dtype, device=self.device)
        zeros_v = torch.empty_like(scales_v)
        ev_v = kivi_native.quantize_values_async(v_dev, packed_v, scales_v, zeros_v, D, G)

        self._pending[i] = {
            "n": flush_len, "direct": False,
            "ev_k": ev_k, "ev_v": ev_v,
            # Inputs are retained until harvest: the kernels read them.
            "in_k": k_trans, "in_v": v_dev,
            "packed_k": packed_k, "scales_k": scales_k, "zeros_k": zeros_k,
            "packed_v": packed_v, "scales_v": scales_v, "zeros_v": zeros_v,
        }

    # -- Arena / staging allocation ---------------------------------------- #

    def _zero_copy_enabled(self) -> bool:
        if self.zero_copy == "auto":
            return (str(self.device).startswith("xpu")
                    and hasattr(kivi_native, "alloc_usm_host"))
        return bool(self.zero_copy)

    def _host_alloc(self, shape, dtype):
        """Arena storage: USM host memory when zero-copy (kernels write it
        directly), ordinary CPU memory otherwise."""
        if self._zero_copy_enabled():
            return kivi_native.alloc_usm_host(list(shape), dtype)
        return torch.empty(shape, dtype=dtype)

    def _staging(self, layer: int, which: str, shape, dtype):
        """A contiguous USM view of `shape`, from this layer's reusable
        staging buffer (reallocated only when too small). Only called at
        submit time, when the layer has no flush in flight."""
        numel = 1
        for d in shape:
            numel *= d
        st = self._stage[layer]
        if st is None:
            st = self._stage[layer] = {}
        buf = st.get(which)
        if buf is None or buf.numel() < numel or buf.dtype != dtype:
            buf = st[which] = kivi_native.alloc_usm_host([numel], dtype)
        return buf[:numel].view(shape)

    # -- Packed store views (used region of each layer's arena) ----------- #

    class _UsedViews:
        """Lazy per-layer sequence over one tensor of the arenas' used
        region: indexing builds only that layer's view. (Building all
        layers' views on every access made get_full_cache() 4x slower.)
        Kept for introspection and tests; the hot paths use the arena."""

        def __init__(self, arenas, idx):
            self._arenas, self._idx = arenas, idx

        def __len__(self):
            return len(self._arenas)

        def __getitem__(self, i):
            a = self._arenas[i]
            if a is None or a.used == 0:
                return None
            views = a.used_keys() if self._idx < 3 else a.used_values()
            return views[self._idx % 3]

        def __iter__(self):
            return (self[i] for i in range(len(self)))

    def _arena_views(self, idx: int):
        return KiviCache._UsedViews(self._arena, idx)

    _packed_k = property(lambda self: self._arena_views(0))
    _scales_k = property(lambda self: self._arena_views(1))
    _zeros_k = property(lambda self: self._arena_views(2))
    _packed_v = property(lambda self: self._arena_views(3))
    _scales_v = property(lambda self: self._arena_views(4))
    _zeros_v = property(lambda self: self._arena_views(5))

    # -- Dequant-on-read -------------------------------------------------- #
    #
    # These are the hot path in serving="rebuild" mode: they run once per
    # layer per step, over the whole history. At TinyLlama-1.1B / 2k context
    # that measured 63 ms per step across 22 layers, which was the entire
    # -27% throughput gap against the FP baseline (238 ms/step vs 175 ms).
    #
    # They are written to minimise *allocation*, not instruction count —
    # the work is memory-bandwidth bound, so every avoided FP-sized
    # temporary is a direct win. The naive expression
    #
    #     q = ((packed.unsqueeze(-1) >> shifts) & 3).to(dtype)
    #     x = q * scales.unsqueeze(-1) + zeros.unsqueeze(-1)
    #     return torch.cat([x.reshape(...).transpose(2, 3), residual], dim=2)
    #
    # allocates three FP-sized temporaries (the `.to`, the multiply, the
    # add) and then a fourth for the concat. The `*_into` form below fuses
    # the multiply-add in place and writes straight into a slice of the
    # caller's output buffer, so the concat disappears too: two temporaries
    # total (one uint8, one float). Measured 33% faster at the shape above,
    # bit-identical output — `x = q*s; x += z` rounds exactly like
    # `x = q*s + z`.

    def _dequant_keys_into(self, out, packed, scales, zeros):
        """Unpack per-channel-quantized keys directly into ``out``.

        packed [B,H,D,T/4] u8, scales/zeros [B,H,D,T/G], out [B,H,T,D].
        The transposed ``copy_`` at the end is the channel-major ->
        token-major turn; it is unavoidable without changing the served
        layout, and it is the one pass that has to be strided."""
        B, H, D, _ = packed.shape
        q = (packed.unsqueeze(-1) >> _UNPACK_SHIFTS) & 0x3   # [B,H,D,T/4,4]
        T = q.numel() // (B * H * D)
        G = T // scales.shape[-1]          # per-layer G, from the shapes
        q = q.reshape(B, H, D, T // G, G).to(out.dtype)
        q.mul_(scales.unsqueeze(-1)).add_(zeros.unsqueeze(-1))
        out.copy_(q.reshape(B, H, D, T).transpose(2, 3))

    def _dequant_values_into(self, out, packed, scales, zeros):
        """Unpack per-token-quantized values directly into ``out``.

        packed [B,H,T,D/4] u8, scales/zeros [B,H,T,D/G], out [B,H,T,D].
        No transpose needed — values are already token-major."""
        B, H, T, _ = packed.shape
        q = (packed.unsqueeze(-1) >> _UNPACK_SHIFTS) & 0x3   # [B,H,T,D/4,4]
        D = q.numel() // (B * H * T)
        G = D // scales.shape[-1]
        q = q.reshape(B, H, T, D // G, G).to(out.dtype)
        q.mul_(scales.unsqueeze(-1)).add_(zeros.unsqueeze(-1))
        out.copy_(q.reshape(B, H, T, D))

    def _dequant_keys_cpu(self, packed, scales, zeros, dtype=None):
        """Allocating form of ``_dequant_keys_into`` — [B,H,T,D] keys in
        ``dtype`` (default: the cache's element dtype)."""
        B, H, D, _ = packed.shape
        T = (packed.numel() // (B * H * D)) * 4
        out = torch.empty((B, H, T, D), dtype=self._out_dtype(dtype, scales))
        self._dequant_keys_into(out, packed, scales, zeros)
        return out

    def _dequant_values_cpu(self, packed, scales, zeros, dtype=None):
        """Allocating form of ``_dequant_values_into`` — [B,H,T,D] values in
        ``dtype`` (default: the cache's element dtype)."""
        B, H, T, _ = packed.shape
        D = (packed.numel() // (B * H * T)) * 4
        out = torch.empty((B, H, T, D), dtype=self._out_dtype(dtype, scales))
        self._dequant_values_into(out, packed, scales, zeros)
        return out

    def _resolve_meta_dtype(self, cache_dtype: torch.dtype) -> torch.dtype:
        """Metadata dtype for a model whose KV dtype is ``cache_dtype``.
        Only (fp32, fp32), (fp32, fp16) and (fp16, fp16) have kernels."""
        if self.metadata_dtype in ("auto", torch.float16):
            return torch.float16
        if cache_dtype != torch.float32:
            raise ValueError(
                f"metadata_dtype=torch.float32 needs an fp32 model, got "
                f"{cache_dtype}: fp32 metadata on fp16 data has no kernel.")
        return torch.float32

    def _out_dtype(self, dtype, scales) -> torch.dtype:
        if dtype is not None:
            return dtype
        return self._cache_dtype if self._cache_dtype is not None else scales.dtype

    def _serve_layer(self, layer_idx: int):
        """Build the full [B,H,T,D] keys/values for one layer.

        Single allocation per tensor: the dequantized history is written
        into the head of the output and the FP residual copied into the
        tail, instead of materialising the history separately and then
        concatenating. Returns (None, None) if the layer is empty."""
        i = layer_idx
        res_k, res_v = self.res_keys[i], self.res_values[i]
        n_res = res_k.shape[2] if res_k is not None else 0
        arena = self._arena[i]
        has_hist = arena is not None and arena.used > 0

        if not has_hist:
            # Nothing quantized yet — the residual *is* the cache. Return it
            # directly; copying it into a fresh buffer would be pure waste.
            if n_res == 0:
                return None, None
            return res_k, res_v

        if self.serving == "cached" and self._deq_serving[i] is not None:
            hist_k, hist_v = self._deq_serving[i]
            n_hist = hist_k.shape[2]
        else:
            hist_k = hist_v = None
            n_hist = arena.used

        if n_res == 0 and hist_k is not None:
            return hist_k, hist_v

        B, H, D = arena.B, arena.H, self.head_dim
        T = n_hist + n_res
        # The model's dtype, not the metadata's: with fp16 metadata on an
        # fp32 model the scales are half but the served cache must be fp32.
        dtype = self._cache_dtype
        out_k = torch.empty((B, H, T, D), dtype=dtype)
        out_v = torch.empty((B, H, T, D), dtype=dtype)

        if hist_k is not None:
            out_k[:, :, :n_hist, :].copy_(hist_k)
            out_v[:, :, :n_hist, :].copy_(hist_v)
        else:
            self._dequant_keys_into(out_k[:, :, :n_hist, :], *arena.used_keys())
            self._dequant_values_into(out_v[:, :, :n_hist, :], *arena.used_values())

        if n_res:
            out_k[:, :, n_hist:, :].copy_(res_k)
            out_v[:, :, n_hist:, :].copy_(res_v)
        return out_k, out_v

    # ------------------------------------------------------------------ #
    #  Stats / Debug                                                       #
    # ------------------------------------------------------------------ #

    @property
    def total_flushes(self) -> int:
        return sum(self._flush_count)

    def get_stats(self, layer: int = 0) -> dict:
        """
        Memory/token statistics for a given layer — every byte count below
        is *measured* from live tensors (``numel() × element_size()``),
        not projected.

        - ``packed_bytes``: resident packed store — the layer's whole arena
          allocation (2-bit payload + scales/zeros, including spare capacity
          and row padding).
        - ``packed_used_bytes``: the part of it holding quantized tokens.
        - ``packed_capacity_tokens``: arena capacity in tokens.
        - ``residual_bytes``: FP residual window (includes tokens covered
          by a not-yet-harvested flush).
        - ``kivi_bytes``: packed + residual — the resident cache footprint
          in serving="rebuild" mode.
        - ``serving_copy_bytes``: the dequantized FP history copy retained
          in serving="cached" mode (0 in "rebuild" mode). Included in the
          compression denominator, so "cached" mode honestly reports
          little-to-no compression.
        - ``fp_baseline_bytes``: what a plain FP cache of the same tokens
          in the model's KV dtype would occupy.

        Note: in "rebuild" mode a *transient* FP reconstruction of the
        history also exists while a forward pass is running (the model
        consumes FP tensors); it is dropped after each ``add_tokens``.
        Peak-during-forward therefore still includes one FP copy — the
        sustained/resident footprint is what this method reports.
        """
        B = self.batch_size or 1
        H = self.num_kv_heads or 1
        i = layer

        pending_n = self._pending[i]["n"] if self._pending[i] is not None else 0
        quant_tokens = self._quant_token_count[i]
        res_tokens = self.res_keys[i].shape[2] if self.res_keys[i] is not None else 0
        total = quant_tokens + res_tokens

        def _nbytes(*tensors):
            return sum(t.numel() * t.element_size()
                       for t in tensors if t is not None)

        arena = self._arena[i]
        # Resident = the whole arena allocation (spare capacity and row
        # padding included); used = the bytes holding quantized tokens.
        packed_bytes = arena.allocated_bytes if arena is not None else 0
        packed_used_bytes = arena.used_bytes() if arena is not None else 0
        residual_bytes = _nbytes(self.res_keys[i], self.res_values[i])
        serving_copy_bytes = 0
        if self._deq_serving[i] is not None:
            serving_copy_bytes = _nbytes(*self._deq_serving[i])

        kivi_bytes = packed_bytes + residual_bytes
        elt = torch.tensor([], dtype=self._cache_dtype or torch.float32).element_size()
        fp_baseline_bytes = 2 * B * H * total * self.head_dim * elt
        denom = kivi_bytes + serving_copy_bytes

        return {
            "group_size": self._G[i],
            "total_tokens": total,
            "quantized_tokens": quant_tokens,
            "residual_tokens": res_tokens,
            "pending_tokens": pending_n,
            "packed_bytes": packed_bytes,
            "packed_used_bytes": packed_used_bytes,
            "packed_capacity_tokens": arena.cap if arena is not None else 0,
            "residual_bytes": residual_bytes,
            "serving_copy_bytes": serving_copy_bytes,
            "kivi_bytes": kivi_bytes,
            "fp_baseline_bytes": fp_baseline_bytes,
            # Backwards-compatible alias (pre-packed-resident scripts)
            "fp32_bytes": fp_baseline_bytes,
            "compression": fp_baseline_bytes / denom if denom > 0 else 0,
        }

    def __repr__(self):
        return (f"KiviCache(layers={self.num_layers}, head_dim={self.head_dim}, "
                f"G={self.G if self.G is not None else self._G}, R={self.R}, device='{self.device}', "
                f"serving='{self.serving}', async_flush={self.async_flush}, "
                f"metadata_dtype={self.metadata_dtype}, zero_copy={self.zero_copy})")
