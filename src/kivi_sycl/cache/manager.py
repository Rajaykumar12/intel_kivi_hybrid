"""KiviCache: the KIVI 2-bit asymmetric KV cache state manager."""

from typing import List, Optional, Tuple

import torch
from transformers.cache_utils import DynamicCache

from ..config.model_config import detect_model_config
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

    Key-residual flush policy (``key_flush``) selects how much full
    precision the *key* window holds. The paper (§3.3, Fig. 3) accumulates
    keys until R and then quantizes the whole buffer, resetting it to
    empty — occupancy sawtooths 0 -> R, averaging R/2. This implementation
    defaults to draining one group at a time, which holds
    ``[R-G, R-1]`` (averaging ~R), roughly twice the paper's key window:

    - ``key_flush="drain"`` (default): incremental, one G-token group per
      flush. Fewer, smaller kernel launches; more full-precision keys held.
    - ``key_flush="reset"``: paper-faithful. Quantizes the whole key
      residual in one submission and leaves it empty, so the key window
      averages R/2. Requires ``residual_length % group_size == 0``.
      Compresses better, at slightly lower key fidelity.

    The value path is unaffected by this flag: values always drain a group
    at a time (the paper's per-token FIFO pop is the same policy at G=1,
    and one launch per group matters far more on an iGPU than on an A100).
    Under ``"reset"`` the key and value histories therefore advance at
    different rates and are tracked by separate token counters.

    Peak vs resident memory (``prefill_chunk``): a flush holds its tokens in
    three places at once until it is harvested — the FP residual, the
    retained device-side kernel inputs, and the packed output. In
    steady-state decode that transient is one group and is negligible.
    During bulk prefill it covers the prompt's entire grouped part, and
    measured peak then runs about 2x the full-precision baseline even though
    resident is several times better. Setting ``prefill_chunk`` ingests a
    long prompt in slices, harvesting between them, which bounds the
    transient to one slice. The packed output is bit-identical either way
    (slice boundaries are multiples of G, so quantization groups do not
    move); the cost is that the prefill flush no longer overlaps a forward
    pass. Decode is unaffected.

    Flush execution: quantization kernels are submitted to the XPU
    asynchronously during ``add_tokens`` and harvested at the start of the
    *next* ``add_tokens`` call, so kernel + transfer time overlaps the
    model's forward pass. Until harvest, the affected tokens remain served
    from the FP residual, so correctness never depends on kernel timing.
    Pass ``async_flush=False`` to harvest immediately (blocking, simpler
    timing for benchmarks).

    Element dtype follows the model: fp32 and fp16 KV tensors are stored
    and quantized natively; bf16 is upcast to fp32 at ingestion (the
    kernels have no bf16 instantiation).

    Args:
        num_layers:       Number of transformer layers.
        head_dim:         Dimension per attention head.
        num_kv_heads:     Number of KV heads (optional, auto-detected on first call).
        device:           Device for quantization kernels ("xpu").
        group_size:       Quantization group size G. Must divide head_dim and be >=4.
        residual_length:  Number of recent full-precision tokens to keep (R in paper).
        serving:          "rebuild" (memory-honest) or "cached" (speed) — see above.
        async_flush:      Overlap flush kernels with the next forward pass.
        key_flush:        "drain" (default) or "reset" — see above.
        prefill_chunk:    If set, ingest a prompt longer than this many
                          tokens in slices of this size, harvesting between
                          them. Bounds peak memory; see below. None (default)
                          keeps the single-submission behaviour every
                          published number in this repo was measured with.
    """

    def __init__(
        self,
        num_layers: int,
        head_dim: int,
        num_kv_heads: Optional[int] = None,
        device: str = "xpu",
        group_size: int = 32,
        residual_length: int = 128,
        serving: str = "rebuild",
        async_flush: bool = True,
        key_flush: str = "drain",
        prefill_chunk: Optional[int] = None,
    ):
        assert head_dim % group_size == 0, \
            f"head_dim ({head_dim}) must be divisible by group_size ({group_size})"
        assert group_size >= 4 and group_size % 4 == 0, \
            f"group_size ({group_size}) must be >= 4 and divisible by 4"
        assert group_size <= residual_length, (
            f"group_size ({group_size}) must not exceed residual_length "
            f"({residual_length}) — flushing requires a full group of tokens "
            f"to be available in the residual buffer, or tokens are silently dropped."
        )
        assert serving in ("rebuild", "cached"), \
            f"serving must be 'rebuild' or 'cached', got {serving!r}"
        assert key_flush in ("drain", "reset"), \
            f"key_flush must be 'drain' or 'reset', got {key_flush!r}"
        if prefill_chunk is not None:
            assert prefill_chunk >= residual_length and prefill_chunk % group_size == 0, (
                f"prefill_chunk ({prefill_chunk}) must be a multiple of "
                f"group_size ({group_size}) and at least residual_length "
                f"({residual_length}) — a smaller or unaligned slice would "
                f"move the quantization group boundaries, so the packed "
                f"output would no longer match the unchunked path."
            )
        if key_flush == "reset":
            # The paper requires R divisible by G ("We note that R should be
            # divisible by G"). Under "drain" the incremental flush makes that
            # unnecessary, which is why it is not a global constraint — but
            # "reset" quantizes the whole buffer at once, so a residual that
            # stops at R with R % G != 0 would leave a permanent sub-group
            # remainder and the key window would never actually reset.
            assert residual_length % group_size == 0, (
                f"key_flush='reset' requires residual_length ({residual_length}) "
                f"to be divisible by group_size ({group_size}) — the paper's "
                f"§3.3 constraint. Under key_flush='drain' it is not needed."
            )

        self.device = device
        self.G = group_size
        self.R = residual_length
        self.head_dim = head_dim
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.serving = serving
        self.async_flush = async_flush
        self.key_flush = key_flush
        self.prefill_chunk = prefill_chunk
        self.batch_size: Optional[int] = None
        self._cache_dtype: Optional[torch.dtype] = None

        # FP residual buffers on CPU (model dtype)
        self.res_keys: List[Optional[torch.Tensor]] = [None] * num_layers
        self.res_values: List[Optional[torch.Tensor]] = [None] * num_layers

        # Packed-resident history on CPU. Keys are stored channel-major:
        #   _packed_k [B,H,D,T/4] u8, _scales_k/_zeros_k [B,H,D,T/G]
        # Values are stored token-major:
        #   _packed_v [B,H,T,D/4] u8, _scales_v/_zeros_v [B,H,T,D/G]
        # T (quantized token count) is always a multiple of G, so chunk
        # concatenation along the token axis preserves group alignment.
        self._packed_k: List[Optional[torch.Tensor]] = [None] * num_layers
        self._scales_k: List[Optional[torch.Tensor]] = [None] * num_layers
        self._zeros_k: List[Optional[torch.Tensor]] = [None] * num_layers
        self._packed_v: List[Optional[torch.Tensor]] = [None] * num_layers
        self._scales_v: List[Optional[torch.Tensor]] = [None] * num_layers
        self._zeros_v: List[Optional[torch.Tensor]] = [None] * num_layers

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

        # Quantized-token counts. These are equal under key_flush="drain"
        # and diverge under "reset" (the key store races ahead of the value
        # store). `_quant_token_count` is the KEY count and is what
        # add_tokens uses to locate new tokens — the two buffers always
        # partition the same total, so either one identifies the boundary,
        # but they must never be used interchangeably for sizing.
        self._quant_token_count: List[int] = [0] * num_layers
        self._quant_token_count_v: List[int] = [0] * num_layers
        self._flush_count: List[int] = [0] * num_layers

        # P9: high-water mark of this layer's real footprint, including the
        # transients get_stats()' resident figures deliberately exclude —
        # the retained device-side flush inputs and the FP view handed to
        # the model. Sampled at every point that can raise it.
        self._peak_bytes: List[int] = [0] * num_layers
        self._peak_at_tokens: List[int] = [0] * num_layers

        # Running mean of residual occupancy, one sample per add_tokens per
        # layer. This is the ρ in the project's compression law
        # `C / (1 + ρ(C-1))`, and it is the quantity key_flush moves: the
        # paper's key window averages R/2, "drain" averages ~R. Measured
        # rather than assumed, so the law can be checked against reality
        # instead of being used to predict it.
        self._res_occ_sum_k: List[int] = [0] * num_layers
        self._res_occ_sum_v: List[int] = [0] * num_layers
        self._res_occ_steps: List[int] = [0] * num_layers

        # Time-averaged resident bytes vs the FP baseline, sampled once per
        # step at a consistent phase. `compression` is a single instantaneous
        # reading, and under key_flush="reset" the key window sweeps the whole
        # 0..R range — so where a run happens to stop moves that number by
        # several percent and two policies can coincide by accident. The
        # averaged figure is what a policy comparison has to use.
        self._bytes_sum_kivi: List[int] = [0] * num_layers
        self._bytes_sum_fp: List[int] = [0] * num_layers
        self._bytes_samples: List[int] = [0] * num_layers

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
        group_size: int = 32,
        residual_length: int = 128,
        serving: str = "rebuild",
        async_flush: bool = True,
        key_flush: str = "drain",
        prefill_chunk: Optional[int] = None,
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
            key_flush=key_flush,
            prefill_chunk=prefill_chunk,
        )

    # ------------------------------------------------------------------ #
    #  Core API                                                            #
    # ------------------------------------------------------------------ #

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

        # Sample resident bytes here, immediately after harvest and before
        # this step's tokens land: the one moment per step at which no
        # in-flight flush is inflating the residual. Consistent phase is the
        # whole point — an inconsistently sampled average is worse than an
        # instantaneous reading, because it looks trustworthy.
        for i in range(self.num_layers):
            self._sample_resident(i)

        for i, (k, v) in enumerate(_iter_kv_layers(past_key_values)):
            if self.batch_size is None:
                self.batch_size = k.shape[0]
                self.num_kv_heads = k.shape[1]
            else:
                assert k.shape[0] == self.batch_size, (
                    f"KiviCache batch size changed mid-session "
                    f"({self.batch_size} -> {k.shape[0]}) at layer {i}. "
                    f"Call reset() before starting a new request/batch."
                )
            if self._cache_dtype is None:
                # bf16 has no kernel instantiation — upcast to fp32.
                # fp32/fp16 are stored and quantized natively.
                self._cache_dtype = (
                    k.dtype if k.dtype in (torch.float32, torch.float16)
                    else torch.float32
                )

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

            # No manual append to the cache view here: when the model grew
            # `self._cache_view` in place (the identity check at the top of
            # this method), the view already contains this token — calling
            # `.update()` again would append it a second time. When the
            # model returned a different object, the view was invalidated
            # above and the next get_full_cache() rebuilds it from the
            # residual, which does include this token.

            chunk = self.prefill_chunk
            n_new = new_k.shape[2]
            if chunk and n_new > chunk:
                # Bounded-peak prefill: ingest the prompt in slices and
                # harvest between them, so the transient described in
                # `_footprint_bytes` covers one slice instead of the whole
                # prompt. Every slice boundary is still flushed to a
                # multiple of G at a multiple of G, so the quantization
                # groups — and therefore the packed output — are identical
                # to the single-submission path. This trades the async
                # overlap on the prefill step only; decode is untouched.
                for s0 in range(0, n_new, chunk):
                    occ = self._ingest(
                        i,
                        new_k[:, :, s0:s0 + chunk, :].contiguous(),
                        new_v[:, :, s0:s0 + chunk, :].contiguous())
                    self._harvest_layer(i)
            else:
                occ = self._ingest(i, new_k, new_v)

            # One occupancy sample per add_tokens per layer, whatever the
            # slicing — otherwise a chunked prefill would contribute many
            # samples and the mean would not be comparable across configs.
            self._res_occ_sum_k[i] += occ[0]
            self._res_occ_sum_v[i] += occ[1]
            self._res_occ_steps[i] += 1

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
            else:
                # The reconstruction the model is about to consume is real
                # resident memory for the duration of the forward pass, and
                # in "rebuild" mode it is the single largest FP allocation
                # the design makes. Peak has to include it.
                self._sample_peak(i, self._view_extra_bytes(i, full_k, full_v))
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

    def _harvest_layer(self, layer_idx: int):
        """Harvest one layer's in-flight flush, if it has one. Split out of
        ``harvest_pending`` so chunked prefill can drain a single layer
        between slices without touching the others."""
        i = layer_idx
        p = self._pending[i]
        if p is None:
            return
        n_k, n_v = p["n_k"], p["n_v"]
        cur_k, cur_v = self._deq_serving[i] or (None, None)

        # Keys are channel-major: token axis is dim 3. Values are
        # token-major: token axis is dim 2. Chunk lengths are multiples
        # of G, so group alignment survives concatenation. Each side is
        # handled independently — under key_flush="reset" only one of
        # them may be present in this record.
        if n_k:
            p["ev_k"].wait()
            packed_k = p["packed_k"].to("cpu")
            scales_k = p["scales_k"].to("cpu")
            zeros_k = p["zeros_k"].to("cpu")
            if self._packed_k[i] is None:
                self._packed_k[i] = packed_k
                self._scales_k[i] = scales_k
                self._zeros_k[i] = zeros_k
            else:
                self._packed_k[i] = torch.cat([self._packed_k[i], packed_k], dim=3)
                self._scales_k[i] = torch.cat([self._scales_k[i], scales_k], dim=3)
                self._zeros_k[i] = torch.cat([self._zeros_k[i], zeros_k], dim=3)
            if self.serving == "cached":
                new_k = self._dequant_keys_cpu(packed_k, scales_k, zeros_k)
                cur_k = new_k if cur_k is None else \
                    torch.cat([cur_k, new_k], dim=2)
            # Trim the flushed tokens from the residual only now — until
            # this point they were still being served in full precision.
            self.res_keys[i] = self.res_keys[i][:, :, n_k:, :].contiguous()
            self._quant_token_count[i] += n_k

        if n_v:
            p["ev_v"].wait()
            packed_v = p["packed_v"].to("cpu")
            scales_v = p["scales_v"].to("cpu")
            zeros_v = p["zeros_v"].to("cpu")
            if self._packed_v[i] is None:
                self._packed_v[i] = packed_v
                self._scales_v[i] = scales_v
                self._zeros_v[i] = zeros_v
            else:
                self._packed_v[i] = torch.cat([self._packed_v[i], packed_v], dim=2)
                self._scales_v[i] = torch.cat([self._scales_v[i], scales_v], dim=2)
                self._zeros_v[i] = torch.cat([self._zeros_v[i], zeros_v], dim=2)
            if self.serving == "cached":
                new_v = self._dequant_values_cpu(packed_v, scales_v, zeros_v)
                cur_v = new_v if cur_v is None else \
                    torch.cat([cur_v, new_v], dim=2)
            self.res_values[i] = self.res_values[i][:, :, n_v:, :].contiguous()
            self._quant_token_count_v[i] += n_v

        if self.serving == "cached":
            self._deq_serving[i] = (cur_k, cur_v)

        self._flush_count[i] += 1
        self._pending[i] = None
        # History composition changed (FP residual tokens replaced by
        # their quantized representation) — the persistent view no
        # longer matches what should be served.
        self._cache_view_valid = False
        self._sample_peak(i)

    def reset(self):
        """Clear all cached data. Reuse the same KiviCache for a new sequence."""
        for i in range(self.num_layers):
            p = self._pending[i]
            if p is not None:
                # Never free tensors a submitted kernel may still touch.
                # Only the sides that were actually submitted have an event:
                # under key_flush="reset" a record often carries just one.
                for ev in ("ev_k", "ev_v"):
                    if p.get(ev) is not None:
                        p[ev].wait()
            self._pending[i] = None
            self._quant_token_count[i] = 0
            self._quant_token_count_v[i] = 0
            self._flush_count[i] = 0
            self._peak_bytes[i] = 0
            self._peak_at_tokens[i] = 0
            self._res_occ_sum_k[i] = 0
            self._res_occ_sum_v[i] = 0
            self._res_occ_steps[i] = 0
            self._bytes_sum_kivi[i] = 0
            self._bytes_sum_fp[i] = 0
            self._bytes_samples[i] = 0
            self.res_keys[i] = None
            self.res_values[i] = None
            self._packed_k[i] = None
            self._scales_k[i] = None
            self._zeros_k[i] = None
            self._packed_v[i] = None
            self._scales_v[i] = None
            self._zeros_v[i] = None
            self._deq_serving[i] = None
        self.batch_size = None
        self._cache_dtype = None
        self._cache_view = None
        self._cache_view_valid = False

    # ------------------------------------------------------------------ #
    #  Internals                                                           #
    # ------------------------------------------------------------------ #

    # -- P9: peak footprint ----------------------------------------------- #
    #
    # get_stats()' byte counts are *resident*: what survives between steps.
    # KIVI's headline memory claim is *peak*, and the two diverge most during
    # bulk prefill, where a flush of the whole grouped part briefly holds
    # those tokens in three places at once — the FP residual (CPU), the
    # retained kernel inputs (device, and `in_k` is a transposed
    # `.contiguous()` copy), and the packed outputs. Reporting resident
    # against a paper that reports peak, without measuring the difference,
    # is the error class already recorded in status_and_roadmap.md §2.1.
    #
    # Counted: packed store, FP residual, retained pending inputs/outputs,
    # the persistent serving copy, and the FP view handed to the model.
    # Not counted: torch's own short-lived temporaries inside `torch.cat`
    # when the packed store grows, and the model's activations. So this is
    # a lower bound on true peak — state it that way, do not round it up.

    @staticmethod
    def _nbytes(*tensors) -> int:
        return sum(t.numel() * t.element_size()
                   for t in tensors if t is not None)

    def _footprint_bytes(self, layer_idx: int, extra: int = 0) -> int:
        """Bytes this layer occupies *right now*, across both memories.

        On a unified-memory part the CPU and XPU allocations come out of the
        same DRAM, so summing them is what the device actually sees. On a
        discrete part this over-counts a single pool and under-counts the
        pressure on the smaller one — say which target the number is for."""
        i = layer_idx
        total = self._nbytes(
            self._packed_k[i], self._scales_k[i], self._zeros_k[i],
            self._packed_v[i], self._scales_v[i], self._zeros_v[i],
            self.res_keys[i], self.res_values[i])
        p = self._pending[i]
        if p is not None:
            total += self._nbytes(*(
                p.get(key) for key in
                ("in_k", "in_v", "packed_k", "scales_k", "zeros_k",
                 "packed_v", "scales_v", "zeros_v")))
        if self._deq_serving[i] is not None:
            total += self._nbytes(*self._deq_serving[i])
        return total + extra

    def _resident_bytes(self, layer_idx: int) -> int:
        """Resident cache footprint for one layer: packed store + FP residual
        + any persistent serving copy. This is ``kivi_bytes`` — the figure
        ``compression`` is computed from — excluding every transient."""
        i = layer_idx
        total = self._nbytes(
            self._packed_k[i], self._scales_k[i], self._zeros_k[i],
            self._packed_v[i], self._scales_v[i], self._zeros_v[i],
            self.res_keys[i], self.res_values[i])
        if self._deq_serving[i] is not None:
            total += self._nbytes(*self._deq_serving[i])
        return total

    def _sample_resident(self, layer_idx: int):
        """Accumulate one resident-vs-baseline sample for the time average."""
        i = layer_idx
        res = self.res_keys[i]
        total = self._quant_token_count[i] + (res.shape[2] if res is not None else 0)
        if total == 0:
            return
        elt = torch.tensor([], dtype=self._cache_dtype or torch.float32).element_size()
        self._bytes_sum_kivi[i] += self._resident_bytes(i)
        self._bytes_sum_fp[i] += (2 * (self.batch_size or 1) *
                                  (self.num_kv_heads or 1) * total *
                                  self.head_dim * elt)
        self._bytes_samples[i] += 1

    def _sample_peak(self, layer_idx: int, extra: int = 0) -> int:
        """Update this layer's high-water mark. Cheap: numel/element_size
        only, no allocation and no device synchronization."""
        i = layer_idx
        b = self._footprint_bytes(i, extra)
        if b > self._peak_bytes[i]:
            self._peak_bytes[i] = b
            res = self.res_keys[i]
            self._peak_at_tokens[i] = (self._quant_token_count[i] +
                                       (res.shape[2] if res is not None else 0))
        return b

    def _view_extra_bytes(self, layer_idx: int, k, v) -> int:
        """Bytes a served view adds beyond what ``_footprint_bytes`` counts.

        ``_serve_layer`` returns the residual tensors themselves when a layer
        has no quantized history, and the persistent dequantized history in
        serving="cached" with an empty residual. Those are aliases, not new
        allocations, and counting them would inflate peak."""
        i = layer_idx
        deq_k, deq_v = self._deq_serving[i] or (None, None)
        total = 0
        for t, res, hist in ((k, self.res_keys[i], deq_k),
                             (v, self.res_values[i], deq_v)):
            if t is None or t is res or t is hist:
                continue
            total += self._nbytes(t)
        return total

    def _ingest(self, layer_idx: int, new_k, new_v):
        """Append one slice of new tokens to the residual and submit whatever
        that pushes past R. Returns the post-flush (logical) occupancy of the
        key and value windows — the quantity the flush policy controls, as
        opposed to the raw buffer length, which also reflects how long an
        async flush has gone unharvested."""
        i = layer_idx
        if self.res_keys[i] is None:
            self.res_keys[i] = new_k
            self.res_values[i] = new_v
        else:
            self.res_keys[i] = torch.cat([self.res_keys[i], new_k], dim=2)
            self.res_values[i] = torch.cat([self.res_values[i], new_v], dim=2)

        # Flush each residual's excess in ONE kernel submission — a long
        # prompt's entire grouped part goes to the XPU in a single launch
        # instead of one launch per G tokens.
        #
        # The key and value buffers are decided independently because
        # key_flush="reset" makes them diverge: the key buffer empties and
        # refills over R steps while the value buffer keeps draining every G.
        # Under "drain" the two expressions coincide exactly (both buffers
        # hold the same token count), so this is bit-identical to the
        # single-policy version it replaces.
        res_k_len = self.res_keys[i].shape[2]
        res_v_len = self.res_values[i].shape[2]
        # The two buffers hold different *residual* lengths under "reset" but
        # must always account for the same total. A drift here is the
        # silent-desync failure mode of split counters: it would not corrupt
        # any single tensor, it would serve keys and values of different
        # lengths.
        assert (self._quant_token_count[i] + res_k_len
                == self._quant_token_count_v[i] + res_v_len), (
            f"KiviCache key/value token accounting diverged at layer {i}: "
            f"keys {self._quant_token_count[i]}+{res_k_len}, "
            f"values {self._quant_token_count_v[i]}+{res_v_len}."
        )
        flush_k = self._key_flush_len(res_k_len)
        flush_v = self._drain_flush_len(res_v_len)
        if flush_k or flush_v:
            self._submit_flush(i, flush_k, flush_v)
        return res_k_len - flush_k, res_v_len - flush_v

    def _drain_flush_len(self, res_len: int) -> int:
        """Incremental policy: flush the largest multiple of G that leaves at
        least ``R - G`` tokens behind, so occupancy stays in ``[R-G, R-1]``."""
        if res_len < self.R:
            return 0
        return ((res_len - (self.R - self.G)) // self.G) * self.G

    def _key_flush_len(self, res_len: int) -> int:
        """Key-side policy, selected by ``key_flush``.

        Under ``"reset"`` the whole buffer goes, reproducing the paper's
        0 -> R sawtooth. ``R % G == 0`` is enforced in ``__init__``, so in
        steady-state decode ``res_len`` is exactly R and the buffer really
        does empty. A prefill chunk of arbitrary length is the one case that
        does not divide evenly: the sub-group remainder is left behind rather
        than zero-padded, matching the paper, which pads only in its
        fake-quantization ablation (§3.1)."""
        if res_len < self.R:
            return 0
        if self.key_flush == "reset":
            return (res_len // self.G) * self.G
        return self._drain_flush_len(res_len)

    def _submit_flush(self, layer_idx: int, flush_len_k: int,
                      flush_len_v: Optional[int] = None):
        """Quantize the oldest residual tokens to the packed 2-bit
        representation on XPU — ``flush_len_k`` keys and ``flush_len_v``
        values. Either may be 0, in which case that side is skipped (under
        key_flush="reset" the two sides fire on different steps). Kernels
        are submitted without waiting; the outputs are collected by
        ``harvest_pending``. The tokens stay in the residual (still served
        in FP) until harvest.

        ``flush_len_v`` defaults to ``flush_len_k`` so the historical
        single-length call still means "flush both sides equally"."""
        i = layer_idx
        G = self.G
        if flush_len_v is None:
            flush_len_v = flush_len_k
        assert self._pending[i] is None, (
            f"Layer {i} already has an in-flight flush — harvest_pending() "
            f"must run before a new flush is submitted."
        )
        assert flush_len_k > 0 or flush_len_v > 0, \
            "_submit_flush called with nothing to flush"
        for name, n, res in (("key", flush_len_k, self.res_keys[i]),
                             ("value", flush_len_v, self.res_values[i])):
            assert n >= 0 and n % G == 0, (
                f"{name} flush_len ({n}) must be a non-negative multiple of "
                f"G ({G})."
            )
            assert res.shape[2] >= n, (
                f"Cannot flush {n} {name} tokens: residual only has "
                f"{res.shape[2]} at layer {i}."
            )

        rec = {"n_k": flush_len_k, "n_v": flush_len_v}

        # --- Keys: per-channel quantization (transpose to [B,H,D,L]) ---
        if flush_len_k:
            k_chunk = self.res_keys[i][:, :, :flush_len_k, :]
            k_trans = k_chunk.transpose(2, 3).contiguous().to(self.device)
            batch, heads, dim, L = k_trans.shape

            packed_k = torch.empty((batch, heads, dim, L // 4),
                                   dtype=torch.uint8, device=self.device)
            scales_k = torch.empty((batch, heads, dim, L // G),
                                   dtype=k_trans.dtype, device=self.device)
            zeros_k = torch.empty_like(scales_k)
            rec.update(
                ev_k=kivi_native.quantize_keys_async(k_trans, packed_k,
                                                     scales_k, zeros_k, G),
                # Inputs are retained until harvest: the kernels read them.
                in_k=k_trans,
                packed_k=packed_k, scales_k=scales_k, zeros_k=zeros_k,
            )

        # --- Values: per-token quantization (natural [B,H,L,D] layout) ---
        if flush_len_v:
            v_chunk = self.res_values[i][:, :, :flush_len_v, :] \
                          .contiguous().to(self.device)
            batch, heads, L = v_chunk.shape[0], v_chunk.shape[1], v_chunk.shape[2]
            groups_per_token = self.head_dim // G

            packed_v = torch.empty((batch, heads, L, self.head_dim // 4),
                                   dtype=torch.uint8, device=self.device)
            scales_v = torch.empty((batch, heads, L, groups_per_token),
                                   dtype=v_chunk.dtype, device=self.device)
            zeros_v = torch.empty_like(scales_v)
            rec.update(
                ev_v=kivi_native.quantize_values_async(v_chunk, packed_v,
                                                       scales_v, zeros_v,
                                                       self.head_dim, G),
                in_v=v_chunk,
                packed_v=packed_v, scales_v=scales_v, zeros_v=zeros_v,
            )

        self._pending[i] = rec
        # The worst moment for this layer: the flushed tokens exist in the FP
        # residual, in the retained device inputs, and in the packed outputs
        # simultaneously. Nothing is trimmed until harvest.
        self._sample_peak(i)

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
        q = q.reshape(B, H, D, T // self.G, self.G).to(scales.dtype)
        q.mul_(scales.unsqueeze(-1)).add_(zeros.unsqueeze(-1))
        out.copy_(q.reshape(B, H, D, T).transpose(2, 3))

    def _dequant_values_into(self, out, packed, scales, zeros):
        """Unpack per-token-quantized values directly into ``out``.

        packed [B,H,T,D/4] u8, scales/zeros [B,H,T,D/G], out [B,H,T,D].
        No transpose needed — values are already token-major."""
        B, H, T, _ = packed.shape
        q = (packed.unsqueeze(-1) >> _UNPACK_SHIFTS) & 0x3   # [B,H,T,D/4,4]
        D = q.numel() // (B * H * T)
        q = q.reshape(B, H, T, D // self.G, self.G).to(scales.dtype)
        q.mul_(scales.unsqueeze(-1)).add_(zeros.unsqueeze(-1))
        out.copy_(q.reshape(B, H, T, D))

    def _dequant_keys_cpu(self, packed, scales, zeros):
        """Allocating form of ``_dequant_keys_into`` — [B,H,T,D] keys."""
        B, H, D, _ = packed.shape
        T = (packed.numel() // (B * H * D)) * 4
        out = torch.empty((B, H, T, D), dtype=scales.dtype)
        self._dequant_keys_into(out, packed, scales, zeros)
        return out

    def _dequant_values_cpu(self, packed, scales, zeros):
        """Allocating form of ``_dequant_values_into`` — [B,H,T,D] values."""
        B, H, T, _ = packed.shape
        D = (packed.numel() // (B * H * T)) * 4
        out = torch.empty((B, H, T, D), dtype=scales.dtype)
        self._dequant_values_into(out, packed, scales, zeros)
        return out

    def _serve_side(self, packed, scales, zeros, hist, res, is_key: bool):
        """Build the full [B,H,T,D] tensor for one side (keys or values).

        Single allocation: the dequantized history is written into the head
        of the output and the FP residual copied into the tail, instead of
        materialising the history separately and then concatenating.

        Keys and values are built independently because key_flush="reset"
        makes their histories different lengths — only the *sum* of history
        and residual is shared between them. Returns None if the side is
        empty.

        ``hist`` is the persistent dequantized history (serving="cached")
        or None, in which case the packed store is dequantized on read."""
        n_res = res.shape[2] if res is not None else 0
        if packed is None:
            # Nothing quantized yet — the residual *is* the cache. Return it
            # directly; copying it into a fresh buffer would be pure waste.
            return res if n_res else None

        n_hist = hist.shape[2] if hist is not None else \
            (packed.shape[3] * 4 if is_key else packed.shape[2])
        if n_res == 0 and hist is not None:
            return hist

        B, H = packed.shape[0], packed.shape[1]
        out = torch.empty((B, H, n_hist + n_res, self.head_dim),
                          dtype=scales.dtype)
        if hist is not None:
            out[:, :, :n_hist, :].copy_(hist)
        elif is_key:
            self._dequant_keys_into(out[:, :, :n_hist, :], packed, scales, zeros)
        else:
            self._dequant_values_into(out[:, :, :n_hist, :], packed, scales, zeros)
        if n_res:
            out[:, :, n_hist:, :].copy_(res)
        return out

    def _serve_layer(self, layer_idx: int):
        """Build the full [B,H,T,D] keys and values for one layer.
        Returns (None, None) if the layer is empty."""
        i = layer_idx
        hist_k = hist_v = None
        if self.serving == "cached" and self._deq_serving[i] is not None:
            hist_k, hist_v = self._deq_serving[i]

        out_k = self._serve_side(self._packed_k[i], self._scales_k[i],
                                 self._zeros_k[i], hist_k, self.res_keys[i],
                                 is_key=True)
        out_v = self._serve_side(self._packed_v[i], self._scales_v[i],
                                 self._zeros_v[i], hist_v, self.res_values[i],
                                 is_key=False)
        if out_k is None or out_v is None:
            return None, None
        # The split counters must not have drifted: keys and values describe
        # the same token stream, so the served lengths have to agree even
        # when the histories behind them do not.
        assert out_k.shape[2] == out_v.shape[2], (
            f"KiviCache served {out_k.shape[2]} keys but {out_v.shape[2]} "
            f"values at layer {i} — key/value token accounting has diverged."
        )
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

        - ``packed_bytes``: resident 2-bit payload + scales/zeros.
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

        Everything above is *resident*: what survives between steps. KIVI's
        headline memory claim is *peak*, so the peak counters below are
        reported alongside and must not be conflated with them:

        - ``peak_bytes``: high-water mark of resident + the transients —
          the retained device-side flush inputs and the FP reconstruction
          handed to the model. A lower bound on true peak (torch's own
          temporaries are not counted); see ``_footprint_bytes``.
        - ``peak_at_tokens``: token count when that mark was set. Compare
          against ``total_tokens`` before quoting ``peak_compression``: the
          two are only apples-to-apples when peak occurred at or near the
          final length, which holds for prefill-dominated runs.
        - ``peak_compression``: ``fp_baseline_bytes / peak_bytes``.

        Residual-occupancy counters, for checking the compression law
        ``C / (1 + ρ(C-1))`` against measurement rather than assuming it:

        - ``mean_residual_tokens`` / ``..._v``: time-averaged key and value
          window occupancy, one sample per ``add_tokens``. Under
          key_flush="reset" the key figure should approach R/2 and the value
          figure R - G/2; under "drain" both approach R - G/2.
        - ``rho``: instantaneous residual fraction — the ρ that corresponds
          to the byte counts reported here.
        - ``rho_mean``: time-averaged occupancy over the current total.
          Under "reset" the key window swings across the whole 0..R range,
          so ``rho`` at any single instant is not representative and this is
          the figure to quote for typical compression.
        - ``mean_compression``: measured time average — the sum of
          FP-baseline bytes over the run divided by the sum of resident
          bytes, sampled once per step at a fixed phase. **Use this, not
          ``compression``, to compare configurations.** A single
          instantaneous reading depends on where in the flush cycle the run
          stopped, and two policies holding very different average windows
          can report the same ``compression`` by coincidence.
        - ``compression_samples``: how many steps that average covers.
        """
        B = self.batch_size or 1
        H = self.num_kv_heads or 1
        i = layer

        p = self._pending[i]
        pending_n = p["n_k"] if p is not None else 0
        pending_n_v = p["n_v"] if p is not None else 0
        quant_tokens = self._quant_token_count[i]
        res_tokens = self.res_keys[i].shape[2] if self.res_keys[i] is not None else 0
        res_tokens_v = (self.res_values[i].shape[2]
                        if self.res_values[i] is not None else 0)
        total = quant_tokens + res_tokens

        _nbytes = self._nbytes
        packed_bytes = _nbytes(
            self._packed_k[i], self._scales_k[i], self._zeros_k[i],
            self._packed_v[i], self._scales_v[i], self._zeros_v[i])
        residual_bytes = _nbytes(self.res_keys[i], self.res_values[i])
        serving_copy_bytes = 0
        if self._deq_serving[i] is not None:
            serving_copy_bytes = _nbytes(*self._deq_serving[i])

        kivi_bytes = packed_bytes + residual_bytes
        elt = torch.tensor([], dtype=self._cache_dtype or torch.float32).element_size()
        fp_baseline_bytes = 2 * B * H * total * self.head_dim * elt
        denom = kivi_bytes + serving_copy_bytes

        peak_bytes = self._peak_bytes[i]
        steps = self._res_occ_steps[i]
        mean_occ_k = self._res_occ_sum_k[i] / steps if steps else 0.0
        mean_occ_v = self._res_occ_sum_v[i] / steps if steps else 0.0
        return {
            "total_tokens": total,
            "quantized_tokens": quant_tokens,
            "residual_tokens": res_tokens,
            "pending_tokens": pending_n,
            # Key/value counts differ only under key_flush="reset"; the
            # unsuffixed keys above stay key-side for backwards compatibility.
            "quantized_tokens_v": self._quant_token_count_v[i],
            "residual_tokens_v": res_tokens_v,
            "pending_tokens_v": pending_n_v,
            "key_flush": self.key_flush,
            "mean_residual_tokens": mean_occ_k,
            "mean_residual_tokens_v": mean_occ_v,
            "rho": ((res_tokens + res_tokens_v) / (2 * total)) if total else 0.0,
            "rho_mean": (((mean_occ_k + mean_occ_v) / (2 * total))
                         if total else 0.0),
            "mean_compression": (self._bytes_sum_fp[i] / self._bytes_sum_kivi[i]
                                 if self._bytes_sum_kivi[i] > 0 else 0),
            "compression_samples": self._bytes_samples[i],
            "peak_bytes": peak_bytes,
            "peak_at_tokens": self._peak_at_tokens[i],
            "peak_compression": (fp_baseline_bytes / peak_bytes
                                 if peak_bytes > 0 else 0),
            "packed_bytes": packed_bytes,
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
                f"G={self.G}, R={self.R}, device='{self.device}', "
                f"serving='{self.serving}', async_flush={self.async_flush}, "
                f"key_flush='{self.key_flush}', "
                f"prefill_chunk={self.prefill_chunk})")
