"""KiviCache: the KIVI 2-bit asymmetric KV cache — Python front end over the
C++ engine (kivi::Engine, src/csrc/engine/).

All cache state and work — the FP residual, flushing its excess into the
packed 2-bit arena on the XPU, asynchronous harvest, reserve(), per-layer
group size, measured stats, dequantize-on-read — lives in the native engine
(Phase 5). This module adapts it to the transformers APIs:

  * ``add_tokens(past)`` / ``get_full_cache()``  the legacy loop: feed the
    model's full ``past_key_values`` after every forward pass, get back a
    full-precision ``DynamicCache`` view for the next one;
  * ``kivi_sycl.enable_kivi`` (hf/) — a native ``transformers.Cache`` built on
    the same engine, with no full-precision cache rebuilt between steps.

What stays in Python is only what is about Python objects: the persistent
``DynamicCache`` view, the ``serving="cached"`` full-precision copy, and
extracting each step's new tokens from the model's cache.
"""

import importlib
import os
from typing import List, Optional, Tuple

import torch
from transformers.cache_utils import DynamicCache

from ..config.model_config import detect_model_config

__all__ = ["KiviCache", "dequantize_keys_cpu", "dequantize_values_cpu"]


def _ext():
    """The compiled extension. Imported directly rather than through the
    extension.loader seam, which tests may point at torch stand-ins."""
    try:
        ext = importlib.import_module("kivi_sycl._C")
    except ImportError as exc:
        raise RuntimeError(
            "KiviCache needs the compiled extension kivi_sycl._C (see README: "
            "Build from source)") from exc
    if not hasattr(ext, "Engine"):
        raise RuntimeError("this build of kivi_sycl._C has no Engine — rebuild the extension")
    return ext


def dequantize_keys_cpu(packed, scales, zeros, dtype=None):
    """Packed per-channel keys [B,H,D,T/4] -> [B,H,T,D] on the CPU, in `dtype`
    (default: the scales' dtype). Same code as the engine's serving path."""
    return _ext().dequant_keys_cpu(packed, scales, zeros, dtype or scales.dtype)


def dequantize_values_cpu(packed, scales, zeros, dtype=None):
    """Packed per-token values [B,H,T,D/4] -> [B,H,T,D] on the CPU."""
    return _ext().dequant_values_cpu(packed, scales, zeros, dtype or scales.dtype)


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
    KIVI 2-bit asymmetric KV cache.

    Plug-and-play with any HuggingFace CausalLM that returns
    ``past_key_values`` as a tuple of (key, value) tensors or a
    `transformers.Cache`. Typical shapes: key/value = [batch, num_kv_heads,
    seq_len, head_dim].

    Storage: flushed tokens live as their real 2-bit packed representation
    (uint8 payload + per-group scales/zeros, fp16 by default) in a per-layer
    arena; the most recent ``residual_length`` tokens stay in full precision.
    On XPU (``zero_copy``, default) the arena is USM host memory the quantize
    kernels write directly. ``get_full_cache()`` reconstructs a
    full-precision view on demand:

    - ``serving="rebuild"`` (default): no full-precision copy of the history
      is retained between steps — the resident footprint is packed +
      residual, and each ``get_full_cache()`` after a change dequantizes.
    - ``serving="cached"``: a dequantized copy of the history is kept and the
      served view persists across steps. Faster, but the resident footprint
      is roughly the FP baseline *plus* the packed store.

    Flushes are submitted to the XPU during ``add_tokens`` and harvested at
    the start of the next call, overlapping the model's forward pass; until
    then the tokens are served from the FP residual. ``async_flush=False``
    harvests immediately.

    Args:
        num_layers:       Number of transformer layers.
        head_dim:         Dimension per attention head.
        num_kv_heads:     Number of KV heads (optional; taken from the first tensors).
        device:           Where the quantize kernels run: "xpu", or "cpu" for the
                          engine's reference quantize (no GPU needed; slow).
        group_size:       Quantization group size G — an int, or one per layer.
                          Each must divide head_dim, be a multiple of 4 and not
                          exceed residual_length.
        residual_length:  Number of recent full-precision tokens (R in the paper).
        serving:          "rebuild" (memory-honest) or "cached" (speed) — see above.
        async_flush:      Overlap flush kernels with the next forward pass.
        metadata_dtype:   "auto" / torch.float16 (default) or torch.float32 (fp32
                          models only).
        zero_copy:        "auto" (on for XPU), True, or False (stage through XPU
                          tensors and copy back at harvest).
        engine:           "auto" or "native". The pure-Python engine was removed in
                          Phase 5 (S1.8); the native one reproduces it bit for bit.
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
        engine="auto",
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
        if engine == "python":
            raise ValueError(
                "engine='python' was removed in Phase 5 (S1.8): the C++ engine "
                "reproduces it bit for bit. Use engine='auto' or 'native'.")
        assert engine in ("auto", "native"), \
            f"engine must be 'auto' or 'native', got {engine!r}"

        self.device = device
        # Per-layer group sizes; self.G is the shared value when uniform.
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
        self.engine = "native"
        self.batch_size: Optional[int] = None
        self._cache_dtype: Optional[torch.dtype] = None
        self._meta_dtype: Optional[torch.dtype] = None
        self._reserve_tokens = 0

        self._native = _ext().Engine(
            num_layers=num_layers, head_dim=head_dim, num_kv_heads=num_kv_heads or 0,
            residual_length=residual_length, group_size=list(group_sizes),
            metadata_dtype={"auto": "auto", torch.float16: "float16",
                            torch.float32: "float32"}[metadata_dtype],
            zero_copy={"auto": "auto", True: "on", False: "off"}[zero_copy],
            async_flush=async_flush, device=str(device))

        # Python-side serving state.
        self._deq_serving: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = \
            [None] * num_layers                       # serving="cached" FP copy
        self._deq_upto: List[int] = [0] * num_layers  # how far that copy reaches
        self._flushes_seen = 0
        # Persistent Cache object returned by get_full_cache(). Reused across
        # non-flush steps in serving="cached" mode (the model grows it in
        # place); dropped after every add_tokens in serving="rebuild" mode.
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
        engine="auto",
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
            engine=engine,
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
        self._native.reserve(total_tokens)

    def add_tokens(self, past_key_values):
        """
        Ingest new KV pairs from a model forward pass.

        Args:
            past_key_values: the model's past_key_values from a forward
                pass — either the legacy tuple of (key, value) per layer, or
                a transformers `Cache`/`DynamicCache` object. Each tensor:
                [B, H, Seq, D].

        Appends each layer's new tokens to its FP residual; when the residual
        reaches ``R`` tokens the excess (the largest multiple of ``G`` that
        leaves at least ``R - G`` behind) is quantized to the packed store in
        a single kernel submission.
        """
        # get_full_cache() keeps the persistent view valid only if the model
        # grew that exact object in place. Models that rebuild their cache
        # (legacy tuples, unported architectures) hand back a different
        # object; serving the stale view again would drop every generated
        # token. Object identity is the exact test.
        if past_key_values is not self._cache_view:
            self._cache_view_valid = False

        e = self._native
        # Collect the flush submitted on the previous step — the XPU had the
        # whole forward pass to finish it, so this wait is effectively free.
        e.harvest_all()
        self._sync()
        for i, (k, v) in enumerate(_iter_kv_layers(past_key_values)):
            self._init_session(k, i)
            res = e.residual(i)
            if res is not None:
                new_start = e.quant_tokens(i) + res[0].shape[2]
                assert k.shape[2] >= new_start, (
                    f"KiviCache desync at layer {i}: expected past length "
                    f"{new_start} but got {k.shape[2]}. add_tokens() must be "
                    f"called exactly once per forward pass, seeded by "
                    f"get_full_cache()."
                )
                new_k, new_v = k[:, :, new_start:, :], v[:, :, new_start:, :]
            else:
                new_start, new_k, new_v = 0, k, v
            assert new_k.shape[2] > 0, (
                f"KiviCache desync at layer {i}: the forward pass produced "
                f"no tokens beyond the {new_start} already tracked. "
                f"add_tokens() was called twice on the same forward output, "
                f"or the model was fed a past_key_values not obtained from "
                f"get_full_cache(). Silently skipping here would drop this "
                f"step's KV and corrupt all subsequent attention."
            )
            e.append(i, new_k, new_v)
        if not self.async_flush:
            e.harvest_all()
            self._sync()
        if self.serving == "rebuild":
            # Drop the FP serving view — the resident footprint between
            # steps is packed history + residual only.
            self._cache_view = None
            self._cache_view_valid = False

    def get_full_cache(self) -> DynamicCache:
        """
        Reconstruct the full KV cache for the next model forward pass, as a
        transformers ``DynamicCache`` of (key, value) per layer on the CPU.

        In serving="rebuild" mode this dequantizes the packed history each
        time the cache composition has changed; the returned FP tensors are
        transient (dropped on the next ``add_tokens``). In "cached" mode the
        dequantized history persists and the same view object is returned
        across non-flush steps.
        """
        if self._cache_view is not None and self._cache_view_valid:
            return self._cache_view
        full_past = []
        for i in range(self.num_layers):
            full_k, full_v = self._serve_layer(i)
            if full_k is None:
                full_k = full_v = torch.empty(0)
            full_past.append((full_k, full_v))
        self._cache_view = _build_cache(full_past)
        self._cache_view_valid = True
        return self._cache_view

    def harvest_pending(self):
        """Collect all in-flight flushes. Called automatically at the start of
        every ``add_tokens``; call it manually before reading ``get_stats()``
        if you need the packed store fully up to date."""
        self._native.harvest_all()
        self._sync()

    def reset(self):
        """Clear all cached data. Reuse the same KiviCache for a new sequence."""
        self._native.reset()
        self._deq_serving = [None] * self.num_layers
        self._deq_upto = [0] * self.num_layers
        self._flushes_seen = 0
        self.batch_size = None
        self._cache_dtype = None
        self._meta_dtype = None
        self._reserve_tokens = 0
        self._cache_view = None
        self._cache_view_valid = False

    # ------------------------------------------------------------------ #
    #  Internals                                                           #
    # ------------------------------------------------------------------ #

    def _init_session(self, k: torch.Tensor, layer_idx: int):
        """Record batch size / KV heads / dtypes from the first tensor seen
        (the engine does the same; this mirrors them for Python callers)."""
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
            self._cache_dtype = (k.dtype if k.dtype in (torch.float32, torch.float16)
                                 else torch.float32)
            if self.metadata_dtype in ("auto", torch.float16):
                self._meta_dtype = torch.float16
            elif self._cache_dtype != torch.float32:
                raise ValueError(
                    f"metadata_dtype=torch.float32 needs an fp32 model, got "
                    f"{self._cache_dtype}: fp32 metadata on fp16 data has no kernel.")
            else:
                self._meta_dtype = torch.float32

    def _sync(self):
        """After a harvest: invalidate the persistent view if the history
        changed, and extend the serving="cached" FP copy."""
        flushes = self._native.total_flushes
        if flushes != self._flushes_seen:
            self._flushes_seen = flushes
            self._cache_view_valid = False
        if self.serving != "cached":
            return
        for i in range(self.num_layers):
            q = self._native.quant_tokens(i)
            t0 = self._deq_upto[i]
            if q <= t0:
                continue
            pk, sk, zk, pv, sv, zv = self._native.history(i)
            G = self._G[i]
            new_k = dequantize_keys_cpu(pk[..., t0 // 4:q // 4], sk[..., t0 // G:q // G],
                                        zk[..., t0 // G:q // G], self._cache_dtype)
            new_v = dequantize_values_cpu(pv[:, :, t0:q], sv[:, :, t0:q], zv[:, :, t0:q],
                                          self._cache_dtype)
            cur = self._deq_serving[i]
            self._deq_serving[i] = (new_k, new_v) if cur is None else (
                torch.cat([cur[0], new_k], dim=2), torch.cat([cur[1], new_v], dim=2))
            self._deq_upto[i] = q

    def _serve_layer(self, i: int):
        """Full [B,H,T,D] keys/values for one layer, or (None, None)."""
        res = self._native.residual(i)
        n_res = res[0].shape[2] if res is not None else 0
        if self._native.quant_tokens(i) == 0:
            # Nothing quantized yet — the residual *is* the cache.
            return (None, None) if n_res == 0 else res
        if self.serving == "cached" and self._deq_serving[i] is not None:
            hk, hv = self._deq_serving[i]
            if n_res == 0:
                return hk, hv
            return torch.cat([hk, res[0]], dim=2), torch.cat([hv, res[1]], dim=2)
        return self._native.full_kv(i)

    # ------------------------------------------------------------------ #
    #  Stats / introspection                                               #
    # ------------------------------------------------------------------ #

    @property
    def total_flushes(self) -> int:
        return self._native.total_flushes

    def get_stats(self, layer: int = 0) -> dict:
        """
        Memory/token statistics for a layer — every byte count is *measured*
        from live tensors, not projected.

        - ``packed_bytes``: the layer's whole arena allocation (2-bit payload
          + scales/zeros, including spare capacity).
        - ``packed_used_bytes``: the part of it holding quantized tokens.
        - ``packed_capacity_tokens``: arena capacity in tokens.
        - ``residual_bytes``: FP residual window (includes tokens covered by a
          not-yet-harvested flush).
        - ``kivi_bytes``: packed + residual — the resident cache footprint in
          serving="rebuild" mode.
        - ``serving_copy_bytes``: the dequantized FP history retained in
          serving="cached" mode (0 in "rebuild" mode), counted against
          compression.
        - ``fp_baseline_bytes``: a plain FP cache of the same tokens in the
          model's KV dtype.

        In "rebuild" mode a *transient* FP reconstruction of the history also
        exists while a forward pass runs (the model consumes FP tensors); the
        sustained footprint is what this reports.
        """
        s = self._native.stats(layer)
        serving_copy_bytes = 0
        if self._deq_serving[layer] is not None:
            serving_copy_bytes = sum(t.numel() * t.element_size()
                                     for t in self._deq_serving[layer])
        denom = s["kivi_bytes"] + serving_copy_bytes
        s.update(serving_copy_bytes=serving_copy_bytes,
                 fp32_bytes=s["fp_baseline_bytes"],   # backwards-compatible alias
                 compression=s["fp_baseline_bytes"] / denom if denom > 0 else 0)
        return s

    def residual(self, layer: int):
        """(keys, values) FP residual [B,H,n,D] of a layer, or None."""
        return self._native.residual(layer)

    def history(self, layer: int):
        """(packed_k, scales_k, zeros_k, packed_v, scales_v, zeros_v) views of
        the quantized prefix, or None before the first harvest."""
        return self._native.history(layer)

    def quantized_tokens(self, layer: int) -> int:
        return self._native.quant_tokens(layer)

    def flush_count(self, layer: int) -> int:
        return self._native.flushes(layer)

    def pending_tokens(self, layer: int) -> int:
        return self._native.pending_tokens(layer)

    def arena_info(self, layer: int):
        """{capacity, used, allocated_bytes, storage_ptr} of a layer's packed
        arena, or None before the first flush. storage_ptr identifies the
        allocation, so callers can tell whether a flush reallocated it."""
        cap = self._native.arena_capacity(layer)
        if cap == 0:
            return None
        h = self._native.history(layer)
        return {"capacity": cap, "used": self._native.quant_tokens(layer),
                "allocated_bytes": self._native.stats(layer)["packed_bytes"],
                "storage_ptr": h[0].data_ptr() if h is not None else None}

    @property
    def group_sizes(self) -> List[int]:
        return list(self._G)

    @property
    def reserved_tokens(self) -> int:
        return self._reserve_tokens

    @property
    def resolved_metadata_dtype(self) -> Optional[torch.dtype]:
        """Scale/zero dtype in use (None until the first tokens arrive)."""
        return self._meta_dtype

    @property
    def has_serving_view(self) -> bool:
        """Whether a full-precision copy of the history is retained between
        steps (persistent view or serving="cached" copy)."""
        return self._cache_view is not None or any(d is not None for d in self._deq_serving)

    def __repr__(self):
        return (f"KiviCache(layers={self.num_layers}, head_dim={self.head_dim}, "
                f"G={self.G if self.G is not None else self._G}, R={self.R}, "
                f"device='{self.device}', serving='{self.serving}', "
                f"async_flush={self.async_flush}, metadata_dtype={self.metadata_dtype}, "
                f"zero_copy={self.zero_copy}, engine='native')")
