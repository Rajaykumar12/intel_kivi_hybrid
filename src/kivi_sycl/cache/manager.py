"""KiviCache: the KIVI 2-bit asymmetric KV cache state manager."""

from typing import List, Optional, Tuple

import torch
from transformers.cache_utils import DynamicCache

from ..config.model_config import detect_model_config
from ..extension.loader import kivi_native

__all__ = ["KiviCache"]


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
    ``past_key_values`` as a tuple of (key, value) tensors.

    Typical shapes: key/value = [batch, num_kv_heads, seq_len, head_dim]

    The cache quantizes old tokens to 2-bit on XPU and keeps the most
    recent ``residual_length`` tokens in full precision for accuracy.

    Args:
        num_layers:       Number of transformer layers.
        head_dim:         Dimension per attention head.
        num_kv_heads:     Number of KV heads (optional, auto-detected on first call).
        device:           Device for quantized storage ("xpu").
        group_size:       Quantization group size G. Must divide head_dim and be >=4.
        residual_length:  Number of recent FP32 tokens to keep (R in paper).
    """

    def __init__(
        self,
        num_layers: int,
        head_dim: int,
        num_kv_heads: Optional[int] = None,
        device: str = "xpu",
        group_size: int = 32,
        residual_length: int = 128,
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

        self.device = device
        self.G = group_size
        self.R = residual_length
        self.head_dim = head_dim
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.batch_size: Optional[int] = None

        # NOTE: the 2-bit packed (packed, scales, zeros) tensors produced by
        # each flush are dequantized to `_deq_history_cpu` immediately and
        # then discarded — get_full_cache() serves exclusively from the CPU
        # FP32 history + residual, never from XPU-resident quantized storage.
        # We therefore keep only byte/token counters here (for get_stats())
        # instead of retaining the XPU tensors themselves, which would sit
        # unused for the lifetime of the cache while duplicating the FP32
        # history already cached on CPU.
        self._quant_token_count: List[int] = [0] * num_layers
        self._quant_bytes: List[int] = [0] * num_layers

        # FP32 residual buffers on CPU
        self.res_keys: List[Optional[torch.Tensor]] = [None] * num_layers
        self.res_values: List[Optional[torch.Tensor]] = [None] * num_layers

        # Dequantized history cached on CPU (updated only on flush)
        self._deq_history_cpu: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = \
            [None] * num_layers
        self._flush_count: List[int] = [0] * num_layers

        # Persistent Cache object returned by get_full_cache(), reused
        # across steps instead of rebuilt from scratch every time: a flush
        # changes the composition of history (raw residual tokens replaced
        # by their quantized-then-dequantized values), which forces a full
        # rebuild, but on the ~97% of steps that don't flush, the newest
        # token is just appended to this object in add_tokens() — matching
        # the O(1)-per-step cost a plain HuggingFace Cache already pays
        # internally, instead of re-wrapping the entire history every step.
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

        Internally appends to the FP32 residual. When the residual reaches
        ``R`` tokens, flushes the oldest ``G`` tokens to 2-bit storage.
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
                new_k = k[:, :, new_start:, :].float()
                new_v = v[:, :, new_start:, :].float()
            else:
                new_k = k.float()
                new_v = v.float()

            assert new_k.shape[2] > 0, (
                f"KiviCache desync at layer {i}: the forward pass produced "
                f"no tokens beyond the {new_start} already tracked. "
                f"add_tokens() was called twice on the same forward output, "
                f"or the model was fed a past_key_values not obtained from "
                f"get_full_cache(). Silently skipping here would drop this "
                f"step's KV and corrupt all subsequent attention."
            )

            # Append to CPU residual
            if self.res_keys[i] is None:
                self.res_keys[i] = new_k
                self.res_values[i] = new_v
            else:
                self.res_keys[i] = torch.cat([self.res_keys[i], new_k], dim=2)
                self.res_values[i] = torch.cat([self.res_values[i], new_v], dim=2)

            # No manual append to the cache view here: when the model grew
            # `self._cache_view` in place (the identity check at the top of
            # this method), the view already contains this token — calling
            # `.update()` again would append it a second time. When the
            # model returned a different object, the view was invalidated
            # above and the next get_full_cache() rebuilds it from the
            # residual, which does include this token.

            # Flush oldest G tokens when residual >= R
            while self.res_keys[i].shape[2] >= self.R:
                self._flush_group(i)
                # Flushing replaces raw residual tokens with their
                # quantized-then-dequantized values, which the persistent
                # cache view doesn't reflect — force a full rebuild on the
                # next get_full_cache() call rather than trying to patch it
                # in place (Cache objects only expose append semantics).
                self._cache_view_valid = False

    def get_full_cache(self) -> DynamicCache:
        """
        Reconstruct the full KV cache for the next model forward pass.

        Returns:
            transformers.Cache (DynamicCache) wrapping (key, value) tensors
            per layer, on CPU. Compatible with HuggingFace
            ``past_key_values``.
        """
        if self._cache_view is not None and self._cache_view_valid:
            return self._cache_view

        full_past = []
        for i in range(self.num_layers):
            parts_k, parts_v = [], []

            # Dequantized history (already on CPU, cached)
            hist = self._deq_history_cpu[i]
            if hist is not None:
                parts_k.append(hist[0])
                parts_v.append(hist[1])

            # FP32 residual (CPU)
            if self.res_keys[i] is not None and self.res_keys[i].shape[2] > 0:
                parts_k.append(self.res_keys[i])
                parts_v.append(self.res_values[i])

            # torch.cat still allocates and copies even for a single-element
            # list; skip it in the (common, since a flush hasn't merged
            # history+residual into one tensor here) case where there's
            # nothing to actually concatenate.
            full_k = parts_k[0] if len(parts_k) == 1 else (
                torch.cat(parts_k, dim=2) if parts_k else torch.empty(0))
            full_v = parts_v[0] if len(parts_v) == 1 else (
                torch.cat(parts_v, dim=2) if parts_v else torch.empty(0))
            full_past.append((full_k, full_v))

        self._cache_view = _build_cache(full_past)
        self._cache_view_valid = True
        return self._cache_view

    def reset(self):
        """Clear all cached data. Reuse the same KiviCache for a new sequence."""
        for i in range(self.num_layers):
            self._quant_token_count[i] = 0
            self._quant_bytes[i] = 0
            self.res_keys[i] = None
            self.res_values[i] = None
            self._deq_history_cpu[i] = None
            self._flush_count[i] = 0
        self.batch_size = None
        self._cache_view = None
        self._cache_view_valid = False

    # ------------------------------------------------------------------ #
    #  Internals                                                           #
    # ------------------------------------------------------------------ #

    def _flush_group(self, layer_idx: int):
        """Round the oldest G tokens from the residual to 2-bit fidelity on XPU."""
        i = layer_idx
        G = self.G

        # Defense in depth: __init__ already asserts group_size <=
        # residual_length, but re-check here too so a future desync between
        # R and G fails loudly instead of silently under-slicing the residual
        # (PyTorch slicing past the end of a dim clips instead of raising).
        assert self.res_keys[i].shape[2] >= G, (
            f"Cannot flush {G} tokens: residual only has "
            f"{self.res_keys[i].shape[2]} at layer {i}."
        )

        # get_full_cache() serves exclusively from _deq_history_cpu (populated
        # below) — the packed/scale/zero-point representation is never read
        # again after this call. quant_dequant_roundtrip_* rounds every value
        # to 2-bit fidelity and writes the reconstructed float straight back
        # out in a single fused kernel launch, without ever allocating or
        # writing the packed byte / scale / zero-point buffers at all — see
        # docs/sycl_kernel_interface.md.

        # --- Keys: per-channel quantization ---
        k_chunk = self.res_keys[i][:, :, :G, :]
        # Transpose to [B, H, D, G] for per-channel quant
        k_trans = k_chunk.transpose(2, 3).contiguous().to(self.device)
        batch, heads, dim, seq = k_trans.shape

        deq_k = torch.empty_like(k_trans)
        kivi_native.quant_dequant_roundtrip_keys(k_trans, deq_k, G)

        # --- Values: per-token quantization ---
        v_chunk = self.res_values[i][:, :, :G, :].contiguous().to(self.device)
        groups_per_token = self.head_dim // G

        deq_v = torch.empty_like(v_chunk)
        kivi_native.quant_dequant_roundtrip_values(v_chunk, deq_v, self.head_dim, G)

        # Track quantized-token accounting analytically (bytes a real 2-bit
        # store would cost) since no packed/scales/zeros tensors exist to
        # read .numel() from in the fused round-trip path.
        self._quant_token_count[i] += G
        self._quant_bytes[i] += (
            batch * heads * dim * (seq // 4)       # packed_k
            + batch * heads * dim * 4 * 2           # scales_k + zeros_k
            + batch * heads * G * (self.head_dim // 4)     # packed_v
            + batch * heads * G * groups_per_token * 4 * 2  # scales_v + zeros_v
        )

        # Remove flushed tokens from residual
        self.res_keys[i] = self.res_keys[i][:, :, G:, :].contiguous()
        self.res_values[i] = self.res_values[i][:, :, G:, :].contiguous()

        new_k = deq_k.transpose(2, 3).to("cpu")
        new_v = deq_v.to("cpu")

        # Append to CPU history
        cur = self._deq_history_cpu[i]
        if cur is not None:
            hist_k = torch.cat([cur[0], new_k], dim=2)
            hist_v = torch.cat([cur[1], new_v], dim=2)
        else:
            hist_k, hist_v = new_k, new_v

        self._flush_count[i] += 1
        self._deq_history_cpu[i] = (hist_k, hist_v)

    # ------------------------------------------------------------------ #
    #  Stats / Debug                                                       #
    # ------------------------------------------------------------------ #

    @property
    def total_flushes(self) -> int:
        return sum(self._flush_count)

    def get_stats(self, layer: int = 0) -> dict:
        """
        Get memory/token statistics for a given layer.

        NOTE: `kivi_bytes` reports the logical 2-bit-quantized footprint
        (what a byte-for-byte 2-bit store would cost) plus the FP32 residual
        window. It does NOT include `_deq_history_cpu`, which holds a full
        FP32 CPU copy of the entire dequantized history for serving — actual
        process memory for long sequences is therefore higher than this
        figure suggests. See the "Logical & Accuracy" section of
        CODE_REVIEW.md for the full explanation of this tradeoff.
        """
        B = self.batch_size or 1
        H = self.num_kv_heads or 1
        i = layer

        quant_tokens = self._quant_token_count[i]
        res_tokens = self.res_keys[i].shape[2] if self.res_keys[i] is not None else 0
        total = quant_tokens + res_tokens

        quant_bytes = self._quant_bytes[i]
        res_bytes = 0
        if self.res_keys[i] is not None:
            res_bytes = (self.res_keys[i].numel() + self.res_values[i].numel()) * 4

        fp32_bytes = 2 * B * H * total * self.head_dim * 4

        return {
            "total_tokens": total,
            "quantized_tokens": quant_tokens,
            "residual_tokens": res_tokens,
            "kivi_bytes": quant_bytes + res_bytes,
            "fp32_bytes": fp32_bytes,
            "compression": fp32_bytes / (quant_bytes + res_bytes)
            if (quant_bytes + res_bytes) > 0 else 0,
        }

    def __repr__(self):
        return (f"KiviCache(layers={self.num_layers}, head_dim={self.head_dim}, "
                f"G={self.G}, R={self.R}, device='{self.device}')")
