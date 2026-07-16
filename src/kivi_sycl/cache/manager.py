"""KiviCache: the KIVI 2-bit asymmetric KV cache state manager."""

from typing import List, Optional, Tuple

import torch

from ..backend.ipex import ensure_ipex
from ..config.model_config import detect_model_config
from ..extension.loader import kivi_native

__all__ = ["KiviCache"]


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
        ensure_ipex()
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
            past_key_values: tuple of (key, value) per layer from the model.
                Each tensor: [B, H, Seq, D].

        Internally appends to the FP32 residual. When the residual reaches
        ``R`` tokens, flushes the oldest ``G`` tokens to 2-bit storage.
        """
        for i, (k, v) in enumerate(past_key_values):
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

            if new_k.shape[2] == 0:
                continue

            # Append to CPU residual
            if self.res_keys[i] is None:
                self.res_keys[i] = new_k
                self.res_values[i] = new_v
            else:
                self.res_keys[i] = torch.cat([self.res_keys[i], new_k], dim=2)
                self.res_values[i] = torch.cat([self.res_values[i], new_v], dim=2)

            # Flush oldest G tokens when residual >= R
            while self.res_keys[i].shape[2] >= self.R:
                self._flush_group(i)

    def get_full_cache(self) -> Tuple:
        """
        Reconstruct the full KV cache for the next model forward pass.

        Returns:
            tuple of (key, value) tensors per layer, on CPU.
            Compatible with HuggingFace ``past_key_values``.
        """
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

            full_k = torch.cat(parts_k, dim=2) if parts_k else torch.empty(0)
            full_v = torch.cat(parts_v, dim=2) if parts_v else torch.empty(0)
            full_past.append((full_k, full_v))

        return tuple(full_past)

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

    # ------------------------------------------------------------------ #
    #  Internals                                                           #
    # ------------------------------------------------------------------ #

    def _flush_group(self, layer_idx: int):
        """Quantize the oldest G tokens from the residual to 2-bit on XPU."""
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

        # --- Keys: per-channel quantization ---
        k_chunk = self.res_keys[i][:, :, :G, :]
        # Transpose to [B, H, D, G] for per-channel quant
        k_trans = k_chunk.transpose(2, 3).contiguous().to(self.device)
        batch, heads, dim, seq = k_trans.shape

        packed_k = torch.empty(
            (batch, heads, dim, seq // 4), dtype=torch.uint8, device=self.device)
        scales_k = torch.empty(
            (batch, heads, dim), dtype=torch.float32, device=self.device)
        zeros_k = torch.empty(
            (batch, heads, dim), dtype=torch.float32, device=self.device)

        kivi_native.quantize_keys(k_trans, packed_k, scales_k, zeros_k, G)

        # --- Values: per-token quantization ---
        v_chunk = self.res_values[i][:, :, :G, :].contiguous().to(self.device)
        groups_per_token = self.head_dim // G

        packed_v = torch.empty(
            (batch, heads, G, self.head_dim // 4),
            dtype=torch.uint8, device=self.device)
        scales_v = torch.empty(
            (batch, heads, G, groups_per_token),
            dtype=torch.float32, device=self.device)
        zeros_v = torch.empty(
            (batch, heads, G, groups_per_token),
            dtype=torch.float32, device=self.device)

        kivi_native.quantize_values(v_chunk, packed_v, scales_v, zeros_v,
                                     self.head_dim, G)

        # Track quantized-token accounting without retaining the packed XPU
        # tensors themselves: get_full_cache() serves exclusively from
        # _deq_history_cpu (populated below), so packed_k/packed_v are never
        # read again after dequantization — keeping them alive here would
        # duplicate the FP32 CPU history in XPU memory for no benefit.
        self._quant_token_count[i] += G
        self._quant_bytes[i] += (
            packed_k.numel() + scales_k.numel() * 4 + zeros_k.numel() * 4
            + packed_v.numel() + scales_v.numel() * 4 + zeros_v.numel() * 4
        )

        # Remove flushed tokens from residual
        self.res_keys[i] = self.res_keys[i][:, :, G:, :].contiguous()
        self.res_values[i] = self.res_values[i][:, :, G:, :].contiguous()

        # Dequantize and cache on CPU (one XPU→CPU transfer per flush)
        deq_k = torch.empty(
            (batch, heads, dim, seq), dtype=torch.float32, device=self.device)
        kivi_native.dequantize_keys(packed_k, scales_k, zeros_k, deq_k, G)
        new_k = deq_k.transpose(2, 3).to("cpu")

        deq_v = torch.empty(
            (batch, heads, G, self.head_dim), dtype=torch.float32, device=self.device)
        kivi_native.dequantize_values(packed_v, scales_v, zeros_v, deq_v,
                                       self.head_dim, G)
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
