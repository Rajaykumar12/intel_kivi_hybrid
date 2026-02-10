"""
kivi_cache — Plug-and-play KIVI 2-bit KV Cache for HuggingFace Models
======================================================================
Drop-in replacement for the standard KV cache in any HuggingFace
CausalLM. Works with GPT-2, LLaMA, Mistral, Phi, Qwen, Falcon, etc.

Usage:
    from kivi_cache import KiviCache, generate

    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

    # Option 1: One-liner generation
    text = generate(model, tokenizer, "Once upon a time", max_new_tokens=200)

    # Option 2: Manual control
    cache = KiviCache.from_model(model)
    # ... use cache.add_tokens() / cache.get_full_cache() in your own loop

Paper: "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache"
       (arXiv:2402.02750)
"""

import torch
import intel_extension_for_pytorch as ipex  # registers XPU backend with PyTorch
import kivi_sycl
from typing import Optional, Tuple, List, Union

__all__ = ["KiviCache", "generate"]


def _detect_model_config(model) -> dict:
    """
    Auto-detect num_layers, num_heads, and head_dim from any HuggingFace model.
    Supports GPT-2, GPT-J, GPT-NeoX, LLaMA, Mistral, Phi, Qwen, Falcon,
    Gemma, StableLM, MPT, BLOOM, OPT, and more.
    """
    config = model.config

    # --- num_layers ---
    for attr in ("n_layer", "num_hidden_layers", "num_layers", "n_layers"):
        if hasattr(config, attr):
            num_layers = getattr(config, attr)
            break
    else:
        raise ValueError(
            f"Cannot detect num_layers from {type(config).__name__}. "
            f"Pass num_layers= explicitly.")

    # --- num_heads (for KV — may differ from query heads in GQA models) ---
    num_kv_heads = None
    for attr in ("num_key_value_heads", "num_kv_heads"):
        if hasattr(config, attr):
            num_kv_heads = getattr(config, attr)
            break
    if num_kv_heads is None:
        for attr in ("n_head", "num_attention_heads", "num_heads"):
            if hasattr(config, attr):
                num_kv_heads = getattr(config, attr)
                break
    if num_kv_heads is None:
        raise ValueError(
            f"Cannot detect num_heads from {type(config).__name__}. "
            f"Pass num_kv_heads= explicitly.")

    # --- head_dim ---
    if hasattr(config, "head_dim"):
        head_dim = config.head_dim
    else:
        hidden_size = getattr(config, "hidden_size",
                              getattr(config, "n_embd", None))
        num_q_heads = getattr(config, "num_attention_heads",
                              getattr(config, "n_head", None))
        if hidden_size and num_q_heads:
            head_dim = hidden_size // num_q_heads
        else:
            raise ValueError(
                f"Cannot detect head_dim from {type(config).__name__}. "
                f"Pass head_dim= explicitly.")

    return {
        "num_layers": num_layers,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
    }


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

        self.device = device
        self.G = group_size
        self.R = residual_length
        self.head_dim = head_dim
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.batch_size: Optional[int] = None

        # Quantized storage on XPU: list of (packed, scales, zeros) per layer
        self.quant_keys: List[list] = [[] for _ in range(num_layers)]
        self.quant_values: List[list] = [[] for _ in range(num_layers)]

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
        cfg = _detect_model_config(model)
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

            # Extract only NEW tokens (model returns full cache)
            if self.res_keys[i] is not None:
                existing_len = self.res_keys[i].shape[2]
                quant_len = sum(p.shape[3] * 4 for p, _, _ in self.quant_keys[i])
                new_start = quant_len + existing_len
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
            self.quant_keys[i].clear()
            self.quant_values[i].clear()
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

        kivi_sycl.quantize_keys(k_trans, packed_k, scales_k, zeros_k, G)
        self.quant_keys[i].append((packed_k, scales_k, zeros_k))

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

        kivi_sycl.quantize_values(v_chunk, packed_v, scales_v, zeros_v,
                                  self.head_dim, G)
        self.quant_values[i].append((packed_v, scales_v, zeros_v))

        # Remove flushed tokens from residual
        self.res_keys[i] = self.res_keys[i][:, :, G:, :].contiguous()
        self.res_values[i] = self.res_values[i][:, :, G:, :].contiguous()

        # Dequantize and cache on CPU (one XPU→CPU transfer per flush)
        deq_k = torch.empty(
            (batch, heads, dim, seq), dtype=torch.float32, device=self.device)
        kivi_sycl.dequantize_keys(packed_k, scales_k, zeros_k, deq_k, G)
        new_k = deq_k.transpose(2, 3).to("cpu")

        deq_v = torch.empty(
            (batch, heads, G, self.head_dim),
            dtype=torch.float32, device=self.device)
        kivi_sycl.dequantize_values(packed_v, scales_v, zeros_v, deq_v,
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
        """Get memory/token statistics for a given layer."""
        B = self.batch_size or 1
        H = self.num_kv_heads or 1
        i = layer

        quant_tokens = sum(p.shape[3] * 4 for p, _, _ in self.quant_keys[i])
        res_tokens = self.res_keys[i].shape[2] if self.res_keys[i] is not None else 0
        total = quant_tokens + res_tokens

        quant_bytes = sum(
            p.numel() + s.numel() * 4 + z.numel() * 4
            for p, s, z in self.quant_keys[i]
        ) + sum(
            p.numel() + s.numel() * 4 + z.numel() * 4
            for p, s, z in self.quant_values[i]
        )
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


# ====================================================================== #
#  High-level generate() — one-liner plug-and-play                        #
# ====================================================================== #

@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    group_size: int = 32,
    residual_length: int = 128,
    device: str = "xpu",
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    do_sample: bool = False,
    verbose: bool = False,
) -> str:
    """
    Generate text with KIVI 2-bit KV cache quantization.

    Works with any HuggingFace CausalLM model. The model stays on CPU;
    quantization kernels run on Intel XPU via SYCL.

    Args:
        model:            HuggingFace CausalLM (on CPU).
        tokenizer:        Matching tokenizer.
        prompt:           Input text.
        max_new_tokens:   Number of tokens to generate.
        group_size:       Quantization group size G (default 32).
        residual_length:  FP32 residual window R (default 128).
        device:           XPU device for quantization (default "xpu").
        temperature:      Sampling temperature (1.0 = greedy with do_sample=False).
        top_k:            Top-k filtering (None = disabled).
        do_sample:        Whether to sample (False = greedy argmax).
        verbose:          Print progress dots.

    Returns:
        Generated text (prompt + completion).

    Example:
        >>> from kivi_cache import generate
        >>> text = generate(model, tokenizer, "Hello world", max_new_tokens=100)
    """
    cache = KiviCache.from_model(
        model, device=device, group_size=group_size,
        residual_length=residual_length,
    )

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    # Prefill
    out = model(input_ids, use_cache=True)
    cache.add_tokens(out.past_key_values)
    next_id = _sample(out.logits[:, -1, :], temperature, top_k, do_sample)
    all_ids = [next_id]

    # Decode
    for step in range(max_new_tokens):
        past = cache.get_full_cache()
        out = model(next_id, past_key_values=past, use_cache=True)
        cache.add_tokens(out.past_key_values)
        next_id = _sample(out.logits[:, -1, :], temperature, top_k, do_sample)
        all_ids.append(next_id)

        if verbose and (step + 1) % 10 == 0:
            print(".", end="", flush=True)

        # Stop on EOS
        if tokenizer.eos_token_id is not None and next_id.item() == tokenizer.eos_token_id:
            break

    if verbose:
        print()

    generated_ids = torch.cat(all_ids, dim=1)
    full_ids = torch.cat([input_ids, generated_ids], dim=1)
    return tokenizer.decode(full_ids[0], skip_special_tokens=True)


def _sample(
    logits: torch.Tensor,
    temperature: float,
    top_k: Optional[int],
    do_sample: bool,
) -> torch.Tensor:
    """Sample or argmax from logits."""
    if not do_sample:
        return logits.argmax(dim=-1).unsqueeze(1)

    logits = logits / max(temperature, 1e-8)
    if top_k is not None:
        topk_vals, _ = logits.topk(top_k, dim=-1)
        logits[logits < topk_vals[:, -1:]] = float("-inf")
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
