"""
KIVI Benchmark: Asymmetric 2-bit KV Cache Quantization for Intel XPU
=====================================================================
Implements the KIVI algorithm (arXiv:2402.02750) as a KV cache manager
that integrates with HuggingFace Transformers models.

Key design from the paper:
  - Keys:   quantized per-channel (outliers along channel dimension)
  - Values: quantized per-token   (outliers along token dimension)
  - Residual window: last R tokens kept in full precision for accuracy

Memory layout:
  Keys are transposed to [B, H, D, Seq] before per-channel quantization,
  then transposed back after dequantization.
  Values stay as [B, H, Seq, D] for per-token quantization.
"""

import torch
import intel_extension_for_pytorch as ipex
import kivi_sycl
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import sys


class KiviManager:
    """
    Manages quantized KV cache using the KIVI algorithm.
    
    During generation, new KV pairs are appended to a residual buffer.
    When the residual exceeds threshold R, groups of G tokens are
    quantized to 2-bit and stored. During attention, quantized history
    is dequantized and concatenated with the full-precision residual.
    
    Args:
        num_layers:      Number of transformer layers
        head_dim:        Dimension per attention head
        device:          Target device ("xpu" for Intel GPU)
        group_size:      Number of elements per quantization group (must divide head_dim, must be ≥4)
        residual_length: Number of recent tokens kept in full precision (R in the paper)
    """
    def __init__(self, num_layers, head_dim, device="xpu", group_size=32, residual_length=128):
        assert head_dim % group_size == 0, \
            f"head_dim ({head_dim}) must be divisible by group_size ({group_size})"
        assert group_size >= 4 and group_size % 4 == 0, \
            f"group_size ({group_size}) must be ≥4 and divisible by 4"
        
        self.device = device
        self.G = group_size
        self.R = residual_length
        self.head_dim = head_dim
        self.num_layers = num_layers
        self.num_heads = None   # populated on first add_tokens
        self.batch_size = None  # populated on first add_tokens

        # Storage: List of (Packed, Scales, Zeros) tuples per layer (on XPU)
        self.quant_keys = [[] for _ in range(num_layers)]
        self.quant_values = [[] for _ in range(num_layers)]

        # Residual buffers: kept on CPU to avoid per-step XPU sync.
        # Only moved to XPU during flush for SYCL quantization.
        self.res_keys = [None for _ in range(num_layers)]   # [B, H, Seq, D] fp32 CPU
        self.res_values = [None for _ in range(num_layers)]  # [B, H, Seq, D] fp32 CPU

        # Dequantized history cache (on CPU). Updated only on flush.
        # Entry: (flush_count, deq_keys_cpu, deq_values_cpu) or None
        self._deq_history_cpu = [None for _ in range(num_layers)]
        self._flush_count = [0 for _ in range(num_layers)]

    def add_tokens(self, model_past_key_values):
        """
        Ingest new KV pairs from a model forward pass.
        
        Residual stays on CPU (model output is already CPU). No XPU
        transfers happen here — only pure CPU tensor slicing/cat.
        
        When the residual buffer reaches size R, groups of G tokens
        are flushed to quantized storage (XPU work happens in _flush_group).
        """
        for i, (k, v) in enumerate(model_past_key_values):
            # k, v shape: [B, H, Seq, D] — already on CPU from model
            if self.batch_size is None:
                self.batch_size = k.shape[0]
                self.num_heads = k.shape[1]
            if self.res_keys[i] is not None:
                # Decode step: model returns full cache, extract only the new token
                existing_len = self.res_keys[i].shape[2]
                quant_len = sum(p.shape[3] * 4 for p, _, _ in self.quant_keys[i])
                new_start = quant_len + existing_len
                new_k = k[:, :, new_start:, :].float()
                new_v = v[:, :, new_start:, :].float()
            else:
                # First call (prefill): take all tokens
                new_k = k.float()
                new_v = v.float()

            if new_k.shape[2] == 0:
                continue

            # Append to residual buffer (CPU-only, no XPU sync)
            if self.res_keys[i] is None:
                self.res_keys[i] = new_k
                self.res_values[i] = new_v
            else:
                self.res_keys[i] = torch.cat([self.res_keys[i], new_k], dim=2)
                self.res_values[i] = torch.cat([self.res_values[i], new_v], dim=2)

            # Flush groups of G tokens when residual exceeds R
            while self.res_keys[i].shape[2] >= self.R:
                self._flush_group(i)

    def _flush_group(self, layer_idx):
        """
        Quantize the oldest G tokens from the residual buffer.
        
        This is the ONLY place where CPU→XPU transfers happen:
        1. Move G tokens to XPU for SYCL quantization
        2. Keep packed result on XPU
        3. Dequantize and transfer result back to CPU for the history cache
        """
        i = layer_idx
        num_flush = self.G
        
        # ---- KEYS: Per-Channel Quantization ----
        # Take first G tokens from CPU residual, move to XPU
        k_to_quant = self.res_keys[i][:, :, :num_flush, :]
        k_trans = k_to_quant.transpose(2, 3).contiguous().to(self.device)  # [B, H, D, G]
        
        batch, heads, dim, seq = k_trans.shape
        packed_k = torch.empty(
            (batch, heads, dim, seq // 4), dtype=torch.uint8, device=self.device)
        scales_k = torch.empty(
            (batch, heads, dim), dtype=torch.float32, device=self.device)
        zeros_k = torch.empty(
            (batch, heads, dim), dtype=torch.float32, device=self.device)
        
        kivi_sycl.quantize_keys(k_trans, packed_k, scales_k, zeros_k, self.G)
        self.quant_keys[i].append((packed_k, scales_k, zeros_k))
        
        # ---- VALUES: Per-Token Quantization ----
        v_to_quant = self.res_values[i][:, :, :num_flush, :].contiguous().to(self.device)
        groups_per_token = self.head_dim // self.G
        
        packed_v = torch.empty(
            (batch, heads, num_flush, self.head_dim // 4),
            dtype=torch.uint8, device=self.device)
        scales_v = torch.empty(
            (batch, heads, num_flush, groups_per_token),
            dtype=torch.float32, device=self.device)
        zeros_v = torch.empty(
            (batch, heads, num_flush, groups_per_token),
            dtype=torch.float32, device=self.device)
        
        kivi_sycl.quantize_values(
            v_to_quant, packed_v, scales_v, zeros_v, self.head_dim, self.G)
        self.quant_values[i].append((packed_v, scales_v, zeros_v))
        
        # Remove flushed tokens from residual (CPU)
        self.res_keys[i] = self.res_keys[i][:, :, num_flush:, :].contiguous()
        self.res_values[i] = self.res_values[i][:, :, num_flush:, :].contiguous()

        # Dequantize the newly quantized group and append to CPU history cache.
        # This is a single XPU→CPU transfer per flush (not per step).
        deq_k_xpu = torch.empty(
            (batch, heads, dim, seq), dtype=torch.float32, device=self.device)
        kivi_sycl.dequantize_keys(packed_k, scales_k, zeros_k, deq_k_xpu, self.G)
        new_deq_k = deq_k_xpu.transpose(2, 3).to("cpu")  # [B, H, G, D]

        deq_v_xpu = torch.empty(
            (batch, heads, num_flush, self.head_dim),
            dtype=torch.float32, device=self.device)
        kivi_sycl.dequantize_values(
            packed_v, scales_v, zeros_v, deq_v_xpu, self.head_dim, self.G)
        new_deq_v = deq_v_xpu.to("cpu")  # [B, H, G, D]

        # Append to the CPU-side dequantized history
        cur = self._deq_history_cpu[i]
        if cur is not None:
            old_k, old_v = cur[1], cur[2]
            hist_k = torch.cat([old_k, new_deq_k], dim=2)
            hist_v = torch.cat([old_v, new_deq_v], dim=2)
        else:
            hist_k, hist_v = new_deq_k, new_deq_v

        self._flush_count[i] += 1
        self._deq_history_cpu[i] = (self._flush_count[i], hist_k, hist_v)

    def get_full_cache(self):
        """
        Reconstruct the full KV cache for model consumption.
        
        Zero-copy fast path: everything is already on CPU.
        - Dequantized history: cached on CPU, updated only on flush
        - Residual: always on CPU
        - Just a CPU torch.cat per layer, no XPU interaction at all
        
        Returns:
            tuple of (key, value) tensors per layer, on CPU.
        """
        full_past = []
        for i in range(self.num_layers):
            parts_k = []
            parts_v = []

            # Dequantized quantized history (already on CPU)
            hist = self._deq_history_cpu[i]
            if hist is not None:
                parts_k.append(hist[1])
                parts_v.append(hist[2])

            # Residual (already on CPU)
            if self.res_keys[i] is not None and self.res_keys[i].shape[2] > 0:
                parts_k.append(self.res_keys[i])
                parts_v.append(self.res_values[i])

            full_k = torch.cat(parts_k, dim=2) if parts_k else torch.empty(0)
            full_v = torch.cat(parts_v, dim=2) if parts_v else torch.empty(0)

            full_past.append((full_k, full_v))

        return tuple(full_past)

    def get_cache_stats(self):
        """Return statistics about the current cache state."""
        B = self.batch_size or 1
        H = self.num_heads or 1
        stats = []
        for i in range(self.num_layers):
            quant_tokens = sum(p.shape[3] * 4 for p, _, _ in self.quant_keys[i])
            res_tokens = self.res_keys[i].shape[2] if self.res_keys[i] is not None else 0
            total_tokens = quant_tokens + res_tokens
            
            # KIVI memory: packed (uint8) + scales (fp32) + zeros (fp32)
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
            
            # FP32 baseline: K + V, each [B, H, total_tokens, head_dim] × 4 bytes
            fp32_bytes = 2 * B * H * total_tokens * self.head_dim * 4
            
            stats.append({
                "layer": i,
                "quantized_tokens": quant_tokens,
                "residual_tokens": res_tokens,
                "total_tokens": total_tokens,
                "kivi_bytes": quant_bytes + res_bytes,
                "fp32_bytes": fp32_bytes,
            })
        return stats


if __name__ == "__main__":
    if not torch.xpu.is_available():
        print("Error: XPU not found.")
        sys.exit(1)

    print(f"XPU Device: {torch.xpu.get_device_name(0)}")

    MODEL_ID = "gpt2"
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    head_dim = model.config.n_embd // model.config.n_head
    print(f"Model: {MODEL_ID}")
    print(f"  Layers: {model.config.n_layer}, Heads: {model.config.n_head}, Head dim: {head_dim}")

    # R = residual window size. Quantization flushes happen when
    # residual exceeds R tokens. For GPT-2, R=64 gives a good balance
    # between quality and actually exercising the quantization path.
    # For production with larger models (7B+), R=128 is recommended.
    R = 64
    G = 32  # group size for quantization
    manager = KiviManager(
        num_layers=model.config.n_layer,
        head_dim=head_dim,
        device="xpu",
        group_size=G,
        residual_length=R,
    )
    print(f"  KIVI config: G={G}, R={R}")

    prompt = "In a shocking finding, scientists discovered a herd of unicorns living in a remote valley. They had never been seen before by humans. The unicorns were"
    input_ids_orig = tokenizer(prompt, return_tensors="pt").input_ids
    num_gen_tokens = 200
    print(f"  Prompt tokens: {input_ids_orig.shape[1]}")
    print(f"  Generating {num_gen_tokens} tokens...")

    # -----------------------------------------------------------
    # 1. FP32 Reference (no quantization)
    # -----------------------------------------------------------
    print(f"\nPrompt: '{prompt}'")
    print(f"\n--- FP32 Reference (no quantization) ---")
    ref_input_ids = input_ids_orig.clone()
    ref_past = None
    ref_tokens = []

    start_ref = time.time()
    with torch.no_grad():
        for step in range(num_gen_tokens):
            out = model(ref_input_ids, past_key_values=ref_past, use_cache=True)
            ref_past = out.past_key_values
            ref_input_ids = out.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            ref_tokens.append(ref_input_ids)
    elapsed_ref = time.time() - start_ref
    ref_text = tokenizer.decode(torch.cat(ref_tokens, dim=1)[0])
    print(f"  {num_gen_tokens} tokens in {elapsed_ref:.2f}s ({num_gen_tokens/elapsed_ref:.1f} tok/s)")
    print(f"  Output: {ref_text[:200]}{'...' if len(ref_text) > 200 else ''}")

    # -----------------------------------------------------------
    # 2. KIVI 2-bit Generation
    # -----------------------------------------------------------
    print(f"\n--- KIVI 2-bit Asymmetric Quantization (G={G}, R={R}) ---")
    print("Generating: ", end="", flush=True)

    input_ids = input_ids_orig.clone()
    past_key_values = None

    # Prefill
    with torch.no_grad():
        out = model(input_ids, use_cache=True)
    manager.add_tokens(out.past_key_values)
    input_ids = out.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)

    all_tokens = [input_ids]

    start = time.time()
    for step in range(num_gen_tokens):
        past_key_values = manager.get_full_cache()

        with torch.no_grad():
            out = model(input_ids, past_key_values=past_key_values, use_cache=True)

        manager.add_tokens(out.past_key_values)
        input_ids = out.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        all_tokens.append(input_ids)
        if (step + 1) % 10 == 0:
            print(".", end="", flush=True)

    elapsed = time.time() - start
    generated = tokenizer.decode(torch.cat(all_tokens, dim=1)[0])

    print(f"\n  {num_gen_tokens} tokens in {elapsed:.2f}s "
          f"({num_gen_tokens/elapsed:.1f} tok/s)")
    print(f"  Output: {generated[:200]}{'...' if len(generated) > 200 else ''}")

    # -----------------------------------------------------------
    # 3. Cache Statistics
    # -----------------------------------------------------------
    stats = manager.get_cache_stats()
    s = stats[0]
    compression = s['fp32_bytes'] / s['kivi_bytes'] if s['kivi_bytes'] > 0 else 0

    # Caching stats
    total_flushes = sum(manager._flush_count)
    total_steps = num_gen_tokens
    # XPU work only happens during flush (quantize + dequantize + transfer)
    # All other steps are pure CPU (torch.cat of cached history + residual)
    xpu_free_steps = total_steps - total_flushes

    print(f"\n{'='*60}")
    print(f"Cache stats (layer 0, per-head × {model.config.n_head} heads):")
    print(f"  Total tokens:      {s['total_tokens']}")
    print(f"  Quantized tokens:  {s['quantized_tokens']}")
    print(f"  Residual tokens:   {s['residual_tokens']}")
    print(f"  KIVI memory:       {s['kivi_bytes']/1024:.1f} KB")
    print(f"  FP32 baseline:     {s['fp32_bytes']/1024:.1f} KB")
    print(f"  Compression ratio: {compression:.2f}x")
    print(f"\nPerformance:")
    print(f"  FP32 speed:        {num_gen_tokens/elapsed_ref:.1f} tok/s")
    print(f"  KIVI speed:        {num_gen_tokens/elapsed:.1f} tok/s")
    kivi_toks = num_gen_tokens / elapsed
    ref_toks = num_gen_tokens / elapsed_ref
    overhead_pct = ((1.0 / kivi_toks) / (1.0 / ref_toks) - 1) * 100
    print(f"  KIVI overhead:     {overhead_pct:.0f}%")
    print(f"  XPU-free steps:    {xpu_free_steps}/{total_steps} "
          f"({100*xpu_free_steps/total_steps:.0f}% pure CPU)")
    print(f"  Flushes:           {total_flushes} (across {model.config.n_layer} layers)")
    print(f"{'='*60}")

    # Quality comparison
    ref_ids = torch.cat(ref_tokens, dim=1)[0]
    kivi_ids = torch.cat(all_tokens, dim=1)[0]
    min_len = min(len(ref_ids), len(kivi_ids))
    match = (ref_ids[:min_len] == kivi_ids[:min_len]).float().mean().item()
    # Find first divergence
    diverge_at = min_len
    for t in range(min_len):
        if ref_ids[t] != kivi_ids[t]:
            diverge_at = t
            break
    print(f"\nQuality comparison:")
    print(f"  Token match rate:     {match*100:.1f}%")
    print(f"  First divergence at:  token {diverge_at} / {min_len}")
    if diverge_at < min_len:
        print(f"  NOTE: GPT-2 (small model, head_dim=64) is sensitive to 2-bit")
        print(f"        quantization. KIVI is designed for larger models (7B+)")
        print(f"        where channel/token outlier patterns are more pronounced.")