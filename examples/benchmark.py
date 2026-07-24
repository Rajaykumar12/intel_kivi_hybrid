"""
KIVI Benchmark — demonstrates plug-and-play usage of kivi_sycl
================================================================
Run: python examples/benchmark.py [model_name] [--tokens N] [--R N] [--G N]

Examples:
    python examples/benchmark.py                          # GPT-2 (default)
    python examples/benchmark.py gpt2-medium              # GPT-2 Medium
    python examples/benchmark.py meta-llama/Llama-2-7b-hf --tokens 100
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import sys
import argparse

from kivi_sycl import KiviCache, generate


def benchmark(model_id, num_gen_tokens, G, R, prompt):
    if not torch.xpu.is_available():
        print("Error: XPU not found.")
        sys.exit(1)

    print(f"XPU Device: {torch.xpu.get_device_name(0)}")
    print(f"\nLoading model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Auto-detect model config
    cache = KiviCache.from_model(model, device="xpu", group_size=G, residual_length=R)
    print(f"  {cache}")

    input_ids_orig = tokenizer(prompt, return_tensors="pt").input_ids
    print(f"  Prompt tokens: {input_ids_orig.shape[1]}")
    print(f"  Generating {num_gen_tokens} tokens...")

    # -----------------------------------------------------------
    # 1. FP32 Reference (no quantization)
    # -----------------------------------------------------------
    print(f"\nPrompt: '{prompt[:80]}{'...' if len(prompt) > 80 else ''}'")
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
            if tokenizer.eos_token_id and ref_input_ids.item() == tokenizer.eos_token_id:
                break
    elapsed_ref = time.time() - start_ref
    actual_ref_tokens = len(ref_tokens)
    ref_text = tokenizer.decode(torch.cat(ref_tokens, dim=1)[0])
    print(f"  {actual_ref_tokens} tokens in {elapsed_ref:.2f}s "
          f"({actual_ref_tokens/elapsed_ref:.1f} tok/s)")
    print(f"  Output: {ref_text[:200]}{'...' if len(ref_text) > 200 else ''}")

    # -----------------------------------------------------------
    # 2. KIVI 2-bit Generation (plug-and-play)
    # -----------------------------------------------------------
    print(f"\n--- KIVI 2-bit (G={G}, R={R}) ---")
    print("Generating: ", end="", flush=True)

    input_ids = input_ids_orig.clone()

    # Prefill
    with torch.no_grad():
        out = model(input_ids, use_cache=True)
    cache.add_tokens(out.past_key_values)
    input_ids = out.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    all_tokens = [input_ids]

    start = time.time()
    # Prefill already produced new token #1 — loop num_gen_tokens - 1 more
    # times so the total matches the FP32 reference loop above.
    for step in range(1, num_gen_tokens):
        past_key_values = cache.get_full_cache()

        with torch.no_grad():
            out = model(input_ids, past_key_values=past_key_values, use_cache=True)

        cache.add_tokens(out.past_key_values)
        input_ids = out.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        all_tokens.append(input_ids)
        if (step + 1) % 10 == 0:
            print(".", end="", flush=True)
        if tokenizer.eos_token_id and input_ids.item() == tokenizer.eos_token_id:
            break

    elapsed = time.time() - start
    actual_kivi_tokens = len(all_tokens)
    generated = tokenizer.decode(torch.cat(all_tokens, dim=1)[0])

    print(f"\n  {actual_kivi_tokens} tokens in {elapsed:.2f}s "
          f"({actual_kivi_tokens/elapsed:.1f} tok/s)")
    print(f"  Output: {generated[:200]}{'...' if len(generated) > 200 else ''}")

    # -----------------------------------------------------------
    # 3. Cache Statistics
    # -----------------------------------------------------------
    s = cache.get_stats(layer=0)

    print(f"\n{'='*60}")
    print(f"Cache stats (layer 0):")
    print(f"  Total tokens:      {s['total_tokens']}")
    print(f"  Quantized tokens:  {s['quantized_tokens']}")
    print(f"  Residual tokens:   {s['residual_tokens']}")
    print(f"  KIVI memory:       {s['kivi_bytes']/1024:.1f} KB")
    print(f"  FP32 baseline:     {s['fp32_bytes']/1024:.1f} KB")
    print(f"  Compression ratio: {s['compression']:.2f}x")
    print(f"\nPerformance:")
    print(f"  FP32 speed:        {actual_ref_tokens/elapsed_ref:.1f} tok/s")
    print(f"  KIVI speed:        {actual_kivi_tokens/elapsed:.1f} tok/s")
    kivi_toks = actual_kivi_tokens / elapsed
    ref_toks = actual_ref_tokens / elapsed_ref
    if kivi_toks >= ref_toks:
        print(f"  KIVI speedup:      {(kivi_toks/ref_toks - 1)*100:.0f}%")
    else:
        overhead_pct = ((1.0 / kivi_toks) / (1.0 / ref_toks) - 1) * 100
        print(f"  KIVI overhead:     {overhead_pct:.0f}%")
    print(f"  Flushes:           {cache.total_flushes} "
          f"(across {cache.num_layers} layers)")
    print(f"{'='*60}")

    # Quality comparison
    ref_ids = torch.cat(ref_tokens, dim=1)[0]
    kivi_ids = torch.cat(all_tokens, dim=1)[0]
    min_len = min(len(ref_ids), len(kivi_ids))
    match = (ref_ids[:min_len] == kivi_ids[:min_len]).float().mean().item()
    diverge_at = min_len
    for t in range(min_len):
        if ref_ids[t] != kivi_ids[t]:
            diverge_at = t
            break
    print(f"\nQuality comparison:")
    print(f"  Token match rate:     {match*100:.1f}%")
    print(f"  First divergence at:  token {diverge_at} / {min_len}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="KIVI 2-bit KV Cache Benchmark")
    parser.add_argument("model", nargs="?", default="gpt2",
                        help="HuggingFace model name/path (default: gpt2)")
    parser.add_argument("--tokens", type=int, default=200,
                        help="Number of tokens to generate (default: 200)")
    parser.add_argument("--G", type=int, default=32,
                        help="Quantization group size (default: 32)")
    parser.add_argument("--R", type=int, default=64,
                        help="Residual window size (default: 64)")
    parser.add_argument("--prompt", type=str,
                        default="In a shocking finding, scientists discovered "
                                "a herd of unicorns living in a remote valley. "
                                "They had never been seen before by humans. "
                                "The unicorns were",
                        help="Input prompt")
    args = parser.parse_args()

    benchmark(args.model, args.tokens, args.G, args.R, args.prompt)