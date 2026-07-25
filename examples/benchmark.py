"""
KIVI Benchmark — demonstrates plug-and-play usage of kivi_sycl
================================================================
Run: python examples/benchmark.py [model_name] [--tokens N] [--R N] [--G N]

Examples:
    python examples/benchmark.py                          # TinyLlama-1.1B (default)
    python examples/benchmark.py gpt2                     # quick check, no 2.2 GB download
    python examples/benchmark.py Qwen/Qwen2-0.5B --tokens 100

Defaults to TinyLlama-1.1B rather than gpt2 because the point of this script is
the measured compression figure, and gpt2's KV cache is far too small for that
number to mean anything (docs/status_and_roadmap.md 2.7). Pass `gpt2` explicitly
for a fast smoke test on a machine without the weights cached.

Note this benchmark generates from a short prompt, so it reports compression in
the short-context regime. For the long-context numbers that actually motivate
KIVI, use examples/eval_longctx.py.
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
    ref_tokens = []

    # Prefill is deliberately OUTSIDE the timed region, matching the KIVI path
    # below. Timing the prompt forward on one side only inflates the other
    # side's apparent speed — the same asymmetry in eval_longctx.py once made
    # KIVI look 2.5x *faster* than the FP baseline at 2k context.
    with torch.no_grad():
        out = model(ref_input_ids, use_cache=True)
    ref_past = out.past_key_values
    ref_input_ids = out.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    ref_tokens.append(ref_input_ids)
    ref_hit_eos = (tokenizer.eos_token_id is not None
                   and ref_input_ids.item() == tokenizer.eos_token_id)

    start_ref = time.time()
    with torch.no_grad():
        for step in range(1, num_gen_tokens):
            if ref_hit_eos:
                break
            out = model(ref_input_ids, past_key_values=ref_past, use_cache=True)
            ref_past = out.past_key_values
            ref_input_ids = out.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            ref_tokens.append(ref_input_ids)
            if tokenizer.eos_token_id and ref_input_ids.item() == tokenizer.eos_token_id:
                break
    elapsed_ref = time.time() - start_ref
    actual_ref_tokens = len(ref_tokens)
    # Rate is over *timed* tokens only: the prefill token is excluded from both
    # the numerator and the denominator, on both paths.
    ref_timed_tokens = actual_ref_tokens - 1
    ref_text = tokenizer.decode(torch.cat(ref_tokens, dim=1)[0])
    print(f"  {ref_timed_tokens} tokens in {elapsed_ref:.2f}s "
          f"({ref_timed_tokens/elapsed_ref:.1f} tok/s), prefill excluded")
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
    # As on the FP path: the prefill-produced token is not in the timed region,
    # so it must not be counted in the rate either. Charging 200 tokens against
    # the time for 199 was overstating KIVI's throughput.
    kivi_timed_tokens = actual_kivi_tokens - 1
    generated = tokenizer.decode(torch.cat(all_tokens, dim=1)[0])

    print(f"\n  {kivi_timed_tokens} tokens in {elapsed:.2f}s "
          f"({kivi_timed_tokens/elapsed:.1f} tok/s), prefill excluded")
    print(f"  Output: {generated[:200]}{'...' if len(generated) > 200 else ''}")

    # -----------------------------------------------------------
    # 3. Cache Statistics
    # -----------------------------------------------------------
    s = cache.get_stats(layer=0)

    print(f"\n{'='*60}")
    print(f"Cache stats (layer 0) — all byte counts measured, not projected:")
    print(f"  Total tokens:        {s['total_tokens']}")
    print(f"  Quantized tokens:    {s['quantized_tokens']}")
    print(f"  Residual tokens:     {s['residual_tokens']}")
    print(f"  Packed (2-bit) KB:   {s['packed_bytes']/1024:.1f}")
    print(f"  Residual (FP) KB:    {s['residual_bytes']/1024:.1f}")
    print(f"  Serving copy KB:     {s['serving_copy_bytes']/1024:.1f} "
          f"(0 unless serving='cached')")
    print(f"  KIVI resident KB:    {s['kivi_bytes']/1024:.1f}")
    print(f"  FP baseline KB:      {s['fp_baseline_bytes']/1024:.1f}")
    print(f"  Compression ratio:   {s['compression']:.2f}x")
    print(f"\nPerformance (decode only — prefill excluded on both paths):")
    print(f"  FP32 speed:        {ref_timed_tokens/elapsed_ref:.1f} tok/s")
    print(f"  KIVI speed:        {kivi_timed_tokens/elapsed:.1f} tok/s")
    kivi_toks = kivi_timed_tokens / elapsed
    ref_toks = ref_timed_tokens / elapsed_ref
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
    if diverge_at < min_len:
        print("  NOTE: greedy decoding diverges irreversibly — once one token")
        print("  differs, every later token is compared against a different")
        print("  prefix, so the match rate mostly encodes *where* it diverged,")
        print("  not how much quality was lost. Read the generated text above,")
        print("  and use examples/eval_perplexity.py for a quantitative answer.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="KIVI 2-bit KV Cache Benchmark")
    parser.add_argument("model", nargs="?",
                        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        help="HuggingFace model name/path "
                             "(default: TinyLlama-1.1B; pass 'gpt2' for a fast "
                             "smoke test with no large download)")
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