"""End-to-end KIVI pipeline on the CPU — no GPU or XPU required.

Runs the real generate()/KiviCache pipeline with device="cpu": the C++
engine's reference quantize (the same math as the SYCL kernels) instead of
the GPU kernels, against a real HuggingFace model, and compares with vanilla
greedy generation. Some token divergence from the FP32 baseline is EXPECTED
(2-bit is lossy); the output must stay coherent, not turn to gibberish.

Needs the compiled extension (kivi_sycl._C) but no GPU. The cache
bookkeeping itself is checked bit-for-bit by the test suite and by
examples/verify_hf_equivalence.py.

Usage:
  python examples/cpu_sim.py                    # gpt2, 2-bit, 120 tokens
  python examples/cpu_sim.py distilgpt2 --tokens 80 --G 16 --R 32
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kivi_sycl import generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", default="gpt2")
    parser.add_argument("--tokens", type=int, default=120)
    parser.add_argument("--G", type=int, default=16)
    parser.add_argument("--R", type=int, default=32,
                        help="small residual on purpose: forces flushes early")
    parser.add_argument("--prompt", default="In a shocking finding, scientists "
                        "discovered a herd of unicorns living in a remote "
                        "valley. The unicorns were")
    args = parser.parse_args()

    print(f"Loading {args.model} (CPU)...")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids

    print("\n--- FP32 baseline (model.generate, greedy) ---")
    with torch.no_grad():
        ref_ids = model.generate(input_ids, max_new_tokens=args.tokens, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)[0]
    print(tokenizer.decode(ref_ids, skip_special_tokens=True))

    print(f"\n--- KIVI pipeline on CPU [2-bit reference, G={args.G}, R={args.R}] ---")
    kivi_text = generate(model, tokenizer, args.prompt, max_new_tokens=args.tokens,
                         group_size=args.G, residual_length=args.R,
                         device="cpu", do_sample=False)
    print(kivi_text)

    kivi_ids = tokenizer(kivi_text, return_tensors="pt").input_ids[0]
    n = min(len(ref_ids), len(kivi_ids))
    match = (ref_ids[:n] == kivi_ids[:n]).float().mean().item()
    diverge = next((t for t in range(n) if ref_ids[t] != kivi_ids[t]), n)
    print(f"\nToken match vs FP32: {match * 100:.1f}%  (first divergence at token {diverge}/{n})")


if __name__ == "__main__":
    main()
