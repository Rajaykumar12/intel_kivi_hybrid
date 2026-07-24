"""End-to-end KIVI pipeline test on CPU — no GPU, XPU, oneAPI, or compiled
extension required.

Replaces the native SYCL kernels with an exact torch reference of the
asymmetric 2-bit round-trip and runs the real generate()/KiviCache
pipeline against a real HuggingFace model, comparing with vanilla greedy
generation.

Two modes:
  default     — real 2-bit rounding. Verifies the whole pipeline and lets
                you judge output quality. Some token divergence from the
                FP32 baseline is EXPECTED (2-bit is lossy); the output
                must stay coherent, not turn to gibberish.
  --identity  — the "quantization" is a lossless copy, so the pipeline
                must reproduce the FP32 baseline token-for-token. This is
                the strongest CPU check of the cache bookkeeping: any
                dropped, reordered, or stale token breaks the 100% match.

Usage:
  python examples/cpu_sim.py                    # gpt2, 2-bit, 120 tokens
  python examples/cpu_sim.py --identity         # must print 100% match
  python examples/cpu_sim.py distilgpt2 --tokens 80 --G 16 --R 32
"""

import argparse
import os
import sys
import types

import torch

REPO_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
if os.path.isdir(REPO_SRC) and REPO_SRC not in sys.path:
    sys.path.insert(0, REPO_SRC)


def _roundtrip_2bit(inp, out, group_size):
    flat = inp.reshape(-1, group_size)
    mn = flat.min(dim=1, keepdim=True).values
    mx = flat.max(dim=1, keepdim=True).values
    scale = (mx - mn) / 3.0 + 1e-10
    q = torch.clamp(torch.round((flat - mn) / scale), 0.0, 3.0)
    out.copy_((q * scale + mn).reshape(inp.shape))


def install_fake_native(identity):
    fake = types.ModuleType("kivi_sycl._C")

    def keys(inp, out, group_size):
        out.copy_(inp) if identity else _roundtrip_2bit(inp, out, group_size)

    def values(inp, out, head_dim, group_size):
        out.copy_(inp) if identity else _roundtrip_2bit(inp, out, group_size)

    fake.quant_dequant_roundtrip_keys = keys
    fake.quant_dequant_roundtrip_values = values
    sys.modules["kivi_sycl._C"] = fake

    import kivi_sycl.cache.manager as manager_module
    manager_module.kivi_native = fake
    manager_module.ensure_ipex = lambda: None  # no IPEX needed on CPU


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", default="gpt2")
    parser.add_argument("--tokens", type=int, default=120)
    parser.add_argument("--G", type=int, default=16)
    parser.add_argument("--R", type=int, default=32,
                        help="small residual on purpose: forces flushes early")
    parser.add_argument("--identity", action="store_true",
                        help="lossless flush — output must match FP32 exactly")
    parser.add_argument("--prompt", default="In a shocking finding, scientists "
                        "discovered a herd of unicorns living in a remote "
                        "valley. The unicorns were")
    args = parser.parse_args()

    install_fake_native(args.identity)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from kivi_sycl import generate

    print(f"Loading {args.model} (CPU)...")
    model = AutoModelForCausalLM.from_pretrained(args.model,
                                                 torch_dtype=torch.float32)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids

    print("\n--- FP32 baseline (model.generate, greedy) ---")
    with torch.no_grad():
        ref_ids = model.generate(input_ids, max_new_tokens=args.tokens,
                                 do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)[0]
    ref_text = tokenizer.decode(ref_ids, skip_special_tokens=True)
    print(ref_text)

    mode = "identity (lossless)" if args.identity else "2-bit reference"
    print(f"\n--- KIVI pipeline on CPU [{mode}, G={args.G}, R={args.R}] ---")
    kivi_text = generate(model, tokenizer, args.prompt,
                         max_new_tokens=args.tokens,
                         group_size=args.G, residual_length=args.R,
                         device="cpu", do_sample=False)
    print(kivi_text)

    kivi_ids = tokenizer(kivi_text, return_tensors="pt").input_ids[0]
    n = min(len(ref_ids), len(kivi_ids))
    match = (ref_ids[:n] == kivi_ids[:n]).float().mean().item()
    diverge = next((t for t in range(n) if ref_ids[t] != kivi_ids[t]), n)
    print(f"\nToken match vs FP32: {match * 100:.1f}%  "
          f"(first divergence at token {diverge}/{n})")

    # Compare decoded text, not re-tokenized ids: BPE re-tokenization of
    # decoded text can shift token boundaries and fake a mismatch.
    if args.identity:
        if kivi_text == ref_text:
            print("PASS: cache bookkeeping is lossless end-to-end.")
        else:
            print("FAIL: identity mode must reproduce the FP32 baseline "
                  "exactly — the cache manager dropped, reordered, or "
                  "staled tokens.")
            sys.exit(1)


if __name__ == "__main__":
    main()
