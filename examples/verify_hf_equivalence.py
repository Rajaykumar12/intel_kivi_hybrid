"""
Model-level equivalence check: legacy loop vs native transformers path.

For each model, greedy-decodes N tokens two ways and requires bit-identical
logits at every step:
  legacy   KiviCache.add_tokens / get_full_cache around a plain model call
  hf       enable_kivi(model, fused=False) — the "kivi" attention's
           dequantize-then-SDPA reference path
Both runs use the C++ cache engine, so this also validates it on real
models. The fused kernel is a different
summation order and is checked separately (tests, eval_perplexity --path hf).

Run:
    python examples/verify_hf_equivalence.py
    python examples/verify_hf_equivalence.py gpt2 Qwen/Qwen2-0.5B --steps 200
"""

import argparse
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kivi_sycl import KiviCache, enable_kivi

DEFAULT_MODELS = ["gpt2", "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "Qwen/Qwen2-0.5B",
                  "Qwen/Qwen2.5-0.5B-Instruct", "facebook/opt-125m"]
PROMPT = "The history of artificial intelligence spans decades of research into machine learning and"


def check(name, steps, G, R):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
    model.set_attn_implementation("sdpa")
    ids = tok(PROMPT, return_tensors="pt").input_ids

    with torch.no_grad():
        legacy = KiviCache.from_model(model, group_size=G, residual_length=R)
        out = model(ids, use_cache=True)
        legacy.add_tokens(out.past_key_values)
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        legacy_logits = []
        for _ in range(steps):
            out = model(nxt, past_key_values=legacy.get_full_cache(), use_cache=True)
            legacy_logits.append(out.logits[:, -1])
            legacy.add_tokens(out.past_key_values)
            nxt = out.logits[:, -1].argmax(-1, keepdim=True)

        cache = enable_kivi(model, group_size=G, residual_length=R, fused=False)
        out = model(ids, past_key_values=cache, use_cache=True)
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        first_diff, max_diff = None, 0.0
        for step in range(steps):
            out = model(nxt, past_key_values=cache, use_cache=True)
            got = out.logits[:, -1]
            d = (got - legacy_logits[step]).abs().max().item()
            max_diff = max(max_diff, d)
            if d != 0 and first_diff is None:
                first_diff = step
            nxt = got.argmax(-1, keepdim=True)
    legacy.harvest_pending()
    cache.harvest_pending()
    quant = cache.kivi.quantized_tokens(0)
    ok = first_diff is None and quant == legacy.quantized_tokens(0) and quant > 0
    print(f"{'OK  ' if ok else 'FAIL'} {name:36s} steps={steps} "
          f"quantized={quant} max|logit diff|={max_diff:.3g}"
          + ("" if first_diff is None else f" first diff at step {first_diff}"))
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--G", type=int, default=32)
    ap.add_argument("--R", type=int, default=64)
    args = ap.parse_args()
    results = [check(m, args.steps, args.G, args.R) for m in args.models]
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
