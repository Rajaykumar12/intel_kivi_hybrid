"""
KIVI Perplexity Evaluation — quality check, KIVI 2-bit vs FP16/FP32 baseline
=============================================================================
Per docs/status_and_roadmap.md G7: cheap, quantitative check that
2-bit quantization hasn't silently degraded model quality — catches
regressions long before a full task-suite eval would.

Computes perplexity over consecutive windows of a WikiText-2 (or C4) slice
using each window's own KV cache (KIVI vs a plain FP forward pass), the
standard sliding-window-free per-segment method used for quick sanity
checks (not the official strided-stride LM-eval protocol).

Run: python examples/eval_perplexity.py [model_name] [--dataset wikitext2|c4]
                                         [--windows N] [--window-len N]

Example:
    python examples/eval_perplexity.py gpt2 --windows 20 --window-len 512
"""

import argparse
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from kivi_sycl import KiviCache


def _load_text(dataset):
    from datasets import load_dataset
    if dataset == "wikitext2":
        # Must be the namespaced id: `datasets` >= 4 / current huggingface_hub
        # reject the legacy bare "wikitext" with
        # `HfUriError: Repository id must be 'namespace/name'`.
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        return "\n\n".join(t for t in ds["text"] if t.strip())
    elif dataset == "c4":
        ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
        texts = []
        for i, ex in enumerate(ds):
            texts.append(ex["text"])
            if i >= 200:
                break
        return "\n\n".join(texts)
    raise ValueError(f"Unknown dataset {dataset!r}")


def ppl_fp(model, ids):
    """Plain FP forward pass perplexity over one window (no external cache)."""
    with torch.no_grad():
        out = model(ids, use_cache=False)
    logits = out.logits[:, :-1, :]
    target = ids[:, 1:]
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
    return loss.item()


def ppl_kivi(model, ids, G, R):
    """Token-by-token forward through KiviCache, accumulating NLL."""
    cache = KiviCache.from_model(model, device="xpu", group_size=G, residual_length=R)
    total_nll = 0.0
    count = 0
    with torch.no_grad():
        out = model(ids[:, :1], use_cache=True)
        cache.add_tokens(out.past_key_values)
        for t in range(1, ids.shape[1]):
            past = cache.get_full_cache()
            logits = out.logits[:, -1, :]
            target = ids[:, t]
            total_nll += F.cross_entropy(logits, target, reduction="sum").item()
            count += 1
            out = model(ids[:, t:t + 1], past_key_values=past, use_cache=True)
            cache.add_tokens(out.past_key_values)
    cache.harvest_pending()
    return total_nll / count


def main():
    parser = argparse.ArgumentParser(description="KIVI perplexity eval vs FP baseline")
    parser.add_argument("model", nargs="?", default="gpt2", help="HuggingFace model")
    parser.add_argument("--dataset", choices=["wikitext2", "c4"], default="wikitext2")
    parser.add_argument("--windows", type=int, default=10, help="Number of eval windows")
    parser.add_argument("--window-len", type=int, default=256, help="Tokens per window")
    parser.add_argument("--G", type=int, default=32, help="Quantization group size")
    parser.add_argument("--R", type=int, default=64, help="Residual window size")
    args = parser.parse_args()

    if not torch.xpu.is_available():
        print("Error: XPU not found.")
        sys.exit(1)

    try:
        text = _load_text(args.dataset)
    except ImportError:
        print("Error: `datasets` package required (pip install datasets).")
        sys.exit(1)

    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model.eval()

    all_ids = tokenizer(text, return_tensors="pt").input_ids
    win = args.window_len
    n_avail = all_ids.shape[1] // win
    n_windows = min(args.windows, n_avail)
    print(f"Dataset: {args.dataset}, {n_windows} windows of {win} tokens "
          f"(G={args.G}, R={args.R})")

    fp_losses, kivi_losses = [], []
    for i in range(n_windows):
        window = all_ids[:, i * win:(i + 1) * win]
        fp_loss = ppl_fp(model, window)
        kivi_loss = ppl_kivi(model, window, args.G, args.R)
        fp_losses.append(fp_loss)
        kivi_losses.append(kivi_loss)
        print(f"  window {i+1}/{n_windows}: FP nll={fp_loss:.4f}  "
              f"KIVI nll={kivi_loss:.4f}  delta={kivi_loss - fp_loss:+.4f}")

    fp_ppl = torch.tensor(fp_losses).mean().exp().item()
    kivi_ppl = torch.tensor(kivi_losses).mean().exp().item()

    print(f"\n{'='*60}")
    print(f"FP perplexity:   {fp_ppl:.3f}")
    print(f"KIVI perplexity: {kivi_ppl:.3f}")
    print(f"Relative delta:  {(kivi_ppl / fp_ppl - 1) * 100:+.2f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
