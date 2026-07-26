"""
KIVI Long-Context Evaluation — resident memory + throughput at 2-4k context
============================================================================
Per docs/status_and_roadmap.md G7: gpt2 @ ~230 tokens is too small to
show KIVI's memory benefit (it only shows up at >=1B params, >=2k context).
This script measures resident cache bytes (via KiviCache.get_stats(), all
measured not projected) and tok/s, KIVI vs an FP baseline, on a small
long-context-capable model at a context length where the memory line
actually diverges.

Run: python examples/eval_longctx.py [model_name] [--ctx N] [--gen N] [--R N] [--G N]

Examples:
    python examples/eval_longctx.py                                   # TinyLlama-1.1B, 2k ctx
    python examples/eval_longctx.py Qwen/Qwen2-0.5B --ctx 4096
"""

import argparse
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kivi_sycl import KiviCache


def _make_prompt(tokenizer, num_tokens):
    """Build a prompt of roughly `num_tokens` tokens by repeating filler text."""
    filler = ("The history of artificial intelligence spans decades of research "
              "into machine learning, symbolic reasoning, and neural networks. ")
    text = filler
    while len(tokenizer(text).input_ids) < num_tokens:
        text += filler
    ids = tokenizer(text, return_tensors="pt").input_ids[:, :num_tokens]
    return tokenizer.decode(ids[0])


def run_fp_baseline(model, input_ids, num_gen_tokens):
    # Prefill is deliberately OUTSIDE the timed region, because run_kivi()
    # also excludes its prefill. Timing the FP prefill here and not KIVI's
    # made KIVI look ~2.5x *faster* at 2k context — that gap was the
    # 2049-token prompt forward pass, not anything about quantization.
    with torch.no_grad():
        out = model(input_ids, use_cache=True)
    past = out.past_key_values
    ids = out.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)

    start = time.time()
    with torch.no_grad():
        for _ in range(1, num_gen_tokens):
            out = model(ids, past_key_values=past, use_cache=True)
            past = out.past_key_values
            ids = out.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    elapsed = time.time() - start
    # FP baseline resident bytes: 2 (K+V) * layers * kv_heads * seq_len * head_dim * elt_size
    cfg = model.config
    num_layers = getattr(cfg, "num_hidden_layers", getattr(cfg, "n_layer", None))
    num_kv_heads = getattr(cfg, "num_key_value_heads",
                            getattr(cfg, "num_attention_heads",
                                    getattr(cfg, "n_head", None)))
    head_dim = getattr(cfg, "head_dim", None)
    if head_dim is None:
        hidden = getattr(cfg, "hidden_size", getattr(cfg, "n_embd", None))
        n_head = getattr(cfg, "num_attention_heads", getattr(cfg, "n_head", None))
        head_dim = hidden // n_head
    seq_len = input_ids.shape[1] + num_gen_tokens
    elt = torch.tensor([], dtype=next(model.parameters()).dtype).element_size()
    fp_bytes = 2 * num_layers * num_kv_heads * seq_len * head_dim * elt
    return elapsed, fp_bytes, num_layers


def run_kivi(model, input_ids, num_gen_tokens, G, R, serving="rebuild",
             key_flush="drain", prefill_chunk=None):
    cache = KiviCache.from_model(model, device="xpu", group_size=G,
                                 residual_length=R, serving=serving,
                                 key_flush=key_flush,
                                 prefill_chunk=prefill_chunk)
    with torch.no_grad():
        out = model(input_ids, use_cache=True)
    cache.add_tokens(out.past_key_values)
    ids = out.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)

    start = time.time()
    with torch.no_grad():
        for _ in range(1, num_gen_tokens):
            past = cache.get_full_cache()
            out = model(ids, past_key_values=past, use_cache=True)
            cache.add_tokens(out.past_key_values)
            ids = out.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    cache.harvest_pending()
    elapsed = time.time() - start
    return elapsed, cache.get_stats(layer=0)


def main():
    parser = argparse.ArgumentParser(description="KIVI long-context memory/throughput eval")
    parser.add_argument("model", nargs="?", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        help="HuggingFace model (default: TinyLlama-1.1B)")
    parser.add_argument("--ctx", type=int, default=2048, help="Prompt context length")
    parser.add_argument("--gen", type=int, default=64, help="Tokens to generate after prefill")
    parser.add_argument("--G", type=int, default=32, help="Quantization group size")
    parser.add_argument("--R", type=int, default=128, help="Residual window size")
    parser.add_argument("--key-flush", choices=["drain", "reset"], default="drain",
                        dest="key_flush",
                        help="drain: quantize a group at a time, key window "
                             "averages ~R (this repo's original policy). "
                             "reset: quantize the whole key buffer at R and "
                             "leave it empty, averaging R/2 (paper-faithful, "
                             "compresses better). Requires R %% G == 0.")
    parser.add_argument("--prefill-chunk", type=int, default=None,
                        dest="prefill_chunk",
                        help="Ingest a long prompt in slices of this many "
                             "tokens, harvesting between them. Bounds peak "
                             "memory; packed output is unchanged. Must be a "
                             "multiple of G and >= R. Default: one submission.")
    parser.add_argument("--serving", choices=["rebuild", "cached"], default="rebuild",
                        help="rebuild: dequantize history on every read (keeps the "
                             "memory win). cached: retain a persistent FP history copy "
                             "(faster reads, trades the memory back).")
    args = parser.parse_args()

    if not torch.xpu.is_available():
        print("Error: XPU not found.")
        sys.exit(1)

    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    prompt = _make_prompt(tokenizer, args.ctx)
    # Re-truncate: _make_prompt slices to args.ctx ids, but decode->re-encode
    # is not round-trip stable and can come back one or two tokens longer,
    # which then overshoots the model's position limit.
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids[:, :args.ctx]
    max_pos = getattr(model.config, "max_position_embeddings", None)
    if max_pos is not None and input_ids.shape[1] + args.gen > max_pos:
        print(f"Error: ctx {input_ids.shape[1]} + gen {args.gen} exceeds this "
              f"model's max_position_embeddings ({max_pos}). Lower --ctx or --gen.")
        sys.exit(1)
    print(f"Prompt tokens: {input_ids.shape[1]}, generating {args.gen} more, "
          f"G={args.G}, R={args.R}")

    print("\n--- FP baseline ---")
    fp_elapsed, fp_bytes, num_layers = run_fp_baseline(model, input_ids.clone(), args.gen)
    fp_toks = (args.gen - 1) / fp_elapsed
    print(f"  {args.gen - 1} tokens in {fp_elapsed:.2f}s ({fp_toks:.1f} tok/s), prefill excluded")
    print(f"  Resident KV, all {num_layers} layers (analytical): {fp_bytes/1024/1024:.1f} MB")

    print(f"\n--- KIVI 2-bit (serving={args.serving}, key_flush={args.key_flush}) ---")
    kivi_elapsed, stats = run_kivi(model, input_ids.clone(), args.gen, args.G,
                                   args.R, args.serving, args.key_flush,
                                   args.prefill_chunk)
    kivi_toks = (args.gen - 1) / kivi_elapsed
    print(f"  {args.gen - 1} tokens in {kivi_elapsed:.2f}s ({kivi_toks:.1f} tok/s)")
    print(f"  Layer 0, measured: packed {stats['packed_bytes']/1024:.1f} KB + "
          f"residual {stats['residual_bytes']/1024:.1f} KB "
          f"= {stats['kivi_bytes']/1024:.1f} KB")
    print(f"  Layer 0, FP baseline (measured): {stats['fp_baseline_bytes']/1024:.1f} KB")
    if stats["serving_copy_bytes"]:
        print(f"  Layer 0, persistent FP serving copy: "
              f"{stats['serving_copy_bytes']/1024:.1f} KB (counted against compression)")
    print(f"  Layer 0, mean FP window: keys {stats['mean_residual_tokens']:.1f} "
          f"tok, values {stats['mean_residual_tokens_v']:.1f} tok "
          f"(rho_mean {stats['rho_mean']:.4f})")

    # Every layer holds identically shaped tensors, so scaling layer 0 by the
    # layer count is exact — and it is the only way to compare against the
    # all-layers FP figure above without mixing scopes.
    kivi_total = (stats["kivi_bytes"] + stats["serving_copy_bytes"]) * num_layers
    fp_total = stats["fp_baseline_bytes"] * num_layers
    peak_total = stats["peak_bytes"] * num_layers

    print(f"\n{'='*64}")
    print(f"Summary @ ctx={args.ctx}, gen={args.gen}, G={args.G}, R={args.R}, "
          f"serving={args.serving}, key_flush={args.key_flush}, "
          f"prefill_chunk={args.prefill_chunk}")
    print(f"  Throughput:  FP {fp_toks:.1f} tok/s  ->  KIVI {kivi_toks:.1f} tok/s "
          f"({(kivi_toks/fp_toks - 1) * 100:+.0f}%)")
    print(f"  Resident KV, all {num_layers} layers (measured):")
    print(f"    FP baseline: {fp_total/1024/1024:8.1f} MB")
    print(f"    KIVI:        {kivi_total/1024/1024:8.1f} MB")
    print(f"    Saved:       {(fp_total - kivi_total)/1024/1024:8.1f} MB "
          f"({stats['compression']:.2f}x compression, final step)")
    # Compare configurations on this line, not the one above: a single
    # instantaneous reading depends on where in the flush cycle the run
    # stopped, and under key_flush="reset" the key window sweeps 0..R.
    print(f"    Time-averaged over {stats['compression_samples']} steps: "
          f"{stats['mean_compression']:.2f}x")
    # KIVI's headline memory claim is PEAK, not resident. Reporting only the
    # resident line against it would be comparing two different metrics —
    # the error class recorded in docs/status_and_roadmap.md §2.1.
    print(f"  Peak KV, all {num_layers} layers (measured high-water mark,")
    print(f"  incl. in-flight flush inputs + the FP view served to the model):")
    print(f"    KIVI peak:   {peak_total/1024/1024:8.1f} MB "
          f"({stats['peak_compression']:.2f}x vs FP baseline)")
    print(f"    reached at {stats['peak_at_tokens']} of "
          f"{stats['total_tokens']} tokens")
    if stats["peak_compression"] < 1.0:
        print(f"    NOTE: peak exceeds the FP baseline. The resident win is "
              f"real, but a peak-constrained\n          deployment sees no "
              f"saving here — report both numbers, not just the first.")
    print(f"{'='*64}")


if __name__ == "__main__":
    main()
