"""
KIVI cache-layer overhead per decode step, without a model forward pass.

Synthetic KV in Qwen2.5-0.5B's cache shape (24 layers x 2 KV heads x 64), so
the numbers are the cache's own cost — add_tokens (flush submit/harvest) and
get_full_cache (dequant-on-read) — free of the model forward's run-to-run
noise, which swamps millisecond effects in end-to-end tok/s.

Run:
    python examples/bench_cache_layer.py                 # staged vs zero-copy
    python examples/bench_cache_layer.py --baseline REV  # also a past KiviCache from git
"""
import argparse
import os
import statistics
import subprocess
import sys
import time
import types

import torch

from kivi_sycl.cache.manager import KiviCache  # noqa: E402

LAYERS, H, D, G, R = 24, 2, 64, 32, 128


def load_baseline(commit):
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = subprocess.check_output(
        ["git", "-C", repo, "show",
         f"{commit}:src/kivi_sycl/cache/manager.py"], text=True)
    src = src.replace("from ..config.model_config import", "from kivi_sycl.config.model_config import")
    src = src.replace("from ..extension.loader import", "from kivi_sycl.extension.loader import")
    mod = types.ModuleType("baseline_manager")
    if "from .arena import" in src:
        # Revisions with the Python engine (before Phase 5 S1.8) import its
        # arena module; load that file from the same revision.
        arena_src = subprocess.check_output(
            ["git", "-C", repo, "show", f"{commit}:src/kivi_sycl/cache/arena.py"], text=True)
        arena = types.ModuleType("baseline_arena")
        exec(compile(arena_src, "baseline_arena.py", "exec"), arena.__dict__)
        mod.__dict__.update({k: v for k, v in arena.__dict__.items() if not k.startswith("__")})
        src = src.replace("from .arena import LayerArena, ensure_capacity\n", "")
    exec(compile(src, "baseline_manager.py", "exec"), mod.__dict__)
    return mod.KiviCache


def run(make_cache, ctx, decode_steps, reserve):
    torch.manual_seed(0)
    cache = make_cache()
    if reserve and hasattr(cache, "reserve"):
        cache.reserve(ctx + decode_steps)
    kv = [(torch.randn(1, H, ctx, D), torch.randn(1, H, ctx, D)) for _ in range(LAYERS)]
    cache.add_tokens(tuple(kv))
    add_t, serve_t, flush_steps = [], [], []
    for step in range(decode_steps):
        t0 = time.perf_counter()
        past = cache.get_full_cache()
        t1 = time.perf_counter()
        kv = [(torch.cat([k, torch.randn(1, H, 1, D)], 2), torch.cat([v, torch.randn(1, H, 1, D)], 2))
              for k, v in kv]
        flushes_before = cache.total_flushes
        t2 = time.perf_counter()
        cache.add_tokens(tuple(kv))
        t3 = time.perf_counter()
        serve_t.append(t1 - t0)
        add_t.append(t3 - t2)
        if cache.total_flushes != flushes_before:
            flush_steps.append(step)
        del past
    cache.harvest_pending()
    s = cache.get_stats(0)
    ms = lambda xs: 1e3 * statistics.mean(xs)
    # Steps that harvested a flush (the step after a submit) carry the copy cost.
    harvest = [add_t[i + 1] for i in flush_steps if i + 1 < len(add_t)]
    return {"add_ms": ms(add_t), "harvest_ms": ms(harvest) if harvest else float("nan"),
            "serve_ms": ms(serve_t), "compression": s["compression"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--baseline", metavar="REV",
                    help="git revision whose src/kivi_sycl/cache/manager.py to compare against")
    ap.add_argument("--ctx", type=int, nargs="+", default=[2048, 8192])
    ap.add_argument("--steps", type=int, default=160, help="decode steps per run")
    args = ap.parse_args()
    variants = {}
    if args.baseline:
        baseline = load_baseline(args.baseline)
        variants[f"{args.baseline} manager"] = lambda: baseline(
            LAYERS, D, H, device="xpu", group_size=G, residual_length=R)
    variants.update({
        "arena staged": lambda: KiviCache(LAYERS, D, H, device="xpu", group_size=G, residual_length=R, zero_copy=False),
        "arena zero-copy": lambda: KiviCache(LAYERS, D, H, device="xpu", group_size=G, residual_length=R, zero_copy=True),
    })
    if not torch.xpu.is_available():
        sys.exit("Error: XPU not found.")
    print(f"{'variant':20s} {'ctx':>5s} {'add_tokens':>11s} {'harvest step':>13s} {'get_full_cache':>15s} {'compr':>6s}")
    for ctx in args.ctx:
        for name, mk in variants.items():
            run(mk, 256, 40, reserve=True)                       # warm-up (JIT, allocator)
            r = run(mk, ctx, args.steps, reserve=True)
            print(f"{name:20s} {ctx:5d} {r['add_ms']:9.2f}ms {r['harvest_ms']:11.2f}ms "
                  f"{r['serve_ms']:13.2f}ms {r['compression']:6.2f}x", flush=True)


if __name__ == "__main__":
    main()
