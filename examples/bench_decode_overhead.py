"""
Decode-step overhead probes (roadmap Phase 4 gate).

Answers "how much of a decode step is outside native code, and what would
removing it buy?" — the question that decides whether a C++/LibTorch runtime
is worth building.

    python examples/bench_decode_overhead.py profile  MODEL CTX [--fp]
        torch.profiler: wall time vs time inside top-level ATen ops and KIVI's
        pybind calls. The gap (Python + dispatch + profiler bookkeeping) is an
        UPPER bound on Python overhead.
    python examples/bench_decode_overhead.py compile  MODEL CTX
        FP model with StaticCache, eager vs torch.compile. Compile removes
        Python *and* dispatch and fuses ops, so its speedup bounds what a
        C++ port (which removes only Python) could gain.
    python examples/bench_decode_overhead.py compile-kivi MODEL CTX
        torch.compile over the KIVI HF path, with the cache update and the
        "kivi" attention excluded from tracing (clean graph breaks).

Median and minimum per step are reported because desktop background load
(browsers, compositors) adds noise; compare medians across runs on an idle
machine before drawing conclusions.
"""

import argparse
import statistics
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache

from kivi_sycl import enable_kivi

STEPS, WARMUP = 24, 6


def _prompt(tok, ctx):
    text = "The history of artificial intelligence spans decades. " * (ctx // 8 + 64)
    return tok(text, return_tensors="pt").input_ids[:, :ctx]


def _decode(model, ids, cache, fwd, static=False):
    """Prefill eagerly, then time STEPS greedy decode steps through `fwd`."""
    ctx = ids.shape[1]
    kw = {"cache_position": torch.arange(ctx)} if static else {}
    with torch.no_grad():
        out = model(ids, past_key_values=cache, use_cache=True, **kw)
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        times, toks, pos = [], [], ctx
        for i in range(STEPS + WARMUP):
            kw = {"cache_position": torch.tensor([pos])} if static else {}
            t0 = time.perf_counter()
            o = fwd(input_ids=nxt, past_key_values=cache, use_cache=True, **kw)
            nxt = o.logits[:, -1].argmax(-1, keepdim=True)
            dt = time.perf_counter() - t0
            pos += 1
            toks.append(nxt.item())
            if i >= WARMUP:
                times.append(1e3 * dt)
    return statistics.median(times), min(times), toks


def cmd_profile(model, ids, fp):
    from torch.profiler import ProfilerActivity, profile, record_function
    from kivi_sycl.extension.loader import kivi_native

    for name in ("fused_decode_attention", "quantize_keys_async",
                 "quantize_values_async", "alloc_usm_host"):
        f = getattr(kivi_native, name)

        def wrapped(*a, _f=f, _n=name, **k):
            with record_function(f"kivi::{_n}"):
                return _f(*a, **k)
        setattr(kivi_native, name, wrapped)

    cache = None if fp else enable_kivi(model, max_tokens=ids.shape[1] + 3 * STEPS)
    with torch.no_grad():
        out = model(ids, past_key_values=cache, use_cache=True)
        state = {"past": out.past_key_values, "nxt": out.logits[:, -1].argmax(-1, keepdim=True)}

        def step():
            o = model(state["nxt"], past_key_values=state["past"], use_cache=True)
            state["past"] = o.past_key_values
            state["nxt"] = o.logits[:, -1].argmax(-1, keepdim=True)

        for _ in range(WARMUP):
            step()
        t0 = time.perf_counter()
        for _ in range(STEPS):
            step()
        plain = 1e3 * (time.perf_counter() - t0) / STEPS
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            t0 = time.perf_counter()
            for _ in range(STEPS):
                step()
            wall = 1e3 * (time.perf_counter() - t0) / STEPS
    native = sum(e.cpu_time_total for e in prof.events() if e.cpu_parent is None) / 1e3 / STEPS
    print(f"{'FP' if fp else 'KIVI'}: step {plain:.1f} ms unprofiled | profiled {wall:.1f} ms, "
          f"native {native:.1f} ms, outside native {wall - native:.1f} ms "
          f"= {100 * (wall - native) / wall:.1f}% (upper bound)")


def cmd_compile(model, ids):
    def static_cache():
        return StaticCache(config=model.config, max_cache_len=ids.shape[1] + 2 * STEPS + 16)
    em, en, et = _decode(model, ids, static_cache(), model.forward, static=True)
    cm, cn, ct = _decode(model, ids, static_cache(),
                         torch.compile(model.forward, dynamic=False), static=True)
    print(f"FP StaticCache: eager {em:.1f} ms (min {en:.1f}) | compiled {cm:.1f} ms "
          f"(min {cn:.1f}) | speedup {em / cm:.2f}x median, {en / cn:.2f}x min | "
          f"same tokens: {et == ct}")


def cmd_compile_kivi(model, ids):
    import kivi_sycl.hf.attention as ha
    import kivi_sycl.hf.cache as hc
    hc.KiviLayer.update = torch.compiler.disable(hc.KiviLayer.update)
    ha.kivi_attention_forward = torch.compiler.disable(ha.kivi_attention_forward)

    def kivi_cache():
        c = enable_kivi(model, max_tokens=ids.shape[1] + 2 * STEPS + 16)
        ha.ALL_ATTENTION_FUNCTIONS.register(ha.ATTN_NAME, ha.kivi_attention_forward)
        return c
    em, en, et = _decode(model, ids, kivi_cache(), model.forward)
    cm, cn, ct = _decode(model, ids, kivi_cache(), torch.compile(model.forward, dynamic=True))
    print(f"KIVI (graph breaks at update/attention): eager {em:.1f} ms (min {en:.1f}) | "
          f"compiled {cm:.1f} ms (min {cn:.1f}) | speedup {em / cm:.2f}x median, "
          f"{en / cn:.2f}x min | same tokens: {et == ct}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("mode", choices=["profile", "compile", "compile-kivi"])
    ap.add_argument("model")
    ap.add_argument("ctx", type=int)
    ap.add_argument("--fp", action="store_true", help="profile: FP baseline instead of KIVI")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()
    ids = _prompt(tok, args.ctx)
    print(f"{args.model} ctx={ids.shape[1]} mode={args.mode}")
    if args.mode == "profile":
        cmd_profile(model, ids, args.fp)
    elif args.mode == "compile":
        cmd_compile(model, ids)
    else:
        cmd_compile_kivi(model, ids)


if __name__ == "__main__":
    main()
