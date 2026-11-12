"""Phase 5 A.10 benchmark: kivi-run vs the Python paths (§4.6).

    python tools/bench_runtime.py --runs 3 --out build-native/bench/results.json

Configurations (model, prompt tokens, generated tokens, artifact dir):
    TinyLlama-1.1B 1.9k + 64, Qwen2.5-0.5B 2k + 64, Qwen2.5-0.5B 8k + 32.
Paths, each run in a FRESH process (so peak RSS and time-to-first-token
include imports and model loading):
    python-fp     HF eager, plain DynamicCache (fp32 KV)
    python-kivi   HF eager + enable_kivi (native Cache, fused 2-bit decode)
    kivi-run      the C++ runtime on the exported artifact
All greedy, no EOS stop, same prompt ids (examples/eval_longctx.py's filler
text), same KIVI settings as the artifact (G, R, fp16 metadata), default
thread count, and logits for the last position only in every path
(transformers' logits_to_keep=1, what generate() uses; the artifacts are
exported that way by default). Metrics:
    prefill tok/s   prompt tokens / time of the prefill forward
    decode tok/s    (gen - 1) / time after the first token (prefill excluded,
                    like eval_longctx.py)
    TTFT            process spawn -> first token sampled, CLOCK_MONOTONIC on
                    both sides (Python time.monotonic, C++ steady_clock)
    peak RSS        the child's ru_maxrss (wait4)
Reported: median and min over --runs, plus whether every path produced the
same greedy tokens as python-fp (KIVI is lossy, so a later divergence is
expected; the first tokens should agree).
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS = [
    ("tinyllama-1.9k", "TinyLlama/TinyLlama-1.1B-Chat-v1.0", 1900, 64, "artifacts/tinyllama"),
    # 2000 + 64 exceeds the 2048-token artifacts/qwen25; the 8k export has the
    # same weights and generated code, only a larger dynamic-shape bound.
    ("qwen2.5-2k", "Qwen/Qwen2.5-0.5B-Instruct", 2000, 64, "artifacts/qwen25_8k"),
    ("qwen2.5-8k", "Qwen/Qwen2.5-0.5B-Instruct", 8192, 32, "artifacts/qwen25_8k"),
]
PATHS = ["python-fp", "python-kivi", "kivi-run"]


# --------------------------------------------------------------------------- worker
def worker(a):
    """Child process for the Python paths. Prints one JSON line."""
    import torch
    from transformers import AutoModelForCausalLM
    with open(a.ids) as fh:
        ids = torch.tensor([json.load(fh)], dtype=torch.long)
    with open(os.path.join(a.artifacts, "kivi_config.json")) as fh:
        cfg = json.load(fh)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()
    kw = {}
    if a.path == "python-kivi":
        from kivi_sycl import enable_kivi
        kw["past_key_values"] = enable_kivi(
            model, group_size=cfg["group_size"], residual_length=cfg["residual_length"],
            fused=True, metadata_dtype="auto", max_tokens=ids.shape[1] + a.gen)
    toks = []
    with torch.no_grad():
        t0 = time.monotonic()
        # logits_to_keep=1: last position only, as generate() does and as
        # kivi-run's export does (logits for all 8k positions would be 5 GB).
        out = model(ids, use_cache=True, logits_to_keep=1, **kw)
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        t1 = time.monotonic()
        toks.append(int(nxt))
        past = out.past_key_values
        for _ in range(a.gen - 1):
            out = model(nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[:, -1].argmax(-1, keepdim=True)
            toks.append(int(nxt))
        t2 = time.monotonic()
    if a.path == "python-kivi":
        kw["past_key_values"].harvest_pending()
    print(json.dumps({"prefill_s": t1 - t0, "decode_s": t2 - t1, "first_token_mono": t1,
                      "tokens": toks}))


# --------------------------------------------------------------------------- driver
def make_ids(model_id, n, path):
    """Prompt ids exactly as examples/eval_longctx.py builds them."""
    sys.path.insert(0, os.path.join(ROOT, "examples"))
    from eval_longctx import _make_prompt
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    ids = tok(_make_prompt(tok, n), return_tensors="pt").input_ids[0, :n].tolist()
    with open(path, "w") as fh:
        json.dump(ids, fh)
    return len(ids)


def run_path(path, model_id, gen, art, ids_file, env):
    py = sys.executable
    if path == "kivi-run":
        with open(ids_file) as fh:
            ids_csv = ",".join(map(str, json.load(fh)))
        ids_txt = ids_file + ".txt"
        with open(ids_txt, "w") as fh:
            fh.write(ids_csv)
        cmd = [os.path.join(ROOT, "build-native", "kivi-run"), "--model", art,
               "--ids-file", ids_txt, "--max-new-tokens", str(gen), "--ignore-eos", "--stats"]
    else:
        cmd = [py, os.path.abspath(__file__), "--worker", "--path", path, "--model", model_id,
               "--ids", ids_file, "--gen", str(gen), "--artifacts", art]
    t_spawn = time.monotonic()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, text=True)
    # wait4 gives this child's own rusage (peak RSS), unlike RUSAGE_CHILDREN.
    out_chunks, err_chunks = [], []
    def pump(src, dst):
        for line in src:
            dst.append(line)
    th = [threading.Thread(target=pump, args=(p.stdout, out_chunks)),
          threading.Thread(target=pump, args=(p.stderr, err_chunks))]
    for t in th:
        t.start()
    _, status, ru = os.wait4(p.pid, 0)
    for t in th:
        t.join()
    p.returncode = os.waitstatus_to_exitcode(status)
    out, err = "".join(out_chunks), "".join(err_chunks)
    if p.returncode != 0:
        raise RuntimeError(f"{path} failed ({p.returncode}):\n{err[-2000:]}")
    peak_mib = ru.ru_maxrss / 1024
    if path == "kivi-run":
        import re
        tokens = [int(x) for x in out.split()]
        m = re.search(r"prefill (\d+) tok in ([\d.]+)s .*decode (\d+) tok, ([\d.]+) tok/s", err)
        mono = float(re.search(r"monotonic ([\d.]+)", err).group(1))
        n_prompt, prefill_s, n_dec, dec_tps = int(m.group(1)), float(m.group(2)), int(m.group(3)), float(m.group(4))
        kv = re.search(r"KV cache .*: ([\d.]+) MiB vs ([\d.]+) MiB fp16 \(([\d.]+)x\)", err)
        r = {"prefill_tps": n_prompt / prefill_s, "decode_tps": dec_tps,
             "ttft_s": mono - t_spawn, "peak_rss_mib": peak_mib, "tokens": tokens,
             "kv_mib": float(kv.group(1)), "kv_fp16_mib": float(kv.group(2)),
             "kv_compression": float(kv.group(3))}
    else:
        j = json.loads(out.strip().splitlines()[-1])
        with open(ids_file) as fh:
            n_prompt = len(json.load(fh))
        r = {"prefill_tps": n_prompt / j["prefill_s"], "decode_tps": (gen - 1) / j["decode_s"],
             "ttft_s": j["first_token_mono"] - t_spawn, "peak_rss_mib": peak_mib,
             "tokens": j["tokens"]}
    return r


def agree(a, b):
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--path"), ap.add_argument("--model"), ap.add_argument("--ids")
    ap.add_argument("--gen", type=int), ap.add_argument("--artifacts")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--only", default=None, help="comma-separated config names")
    ap.add_argument("--out", default=os.path.join(ROOT, "build-native", "bench", "results.json"))
    a = ap.parse_args()
    if a.worker:
        return worker(a)

    env = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}
    env.setdefault("HF_HUB_OFFLINE", "1")
    env["PYTHONPATH"] = os.path.join(ROOT, "src")
    env.setdefault("TMPDIR", os.path.expanduser("~/.cache/kivi_tmp"))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    results = {"machine": machine_info(), "configs": {}}
    configs = [c for c in CONFIGS if a.only is None or c[0] in a.only.split(",")]
    for name, model_id, ctx, gen, art in configs:
        art = os.path.join(ROOT, art)
        ids_file = os.path.join(os.path.dirname(a.out), f"{name}.ids.json")
        n_prompt = make_ids(model_id, ctx, ids_file)
        runs = {p: [] for p in PATHS}
        for i in range(a.runs):
            for path in PATHS:              # interleaved, so drift hits all paths alike
                r = run_path(path, model_id, gen, art, ids_file, env)
                runs[path].append(r)
                print(f"[{name}] run {i + 1}/{a.runs} {path:12s} prefill {r['prefill_tps']:7.1f} tok/s  "
                      f"decode {r['decode_tps']:6.2f} tok/s  TTFT {r['ttft_s']:6.2f}s  "
                      f"peak {r['peak_rss_mib']:6.0f} MiB", flush=True)
        summary = {}
        ref = runs["python-fp"][0]["tokens"]
        for path, rs in runs.items():
            s = {}
            for k in ("prefill_tps", "decode_tps", "ttft_s", "peak_rss_mib"):
                vals = [r[k] for r in rs]
                s[k] = {"median": statistics.median(vals), "min": min(vals), "max": max(vals)}
            s["tokens_agree_with_fp"] = agree(rs[0]["tokens"], ref)
            s["tokens_stable_across_runs"] = all(r["tokens"] == rs[0]["tokens"] for r in rs)
            if "kv_compression" in rs[0]:
                s["kv_mib"], s["kv_fp16_mib"], s["kv_compression"] = (
                    rs[0]["kv_mib"], rs[0]["kv_fp16_mib"], rs[0]["kv_compression"])
            summary[path] = s
        results["configs"][name] = {"model": model_id, "prompt_tokens": n_prompt, "gen": gen,
                                    "runs": runs, "summary": summary}
        with open(a.out, "w") as fh:
            json.dump(results, fh, indent=1)
    print_table(results)


def machine_info():
    def sh(c):
        try:
            return subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
        except OSError:
            return ""
    return {"cpu": sh("grep -m1 'model name' /proc/cpuinfo | cut -d: -f2"),
            "power_profile": sh("powerprofilesctl get"),
            "ac_online": sh("cat /sys/class/power_supply/A*/online"),
            "loadavg_start": sh("cat /proc/loadavg"), "kernel": sh("uname -r"),
            "date": time.strftime("%Y-%m-%d %H:%M")}


def print_table(res):
    print("\n| config | path | prefill tok/s | decode tok/s | TTFT s | peak RSS MiB | greedy = FP |")
    print("|---|---|---|---|---|---|---|")
    for name, c in res["configs"].items():
        for path, s in c["summary"].items():
            f = lambda k, d=1: f"{s[k]['median']:.{d}f} ({s[k]['min']:.{d}f})" if k != "ttft_s" \
                else f"{s[k]['median']:.2f} ({s[k]['min']:.2f})"
            print(f"| {name} | {path} | {f('prefill_tps')} | {f('decode_tps', 2)} | {f('ttft_s')} | "
                  f"{s['peak_rss_mib']['median']:.0f} | {s['tokens_agree_with_fp']}/{c['gen']} |")
    print("\nmedian (min) for rates and TTFT over the runs; machine:", res["machine"])


if __name__ == "__main__":
    sys.exit(main())
