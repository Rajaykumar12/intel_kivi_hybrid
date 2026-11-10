"""python -m kivi_sycl.export --model <id|dir> --out <artifact dir> [options]

Exports a HuggingFace causal LM into an artifact directory for the
Python-free KIVI runtime (see kivi_sycl/export/artifacts.py)."""

import argparse
import os
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--model", required=True, help="HuggingFace model id or local snapshot dir")
    ap.add_argument("--out", required=True, help="artifact directory to write")
    ap.add_argument("--max-tokens", type=int, default=2048,
                    help="largest number of tokens per call (prefill chunk); decode is 1")
    ap.add_argument("--group-size", type=str, default="32", help="G, or a comma list per layer")
    ap.add_argument("--residual", type=int, default=128, help="residual window R")
    ap.add_argument("--metadata-dtype", choices=["float16", "float32"], default="float16")
    args = ap.parse_args()

    # Inductor stages builds in TORCHINDUCTOR_CACHE_DIR, default /tmp — often a
    # RAM-backed tmpfs, which a 1B model's build can exhaust. Default to disk.
    # (No spaces: Inductor does not quote paths in its g++ command.)
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR",
                          os.path.join(os.path.expanduser("~"), ".cache", "kivi_inductor"))

    from .artifacts import write_artifacts
    gs = [int(g) for g in args.group_size.split(",")]
    t0 = time.perf_counter()
    cfg = write_artifacts(args.model, args.out, max_tokens=args.max_tokens,
                          group_size=gs[0] if len(gs) == 1 else gs,
                          residual_length=args.residual, metadata_dtype=args.metadata_dtype)
    size = sum(os.path.getsize(os.path.join(args.out, f)) for f in os.listdir(args.out))
    print(f"exported {args.model} -> {args.out} in {time.perf_counter() - t0:.1f}s "
          f"({size / 2**20:.0f} MiB; {cfg['num_layers']} layers, {cfg['num_kv_heads']} KV heads, "
          f"head_dim {cfg['head_dim']}, max_tokens {cfg['max_tokens']})")


if __name__ == "__main__":
    main()
