"""Golden reference for kivi-run (Phase 5 A.4).

Runs greedy decoding in Python eager op mode (HF model + kivi_op attention,
the reference the exported artifacts were verified against) and writes
{model, prompt, prompt_ids, tokens} for `kivi-run --golden`:

    python tools/runtime_golden.py --artifacts artifacts/tinyllama \
        --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --steps 200 --out golden_run.json
    kivi-run --model artifacts/tinyllama --golden golden_run.json

The tokenizer files travel with the artifacts, so the prompt ids come from
the artifact directory. No EOS stop: exactly --steps tokens are generated.
"""

import argparse
import json

import torch

_META = {"auto": "auto", "float16": torch.float16, "float32": torch.float32}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--artifacts", required=True, help="artifact directory (kivi_config.json)")
    ap.add_argument("--model", required=True, help="HF id or directory the artifacts came from")
    ap.add_argument("--prompt", default="Once upon a time, in a small village by the sea,")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from kivi_sycl.hf.compiled import enable_kivi_op

    with open(f"{a.artifacts}/kivi_config.json") as fh:
        cfg = json.load(fh)
    tok = AutoTokenizer.from_pretrained(a.artifacts)
    ids = tok(a.prompt, return_tensors="pt").input_ids
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()
    cache = enable_kivi_op(model, group_size=cfg["group_size"],
                           residual_length=cfg["residual_length"],
                           metadata_dtype=_META[cfg["metadata_dtype"]])
    kw, out = cache.model_kwargs, []
    with torch.no_grad():
        o = model(ids, past_key_values=cache, use_cache=True,
                  position_ids=torch.arange(ids.shape[1])[None], **kw)
        nxt, pos = o.logits[:, -1].argmax(-1, keepdim=True), ids.shape[1]
        for _ in range(a.steps):
            out.append(int(nxt.item()))
            o = model(nxt, past_key_values=cache, use_cache=True,
                      position_ids=torch.tensor([[pos]]), **kw)
            nxt, pos = o.logits[:, -1].argmax(-1, keepdim=True), pos + 1
    cache.close()
    with open(a.out, "w") as fh:
        json.dump({"model": a.model, "prompt": a.prompt, "prompt_ids": ids[0].tolist(),
                   "tokens": out}, fh)
    print(f"wrote {a.out}: {ids.shape[1]} prompt ids, {len(out)} tokens")
    print("text:", repr(tok.decode(out))[:160])


if __name__ == "__main__":
    main()
