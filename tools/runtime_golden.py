"""Golden reference for kivi-run (Phase 5 A.4).

Runs greedy decoding in Python eager op mode (HF model + kivi_op attention,
the reference the exported artifacts were verified against) and writes
{model, prompt, prompt_ids, tokens} for `kivi-run --golden`:

    python tools/runtime_golden.py --artifacts artifacts/tinyllama \
        --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --steps 200 --out golden_run.json
    kivi-run --model artifacts/tinyllama --golden golden_run.json

The tokenizer files travel with the artifacts, so the prompt ids come from
the artifact directory. No EOS stop: exactly --steps tokens are generated.

Chat golden (A.8): with --chat-turn (repeatable) and optional --system, the
reference is HF generate() replying greedily turn by turn, re-rendering the
whole conversation with the chat template each time — what a transformers
user gets. `kivi-run --golden` replays the turns through its incremental
chat loop (KV cache kept, only each turn's delta fed) and compares the reply
texts. The conversation must stay within the FP residual window, where KIVI
has quantized nothing and the two are exactly comparable (the quantized
numerics are covered by the token golden above):

    python tools/runtime_golden.py --artifacts artifacts/qwen25 --model Qwen/Qwen2.5-0.5B-Instruct \
        --chat-turn "Name a colour." --chat-turn "Another?" --out build-native/golden/qwen25_chat.json
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
    ap.add_argument("--chat-turn", action="append", default=[], help="user turn (repeatable)")
    ap.add_argument("--system", default=None)
    ap.add_argument("--reply-tokens", type=int, default=40)
    a = ap.parse_args()
    if a.chat_turn:
        return chat_golden(a)

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


def chat_golden(a):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    with open(f"{a.artifacts}/kivi_config.json") as fh:
        cfg = json.load(fh)
    tok = AutoTokenizer.from_pretrained(a.artifacts)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()
    conv = [{"role": "system", "content": a.system}] if a.system is not None else []
    replies, total = [], 0
    for turn in a.chat_turn:
        conv.append({"role": "user", "content": turn})
        ids = tok.apply_chat_template(conv, tokenize=True, add_generation_prompt=True,
                                      return_tensors="pt")
        ids = ids["input_ids"] if hasattr(ids, "keys") else ids
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=a.reply_tokens, do_sample=False,
                                 eos_token_id=cfg["eos_token_ids"], pad_token_id=cfg["eos_token_ids"][0])
        new = out[0, ids.shape[1]:]
        total = ids.shape[1] + len(new)
        reply = tok.decode(new, skip_special_tokens=True)
        replies.append(reply)
        conv.append({"role": "assistant", "content": reply})
    if total > cfg["residual_length"]:
        raise SystemExit(f"conversation reached {total} tokens > residual_length "
                         f"{cfg['residual_length']}: shorten it (see the docstring)")
    with open(a.out, "w") as fh:
        json.dump({"model": a.model, "chat": {"system": a.system, "turns": a.chat_turn,
                                              "reply_tokens": a.reply_tokens},
                   "replies": replies}, fh, ensure_ascii=False)
    print(f"wrote {a.out}: {len(replies)} turns, {total} tokens")
    for t, r in zip(a.chat_turn, replies):
        print(f"  > {t}\n  {r!r}")


if __name__ == "__main__":
    main()
