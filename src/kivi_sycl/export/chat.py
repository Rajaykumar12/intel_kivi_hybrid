"""Chat templates for the Python-free runtime (Phase 5 A.8).

HF chat templates are Jinja programs; the C++ runtime does not embed Jinja.
Instead the exporter *derives* the template's literal pieces by rendering it
with sentinel contents and taking differences:

    no_system_prefix                       text before the first user message
                                           when there is no system message
                                           (e.g. Qwen's default system block)
    system   [prefix, suffix]              around a leading system message
    user / assistant [prefix, suffix]      around each message's content
    generation_prompt                      appended by add_generation_prompt

The rendering this describes —

    (system.prefix + S + system.suffix  |  no_system_prefix)
    + for each later message: role.prefix + content + role.suffix
    + generation_prompt

— is then checked against the real template on a set of conversations
(with and without a system message, 1–3 turns, contents with whitespace,
newlines, quotes and special-token text). Only if every one renders
byte-identically is the pattern written; otherwise the artifact has no chat
template and ``kivi-run --chat`` refuses rather than guessing.
"""

import json
import os

__all__ = ["derive_chat_template", "render_chat", "write_generation_json", "VERIFY_CONVERSATIONS"]

_S = ["⟦K%d⟧" % i for i in range(8)]      # sentinels: ⟦K0⟧ ...


def _render(tok, messages, gen):
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=gen)


def _between(text, a, b):
    i = text.index(a) + len(a)
    return text[i:text.index(b, i)]


def _split_delta(longer, shorter, content):
    """`longer` = `shorter` + prefix + content + suffix  ->  (prefix, suffix)."""
    if not longer.startswith(shorter):
        raise ValueError("template is not append-only")
    delta = longer[len(shorter):]
    if delta.count(content) != 1:
        raise ValueError("content not found exactly once")
    p, s = delta.split(content)
    return p, s


def render_chat(pattern, messages, add_generation_prompt=True):
    """Python mirror of the C++ ChatTemplate::render."""
    out, rest = "", list(messages)
    if rest and rest[0]["role"] == "system":
        p, s = pattern["system"]
        out += p + rest[0]["content"] + s
        rest = rest[1:]
    else:
        out += pattern["no_system_prefix"]
    for m in rest:
        if m["role"] not in ("user", "assistant"):
            raise ValueError(f"unsupported role {m['role']!r} here")
        p, s = pattern[m["role"]]
        out += p + m["content"] + s
    if add_generation_prompt:
        out += pattern["generation_prompt"]
    return out


VERIFY_CONVERSATIONS = [
    [{"role": "user", "content": "Hello!"}],
    [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Hi"}],
    [{"role": "user", "content": "  padded  "}, {"role": "assistant", "content": "ok"},
     {"role": "user", "content": "line1\nline2\n"}],
    [{"role": "system", "content": "S\nwith newline"}, {"role": "user", "content": "q1"},
     {"role": "assistant", "content": "a1"}, {"role": "user", "content": "q2"},
     {"role": "assistant", "content": "a2 \"quoted\" {braces} {{jinja}}"},
     {"role": "user", "content": "q3 <|im_end|> </s> <s>"}],
    [{"role": "user", "content": ""}, {"role": "assistant", "content": ""},
     {"role": "user", "content": "empty turns before"}],
    [{"role": "user", "content": "日本語 🤖 naïve"}],
]


def derive_chat_template(tok):
    """Return (pattern, None) or (None, reason)."""
    if not getattr(tok, "chat_template", None):
        return None, "tokenizer has no chat_template"
    try:
        s0, u1, a1, u2 = _S[0], _S[1], _S[2], _S[3]
        sys_ = [{"role": "system", "content": s0}]
        user1 = [{"role": "user", "content": u1}]
        asst1 = [{"role": "assistant", "content": a1}]
        user2 = [{"role": "user", "content": u2}]

        # user / assistant: differences of growing conversations.
        base = _render(tok, sys_ + user1, False)
        with_a = _render(tok, sys_ + user1 + asst1, False)
        with_u2 = _render(tok, sys_ + user1 + asst1 + user2, False)
        a_p, a_s = _split_delta(with_a, base, a1)
        u_p, u_s = _split_delta(with_u2, with_a, u2)
        # system: everything before / after its content, up to the first user prefix.
        pre, post = base.split(s0)
        if not post.endswith(u_p + u1 + u_s):
            raise ValueError("system block is not followed by the user message")
        sys_p, sys_s = pre, post[: len(post) - len(u_p + u1 + u_s)]
        # no system message: whatever precedes the first user block.
        nosys = _render(tok, user1, False)
        if not nosys.endswith(u_p + u1 + u_s):
            raise ValueError("unexpected rendering without a system message")
        nosys_p = nosys[: len(nosys) - len(u_p + u1 + u_s)]
        g_on = _render(tok, sys_ + user1, True)
        if not g_on.startswith(base):
            raise ValueError("generation prompt is not appended")
        pattern = {"no_system_prefix": nosys_p, "system": [sys_p, sys_s],
                   "user": [u_p, u_s], "assistant": [a_p, a_s],
                   "generation_prompt": g_on[len(base):]}
    except Exception as e:                                     # noqa: BLE001 — report, don't guess
        return None, f"template is not a simple concatenation ({e})"
    for conv in VERIFY_CONVERSATIONS:
        for gen in (False, True):
            want = _render(tok, conv, gen)
            got = render_chat(pattern, conv, gen)
            if got != want:
                return None, f"derived pattern mismatches the template on {conv!r} (gen={gen})"
    return pattern, None


def write_generation_json(snapshot: str, out_dir: str) -> dict:
    """Write out_dir/generation.json: chat pattern (if derivable) and the
    model's default sampling settings."""
    from transformers import AutoTokenizer, GenerationConfig
    gen = {"chat": None, "chat_error": None, "sampling": None}
    has_tok = any(os.path.exists(os.path.join(snapshot, f))
                  for f in ("tokenizer.json", "tokenizer_config.json"))
    if has_tok:
        tok = AutoTokenizer.from_pretrained(snapshot)
        gen["chat"], gen["chat_error"] = derive_chat_template(tok)
        if gen["chat"] is not None:
            gen["chat"]["eos_token"] = tok.eos_token
    try:
        gc = GenerationConfig.from_pretrained(snapshot)
        if gc.do_sample:
            gen["sampling"] = {"temperature": gc.temperature or 1.0, "top_k": gc.top_k or 0,
                               "top_p": gc.top_p if gc.top_p is not None else 1.0,
                               "repetition_penalty": gc.repetition_penalty or 1.0}
    except OSError:
        pass                                                   # no generation_config.json
    with open(os.path.join(out_dir, "generation.json"), "w", encoding="utf-8") as fh:
        json.dump(gen, fh, indent=1, ensure_ascii=False)
    return gen
