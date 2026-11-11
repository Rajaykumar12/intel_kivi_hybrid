"""Chat template derivation for the C++ runtime (Phase 5 A.8)."""

import pytest

from kivi_sycl.export.chat import VERIFY_CONVERSATIONS, derive_chat_template, render_chat

MODELS = ["TinyLlama/TinyLlama-1.1B-Chat-v1.0", "Qwen/Qwen2.5-0.5B-Instruct"]


def _tokenizer(repo):
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(repo, local_files_only=True)
    except OSError:
        pytest.skip(f"{repo} tokenizer not in the local HF cache")


@pytest.mark.parametrize("repo", MODELS)
def test_derived_pattern_reproduces_template(repo):
    tok = _tokenizer(repo)
    pattern, err = derive_chat_template(tok)
    assert err is None and pattern is not None
    for conv in VERIFY_CONVERSATIONS:
        for gen in (False, True):
            want = tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=gen)
            assert render_chat(pattern, conv, gen) == want
    # multi-turn chat in kivi-run relies on these two being the same text
    assert pattern["generation_prompt"] == pattern["assistant"][0]


def test_non_concatenative_template_is_rejected():
    tok = _tokenizer(MODELS[0])
    tok.chat_template = ("{% for m in messages %}<|{{ m['role'] }}|>\n{{ m['content'] | trim }}"
                         "</s>\n{% endfor %}{% if add_generation_prompt %}<|assistant|>\n{% endif %}")
    pattern, err = derive_chat_template(tok)
    assert pattern is None and "mismatch" in err


def test_no_template():
    tok = _tokenizer(MODELS[0])
    tok.chat_template = None
    pattern, err = derive_chat_template(tok)
    assert pattern is None and "no chat_template" in err
