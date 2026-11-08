"""Native transformers integration (roadmap G9): KiviHFCache + "kivi" attention.

The HF-native path must be numerically identical to the add_tokens /
get_full_cache path it replaces — same flush timing, same packed bytes, same
SDPA over the same tensors — so every comparison here is exact (torch.equal),
not a tolerance. Tiny randomly initialised models keep it download-free.
"""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
    pytest.skip("XPU device not available", allow_module_level=True)

from transformers import (GPT2Config, GPT2LMHeadModel, LlamaConfig,  # noqa: E402
                          LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM)
from transformers.cache_utils import Cache  # noqa: E402

from kivi_sycl import KiviCache, KiviHFCache, enable_kivi  # noqa: E402

G, R = 8, 16
PROMPT, STEPS = 11, 40          # crosses several flushes at R=16


def _tiny(kind):
    torch.manual_seed(0)
    if kind == "gpt2":
        cfg = GPT2Config(n_layer=2, n_head=2, n_embd=128, vocab_size=97, n_positions=128)
        m = GPT2LMHeadModel(cfg)
    elif kind == "llama-gqa":
        cfg = LlamaConfig(num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                          hidden_size=128, intermediate_size=256, vocab_size=97,
                          max_position_embeddings=128)
        m = LlamaForCausalLM(cfg)
    else:
        cfg = Qwen2Config(num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=1,
                          hidden_size=128, intermediate_size=256, vocab_size=97,
                          max_position_embeddings=128)
        m = Qwen2ForCausalLM(cfg)
    m.set_attn_implementation("sdpa")
    return m.eval()


def _old_path(model, ids, async_flush=True):
    """Greedy decode via add_tokens / get_full_cache (the pre-G9 API)."""
    c = KiviCache.from_model(model, group_size=G, residual_length=R, async_flush=async_flush)
    logits = []
    with torch.no_grad():
        out = model(ids, use_cache=True)
        c.add_tokens(out.past_key_values)
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        for _ in range(STEPS):
            out = model(nxt, past_key_values=c.get_full_cache(), use_cache=True)
            logits.append(out.logits[:, -1])
            c.add_tokens(out.past_key_values)
            nxt = out.logits[:, -1].argmax(-1, keepdim=True)
    c.harvest_pending()
    return logits, c


def _new_path(model, ids, async_flush=True, max_tokens=0):
    """Greedy decode with the HF-native cache, stepping the model by hand."""
    cache = enable_kivi(model, group_size=G, residual_length=R,
                        async_flush=async_flush, max_tokens=max_tokens)
    logits = []
    with torch.no_grad():
        out = model(ids, past_key_values=cache, use_cache=True)
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        for _ in range(STEPS):
            out = model(nxt, past_key_values=cache, use_cache=True)
            logits.append(out.logits[:, -1])
            nxt = out.logits[:, -1].argmax(-1, keepdim=True)
    cache.harvest_pending()
    return logits, cache


@pytest.mark.parametrize("kind", ["gpt2", "llama-gqa", "qwen2-gqa"])
@pytest.mark.parametrize("async_flush", [True, False])
def test_hf_path_bit_identical_to_add_tokens_path(kind, async_flush):
    model = _tiny(kind)
    ids = torch.randint(0, 97, (1, PROMPT))
    old_logits, old = _old_path(model, ids, async_flush)
    new_logits, new = _new_path(model, ids, async_flush)
    assert old._quant_token_count == new.kivi._quant_token_count
    assert new.kivi._quant_token_count[0] >= 2 * G      # history actually used
    for step, (a, b) in enumerate(zip(old_logits, new_logits)):
        assert torch.equal(a, b), f"{kind}: logits differ at decode step {step}"
    # Same packed bytes in the store.
    for name in ("_packed_k", "_scales_k", "_packed_v", "_zeros_v"):
        assert torch.equal(getattr(old, name)[0], getattr(new.kivi, name)[0]), name


def test_generate_api_and_stats():
    model = _tiny("llama-gqa")
    ids = torch.randint(0, 97, (1, PROMPT))
    cache = enable_kivi(model, group_size=G, residual_length=R, max_tokens=PROMPT + STEPS)
    assert isinstance(cache, Cache) and isinstance(cache, KiviHFCache)
    with torch.no_grad():
        out = model.generate(ids, past_key_values=cache, max_new_tokens=STEPS,
                             do_sample=False, pad_token_id=0)
    assert out.shape[1] == PROMPT + STEPS
    cache.harvest_pending()
    s = cache.get_stats()
    assert s["total_tokens"] == PROMPT + STEPS - 1     # last token never fed back
    assert s["quantized_tokens"] > 0 and s["compression"] > 1.0
    # The engine never builds a full-length FP serving view on this path.
    assert cache.kivi._cache_view is None
    assert cache.get_seq_length() == s["total_tokens"]


def test_untagged_tensors_fall_back_to_sdpa():
    """With "kivi" selected but a plain DynamicCache passed, attention must
    behave exactly like sdpa."""
    model = _tiny("gpt2")
    ids = torch.randint(0, 97, (1, PROMPT))
    with torch.no_grad():
        ref = model(ids).logits
        enable_kivi(model)
        got = model(ids).logits
    assert torch.equal(ref, got)


def test_unsupported_cache_ops_raise():
    cache = enable_kivi(_tiny("gpt2"), group_size=G, residual_length=R)
    with pytest.raises(NotImplementedError):
        cache.reorder_cache(torch.tensor([0]))
