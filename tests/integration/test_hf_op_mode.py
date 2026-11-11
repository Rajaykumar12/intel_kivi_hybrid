"""Op mode (Phase 5 S1.7): KIVI as one torch.ops.kivi.layer_attention call per
layer, so torch.compile captures a whole decode step with no graph breaks.

Measured on real models (2026-11-10): 0 graph breaks (default path: 48 per
step on Qwen2.5-0.5B); compiled decode 1.20x on Qwen2.5-0.5B @2k and 1.11x on
TinyLlama-1.1B @1.9k, greedy tokens unchanged. These tests pin the properties
behind those numbers on tiny random models.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
    pytest.skip("XPU device not available", allow_module_level=True)

from transformers import (GPT2Config, GPT2LMHeadModel, LlamaConfig,  # noqa: E402
                          LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM)

from kivi_sycl import _C, enable_kivi  # noqa: E402
from kivi_sycl.hf.compiled import enable_kivi_op  # noqa: E402

G, R, PROMPT, STEPS = 8, 16, 11, 30


def _tiny(kind):
    torch.manual_seed(0)
    if kind == "gpt2":
        m = GPT2LMHeadModel(GPT2Config(n_layer=2, n_head=2, n_embd=128, vocab_size=97, n_positions=128))
    elif kind == "llama-gqa":
        m = LlamaForCausalLM(LlamaConfig(num_hidden_layers=2, num_attention_heads=4,
                                         num_key_value_heads=2, hidden_size=128,
                                         intermediate_size=256, vocab_size=97,
                                         max_position_embeddings=128))
    else:
        m = Qwen2ForCausalLM(Qwen2Config(num_hidden_layers=2, num_attention_heads=4,
                                         num_key_value_heads=1, hidden_size=128,
                                         intermediate_size=256, vocab_size=97,
                                         max_position_embeddings=128))
    return m.eval()


def _default_path(model, ids):
    cache = enable_kivi(model, group_size=G, residual_length=R, engine="native")
    out, logits = model(ids, past_key_values=cache, use_cache=True), []
    nxt = out.logits[:, -1].argmax(-1, keepdim=True)
    for _ in range(STEPS):
        out = model(nxt, past_key_values=cache, use_cache=True)
        logits.append(out.logits[:, -1])
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
    return logits


def _op_path(model, ids, fwd=None):
    cache = enable_kivi_op(model, group_size=G, residual_length=R)
    fwd = fwd or model.forward
    kw = cache.model_kwargs
    out = model(ids, past_key_values=cache, use_cache=True,
                position_ids=torch.arange(ids.shape[1])[None], **kw)
    nxt, pos, logits = out.logits[:, -1].argmax(-1, keepdim=True), ids.shape[1], []
    for _ in range(STEPS):
        out = fwd(input_ids=nxt, past_key_values=cache, use_cache=True,
                  position_ids=torch.tensor([[pos]]), **kw)
        logits.append(out.logits[:, -1])
        nxt, pos = out.logits[:, -1].argmax(-1, keepdim=True), pos + 1
    return logits, cache


@pytest.mark.parametrize("kind", ["gpt2", "llama-gqa", "qwen2-gqa"])
def test_op_mode_bit_identical_to_default_fused_path(kind):
    model = _tiny(kind)
    ids = torch.randint(0, 97, (1, PROMPT))
    with torch.no_grad():
        ref = _default_path(model, ids)
        got, cache = _op_path(model, ids)
    for step, (a, b) in enumerate(zip(ref, got)):
        assert torch.equal(a, b), f"{kind}: step {step}"
    layers = cache.kivi.num_layers
    assert int(cache.state.item()) == (STEPS + 1) * layers     # one op call per layer per forward
    assert cache.kivi.quantized_tokens(0) >= 2 * G
    assert cache.get_seq_length() == PROMPT + STEPS


def test_compiled_decode_step_has_no_graph_breaks():
    model = _tiny("llama-gqa")
    ids = torch.randint(0, 97, (1, PROMPT))
    cache = enable_kivi_op(model, group_size=G, residual_length=R)
    kw = cache.model_kwargs
    with torch.no_grad():
        model(ids, past_key_values=cache, use_cache=True,
              position_ids=torch.arange(PROMPT)[None], **kw)
        ex = torch._dynamo.explain(model.forward)(
            input_ids=ids[:, -1:], past_key_values=cache, use_cache=True,
            position_ids=torch.tensor([[PROMPT]]), **kw)
    torch._dynamo.reset()
    assert ex.graph_break_count == 0, [str(r.reason) for r in ex.break_reasons]
    assert ex.graph_count == 1


def test_compiled_op_mode_matches_eager_tokens():
    model = _tiny("qwen2-gqa")
    ids = torch.randint(0, 97, (1, PROMPT))
    with torch.no_grad():
        eager, _ = _op_path(model, ids)
        torch._dynamo.reset()
        compiled, cache = _op_path(model, ids, torch.compile(model.forward, dynamic=False))
    torch._dynamo.reset()
    for a, b in zip(eager, compiled):
        torch.testing.assert_close(b, a, rtol=1e-4, atol=1e-4)
        assert a.argmax() == b.argmax()
    assert int(cache.state.item()) == (STEPS + 1) * cache.kivi.num_layers


def test_registry_lifecycle():
    model = _tiny("gpt2")
    cache = enable_kivi_op(model, group_size=G, residual_length=R)
    eid = cache.engine_id
    assert eid >= 1                                    # 0 is reserved for the C++ runtime
    cache.close()
    q = torch.randn(1, 2, 1, 64)
    with pytest.raises(RuntimeError, match=f"no engine registered with id {eid}"):
        torch.ops.kivi.layer_attention(eid, 0, q, q, q, 0.125, torch.zeros(1, dtype=torch.int64))
    e = _C.Engine(num_layers=1, head_dim=64, residual_length=16, group_size=[8], device="cpu")
    assert _C.register_engine(e, 4242) == 4242
    with pytest.raises(RuntimeError, match="already in use"):
        _C.register_engine(e, 4242)
    _C.unregister_engine(4242)
