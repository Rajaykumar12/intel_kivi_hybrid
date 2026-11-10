"""Exported prefill/decode programs with torch.ops.kivi.layer_attention
(Phase 5 A.1): one op per layer in the graph, and running the programs
against an engine registered as id 0 reproduces eager op mode bit for bit."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
    pytest.skip("XPU device not available", allow_module_level=True)

from transformers import LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM  # noqa: E402

from kivi_sycl import KiviCache, _C  # noqa: E402
from kivi_sycl.export import RUNTIME_ENGINE_ID, export_programs  # noqa: E402
from kivi_sycl.hf.compiled import enable_kivi_op  # noqa: E402

G, R, PROMPT, STEPS = 8, 16, 13, 30


def _tiny(kind):
    torch.manual_seed(0)
    if kind == "llama-gqa":
        return LlamaForCausalLM(LlamaConfig(num_hidden_layers=2, num_attention_heads=4,
                                            num_key_value_heads=2, hidden_size=128,
                                            intermediate_size=256, vocab_size=97,
                                            max_position_embeddings=256)).eval()
    return Qwen2ForCausalLM(Qwen2Config(num_hidden_layers=2, num_attention_heads=4,
                                        num_key_value_heads=1, hidden_size=128,
                                        intermediate_size=256, vocab_size=97,
                                        max_position_embeddings=256)).eval()


def _eager_op_mode(model, ids):
    cache = enable_kivi_op(model, group_size=G, residual_length=R)
    kw = cache.model_kwargs
    with torch.no_grad():
        out = model(ids, past_key_values=cache, use_cache=True,
                    position_ids=torch.arange(ids.shape[1])[None], **kw)
        nxt, pos, logits = out.logits[:, -1].argmax(-1, keepdim=True), ids.shape[1], []
        for _ in range(STEPS):
            out = model(nxt, past_key_values=cache, use_cache=True,
                        position_ids=torch.tensor([[pos]]), **kw)
            logits.append(out.logits[:, -1])
            nxt, pos = out.logits[:, -1].argmax(-1, keepdim=True), pos + 1
    cache.close()
    return logits


@pytest.mark.parametrize("kind", ["llama-gqa", "qwen2-gqa"])
def test_exported_programs_reproduce_eager_op_mode(kind):
    model = _tiny(kind)
    ids = torch.randint(0, 97, (1, PROMPT))
    prefill, decode = export_programs(model, max_prompt=64)
    for ep in (prefill, decode):
        ops = [n for n in ep.graph.nodes if "kivi.layer_attention" in str(n.target)]
        assert len(ops) == 2, "one layer_attention per layer"

    cache = KiviCache.from_model(model, group_size=G, residual_length=R)
    _C.register_engine(cache._native, RUNTIME_ENGINE_ID)
    try:
        pre, dec = prefill.module(), decode.module()
        state = torch.zeros(1, dtype=torch.int64)
        with torch.no_grad():
            lg = pre(ids, torch.arange(PROMPT)[None], state)
            nxt, pos, got = lg[:, -1].argmax(-1, keepdim=True), PROMPT, []
            for _ in range(STEPS):
                lg = dec(nxt, torch.tensor([[pos]]), state)
                got.append(lg[:, -1])
                nxt, pos = lg[:, -1].argmax(-1, keepdim=True), pos + 1
    finally:
        _C.unregister_engine(RUNTIME_ENGINE_ID)
    assert int(state.item()) == (STEPS + 1) * 2
    assert cache.quantized_tokens(0) >= 2 * G

    ref = _eager_op_mode(model, ids)
    for step, (a, b) in enumerate(zip(ref, got)):
        assert torch.equal(a, b), f"{kind}: step {step}"
