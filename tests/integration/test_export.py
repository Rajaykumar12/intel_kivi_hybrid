"""Exported program with torch.ops.kivi.layer_attention (Phase 5 A.1–A.2):
one op per layer in the graph; the exported program (and its AOTInductor
package) run against an engine registered as id 0 reproduce eager op mode."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
    pytest.skip("XPU device not available", allow_module_level=True)

from transformers import LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM  # noqa: E402

from kivi_sycl import KiviCache, _C  # noqa: E402
from kivi_sycl.export import (RUNTIME_ENGINE_ID, compile_package, export_program,  # noqa: E402
                              exported_constants, load_package)
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


def _run_program(fn, model, ids):
    """Prefill + STEPS decode calls of `fn(ids, positions, state)` against a
    fresh engine registered under the runtime id."""
    cache = KiviCache.from_model(model, group_size=G, residual_length=R)
    _C.register_engine(cache._native, RUNTIME_ENGINE_ID)
    state = torch.zeros(1, dtype=torch.int64)
    try:
        with torch.no_grad():
            lg = fn(ids, torch.arange(ids.shape[1])[None], state)
            nxt, pos, got = lg[:, -1].argmax(-1, keepdim=True), ids.shape[1], []
            for _ in range(STEPS):
                lg = fn(nxt, torch.tensor([[pos]]), state)
                got.append(lg[:, -1])
                nxt, pos = lg[:, -1].argmax(-1, keepdim=True), pos + 1
    finally:
        _C.unregister_engine(RUNTIME_ENGINE_ID)
    assert int(state.item()) == (STEPS + 1) * 2       # one engine call per layer per call
    assert cache.quantized_tokens(0) >= 2 * G
    return got


@pytest.mark.parametrize("kind", ["llama-gqa", "qwen2-gqa"])
def test_exported_program_reproduces_eager_op_mode(kind):
    model = _tiny(kind)
    ids = torch.randint(0, 97, (1, PROMPT))
    ep = export_program(model, max_tokens=64)
    ops = [n for n in ep.graph.nodes if "kivi.layer_attention" in str(n.target)]
    assert len(ops) == 2, "one layer_attention per layer"
    got = _run_program(ep.module(), model, ids)
    ref = _eager_op_mode(model, ids)
    for step, (a, b) in enumerate(zip(ref, got)):
        assert torch.equal(a, b), f"{kind}: step {step}"


def test_aoti_package_matches_eager_op_mode(tmp_path):
    model = _tiny("llama-gqa")
    ids = torch.randint(0, 97, (1, PROMPT))
    ep = export_program(model, max_tokens=64)
    path = compile_package(ep, str(tmp_path / "model.pt2"))
    pkg = load_package(path, exported_constants(ep))
    got = _run_program(pkg, model, ids)
    ref = _eager_op_mode(model, ids)
    for a, b in zip(ref, got):
        torch.testing.assert_close(b, a, rtol=1e-4, atol=1e-5)   # Inductor reorders arithmetic
        assert a.argmax() == b.argmax()


@pytest.mark.parametrize("tied", [False, True])
def test_artifact_directory_roundtrip(tmp_path, tied):
    """write_artifacts -> load_artifacts (the C++ runtime's procedure in
    Python) reproduces eager op mode; tied embeddings resolve to the stored
    tensor and buffers absent from the weights land in extra_constants."""
    import json
    from kivi_sycl.export import load_artifacts, write_artifacts
    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(num_hidden_layers=2, num_attention_heads=4,
                                         num_key_value_heads=2, hidden_size=128,
                                         intermediate_size=256, vocab_size=97,
                                         max_position_embeddings=256,
                                         tie_word_embeddings=tied)).eval()
    snap = tmp_path / "snapshot"
    model.save_pretrained(snap)
    out = tmp_path / "artifacts"
    write_artifacts(str(snap), str(out), max_tokens=64, group_size=G, residual_length=R)
    manifest = json.loads((out / "constants.json").read_text())
    assert manifest["model.lm_head.weight"]["tensor"] == (
        "model.embed_tokens.weight" if tied else "lm_head.weight")
    assert any(m["file"] == "extra_constants.safetensors" for m in manifest.values())

    ids = torch.randint(0, 97, (1, PROMPT))
    run = load_artifacts(str(out))
    try:
        with torch.no_grad():
            lg = run(ids, torch.arange(PROMPT)[None])
            nxt, pos, got = lg[:, -1].argmax(-1, keepdim=True), PROMPT, []
            for _ in range(STEPS):
                lg = run(nxt, torch.tensor([[pos]]))
                got.append(lg[:, -1])
                nxt, pos = lg[:, -1].argmax(-1, keepdim=True), pos + 1
    finally:
        run.close()
    ref = _eager_op_mode(model, ids)
    for a, b in zip(ref, got):
        torch.testing.assert_close(b, a, rtol=1e-4, atol=1e-5)
        assert a.argmax() == b.argmax()


@pytest.mark.parametrize("last_only", [True, False])
def test_logits_last_position_only(last_only):
    """Default export returns logits for the last position only ([1, 1, V]);
    last_logits_only=False returns every position. The last row is the same."""
    model = _tiny("llama-gqa")
    ids = torch.randint(0, 97, (1, PROMPT))
    ep = export_program(model, max_tokens=64, last_logits_only=last_only)
    cache = KiviCache.from_model(model, group_size=G, residual_length=R)
    _C.register_engine(cache._native, RUNTIME_ENGINE_ID)
    try:
        with torch.no_grad():
            lg = ep.module()(ids, torch.arange(PROMPT)[None], torch.zeros(1, dtype=torch.int64))
    finally:
        _C.unregister_engine(RUNTIME_ENGINE_ID)
    assert lg.shape == ((1, 1, 97) if last_only else (1, PROMPT, 97))
    with torch.no_grad():
        ref = model(ids).logits[:, -1]        # model is in op mode now; first call == plain prefill
    torch.testing.assert_close(lg[:, -1], ref, rtol=1e-4, atol=1e-4)
