"""kivi::Engine (_C.Engine) directly: configuration, validation, lifecycle.

The cache *behaviour* — flush timing, residual window, packed bytes, reserve
and growth — is checked against the KIVI reference math in
test_manager_bookkeeping.py and on the real kernels in tests/integration/.
(Until Phase 5 S1.8 the engine was also diffed step by step against the
former Python engine: identical residual, packed bytes and stats at every
step, CPU and XPU, both update orders.)
"""

import pytest

torch = pytest.importorskip("torch")
try:
    from kivi_sycl import _C
except ImportError:
    pytest.skip("compiled extension not built", allow_module_level=True)
if not hasattr(_C, "Engine"):
    pytest.skip("extension built without the native engine", allow_module_level=True)


def _feed(e, steps, heads=2, D=8, prefill=11):
    torch.manual_seed(0)
    for n in [prefill] + [1] * steps:
        e.update(0, torch.randn(1, heads, n, D), torch.randn(1, heads, n, D))


def test_rejects_fp32_metadata_for_fp16_model():
    e = _C.Engine(num_layers=1, head_dim=8, residual_length=8, group_size=[4],
                  metadata_dtype="float32", device="cpu")
    k = torch.randn(1, 2, 3, 8, dtype=torch.float16)
    with pytest.raises(ValueError, match="fp32 model"):
        e.update(0, k, k.clone())


def test_validates_config():
    with pytest.raises(RuntimeError, match="entries for 3 layers"):
        _C.Engine(num_layers=3, head_dim=8, residual_length=8, group_size=[4, 8], device="cpu")
    with pytest.raises(RuntimeError, match="divisible by group_size"):
        _C.Engine(num_layers=1, head_dim=8, residual_length=16, group_size=[12], device="cpu")
    with pytest.raises(RuntimeError, match="must not exceed residual_length"):
        _C.Engine(num_layers=1, head_dim=64, residual_length=16, group_size=[32], device="cpu")


def test_reset_clears_everything():
    e = _C.Engine(num_layers=1, head_dim=8, residual_length=8, group_size=[4], device="cpu")
    _feed(e, 20)
    assert e.total_flushes > 0 and e.history(0) is not None
    e.reset()
    assert e.residual(0) is None and e.history(0) is None
    assert e.quant_tokens(0) == 0 and e.total_flushes == 0 and e.arena_capacity(0) == 0


def test_residual_window_and_counts():
    e = _C.Engine(num_layers=1, head_dim=8, residual_length=8, group_size=[4], device="cpu")
    _feed(e, 40)
    e.harvest_all()
    q, r = e.quant_tokens(0), e.residual(0)[0].shape[2]
    assert q % 4 == 0 and q + r == e.seq_length(0) == 11 + 40
    assert 8 - 4 <= r <= 8 - 1
    s = e.stats(0)
    assert s["quantized_tokens"] == q and s["residual_tokens"] == r
    assert s["packed_bytes"] >= s["packed_used_bytes"] > 0
