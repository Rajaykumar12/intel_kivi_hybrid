"""C++ kivi::Engine vs the Python KiviCache, step by step, on the CPU
(Phase 5 S1.4). The Python side uses the torch fake kernels from
test_manager_bookkeeping; the native side its CPU reference quantize — the
same math, so every residual, packed byte, counter and stat must match."""

import os
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
try:
    from kivi_sycl import _C
except ImportError:
    pytest.skip("compiled extension not built", allow_module_level=True)
if not hasattr(_C, "Engine"):
    pytest.skip("extension built without the native engine", allow_module_level=True)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import engine_driver  # noqa: E402
import kivi_sycl.cache.manager as manager_module  # noqa: E402
from test_manager_bookkeeping import _fake_native  # noqa: E402


@pytest.fixture(autouse=True)
def _use_fake_native(monkeypatch):
    monkeypatch.setattr(manager_module, "kivi_native", _fake_native)


@pytest.mark.parametrize("order", ["hf", "legacy"])
@pytest.mark.parametrize("async_flush", [True, False])
@pytest.mark.parametrize("G", [4, [4, 8]])
def test_native_engine_matches_python_cpu(order, async_flush, G):
    engine_driver.run(_C, order, G=G, async_flush=async_flush)


@pytest.mark.parametrize("reserve", [None, 11 + 40, 11 + 400])
def test_native_engine_reserve_and_growth_cpu(reserve):
    engine_driver.run(_C, "hf", reserve=reserve, steps=120)


def test_native_engine_fp32_metadata_cpu():
    engine_driver.run(_C, "hf", meta=torch.float32)


def test_native_engine_rejects_fp32_meta_for_fp16_model():
    nat = _C.Engine(num_layers=1, head_dim=8, residual_length=8, group_size=[4],
                    metadata_dtype="float32", device="cpu")
    k = torch.randn(1, 2, 3, 8, dtype=torch.float16)
    with pytest.raises(ValueError, match="fp32 model"):
        nat.update(0, k, k.clone())


def test_native_engine_reset_cpu():
    _, nat = engine_driver.run(_C, "hf", steps=20)
    nat.reset()
    assert nat.residual(0) is None and nat.history(0) is None
    assert nat.quant_tokens(0) == 0 and nat.total_flushes == 0


def test_native_engine_validates_config():
    with pytest.raises(RuntimeError, match="entries for 3 layers"):
        _C.Engine(num_layers=3, head_dim=8, residual_length=8, group_size=[4, 8], device="cpu")
    with pytest.raises(RuntimeError, match="divisible by group_size"):
        _C.Engine(num_layers=1, head_dim=8, residual_length=16, group_size=[12], device="cpu")
