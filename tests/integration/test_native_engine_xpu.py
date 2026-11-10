"""C++ kivi::Engine vs the Python KiviCache on the real XPU kernels (Phase 5
S1.4): zero-copy (USM arena) and staged paths, both update orders."""

import os
import sys

import pytest

torch = pytest.importorskip("torch")
if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
    pytest.skip("XPU device not available", allow_module_level=True)

from kivi_sycl import _C  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import engine_driver  # noqa: E402


@pytest.mark.parametrize("order", ["hf", "legacy"])
@pytest.mark.parametrize("zero_copy", [True, False])
@pytest.mark.parametrize("async_flush", [True, False])
def test_native_engine_matches_python_xpu(order, zero_copy, async_flush):
    engine_driver.run(_C, order, device="xpu", head_dim=64, G=32, R=64, heads=2,
                      zero_copy=zero_copy, async_flush=async_flush, prefill=70, steps=80)


def test_native_engine_mixed_g_reserve_xpu():
    engine_driver.run(_C, "hf", device="xpu", head_dim=64, G=[16, 32], R=64,
                      reserve=70 + 80, prefill=70, steps=80)
