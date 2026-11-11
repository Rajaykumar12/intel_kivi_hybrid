"""kivi::Engine on the real XPU kernels: the zero-copy (USM arena written
directly by the kernels) and staged (device temporaries copied in at harvest)
paths must store identical bytes, in both update orders."""

import pytest

torch = pytest.importorskip("torch")
if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
    pytest.skip("XPU device not available", allow_module_level=True)

from kivi_sycl import _C  # noqa: E402


def _run(zero_copy, order, async_flush):
    torch.manual_seed(0)
    e = _C.Engine(num_layers=2, head_dim=64, residual_length=64, group_size=[16, 32],
                  zero_copy=zero_copy, async_flush=async_flush, device="xpu")
    for n in [70] + [1] * 80:
        for i in range(2):
            k, v = torch.randn(1, 2, n, 64), torch.randn(1, 2, n, 64)
            if order == "update":
                e.update(i, k, v)
            else:
                e.append(i, k, v)
        if order == "append":
            e.harvest_all()
    e.harvest_all()
    return e


@pytest.mark.parametrize("order", ["update", "append"])
@pytest.mark.parametrize("async_flush", [True, False])
def test_zero_copy_and_staged_paths_identical(order, async_flush):
    a = _run("on", order, async_flush)
    b = _run("off", order, async_flush)
    assert a.zero_copy_enabled and not b.zero_copy_enabled
    for i in range(2):
        assert a.quant_tokens(i) == b.quant_tokens(i) > 0
        for x, y in zip(a.history(i), b.history(i)):
            assert torch.equal(x, y)
        assert all(torch.equal(x, y) for x, y in zip(a.residual(i), b.residual(i)))
