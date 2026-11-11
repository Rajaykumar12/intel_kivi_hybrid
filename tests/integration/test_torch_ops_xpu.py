"""torch.ops.kivi quantize ops on XPU tensors and USM-backed CPU tensors."""

import pytest

torch = pytest.importorskip("torch")
if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
    pytest.skip("XPU device not available", allow_module_level=True)

from kivi_sycl import _C  # noqa: E402

B, H, D, G = 1, 2, 64, 32


def _key_args(where, meta=torch.float16, L=2 * G):
    x = torch.randn(B, H, D, L, device="xpu").contiguous()
    if where == "xpu":
        mk = lambda shape, dt: torch.empty(shape, dtype=dt, device="xpu")
    else:
        mk = lambda shape, dt: _C.alloc_usm_host(list(shape), dt)
    return (x, mk((B, H, D, L // 4), torch.uint8), mk((B, H, D, L // G), meta),
            mk((B, H, D, L // G), meta), G)


@pytest.mark.parametrize("where", ["xpu", "usm"])
def test_quantize_keys_op_matches_pybind(where):
    torch.manual_seed(0)
    a = _key_args(where)
    b = (a[0],) + tuple(torch.empty_like(t) if where == "xpu" else
                        _C.alloc_usm_host(list(t.shape), t.dtype) for t in a[1:4]) + (G,)
    torch.ops.kivi.quantize_keys(*a)
    _C.quantize_keys(*b)
    for x, y in zip(a[1:4], b[1:4]):
        assert torch.equal(x.cpu(), y.cpu())


# Why not a full torch.library.opcheck for the quantize ops: its test_schema
# step (a) clones arguments into ordinary CPU memory, which these ops rightly
# reject for USM-only inputs, and (b) compares XPU tensors with
# torch.allclose, which hits torch 2.13's broken XPU dtype-cast kernel on Iris
# Xe (UR_RESULT_ERROR_INVALID_ARGUMENT; docs/status_and_roadmap.md §6.2).
# The properties opcheck guards are covered here instead: FakeTensor
# behaviour (opcheck's test_faketensor), mutation annotations in the schema
# (unit test), and graph capture matching eager.


def _key_args_xpu(meta, L=2 * G):
    x = torch.randn(B, H, D, L, device="xpu").contiguous()
    return (x, torch.empty(B, H, D, L // 4, dtype=torch.uint8, device="xpu"),
            torch.empty(B, H, D, L // G, dtype=meta, device="xpu"),
            torch.empty(B, H, D, L // G, dtype=meta, device="xpu"), G)


@pytest.mark.parametrize("meta", [torch.float16, torch.float32])
def test_opcheck_faketensor_quantize_ops(meta):
    torch.manual_seed(1)
    torch.library.opcheck(torch.ops.kivi.quantize_keys.default, _key_args_xpu(meta),
                          test_utils=("test_faketensor",))
    xv = torch.randn(B, H, 3 * G, D, device="xpu")
    pv = torch.empty(B, H, 3 * G, D // 4, dtype=torch.uint8, device="xpu")
    sv = torch.empty(B, H, 3 * G, D // G, dtype=meta, device="xpu")
    torch.library.opcheck(torch.ops.kivi.quantize_values.default,
                          (xv, pv, sv, torch.empty_like(sv), D, G),
                          test_utils=("test_faketensor",))


def test_compiled_graph_keeps_quantize_writes():
    """A torch.compile'd function calling the mutating ops must produce the
    same packed bytes as eager — i.e. capture keeps the writes."""
    def f(x, p, s, z):
        torch.ops.kivi.quantize_keys(x, p, s, z, G)
        return p, s, z

    torch.manual_seed(2)
    eager = _key_args_xpu(torch.float16)
    comp = (eager[0],) + tuple(torch.empty_like(t) for t in eager[1:4])
    f(*eager[:4])
    torch.compile(f, fullgraph=True)(*comp)
    for a, b in zip(eager[1:4], comp[1:]):
        assert torch.equal(a.cpu(), b.cpu())
