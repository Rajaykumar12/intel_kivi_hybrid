"""torch.ops.kivi.* — dispatcher-registered ops (Phase 5 S1.2).

torch.library.opcheck validates each op's schema (incl. the mutation
annotations on the quantize outputs), its Meta kernel under FakeTensor, and
AOT-dispatch tracing — the properties torch.compile / torch.export rely on.
"""

import pytest

torch = pytest.importorskip("torch")
try:
    import kivi_sycl._C  # noqa: F401  (registers torch.ops.kivi)
except ImportError:
    pytest.skip("compiled extension not built", allow_module_level=True)

import test_fused_attention as fa  # noqa: E402  (random packed-history helpers)

G, D = 32, 64


def _fused_args(Hq=4, Hkv=2, T=128, NR=9):
    torch.manual_seed(0)
    q = torch.randn(1, Hq, 1, D)
    kr = torch.randn(1, Hkv, NR, D)
    vr = torch.randn(1, Hkv, NR, D)
    hist = fa._random_history(1, Hkv, T, D, G, torch.float16)
    return (q, kr, vr, *hist, D ** -0.5, G)


def test_ops_are_registered():
    for name in ("quantize_keys", "quantize_values", "fused_decode_attention"):
        assert hasattr(torch.ops.kivi, name)
    schema = str(torch.ops.kivi.quantize_keys.default._schema)
    assert "Tensor(a!) packed" in schema and "Tensor(b!) scales" in schema


def test_fused_op_matches_pybind_function():
    from kivi_sycl import _C
    args = _fused_args()
    assert torch.equal(torch.ops.kivi.fused_decode_attention(*args),
                       _C.fused_decode_attention(*args))


@pytest.mark.parametrize("shape", [(4, 2, 128, 9), (14, 2, 320, 0), (4, 4, 32, 1)])
def test_opcheck_fused_decode_attention(shape):
    torch.library.opcheck(torch.ops.kivi.fused_decode_attention.default, _fused_args(*shape))


def test_fused_op_meta_shape():
    args = _fused_args(Hq=14, Hkv=2)
    meta = [a.to("meta") if isinstance(a, torch.Tensor) else a for a in args]
    out = torch.ops.kivi.fused_decode_attention(*meta)
    assert out.device.type == "meta" and tuple(out.shape) == (1, 1, 14, D)
