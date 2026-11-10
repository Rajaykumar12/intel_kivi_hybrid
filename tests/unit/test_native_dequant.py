"""C++ dequantize-on-read (kivi::dequant_*) vs the Python KiviCache version.

Phase 5 S1.3 exit check: the port uses the same ATen op sequence, so the
output must be bit-identical (torch.equal), including on strided arena views
and with per-layer group sizes.
"""

import pytest

torch = pytest.importorskip("torch")
try:
    from kivi_sycl import _C
except ImportError:
    pytest.skip("compiled extension not built", allow_module_level=True)
if not hasattr(_C, "dequant_keys_cpu"):
    pytest.skip("extension built without the native dequant", allow_module_level=True)

from kivi_sycl.cache.manager import KiviCache  # noqa: E402


@pytest.mark.parametrize("B,H,D,T,G,meta,out,cap", [
    (1, 2, 64, 128, 32, torch.float16, torch.float32, None),
    (2, 3, 64, 96, 16, torch.float32, torch.float32, 160),     # strided arena views
    (1, 4, 128, 256, 64, torch.float16, torch.float16, None),
])
def test_native_dequant_bit_identical_to_python(B, H, D, T, G, meta, out, cap):
    torch.manual_seed(0)
    cap = cap or T
    pk = torch.randint(0, 256, (B, H, D, cap // 4), dtype=torch.uint8)[..., :T // 4]
    sk = (torch.rand(B, H, D, cap // G) + 0.01).to(meta)[..., :T // G]
    zk = torch.randn(B, H, D, cap // G).to(meta)[..., :T // G]
    pv = torch.randint(0, 256, (B, H, cap, D // 4), dtype=torch.uint8)[:, :, :T]
    sv = (torch.rand(B, H, cap, D // G) + 0.01).to(meta)[:, :, :T]
    zv = torch.randn(B, H, cap, D // G).to(meta)[:, :, :T]

    cache = KiviCache(num_layers=1, head_dim=D, num_kv_heads=H, device="cpu",
                      group_size=G, residual_length=max(G, 32))
    assert torch.equal(_C.dequant_keys_cpu(pk, sk, zk, out),
                       cache._dequant_keys_cpu(pk, sk, zk, dtype=out))
    assert torch.equal(_C.dequant_values_cpu(pv, sv, zv, out),
                       cache._dequant_values_cpu(pv, sv, zv, dtype=out))
