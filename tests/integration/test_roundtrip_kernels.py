"""Fused round-trip kernel parity tests (XPU).

The flush path in KiviCache uses quant_dequant_roundtrip_keys/values
exclusively, but the original test suite only exercised the separate
quantize_* + dequantize_* pairs. These tests pin the fused kernels to
(1) the separate-pair result and (2) an exact torch reference, so a
divergence in the fused path can never again reach generation unseen.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("intel_extension_for_pytorch")

from kivi_sycl import _C as kivi_native

DEVICE = "xpu"


def _reference_roundtrip(inp, group_size):
    flat = inp.reshape(-1, group_size)
    mn = flat.min(dim=1, keepdim=True).values
    mx = flat.max(dim=1, keepdim=True).values
    scale = (mx - mn) / 3.0 + 1e-10
    q = torch.clamp(torch.round((flat - mn) / scale), 0.0, 3.0)
    return (q * scale + mn).reshape(inp.shape)


def test_roundtrip_keys_matches_pair_and_reference():
    torch.manual_seed(0)
    B, H, D, S = 2, 3, 64, 32  # S == group_size: one group per channel
    group_size = S
    x = torch.randn(B, H, D, S, dtype=torch.float32, device=DEVICE).contiguous()

    fused = torch.empty_like(x)
    kivi_native.quant_dequant_roundtrip_keys(x, fused, group_size)

    packed = torch.empty(B, H, D, S // 4, dtype=torch.uint8, device=DEVICE)
    scales = torch.empty(B, H, D, dtype=torch.float32, device=DEVICE)
    zeros = torch.empty(B, H, D, dtype=torch.float32, device=DEVICE)
    kivi_native.quantize_keys(x, packed, scales, zeros, group_size)
    paired = torch.empty_like(x)
    kivi_native.dequantize_keys(packed, scales, zeros, paired, group_size)
    torch.xpu.synchronize()

    assert torch.equal(fused.cpu(), paired.cpu()), \
        "fused round-trip diverges from quantize_keys + dequantize_keys"
    ref = _reference_roundtrip(x.cpu(), group_size)
    assert torch.allclose(fused.cpu(), ref, atol=1e-5), \
        "fused round-trip diverges from the torch reference"


def test_roundtrip_values_matches_pair_and_reference():
    torch.manual_seed(1)
    B, H, S, D = 2, 3, 8, 64
    group_size = 32
    x = torch.randn(B, H, S, D, dtype=torch.float32, device=DEVICE).contiguous()

    fused = torch.empty_like(x)
    kivi_native.quant_dequant_roundtrip_values(x, fused, D, group_size)

    gpt = D // group_size
    packed = torch.empty(B, H, S, D // 4, dtype=torch.uint8, device=DEVICE)
    scales = torch.empty(B, H, S, gpt, dtype=torch.float32, device=DEVICE)
    zeros = torch.empty(B, H, S, gpt, dtype=torch.float32, device=DEVICE)
    kivi_native.quantize_values(x, packed, scales, zeros, D, group_size)
    paired = torch.empty_like(x)
    kivi_native.dequantize_values(packed, scales, zeros, paired, D, group_size)
    torch.xpu.synchronize()

    assert torch.equal(fused.cpu(), paired.cpu()), \
        "fused round-trip diverges from quantize_values + dequantize_values"
    ref = _reference_roundtrip(x.cpu(), group_size)
    assert torch.allclose(fused.cpu(), ref, atol=1e-5), \
        "fused round-trip diverges from the torch reference"


def test_roundtrip_constant_input_is_stable():
    x = torch.full((1, 1, 4, 32), 5.0, dtype=torch.float32,
                   device=DEVICE).contiguous()
    out = torch.empty_like(x)
    kivi_native.quant_dequant_roundtrip_keys(x, out, 32)
    torch.xpu.synchronize()
    out_cpu = out.cpu()
    assert not torch.isnan(out_cpu).any() and not torch.isinf(out_cpu).any()
    assert torch.allclose(out_cpu, torch.full_like(out_cpu, 5.0), atol=1e-4)
