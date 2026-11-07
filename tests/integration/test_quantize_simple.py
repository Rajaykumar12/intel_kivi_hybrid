"""
Known-value check for the asymmetric 2-bit key quantization kernel.

Input [1.0, 2.0, 3.0, 4.0] replicated across 4 channels:
  min = 1.0, max = 4.0, scale = (4.0 - 1.0) / 3.0 = 1.0
  q = [0, 1, 2, 3]
  packed = 0 | (1<<2) | (2<<4) | (3<<6) = 228

Uses D=4 channels (not D=1) because single-element float32 tensors on Iris
Xe can hit a SYCL workgroup cache-flush edge case where 1 work-item out of
256 writes a single float that doesn't become visible to the host readback.
"""

import pytest

torch = pytest.importorskip("torch")
if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
    pytest.skip("XPU device not available", allow_module_level=True)

from kivi_sycl import _C as kivi_native  # noqa: E402

DEVICE = "xpu"
NUM_CHANNELS = 4
GROUP_SIZE = 4
ROW = [1.0, 2.0, 3.0, 4.0]
EXPECTED_PACKED = 0 | (1 << 2) | (2 << 4) | (3 << 6)  # 228


def _quantize_known_input():
    row = torch.tensor(ROW, dtype=torch.float32)
    x = row.unsqueeze(0).expand(NUM_CHANNELS, -1)          # [D=4, Seq=4]
    x = x.unsqueeze(0).unsqueeze(0).contiguous().to(DEVICE)  # [1, 1, 4, 4]

    packed = torch.empty(1, 1, NUM_CHANNELS, 1, dtype=torch.uint8, device=DEVICE)
    scales = torch.empty(1, 1, NUM_CHANNELS, dtype=torch.float32, device=DEVICE)
    zeros = torch.empty(1, 1, NUM_CHANNELS, dtype=torch.float32, device=DEVICE)
    kivi_native.quantize_keys(x, packed, scales, zeros, GROUP_SIZE)
    torch.xpu.synchronize()
    return x, packed, scales, zeros


def test_known_input_scale_zero_and_packing():
    _, packed, scales, zeros = _quantize_known_input()

    scales_c, zeros_c, packed_c = scales.cpu(), zeros.cpu(), packed.cpu()
    assert torch.allclose(scales_c, torch.ones_like(scales_c), atol=1e-2), scales_c
    assert torch.allclose(zeros_c, torch.ones_like(zeros_c), atol=1e-2), zeros_c
    assert packed_c.flatten().tolist() == [EXPECTED_PACKED] * NUM_CHANNELS

    byte = packed_c[0, 0, 0, 0].item()
    codes = [(byte >> (j * 2)) & 0x3 for j in range(4)]
    assert codes == [0, 1, 2, 3]


def test_known_input_roundtrip():
    x, packed, scales, zeros = _quantize_known_input()

    out = torch.empty_like(x)
    kivi_native.dequantize_keys(packed, scales, zeros, out, GROUP_SIZE)
    torch.xpu.synchronize()

    out_c = out.cpu()
    expected = torch.tensor(ROW).expand(NUM_CHANNELS, -1)
    assert (out_c[0, 0] - expected).abs().max().item() < 0.1
