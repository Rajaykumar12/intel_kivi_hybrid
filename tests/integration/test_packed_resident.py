"""Integration tests for the packed-resident storage path (real XPU).

Covers the three seams the CPU-only unit tests cannot:
1. fp16 kernel instantiations (quantize/dequantize/roundtrip) behave and
   stay self-consistent on device.
2. The manager's torch-native CPU dequant-on-read reconstructs exactly
   what the XPU dequantize kernels produce from the same packed data.
3. KiviCache end-to-end on XPU: packed store + async flush + serving
   match the pure-torch KIVI reference.
"""

import os
import sys

import pytest
import torch

# NOTE: never call `.float()`/`.half()`/`.to(dtype)` on an XPU tensor in this
# file. torch 2.13.0+xpu's dtype-conversion kernel fails on Iris Xe with
# `UR_RESULT_ERROR_INVALID_ARGUMENT` for *any* fp16 source or destination,
# with no kivi kernel involved (see docs/status_and_roadmap.md). Cast on the host
# instead: `t.cpu().float()`.

_SRC = os.path.join(os.path.dirname(__file__), "..", "..", "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from kivi_sycl import _C as kivi_native  # noqa: E402
from kivi_sycl.cache.manager import KiviCache, _iter_kv_layers  # noqa: E402

B, H, D = 2, 3, 64
G = 32


def _quant_keys_xpu(x_trans, dtype):
    """Run the real key-quantize kernel; x_trans is [B,H,D,L] on XPU."""
    b, h, d, L = x_trans.shape
    packed = torch.empty((b, h, d, L // 4), dtype=torch.uint8, device="xpu")
    scales = torch.empty((b, h, d, L // G), dtype=dtype, device="xpu")
    zeros = torch.empty_like(scales)
    kivi_native.quantize_keys(x_trans, packed, scales, zeros, G)
    return packed, scales, zeros


def _quant_values_xpu(x, dtype):
    """Run the real value-quantize kernel; x is [B,H,L,D] on XPU."""
    b, h, L, d = x.shape
    packed = torch.empty((b, h, L, d // 4), dtype=torch.uint8, device="xpu")
    scales = torch.empty((b, h, L, d // G), dtype=dtype, device="xpu")
    zeros = torch.empty_like(scales)
    kivi_native.quantize_values(x, packed, scales, zeros, d, G)
    return packed, scales, zeros


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_fp_dtypes_roundtrip_error_bounded(dtype):
    """quantize -> dequantize on device: max error <= scale/2 + dtype eps,
    for both element dtypes and both key/value layouts."""
    torch.manual_seed(0)
    L = 4 * G
    x_k = torch.randn(B, H, D, L, dtype=dtype, device="xpu").contiguous()
    packed, scales, zeros = _quant_keys_xpu(x_k, dtype)
    out = torch.empty_like(x_k)
    kivi_native.dequantize_keys(packed, scales, zeros, out, G)
    err = (out.cpu().float() - x_k.cpu().float()).abs().max().item()
    max_scale = scales.cpu().float().max().item()
    assert err <= max_scale / 2 + 1e-2, f"key error {err} vs scale {max_scale}"

    x_v = torch.randn(B, H, L, D, dtype=dtype, device="xpu").contiguous()
    packed, scales, zeros = _quant_values_xpu(x_v, dtype)
    out = torch.empty_like(x_v)
    kivi_native.dequantize_values(packed, scales, zeros, out, D, G)
    err = (out.cpu().float() - x_v.cpu().float()).abs().max().item()
    max_scale = scales.cpu().float().max().item()
    assert err <= max_scale / 2 + 1e-2, f"value error {err} vs scale {max_scale}"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_fused_roundtrip_matches_quant_then_dequant(dtype):
    """The fused round-trip kernel must be bit-identical to quantize
    followed by dequantize (scale/zero are rounded through the storage
    dtype in both paths)."""
    torch.manual_seed(1)
    L = 2 * G
    x = torch.randn(B, H, D, L, dtype=dtype, device="xpu").contiguous()
    packed, scales, zeros = _quant_keys_xpu(x, dtype)
    deq = torch.empty_like(x)
    kivi_native.dequantize_keys(packed, scales, zeros, deq, G)
    fused = torch.empty_like(x)
    kivi_native.quant_dequant_roundtrip_keys(x, fused, G)
    assert torch.equal(deq.cpu(), fused.cpu())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_cpu_dequant_matches_xpu_kernel(dtype):
    """The manager's torch-native unpack must reconstruct what the XPU
    dequantize kernels produce from the same packed representation.

    Not bit-exact by design, so this is a tolerance check, not
    `torch.equal`: the kernel's `qi * scale + zero` is contracted into a
    single FMA by icpx (`-ffp-contract=fast`, on by default), so it rounds
    once; the torch CPU path does a separate multiply and add and rounds
    twice. That is a ~1 ULP disagreement on roughly 4% of elements at
    fp32 — verified to be exactly the FMA/non-FMA split, not an unpack
    ordering bug. For fp16 the CPU path additionally does its arithmetic in
    fp16 while the kernel accumulates in fp32 and rounds on store, which
    widens the gap to ~1 ULP of fp16. Both are well inside the 2-bit
    quantization error the values already carry."""
    torch.manual_seed(2)
    L = 4 * G
    cache = KiviCache(num_layers=1, head_dim=D, num_kv_heads=H,
                      device="xpu", group_size=G, residual_length=128)

    # Keys
    x_k = torch.randn(B, H, D, L, dtype=dtype, device="xpu").contiguous()
    packed, scales, zeros = _quant_keys_xpu(x_k, dtype)
    kernel_out = torch.empty_like(x_k)
    kivi_native.dequantize_keys(packed, scales, zeros, kernel_out, G)
    cpu_out = cache._dequant_keys_cpu(packed.cpu(), scales.cpu(), zeros.cpu())
    # cpu_out is [B,H,L,D]; kernel output is [B,H,D,L]
    tol = 1e-6 if dtype == torch.float32 else 2e-3
    assert torch.allclose(cpu_out, kernel_out.cpu().transpose(2, 3),
                          atol=tol, rtol=tol)

    # Values
    x_v = torch.randn(B, H, L, D, dtype=dtype, device="xpu").contiguous()
    packed, scales, zeros = _quant_values_xpu(x_v, dtype)
    kernel_out = torch.empty_like(x_v)
    kivi_native.dequantize_values(packed, scales, zeros, kernel_out, D, G)
    cpu_out = cache._dequant_values_cpu(packed.cpu(), scales.cpu(), zeros.cpu())
    assert torch.allclose(cpu_out, kernel_out.cpu(), atol=tol, rtol=tol)


def _reference_roundtrip(inp, group_size):
    flat = inp.reshape(-1, group_size)
    mn = flat.min(dim=1, keepdim=True).values
    mx = flat.max(dim=1, keepdim=True).values
    scale = (mx - mn) / 3.0 + 1e-10
    q = torch.clamp(torch.round((flat - mn) / scale), 0.0, 3.0)
    return (q * scale + mn).reshape(inp.shape)


@pytest.mark.parametrize("async_flush", [True, False])
def test_manager_end_to_end_on_xpu(async_flush):
    """Full KiviCache loop with the real kernels: packed store resident,
    serving matches the torch KIVI reference, stats measured."""
    torch.manual_seed(3)
    R = 2 * G
    layers = 2
    cache = KiviCache(num_layers=layers, head_dim=D, num_kv_heads=H,
                      device="xpu", group_size=G, residual_length=R,
                      serving="rebuild", async_flush=async_flush)

    truth_k = [None] * layers
    truth_v = [None] * layers
    total = 0

    def step(n_new):
        nonlocal total
        past = []
        for i in range(layers):
            k_new = torch.randn(1, H, n_new, D)
            v_new = torch.randn(1, H, n_new, D)
            truth_k[i] = k_new if truth_k[i] is None else \
                torch.cat([truth_k[i], k_new], dim=2)
            truth_v[i] = v_new if truth_v[i] is None else \
                torch.cat([truth_v[i], v_new], dim=2)
            past.append((truth_k[i], truth_v[i]))
        total += n_new
        cache.add_tokens(tuple(past))

    step(3 * G + 5)          # bulk prefill: one flush past R
    for _ in range(G + 3):   # decode steps, crossing another flush boundary
        step(1)
    cache.harvest_pending()

    for i, (k, v) in enumerate(_iter_kv_layers(cache.get_full_cache())):
        assert k.shape[2] == total
        n_hist = cache._quant_token_count[i]
        assert n_hist > 0 and n_hist % G == 0
        # Residual exact
        assert torch.equal(k[:, :, n_hist:, :], truth_k[i][:, :, n_hist:, :])
        assert torch.equal(v[:, :, n_hist:, :], truth_v[i][:, :, n_hist:, :])
        # History matches the KIVI reference (keys per-channel in G-token
        # groups, values per-token in G-channel groups)
        for c0 in range(0, n_hist, G):
            chunk = truth_k[i][:, :, c0:c0 + G, :].transpose(2, 3).contiguous()
            expect = _reference_roundtrip(chunk, G).transpose(2, 3)
            assert torch.allclose(k[:, :, c0:c0 + G, :], expect,
                                  atol=1e-6, rtol=0)
        expect_v = _reference_roundtrip(
            truth_v[i][:, :, :n_hist, :].contiguous(), G)
        assert torch.allclose(v[:, :, :n_hist, :], expect_v, atol=1e-6, rtol=0)

    s = cache.get_stats(layer=0)
    assert s["packed_bytes"] > 0
    assert s["serving_copy_bytes"] == 0
    assert s["quantized_tokens"] + s["residual_tokens"] == s["total_tokens"]
    # 2-bit payload + metadata + FP residual must actually compress
    assert s["compression"] > 2.0, s
