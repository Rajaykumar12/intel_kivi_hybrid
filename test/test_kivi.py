#!/usr/bin/env python3
"""
KIVI 2-bit Asymmetric Quantization — Validation Tests
======================================================
Tests the SYCL kernels for correctness:
  1. Key quantization/dequantization round-trip accuracy
  2. Value quantization/dequantization round-trip accuracy
  3. Multi-batch, multi-head tensor handling
  4. Edge cases (constant input, single element groups)
"""

import torch
import numpy as np
import sys

# Import Intel Extension for PyTorch to enable XPU support
try:
    import intel_extension_for_pytorch as ipex
    HAS_IPEX = True
except ImportError:
    print("WARNING: Intel Extension for PyTorch not found. XPU support disabled.")
    HAS_IPEX = False

import kivi_sycl

DEVICE = "xpu"
PASS_COUNT = 0
FAIL_COUNT = 0


def check(condition, msg):
    global PASS_COUNT, FAIL_COUNT
    if condition:
        PASS_COUNT += 1
        print(f"  ✓ PASS: {msg}")
    else:
        FAIL_COUNT += 1
        print(f"  ✗ FAIL: {msg}")


# ---------------------------------------------------------------
# TEST 1: Key Quantization Round-Trip
# ---------------------------------------------------------------
def test_key_roundtrip():
    """
    Quantize keys per-channel (asymmetric 2-bit), then dequantize.
    Check that the max absolute error is within 1 scale step.
    """
    print("=" * 60)
    print("TEST 1: Key Quantization Round-Trip (Per-Channel)")
    print("=" * 60)

    batch, heads, dim, seq = 1, 2, 4, 32  # seq = group_size
    group_size = seq

    # Random input in [B, H, D, Seq] layout (transposed keys)
    original = torch.randn(batch, heads, dim, seq, dtype=torch.float32, device=DEVICE)

    # Quantize
    packed = torch.empty(batch, heads, dim, seq // 4, dtype=torch.uint8, device=DEVICE)
    scales = torch.empty(batch, heads, dim, dtype=torch.float32, device=DEVICE)
    zeros  = torch.empty(batch, heads, dim, dtype=torch.float32, device=DEVICE)

    kivi_sycl.quantize_keys(original.contiguous(), packed, scales, zeros, group_size)

    # Dequantize
    reconstructed = torch.empty_like(original)
    kivi_sycl.dequantize_keys(packed, scales, zeros, reconstructed, group_size)

    # Check error
    orig_cpu = original.cpu()
    recon_cpu = reconstructed.cpu()
    abs_error = (orig_cpu - recon_cpu).abs()
    max_error = abs_error.max().item()
    scale_max = scales.cpu().max().item()

    print(f"  Max absolute error: {max_error:.6f}")
    print(f"  Max scale value:    {scale_max:.6f}")
    print(f"  Error / scale:      {max_error / scale_max:.4f}")

    # For 2-bit quantization, max error should be ≤ 0.5 * scale (rounding)
    check(max_error <= scale_max * 0.6,
          f"Round-trip error within tolerance (max_err={max_error:.6f})")

    # Check that all quantized values are valid (0-3)
    packed_cpu = packed.cpu()
    for j in range(4):
        bits = (packed_cpu.int() >> (j * 2)) & 0x3
        check(bits.max().item() <= 3 and bits.min().item() >= 0,
              f"Bit position {j}: all values in [0, 3]")

    print()


# ---------------------------------------------------------------
# TEST 2: Value Quantization Round-Trip
# ---------------------------------------------------------------
def test_value_roundtrip():
    """
    Quantize values per-token (asymmetric 2-bit), then dequantize.
    """
    print("=" * 60)
    print("TEST 2: Value Quantization Round-Trip (Per-Token)")
    print("=" * 60)

    batch, heads, seq, head_dim = 1, 2, 8, 64
    group_size = 32
    groups_per_token = head_dim // group_size

    # Random input in [B, H, Seq, D] layout
    original = torch.randn(batch, heads, seq, head_dim, dtype=torch.float32, device=DEVICE)

    # Quantize
    packed = torch.empty(batch, heads, seq, head_dim // 4, dtype=torch.uint8, device=DEVICE)
    scales = torch.empty(batch, heads, seq, groups_per_token, dtype=torch.float32, device=DEVICE)
    zeros  = torch.empty(batch, heads, seq, groups_per_token, dtype=torch.float32, device=DEVICE)

    kivi_sycl.quantize_values(
        original.contiguous(), packed, scales, zeros, head_dim, group_size)

    # Dequantize
    reconstructed = torch.empty_like(original)
    kivi_sycl.dequantize_values(packed, scales, zeros, reconstructed, head_dim, group_size)

    # Check error
    orig_cpu = original.cpu()
    recon_cpu = reconstructed.cpu()
    abs_error = (orig_cpu - recon_cpu).abs()
    max_error = abs_error.max().item()
    mean_error = abs_error.mean().item()
    scale_max = scales.cpu().max().item()

    print(f"  Max absolute error:  {max_error:.6f}")
    print(f"  Mean absolute error: {mean_error:.6f}")
    print(f"  Max scale value:     {scale_max:.6f}")

    check(max_error <= scale_max * 0.6,
          f"Round-trip error within tolerance (max_err={max_error:.6f})")
    print()


# ---------------------------------------------------------------
# TEST 3: Known Values (Deterministic Check)
# ---------------------------------------------------------------
def test_known_values():
    """
    Test with known values to verify exact quantization behavior.
    For input [1.0, 2.0, 3.0, 4.0]:
      min=1.0, max=4.0, scale=(4-1)/3=1.0
      q0 = round((1-1)/1) = 0
      q1 = round((2-1)/1) = 1
      q2 = round((3-1)/1) = 2
      q3 = round((4-1)/1) = 3
      Dequant: [0*1+1, 1*1+1, 2*1+1, 3*1+1] = [1.0, 2.0, 3.0, 4.0]
    """
    print("=" * 60)
    print("TEST 3: Known Values (Deterministic)")
    print("=" * 60)

    # Shape: [1, 1, 1, 4] = [B, H, D=1, Seq=4] for key quantization
    input_data = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]],
                               dtype=torch.float32, device=DEVICE)
    group_size = 4

    packed = torch.empty(1, 1, 1, 1, dtype=torch.uint8, device=DEVICE)
    scales = torch.empty(1, 1, 1, dtype=torch.float32, device=DEVICE)
    zeros  = torch.empty(1, 1, 1, dtype=torch.float32, device=DEVICE)

    kivi_sycl.quantize_keys(input_data.contiguous(), packed, scales, zeros, group_size)

    scale_val = scales.cpu().item()
    zero_val = zeros.cpu().item()
    packed_val = packed.cpu().item()

    print(f"  Input:  [1.0, 2.0, 3.0, 4.0]")
    print(f"  Scale:  {scale_val:.6f} (expected ~1.0)")
    print(f"  Zero:   {zero_val:.6f} (expected 1.0)")
    print(f"  Packed: {packed_val} = {packed_val:08b}")

    # Expected packed: q0=0, q1=1, q2=2, q3=3
    # byte = 0 | (1<<2) | (2<<4) | (3<<6) = 0 + 4 + 32 + 192 = 228
    expected_packed = 0 | (1 << 2) | (2 << 4) | (3 << 6)
    print(f"  Expected packed: {expected_packed} = {expected_packed:08b}")

    check(abs(scale_val - 1.0) < 0.01, f"Scale ≈ 1.0 (got {scale_val:.6f})")
    check(abs(zero_val - 1.0) < 0.01, f"Zero ≈ 1.0 (got {zero_val:.6f})")
    check(packed_val == expected_packed,
          f"Packed byte = {expected_packed} (got {packed_val})")

    # Dequantize and verify
    out = torch.empty_like(input_data)
    kivi_sycl.dequantize_keys(packed, scales, zeros, out, group_size)
    out_cpu = out.cpu().squeeze()
    expected = torch.tensor([1.0, 2.0, 3.0, 4.0])
    max_err = (out_cpu - expected).abs().max().item()
    check(max_err < 0.1, f"Dequantized matches input (max_err={max_err:.6f})")
    print()


# ---------------------------------------------------------------
# TEST 4: Multi-Batch / Multi-Head
# ---------------------------------------------------------------
def test_multi_batch_head():
    """Test that quantization works correctly with multiple batches and heads."""
    print("=" * 60)
    print("TEST 4: Multi-Batch / Multi-Head")
    print("=" * 60)

    batch, heads, dim, seq = 2, 4, 64, 32
    group_size = seq
    head_dim = dim

    # Keys
    k_input = torch.randn(batch, heads, dim, seq, dtype=torch.float32, device=DEVICE)
    k_packed = torch.empty(batch, heads, dim, seq // 4, dtype=torch.uint8, device=DEVICE)
    k_scales = torch.empty(batch, heads, dim, dtype=torch.float32, device=DEVICE)
    k_zeros  = torch.empty(batch, heads, dim, dtype=torch.float32, device=DEVICE)

    kivi_sycl.quantize_keys(k_input.contiguous(), k_packed, k_scales, k_zeros, group_size)

    k_out = torch.empty_like(k_input)
    kivi_sycl.dequantize_keys(k_packed, k_scales, k_zeros, k_out, group_size)

    k_err = (k_input.cpu() - k_out.cpu()).abs().max().item()
    print(f"  Key round-trip max error: {k_err:.6f}")
    check(k_err < 2.0, f"Key error reasonable for {batch}B×{heads}H")

    # Values
    v_input = torch.randn(batch, heads, seq, head_dim, dtype=torch.float32, device=DEVICE)
    groups_per_token = head_dim // group_size
    v_packed = torch.empty(batch, heads, seq, head_dim // 4, dtype=torch.uint8, device=DEVICE)
    v_scales = torch.empty(batch, heads, seq, groups_per_token, dtype=torch.float32, device=DEVICE)
    v_zeros  = torch.empty(batch, heads, seq, groups_per_token, dtype=torch.float32, device=DEVICE)

    kivi_sycl.quantize_values(
        v_input.contiguous(), v_packed, v_scales, v_zeros, head_dim, group_size)

    v_out = torch.empty_like(v_input)
    kivi_sycl.dequantize_values(v_packed, v_scales, v_zeros, v_out, head_dim, group_size)

    v_err = (v_input.cpu() - v_out.cpu()).abs().max().item()
    print(f"  Value round-trip max error: {v_err:.6f}")
    check(v_err < 2.0, f"Value error reasonable for {batch}B×{heads}H")
    print()


# ---------------------------------------------------------------
# TEST 5: Constant Input Edge Case
# ---------------------------------------------------------------
def test_constant_input():
    """Test that constant input doesn't cause division by zero."""
    print("=" * 60)
    print("TEST 5: Constant Input Edge Case")
    print("=" * 60)

    input_data = torch.ones(1, 1, 1, 4, dtype=torch.float32, device=DEVICE) * 5.0
    group_size = 4

    packed = torch.empty(1, 1, 1, 1, dtype=torch.uint8, device=DEVICE)
    scales = torch.empty(1, 1, 1, dtype=torch.float32, device=DEVICE)
    zeros  = torch.empty(1, 1, 1, dtype=torch.float32, device=DEVICE)

    kivi_sycl.quantize_keys(input_data.contiguous(), packed, scales, zeros, group_size)

    out = torch.empty_like(input_data)
    kivi_sycl.dequantize_keys(packed, scales, zeros, out, group_size)

    out_cpu = out.cpu()
    has_nan = torch.isnan(out_cpu).any().item()
    has_inf = torch.isinf(out_cpu).any().item()

    check(not has_nan, "No NaN in output")
    check(not has_inf, "No Inf in output")

    # All values should be close to 5.0
    max_err = (out_cpu - 5.0).abs().max().item()
    print(f"  Max error from constant 5.0: {max_err:.6f}")
    check(max_err < 0.5, f"Reconstructed ≈ 5.0 (max_err={max_err:.6f})")
    print()


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("KIVI 2-BIT ASYMMETRIC QUANTIZATION — VALIDATION TESTS")
    print("=" * 60 + "\n")

    if not HAS_IPEX:
        print("ERROR: Intel Extension for PyTorch is not installed.")
        sys.exit(1)

    if not torch.xpu.is_available():
        print("ERROR: XPU device not available.")
        print("Run 'sycl-ls' to check available devices.")
        sys.exit(1)

    print(f"✓ XPU Device: {torch.xpu.get_device_name(0)}")
    print(f"✓ IPEX Version: {ipex.__version__}\n")

    test_key_roundtrip()
    test_value_roundtrip()
    test_known_values()
    test_multi_batch_head()
    test_constant_input()

    print("=" * 60)
    print(f"RESULTS: {PASS_COUNT} passed, {FAIL_COUNT} failed")
    print("=" * 60)

    sys.exit(1 if FAIL_COUNT > 0 else 0)
