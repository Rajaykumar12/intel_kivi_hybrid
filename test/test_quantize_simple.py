#!/usr/bin/env python3
"""
Simple test to verify KIVI asymmetric key quantization kernel.

Tests with a known input [1.0, 2.0, 3.0, 4.0] replicated across 4 channels:
  min = 1.0, max = 4.0
  scale = (4.0 - 1.0) / 3.0 = 1.0
  q0 = round((1.0 - 1.0) / 1.0) = 0
  q1 = round((2.0 - 1.0) / 1.0) = 1
  q2 = round((3.0 - 1.0) / 1.0) = 2
  q3 = round((4.0 - 1.0) / 1.0) = 3
  packed = 0 | (1<<2) | (2<<4) | (3<<6) = 0 + 4 + 32 + 192 = 228

Note: Uses D=4 channels (not D=1) because single-element float32
tensors on Iris Xe can hit a SYCL workgroup cache-flush edge case
where 1 work-item out of 256 writes a single float that doesn't
get visible to the host readback path.
"""

import torch
import intel_extension_for_pytorch as ipex
import kivi_sycl

device = "xpu"
NUM_CHANNELS = 4  # D=4 avoids single-element scale/zero tensor edge case
group_size = 4

# Input shape: [B=1, H=1, D=4, Seq=4] — same [1,2,3,4] in every channel
row = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
input_data = row.unsqueeze(0).expand(NUM_CHANNELS, -1)  # [4, 4]
input_data = input_data.unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, 4, 4]

print(f"Input (per channel): {row.tolist()}")
print(f"Input device: {input_data.device}")
print(f"Input shape: {input_data.shape}  (D={NUM_CHANNELS} channels × Seq={group_size})")

# Allocate outputs — D=4 channels means scales/zeros are [1,1,4] = 4 floats
packed_out = torch.empty(1, 1, NUM_CHANNELS, 1, dtype=torch.uint8, device=device)
scales_out = torch.empty(1, 1, NUM_CHANNELS, dtype=torch.float32, device=device)
zeros_out  = torch.empty(1, 1, NUM_CHANNELS, dtype=torch.float32, device=device)

print(f"\nCalling kivi_sycl.quantize_keys...")
kivi_sycl.quantize_keys(input_data.contiguous(), packed_out, scales_out, zeros_out, group_size)
torch.xpu.synchronize()

# Check channel 0 (all channels should be identical)
packed_val = packed_out.cpu()[0, 0, 0, 0].item()
scale_val = scales_out.cpu()[0, 0, 0].item()
zero_val = zeros_out.cpu()[0, 0, 0].item()

print(f"\nResults (channel 0):")
print(f"  Scale:  {scale_val:.6f} (expected ~1.0)")
print(f"  Zero:   {zero_val:.6f} (expected 1.0)")
print(f"  Packed: {packed_val} = {packed_val:08b}")

# Verify all channels agree
all_scales = scales_out.cpu().squeeze().tolist()
all_zeros = zeros_out.cpu().squeeze().tolist()
all_packed = packed_out.cpu().squeeze().tolist()
channels_consistent = all(abs(s - 1.0) < 0.01 for s in all_scales) and \
                      all(abs(z - 1.0) < 0.01 for z in all_zeros)
print(f"  All {NUM_CHANNELS} channels consistent: "
      f"{'✓ YES' if channels_consistent else '✗ NO'}")

# Manually extract 2-bit values from channel 0
print(f"\nExtracted 2-bit values:")
for j in range(4):
    bits = (packed_val >> (j * 2)) & 0x3
    dequant = bits * scale_val + zero_val
    print(f"  q{j} = {bits} → dequant = {dequant:.4f} (original = {j+1}.0)")

# Expected: packed = 228 = 11100100
expected = 0 | (1 << 2) | (2 << 4) | (3 << 6)
print(f"\nExpected packed: {expected} = {expected:08b}")
print(f"Match: {'✓ PASS' if packed_val == expected else '✗ FAIL'}")

# Full round-trip
print(f"\n--- Full Round-Trip ---")
out = torch.empty_like(input_data)
kivi_sycl.dequantize_keys(packed_out, scales_out, zeros_out, out, group_size)
torch.xpu.synchronize()
out_ch0 = out.cpu()[0, 0, 0, :].tolist()  # channel 0
print(f"Dequantized: {out_ch0}")
print(f"Original:    {row.tolist()}")
max_err = max(abs(a - b) for a, b in zip(out_ch0, [1.0, 2.0, 3.0, 4.0]))
print(f"Max error:   {max_err:.6f}")
print(f"Result:      {'✓ PASS' if max_err < 0.1 else '✗ FAIL'}")
