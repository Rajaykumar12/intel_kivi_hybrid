#!/usr/bin/env python3
"""Simple test to debug quantization kernel"""

import torch
import intel_extension_for_pytorch as ipex
import kivi_sycl

device = "xpu"

# Test input: [1.0, 2.0, 3.0, 4.0]
input_vec = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32, device=device)

print(f"Input: {input_vec}")
print(f"Input device: {input_vec.device}")
print(f"Input is_contiguous: {input_vec.is_contiguous()}")

# Allocate outputs
num_groups = 1
packed_out = torch.zeros(num_groups, dtype=torch.uint8, device=device)
scales_out = torch.zeros(num_groups, dtype=torch.float32, device=device)

print(f"\nBefore quantize:")
print(f"  packed_out: {packed_out}")
print(f"  scales_out: {scales_out}")
print(f"  packed_out is_contiguous: {packed_out.is_contiguous()}")
print(f"  scales_out is_contiguous: {scales_out.is_contiguous()}")

# Call kernel
print(f"\nCalling kivi_sycl.quantize...")
kivi_sycl.quantize(input_vec, packed_out, scales_out)

print(f"\nAfter quantize:")
print(f"  packed_out: {packed_out} ({packed_out.item():08b})")
print(f"  scales_out: {scales_out}")

# Expected:
# max_val = 4.0
# scale = 4.0 / 3.0 = 1.333...
# q0 = int(1.0 / 1.333 + 1.5) = int(0.75 + 1.5) = int(2.25) = 2
# q1 = int(2.0 / 1.333 + 1.5) = int(1.5 + 1.5) = int(3.0) = 3
# q2 = int(3.0 / 1.333 + 1.5) = int(2.25 + 1.5) = int(3.75) = 3
# q3 = int(4.0 / 1.333 + 1.5) = int(3.0 + 1.5) = int(4.5) = 4 -> clamped to 3
# packed = (3<<6) | (3<<4) | (3<<2) | 2 = 192 + 48 + 12 + 2 = 254

print(f"\nExpected:")
print(f"  scale: ~1.333")
print(f"  packed: 254 (11111110)")
