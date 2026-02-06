#!/usr/bin/env python3
"""
Test the full KIVI quantize → dequantize → attention pipeline.

This test verifies that:
1. Keys can be quantized per-channel and dequantized accurately
2. Values can be quantized per-token and dequantized accurately
3. Attention computed with dequantized KV is close to FP32 attention

SYCL kernels run on XPU; attention math runs on CPU (some XPU devices
don't support all matmul shapes via oneDNN).
"""

import torch
import intel_extension_for_pytorch as ipex
import kivi_sycl

xpu = "xpu"

# Test configuration
batch_size = 1
num_heads = 2
seq_len = 3      # query sequence length
cache_len = 32   # cached KV length (= group_size for single group)
head_dim = 64
group_size = 32

print(f"KIVI Attention Pipeline Test")
print(f"  batch={batch_size}, heads={num_heads}")
print(f"  query_len={seq_len}, cache_len={cache_len}, head_dim={head_dim}")
print(f"  group_size={group_size}")

# Create test data on CPU, copy to XPU only for SYCL kernels
torch.manual_seed(42)
query = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32)
key   = torch.randn(batch_size, num_heads, cache_len, head_dim, dtype=torch.float32)
value = torch.randn(batch_size, num_heads, cache_len, head_dim, dtype=torch.float32)

# ---- Reference: FP32 Attention (CPU) ----
print("\n1. Computing FP32 reference attention...")
scale_factor = head_dim ** -0.5
attn_weights_ref = torch.matmul(query, key.transpose(-2, -1)) * scale_factor
attn_weights_ref = torch.softmax(attn_weights_ref, dim=-1)
attn_output_ref = torch.matmul(attn_weights_ref, value)
print(f"   Reference output shape: {attn_output_ref.shape}")

# ---- KIVI: Quantize Keys on XPU ----
print("\n2. Quantizing keys (per-channel, asymmetric 2-bit)...")
k_xpu = key.to(xpu)
k_trans = k_xpu.transpose(2, 3).contiguous()  # [B, H, D, Seq]
B, H, D, S = k_trans.shape

k_packed = torch.empty(B, H, D, S // 4, dtype=torch.uint8, device=xpu)
k_scales = torch.empty(B, H, D, dtype=torch.float32, device=xpu)
k_zeros  = torch.empty(B, H, D, dtype=torch.float32, device=xpu)

kivi_sycl.quantize_keys(k_trans, k_packed, k_scales, k_zeros, group_size)
print(f"   Packed shape: {k_packed.shape}")

# Dequantize keys
k_deq_xpu = torch.empty_like(k_trans)
kivi_sycl.dequantize_keys(k_packed, k_scales, k_zeros, k_deq_xpu, group_size)
torch.xpu.synchronize()
key_deq = k_deq_xpu.transpose(2, 3).cpu()  # Back to [B, H, Seq, D] on CPU

key_err = (key - key_deq).abs().max().item()
print(f"   Key dequant max error: {key_err:.6f}")

# ---- KIVI: Quantize Values on XPU ----
print("\n3. Quantizing values (per-token, asymmetric 2-bit)...")
v_xpu = value.to(xpu)
groups_per_token = head_dim // group_size

v_packed = torch.empty(B, H, cache_len, head_dim // 4, dtype=torch.uint8, device=xpu)
v_scales = torch.empty(B, H, cache_len, groups_per_token, dtype=torch.float32, device=xpu)
v_zeros  = torch.empty(B, H, cache_len, groups_per_token, dtype=torch.float32, device=xpu)

kivi_sycl.quantize_values(v_xpu.contiguous(), v_packed, v_scales, v_zeros, head_dim, group_size)
print(f"   Packed shape: {v_packed.shape}")

# Dequantize values
v_deq_xpu = torch.empty_like(v_xpu)
kivi_sycl.dequantize_values(v_packed, v_scales, v_zeros, v_deq_xpu, head_dim, group_size)
torch.xpu.synchronize()
value_deq = v_deq_xpu.cpu()  # Back to CPU

value_err = (value - value_deq).abs().max().item()
print(f"   Value dequant max error: {value_err:.6f}")

# ---- Attention with dequantized KV (CPU) ----
print("\n4. Computing attention with dequantized KV...")
attn_weights_kivi = torch.matmul(query, key_deq.transpose(-2, -1)) * scale_factor
attn_weights_kivi = torch.softmax(attn_weights_kivi, dim=-1)
attn_output_kivi = torch.matmul(attn_weights_kivi, value_deq)
print(f"   KIVI output shape: {attn_output_kivi.shape}")

# ---- Compare ----
print("\n5. Comparing outputs...")
output_diff = (attn_output_ref - attn_output_kivi).abs()
max_diff = output_diff.max().item()
mean_diff = output_diff.mean().item()
cosine_sim = torch.nn.functional.cosine_similarity(
    attn_output_ref.view(-1).unsqueeze(0),
    attn_output_kivi.view(-1).unsqueeze(0)
).item()

print(f"   Max absolute difference:  {max_diff:.6f}")
print(f"   Mean absolute difference: {mean_diff:.6f}")
print(f"   Cosine similarity:        {cosine_sim:.6f}")

# For 2-bit quantization (4 levels), some error is expected.
# Cosine similarity > 0.85 indicates attention structure is well preserved.
if cosine_sim > 0.85:
    print(f"\n✓ PASS: Cosine similarity {cosine_sim:.4f} > 0.85")
else:
    print(f"\n✗ FAIL: Cosine similarity {cosine_sim:.4f} too low")

# Check for NaN/Inf
has_nan = torch.isnan(attn_output_kivi).any().item()
has_inf = torch.isinf(attn_output_kivi).any().item()
if not has_nan and not has_inf:
    print("✓ PASS: No NaN/Inf in KIVI output")
else:
    print(f"✗ FAIL: NaN={has_nan}, Inf={has_inf}")
