#!/usr/bin/env python3
"""Test the attention kernel fix"""

import torch
import intel_extension_for_pytorch as ipex
import kivi_sycl

device = "xpu"

# Test configuration
batch_size = 1
num_heads = 2
seq_len = 3  # 3 query tokens
cache_len = 5  # 5 cached tokens
head_dim = 4
groups_per_head = head_dim // 4

print(f"Testing attention kernel:")
print(f"  batch={batch_size}, heads={num_heads}")
print(f"  seq_len={seq_len}, cache_len={cache_len}, head_dim={head_dim}")
print(f"  Expected output shape: [{batch_size}, {num_heads}, {seq_len}, {cache_len}]")
print(f"  Expected total elements: {batch_size * num_heads * seq_len * cache_len}")

# Create test data
cache_shape = (batch_size, num_heads, cache_len, groups_per_head)
k_cache_packed = torch.randint(0, 4, cache_shape, dtype=torch.uint8, device=device)
k_scales = torch.ones(cache_shape, dtype=torch.float32, device=device) * 0.5

query = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32, device=device)

# Flatten
cache_flat = k_cache_packed.view(-1).contiguous()
scales_flat = k_scales.view(-1).contiguous()
query_flat = query.contiguous().view(-1)

# Output
attn_scores_flat = torch.zeros(
    batch_size * num_heads * seq_len * cache_len,
    dtype=torch.float32,
    device=device
)

print(f"\nInput tensor sizes:")
print(f"  cache_flat: {cache_flat.shape[0]} elements")
print(f"  scales_flat: {scales_flat.shape[0]} elements")
print(f"  query_flat: {query_flat.shape[0]} elements")
print(f"  attn_scores_flat: {attn_scores_flat.shape[0]} elements")

# Call kernel
try:
    kivi_sycl.batched_attention(
        cache_flat,
        scales_flat,
        query_flat,
        attn_scores_flat,
        batch_size,
        num_heads,
        seq_len,
        cache_len
    )
    
    # Reshape
    attn_scores = attn_scores_flat.view(batch_size, num_heads, seq_len, cache_len)
    
    print(f"\n✓ Kernel executed successfully!")
    print(f"Output shape: {attn_scores.shape}")
    print(f"\nAttention scores (head 0):")
    print(attn_scores[0, 0].cpu().numpy())
    
    # Check that we have different scores for different cache tokens
    # (not all the same, which would indicate the bug)
    for h in range(num_heads):
        for q in range(seq_len):
            scores = attn_scores[0, h, q, :].cpu()
            unique_count = len(torch.unique(scores))
            print(f"  Head {h}, Query {q}: {unique_count} unique scores out of {cache_len}")
            if unique_count < 2:
                print(f"    ⚠️  WARNING: All scores are the same! Bug may still exist.")
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
