#!/usr/bin/env python3
"""
KIVI 2-bit Quantization Validation Test
Tests quantization accuracy and end-to-end inference correctness
"""

import torch
import numpy as np

# Import Intel Extension for PyTorch to enable XPU support
try:
    import intel_extension_for_pytorch as ipex
    HAS_IPEX = True
except ImportError:
    print("WARNING: Intel Extension for PyTorch not found. XPU support disabled.")
    HAS_IPEX = False

import kivi_sycl

def test_quantization_accuracy():
    """Test that quantization/dequantization preserves values within acceptable error"""
    print("=" * 60)
    print("TEST 1: Quantization Accuracy")
    print("=" * 60)
    
    # Test with known values
    test_cases = [
        torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32),
        torch.tensor([-1.0, -2.0, -3.0, -4.0], dtype=torch.float32),
        torch.tensor([0.5, -0.5, 1.5, -1.5], dtype=torch.float32),
        torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32),
    ]
    
    device = "xpu"
    
    for i, input_vec in enumerate(test_cases):
        print(f"\nTest Case {i+1}: {input_vec.tolist()}")
        
        # Move to XPU
        input_xpu = input_vec.to(device)
        
        # Allocate outputs
        num_groups = 1  # 4 elements = 1 group
        packed_out = torch.zeros(num_groups, dtype=torch.uint8, device=device)
        scales_out = torch.zeros(num_groups, dtype=torch.float32, device=device)
        
        # Quantize
        kivi_sycl.quantize(input_xpu, packed_out, scales_out)
        
        # Move back to CPU for inspection
        packed_cpu = packed_out.cpu()
        scales_cpu = scales_out.cpu()
        
        print(f"  Packed byte: {packed_cpu.item():08b} ({packed_cpu.item()})")
        print(f"  Scale: {scales_cpu.item():.6f}")
        
        # Manually dequantize to verify
        packed_val = packed_cpu.item()
        scale = scales_cpu.item()
        
        dequant = []
        for j in range(4):
            bits = (packed_val >> (j * 2)) & 0x3
            dequant_val = bits * scale
            dequant.append(dequant_val)
        
        print(f"  Dequantized: {dequant}")
        
        # Calculate error
        original = input_vec.numpy()
        dequant_np = np.array(dequant)
        error = np.abs(original - dequant_np)
        max_error = np.max(error)
        
        print(f"  Max Error: {max_error:.6f}")
        
        # Check if error is acceptable (within 1 scale unit)
        if max_error <= scale * 1.5:
            print(f"  ✓ PASS")
        else:
            print(f"  ✗ FAIL (error too large)")
    
    print()

def test_attention_kernel():
    """Test the attention kernel with small inputs"""
    print("=" * 60)
    print("TEST 2: Attention Kernel")
    print("=" * 60)
    
    device = "xpu"
    
    # Small test case
    batch_size = 1
    num_heads = 2
    seq_len = 2
    cache_len = 3
    head_dim = 4
    groups_per_head = head_dim // 4
    
    print(f"\nConfiguration:")
    print(f"  batch_size={batch_size}, num_heads={num_heads}")
    print(f"  seq_len={seq_len}, cache_len={cache_len}, head_dim={head_dim}")
    
    # Create dummy cache (already quantized)
    cache_shape = (batch_size, num_heads, cache_len, groups_per_head)
    k_cache_packed = torch.randint(0, 256, cache_shape, dtype=torch.uint8, device=device)
    k_scales = torch.rand(cache_shape, dtype=torch.float32, device=device) * 0.5
    
    # Create query
    query = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32, device=device)
    
    # Flatten for kernel
    cache_flat = k_cache_packed.view(-1).contiguous()
    scales_flat = k_scales.view(-1).contiguous()
    query_flat = query.contiguous().view(-1)
    
    # Output
    attn_scores_flat = torch.zeros(
        batch_size * num_heads * seq_len * cache_len,
        dtype=torch.float32,
        device=device
    )
    
    print(f"\nTensor shapes:")
    print(f"  cache_flat: {cache_flat.shape}")
    print(f"  scales_flat: {scales_flat.shape}")
    print(f"  query_flat: {query_flat.shape}")
    print(f"  attn_scores_flat: {attn_scores_flat.shape}")
    
    try:
        # Call kernel
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
        
        # Reshape output
        attn_scores = attn_scores_flat.view(batch_size, num_heads, seq_len, cache_len)
        
        print(f"\nAttention scores shape: {attn_scores.shape}")
        print(f"Attention scores (first head, first query token):")
        print(attn_scores[0, 0, 0, :].cpu().numpy())
        
        # Check for NaN or Inf
        if torch.isnan(attn_scores).any():
            print("✗ FAIL: NaN detected in output")
        elif torch.isinf(attn_scores).any():
            print("✗ FAIL: Inf detected in output")
        else:
            print("✓ PASS: No NaN/Inf in output")
            
    except Exception as e:
        print(f"✗ FAIL: Exception occurred: {e}")
        import traceback
        traceback.print_exc()
    
    print()

def test_dimension_validation():
    """Test that dimension validation works correctly"""
    print("=" * 60)
    print("TEST 3: Dimension Validation")
    print("=" * 60)
    
    device = "xpu"
    
    # Test case 1: head_dim not divisible by 4
    print("\nTest Case 1: Invalid head_dim (not divisible by 4)")
    try:
        batch_size = 1
        num_heads = 1
        seq_len = 1
        cache_len = 1
        head_dim = 5  # Invalid!
        
        cache_flat = torch.zeros(10, dtype=torch.uint8, device=device)
        scales_flat = torch.zeros(10, dtype=torch.float32, device=device)
        query_flat = torch.zeros(head_dim, dtype=torch.float32, device=device)
        out_flat = torch.zeros(cache_len, dtype=torch.float32, device=device)
        
        kivi_sycl.batched_attention(
            cache_flat, scales_flat, query_flat, out_flat,
            batch_size, num_heads, seq_len, cache_len
        )
        print("  ✗ FAIL: Should have raised error")
    except RuntimeError as e:
        if "divisible by 4" in str(e):
            print(f"  ✓ PASS: Caught expected error: {e}")
        else:
            print(f"  ✗ FAIL: Wrong error message: {e}")
    
    # Test case 2: Negative dimensions
    print("\nTest Case 2: Negative dimensions")
    try:
        kivi_sycl.batched_attention(
            cache_flat, scales_flat, query_flat, out_flat,
            -1, 1, 1, 1  # Negative batch_size
        )
        print("  ✗ FAIL: Should have raised error")
    except RuntimeError as e:
        if "positive" in str(e):
            print(f"  ✓ PASS: Caught expected error: {e}")
        else:
            print(f"  ✗ FAIL: Wrong error message: {e}")
    
    print()

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("KIVI 2-BIT QUANTIZATION VALIDATION TESTS")
    print("=" * 60 + "\n")
    
    try:
        # Check if IPEX and XPU are available
        if not HAS_IPEX:
            print("ERROR: Intel Extension for PyTorch is not installed.")
            print("\nPlease install it with:")
            print("  pip install intel-extension-for-pytorch")
            print("\nOr follow: https://github.com/intel/intel-extension-for-pytorch")
            exit(1)
        
        if not hasattr(torch, 'xpu'):
            print("ERROR: torch.xpu not available even with IPEX installed.")
            print("This may indicate an installation issue.")
            exit(1)
            
        if not torch.xpu.is_available():
            print("ERROR: XPU device not available.")
            print("\nPossible causes:")
            print("  1. Intel GPU drivers not installed")
            print("  2. oneAPI environment not sourced (run: source /opt/intel/oneapi/setvars.sh)")
            print("  3. No Intel GPU detected")
            print("\nRun 'sycl-ls' to check available devices.")
            exit(1)
        
        print(f"✓ XPU Device: {torch.xpu.get_device_name(0)}")
        print(f"✓ IPEX Version: {ipex.__version__}\n")
        
        # Run tests
        test_quantization_accuracy()
        test_attention_kernel()
        test_dimension_validation()
        
        print("=" * 60)
        print("ALL TESTS COMPLETED")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
