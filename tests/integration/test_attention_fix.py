"""
Full KIVI quantize -> dequantize -> attention pipeline.

Verifies that:
1. Keys can be quantized per-channel and dequantized accurately
2. Values can be quantized per-token and dequantized accurately
3. Attention computed with dequantized KV stays close to FP32 attention

SYCL kernels run on XPU; attention math runs on CPU (some XPU devices don't
support all matmul shapes via oneDNN).
"""

import pytest

torch = pytest.importorskip("torch")
if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
    pytest.skip("XPU device not available", allow_module_level=True)

from kivi_sycl import _C as kivi_native  # noqa: E402

DEVICE = "xpu"
BATCH = 1
HEADS = 2
QUERY_LEN = 3
CACHE_LEN = 32   # = GROUP_SIZE, so keys form a single group per channel
HEAD_DIM = 64
GROUP_SIZE = 32

# 2-bit quantization has 4 levels, so some error is expected. Cosine similarity
# above 0.85 means the attention structure is preserved.
MIN_COSINE = 0.85


def _quant_dequant_keys(key):
    k_trans = key.to(DEVICE).transpose(2, 3).contiguous()   # [B, H, D, Seq]
    B, H, D, S = k_trans.shape
    packed = torch.empty(B, H, D, S // 4, dtype=torch.uint8, device=DEVICE)
    scales = torch.empty(B, H, D, S // GROUP_SIZE, dtype=torch.float32, device=DEVICE)
    zeros = torch.empty_like(scales)
    kivi_native.quantize_keys(k_trans, packed, scales, zeros, GROUP_SIZE)

    out = torch.empty_like(k_trans)
    kivi_native.dequantize_keys(packed, scales, zeros, out, GROUP_SIZE)
    torch.xpu.synchronize()
    return out.cpu().transpose(2, 3)                        # back to [B, H, Seq, D]


def _quant_dequant_values(value):
    v = value.to(DEVICE).contiguous()
    B, H, S, D = v.shape
    packed = torch.empty(B, H, S, D // 4, dtype=torch.uint8, device=DEVICE)
    scales = torch.empty(B, H, S, D // GROUP_SIZE, dtype=torch.float32, device=DEVICE)
    zeros = torch.empty_like(scales)
    kivi_native.quantize_values(v, packed, scales, zeros, D, GROUP_SIZE)

    out = torch.empty_like(v)
    kivi_native.dequantize_values(packed, scales, zeros, out, D, GROUP_SIZE)
    torch.xpu.synchronize()
    return out.cpu()


def _attention(q, k, v):
    w = torch.softmax(q @ k.transpose(-2, -1) * HEAD_DIM ** -0.5, dim=-1)
    return w @ v


def test_attention_with_dequantized_kv_matches_fp32():
    torch.manual_seed(42)
    query = torch.randn(BATCH, HEADS, QUERY_LEN, HEAD_DIM)
    key = torch.randn(BATCH, HEADS, CACHE_LEN, HEAD_DIM)
    value = torch.randn(BATCH, HEADS, CACHE_LEN, HEAD_DIM)

    key_deq = _quant_dequant_keys(key)
    value_deq = _quant_dequant_values(value)

    # Per-element error is bounded by half a quantization step (scale/2);
    # with 4 levels over a randn range that is well under 2.
    assert (key - key_deq).abs().max().item() < 2.0
    assert (value - value_deq).abs().max().item() < 2.0

    ref = _attention(query, key, value)
    kivi = _attention(query, key_deq, value_deq)

    assert torch.isfinite(kivi).all()
    cos = torch.nn.functional.cosine_similarity(
        ref.reshape(1, -1), kivi.reshape(1, -1)).item()
    assert cos > MIN_COSINE, f"cosine similarity {cos:.4f} <= {MIN_COSINE}"
