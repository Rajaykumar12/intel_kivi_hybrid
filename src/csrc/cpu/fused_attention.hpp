#pragma once
// KIVI fused decode attention on the CPU — see fused_attention.cpp.
#include <torch/extension.h>

// One decode step (query length 1) of attention for all heads, computed
// directly from the packed 2-bit history and the FP residual.
//   query     [B, Hq, 1, D]               (float32 or float16)
//   k_res/v_res [B, Hkv, n_res, D]        FP residual (n_res may be 0)
//   packed_k  [B, Hkv, D, T/4] uint8      scales_k/zeros_k [B, Hkv, D, T/G]
//   packed_v  [B, Hkv, T, D/4] uint8      scales_v/zeros_v [B, Hkv, T, D/G]
//   (history tensors may be arena views; empty when T == 0)
// Returns [B, 1, Hq, D] in query's dtype — the layout transformers' attention
// functions return (attn_output.transpose(1, 2)).
torch::Tensor kivi_fused_decode_attention(
    torch::Tensor query, torch::Tensor k_res, torch::Tensor v_res,
    torch::Tensor packed_k, torch::Tensor scales_k, torch::Tensor zeros_k,
    torch::Tensor packed_v, torch::Tensor scales_v, torch::Tensor zeros_v,
    double scaling, int64_t group_size);

// Whether the AVX2/FMA build of the hot loops is in use on this CPU.
bool kivi_fused_attention_uses_avx2();
