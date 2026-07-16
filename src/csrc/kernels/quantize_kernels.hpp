#pragma once

#include <cstdint>

// --------------------------------------------------------------------------
// 1. KEY QUANTIZATION — Per-Channel Asymmetric 2-bit
// --------------------------------------------------------------------------
// Input layout:  [B, H, D, Seq]  (transposed keys, contiguous)
// Each "channel" = one (batch, head, dim) slice across Seq tokens.
// group_size = number of tokens per quantization group (= Seq for single group).
//
// Output packed: [B, H, D, Seq/4]   dtype=uint8
// Output scales: [B, H, D]          dtype=float32
// Output zeros:  [B, H, D]          dtype=float32
// --------------------------------------------------------------------------
void quantize_keys_per_channel(
    const float* input,
    uint8_t*     output,
    float*       scales,
    float*       zeros,
    int num_channels,
    int group_size);

// --------------------------------------------------------------------------
// 2. VALUE QUANTIZATION — Per-Token Asymmetric 2-bit
// --------------------------------------------------------------------------
// Input layout:  [B, H, Seq, D]  (values, contiguous)
// Each "group" = group_size consecutive elements within one token's D dim.
//
// Output packed: [B, H, Seq, D/4]                dtype=uint8
// Output scales: [B, H, Seq, D/group_size]       dtype=float32
// Output zeros:  [B, H, Seq, D/group_size]       dtype=float32
// --------------------------------------------------------------------------
void quantize_values_per_token(
    const float* input,
    uint8_t*     output,
    float*       scales,
    float*       zeros,
    int num_tokens,
    int head_dim,
    int group_size);
