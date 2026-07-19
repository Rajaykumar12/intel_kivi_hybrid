#pragma once

#include <cstdint>
#include <sycl/sycl.hpp>

// --------------------------------------------------------------------------
// Non-blocking variants: submit the kernel and return immediately with a
// sycl::event, instead of blocking on wait_and_throw() internally. Callers
// that need the result must call `.wait_and_throw()` on the returned event
// (or otherwise synchronize) before reading the output tensors. Used by the
// async flush path (KiviCache) to overlap XPU kernel execution with CPU-side
// decode work instead of stalling on every flush. The blocking
// quantize_*_per_channel/per_token functions below are unchanged and simply
// call the *_submit variant followed by an immediate wait.
// --------------------------------------------------------------------------
sycl::event quantize_keys_per_channel_submit(
    const float* input,
    uint8_t*     output,
    float*       scales,
    float*       zeros,
    int num_channels,
    int group_size);

sycl::event quantize_values_per_token_submit(
    const float* input,
    uint8_t*     output,
    float*       scales,
    float*       zeros,
    int num_tokens,
    int head_dim,
    int group_size);

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
