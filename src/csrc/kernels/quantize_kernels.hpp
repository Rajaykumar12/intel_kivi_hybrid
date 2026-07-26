#pragma once

#include <cstdint>
#include <sycl/sycl.hpp>

// --------------------------------------------------------------------------
// All kernels are templated on the element type T. Supported instantiations
// (provided explicitly in quantize_kernels.cpp): float and sycl::half.
// Loads/stores happen in T; min/max reduction, scale computation, and
// rounding always accumulate in float (see internal/quant_tile.hpp).
// scales/zeros are stored in T as well — the freshly computed scale/zero
// are rounded through T before being used for packing, so a subsequent
// dequantize that reads them back as T reconstructs the exact same values.
//
// Non-blocking variants: submit the kernel and return immediately with a
// sycl::event, instead of blocking on wait_and_throw() internally. Callers
// that need the result must call `.wait_and_throw()` on the returned event
// (or otherwise synchronize) before reading the output tensors. Used by the
// async flush path (KiviCache) to overlap XPU kernel execution with CPU-side
// decode work instead of stalling on every flush. The blocking
// quantize_*_per_channel/per_token functions below are unchanged and simply
// call the *_submit variant followed by an immediate wait.
// --------------------------------------------------------------------------
template <typename T>
sycl::event quantize_keys_per_channel_submit(
    const T*  input,
    uint8_t*  output,
    T*        scales,
    T*        zeros,
    int num_channels,
    int group_size);

template <typename T>
sycl::event quantize_values_per_token_submit(
    const T*  input,
    uint8_t*  output,
    T*        scales,
    T*        zeros,
    int num_tokens,
    int head_dim,
    int group_size);

// --------------------------------------------------------------------------
// 1. KEY QUANTIZATION — Per-Channel Asymmetric 2-bit
// --------------------------------------------------------------------------
// Input layout:  [B, H, D, Seq]  (transposed keys, contiguous)
// Each "channel" = one (batch, head, dim) slice across Seq tokens.
// group_size = number of tokens per quantization group. Seq may be any
// multiple of group_size: the kernel is layout-flat, so a chunk of
// k*group_size tokens per channel simply becomes k channel-groups with one
// scale/zero pair each (num_channels = numel / group_size).
//
// Output packed: [B, H, D, Seq/4]        dtype=uint8
// Output scales: [B, H, D, Seq/G]        dtype=T
// Output zeros:  [B, H, D, Seq/G]        dtype=T
// --------------------------------------------------------------------------
template <typename T>
inline void quantize_keys_per_channel(
    const T*  input,
    uint8_t*  output,
    T*        scales,
    T*        zeros,
    int num_channels,
    int group_size)
{
    quantize_keys_per_channel_submit<T>(input, output, scales, zeros,
                                        num_channels, group_size)
        .wait_and_throw();
}

// --------------------------------------------------------------------------
// 2. VALUE QUANTIZATION — Per-Token Asymmetric 2-bit
// --------------------------------------------------------------------------
// Input layout:  [B, H, Seq, D]  (values, contiguous)
// Each "group" = group_size consecutive elements within one token's D dim.
//
// Output packed: [B, H, Seq, D/4]                dtype=uint8
// Output scales: [B, H, Seq, D/group_size]       dtype=T
// Output zeros:  [B, H, Seq, D/group_size]       dtype=T
// --------------------------------------------------------------------------
template <typename T>
inline void quantize_values_per_token(
    const T*  input,
    uint8_t*  output,
    T*        scales,
    T*        zeros,
    int num_tokens,
    int head_dim,
    int group_size)
{
    quantize_values_per_token_submit<T>(input, output, scales, zeros,
                                        num_tokens, head_dim, group_size)
        .wait_and_throw();
}
