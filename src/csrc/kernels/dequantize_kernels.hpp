#pragma once

#include <cstdint>
#include <sycl/sycl.hpp>

// Templated on element type T (float / sycl::half — explicit instantiations
// in dequantize_kernels.cpp). scales/zeros/output share dtype T; arithmetic
// runs in float. Non-blocking *_submit variants — see quantize_kernels.hpp
// for the rationale.
template <typename T>
sycl::event dequantize_keys_per_channel_submit(
    const uint8_t* input,
    const T*       scales,
    const T*       zeros,
    T*             output,
    int num_channels,
    int group_size);

template <typename T>
sycl::event dequantize_values_per_token_submit(
    const uint8_t* input,
    const T*       scales,
    const T*       zeros,
    T*             output,
    int num_groups,
    int group_size);

// --------------------------------------------------------------------------
// 3. KEY DEQUANTIZATION — Per-Channel Asymmetric
// --------------------------------------------------------------------------
template <typename T>
inline void dequantize_keys_per_channel(
    const uint8_t* input,
    const T*       scales,
    const T*       zeros,
    T*             output,
    int num_channels,
    int group_size)
{
    dequantize_keys_per_channel_submit<T>(input, scales, zeros, output,
                                          num_channels, group_size)
        .wait_and_throw();
}

// --------------------------------------------------------------------------
// 4. VALUE DEQUANTIZATION — Per-Token Asymmetric
// --------------------------------------------------------------------------
template <typename T>
inline void dequantize_values_per_token(
    const uint8_t* input,
    const T*       scales,
    const T*       zeros,
    T*             output,
    int num_groups,
    int group_size)
{
    dequantize_values_per_token_submit<T>(input, scales, zeros, output,
                                          num_groups, group_size)
        .wait_and_throw();
}
