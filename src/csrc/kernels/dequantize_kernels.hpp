#pragma once

#include <cstdint>
#include <sycl/sycl.hpp>

// Non-blocking variants — see quantize_kernels.hpp for the rationale.
sycl::event dequantize_keys_per_channel_submit(
    const uint8_t* input,
    const float*   scales,
    const float*   zeros,
    float*         output,
    int num_channels,
    int group_size);

sycl::event dequantize_values_per_token_submit(
    const uint8_t* input,
    const float*   scales,
    const float*   zeros,
    float*         output,
    int num_groups,
    int group_size);

// --------------------------------------------------------------------------
// 3. KEY DEQUANTIZATION — Per-Channel Asymmetric
// --------------------------------------------------------------------------
void dequantize_keys_per_channel(
    const uint8_t* input,
    const float*   scales,
    const float*   zeros,
    float*         output,
    int num_channels,
    int group_size);

// --------------------------------------------------------------------------
// 4. VALUE DEQUANTIZATION — Per-Token Asymmetric
// --------------------------------------------------------------------------
void dequantize_values_per_token(
    const uint8_t* input,
    const float*   scales,
    const float*   zeros,
    float*         output,
    int num_groups,
    int group_size);
