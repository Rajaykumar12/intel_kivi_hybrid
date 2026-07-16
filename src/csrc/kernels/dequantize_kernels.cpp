// ==========================================================================
// KIVI: Asymmetric 2-bit dequantization SYCL kernels
// ==========================================================================
// Dequantize: x ~= q * scale + zero
// ==========================================================================

#include "dequantize_kernels.hpp"
#include "kivi/common.hpp"

void dequantize_keys_per_channel(
    const uint8_t* __restrict__ input,
    const float*   __restrict__ scales,
    const float*   __restrict__ zeros,
    float*         __restrict__ output,
    int num_channels,
    int group_size)
{
    sycl::queue& q = kivi::get_queue();
    int bytes_per_channel = group_size / 4;
    int total_bytes       = num_channels * bytes_per_channel;
    const int local_size  = 256;
    const int global_size = ((total_bytes + local_size - 1) / local_size) * local_size;

    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(global_size, local_size),
            [=](sycl::nd_item<1> item) {
                int idx = item.get_global_id(0);
                if (idx >= total_bytes) return;

                int ch         = idx / bytes_per_channel;
                int byte_in_ch = idx % bytes_per_channel;

                float scale    = scales[ch];
                float zero     = zeros[ch];
                uint8_t packed = input[idx];

                int out_base = ch * group_size + byte_in_ch * 4;
                for (int j = 0; j < 4; ++j) {
                    uint8_t qi = (packed >> (j * 2)) & 0x3;
                    output[out_base + j] = static_cast<float>(qi) * scale + zero;
                }
            });
    }).wait_and_throw();
}

void dequantize_values_per_token(
    const uint8_t* __restrict__ input,
    const float*   __restrict__ scales,
    const float*   __restrict__ zeros,
    float*         __restrict__ output,
    int num_groups,
    int group_size)
{
    sycl::queue& q = kivi::get_queue();
    int bytes_per_group   = group_size / 4;
    int total_bytes       = num_groups * bytes_per_group;
    const int local_size  = 256;
    const int global_size = ((total_bytes + local_size - 1) / local_size) * local_size;

    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(global_size, local_size),
            [=](sycl::nd_item<1> item) {
                int byte_idx = item.get_global_id(0);
                if (byte_idx >= total_bytes) return;

                int group_idx   = byte_idx / bytes_per_group;
                int byte_in_grp = byte_idx % bytes_per_group;

                float scale    = scales[group_idx];
                float zero     = zeros[group_idx];
                uint8_t packed = input[byte_idx];

                // Output: group_idx * group_size + byte_in_grp * 4
                int out_base = group_idx * group_size + byte_in_grp * 4;
                for (int j = 0; j < 4; ++j) {
                    uint8_t qi = (packed >> (j * 2)) & 0x3;
                    output[out_base + j] = static_cast<float>(qi) * scale + zero;
                }
            });
    }).wait_and_throw();
}
