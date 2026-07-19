// ==========================================================================
// KIVI: Asymmetric 2-bit dequantization SYCL kernels
// ==========================================================================
// Dequantize: x ~= q * scale + zero
//
// One work-item per packed byte (4 output values). No reduction is needed
// here (scale/zero are already computed), so the only optimization that
// applies is vectorizing the unpack+store as a single float4 write instead
// of four scalar stores.
// ==========================================================================

#include "dequantize_kernels.hpp"
#include "kivi/common.hpp"

namespace {

inline void unpack_and_store(
    uint8_t packed, float scale, float zero,
    float* __restrict__ output, int out_base)
{
    sycl::vec<float, 4> vals;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        uint8_t qi = (packed >> (j * 2)) & 0x3;
        vals[j] = static_cast<float>(qi) * scale + zero;
    }
    vals.store(0, sycl::global_ptr<float>(output + out_base));
}

}  // namespace

sycl::event dequantize_keys_per_channel_submit(
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

    return q.submit([&](sycl::handler& h) {
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
                unpack_and_store(packed, scale, zero, output, out_base);
            });
    });
}

void dequantize_keys_per_channel(
    const uint8_t* __restrict__ input,
    const float*   __restrict__ scales,
    const float*   __restrict__ zeros,
    float*         __restrict__ output,
    int num_channels,
    int group_size)
{
    dequantize_keys_per_channel_submit(input, scales, zeros, output,
                                        num_channels, group_size)
        .wait_and_throw();
}

sycl::event dequantize_values_per_token_submit(
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

    return q.submit([&](sycl::handler& h) {
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

                int out_base = group_idx * group_size + byte_in_grp * 4;
                unpack_and_store(packed, scale, zero, output, out_base);
            });
    });
}

void dequantize_values_per_token(
    const uint8_t* __restrict__ input,
    const float*   __restrict__ scales,
    const float*   __restrict__ zeros,
    float*         __restrict__ output,
    int num_groups,
    int group_size)
{
    dequantize_values_per_token_submit(input, scales, zeros, output,
                                        num_groups, group_size)
        .wait_and_throw();
}
