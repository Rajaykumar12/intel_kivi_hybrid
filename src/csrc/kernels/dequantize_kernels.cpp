// ==========================================================================
// KIVI: Asymmetric 2-bit dequantization SYCL kernels
// ==========================================================================
// Dequantize: x ~= q * scale + zero
//
// One work-item per packed byte (4 output values). No reduction is needed
// here (scale/zero are already computed), so the only optimization that
// applies is vectorizing the unpack+store as a single 4-wide write instead
// of four scalar stores.
//
// Templated on element type T; explicit instantiations for float and
// sycl::half at the bottom. scales/zeros are read as T and promoted to
// float for the multiply-add.
// ==========================================================================

#include "dequantize_kernels.hpp"
#include "kivi/common.hpp"

namespace {

template <typename T>
inline void unpack_and_store(
    uint8_t packed, float scale, float zero,
    T* __restrict__ output, int out_base)
{
    sycl::vec<T, 4> vals;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        uint8_t qi = (packed >> (j * 2)) & 0x3;
        vals[j] = static_cast<T>(static_cast<float>(qi) * scale + zero);
    }
    vals.store(0, sycl::global_ptr<T>(output + out_base));
}

}  // namespace

template <typename T>
sycl::event dequantize_keys_per_channel_submit(
    const uint8_t* __restrict__ input,
    const T*       __restrict__ scales,
    const T*       __restrict__ zeros,
    T*             __restrict__ output,
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

                float scale    = static_cast<float>(scales[ch]);
                float zero     = static_cast<float>(zeros[ch]);
                uint8_t packed = input[idx];

                int out_base = ch * group_size + byte_in_ch * 4;
                unpack_and_store(packed, scale, zero, output, out_base);
            });
    });
}

template <typename T>
sycl::event dequantize_values_per_token_submit(
    const uint8_t* __restrict__ input,
    const T*       __restrict__ scales,
    const T*       __restrict__ zeros,
    T*             __restrict__ output,
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

                float scale    = static_cast<float>(scales[group_idx]);
                float zero     = static_cast<float>(zeros[group_idx]);
                uint8_t packed = input[byte_idx];

                int out_base = group_idx * group_size + byte_in_grp * 4;
                unpack_and_store(packed, scale, zero, output, out_base);
            });
    });
}

// Explicit instantiations — the only element types the ops layer dispatches.
template sycl::event dequantize_keys_per_channel_submit<float>(
    const uint8_t*, const float*, const float*, float*, int, int);
template sycl::event dequantize_keys_per_channel_submit<sycl::half>(
    const uint8_t*, const sycl::half*, const sycl::half*, sycl::half*, int, int);
template sycl::event dequantize_values_per_token_submit<float>(
    const uint8_t*, const float*, const float*, float*, int, int);
template sycl::event dequantize_values_per_token_submit<sycl::half>(
    const uint8_t*, const sycl::half*, const sycl::half*, sycl::half*, int, int);
