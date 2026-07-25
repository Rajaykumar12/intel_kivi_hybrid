// ==========================================================================
// KIVI: Asymmetric 2-bit quantization SYCL kernels
// ==========================================================================
// Quantization: Asymmetric 2-bit
//   q = clamp(round((x - min) / scale), 0, 3)
//   scale = (max - min) / 3
//
// Packing: 4 x 2-bit values per uint8
//   byte = q0 | (q1 << 2) | (q2 << 4) | (q3 << 6)
//
// One work-item per channel/group (see internal/quant_tile.hpp for why
// sub-group-cooperative parallelization was tried and reverted — it
// regressed on KIVI's actual small group_size workload). Each thread
// vectorizes its own min/max reduction and packing pass as 4-wide
// loads/stores instead of scalar ones.
//
// Templated on element type T; explicit instantiations for float and
// sycl::half are at the bottom of this file. Arithmetic accumulates in
// float for both.
// ==========================================================================

#include "quantize_kernels.hpp"
#include "kivi/common.hpp"
#include "internal/quant_tile.hpp"

using namespace kivi::detail;

template <typename T>
sycl::event quantize_keys_per_channel_submit(
    const T* __restrict__ input,
    uint8_t* __restrict__ output,
    T*       __restrict__ scales,
    T*       __restrict__ zeros,
    int num_channels,
    int group_size)
{
    sycl::queue& q = kivi::get_queue();
    const int global_size = ((num_channels + kLocalSize - 1) / kLocalSize) * kLocalSize;

    return q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(global_size, kLocalSize),
            [=](sycl::nd_item<1> item) {
                int ch = item.get_global_id(0);
                if (ch >= num_channels) return;

                int base = ch * group_size;
                float min_val, max_val;
                reduce_min_max_vec(input, base, group_size, min_val, max_val);

                float scale = (max_val - min_val) / 3.0f + 1e-10f;
                float scale_r, zero_r;
                store_scale_zero(scale, min_val, scales, zeros, ch,
                                 scale_r, zero_r);

                int out_base = ch * (group_size / 4);
                pack_vec(input, base, group_size, zero_r, scale_r, output, out_base);
            });
    });
}

template <typename T>
sycl::event quantize_values_per_token_submit(
    const T* __restrict__ input,
    uint8_t* __restrict__ output,
    T*       __restrict__ scales,
    T*       __restrict__ zeros,
    int num_tokens,
    int head_dim,
    int group_size)
{
    sycl::queue& q = kivi::get_queue();
    int groups_per_token = head_dim / group_size;
    int total_groups     = num_tokens * groups_per_token;
    const int global_size = ((total_groups + kLocalSize - 1) / kLocalSize) * kLocalSize;
    int packed_per_token  = head_dim / 4;

    return q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(global_size, kLocalSize),
            [=](sycl::nd_item<1> item) {
                int gid = item.get_global_id(0);
                if (gid >= total_groups) return;

                int token_idx    = gid / groups_per_token;
                int group_in_tok = gid % groups_per_token;
                int base = token_idx * head_dim + group_in_tok * group_size;

                float min_val, max_val;
                reduce_min_max_vec(input, base, group_size, min_val, max_val);

                float scale = (max_val - min_val) / 3.0f + 1e-10f;
                float scale_r, zero_r;
                store_scale_zero(scale, min_val, scales, zeros, gid,
                                 scale_r, zero_r);

                int out_base = token_idx * packed_per_token + group_in_tok * (group_size / 4);
                pack_vec(input, base, group_size, zero_r, scale_r, output, out_base);
            });
    });
}

// Explicit instantiations — the only element types the ops layer dispatches.
template sycl::event quantize_keys_per_channel_submit<float>(
    const float*, uint8_t*, float*, float*, int, int);
template sycl::event quantize_keys_per_channel_submit<sycl::half>(
    const sycl::half*, uint8_t*, sycl::half*, sycl::half*, int, int);
template sycl::event quantize_values_per_token_submit<float>(
    const float*, uint8_t*, float*, float*, int, int, int);
template sycl::event quantize_values_per_token_submit<sycl::half>(
    const sycl::half*, uint8_t*, sycl::half*, sycl::half*, int, int, int);
