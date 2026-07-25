// ==========================================================================
// KIVI: fused quantize->dequantize round-trip kernels.
// See roundtrip_kernels.hpp for the rationale, internal/quant_tile.hpp for
// why parallelism is one work-item per channel/group with vectorized
// 4-wide loads/stores rather than sub-group-cooperative reduction.
// ==========================================================================

#include "roundtrip_kernels.hpp"
#include "kivi/common.hpp"
#include "internal/quant_tile.hpp"

using namespace kivi::detail;

namespace {

// Round scale/zero through T so the round-trip result matches
// quantize<T> -> dequantize<T> exactly (identity for T=float). The narrowing
// must go through round_through_storage — see the comment on that helper for
// why an inline `float(T(x))` here does not survive -O3.
template <typename T>
inline void round_scale_zero(float& scale, float& zero)
{
    scale = round_through_storage<T>(scale);
    zero  = round_through_storage<T>(zero);
}

}  // namespace

template <typename T>
sycl::event quant_dequant_roundtrip_keys_submit(
    const T* __restrict__ input,
    T*       __restrict__ output,
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
                round_scale_zero<T>(scale, min_val);

                roundtrip_vec(input, base, group_size, min_val, scale, output, base);
            });
    });
}

template <typename T>
sycl::event quant_dequant_roundtrip_values_submit(
    const T* __restrict__ input,
    T*       __restrict__ output,
    int num_tokens,
    int head_dim,
    int group_size)
{
    sycl::queue& q = kivi::get_queue();
    int groups_per_token = head_dim / group_size;
    int total_groups     = num_tokens * groups_per_token;
    const int global_size = ((total_groups + kLocalSize - 1) / kLocalSize) * kLocalSize;

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
                round_scale_zero<T>(scale, min_val);

                roundtrip_vec(input, base, group_size, min_val, scale, output, base);
            });
    });
}

// Explicit instantiations — the only element types the ops layer dispatches.
template sycl::event quant_dequant_roundtrip_keys_submit<float>(
    const float*, float*, int, int);
template sycl::event quant_dequant_roundtrip_keys_submit<sycl::half>(
    const sycl::half*, sycl::half*, int, int);
template sycl::event quant_dequant_roundtrip_values_submit<float>(
    const float*, float*, int, int, int);
template sycl::event quant_dequant_roundtrip_values_submit<sycl::half>(
    const sycl::half*, sycl::half*, int, int, int);
