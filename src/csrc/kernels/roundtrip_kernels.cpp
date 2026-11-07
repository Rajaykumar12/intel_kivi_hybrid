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

namespace {

// One work-item per group; group `gid` is the contiguous run
// input[gid*G .. gid*G+G) for both keys and values (see quantize_kernels.cpp),
// and the round-trip writes it back in place, so both ops share this kernel.
template <typename T, typename M>
sycl::event roundtrip_groups_submit(
    const T* __restrict__ input,
    T*       __restrict__ output,
    int64_t num_groups,
    int group_size)
{
    sycl::queue& q = kivi::get_queue();
    const int64_t global_size =
        ((num_groups + kLocalSize - 1) / kLocalSize) * kLocalSize;

    return q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(static_cast<size_t>(global_size), kLocalSize),
            [=](sycl::nd_item<1> item) {
                const int64_t gid = static_cast<int64_t>(item.get_global_id(0));
                if (gid >= num_groups) return;

                const int64_t base = gid * group_size;
                float min_val, max_val;
                reduce_min_max_vec(input, base, group_size, min_val, max_val);
                float scale = compute_scale<M>(min_val, max_val);
                min_val = saturate_zero<M>(min_val);
                round_scale_zero<M>(scale, min_val);

                roundtrip_vec(input, base, group_size, min_val, scale, output, base);
            });
    });
}

}  // namespace

template <typename T, typename M>
sycl::event quant_dequant_roundtrip_keys_submit(
    const T* input, T* output, int num_channels, int group_size)
{
    return roundtrip_groups_submit<T, M>(input, output, num_channels, group_size);
}

template <typename T, typename M>
sycl::event quant_dequant_roundtrip_values_submit(
    const T* input, T* output, int num_tokens, int head_dim, int group_size)
{
    return roundtrip_groups_submit<T, M>(
        input, output,
        static_cast<int64_t>(num_tokens) * (head_dim / group_size), group_size);
}

// Explicit instantiations — the (element, metadata) pairs the ops layer
// dispatches: fp32 with fp32 or fp16 metadata, and fp16 with fp16 metadata.
// <half, float> is deliberately absent: widening metadata past the element
// precision buys nothing and doubles its footprint.
template sycl::event quant_dequant_roundtrip_keys_submit<float, float>(
    const float*, float*, int, int);
template sycl::event quant_dequant_roundtrip_keys_submit<float, sycl::half>(
    const float*, float*, int, int);
template sycl::event quant_dequant_roundtrip_keys_submit<sycl::half, sycl::half>(
    const sycl::half*, sycl::half*, int, int);

template sycl::event quant_dequant_roundtrip_values_submit<float, float>(
    const float*, float*, int, int, int);
template sycl::event quant_dequant_roundtrip_values_submit<float, sycl::half>(
    const float*, float*, int, int, int);
template sycl::event quant_dequant_roundtrip_values_submit<sycl::half, sycl::half>(
    const sycl::half*, sycl::half*, int, int, int);
