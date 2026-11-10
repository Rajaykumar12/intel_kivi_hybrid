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
// Templated on element type T and metadata type M; explicit instantiations
// are at the bottom of this file. Arithmetic accumulates in float.
// ==========================================================================

#include "quantize_kernels.hpp"
#include "kivi/common.hpp"
#include "internal/quant_tile.hpp"

using namespace kivi::detail;

namespace {

// One work-item per quantization group: reduce, store scale/zero, pack.
// Keys and values differ only in how the caller laid out the input (keys
// transposed to [.., D, L] so groups run along tokens; values [.., L, D] so
// groups run along channels) — either way group `gid` is the contiguous run
// input[gid*G .. gid*G+G), so both ops share this kernel. Output placement
// follows `layout` (see kivi/layout.hpp).
template <typename T, typename M>
sycl::event quantize_groups_submit(
    const T* __restrict__ input,
    uint8_t* __restrict__ output,
    M*       __restrict__ scales,
    M*       __restrict__ zeros,
    int64_t num_groups,
    int group_size,
    kivi::KiviLayout layout)
{
    sycl::queue& q = kivi::get_queue();
    const int64_t global_size =
        ((num_groups + kLocalSize - 1) / kLocalSize) * kLocalSize;
    const int64_t gpr = layout.groups_per_row;
    const int64_t out_stride = layout.out_row_stride;
    const int64_t meta_stride = layout.meta_row_stride;
    const int bytes_per_group = group_size / 4;

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

                const int64_t row = gid / gpr;
                const int64_t g   = gid - row * gpr;
                float scale_r, zero_r;
                store_scale_zero(scale, min_val, scales, zeros,
                                 row * meta_stride + g, scale_r, zero_r);

                pack_vec(input, base, group_size, zero_r, scale_r, output,
                         row * out_stride + g * bytes_per_group);
            });
    });
}

}  // namespace

template <typename T, typename M>
sycl::event quantize_keys_per_channel_submit(
    const T* input, uint8_t* output, M* scales, M* zeros,
    int num_channels, int group_size, kivi::KiviLayout layout)
{
    return quantize_groups_submit<T, M>(input, output, scales, zeros,
                                        num_channels, group_size, layout);
}

template <typename T, typename M>
sycl::event quantize_values_per_token_submit(
    const T* input, uint8_t* output, M* scales, M* zeros,
    int num_tokens, int head_dim, int group_size, kivi::KiviLayout layout)
{
    const int64_t num_groups =
        static_cast<int64_t>(num_tokens) * (head_dim / group_size);
    return quantize_groups_submit<T, M>(input, output, scales, zeros,
                                        num_groups, group_size, layout);
}

// Explicit instantiations — the (element, metadata) pairs the ops layer
// dispatches: fp32 with fp32 or fp16 metadata, and fp16 with fp16 metadata.
// <half, float> is deliberately absent: widening metadata past the element
// precision buys nothing and doubles its footprint.
template sycl::event quantize_keys_per_channel_submit<float, float>(
    const float*, uint8_t*, float*, float*, int, int, kivi::KiviLayout);
template sycl::event quantize_keys_per_channel_submit<float, sycl::half>(
    const float*, uint8_t*, sycl::half*, sycl::half*, int, int, kivi::KiviLayout);
template sycl::event quantize_keys_per_channel_submit<sycl::half, sycl::half>(
    const sycl::half*, uint8_t*, sycl::half*, sycl::half*, int, int, kivi::KiviLayout);

template sycl::event quantize_values_per_token_submit<float, float>(
    const float*, uint8_t*, float*, float*, int, int, int, kivi::KiviLayout);
template sycl::event quantize_values_per_token_submit<float, sycl::half>(
    const float*, uint8_t*, sycl::half*, sycl::half*, int, int, int, kivi::KiviLayout);
template sycl::event quantize_values_per_token_submit<sycl::half, sycl::half>(
    const sycl::half*, uint8_t*, sycl::half*, sycl::half*, int, int, int, kivi::KiviLayout);
