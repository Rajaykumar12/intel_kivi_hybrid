#pragma once
// ==========================================================================
// KIVI: shared device-side helpers for quantize_kernels.cpp and
// roundtrip_kernels.cpp.
//
// Design note: an earlier version of this file cooperatively parallelized
// each channel's min/max reduction across a 16-wide sub-group, staging the
// group into SLM first. Measured on real Iris Xe hardware with KIVI's
// actual workload shape (group_size=32, ~768 channels per flush for
// GPT-2), that design regressed flush overhead from ~13% to ~21% versus
// the original one-thread-per-channel scalar kernel: spreading a 32-element
// reduction across 16 lanes dispatches 16x more work-items than necessary
// and pays a work-group barrier + SLM setup cost that a plain serial scan
// over 32 elements never needed. Sub-group reduction only pays off when the
// per-channel reduction itself is large; KIVI's group sizes (32-128) are
// too small for that trade on this hardware.
//
// What's kept from that attempt: vectorized float4 loads/stores, which are
// a pure win at any group size since they don't change the thread count or
// add synchronization — only how many memory transactions each thread
// issues. Parallelism is back to one work-item per channel/group, matching
// the original kernel's launch shape.
// ==========================================================================

#include <sycl/sycl.hpp>
#include <cstdint>

namespace kivi::detail {

constexpr int kLocalSize = 256;

// Vectorized (float4) min/max reduction over one channel's group_size
// elements, done serially by a single work-item. group_size is guaranteed
// (by kv_cache_ops.cpp validation) to be a multiple of 4.
inline void reduce_min_max_vec(
    const float* __restrict__ input, int base, int group_size,
    float& min_val, float& max_val)
{
    min_val = 1e30f;
    max_val = -1e30f;
    for (int i = 0; i < group_size; i += 4) {
        sycl::vec<float, 4> v;
        v.load(0, sycl::global_ptr<const float>(input + base + i));
        float lmax = sycl::fmax(sycl::fmax(v[0], v[1]), sycl::fmax(v[2], v[3]));
        float lmin = sycl::fmin(sycl::fmin(v[0], v[1]), sycl::fmin(v[2], v[3]));
        max_val = sycl::fmax(max_val, lmax);
        min_val = sycl::fmin(min_val, lmin);
    }
}

// Vectorized pack: reload the same group_size elements (float4 at a time)
// and write one packed byte per 4 values.
inline void pack_vec(
    const float* __restrict__ input, int base, int group_size,
    float min_val, float scale,
    uint8_t* __restrict__ output, int out_base)
{
    for (int i = 0; i < group_size; i += 4) {
        sycl::vec<float, 4> v;
        v.load(0, sycl::global_ptr<const float>(input + base + i));
        uint8_t packed = 0;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int qi = static_cast<int>(sycl::round((v[j] - min_val) / scale));
            qi = sycl::clamp(qi, 0, 3);
            packed |= (static_cast<uint8_t>(qi) << (j * 2));
        }
        output[out_base + i / 4] = packed;
    }
}

// Vectorized round-trip: reload, round to 2-bit fidelity, write the
// reconstructed float straight back out. No packed byte is ever stored.
inline void roundtrip_vec(
    const float* __restrict__ input, int base, int group_size,
    float min_val, float scale,
    float* __restrict__ output, int out_base)
{
    for (int i = 0; i < group_size; i += 4) {
        sycl::vec<float, 4> v;
        v.load(0, sycl::global_ptr<const float>(input + base + i));
        sycl::vec<float, 4> outv;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int qi = static_cast<int>(sycl::round((v[j] - min_val) / scale));
            qi = sycl::clamp(qi, 0, 3);
            outv[j] = static_cast<float>(qi) * scale + min_val;
        }
        outv.store(0, sycl::global_ptr<float>(output + out_base + i));
    }
}

}  // namespace kivi::detail
