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
// What's kept from that attempt: vectorized 4-wide loads/stores, which are
// a pure win at any group size since they don't change the thread count or
// add synchronization — only how many memory transactions each thread
// issues. Parallelism is back to one work-item per channel/group, matching
// the original kernel's launch shape.
//
// Everything here is templated on the element type T (float or
// sycl::half). Loads/stores happen in T; all arithmetic (min/max, scale,
// rounding) is accumulated in float regardless of T, so the fp16 path
// cannot overflow or lose the scale computation to half precision.
// ==========================================================================

#include <sycl/sycl.hpp>
#include <cstdint>

namespace kivi::detail {

constexpr int kLocalSize = 256;

// Narrow a float to the storage dtype T and widen it back, i.e. produce the
// value a caller would see after writing it to a T-typed buffer and reading
// it again. Identity for T=float.
//
// The `volatile` here is load-bearing, not defensive style. Written as a
// plain `float(T(v))`, the fpext(fptrunc(v)) pair is folded away at -O3 and
// the un-narrowed fp32 value comes back out. That silently broke the fused
// round-trip kernel for T=half: it reconstructed from full-precision
// scale/zero while the quantize->dequantize path reloads them from memory as
// fp16, so the two paths disagreed by up to 2 ULP on ~half the elements.
// An integer bit_cast round trip is *not* enough to stop it (the bitcast pair
// folds first, exposing the same fpext/fptrunc pair); forcing the narrowed
// bit pattern through a volatile private slot is. Cost is one private
// store/load per work-item, against a group_size-long loop — immaterial.
template <typename T> struct StorageBits;
template <> struct StorageBits<float>      { using type = uint32_t; };
template <> struct StorageBits<sycl::half> { using type = uint16_t; };

template <typename T>
inline float round_through_storage(float v)
{
    using U = typename StorageBits<T>::type;
    volatile U slot = sycl::bit_cast<U>(static_cast<T>(v));
    U bits = slot;
    return static_cast<float>(sycl::bit_cast<T>(bits));
}

// Vectorized (4-wide) min/max reduction over one channel's group_size
// elements, done serially by a single work-item. group_size is guaranteed
// (by kv_cache_ops.cpp validation) to be a multiple of 4.
template <typename T>
inline void reduce_min_max_vec(
    const T* __restrict__ input, int base, int group_size,
    float& min_val, float& max_val)
{
    min_val = 1e30f;
    max_val = -1e30f;
    for (int i = 0; i < group_size; i += 4) {
        sycl::vec<T, 4> v;
        v.load(0, sycl::global_ptr<const T>(input + base + i));
        float f0 = static_cast<float>(v[0]);
        float f1 = static_cast<float>(v[1]);
        float f2 = static_cast<float>(v[2]);
        float f3 = static_cast<float>(v[3]);
        float lmax = sycl::fmax(sycl::fmax(f0, f1), sycl::fmax(f2, f3));
        float lmin = sycl::fmin(sycl::fmin(f0, f1), sycl::fmin(f2, f3));
        max_val = sycl::fmax(max_val, lmax);
        min_val = sycl::fmin(min_val, lmin);
    }
}

// Round the freshly computed scale/zero through the storage dtype T before
// using them, so quantize followed by dequantize (which reads them back as
// T) is self-consistent. For T=float this is the identity.
template <typename T>
inline void store_scale_zero(
    float scale_f, float zero_f,
    T* __restrict__ scales, T* __restrict__ zeros, int idx,
    float& scale_out, float& zero_out)
{
    scales[idx] = static_cast<T>(scale_f);
    zeros[idx]  = static_cast<T>(zero_f);
    scale_out = round_through_storage<T>(scale_f);
    zero_out  = round_through_storage<T>(zero_f);
}

// Vectorized pack: reload the same group_size elements (4 at a time)
// and write one packed byte per 4 values.
template <typename T>
inline void pack_vec(
    const T* __restrict__ input, int base, int group_size,
    float min_val, float scale,
    uint8_t* __restrict__ output, int out_base)
{
    for (int i = 0; i < group_size; i += 4) {
        sycl::vec<T, 4> v;
        v.load(0, sycl::global_ptr<const T>(input + base + i));
        uint8_t packed = 0;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int qi = static_cast<int>(
                sycl::round((static_cast<float>(v[j]) - min_val) / scale));
            qi = sycl::clamp(qi, 0, 3);
            packed |= (static_cast<uint8_t>(qi) << (j * 2));
        }
        output[out_base + i / 4] = packed;
    }
}

// Vectorized round-trip: reload, round to 2-bit fidelity, write the
// reconstructed value straight back out. No packed byte is ever stored.
template <typename T>
inline void roundtrip_vec(
    const T* __restrict__ input, int base, int group_size,
    float min_val, float scale,
    T* __restrict__ output, int out_base)
{
    for (int i = 0; i < group_size; i += 4) {
        sycl::vec<T, 4> v;
        v.load(0, sycl::global_ptr<const T>(input + base + i));
        sycl::vec<T, 4> outv;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int qi = static_cast<int>(
                sycl::round((static_cast<float>(v[j]) - min_val) / scale));
            qi = sycl::clamp(qi, 0, 3);
            outv[j] = static_cast<T>(
                static_cast<float>(qi) * scale + min_val);
        }
        outv.store(0, sycl::global_ptr<T>(output + out_base + i));
    }
}

}  // namespace kivi::detail
