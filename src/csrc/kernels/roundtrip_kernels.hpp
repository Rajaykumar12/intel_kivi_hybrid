#pragma once

#include <sycl/sycl.hpp>

// ==========================================================================
// KIVI: fused quantize->dequantize round-trip kernels.
// ==========================================================================
// These entry points compute the round-trip (quantize to 2-bit fidelity,
// then immediately reconstruct) in a single kernel launch, without ever
// writing the packed uint8 buffer (or scales/zeros) to global memory. Use
// these instead of a quantize_*+dequantize_* pair whenever the packed
// representation itself is not needed by the caller.
//
// NOTE: since the packed-resident storage change (see
// docs/architecture_and_state.md), KiviCache._submit_flush no longer uses
// these — the flush persists the packed representation, so it calls
// quantize_* and keeps the outputs. The round-trip kernels remain for
// fidelity-simulation use cases and the direct kernel API.
//
// Templated on element type T (float / sycl::half — explicit
// instantiations in roundtrip_kernels.cpp). The internally computed
// scale/zero are rounded through T before reconstruction so the result is
// bit-identical to quantize_*<T> followed by dequantize_*<T>.
// --------------------------------------------------------------------------

// Non-blocking variants — see quantize_kernels.hpp for the rationale.
template <typename T>
sycl::event quant_dequant_roundtrip_keys_submit(
    const T* input,
    T*       output,
    int num_channels,
    int group_size);

template <typename T>
sycl::event quant_dequant_roundtrip_values_submit(
    const T* input,
    T*       output,
    int num_tokens,
    int head_dim,
    int group_size);

template <typename T>
inline void quant_dequant_roundtrip_keys(
    const T* input,
    T*       output,
    int num_channels,
    int group_size)
{
    quant_dequant_roundtrip_keys_submit<T>(input, output, num_channels, group_size)
        .wait_and_throw();
}

template <typename T>
inline void quant_dequant_roundtrip_values(
    const T* input,
    T*       output,
    int num_tokens,
    int head_dim,
    int group_size)
{
    quant_dequant_roundtrip_values_submit<T>(input, output, num_tokens,
                                             head_dim, group_size)
        .wait_and_throw();
}
