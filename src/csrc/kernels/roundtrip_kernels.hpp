#pragma once

#include <sycl/sycl.hpp>

// ==========================================================================
// KIVI: fused quantize->dequantize round-trip kernels.
// ==========================================================================
// The CPU-residual flush path (KiviCache._flush_group) always calls
// quantize_* immediately followed by dequantize_* on the same data, and
// never reads the intermediate packed/scales/zeros buffers again — see
// docs/architecture_and_state.md's "critical invariant" and
// manager.py::_flush_group. These entry points compute the round-trip
// (quantize to 2-bit fidelity, then immediately reconstruct as float) in a
// single kernel launch, without ever writing the packed uint8 buffer (or
// scales/zeros) to global memory. Use these instead of a
// quantize_*+dequantize_* pair whenever the packed representation itself
// is not needed by the caller.
//
// The original quantize_*/dequantize_* entry points in quantize_kernels.hpp
// / dequantize_kernels.hpp remain unchanged and are still required for any
// path that persists the 2-bit representation (e.g. paged/long-context
// storage).
// --------------------------------------------------------------------------

void quant_dequant_roundtrip_keys(
    const float* input,
    float*       output,
    int num_channels,
    int group_size);

void quant_dequant_roundtrip_values(
    const float* input,
    float*       output,
    int num_tokens,
    int head_dim,
    int group_size);

// Non-blocking variants — see quantize_kernels.hpp for the rationale.
sycl::event quant_dequant_roundtrip_keys_submit(
    const float* input,
    float*       output,
    int num_channels,
    int group_size);

sycl::event quant_dequant_roundtrip_values_submit(
    const float* input,
    float*       output,
    int num_tokens,
    int head_dim,
    int group_size);
