#pragma once

#include <torch/extension.h>
#include "kivi/event.hpp"

// ==========================================================================
// Torch Tensor Ops — validation + dispatch to SYCL kernels
// ==========================================================================

// --------------------------------------------------------------------------
// Fused round-trip ops: quantize to 2-bit fidelity and immediately
// dequantize back to float, in a single kernel launch, without ever
// materializing the packed/scales/zeros buffers. Use these instead of a
// quantize_+dequantize_ pair whenever the packed representation itself is
// discarded by the caller (this is the flush path's actual usage pattern —
// see roundtrip_kernels.hpp).
// --------------------------------------------------------------------------
void kivi_quant_dequant_roundtrip_keys(
    torch::Tensor input,
    torch::Tensor output,
    int group_size);

void kivi_quant_dequant_roundtrip_values(
    torch::Tensor input,
    torch::Tensor output,
    int head_dim,
    int group_size);

// --------------------------------------------------------------------------
// Non-blocking variants of every op above: submit the kernel and return a
// KiviEvent immediately instead of blocking until it completes. The caller
// must call `.wait()` on the returned event before reading any output
// tensor. Tensor validation still happens synchronously before submission
// (shape/dtype/device checks are host-side and cheap; only the kernel
// execution itself is deferred).
// --------------------------------------------------------------------------
kivi::KiviEvent kivi_quant_keys_async(
    torch::Tensor input, torch::Tensor output,
    torch::Tensor scales, torch::Tensor zeros, int group_size);

kivi::KiviEvent kivi_quant_values_async(
    torch::Tensor input, torch::Tensor output,
    torch::Tensor scales, torch::Tensor zeros, int head_dim, int group_size);

kivi::KiviEvent kivi_dequant_keys_async(
    torch::Tensor input, torch::Tensor scales,
    torch::Tensor zeros, torch::Tensor output, int group_size);

kivi::KiviEvent kivi_dequant_values_async(
    torch::Tensor input, torch::Tensor scales,
    torch::Tensor zeros, torch::Tensor output, int head_dim, int group_size);

kivi::KiviEvent kivi_quant_dequant_roundtrip_keys_async(
    torch::Tensor input, torch::Tensor output, int group_size);

kivi::KiviEvent kivi_quant_dequant_roundtrip_values_async(
    torch::Tensor input, torch::Tensor output, int head_dim, int group_size);

void kivi_quant_keys(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor scales,
    torch::Tensor zeros,
    int group_size);

void kivi_quant_values(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor scales,
    torch::Tensor zeros,
    int head_dim,
    int group_size);

void kivi_dequant_keys(
    torch::Tensor input,
    torch::Tensor scales,
    torch::Tensor zeros,
    torch::Tensor output,
    int group_size);

void kivi_dequant_values(
    torch::Tensor input,
    torch::Tensor scales,
    torch::Tensor zeros,
    torch::Tensor output,
    int head_dim,
    int group_size);
