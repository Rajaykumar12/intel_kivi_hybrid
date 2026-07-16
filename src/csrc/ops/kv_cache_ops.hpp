#pragma once

#include <torch/extension.h>

// ==========================================================================
// Torch Tensor Ops — validation + dispatch to SYCL kernels
// ==========================================================================

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
