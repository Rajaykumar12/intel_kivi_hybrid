// ==========================================================================
// KIVI: Torch tensor wrappers — shape/dtype validation, then dispatch to
// the SYCL kernels in kernels/. Kept separate from bindings/ so this
// validation logic could be reused outside a pybind11 module if needed.
// ==========================================================================

#include "kv_cache_ops.hpp"
#include "kivi/common.hpp"
#include "kernels/quantize_kernels.hpp"
#include "kernels/dequantize_kernels.hpp"

void kivi_quant_keys(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor scales,
    torch::Tensor zeros,
    int group_size)
{
    kivi::check_tensor(input,  "input",  torch::kFloat32);
    kivi::check_tensor(output, "output", torch::kUInt8);
    kivi::check_tensor(scales, "scales", torch::kFloat32);
    kivi::check_tensor(zeros,  "zeros",  torch::kFloat32);
    TORCH_CHECK(group_size > 0 && group_size % 4 == 0,
                "group_size must be positive and divisible by 4, got ", group_size);

    int64_t num_channels = scales.numel();
    TORCH_CHECK(num_channels > 0, "scales must be non-empty");
    TORCH_CHECK(zeros.numel() == num_channels,
                "zeros size (", zeros.numel(), ") must match scales size (",
                num_channels, ")");
    TORCH_CHECK(input.numel() == num_channels * group_size,
                "input size (", input.numel(), ") must equal num_channels(",
                num_channels, ") * group_size(", group_size, ") = ",
                num_channels * group_size);
    TORCH_CHECK(output.numel() == num_channels * (group_size / 4),
                "output size (", output.numel(), ") must equal num_channels(",
                num_channels, ") * (group_size/4) = ",
                num_channels * (group_size / 4));

    quantize_keys_per_channel(
        input.data_ptr<float>(),
        output.data_ptr<uint8_t>(),
        scales.data_ptr<float>(),
        zeros.data_ptr<float>(),
        static_cast<int>(num_channels),
        group_size);
}

void kivi_quant_values(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor scales,
    torch::Tensor zeros,
    int head_dim,
    int group_size)
{
    kivi::check_tensor(input,  "input",  torch::kFloat32);
    kivi::check_tensor(output, "output", torch::kUInt8);
    kivi::check_tensor(scales, "scales", torch::kFloat32);
    kivi::check_tensor(zeros,  "zeros",  torch::kFloat32);
    TORCH_CHECK(group_size > 0 && group_size % 4 == 0,
                "group_size must be positive and divisible by 4");
    TORCH_CHECK(head_dim > 0 && head_dim % group_size == 0,
                "head_dim must be positive and divisible by group_size");
    TORCH_CHECK(input.numel() % head_dim == 0,
                "input.numel() (", input.numel(), ") must be divisible by head_dim (",
                head_dim, ") — refusing to silently truncate trailing elements");

    int64_t num_tokens = input.numel() / head_dim;
    int64_t groups_per_token = head_dim / group_size;
    TORCH_CHECK(num_tokens > 0, "input must be non-empty");
    TORCH_CHECK(scales.numel() == num_tokens * groups_per_token,
                "scales size (", scales.numel(), ") must equal num_tokens(",
                num_tokens, ") * groups_per_token(", groups_per_token, ")");
    TORCH_CHECK(zeros.numel() == scales.numel(),
                "zeros size (", zeros.numel(), ") must match scales size (",
                scales.numel(), ")");
    TORCH_CHECK(output.numel() == num_tokens * (head_dim / 4),
                "output size (", output.numel(), ") must equal num_tokens(",
                num_tokens, ") * (head_dim/4) = ", num_tokens * (head_dim / 4));

    quantize_values_per_token(
        input.data_ptr<float>(),
        output.data_ptr<uint8_t>(),
        scales.data_ptr<float>(),
        zeros.data_ptr<float>(),
        static_cast<int>(num_tokens),
        head_dim,
        group_size);
}

void kivi_dequant_keys(
    torch::Tensor input,
    torch::Tensor scales,
    torch::Tensor zeros,
    torch::Tensor output,
    int group_size)
{
    kivi::check_tensor(input,  "input",  torch::kUInt8);
    kivi::check_tensor(scales, "scales", torch::kFloat32);
    kivi::check_tensor(zeros,  "zeros",  torch::kFloat32);
    kivi::check_tensor(output, "output", torch::kFloat32);
    TORCH_CHECK(group_size > 0 && group_size % 4 == 0,
                "group_size must be positive and divisible by 4, got ", group_size);

    int64_t num_channels = scales.numel();
    TORCH_CHECK(num_channels > 0, "scales must be non-empty");
    TORCH_CHECK(zeros.numel() == num_channels,
                "zeros size (", zeros.numel(), ") must match scales size (",
                num_channels, ")");
    TORCH_CHECK(input.numel() == num_channels * (group_size / 4),
                "input (packed) size (", input.numel(), ") must equal num_channels(",
                num_channels, ") * (group_size/4) = ",
                num_channels * (group_size / 4));
    TORCH_CHECK(output.numel() == num_channels * group_size,
                "output size (", output.numel(), ") must equal num_channels(",
                num_channels, ") * group_size = ", num_channels * group_size);

    dequantize_keys_per_channel(
        input.data_ptr<uint8_t>(),
        scales.data_ptr<float>(),
        zeros.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(num_channels),
        group_size);
}

void kivi_dequant_values(
    torch::Tensor input,
    torch::Tensor scales,
    torch::Tensor zeros,
    torch::Tensor output,
    int head_dim,
    int group_size)
{
    kivi::check_tensor(input,  "input",  torch::kUInt8);
    kivi::check_tensor(scales, "scales", torch::kFloat32);
    kivi::check_tensor(zeros,  "zeros",  torch::kFloat32);
    kivi::check_tensor(output, "output", torch::kFloat32);
    TORCH_CHECK(group_size > 0 && group_size % 4 == 0,
                "group_size must be positive and divisible by 4, got ", group_size);
    TORCH_CHECK(head_dim > 0 && head_dim % group_size == 0,
                "head_dim must be positive and divisible by group_size");

    int64_t num_groups = scales.numel();
    TORCH_CHECK(num_groups > 0, "scales must be non-empty");
    TORCH_CHECK(zeros.numel() == num_groups,
                "zeros size (", zeros.numel(), ") must match scales size (",
                num_groups, ")");
    TORCH_CHECK(input.numel() == num_groups * (group_size / 4),
                "input (packed) size (", input.numel(), ") must equal num_groups(",
                num_groups, ") * (group_size/4) = ", num_groups * (group_size / 4));
    TORCH_CHECK(output.numel() == num_groups * group_size,
                "output size (", output.numel(), ") must equal num_groups(",
                num_groups, ") * group_size = ", num_groups * group_size);

    dequantize_values_per_token(
        input.data_ptr<uint8_t>(),
        scales.data_ptr<float>(),
        zeros.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(num_groups),
        group_size);
}
