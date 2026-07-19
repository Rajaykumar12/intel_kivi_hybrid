// ==========================================================================
// KIVI: Torch tensor wrappers — shape/dtype validation, then dispatch to
// the SYCL kernels in kernels/. Kept separate from bindings/ so this
// validation logic could be reused outside a pybind11 module if needed.
//
// Validation is factored into standalone validate_* functions so the
// blocking and non-blocking (_async) entry points for the same op share
// identical checks instead of duplicating TORCH_CHECK blocks.
// ==========================================================================

#include "kv_cache_ops.hpp"
#include "kivi/common.hpp"
#include "kernels/quantize_kernels.hpp"
#include "kernels/dequantize_kernels.hpp"
#include "kernels/roundtrip_kernels.hpp"

namespace {

int64_t validate_quant_keys_args(
    const torch::Tensor& input, const torch::Tensor& output,
    const torch::Tensor& scales, const torch::Tensor& zeros, int group_size)
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
    return num_channels;
}

int64_t validate_quant_values_args(
    const torch::Tensor& input, const torch::Tensor& output,
    const torch::Tensor& scales, const torch::Tensor& zeros,
    int head_dim, int group_size)
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
    return num_tokens;
}

int64_t validate_dequant_keys_args(
    const torch::Tensor& input, const torch::Tensor& scales,
    const torch::Tensor& zeros, const torch::Tensor& output, int group_size)
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
    return num_channels;
}

int64_t validate_dequant_values_args(
    const torch::Tensor& input, const torch::Tensor& scales,
    const torch::Tensor& zeros, const torch::Tensor& output,
    int head_dim, int group_size)
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
    return num_groups;
}

int64_t validate_roundtrip_keys_args(
    const torch::Tensor& input, const torch::Tensor& output, int group_size)
{
    kivi::check_tensor(input,  "input",  torch::kFloat32);
    kivi::check_tensor(output, "output", torch::kFloat32);
    TORCH_CHECK(group_size > 0 && group_size % 4 == 0,
                "group_size must be positive and divisible by 4, got ", group_size);
    TORCH_CHECK(input.numel() % group_size == 0,
                "input.numel() (", input.numel(), ") must be divisible by group_size (",
                group_size, ")");
    TORCH_CHECK(output.numel() == input.numel(),
                "output size (", output.numel(), ") must equal input size (",
                input.numel(), ") — round-trip is shape-preserving");

    int64_t num_channels = input.numel() / group_size;
    TORCH_CHECK(num_channels > 0, "input must be non-empty");
    return num_channels;
}

int64_t validate_roundtrip_values_args(
    const torch::Tensor& input, const torch::Tensor& output,
    int head_dim, int group_size)
{
    kivi::check_tensor(input,  "input",  torch::kFloat32);
    kivi::check_tensor(output, "output", torch::kFloat32);
    TORCH_CHECK(group_size > 0 && group_size % 4 == 0,
                "group_size must be positive and divisible by 4");
    TORCH_CHECK(head_dim > 0 && head_dim % group_size == 0,
                "head_dim must be positive and divisible by group_size");
    TORCH_CHECK(input.numel() % head_dim == 0,
                "input.numel() (", input.numel(), ") must be divisible by head_dim (",
                head_dim, ")");
    TORCH_CHECK(output.numel() == input.numel(),
                "output size (", output.numel(), ") must equal input size (",
                input.numel(), ") — round-trip is shape-preserving");

    int64_t num_tokens = input.numel() / head_dim;
    TORCH_CHECK(num_tokens > 0, "input must be non-empty");
    return num_tokens;
}

}  // namespace

// -------------------------------------------------------------------------
// Blocking entry points
// -------------------------------------------------------------------------

void kivi_quant_keys(
    torch::Tensor input, torch::Tensor output,
    torch::Tensor scales, torch::Tensor zeros, int group_size)
{
    int64_t num_channels = validate_quant_keys_args(input, output, scales, zeros, group_size);
    quantize_keys_per_channel(
        input.data_ptr<float>(), output.data_ptr<uint8_t>(),
        scales.data_ptr<float>(), zeros.data_ptr<float>(),
        static_cast<int>(num_channels), group_size);
}

void kivi_quant_values(
    torch::Tensor input, torch::Tensor output,
    torch::Tensor scales, torch::Tensor zeros, int head_dim, int group_size)
{
    int64_t num_tokens = validate_quant_values_args(input, output, scales, zeros, head_dim, group_size);
    quantize_values_per_token(
        input.data_ptr<float>(), output.data_ptr<uint8_t>(),
        scales.data_ptr<float>(), zeros.data_ptr<float>(),
        static_cast<int>(num_tokens), head_dim, group_size);
}

void kivi_dequant_keys(
    torch::Tensor input, torch::Tensor scales,
    torch::Tensor zeros, torch::Tensor output, int group_size)
{
    int64_t num_channels = validate_dequant_keys_args(input, scales, zeros, output, group_size);
    dequantize_keys_per_channel(
        input.data_ptr<uint8_t>(), scales.data_ptr<float>(), zeros.data_ptr<float>(),
        output.data_ptr<float>(), static_cast<int>(num_channels), group_size);
}

void kivi_dequant_values(
    torch::Tensor input, torch::Tensor scales,
    torch::Tensor zeros, torch::Tensor output, int head_dim, int group_size)
{
    int64_t num_groups = validate_dequant_values_args(input, scales, zeros, output, head_dim, group_size);
    dequantize_values_per_token(
        input.data_ptr<uint8_t>(), scales.data_ptr<float>(), zeros.data_ptr<float>(),
        output.data_ptr<float>(), static_cast<int>(num_groups), group_size);
}

void kivi_quant_dequant_roundtrip_keys(
    torch::Tensor input, torch::Tensor output, int group_size)
{
    int64_t num_channels = validate_roundtrip_keys_args(input, output, group_size);
    quant_dequant_roundtrip_keys(
        input.data_ptr<float>(), output.data_ptr<float>(),
        static_cast<int>(num_channels), group_size);
}

void kivi_quant_dequant_roundtrip_values(
    torch::Tensor input, torch::Tensor output, int head_dim, int group_size)
{
    int64_t num_tokens = validate_roundtrip_values_args(input, output, head_dim, group_size);
    quant_dequant_roundtrip_values(
        input.data_ptr<float>(), output.data_ptr<float>(),
        static_cast<int>(num_tokens), head_dim, group_size);
}

// -------------------------------------------------------------------------
// Non-blocking entry points — validation runs synchronously (host-side,
// cheap); only the kernel submission is deferred. The returned KiviEvent
// must be waited on before any output tensor is read.
// -------------------------------------------------------------------------

kivi::KiviEvent kivi_quant_keys_async(
    torch::Tensor input, torch::Tensor output,
    torch::Tensor scales, torch::Tensor zeros, int group_size)
{
    int64_t num_channels = validate_quant_keys_args(input, output, scales, zeros, group_size);
    return kivi::KiviEvent(quantize_keys_per_channel_submit(
        input.data_ptr<float>(), output.data_ptr<uint8_t>(),
        scales.data_ptr<float>(), zeros.data_ptr<float>(),
        static_cast<int>(num_channels), group_size));
}

kivi::KiviEvent kivi_quant_values_async(
    torch::Tensor input, torch::Tensor output,
    torch::Tensor scales, torch::Tensor zeros, int head_dim, int group_size)
{
    int64_t num_tokens = validate_quant_values_args(input, output, scales, zeros, head_dim, group_size);
    return kivi::KiviEvent(quantize_values_per_token_submit(
        input.data_ptr<float>(), output.data_ptr<uint8_t>(),
        scales.data_ptr<float>(), zeros.data_ptr<float>(),
        static_cast<int>(num_tokens), head_dim, group_size));
}

kivi::KiviEvent kivi_dequant_keys_async(
    torch::Tensor input, torch::Tensor scales,
    torch::Tensor zeros, torch::Tensor output, int group_size)
{
    int64_t num_channels = validate_dequant_keys_args(input, scales, zeros, output, group_size);
    return kivi::KiviEvent(dequantize_keys_per_channel_submit(
        input.data_ptr<uint8_t>(), scales.data_ptr<float>(), zeros.data_ptr<float>(),
        output.data_ptr<float>(), static_cast<int>(num_channels), group_size));
}

kivi::KiviEvent kivi_dequant_values_async(
    torch::Tensor input, torch::Tensor scales,
    torch::Tensor zeros, torch::Tensor output, int head_dim, int group_size)
{
    int64_t num_groups = validate_dequant_values_args(input, scales, zeros, output, head_dim, group_size);
    return kivi::KiviEvent(dequantize_values_per_token_submit(
        input.data_ptr<uint8_t>(), scales.data_ptr<float>(), zeros.data_ptr<float>(),
        output.data_ptr<float>(), static_cast<int>(num_groups), group_size));
}

kivi::KiviEvent kivi_quant_dequant_roundtrip_keys_async(
    torch::Tensor input, torch::Tensor output, int group_size)
{
    int64_t num_channels = validate_roundtrip_keys_args(input, output, group_size);
    return kivi::KiviEvent(quant_dequant_roundtrip_keys_submit(
        input.data_ptr<float>(), output.data_ptr<float>(),
        static_cast<int>(num_channels), group_size));
}

kivi::KiviEvent kivi_quant_dequant_roundtrip_values_async(
    torch::Tensor input, torch::Tensor output, int head_dim, int group_size)
{
    int64_t num_tokens = validate_roundtrip_values_args(input, output, head_dim, group_size);
    return kivi::KiviEvent(quant_dequant_roundtrip_values_submit(
        input.data_ptr<float>(), output.data_ptr<float>(),
        static_cast<int>(num_tokens), head_dim, group_size));
}
