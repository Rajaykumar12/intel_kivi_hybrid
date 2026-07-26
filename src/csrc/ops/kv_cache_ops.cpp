// ==========================================================================
// KIVI: Torch tensor wrappers — shape/dtype validation, then dispatch to
// the SYCL kernels in kernels/. Kept separate from bindings/ so this
// validation logic could be reused outside a pybind11 module if needed.
//
// Validation is factored into standalone validate_* functions so the
// blocking and non-blocking (_async) entry points for the same op share
// identical checks instead of duplicating TORCH_CHECK blocks.
//
// Element dtype: every op accepts float32 or float16 tensors (bfloat16
// callers must upcast — there is no bf16 kernel instantiation). The float
// dtype is taken from the op's float-typed input tensor; scales, zeros,
// and the float output must all match it exactly.
// ==========================================================================

#include "kv_cache_ops.hpp"
#include "kivi/common.hpp"
#include "kernels/quantize_kernels.hpp"
#include "kernels/dequantize_kernels.hpp"
#include "kernels/roundtrip_kernels.hpp"

namespace {

// Map the SYCL element type to the ATen element type used by data_ptr<>.
// sycl::half and at::Half are layout-identical IEEE binary16.
template <typename T> struct AtenElem;
template <> struct AtenElem<float>      { using type = float; };
template <> struct AtenElem<sycl::half> { using type = at::Half; };

template <typename T>
const T* const_elem_ptr(const torch::Tensor& t) {
    return reinterpret_cast<const T*>(
        t.data_ptr<typename AtenElem<T>::type>());
}

template <typename T>
T* elem_ptr(torch::Tensor& t) {
    return reinterpret_cast<T*>(t.data_ptr<typename AtenElem<T>::type>());
}

torch::ScalarType float_dtype_of(const torch::Tensor& t, const char* op) {
    auto dt = t.scalar_type();
    TORCH_CHECK(dt == torch::kFloat32 || dt == torch::kHalf,
                op, ": element dtype must be float32 or float16, got ",
                torch::toString(dt),
                " (bfloat16 callers must upcast to float32 first)");
    return dt;
}

int64_t validate_quant_keys_args(
    const torch::Tensor& input, const torch::Tensor& output,
    const torch::Tensor& scales, const torch::Tensor& zeros,
    int group_size, torch::ScalarType dt)
{
    kivi::check_tensor(input,  "input",  dt);
    kivi::check_tensor(output, "output", torch::kUInt8);
    kivi::check_tensor(scales, "scales", dt);
    kivi::check_tensor(zeros,  "zeros",  dt);
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
    int head_dim, int group_size, torch::ScalarType dt)
{
    kivi::check_tensor(input,  "input",  dt);
    kivi::check_tensor(output, "output", torch::kUInt8);
    kivi::check_tensor(scales, "scales", dt);
    kivi::check_tensor(zeros,  "zeros",  dt);
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
    const torch::Tensor& zeros, const torch::Tensor& output,
    int group_size, torch::ScalarType dt)
{
    kivi::check_tensor(input,  "input",  torch::kUInt8);
    kivi::check_tensor(scales, "scales", dt);
    kivi::check_tensor(zeros,  "zeros",  dt);
    kivi::check_tensor(output, "output", dt);
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
    int head_dim, int group_size, torch::ScalarType dt)
{
    kivi::check_tensor(input,  "input",  torch::kUInt8);
    kivi::check_tensor(scales, "scales", dt);
    kivi::check_tensor(zeros,  "zeros",  dt);
    kivi::check_tensor(output, "output", dt);
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
    const torch::Tensor& input, const torch::Tensor& output,
    int group_size, torch::ScalarType dt)
{
    kivi::check_tensor(input,  "input",  dt);
    kivi::check_tensor(output, "output", dt);
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
    int head_dim, int group_size, torch::ScalarType dt)
{
    kivi::check_tensor(input,  "input",  dt);
    kivi::check_tensor(output, "output", dt);
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

// --- Typed submit helpers: one per op, shared by blocking/async paths ---

template <typename T>
sycl::event submit_quant_keys(
    torch::Tensor& input, torch::Tensor& output,
    torch::Tensor& scales, torch::Tensor& zeros,
    int64_t num_channels, int group_size)
{
    return quantize_keys_per_channel_submit<T>(
        const_elem_ptr<T>(input), output.data_ptr<uint8_t>(),
        elem_ptr<T>(scales), elem_ptr<T>(zeros),
        static_cast<int>(num_channels), group_size);
}

template <typename T>
sycl::event submit_quant_values(
    torch::Tensor& input, torch::Tensor& output,
    torch::Tensor& scales, torch::Tensor& zeros,
    int64_t num_tokens, int head_dim, int group_size)
{
    return quantize_values_per_token_submit<T>(
        const_elem_ptr<T>(input), output.data_ptr<uint8_t>(),
        elem_ptr<T>(scales), elem_ptr<T>(zeros),
        static_cast<int>(num_tokens), head_dim, group_size);
}

template <typename T>
sycl::event submit_dequant_keys(
    torch::Tensor& input, torch::Tensor& scales,
    torch::Tensor& zeros, torch::Tensor& output,
    int64_t num_channels, int group_size)
{
    return dequantize_keys_per_channel_submit<T>(
        input.data_ptr<uint8_t>(),
        const_elem_ptr<T>(scales), const_elem_ptr<T>(zeros),
        elem_ptr<T>(output), static_cast<int>(num_channels), group_size);
}

template <typename T>
sycl::event submit_dequant_values(
    torch::Tensor& input, torch::Tensor& scales,
    torch::Tensor& zeros, torch::Tensor& output,
    int64_t num_groups, int group_size)
{
    return dequantize_values_per_token_submit<T>(
        input.data_ptr<uint8_t>(),
        const_elem_ptr<T>(scales), const_elem_ptr<T>(zeros),
        elem_ptr<T>(output), static_cast<int>(num_groups), group_size);
}

template <typename T>
sycl::event submit_roundtrip_keys(
    torch::Tensor& input, torch::Tensor& output,
    int64_t num_channels, int group_size)
{
    return quant_dequant_roundtrip_keys_submit<T>(
        const_elem_ptr<T>(input), elem_ptr<T>(output),
        static_cast<int>(num_channels), group_size);
}

template <typename T>
sycl::event submit_roundtrip_values(
    torch::Tensor& input, torch::Tensor& output,
    int64_t num_tokens, int head_dim, int group_size)
{
    return quant_dequant_roundtrip_values_submit<T>(
        const_elem_ptr<T>(input), elem_ptr<T>(output),
        static_cast<int>(num_tokens), head_dim, group_size);
}

}  // namespace

// -------------------------------------------------------------------------
// Blocking entry points
// -------------------------------------------------------------------------

void kivi_quant_keys(
    torch::Tensor input, torch::Tensor output,
    torch::Tensor scales, torch::Tensor zeros, int group_size)
{
    auto dt = float_dtype_of(input, "quantize_keys");
    int64_t nc = validate_quant_keys_args(input, output, scales, zeros, group_size, dt);
    auto ev = (dt == torch::kFloat32)
        ? submit_quant_keys<float>(input, output, scales, zeros, nc, group_size)
        : submit_quant_keys<sycl::half>(input, output, scales, zeros, nc, group_size);
    ev.wait_and_throw();
}

void kivi_quant_values(
    torch::Tensor input, torch::Tensor output,
    torch::Tensor scales, torch::Tensor zeros, int head_dim, int group_size)
{
    auto dt = float_dtype_of(input, "quantize_values");
    int64_t nt = validate_quant_values_args(input, output, scales, zeros,
                                            head_dim, group_size, dt);
    auto ev = (dt == torch::kFloat32)
        ? submit_quant_values<float>(input, output, scales, zeros, nt, head_dim, group_size)
        : submit_quant_values<sycl::half>(input, output, scales, zeros, nt, head_dim, group_size);
    ev.wait_and_throw();
}

void kivi_dequant_keys(
    torch::Tensor input, torch::Tensor scales,
    torch::Tensor zeros, torch::Tensor output, int group_size)
{
    auto dt = float_dtype_of(output, "dequantize_keys");
    int64_t nc = validate_dequant_keys_args(input, scales, zeros, output, group_size, dt);
    auto ev = (dt == torch::kFloat32)
        ? submit_dequant_keys<float>(input, scales, zeros, output, nc, group_size)
        : submit_dequant_keys<sycl::half>(input, scales, zeros, output, nc, group_size);
    ev.wait_and_throw();
}

void kivi_dequant_values(
    torch::Tensor input, torch::Tensor scales,
    torch::Tensor zeros, torch::Tensor output, int head_dim, int group_size)
{
    auto dt = float_dtype_of(output, "dequantize_values");
    int64_t ng = validate_dequant_values_args(input, scales, zeros, output,
                                              head_dim, group_size, dt);
    auto ev = (dt == torch::kFloat32)
        ? submit_dequant_values<float>(input, scales, zeros, output, ng, group_size)
        : submit_dequant_values<sycl::half>(input, scales, zeros, output, ng, group_size);
    ev.wait_and_throw();
}

void kivi_quant_dequant_roundtrip_keys(
    torch::Tensor input, torch::Tensor output, int group_size)
{
    auto dt = float_dtype_of(input, "quant_dequant_roundtrip_keys");
    int64_t nc = validate_roundtrip_keys_args(input, output, group_size, dt);
    auto ev = (dt == torch::kFloat32)
        ? submit_roundtrip_keys<float>(input, output, nc, group_size)
        : submit_roundtrip_keys<sycl::half>(input, output, nc, group_size);
    ev.wait_and_throw();
}

void kivi_quant_dequant_roundtrip_values(
    torch::Tensor input, torch::Tensor output, int head_dim, int group_size)
{
    auto dt = float_dtype_of(input, "quant_dequant_roundtrip_values");
    int64_t nt = validate_roundtrip_values_args(input, output, head_dim, group_size, dt);
    auto ev = (dt == torch::kFloat32)
        ? submit_roundtrip_values<float>(input, output, nt, head_dim, group_size)
        : submit_roundtrip_values<sycl::half>(input, output, nt, head_dim, group_size);
    ev.wait_and_throw();
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
    auto dt = float_dtype_of(input, "quantize_keys_async");
    int64_t nc = validate_quant_keys_args(input, output, scales, zeros, group_size, dt);
    return kivi::KiviEvent((dt == torch::kFloat32)
        ? submit_quant_keys<float>(input, output, scales, zeros, nc, group_size)
        : submit_quant_keys<sycl::half>(input, output, scales, zeros, nc, group_size));
}

kivi::KiviEvent kivi_quant_values_async(
    torch::Tensor input, torch::Tensor output,
    torch::Tensor scales, torch::Tensor zeros, int head_dim, int group_size)
{
    auto dt = float_dtype_of(input, "quantize_values_async");
    int64_t nt = validate_quant_values_args(input, output, scales, zeros,
                                            head_dim, group_size, dt);
    return kivi::KiviEvent((dt == torch::kFloat32)
        ? submit_quant_values<float>(input, output, scales, zeros, nt, head_dim, group_size)
        : submit_quant_values<sycl::half>(input, output, scales, zeros, nt, head_dim, group_size));
}

kivi::KiviEvent kivi_dequant_keys_async(
    torch::Tensor input, torch::Tensor scales,
    torch::Tensor zeros, torch::Tensor output, int group_size)
{
    auto dt = float_dtype_of(output, "dequantize_keys_async");
    int64_t nc = validate_dequant_keys_args(input, scales, zeros, output, group_size, dt);
    return kivi::KiviEvent((dt == torch::kFloat32)
        ? submit_dequant_keys<float>(input, scales, zeros, output, nc, group_size)
        : submit_dequant_keys<sycl::half>(input, scales, zeros, output, nc, group_size));
}

kivi::KiviEvent kivi_dequant_values_async(
    torch::Tensor input, torch::Tensor scales,
    torch::Tensor zeros, torch::Tensor output, int head_dim, int group_size)
{
    auto dt = float_dtype_of(output, "dequantize_values_async");
    int64_t ng = validate_dequant_values_args(input, scales, zeros, output,
                                              head_dim, group_size, dt);
    return kivi::KiviEvent((dt == torch::kFloat32)
        ? submit_dequant_values<float>(input, scales, zeros, output, ng, group_size)
        : submit_dequant_values<sycl::half>(input, scales, zeros, output, ng, group_size));
}

kivi::KiviEvent kivi_quant_dequant_roundtrip_keys_async(
    torch::Tensor input, torch::Tensor output, int group_size)
{
    auto dt = float_dtype_of(input, "quant_dequant_roundtrip_keys_async");
    int64_t nc = validate_roundtrip_keys_args(input, output, group_size, dt);
    return kivi::KiviEvent((dt == torch::kFloat32)
        ? submit_roundtrip_keys<float>(input, output, nc, group_size)
        : submit_roundtrip_keys<sycl::half>(input, output, nc, group_size));
}

kivi::KiviEvent kivi_quant_dequant_roundtrip_values_async(
    torch::Tensor input, torch::Tensor output, int head_dim, int group_size)
{
    auto dt = float_dtype_of(input, "quant_dequant_roundtrip_values_async");
    int64_t nt = validate_roundtrip_values_args(input, output, head_dim, group_size, dt);
    return kivi::KiviEvent((dt == torch::kFloat32)
        ? submit_roundtrip_values<float>(input, output, nt, head_dim, group_size)
        : submit_roundtrip_values<sycl::half>(input, output, nt, head_dim, group_size));
}
