// ==========================================================================
// KIVI: Tuning-Free Asymmetric 2-bit KV Cache Quantization for Intel GPUs
// ==========================================================================
// Implements the KIVI algorithm (arXiv:2402.02750) using SYCL for Intel XPU.
//
// Key insight from paper:
//   - Keys exhibit per-channel outliers  → quantize per-channel (along token dim)
//   - Values exhibit per-token outliers  → quantize per-token (along channel dim)
//   - A residual window of R recent tokens is kept in FP16/FP32 for accuracy.
//
// Quantization: Asymmetric 2-bit
//   q = clamp(round((x - min) / scale), 0, 3)
//   scale = (max - min) / 3
//   Dequantize: x ≈ q * scale + min
//
// Packing: 4 × 2-bit values per uint8
//   byte = q0 | (q1 << 2) | (q2 << 4) | (q3 << 6)
// ==========================================================================

#include <sycl/sycl.hpp>
#include <torch/extension.h>
#include <vector>
#include <iostream>
#include <stdexcept>

// --------------------------------------------------------------------------
// SYCL Queue Management
// --------------------------------------------------------------------------
// Uses in_order queue for deterministic execution. When called from Python
// with XPU tensors allocated via IPEX, USM pointers are valid because IPEX
// uses Level Zero USM allocations on the same device.
// --------------------------------------------------------------------------
sycl::queue& get_queue() {
    static sycl::queue* q_ptr = nullptr;
    if (!q_ptr) {
        try {
            q_ptr = new sycl::queue(sycl::gpu_selector_v,
                                    sycl::property::queue::in_order{});
            std::cout << "[KIVI] SYCL device: "
                      << q_ptr->get_device().get_info<sycl::info::device::name>()
                      << std::endl;
        } catch (const sycl::exception& e) {
            throw std::runtime_error(
                std::string("[KIVI] Failed to create SYCL queue: ") + e.what());
        }
    }
    return *q_ptr;
}

// --------------------------------------------------------------------------
// 1. KEY QUANTIZATION — Per-Channel Asymmetric 2-bit
// --------------------------------------------------------------------------
// Input layout:  [B, H, D, Seq]  (transposed keys, contiguous)
// Each "channel" = one (batch, head, dim) slice across Seq tokens.
// group_size = number of tokens per quantization group (= Seq for single group).
//
// Output packed: [B, H, D, Seq/4]   dtype=uint8
// Output scales: [B, H, D]          dtype=float32
// Output zeros:  [B, H, D]          dtype=float32
// --------------------------------------------------------------------------
void quantize_keys_per_channel(
    const float* __restrict__ input,
    uint8_t*     __restrict__ output,
    float*       __restrict__ scales,
    float*       __restrict__ zeros,
    int num_channels,
    int group_size)
{
    sycl::queue& q = get_queue();
    const int local_size  = 256;
    const int global_size = ((num_channels + local_size - 1) / local_size) * local_size;

    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(global_size, local_size),
            [=](sycl::nd_item<1> item) {
                int ch = item.get_global_id(0);
                if (ch >= num_channels) return;

                // --- Find min/max across the group ---
                float max_val = -1e30f;
                float min_val =  1e30f;
                int base = ch * group_size;

                for (int i = 0; i < group_size; ++i) {
                    float val = input[base + i];
                    max_val = sycl::fmax(max_val, val);
                    min_val = sycl::fmin(min_val, val);
                }

                // --- Asymmetric scale & zero ---
                float scale = (max_val - min_val) / 3.0f + 1e-10f;
                scales[ch] = scale;
                zeros[ch]  = min_val;

                // --- Pack 4 × 2-bit values per byte ---
                int out_base = ch * (group_size / 4);
                for (int i = 0; i < group_size / 4; ++i) {
                    uint8_t packed = 0;
                    for (int j = 0; j < 4; ++j) {
                        float val = input[base + i * 4 + j];
                        float qf  = (val - min_val) / scale;
                        int   qi  = static_cast<int>(sycl::round(qf));
                        qi = sycl::clamp(qi, 0, 3);
                        packed |= (static_cast<uint8_t>(qi) << (j * 2));
                    }
                    output[out_base + i] = packed;
                }
            });
    }).wait();
}

// --------------------------------------------------------------------------
// 2. VALUE QUANTIZATION — Per-Token Asymmetric 2-bit
// --------------------------------------------------------------------------
// Input layout:  [B, H, Seq, D]  (values, contiguous)
// Each "group" = group_size consecutive elements within one token's D dim.
//
// Output packed: [B, H, Seq, D/4]                dtype=uint8
// Output scales: [B, H, Seq, D/group_size]       dtype=float32
// Output zeros:  [B, H, Seq, D/group_size]       dtype=float32
// --------------------------------------------------------------------------
void quantize_values_per_token(
    const float* __restrict__ input,
    uint8_t*     __restrict__ output,
    float*       __restrict__ scales,
    float*       __restrict__ zeros,
    int num_tokens,
    int head_dim,
    int group_size)
{
    sycl::queue& q = get_queue();
    int groups_per_token = head_dim / group_size;
    int total_groups     = num_tokens * groups_per_token;
    const int local_size  = 256;
    const int global_size = ((total_groups + local_size - 1) / local_size) * local_size;

    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(global_size, local_size),
            [=](sycl::nd_item<1> item) {
                int gid = item.get_global_id(0);
                if (gid >= total_groups) return;

                int token_idx    = gid / groups_per_token;
                int group_in_tok = gid % groups_per_token;
                int start_feat   = group_in_tok * group_size;
                int base_input   = token_idx * head_dim + start_feat;

                // --- Find min/max within the group ---
                float max_val = -1e30f;
                float min_val =  1e30f;
                for (int i = 0; i < group_size; ++i) {
                    float val = input[base_input + i];
                    max_val = sycl::fmax(max_val, val);
                    min_val = sycl::fmin(min_val, val);
                }

                float scale = (max_val - min_val) / 3.0f + 1e-10f;
                scales[gid] = scale;
                zeros[gid]  = min_val;

                // --- Pack ---
                // Output packed index: each token has head_dim/4 packed bytes
                int packed_per_token = head_dim / 4;
                int out_base = token_idx * packed_per_token + group_in_tok * (group_size / 4);

                for (int i = 0; i < group_size / 4; ++i) {
                    uint8_t packed = 0;
                    for (int j = 0; j < 4; ++j) {
                        float val = input[base_input + i * 4 + j];
                        float qf  = (val - min_val) / scale;
                        int   qi  = static_cast<int>(sycl::round(qf));
                        qi = sycl::clamp(qi, 0, 3);
                        packed |= (static_cast<uint8_t>(qi) << (j * 2));
                    }
                    output[out_base + i] = packed;
                }
            });
    }).wait();
}

// --------------------------------------------------------------------------
// 3. KEY DEQUANTIZATION — Per-Channel Asymmetric
// --------------------------------------------------------------------------
void dequantize_keys_per_channel(
    const uint8_t* __restrict__ input,
    const float*   __restrict__ scales,
    const float*   __restrict__ zeros,
    float*         __restrict__ output,
    int num_channels,
    int group_size)
{
    sycl::queue& q = get_queue();
    int bytes_per_channel = group_size / 4;
    int total_bytes       = num_channels * bytes_per_channel;
    const int local_size  = 256;
    const int global_size = ((total_bytes + local_size - 1) / local_size) * local_size;

    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(global_size, local_size),
            [=](sycl::nd_item<1> item) {
                int idx = item.get_global_id(0);
                if (idx >= total_bytes) return;

                int ch         = idx / bytes_per_channel;
                int byte_in_ch = idx % bytes_per_channel;

                float scale    = scales[ch];
                float zero     = zeros[ch];
                uint8_t packed = input[idx];

                int out_base = ch * group_size + byte_in_ch * 4;
                for (int j = 0; j < 4; ++j) {
                    uint8_t qi = (packed >> (j * 2)) & 0x3;
                    output[out_base + j] = static_cast<float>(qi) * scale + zero;
                }
            });
    }).wait();
}

// --------------------------------------------------------------------------
// 4. VALUE DEQUANTIZATION — Per-Token Asymmetric
// --------------------------------------------------------------------------
void dequantize_values_per_token(
    const uint8_t* __restrict__ input,
    const float*   __restrict__ scales,
    const float*   __restrict__ zeros,
    float*         __restrict__ output,
    int num_groups,
    int group_size)
{
    sycl::queue& q = get_queue();
    int bytes_per_group   = group_size / 4;
    int total_bytes       = num_groups * bytes_per_group;
    const int local_size  = 256;
    const int global_size = ((total_bytes + local_size - 1) / local_size) * local_size;

    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(global_size, local_size),
            [=](sycl::nd_item<1> item) {
                int byte_idx = item.get_global_id(0);
                if (byte_idx >= total_bytes) return;

                int group_idx   = byte_idx / bytes_per_group;
                int byte_in_grp = byte_idx % bytes_per_group;

                float scale    = scales[group_idx];
                float zero     = zeros[group_idx];
                uint8_t packed = input[byte_idx];

                // Output: group_idx * group_size + byte_in_grp * 4
                int out_base = group_idx * group_size + byte_in_grp * 4;
                for (int j = 0; j < 4; ++j) {
                    uint8_t qi = (packed >> (j * 2)) & 0x3;
                    output[out_base + j] = static_cast<float>(qi) * scale + zero;
                }
            });
    }).wait();
}


// ==========================================================================
// Python Bindings — Torch Tensor Wrappers with Validation
// ==========================================================================

static void check_tensor(const torch::Tensor& t, const char* name,
                          torch::ScalarType expected_dtype) {
    TORCH_CHECK(t.is_contiguous(),
                name, " must be contiguous");
    TORCH_CHECK(t.scalar_type() == expected_dtype,
                name, " must be ", torch::toString(expected_dtype),
                " but got ", torch::toString(t.scalar_type()));
}

void kivi_quant_keys(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor scales,
    torch::Tensor zeros,
    int group_size)
{
    check_tensor(input,  "input",  torch::kFloat32);
    check_tensor(output, "output", torch::kUInt8);
    check_tensor(scales, "scales", torch::kFloat32);
    check_tensor(zeros,  "zeros",  torch::kFloat32);
    TORCH_CHECK(group_size > 0 && group_size % 4 == 0,
                "group_size must be positive and divisible by 4, got ", group_size);

    int num_channels = scales.numel();
    quantize_keys_per_channel(
        input.data_ptr<float>(),
        output.data_ptr<uint8_t>(),
        scales.data_ptr<float>(),
        zeros.data_ptr<float>(),
        num_channels,
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
    check_tensor(input,  "input",  torch::kFloat32);
    check_tensor(output, "output", torch::kUInt8);
    check_tensor(scales, "scales", torch::kFloat32);
    check_tensor(zeros,  "zeros",  torch::kFloat32);
    TORCH_CHECK(group_size > 0 && group_size % 4 == 0,
                "group_size must be positive and divisible by 4");
    TORCH_CHECK(head_dim > 0 && head_dim % group_size == 0,
                "head_dim must be positive and divisible by group_size");

    int num_tokens = input.numel() / head_dim;
    quantize_values_per_token(
        input.data_ptr<float>(),
        output.data_ptr<uint8_t>(),
        scales.data_ptr<float>(),
        zeros.data_ptr<float>(),
        num_tokens,
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
    check_tensor(input,  "input",  torch::kUInt8);
    check_tensor(scales, "scales", torch::kFloat32);
    check_tensor(zeros,  "zeros",  torch::kFloat32);
    check_tensor(output, "output", torch::kFloat32);

    int num_channels = scales.numel();
    dequantize_keys_per_channel(
        input.data_ptr<uint8_t>(),
        scales.data_ptr<float>(),
        zeros.data_ptr<float>(),
        output.data_ptr<float>(),
        num_channels,
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
    check_tensor(input,  "input",  torch::kUInt8);
    check_tensor(scales, "scales", torch::kFloat32);
    check_tensor(zeros,  "zeros",  torch::kFloat32);
    check_tensor(output, "output", torch::kFloat32);

    int num_groups = scales.numel();
    dequantize_values_per_token(
        input.data_ptr<uint8_t>(),
        scales.data_ptr<float>(),
        zeros.data_ptr<float>(),
        output.data_ptr<float>(),
        num_groups,
        group_size);
}

// ==========================================================================
// Module Registration
// ==========================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "KIVI: Asymmetric 2-bit KV Cache Quantization for Intel XPU (SYCL)";

    m.def("quantize_keys",     &kivi_quant_keys,
          "Asymmetric 2-bit key quantization (per-channel)",
          py::arg("input"), py::arg("output"), py::arg("scales"),
          py::arg("zeros"), py::arg("group_size"));

    m.def("quantize_values",   &kivi_quant_values,
          "Asymmetric 2-bit value quantization (per-token)",
          py::arg("input"), py::arg("output"), py::arg("scales"),
          py::arg("zeros"), py::arg("head_dim"), py::arg("group_size"));

    m.def("dequantize_keys",   &kivi_dequant_keys,
          "Asymmetric 2-bit key dequantization (per-channel)",
          py::arg("input"), py::arg("scales"), py::arg("zeros"),
          py::arg("output"), py::arg("group_size"));

    m.def("dequantize_values", &kivi_dequant_values,
          "Asymmetric 2-bit value dequantization (per-token)",
          py::arg("input"), py::arg("scales"), py::arg("zeros"),
          py::arg("output"), py::arg("head_dim"), py::arg("group_size"));
}