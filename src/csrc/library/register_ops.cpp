// ==========================================================================
// KIVI ops as real torch operators: torch.ops.kivi.* (Phase 5 S1.2).
//
// pybind functions are invisible to torch.compile / torch.export / AOTInductor;
// dispatcher-registered ops are not. This file is compiled into BOTH the
// Python extension (kivi_sycl._C) and the Python-free libkivi_core.so. A
// process must load only one of them — TORCH_LIBRARY(kivi) may be defined
// once per process.
//
// Schemas mark the outputs the quantize ops write as mutated (Tensor(a!)), so
// functionalization and graph capture keep the writes and their order.
// Meta (shape-only) kernels let FakeTensor tracing run without a device.
//
// Dispatch keys: the quantize ops accept XPU tensors and USM-backed CPU
// tensors (zero-copy arena views), so they are registered for both CPU and
// XPU; device accessibility is still checked inside the op. The fused
// attention runs on the CPU only.
// ==========================================================================

#include <torch/library.h>
#include <torch/torch.h>

#include "cpu/fused_attention.hpp"
#include "ops/kv_cache_ops.hpp"

namespace {

void quantize_keys(const at::Tensor& input, const at::Tensor& packed,
                   const at::Tensor& scales, const at::Tensor& zeros, int64_t group_size) {
    kivi_quant_keys(input, packed, scales, zeros, static_cast<int>(group_size));
}

void quantize_values(const at::Tensor& input, const at::Tensor& packed,
                     const at::Tensor& scales, const at::Tensor& zeros,
                     int64_t head_dim, int64_t group_size) {
    kivi_quant_values(input, packed, scales, zeros, static_cast<int>(head_dim),
                      static_cast<int>(group_size));
}

at::Tensor fused_decode_attention(
    const at::Tensor& query, const at::Tensor& k_res, const at::Tensor& v_res,
    const at::Tensor& packed_k, const at::Tensor& scales_k, const at::Tensor& zeros_k,
    const at::Tensor& packed_v, const at::Tensor& scales_v, const at::Tensor& zeros_v,
    double scaling, int64_t group_size) {
    return kivi_fused_decode_attention(query, k_res, v_res, packed_k, scales_k, zeros_k,
                                       packed_v, scales_v, zeros_v, scaling, group_size);
}

// --- Meta kernels: shapes only ---------------------------------------------

void quantize_keys_meta(const at::Tensor&, const at::Tensor&, const at::Tensor&,
                        const at::Tensor&, int64_t) {}

void quantize_values_meta(const at::Tensor&, const at::Tensor&, const at::Tensor&,
                          const at::Tensor&, int64_t, int64_t) {}

at::Tensor fused_decode_attention_meta(
    const at::Tensor& query, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, const at::Tensor&, double, int64_t) {
    TORCH_CHECK(query.dim() == 4 && query.size(2) == 1,
                "fused_decode_attention: query must be [B, Hq, 1, D]");
    // [B, 1, Hq, D] — the layout transformers attention functions return.
    return at::empty({query.size(0), 1, query.size(1), query.size(3)},
                     query.options());
}

}  // namespace

TORCH_LIBRARY(kivi, m) {
    m.def("quantize_keys(Tensor input, Tensor(a!) packed, Tensor(b!) scales, "
          "Tensor(c!) zeros, int group_size) -> ()");
    m.def("quantize_values(Tensor input, Tensor(a!) packed, Tensor(b!) scales, "
          "Tensor(c!) zeros, int head_dim, int group_size) -> ()");
    m.def("fused_decode_attention(Tensor query, Tensor k_res, Tensor v_res, "
          "Tensor packed_k, Tensor scales_k, Tensor zeros_k, "
          "Tensor packed_v, Tensor scales_v, Tensor zeros_v, "
          "float scaling, int group_size) -> Tensor");
}

TORCH_LIBRARY_IMPL(kivi, CPU, m) {
    m.impl("quantize_keys", &quantize_keys);
    m.impl("quantize_values", &quantize_values);
    m.impl("fused_decode_attention", &fused_decode_attention);
}

TORCH_LIBRARY_IMPL(kivi, XPU, m) {
    m.impl("quantize_keys", &quantize_keys);
    m.impl("quantize_values", &quantize_values);
}

TORCH_LIBRARY_IMPL(kivi, Meta, m) {
    m.impl("quantize_keys", &quantize_keys_meta);
    m.impl("quantize_values", &quantize_values_meta);
    m.impl("fused_decode_attention", &fused_decode_attention_meta);
}
