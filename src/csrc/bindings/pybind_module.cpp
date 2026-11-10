// ==========================================================================
// KIVI: pybind11 module registration — thin glue only.
// Installed as the kivi_sycl._C submodule (see setup.py).
// ==========================================================================

#include <torch/extension.h>

#include "ops/kv_cache_ops.hpp"
#include "kivi/event.hpp"
#include "kivi/usm.hpp"
#include "cpu/fused_attention.hpp"
#include "engine/dequant_cpu.hpp"
#include "engine/engine.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "KIVI: Asymmetric 2-bit KV Cache Quantization for Intel XPU (SYCL)";

    py::class_<kivi::KiviEvent>(m, "KiviEvent",
        "Handle for a kernel submitted via a *_async op. Call .wait() before "
        "reading any tensor the kernel writes.")
        .def("wait", &kivi::KiviEvent::wait,
             // Release the GIL while blocked: a background thread waiting on a
             // flush must not stall the Python thread running the model.
             py::call_guard<py::gil_scoped_release>(),
             "Block until the kernel completes; rethrows any async SYCL error.");

    // --- USM host memory (zero-copy on integrated GPUs; see kivi/usm.hpp)
    m.def("alloc_usm_host",
          [](std::vector<int64_t> sizes, at::ScalarType dtype, int64_t align) {
              return kivi::alloc_usm_host(sizes, dtype, align);
          },
          "Allocate an uninitialized CPU tensor backed by SYCL USM host memory, "
          "aligned to `align` bytes. Torch treats it as an ordinary CPU tensor; "
          "the KIVI kernels can also read and write it directly, with no "
          "host<->device copy.",
          py::arg("sizes"), py::arg("dtype"), py::arg("align") = 64);

    m.def("is_kernel_accessible", &kivi::is_kernel_accessible,
          "True if KIVI kernels can dereference this tensor's memory: an XPU "
          "tensor, or a CPU tensor allocated with alloc_usm_host.",
          py::arg("tensor"));

    // --- CPU fused decode attention over the packed history (Phase 2 §B.4)
    m.def("fused_decode_attention", &kivi_fused_decode_attention,
          "One decode step of attention computed directly from the packed 2-bit "
          "history plus the FP residual, without materialising the dequantized "
          "history. Returns [B, 1, Hq, D].",
          py::arg("query"), py::arg("k_res"), py::arg("v_res"),
          py::arg("packed_k"), py::arg("scales_k"), py::arg("zeros_k"),
          py::arg("packed_v"), py::arg("scales_v"), py::arg("zeros_v"),
          py::arg("scaling"), py::arg("group_size"),
          py::call_guard<py::gil_scoped_release>());
    // --- kivi::Engine — the cache engine in C++ (Phase 5 S1.4) -------------
    py::class_<kivi::Engine>(m, "Engine",
        "KIVI cache engine (C++ port of KiviCache's state machine). update() is "
        "the HF-native order, append() the legacy add_tokens order.")
        .def(py::init([](int64_t num_layers, int64_t head_dim, int64_t num_kv_heads,
                         int64_t residual_length, std::vector<int64_t> group_size,
                         const std::string& metadata_dtype, const std::string& zero_copy,
                         bool async_flush, const std::string& device, int64_t row_align) {
                 kivi::EngineConfig c;
                 c.num_layers = num_layers; c.head_dim = head_dim;
                 c.num_kv_heads = num_kv_heads; c.residual_length = residual_length;
                 c.group_size = std::move(group_size);
                 if (metadata_dtype == "float16") c.metadata_dtype = at::kHalf;
                 else if (metadata_dtype == "float32") c.metadata_dtype = at::kFloat;
                 else TORCH_CHECK(metadata_dtype == "auto", "metadata_dtype must be auto/float16/float32");
                 c.zero_copy = zero_copy == "on" ? kivi::ZeroCopy::On
                             : zero_copy == "off" ? kivi::ZeroCopy::Off : kivi::ZeroCopy::Auto;
                 c.async_flush = async_flush; c.device = device; c.row_align = row_align;
                 return std::make_unique<kivi::Engine>(std::move(c));
             }),
             py::arg("num_layers"), py::arg("head_dim"), py::arg("num_kv_heads") = 0,
             py::arg("residual_length") = 128, py::arg("group_size") = std::vector<int64_t>{32},
             py::arg("metadata_dtype") = "auto", py::arg("zero_copy") = "auto",
             py::arg("async_flush") = true, py::arg("device") = "xpu", py::arg("row_align") = 1)
        .def("update", &kivi::Engine::update, py::arg("layer"), py::arg("k"), py::arg("v"))
        .def("append", &kivi::Engine::append, py::arg("layer"), py::arg("k"), py::arg("v"))
        .def("harvest", &kivi::Engine::harvest, py::arg("layer"))
        .def("harvest_all", &kivi::Engine::harvest_all)
        .def("reserve", &kivi::Engine::reserve, py::arg("total_tokens"))
        .def("reset", &kivi::Engine::reset)
        .def("residual", [](const kivi::Engine& e, int64_t i) -> py::object {
                 auto r = e.residual(i);
                 if (!r) return py::none();
                 return py::make_tuple(r->first, r->second);
             }, py::arg("layer"))
        .def("history", [](const kivi::Engine& e, int64_t i) -> py::object {
                 auto h = e.history(i);
                 if (!h) return py::none();
                 return py::make_tuple((*h)[0], (*h)[1], (*h)[2], (*h)[3], (*h)[4], (*h)[5]);
             }, py::arg("layer"))
        .def("quant_tokens", &kivi::Engine::quant_tokens, py::arg("layer"))
        .def("flushes", &kivi::Engine::flushes, py::arg("layer"))
        .def("pending_tokens", &kivi::Engine::pending_tokens, py::arg("layer"))
        .def("arena_capacity", &kivi::Engine::arena_capacity, py::arg("layer"))
        .def("seq_length", &kivi::Engine::seq_length, py::arg("layer"))
        .def_property_readonly("total_flushes", &kivi::Engine::total_flushes)
        .def_property_readonly("zero_copy_enabled", &kivi::Engine::zero_copy_enabled)
        .def("stats", [](const kivi::Engine& e, int64_t i) {
                 auto s = e.stats(i);
                 py::dict d;
                 d["group_size"] = s.group_size; d["total_tokens"] = s.total_tokens;
                 d["quantized_tokens"] = s.quantized_tokens; d["residual_tokens"] = s.residual_tokens;
                 d["pending_tokens"] = s.pending_tokens; d["packed_bytes"] = s.packed_bytes;
                 d["packed_used_bytes"] = s.packed_used_bytes;
                 d["packed_capacity_tokens"] = s.packed_capacity_tokens;
                 d["residual_bytes"] = s.residual_bytes; d["kivi_bytes"] = s.kivi_bytes;
                 d["fp_baseline_bytes"] = s.fp_baseline_bytes; d["compression"] = s.compression;
                 return d;
             }, py::arg("layer") = 0);

    // --- C++ dequantize-on-read (Phase 5 S1.3; port of KiviCache._dequant_*)
    m.def("dequant_keys_cpu", &kivi::dequant_keys,
          "Dequantize packed per-channel keys [B,H,D,T/4] -> [B,H,T,D] on the CPU.",
          py::arg("packed"), py::arg("scales"), py::arg("zeros"), py::arg("dtype"));
    m.def("dequant_values_cpu", &kivi::dequant_values,
          "Dequantize packed per-token values [B,H,T,D/4] -> [B,H,T,D] on the CPU.",
          py::arg("packed"), py::arg("scales"), py::arg("zeros"), py::arg("dtype"));

    m.def("fused_attention_uses_avx2", &kivi_fused_attention_uses_avx2,
          "Whether the fused attention runs its AVX2/FMA code path on this CPU.");

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

    m.def("quant_dequant_roundtrip_keys", &kivi_quant_dequant_roundtrip_keys,
          "Fused quantize->dequantize round-trip (per-channel). Rounds "
          "values to 2-bit fidelity and writes the reconstructed float "
          "straight back out, without ever materializing the packed byte, "
          "scale, or zero-point buffers. Use instead of quantize_keys + "
          "dequantize_keys whenever the packed representation itself is "
          "discarded by the caller. half_metadata=True rounds the internal "
          "scale/zero through fp16, matching quantize_keys with fp16 scales.",
          py::arg("input"), py::arg("output"), py::arg("group_size"),
          py::arg("half_metadata") = false);

    m.def("quant_dequant_roundtrip_values", &kivi_quant_dequant_roundtrip_values,
          "Fused quantize->dequantize round-trip (per-token). See "
          "quant_dequant_roundtrip_keys.",
          py::arg("input"), py::arg("output"), py::arg("head_dim"), py::arg("group_size"),
          py::arg("half_metadata") = false);

    // --- Non-blocking variants: submit and return a KiviEvent immediately.
    // Caller must call .wait() on the returned event before reading output
    // tensors. See docs/sycl_kernel_interface.md §3.
    m.def("quantize_keys_async",     &kivi_quant_keys_async,
          "Non-blocking quantize_keys — returns a KiviEvent, does not wait.",
          py::arg("input"), py::arg("output"), py::arg("scales"),
          py::arg("zeros"), py::arg("group_size"));

    m.def("quantize_values_async",   &kivi_quant_values_async,
          "Non-blocking quantize_values — returns a KiviEvent, does not wait.",
          py::arg("input"), py::arg("output"), py::arg("scales"),
          py::arg("zeros"), py::arg("head_dim"), py::arg("group_size"));

    m.def("dequantize_keys_async",   &kivi_dequant_keys_async,
          "Non-blocking dequantize_keys — returns a KiviEvent, does not wait.",
          py::arg("input"), py::arg("scales"), py::arg("zeros"),
          py::arg("output"), py::arg("group_size"));

    m.def("dequantize_values_async", &kivi_dequant_values_async,
          "Non-blocking dequantize_values — returns a KiviEvent, does not wait.",
          py::arg("input"), py::arg("scales"), py::arg("zeros"),
          py::arg("output"), py::arg("head_dim"), py::arg("group_size"));

    m.def("quant_dequant_roundtrip_keys_async", &kivi_quant_dequant_roundtrip_keys_async,
          "Non-blocking quant_dequant_roundtrip_keys — returns a KiviEvent, does not wait.",
          py::arg("input"), py::arg("output"), py::arg("group_size"),
          py::arg("half_metadata") = false);

    m.def("quant_dequant_roundtrip_values_async", &kivi_quant_dequant_roundtrip_values_async,
          "Non-blocking quant_dequant_roundtrip_values — returns a KiviEvent, does not wait.",
          py::arg("input"), py::arg("output"), py::arg("head_dim"), py::arg("group_size"),
          py::arg("half_metadata") = false);
}
