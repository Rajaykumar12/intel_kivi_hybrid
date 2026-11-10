// ==========================================================================
// KIVI: pybind11 module registration — thin glue only.
// Installed as the kivi_sycl._C submodule (see setup.py).
// ==========================================================================

#include "ops/kv_cache_ops.hpp"
#include "kivi/event.hpp"
#include "kivi/usm.hpp"
#include "cpu/fused_attention.hpp"

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
