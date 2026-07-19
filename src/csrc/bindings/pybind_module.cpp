// ==========================================================================
// KIVI: pybind11 module registration — thin glue only.
// Installed as the kivi_sycl._C submodule (see setup.py).
// ==========================================================================

#include "ops/kv_cache_ops.hpp"
#include "kivi/event.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "KIVI: Asymmetric 2-bit KV Cache Quantization for Intel XPU (SYCL)";

    py::class_<kivi::KiviEvent>(m, "KiviEvent",
        "Handle for a kernel submitted via a *_async op. Call .wait() before "
        "reading any tensor the kernel writes.")
        .def("wait", &kivi::KiviEvent::wait,
             "Block until the kernel completes; rethrows any async SYCL error.");

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
          "discarded by the caller.",
          py::arg("input"), py::arg("output"), py::arg("group_size"));

    m.def("quant_dequant_roundtrip_values", &kivi_quant_dequant_roundtrip_values,
          "Fused quantize->dequantize round-trip (per-token). See "
          "quant_dequant_roundtrip_keys.",
          py::arg("input"), py::arg("output"), py::arg("head_dim"), py::arg("group_size"));

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
          py::arg("input"), py::arg("output"), py::arg("group_size"));

    m.def("quant_dequant_roundtrip_values_async", &kivi_quant_dequant_roundtrip_values_async,
          "Non-blocking quant_dequant_roundtrip_values — returns a KiviEvent, does not wait.",
          py::arg("input"), py::arg("output"), py::arg("head_dim"), py::arg("group_size"));
}
