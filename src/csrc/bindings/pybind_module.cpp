// ==========================================================================
// KIVI: pybind11 module registration — thin glue only.
// Installed as the kivi_sycl._C submodule (see setup.py).
// ==========================================================================

#include "ops/kv_cache_ops.hpp"

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
