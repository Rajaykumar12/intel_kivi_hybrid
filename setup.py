"""
KIVI-SYCL: 2-bit KV Cache Quantization for Intel GPUs
======================================================
Build & install:
    pip install . --no-build-isolation

Requires:
    - Intel oneAPI DPC++ compiler (icpx) whose SYCL runtime is compatible with
      the one bundled in the installed torch XPU wheel. See
      docs/optimization_roadmap/README.md §5 for the working recipe when
      setvars.sh resolves to a newer oneAPI than torch was built with.
    - PyTorch XPU build (torch >= 2.4 ships libc10_xpu; IPEX optional)
"""

import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

# --------------------------------------------------------------------------
# Build flags
# --------------------------------------------------------------------------
# Note: no explicit -I<oneapi>/include/sycl is added here. icpx already
# injects its own SYCL headers (in the correct order) whenever -fsycl is
# passed; adding that path again via -I causes stl_wrappers/cmath to be
# picked up before the compiler's own macro setup, which breaks the build
# on newer glibc (e.g. Fedora) with errors like "unknown type name
# '__DPCPP_SYCL_EXTERNAL_LIBC'".
cxx_flags = os.environ.get("CXXFLAGS", "").split()
ld_flags = os.environ.get("LDFLAGS", "").split()

# Floating point: icpx's default model is already "fast" — unsafe-math
# reassociation, approximate libm (-fapprox-func), FTZ/DAZ denormals, FMA
# contraction — but it still honours NaN/Inf. Measured 2026-11-09: forcing
# -fp-model=precise made the CPU fused attention ~1.6x slower (median 1.44 vs
# 0.91 ms per layer at 8k over repeated runs; its accumulation loops lose
# reassociation) with no accuracy gain (1.5e-6 vs 1.0e-6 max error against the
# dequantize+SDPA reference) and all tests passing either way.
# Do NOT add -ffast-math: on top of the default it implies finite-math-only,
# which lets the compiler drop the NaN/Inf handling the fp16 scale floor and
# saturation in kernels/internal/quant_tile.hpp exist for.
compile_flags = ["-fsycl", "-fPIC", "-std=c++17", "-O3", "-w"] + cxx_flags
link_flags = compile_flags + ld_flags

# Ahead-of-time device compilation (optional). Without it the SPIR-V device
# code is JIT-compiled by the GPU driver on first use — measured ~1.5 s on
# Iris Xe for the dequantize image on a cold driver cache. KIVI_AOT_DEVICE is
# passed to ocloc's -device: a device (adl-p for Alder Lake Iris Xe, tgllp,
# dg2, ...), a family (xe-lp, gen12lp, xe-hpg, ...), a PCI id (0x46a8) or a
# comma list; `ocloc compile --help` lists them. The generic spir64 target is
# kept alongside, so the binary still JIT-runs on any other Intel GPU.
_aot_device = os.environ.get("KIVI_AOT_DEVICE", "").strip()
if _aot_device:
    _targets = ["-fsycl-targets=spir64_gen,spir64"]
    compile_flags += _targets
    link_flags += _targets + ["-Xsycl-target-backend=spir64_gen", f"-device {_aot_device}"]

# Submit kernels on the SYCL queue backing PyTorch's current XPU stream
# (see src/csrc/include/kivi/common.hpp) when the installed torch ships
# libc10_xpu (torch >= 2.4 XPU builds). Older torch/IPEX stacks fall back
# to the private-queue path, so this stays a no-op there.
import torch as _torch  # noqa: E402  (needed for lib path detection)

_torch_lib_dir = os.path.join(os.path.dirname(_torch.__file__), "lib")
_has_c10_xpu = os.path.exists(os.path.join(_torch_lib_dir, "libc10_xpu.so"))
extra_libraries = []
if _has_c10_xpu:
    compile_flags.append("-DKIVI_USE_TORCH_XPU_QUEUE")
    extra_libraries.append("c10_xpu")

# --------------------------------------------------------------------------
# Extension build only. Package metadata (name, version, license, deps,
# classifiers) lives in pyproject.toml; the version comes from
# src/kivi_sycl/_version.py.
# --------------------------------------------------------------------------
CSRC = os.path.join("src", "csrc")

setup(
    ext_modules=[
        CppExtension(
            name="kivi_sycl._C",
            sources=[
                os.path.join(CSRC, "bindings", "pybind_module.cpp"),
                os.path.join(CSRC, "kernels", "quantize_kernels.cpp"),
                os.path.join(CSRC, "kernels", "dequantize_kernels.cpp"),
                os.path.join(CSRC, "kernels", "roundtrip_kernels.cpp"),
                os.path.join(CSRC, "ops", "kv_cache_ops.cpp"),
                os.path.join(CSRC, "cpu", "fused_attention.cpp"),
                os.path.join(CSRC, "library", "register_ops.cpp"),
            ],
            include_dirs=[CSRC, os.path.join(CSRC, "include")],
            libraries=extra_libraries,
            library_dirs=[_torch_lib_dir] if extra_libraries else [],
            extra_compile_args=compile_flags,
            extra_link_args=link_flags,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
