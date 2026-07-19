"""
KIVI-SYCL: 2-bit KV Cache Quantization for Intel GPUs
======================================================
Build & install:
    pip install . --no-build-isolation

Requires:
    - Intel oneAPI DPC++ compiler (source setvars.sh first)
    - PyTorch with Intel Extension for PyTorch (IPEX)
"""

import os
from setuptools import setup, find_packages
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

compile_flags = ["-fsycl", "-fPIC", "-std=c++17", "-O3", "-w"] + cxx_flags
link_flags = compile_flags + ld_flags

# --------------------------------------------------------------------------
# Package metadata
# --------------------------------------------------------------------------
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

CSRC = os.path.join("src", "csrc")

setup(
    name="kivi-sycl",
    version="0.1.0",
    author="Rajay",
    description="KIVI: Tuning-Free Asymmetric 2-bit KV Cache Quantization for Intel GPUs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Rajaykumar12/intel_kivi_hybrid",  # TODO: update with your repo URL
    license="MIT",
    python_requires=">=3.10",
    install_requires=[
        "torch",
        "transformers",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    ext_modules=[
        CppExtension(
            name="kivi_sycl._C",
            sources=[
                os.path.join(CSRC, "bindings", "pybind_module.cpp"),
                os.path.join(CSRC, "kernels", "quantize_kernels.cpp"),
                os.path.join(CSRC, "kernels", "dequantize_kernels.cpp"),
                os.path.join(CSRC, "kernels", "roundtrip_kernels.cpp"),
                os.path.join(CSRC, "ops", "kv_cache_ops.cpp"),
            ],
            include_dirs=[CSRC, os.path.join(CSRC, "include")],
            extra_compile_args=compile_flags,
            extra_link_args=link_flags,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: POSIX :: Linux",
    ],
    keywords="kv-cache quantization 2-bit sycl intel xpu llm inference",
)