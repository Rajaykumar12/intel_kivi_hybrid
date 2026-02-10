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
import glob
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension

# --------------------------------------------------------------------------
# Auto-detect oneAPI SYCL include path (portable across versions)
# --------------------------------------------------------------------------
oneapi_root = os.environ.get("ONEAPI_ROOT", "/opt/intel/oneapi")
sycl_include = os.environ.get("SYCL_INCLUDE_DIR", "")

if not sycl_include:
    # Search common locations
    candidates = sorted(
        glob.glob(os.path.join(oneapi_root, "compiler", "*", "include", "sycl")),
        reverse=True,  # newest version first
    )
    if candidates:
        sycl_include = candidates[0]
    else:
        # Fallback: try the 'latest' symlink
        fallback = os.path.join(oneapi_root, "compiler", "latest", "include", "sycl")
        if os.path.isdir(fallback):
            sycl_include = fallback

if sycl_include:
    print(f"[KIVI] SYCL include: {sycl_include}")
else:
    print("[KIVI] WARNING: Could not find SYCL include dir. "
          "Set SYCL_INCLUDE_DIR or source setvars.sh.")

# --------------------------------------------------------------------------
# Build flags
# --------------------------------------------------------------------------
cxx_flags = os.environ.get("CXXFLAGS", "").split()
ld_flags = os.environ.get("LDFLAGS", "").split()

compile_flags = ["-fsycl", "-fPIC", "-std=c++17", "-O3", "-w"] + cxx_flags
if sycl_include:
    compile_flags.append(f"-I{sycl_include}")

link_flags = compile_flags + ld_flags

# --------------------------------------------------------------------------
# Package metadata
# --------------------------------------------------------------------------
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="kivi-sycl",
    version="0.1.0",
    author="Rajay",
    description="KIVI: Tuning-Free Asymmetric 2-bit KV Cache Quantization for Intel GPUs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rajay/kivi-sycl",  # TODO: update with your repo URL
    license="MIT",
    python_requires=">=3.10",
    install_requires=[
        "torch",
        "intel-extension-for-pytorch",
        "transformers",
    ],
    py_modules=["kivi_cache"],
    ext_modules=[
        CppExtension(
            name="kivi_sycl",
            sources=["src/kivi_optimized.cpp"],
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