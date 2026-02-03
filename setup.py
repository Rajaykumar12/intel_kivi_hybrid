import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

# ==============================================================================
# 1. HARDCODED INCLUDE PATH (Derived from your 'find' command)
# ==============================================================================
# We point to the folder containing 'ext/'
oneapi_sycl_include = "/opt/intel/oneapi/compiler/2025.3/include/sycl"

print(f"🔹 Manually adding SYCL Include Path: {oneapi_sycl_include}")

# 2. STANDARD SETUP
cxx_flags = os.environ.get("CXXFLAGS", "").split()
ld_flags = os.environ.get("LDFLAGS", "").split()

# 3. BUILD FLAGS
# -I adds the include path so the compiler finds bfloat16.hpp
flags = ['-fsycl', '-fPIC', '-std=c++17', '-O3', '-w'] + cxx_flags
flags.append(f"-I{oneapi_sycl_include}")

setup(
    name='kivi_sycl',
    ext_modules=[
        CppExtension(
            name='kivi_sycl',
            sources=['src/kivi_optimized.cpp'], 
            extra_compile_args=flags,
            extra_link_args=flags + ld_flags
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)