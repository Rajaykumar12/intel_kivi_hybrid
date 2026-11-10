# ==========================================================================
# KIVI native build (Phase 5): Python-free libkivi_core.so + C++ programs.
#
# The Python extension (kivi_sycl._C) is still built by setup.py from the
# same sources plus src/csrc/bindings/. This CMake build produces the parts a
# Python-free runtime needs:
#   kivi_core          shared library: SYCL kernels, ops, CPU fused attention
#   kivi_native_tests  C++ test program (ctest)
#
# Configure via scripts/build_native.sh, which selects a oneAPI compiler
# compatible with the SYCL runtime in the torch wheel and points CMake at
# torch's own CMake config (find_package(Torch)).
# ==========================================================================
cmake_minimum_required(VERSION 3.18)
project(kivi_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Torch REQUIRED)

# Same queue policy as setup.py: submit on torch's XPU stream when the torch
# build ships libc10_xpu (see src/csrc/include/kivi/common.hpp).
get_filename_component(KIVI_TORCH_LIB_DIR "${TORCH_LIBRARY}" DIRECTORY)
find_library(KIVI_C10_XPU c10_xpu PATHS "${KIVI_TORCH_LIB_DIR}" NO_DEFAULT_PATH)

# SYCL runtime: link against the one torch itself loads (the torch XPU wheel
# resolves libsycl.so.9 / libur_loader.so.0 via RPATH $ORIGIN/../../../..),
# not the compiler's newer copy. Otherwise an executable fails to link —
# the compiler's libsycl needs Unified Runtime symbols (urGraphGetIdExp, …)
# that torch's bundled loader does not export — and at run time two SYCL
# runtimes would meet in one process. Override with -DKIVI_SYCL_RUNTIME_DIR.
get_filename_component(_kivi_default_rt "${KIVI_TORCH_LIB_DIR}/../../../.." ABSOLUTE)
set(KIVI_SYCL_RUNTIME_DIR "${_kivi_default_rt}" CACHE PATH "directory with the libsycl.so torch uses")
if(NOT EXISTS "${KIVI_SYCL_RUNTIME_DIR}/libsycl.so")
  message(FATAL_ERROR "kivi: no libsycl.so in KIVI_SYCL_RUNTIME_DIR=${KIVI_SYCL_RUNTIME_DIR}")
endif()
message(STATUS "kivi: SYCL runtime from ${KIVI_SYCL_RUNTIME_DIR}")
set(KIVI_RPATH "${KIVI_TORCH_LIB_DIR};${KIVI_SYCL_RUNTIME_DIR}")

set(KIVI_CSRC ${CMAKE_CURRENT_SOURCE_DIR}/src/csrc)
add_library(kivi_core SHARED
  ${KIVI_CSRC}/kernels/quantize_kernels.cpp
  ${KIVI_CSRC}/kernels/dequantize_kernels.cpp
  ${KIVI_CSRC}/kernels/roundtrip_kernels.cpp
  ${KIVI_CSRC}/ops/kv_cache_ops.cpp
  ${KIVI_CSRC}/cpu/fused_attention.cpp
)
target_include_directories(kivi_core PUBLIC ${KIVI_CSRC} ${KIVI_CSRC}/include)
# Flags mirror setup.py (fp model rationale is documented there).
target_compile_options(kivi_core PUBLIC -fsycl -O3)
# -L first, so the driver's implicit -lsycl resolves to torch's runtime.
target_link_options(kivi_core PUBLIC "-L${KIVI_SYCL_RUNTIME_DIR}" -fsycl)
target_link_libraries(kivi_core PUBLIC ${TORCH_LIBRARIES})
if(KIVI_C10_XPU)
  target_compile_definitions(kivi_core PUBLIC KIVI_USE_TORCH_XPU_QUEUE)
  target_link_libraries(kivi_core PUBLIC ${KIVI_C10_XPU})
  message(STATUS "kivi: using torch XPU stream queue (${KIVI_C10_XPU})")
endif()
# Ahead-of-time device code, same switch as setup.py (KIVI_AOT_DEVICE=adl-p ...).
if(DEFINED ENV{KIVI_AOT_DEVICE} AND NOT "$ENV{KIVI_AOT_DEVICE}" STREQUAL "")
  target_compile_options(kivi_core PUBLIC -fsycl-targets=spir64_gen,spir64)
  target_link_options(kivi_core PUBLIC -fsycl-targets=spir64_gen,spir64
                      "SHELL:-Xsycl-target-backend=spir64_gen \"-device $ENV{KIVI_AOT_DEVICE}\"")
  message(STATUS "kivi: AOT device code for $ENV{KIVI_AOT_DEVICE}")
endif()
# Find the torch libraries next to the binary's recorded torch location.
set_target_properties(kivi_core PROPERTIES BUILD_RPATH "${KIVI_RPATH}")

enable_testing()
add_executable(kivi_native_tests tests/native/test_core.cpp)
target_link_libraries(kivi_native_tests PRIVATE kivi_core)
set_target_properties(kivi_native_tests PROPERTIES
  BUILD_RPATH "${KIVI_RPATH};$<TARGET_FILE_DIR:kivi_core>")
add_test(NAME kivi_native_tests COMMAND kivi_native_tests)
