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

# SYCL runtime: use the one torch itself loads (the torch XPU wheel resolves
# libsycl.so.9 / libur_loader.so.0 via RPATH $ORIGIN/../../../..), not the
# compiler's newer copy. torch's FindSYCLToolkit.cmake would otherwise cache
# the compiler's libsycl.so as SYCL_LIBRARY and link it into every target via
# torch::xpurt — which then fails to link executables (the compiler's libsycl
# needs Unified Runtime symbols such as urGraphGetIdExp that torch's bundled
# loader does not export) and would put two SYCL runtimes in one process.
# Seeding SYCL_LIBRARY before find_package(Torch) makes torch's own find step
# keep this value. Override with -DKIVI_SYCL_RUNTIME_DIR=<dir>.
if(NOT KIVI_SYCL_RUNTIME_DIR)
  foreach(_p ${CMAKE_PREFIX_PATH})
    if(EXISTS "${_p}/../../lib/libtorch.so")
      get_filename_component(KIVI_SYCL_RUNTIME_DIR "${_p}/../../lib/../../../.." ABSOLUTE)
    endif()
  endforeach()
endif()
if(NOT EXISTS "${KIVI_SYCL_RUNTIME_DIR}/libsycl.so")
  message(FATAL_ERROR "kivi: no libsycl.so in KIVI_SYCL_RUNTIME_DIR=${KIVI_SYCL_RUNTIME_DIR} "
                      "(pass -DKIVI_SYCL_RUNTIME_DIR=<dir of the libsycl torch loads>)")
endif()
set(KIVI_SYCL_RUNTIME_DIR "${KIVI_SYCL_RUNTIME_DIR}" CACHE PATH "directory with the libsycl.so torch uses")
set(SYCL_LIBRARY "${KIVI_SYCL_RUNTIME_DIR}/libsycl.so" CACHE FILEPATH "SYCL runtime torch uses" FORCE)
set(SYCL_LIBRARY_DIR "${KIVI_SYCL_RUNTIME_DIR}" CACHE FILEPATH "SYCL runtime dir torch uses" FORCE)
message(STATUS "kivi: SYCL runtime from ${KIVI_SYCL_RUNTIME_DIR}")

find_package(Torch REQUIRED)

# Same queue policy as setup.py: submit on torch's XPU stream when the torch
# build ships libc10_xpu (see src/csrc/include/kivi/common.hpp).
get_filename_component(KIVI_TORCH_LIB_DIR "${TORCH_LIBRARY}" DIRECTORY)
find_library(KIVI_C10_XPU c10_xpu PATHS "${KIVI_TORCH_LIB_DIR}" NO_DEFAULT_PATH)

set(KIVI_RPATH "${KIVI_TORCH_LIB_DIR};${KIVI_SYCL_RUNTIME_DIR}")

set(KIVI_CSRC ${CMAKE_CURRENT_SOURCE_DIR}/src/csrc)
add_library(kivi_core SHARED
  ${KIVI_CSRC}/kernels/quantize_kernels.cpp
  ${KIVI_CSRC}/kernels/dequantize_kernels.cpp
  ${KIVI_CSRC}/kernels/roundtrip_kernels.cpp
  ${KIVI_CSRC}/ops/kv_cache_ops.cpp
  ${KIVI_CSRC}/cpu/fused_attention.cpp
  ${KIVI_CSRC}/library/register_ops.cpp
  ${KIVI_CSRC}/engine/arena.cpp
  ${KIVI_CSRC}/engine/dequant_cpu.cpp
  ${KIVI_CSRC}/engine/engine.cpp
  ${KIVI_CSRC}/engine/registry.cpp
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

# kivi-run: the Python-free runtime (Phase 5 Route A). Links libkivi_core,
# which registers the kivi:: ops the AOTInductor package calls, and holds the
# engine registry the package's layer_attention calls resolve id 0 against.
# Tokenizer: plain C++ (no torch, no SYCL), so it is its own static library.
add_library(kivi_tokenizer STATIC src/runtime/tokenizer/tokenizer.cpp)
target_include_directories(kivi_tokenizer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/runtime
                           ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
set_target_properties(kivi_tokenizer PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(kivi-run
  src/runtime/main.cpp
  src/runtime/model.cpp
  src/runtime/safetensors.cpp
)
target_include_directories(kivi-run PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/runtime
                           ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
target_link_libraries(kivi-run PRIVATE kivi_core kivi_tokenizer)
set_target_properties(kivi-run PROPERTIES
  BUILD_RPATH "${KIVI_RPATH};$<TARGET_FILE_DIR:kivi_core>")

enable_testing()
add_executable(kivi_native_tests tests/native/test_core.cpp)
target_link_libraries(kivi_native_tests PRIVATE kivi_core)
set_target_properties(kivi_native_tests PROPERTIES
  BUILD_RPATH "${KIVI_RPATH};$<TARGET_FILE_DIR:kivi_core>")
add_test(NAME kivi_native_tests COMMAND kivi_native_tests)

add_executable(kivi_arena_tests tests/native/test_arena.cpp)
target_link_libraries(kivi_arena_tests PRIVATE kivi_core)
set_target_properties(kivi_arena_tests PROPERTIES
  BUILD_RPATH "${KIVI_RPATH};$<TARGET_FILE_DIR:kivi_core>")
add_test(NAME kivi_arena_tests COMMAND kivi_arena_tests)

# Runtime golden test (Phase 5 A.4): kivi-run must reproduce Python's greedy
# tokens. Artifacts (GBs) and the golden file are generated, not checked in:
#   python -m kivi_sycl.export --model <id> --out artifacts/tinyllama
#   python tools/runtime_golden.py --artifacts artifacts/tinyllama --model <id> \
#       --out build-native/golden/tinyllama.json
# The test is registered only when both exist (re-run cmake after creating them).
set(KIVI_GOLDEN_ARTIFACTS "${CMAKE_CURRENT_SOURCE_DIR}/artifacts/tinyllama" CACHE PATH
    "artifact directory for the kivi-run golden test")
set(KIVI_GOLDEN_FILE "${CMAKE_CURRENT_BINARY_DIR}/golden/tinyllama.json" CACHE FILEPATH
    "golden tokens (tools/runtime_golden.py) for the kivi-run golden test")
if(EXISTS "${KIVI_GOLDEN_ARTIFACTS}/kivi_config.json" AND EXISTS "${KIVI_GOLDEN_FILE}")
  add_test(NAME kivi_run_golden
           COMMAND kivi-run --model "${KIVI_GOLDEN_ARTIFACTS}" --golden "${KIVI_GOLDEN_FILE}")
  message(STATUS "kivi: runtime golden test on ${KIVI_GOLDEN_ARTIFACTS}")
endif()

# Tokenizer conformance (Phase 5 A.5): tools/tokenizer_golden.py corpus.
add_executable(test_tokenizer tests/native/test_tokenizer.cpp)
target_link_libraries(test_tokenizer PRIVATE kivi_tokenizer)
file(GLOB KIVI_TOKENIZER_GOLDEN "${CMAKE_CURRENT_BINARY_DIR}/golden/*_tokenizer.json")
foreach(_g ${KIVI_TOKENIZER_GOLDEN})
  get_filename_component(_name "${_g}" NAME_WE)            # e.g. tinyllama_tokenizer
  string(REPLACE "_tokenizer" "" _model "${_name}")
  set(_tj "${CMAKE_CURRENT_SOURCE_DIR}/artifacts/${_model}/tokenizer.json")
  if(EXISTS "${_tj}")
    add_test(NAME tokenizer_${_model} COMMAND test_tokenizer "${_tj}" "${_g}")
    message(STATUS "kivi: tokenizer conformance test on ${_model}")
  endif()
endforeach()
