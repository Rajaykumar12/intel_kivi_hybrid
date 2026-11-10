#!/bin/bash
# ==========================================================================
# Build the Python-free native targets (CMakeLists.txt): libkivi_core.so and
# the C++ test program, then run the tests.
#
# Compiler selection matches scripts/dev_build.sh: a specific oneAPI icpx
# (KIVI_ICPX_VER, default 2026.1) rather than setvars.sh, and the torch env
# (KIVI_ENV) only to locate torch's CMake config. Python is used at
# configure time only — nothing built here links or loads it.
#
#   bash scripts/build_native.sh            # configure + build + ctest
#   KIVI_AOT_DEVICE=adl-p bash scripts/build_native.sh
# ==========================================================================
set -e
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
ICPX_VER="${KIVI_ICPX_VER:-2026.1}"
ENV_DIR="${KIVI_ENV:-$HOME/miniforge3/envs/intel_xpu}"
BUILD_DIR="${KIVI_NATIVE_BUILD_DIR:-build-native}"

TORCH_CMAKE="$(env -u LD_LIBRARY_PATH "$ENV_DIR/bin/python" -c 'import torch; print(torch.utils.cmake_prefix_path)')"

# shellcheck disable=SC1090
source "/opt/intel/oneapi/compiler/${ICPX_VER}/env/vars.sh" >/dev/null 2>&1
export LD_LIBRARY_PATH="$ENV_DIR/lib:$LD_LIBRARY_PATH"
icpx --version | head -1

cmake -S . -B "$BUILD_DIR" -DCMAKE_CXX_COMPILER=icpx -DCMAKE_PREFIX_PATH="$TORCH_CMAKE" \
      -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" -j"$(nproc)"

# Run the tests without the oneAPI env: the binaries must load against the
# SYCL runtime bundled with torch, exactly as a deployed runtime would.
cd "$BUILD_DIR"
env -u LD_LIBRARY_PATH ctest --output-on-failure
