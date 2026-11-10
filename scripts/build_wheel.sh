#!/bin/bash
# ==========================================================================
# Build a pre-built wheel for kivi-sycl
# ==========================================================================
# Compiles the SYCL kernels and packages everything into a .whl file.
# Users who install this wheel do NOT need the oneAPI compiler — only the
# Intel GPU driver and the SYCL runtime bundled with the torch XPU wheel.
#
# Usage:
#   bash scripts/build_wheel.sh                         # portable (JIT) device code
#   KIVI_AOT_DEVICE=adl-p bash scripts/build_wheel.sh   # + AOT code for Alder Lake Iris Xe
#
# The compiler environment matches scripts/dev_build.sh (KIVI_ICPX_VER,
# KIVI_ENV): a oneAPI compiler compatible with the SYCL runtime bundled in
# torch, not whatever /opt/intel/oneapi/setvars.sh resolves to.
#
# KIVI_AOT_DEVICE takes any ocloc -device value (device, family such as
# xe-lp, or PCI id); see setup.py. AOT removes the first-use JIT compile of
# the kernels (~100 ms per process on a cold driver cache) and keeps a
# generic SPIR-V copy so the wheel still runs on other Intel GPUs.
#
# Output: dist/kivi_sycl-<version>-cp3XX-cp3XX-linux_x86_64.whl
# ==========================================================================

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
ICPX_VER="${KIVI_ICPX_VER:-2026.1}"
ENV_DIR="${KIVI_ENV:-$HOME/miniforge3/envs/intel_xpu}"

echo "=== Building kivi-sycl wheel ==="

# shellcheck disable=SC1090
source "/opt/intel/oneapi/compiler/${ICPX_VER}/env/vars.sh" >/dev/null 2>&1 || {
    echo "ERROR: /opt/intel/oneapi/compiler/${ICPX_VER} not found (set KIVI_ICPX_VER)."; exit 1; }
export LD_LIBRARY_PATH="$ENV_DIR/lib:$LD_LIBRARY_PATH"
export PATH="$ENV_DIR/bin:$PATH"
command -v icpx >/dev/null || { echo "ERROR: icpx not found."; exit 1; }
icpx --version | head -1

python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import intel_extension_for_pytorch as i; print(f'IPEX: {i.__version__}')" 2>/dev/null \
    || echo "IPEX: not installed (not required with torch >= 2.4 XPU builds)"
echo "AOT device: ${KIVI_AOT_DEVICE:-none (portable SPIR-V, JIT at first use)}"

rm -rf build/ dist/ ./*.egg-info/ src/*.egg-info/

CXX=icpx CC=icx python -m pip wheel . --no-build-isolation --no-deps -w dist

echo ""
echo "=== Wheel built successfully ==="
ls -lh dist/*.whl
echo ""
echo "Users install with:"
echo "  pip install dist/$(ls dist/)"
