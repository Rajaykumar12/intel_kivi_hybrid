#!/bin/bash
# ==========================================================================
# Clean in-place rebuild of kivi_sycl._C for development.
#
# Uses a specific oneAPI compiler instead of /opt/intel/oneapi/setvars.sh:
# setvars resolves to the newest installed oneAPI, whose UR loader can be
# newer than the SYCL runtime bundled in the torch XPU wheel — then
# `import torch` (which setup.py needs) fails with
#   undefined symbol: urKernelSetArgLocal, version LIBUR_LOADER_0.12
# The torch env's own libs are put first so torch imports during the build.
#
#   KIVI_ICPX_VER=2026.1 KIVI_ENV=$HOME/miniforge3/envs/intel_xpu bash scripts/dev_build.sh
# ==========================================================================
set -e
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
ICPX_VER="${KIVI_ICPX_VER:-2026.1}"
ENV_DIR="${KIVI_ENV:-$HOME/miniforge3/envs/intel_xpu}"

# shellcheck disable=SC1090
source "/opt/intel/oneapi/compiler/${ICPX_VER}/env/vars.sh" >/dev/null 2>&1
export LD_LIBRARY_PATH="$ENV_DIR/lib:$LD_LIBRARY_PATH"
export PATH="$ENV_DIR/bin:$PATH"
icpx --version | head -1

rm -rf build/   # stale objects survive header-only edits otherwise
CXX=icpx CC=icx python -m pip install -e . --no-build-isolation --force-reinstall --no-deps -q
ls -la src/kivi_sycl/*.so
