#!/bin/bash
# ==========================================================================
# Warning audit for the project's own C++/SYCL sources.
#
# setup.py builds with -w, because torch's extension builder passes the
# torch headers with -I and they would otherwise bury the build log. This
# script syntax-checks our sources with -Wall -Wextra, treating the torch
# and Python headers as system headers, and fails if any warning comes from
# src/csrc. Same compiler selection as scripts/dev_build.sh.
#   bash scripts/check_warnings.sh
# ==========================================================================
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
ICPX_VER="${KIVI_ICPX_VER:-2026.1}"
ENV_DIR="${KIVI_ENV:-$HOME/miniforge3/envs/intel_xpu}"
# shellcheck disable=SC1090
source "/opt/intel/oneapi/compiler/${ICPX_VER}/env/vars.sh" >/dev/null 2>&1
TORCH_INC="$("$ENV_DIR/bin/python" -c 'import torch.utils.cpp_extension as c; print(" ".join(c.include_paths()))' 2>/dev/null)"
PY_INC="$("$ENV_DIR/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["include"])')"
SYS=()
for d in $TORCH_INC "$PY_INC"; do SYS+=(-isystem "$d"); done

status=0
for f in $(find src/csrc -name '*.cpp' | sort); do
    out="$(icpx -fsycl -std=c++17 -fsyntax-only -Wall -Wextra \
        -DKIVI_USE_TORCH_XPU_QUEUE -DTORCH_EXTENSION_NAME=_C \
        -Isrc/csrc -Isrc/csrc/include "${SYS[@]}" "$f" 2>&1)" || { echo "$out"; status=1; continue; }
    w="$(grep -E '^src/csrc/.*warning:' <<<"$out")"
    if [ -n "$w" ]; then echo "$w"; status=1; else echo "clean  $f"; fi
done
exit $status
