#!/bin/bash
# Run the test suite WITHOUT the oneAPI environment: the extension loads
# against the SYCL runtime bundled in the torch XPU wheel, and a sourced
# oneAPI on LD_LIBRARY_PATH can shadow it. Extra args go to pytest.
#   bash scripts/dev_test.sh                 # everything
#   bash scripts/dev_test.sh tests/unit -x
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
ENV_DIR="${KIVI_ENV:-$HOME/miniforge3/envs/intel_xpu}"
# torch.compile / AOTInductor stage build files in TORCHINDUCTOR_CACHE_DIR,
# default /tmp — a RAM-backed tmpfs on many Linux desktops (here: compiles
# of 1B models filled it and ran the machine out of memory). Default both to
# a disk path; it must not contain spaces (Inductor does not quote paths in
# its g++ command line).
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$HOME/.cache/kivi_inductor}"
export TMPDIR="${KIVI_TMPDIR:-$HOME/.cache/kivi_tmp}"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TMPDIR"
exec env -u LD_LIBRARY_PATH "$ENV_DIR/bin/python" -m pytest -q -p no:cacheprovider "${@:-tests/}"
