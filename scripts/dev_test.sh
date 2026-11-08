#!/bin/bash
# Run the test suite WITHOUT the oneAPI environment: the extension loads
# against the SYCL runtime bundled in the torch XPU wheel, and a sourced
# oneAPI on LD_LIBRARY_PATH can shadow it. Extra args go to pytest.
#   bash scripts/dev_test.sh                 # everything
#   bash scripts/dev_test.sh tests/unit -x
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
ENV_DIR="${KIVI_ENV:-$HOME/miniforge3/envs/intel_xpu}"
exec env -u LD_LIBRARY_PATH "$ENV_DIR/bin/python" -m pytest -q -p no:cacheprovider "${@:-tests/}"
