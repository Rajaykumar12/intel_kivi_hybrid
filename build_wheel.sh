#!/bin/bash
# ==========================================================================
# Build pre-built wheel for kivi-sycl
# ==========================================================================
# This compiles the SYCL kernels and packages everything into a .whl file.
# Users who install this wheel do NOT need the oneAPI compiler — only the
# Intel GPU driver + SYCL runtime (shipped with the driver).
#
# Usage:
#   source /opt/intel/oneapi/setvars.sh
#   conda activate intel_xpu
#   bash build_wheel.sh
#
# Output:
#   dist/kivi_sycl-0.1.0-cp310-cp310-linux_x86_64.whl
# ==========================================================================

set -e

echo "=== Building kivi-sycl wheel ==="

# Verify environment
if ! command -v icpx &> /dev/null; then
    echo "ERROR: icpx (Intel DPC++ compiler) not found."
    echo "Run: source /opt/intel/oneapi/setvars.sh"
    exit 1
fi

python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import intel_extension_for_pytorch; print(f'IPEX: {intel_extension_for_pytorch.__version__}')"

# Clean previous builds
rm -rf build/ dist/ *.egg-info/

# Build the wheel (includes compiled .so)
python setup.py bdist_wheel

echo ""
echo "=== Wheel built successfully ==="
ls -lh dist/*.whl
echo ""
echo "Users install with:"
echo "  pip install dist/$(ls dist/)"
echo ""
echo "To upload to PyPI:"
echo "  pip install twine"
echo "  twine upload dist/*"
