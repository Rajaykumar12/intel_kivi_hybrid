"""IPEX (Intel Extension for PyTorch) detection.

IPEX registers the XPU backend with PyTorch. It's a soft dependency — not
on standard PyPI — so we check at runtime and give users a clear install
command instead of a cryptic ImportError.
"""

try:
    import intel_extension_for_pytorch as ipex  # noqa: F401
except ImportError:
    _ipex_available = False
else:
    _ipex_available = True

__all__ = ["ensure_ipex"]


def ensure_ipex():
    """Raise a helpful error if IPEX is not installed."""
    if not _ipex_available:
        raise RuntimeError(
            "Intel Extension for PyTorch (IPEX) is required but not installed.\n"
            "IPEX is not on standard PyPI — install it with:\n\n"
            "  pip install intel-extension-for-pytorch "
            "--extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/\n\n"
            "Or via conda:\n\n"
            "  conda install intel-extension-for-pytorch -c intel\n"
        )
