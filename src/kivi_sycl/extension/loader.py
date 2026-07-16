"""Single import seam for the compiled SYCL extension (kivi_sycl._C).

Every module that needs the native kernels imports `kivi_native` from here
instead of importing `kivi_sycl._C` directly, so there is exactly one place
that needs to explain a missing build.
"""

try:
    from kivi_sycl import _C as kivi_native
except ImportError as exc:
    raise ImportError(
        "kivi_sycl._C (the compiled SYCL extension) was not found.\n"
        "Build it in-place with:\n\n"
        "  source /opt/intel/oneapi/setvars.sh\n"
        "  pip install . --no-build-isolation\n\n"
        "or run scripts/build_wheel.sh to produce an installable wheel."
    ) from exc

__all__ = ["kivi_native"]
