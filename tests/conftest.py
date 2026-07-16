import pytest


def pytest_collection_modifyitems(config, items):
    """Skip tests/integration/* automatically when no XPU device is present."""
    try:
        import torch
        xpu_available = torch.xpu.is_available()
    except Exception:
        xpu_available = False

    if xpu_available:
        return

    skip_xpu = pytest.mark.skip(reason="requires an Intel XPU device")
    for item in items:
        if "integration" in str(item.fspath):
            item.add_marker(skip_xpu)
