"""USM host allocations: kernels read and write CPU tensors with no copies.

Phase 1 step 8 (docs/optimization_roadmap/phase1_memory_and_metadata.md §B.1).
"""

import pytest

torch = pytest.importorskip("torch")
if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
    pytest.skip("XPU device not available", allow_module_level=True)

from kivi_sycl import _C as kivi_native  # noqa: E402

B, H, D, G = 2, 3, 64, 32


def _usm(shape, dtype):
    return kivi_native.alloc_usm_host(list(shape), dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.uint8])
@pytest.mark.parametrize("align", [64, 4096])
def test_alloc_is_cpu_aligned_and_accessible(dtype, align):
    t = kivi_native.alloc_usm_host([B, H, D, 7], dtype, align)
    assert t.device.type == "cpu"
    assert t.dtype == dtype
    assert tuple(t.shape) == (B, H, D, 7)
    assert t.data_ptr() % align == 0
    assert kivi_native.is_kernel_accessible(t)
    # ordinary torch ops work on it
    t.fill_(1)
    assert t.sum().item() == B * H * D * 7


def test_accessibility_classification():
    assert kivi_native.is_kernel_accessible(torch.empty(4, device="xpu"))
    assert not kivi_native.is_kernel_accessible(torch.empty(4))


def test_alloc_rejects_bad_alignment():
    with pytest.raises(RuntimeError, match="power of two"):
        kivi_native.alloc_usm_host([4], torch.float32, 48)


def test_zero_sized_alloc():
    t = kivi_native.alloc_usm_host([0, 4], torch.float32)
    assert t.numel() == 0 and kivi_native.is_kernel_accessible(t)


@pytest.mark.parametrize("meta", [torch.float32, torch.float16])
def test_quantize_keys_usm_matches_xpu(meta):
    """Input and all outputs in USM host memory: same bytes as the XPU path,
    and the CPU sees the kernel's writes with no .to('cpu')."""
    torch.manual_seed(0)
    x_cpu = torch.randn(B, H, D, 2 * G)

    x_usm = _usm(x_cpu.shape, torch.float32)
    x_usm.copy_(x_cpu)
    p_usm = _usm((B, H, D, 2 * G // 4), torch.uint8)
    s_usm = _usm((B, H, D, 2), meta)
    z_usm = _usm((B, H, D, 2), meta)
    kivi_native.quantize_keys(x_usm, p_usm, s_usm, z_usm, G)

    x_x = x_cpu.to("xpu")
    p_x = torch.empty((B, H, D, 2 * G // 4), dtype=torch.uint8, device="xpu")
    s_x = torch.empty((B, H, D, 2), dtype=meta, device="xpu")
    z_x = torch.empty_like(s_x)
    kivi_native.quantize_keys(x_x, p_x, s_x, z_x, G)

    assert torch.equal(p_usm, p_x.cpu())
    assert torch.equal(s_usm, s_x.cpu())
    assert torch.equal(z_usm, z_x.cpu())


def test_quantize_values_async_usm_outputs_visible_after_wait():
    torch.manual_seed(1)
    L = 3 * G
    x = torch.randn(B, H, L, D).to("xpu")              # XPU input, USM outputs
    p = _usm((B, H, L, D // 4), torch.uint8)
    s = _usm((B, H, L, D // G), torch.float16)
    z = _usm((B, H, L, D // G), torch.float16)
    p.zero_(); s.zero_()
    ev = kivi_native.quantize_values_async(x, p, s, z, D, G)
    ev.wait()
    assert (s > 0).all()

    out = _usm((B, H, L, D), torch.float32)
    kivi_native.dequantize_values(p, s, z, out, D, G)
    err = (out - x.cpu()).abs().max().item()
    assert err <= s.float().max().item() / 2 + 1e-2


def test_plain_cpu_tensor_rejected():
    x = torch.randn(B, H, D, G)                         # malloc-backed
    p = _usm((B, H, D, G // 4), torch.uint8)
    s = _usm((B, H, D, 1), torch.float32)
    z = _usm((B, H, D, 1), torch.float32)
    with pytest.raises(RuntimeError, match="not backed by USM"):
        kivi_native.quantize_keys(x, p, s, z, G)


def _rss_mib():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    pytest.skip("no /proc/self/status")


def test_allocations_are_freed():
    """Allocate, fully touch and drop 16 x 64 MiB. If the deleter leaked,
    resident memory would grow by ~1 GiB; it must stay roughly flat."""
    before = _rss_mib()
    for _ in range(16):
        t = kivi_native.alloc_usm_host([16 * 1024 * 1024], torch.float32)
        t.fill_(1.0)                                    # touch every page
        del t
    grown = _rss_mib() - before
    assert grown < 256, f"RSS grew {grown:.0f} MiB — USM deleter is leaking"
