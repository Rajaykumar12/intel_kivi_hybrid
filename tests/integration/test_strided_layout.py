"""Quantize kernels writing into arena views (KiviLayout).

Phase 1 step 9 (docs/optimization_roadmap/phase1_memory_and_metadata.md §B.2):
a flush chunk is written straight into a slice of a preallocated
[B,H,D,cap/4] key arena / [B,H,cap,D/4] value arena, and must land exactly
where a contiguous quantize of the same chunk would put it.
"""

import pytest

torch = pytest.importorskip("torch")
if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
    pytest.skip("XPU device not available", allow_module_level=True)

from kivi_sycl import _C as kivi_native  # noqa: E402

B, H, D, G = 2, 3, 64, 32
CAP = 8 * G          # arena token capacity


def _alloc(where, shape, dtype):
    if where == "usm":
        return kivi_native.alloc_usm_host(list(shape), dtype)
    return torch.empty(shape, dtype=dtype, device="xpu")


def _contig_keys(x_trans, meta):
    b, h, d, L = x_trans.shape
    p = torch.empty((b, h, d, L // 4), dtype=torch.uint8, device="xpu")
    s = torch.empty((b, h, d, L // G), dtype=meta, device="xpu")
    z = torch.empty_like(s)
    kivi_native.quantize_keys(x_trans, p, s, z, G)
    return p.cpu(), s.cpu(), z.cpu()


def _contig_values(x, meta):
    b, h, L, d = x.shape
    p = torch.empty((b, h, L, d // 4), dtype=torch.uint8, device="xpu")
    s = torch.empty((b, h, L, d // G), dtype=meta, device="xpu")
    z = torch.empty_like(s)
    kivi_native.quantize_values(x, p, s, z, d, G)
    return p.cpu(), s.cpu(), z.cpu()


@pytest.mark.parametrize("where", ["xpu", "usm"])
@pytest.mark.parametrize("meta", [torch.float32, torch.float16])
def test_key_chunks_into_arena_match_contiguous(where, meta):
    torch.manual_seed(0)
    pk = _alloc(where, (B, H, D, CAP // 4), torch.uint8)
    sk = _alloc(where, (B, H, D, CAP // G), meta)
    zk = _alloc(where, (B, H, D, CAP // G), meta)
    pk.zero_(); sk.zero_(); zk.zero_()

    chunks = [(0, 2 * G), (2 * G, 3 * G)]   # (token offset, length)
    refs = []
    for off, L in chunks:
        x = torch.randn(B, H, D, L, device="xpu").contiguous()
        kivi_native.quantize_keys(
            x,
            pk[..., off // 4:(off + L) // 4],
            sk[..., off // G:(off + L) // G],
            zk[..., off // G:(off + L) // G],
            G)
        refs.append((off, L, _contig_keys(x, meta)))

    pk_c, sk_c, zk_c = pk.cpu(), sk.cpu(), zk.cpu()
    for off, L, (p, s, z) in refs:
        assert torch.equal(pk_c[..., off // 4:(off + L) // 4], p)
        assert torch.equal(sk_c[..., off // G:(off + L) // G], s)
        assert torch.equal(zk_c[..., off // G:(off + L) // G], z)
    # capacity beyond the written chunks is untouched
    used = sum(L for _, L in chunks)
    assert (pk_c[..., used // 4:] == 0).all()
    assert (sk_c[..., used // G:] == 0).all()


@pytest.mark.parametrize("where", ["xpu", "usm"])
@pytest.mark.parametrize("meta", [torch.float32, torch.float16])
def test_value_chunks_into_arena_match_contiguous(where, meta):
    torch.manual_seed(1)
    pv = _alloc(where, (B, H, CAP, D // 4), torch.uint8)
    sv = _alloc(where, (B, H, CAP, D // G), meta)
    zv = _alloc(where, (B, H, CAP, D // G), meta)
    pv.zero_(); sv.zero_(); zv.zero_()

    chunks = [(0, G), (G, 4 * G)]
    refs = []
    for off, L in chunks:
        x = torch.randn(B, H, L, D, device="xpu").contiguous()
        kivi_native.quantize_values(
            x, pv[:, :, off:off + L], sv[:, :, off:off + L], zv[:, :, off:off + L],
            D, G)
        refs.append((off, L, _contig_values(x, meta)))

    pv_c, sv_c, zv_c = pv.cpu(), sv.cpu(), zv.cpu()
    for off, L, (p, s, z) in refs:
        assert torch.equal(pv_c[:, :, off:off + L], p)
        assert torch.equal(sv_c[:, :, off:off + L], s)
        assert torch.equal(zv_c[:, :, off:off + L], z)
    used = sum(L for _, L in chunks)
    assert (pv_c[:, :, used:] == 0).all()


def test_async_strided_write_into_usm_arena():
    torch.manual_seed(2)
    pk = kivi_native.alloc_usm_host([B, H, D, CAP // 4], torch.uint8)
    sk = kivi_native.alloc_usm_host([B, H, D, CAP // G], torch.float16)
    zk = kivi_native.alloc_usm_host([B, H, D, CAP // G], torch.float16)
    x = torch.randn(B, H, D, 2 * G, device="xpu").contiguous()
    ev = kivi_native.quantize_keys_async(
        x, pk[..., G // 4:3 * G // 4], sk[..., 1:3], zk[..., 1:3], G)
    ev.wait()
    p, s, z = _contig_keys(x, torch.float16)
    assert torch.equal(pk[..., G // 4:3 * G // 4], p)
    assert torch.equal(sk[..., 1:3], s)


def test_view_with_noncontiguous_inner_dim_rejected():
    x = torch.randn(B, H, D, G, device="xpu").contiguous()
    pk = torch.empty((B, H, D, 2 * G // 4), dtype=torch.uint8, device="xpu")
    sk = torch.empty((B, H, D, 2), dtype=torch.float32, device="xpu")
    zk = torch.empty_like(sk)
    with pytest.raises(RuntimeError, match="contiguous"):
        kivi_native.quantize_keys(x, pk[..., ::2], sk[..., :1], zk[..., :1], G)


def test_zeros_strides_must_match_scales():
    x = torch.randn(B, H, D, G, device="xpu").contiguous()
    pk = torch.empty((B, H, D, CAP // 4), dtype=torch.uint8, device="xpu")
    sk = torch.empty((B, H, D, CAP // G), dtype=torch.float32, device="xpu")
    zk = torch.empty((B, H, D, 1), dtype=torch.float32, device="xpu")  # contiguous, different strides
    with pytest.raises(RuntimeError, match="same shape and strides"):
        kivi_native.quantize_keys(x, pk[..., :G // 4], sk[..., :1], zk, G)
