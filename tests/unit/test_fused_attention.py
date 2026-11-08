"""CPU fused decode attention vs a dequantize-then-attend reference.

Needs the compiled extension but no XPU: the packed history here is random
codes and scales built on the CPU, in exactly the layout the arena stores.
"""

import os
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
try:
    from kivi_sycl import _C as kivi_native
except ImportError:
    pytest.skip("compiled extension not built", allow_module_level=True)
if not hasattr(kivi_native, "fused_decode_attention"):
    pytest.skip("extension built without fused attention", allow_module_level=True)

SHIFTS = torch.tensor([0, 2, 4, 6], dtype=torch.uint8)


def _random_history(B, H, T, D, G, meta_dtype, cap=None):
    """Packed history in arena layout; with cap > T, views into a larger store."""
    cap = cap or T
    pk = torch.randint(0, 256, (B, H, D, cap // 4), dtype=torch.uint8)[..., :T // 4]
    sk = (torch.rand(B, H, D, cap // G) * 0.5 + 0.01).to(meta_dtype)[..., :T // G]
    zk = (torch.randn(B, H, D, cap // G)).to(meta_dtype)[..., :T // G]
    pv = torch.randint(0, 256, (B, H, cap, D // 4), dtype=torch.uint8)[:, :, :T]
    sv = (torch.rand(B, H, cap, D // G) * 0.5 + 0.01).to(meta_dtype)[:, :, :T]
    zv = (torch.randn(B, H, cap, D // G)).to(meta_dtype)[:, :, :T]
    return pk, sk, zk, pv, sv, zv


def _dequant(pk, sk, zk, pv, sv, zv, G):
    B, H, D, T4 = pk.shape
    T = T4 * 4
    ck = ((pk.unsqueeze(-1) >> SHIFTS) & 3).reshape(B, H, D, T).float()
    k = ck * sk.float().repeat_interleave(G, -1) + zk.float().repeat_interleave(G, -1)
    cv = ((pv.unsqueeze(-1) >> SHIFTS) & 3).reshape(B, H, T, D).float()
    v = cv * sv.float().repeat_interleave(G, -1) + zv.float().repeat_interleave(G, -1)
    return k.transpose(2, 3), v                          # both [B, H, T, D]


def _reference(q, k_res, v_res, hist, scaling, G):
    k, v = k_res.float(), v_res.float()
    if hist is not None:
        hk, hv = _dequant(*hist, G)
        k, v = torch.cat([hk, k], 2), torch.cat([hv, v], 2)
    n_rep = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(n_rep, 1)
    v = v.repeat_interleave(n_rep, 1)
    p = torch.softmax(q.float() @ k.transpose(2, 3) * scaling, -1)
    return (p @ v).transpose(1, 2)                       # [B, 1, Hq, D]


def _run(B=1, Hq=4, Hkv=2, T=320, NR=17, D=64, G=32, dtype=torch.float32,
         meta=torch.float16, cap=None, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(B, Hq, 1, D).to(dtype)
    k_res = torch.randn(B, Hkv, NR, D).to(dtype)
    v_res = torch.randn(B, Hkv, NR, D).to(dtype)
    scaling = D ** -0.5
    if T:
        hist = _random_history(B, Hkv, T, D, G, meta, cap)
    else:
        e8, em = torch.empty(0, dtype=torch.uint8), torch.empty(0, dtype=meta)
        hist = None
    args = hist if hist is not None else (e8, em, em, e8, em, em)
    out = kivi_native.fused_decode_attention(q, k_res, v_res, *args, scaling, G)
    ref = _reference(q, k_res, v_res, hist, scaling, G)
    return out, ref


@pytest.mark.parametrize("Hq,Hkv", [(2, 2), (4, 2), (8, 2), (14, 2)])
def test_matches_reference_mha_and_gqa(Hq, Hkv):
    out, ref = _run(Hq=Hq, Hkv=Hkv)
    assert out.shape == ref.shape and out.dtype == torch.float32
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("T,NR", [(0, 9), (256, 0), (32, 1), (512 + 96, 64), (2048, 100)])
def test_history_and_residual_edge_lengths(T, NR):
    out, ref = _run(T=T, NR=NR)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-5)


def test_strided_arena_views_and_batch():
    out, ref = _run(B=2, T=384, cap=1024)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("G", [8, 16, 64])
def test_group_sizes(G):
    out, ref = _run(G=G, T=4 * 64)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-5)


def test_fp16_model_dtype():
    out, ref = _run(dtype=torch.float16)
    assert out.dtype == torch.float16
    torch.testing.assert_close(out.float(), ref, rtol=2e-3, atol=2e-3)


def test_fp32_metadata():
    out, ref = _run(meta=torch.float32)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-5)


def test_rejects_prefill_queries():
    q = torch.randn(1, 2, 3, 64)
    r = torch.randn(1, 2, 4, 64)
    e8, em = torch.empty(0, dtype=torch.uint8), torch.empty(0)
    with pytest.raises(RuntimeError, match="decode step"):
        kivi_native.fused_decode_attention(q, r, r, e8, em, em, e8, em, em, 0.125, 32)


def test_generic_code_path_matches_too():
    """KIVI_FUSED_GENERIC=1 forces the portable build of the hot loops."""
    code = ("import torch, sys; sys.path.insert(0, %r); import test_fused_attention as t; "
            "from kivi_sycl import _C; assert not _C.fused_attention_uses_avx2(); "
            "out, ref = t._run(T=640, NR=33, Hq=8); "
            "torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-5); print('ok')"
            % os.path.dirname(__file__))
    env = dict(os.environ, KIVI_FUSED_GENERIC="1")
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr[-2000:]
