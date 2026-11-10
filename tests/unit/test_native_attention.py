"""kivi::Engine::attend vs the Python "kivi" attention (Phase 5 S1.5).

Reference path: Python's own dequantize of the engine's history + residual,
then transformers' sdpa_attention_forward — must be bit-identical. Fused
decode path: within fp32 tolerance of it. Chunked prefill without a mask:
the engine's bottom-right causal mask, checked against an explicit mask.
Runs on the CPU engine (reference quantize) — no XPU needed.
"""

import os
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
try:
    from kivi_sycl import _C
except ImportError:
    pytest.skip("compiled extension not built", allow_module_level=True)
if not hasattr(_C, "Engine") or not hasattr(_C.Engine, "attend"):
    pytest.skip("extension built without Engine.attend", allow_module_level=True)

from transformers.integrations.sdpa_attention import sdpa_attention_forward  # noqa: E402

from kivi_sycl.cache.manager import KiviCache  # noqa: E402

D, G, R = 64, 32, 64


class _Module:                      # what sdpa_attention_forward reads off the module
    is_causal = True

    def __init__(self, n_rep):
        self.num_key_value_groups = n_rep


def _engine(Hkv, steps, prefill=70, seed=0):
    """Engine (CPU) with history + residual after `prefill` + `steps` tokens."""
    torch.manual_seed(seed)
    e = _C.Engine(num_layers=1, head_dim=D, residual_length=R, group_size=[G], device="cpu")
    for n in [prefill] + [1] * steps:
        e.update(0, torch.randn(1, Hkv, n, D), torch.randn(1, Hkv, n, D))
    return e


def _python_reference(e, q, mask=None):
    """The Python "kivi" attention's reference computation on the same state."""
    py = KiviCache(num_layers=1, head_dim=D, num_kv_heads=q.shape[1], device="cpu",
                   group_size=G, residual_length=R)
    py._cache_dtype = torch.float32
    k, v = e.residual(0)
    h = e.history(0)
    if h is not None:
        k = torch.cat([py._dequant_keys_cpu(*h[:3]), k], 2)
        v = torch.cat([py._dequant_values_cpu(*h[3:]), v], 2)
    out, _ = sdpa_attention_forward(_Module(q.shape[1] // k.shape[1]), q, k, v, mask,
                                    scaling=D ** -0.5)
    return out


@pytest.mark.parametrize("Hq,Hkv", [(2, 2), (8, 2), (14, 2)])
def test_reference_path_bit_identical_decode(Hq, Hkv):
    e = _engine(Hkv, steps=40)
    assert e.history(0) is not None
    q = torch.randn(1, Hq, 1, D)
    assert torch.equal(e.attend(0, q, D ** -0.5, None, False), _python_reference(e, q))
    assert e.reference_calls == 1 and e.fused_calls == 0


def test_reference_path_bit_identical_prefill():
    torch.manual_seed(1)
    e = _C.Engine(num_layers=1, head_dim=D, residual_length=R, group_size=[G], device="cpu")
    k, v = torch.randn(1, 2, 20, D), torch.randn(1, 2, 20, D)
    e.update(0, k, v)                               # prefill: no history yet
    q = torch.randn(1, 8, 20, D)
    assert torch.equal(e.attend(0, q, D ** -0.5), _python_reference(e, q))


@pytest.mark.parametrize("Hq,Hkv", [(2, 2), (14, 2)])
def test_fused_decode_matches_reference(Hq, Hkv):
    e = _engine(Hkv, steps=40)
    q = torch.randn(1, Hq, 1, D)
    fused = e.attend(0, q, D ** -0.5)
    assert e.fused_calls == 1
    torch.testing.assert_close(fused, _python_reference(e, q), rtol=1e-4, atol=1e-5)
    # an all-visible HF mask still takes the fused path
    kv_len = e.seq_length(0)
    e.attend(0, q, D ** -0.5, torch.ones(1, 1, 1, kv_len, dtype=torch.bool))
    assert e.fused_calls == 2


def test_padding_mask_takes_reference_path():
    e = _engine(2, steps=40)
    q = torch.randn(1, 8, 1, D)
    kv_len = e.seq_length(0)
    mask = torch.ones(1, 1, 1, kv_len, dtype=torch.bool)
    mask[..., :5] = False                           # left padding
    out = e.attend(0, q, D ** -0.5, mask)
    assert e.fused_calls == 0 and e.reference_calls == 1
    assert torch.equal(out, _python_reference(e, q, mask))


def test_chunked_prefill_builds_bottom_right_causal_mask():
    """History exists, then a 5-token chunk arrives with no mask: token t of
    the chunk must see every earlier token and chunk tokens 0..t."""
    e = _engine(2, steps=10)
    T = 5
    e.update(0, torch.randn(1, 2, T, D), torch.randn(1, 2, T, D))
    q = torch.randn(1, 8, T, D)
    kv_len = e.seq_length(0)
    explicit = torch.arange(kv_len)[None, :] <= (torch.arange(T)[:, None] + kv_len - T)
    got = e.attend(0, q, D ** -0.5)
    assert torch.equal(got, _python_reference(e, q, explicit))
