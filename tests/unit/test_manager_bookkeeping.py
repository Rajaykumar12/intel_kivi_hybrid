"""CPU-only KiviCache bookkeeping tests — no XPU, IPEX, or compiled
extension required.

The native kernels are replaced with an exact torch reference
implementation of the asymmetric 2-bit round-trip (plus an identity
variant), so these tests isolate the cache state machine itself:
new-token extraction, flush ordering, residual/history concatenation,
and the persistent cache view.

Regression coverage: a model that does NOT mutate the passed Cache in
place (legacy tuple returns, architectures not ported to the Cache API)
previously left the persistent `_cache_view` stale after the first
decode step — the served cache stopped growing and every subsequent
token's KV was silently dropped, degrading generation into gibberish.
"""

import os
import sys
import types

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers.cache_utils import DynamicCache

# Make the src-layout package importable when running from the repo, and
# stub the compiled extension BEFORE importing kivi_sycl so the import
# seam in extension/loader.py doesn't fail on machines without the build.
_SRC = os.path.join(os.path.dirname(__file__), "..", "..", "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)


def _roundtrip_2bit_reference(inp, out, group_size):
    """Exact torch mirror of the fused SYCL round-trip kernels.

    Both kernels reduce over contiguous runs of `group_size` elements, so
    a flat [-1, group_size] view reproduces their grouping for keys
    ([B,H,D,G] input) and values ([B,H,G,D] input) alike.
    """
    flat = inp.reshape(-1, group_size)
    mn = flat.min(dim=1, keepdim=True).values
    mx = flat.max(dim=1, keepdim=True).values
    scale = (mx - mn) / 3.0 + 1e-10
    q = torch.clamp(torch.round((flat - mn) / scale), 0.0, 3.0)
    out.copy_((q * scale + mn).reshape(inp.shape))


class _FakeNative(types.ModuleType):
    """Stand-in for kivi_sycl._C. `identity=True` makes flushes lossless
    so served tensors can be compared exactly against ground truth."""

    def __init__(self):
        super().__init__("kivi_sycl._C")
        self.identity = False

    def quant_dequant_roundtrip_keys(self, inp, out, group_size):
        if self.identity:
            out.copy_(inp)
        else:
            _roundtrip_2bit_reference(inp, out, group_size)

    def quant_dequant_roundtrip_values(self, inp, out, head_dim, group_size):
        if self.identity:
            out.copy_(inp)
        else:
            _roundtrip_2bit_reference(inp, out, group_size)


_fake_native = _FakeNative()
sys.modules.setdefault("kivi_sycl._C", _fake_native)

import kivi_sycl.cache.manager as manager_module  # noqa: E402
from kivi_sycl.cache.manager import KiviCache, _iter_kv_layers  # noqa: E402

# Override the seams regardless of whether a real build was importable.
manager_module.kivi_native = _fake_native
manager_module.ensure_ipex = lambda: None


class FakeModel:
    """Emulates the HuggingFace forward-pass KV-cache contract.

    mode="in_place":     appends to the passed DynamicCache and returns
                         the same object (modern transformers).
    mode="legacy_tuple": returns a freshly built tuple of full-length
                         (k, v) per layer without touching the passed
                         cache (pre-Cache-API transformers / unported
                         architectures). This is the mode that exposed
                         the stale persistent-view regression.
    """

    def __init__(self, num_layers, num_heads, head_dim, mode):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.mode = mode
        self.total_tokens = 0
        # Ground truth: full FP32 KV history per layer, [1, H, T, D].
        self.truth = [None] * num_layers

    def _new_kv(self, layer, n_new):
        # Token t gets the distinct fill value t (plus small per-position
        # jitter) so dropped/reordered tokens are caught exactly.
        t0 = self.total_tokens
        toks = torch.arange(t0, t0 + n_new, dtype=torch.float32)
        base = toks.view(1, 1, n_new, 1).expand(
            1, self.num_heads, n_new, self.head_dim)
        jitter = torch.randn(1, self.num_heads, n_new, self.head_dim) * 0.01
        k = (base + jitter).contiguous()
        v = (base - jitter).contiguous()
        return k, v

    def forward(self, n_new, past=None):
        new_per_layer = [self._new_kv(i, n_new) for i in range(self.num_layers)]
        for i, (k, v) in enumerate(new_per_layer):
            self.truth[i] = k if self.truth[i] is None else \
                torch.cat([self.truth[i], k], dim=2)
        past_tensors = None
        if past is not None:
            past_tensors = list(_iter_kv_layers(past))
            for i, (pk, _) in enumerate(past_tensors):
                assert pk.shape[2] == self.total_tokens, (
                    f"model was served {pk.shape[2]} past tokens at layer "
                    f"{i}, expected {self.total_tokens}")
        self.total_tokens += n_new

        if self.mode == "in_place":
            cache = past if isinstance(past, DynamicCache) else DynamicCache()
            for i, (k, v) in enumerate(new_per_layer):
                cache.update(k, v, i)
            return cache

        # legacy_tuple: rebuild full-length tensors, never mutate `past`.
        out = []
        for i, (k, v) in enumerate(new_per_layer):
            if past_tensors is not None:
                pk, pv = past_tensors[i]
                k = torch.cat([pk, k], dim=2)
                v = torch.cat([pv, v], dim=2)
            out.append((k, v))
        return tuple(out)


def _run_generation(mode, identity, num_steps=24, prefill=6,
                    G=4, R=8, head_dim=8, layers=2, heads=2):
    _fake_native.identity = identity
    cache = KiviCache(num_layers=layers, head_dim=head_dim,
                      num_kv_heads=heads, device="cpu",
                      group_size=G, residual_length=R)
    model = FakeModel(layers, heads, head_dim, mode)

    cache.add_tokens(model.forward(prefill))
    for _ in range(num_steps):
        past = cache.get_full_cache()
        served_len = next(iter(_iter_kv_layers(past)))[0].shape[2]
        assert served_len == model.total_tokens, (
            f"stale cache served: {served_len} tokens, model has produced "
            f"{model.total_tokens} — generated tokens are being dropped")
        cache.add_tokens(model.forward(1, past))
    return cache, model


@pytest.mark.parametrize("mode", ["in_place", "legacy_tuple"])
def test_served_cache_matches_ground_truth_exactly(mode):
    """With identity 'quantization', the served cache must equal the
    model's FP32 KV history bit-for-bit, in order, for every layer."""
    cache, model = _run_generation(mode, identity=True)
    for i, (k, v) in enumerate(_iter_kv_layers(cache.get_full_cache())):
        assert torch.equal(k, model.truth[i]), f"key mismatch, layer {i}"
        assert k.shape[2] == model.total_tokens


@pytest.mark.parametrize("mode", ["in_place", "legacy_tuple"])
def test_flushed_history_matches_kivi_reference(mode):
    """With real 2-bit rounding, flushed tokens must equal the reference
    KIVI round-trip applied per flush chunk; residual tokens stay exact."""
    G = 4
    cache, model = _run_generation(mode, identity=False, G=G)
    for i, (k, v) in enumerate(_iter_kv_layers(cache.get_full_cache())):
        n_hist = cache._quant_token_count[i]
        assert n_hist > 0, "test must run long enough to trigger flushes"
        # Residual tail is served in full precision.
        assert torch.equal(k[:, :, n_hist:, :], model.truth[i][:, :, n_hist:, :])
        # History: replay the manager's flush transform chunk by chunk.
        for c0 in range(0, n_hist, G):
            chunk = model.truth[i][:, :, c0:c0 + G, :]
            k_trans = chunk.transpose(2, 3).contiguous()
            expect = torch.empty_like(k_trans)
            _roundtrip_2bit_reference(k_trans, expect, G)
            expect = expect.transpose(2, 3)
            assert torch.equal(k[:, :, c0:c0 + G, :], expect), (
                f"layer {i} flush chunk at token {c0} does not match the "
                f"KIVI per-channel reference")


def test_double_add_tokens_fails_loudly():
    """Feeding the same forward output twice must raise, not silently
    no-op — a silent skip here is indistinguishable from dropped KV."""
    _fake_native.identity = True
    cache = KiviCache(num_layers=1, head_dim=8, num_kv_heads=2,
                      device="cpu", group_size=4, residual_length=8)
    model = FakeModel(1, 2, 8, "in_place")
    out = model.forward(5)
    cache.add_tokens(out)
    with pytest.raises(AssertionError, match="desync"):
        cache.add_tokens(out)
