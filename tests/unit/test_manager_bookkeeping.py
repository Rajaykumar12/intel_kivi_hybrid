"""CPU-only KiviCache bookkeeping tests — no XPU, IPEX, or compiled
extension required.

The native quantize kernels are replaced with an exact torch reference
implementation of the asymmetric 2-bit pack (bit-identical to the SYCL
kernels, including the LSB-first byte layout), so these tests isolate the
cache state machine itself: new-token extraction, bulk flush sizing,
async submit/harvest ordering, packed-store concatenation, the
torch-native dequant-on-read path, and the persistent cache view.

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


def _quantize_2bit_reference(inp, packed_out, scales_out, zeros_out, group_size):
    """Exact torch mirror of the SYCL quantize kernels.

    Both kernels reduce over contiguous runs of `group_size` elements, so
    a flat [-1, group_size] view reproduces their grouping for keys
    ([B,H,D,L] input, groups along tokens) and values ([B,H,L,D] input,
    groups along channels) alike. Packing is LSB-first:
    byte = q0 | q1<<2 | q2<<4 | q3<<6.
    """
    flat = inp.reshape(-1, group_size)
    mn = flat.min(dim=1, keepdim=True).values
    mx = flat.max(dim=1, keepdim=True).values
    scale = (mx - mn) / 3.0 + 1e-10
    q = torch.clamp(torch.round((flat - mn) / scale), 0.0, 3.0).to(torch.uint8)
    qb = q.reshape(-1, group_size // 4, 4)
    byte = (qb[..., 0] | (qb[..., 1] << 2) | (qb[..., 2] << 4)
            | (qb[..., 3] << 6))
    packed_out.copy_(byte.reshape(packed_out.shape))
    scales_out.copy_(scale.reshape(scales_out.shape))
    zeros_out.copy_(mn.reshape(zeros_out.shape))


def _roundtrip_2bit_reference(inp, group_size):
    """Quantize + dequantize in one go — the expected served value of a
    flushed chunk."""
    flat = inp.reshape(-1, group_size)
    mn = flat.min(dim=1, keepdim=True).values
    mx = flat.max(dim=1, keepdim=True).values
    scale = (mx - mn) / 3.0 + 1e-10
    q = torch.clamp(torch.round((flat - mn) / scale), 0.0, 3.0)
    return (q * scale + mn).reshape(inp.shape)


class _FakeEvent:
    def wait(self):
        pass


class _FakeNative(types.ModuleType):
    """Stand-in for kivi_sycl._C 's async quantize entry points."""

    def __init__(self):
        super().__init__("kivi_sycl._C")

    def quantize_keys_async(self, inp, out, scales, zeros, group_size):
        _quantize_2bit_reference(inp, out, scales, zeros, group_size)
        return _FakeEvent()

    def quantize_values_async(self, inp, out, scales, zeros, head_dim, group_size):
        _quantize_2bit_reference(inp, out, scales, zeros, group_size)
        return _FakeEvent()


_fake_native = _FakeNative()

# Only claim the `kivi_sycl._C` slot on machines with no compiled build —
# otherwise a combined `pytest tests/` run would hand the fake to
# tests/integration/, whose `from kivi_sycl import _C` is meant to get the
# real kernels. (setdefault alone is not enough: whether the real extension
# has been imported yet depends on test collection order.)
try:
    import kivi_sycl._C  # noqa: F401
except ImportError:
    sys.modules["kivi_sycl._C"] = _fake_native

import kivi_sycl.cache.manager as manager_module  # noqa: E402
from kivi_sycl.cache.manager import KiviCache, _iter_kv_layers  # noqa: E402


@pytest.fixture(autouse=True)
def _use_fake_native(monkeypatch):
    """Point the manager's native seam at the torch stand-in for the
    duration of each test, and put the real one back afterwards.

    This must be a fixture, not a module-level assignment: pytest imports
    every test module up front, so a plain `manager_module.kivi_native =
    _fake_native` at import time also silently rebinds the seam for
    tests/integration/, which then "passes" against the fake instead of the
    real kernels it exists to exercise."""
    monkeypatch.setattr(manager_module, "kivi_native", _fake_native)


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

    def __init__(self, num_layers, num_heads, head_dim, mode, batch=1):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.mode = mode
        self.batch = batch
        self.total_tokens = 0
        # Ground truth: full FP32 KV history per layer, [B, H, T, D].
        self.truth = [None] * num_layers
        self.truth_v = [None] * num_layers

    def _new_kv(self, layer, n_new):
        # Token t gets the distinct fill value t (plus small per-position
        # jitter) so dropped/reordered tokens are caught exactly.
        t0 = self.total_tokens
        toks = torch.arange(t0, t0 + n_new, dtype=torch.float32)
        base = toks.view(1, 1, n_new, 1).expand(
            self.batch, self.num_heads, n_new, self.head_dim)
        jitter = torch.randn(
            self.batch, self.num_heads, n_new, self.head_dim) * 0.01
        k = (base + jitter).contiguous()
        v = (base - jitter).contiguous()
        return k, v

    def forward(self, n_new, past=None):
        new_per_layer = [self._new_kv(i, n_new) for i in range(self.num_layers)]
        for i, (k, v) in enumerate(new_per_layer):
            self.truth[i] = k if self.truth[i] is None else \
                torch.cat([self.truth[i], k], dim=2)
            self.truth_v[i] = v if self.truth_v[i] is None else \
                torch.cat([self.truth_v[i], v], dim=2)
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


def _run_generation(mode, num_steps=24, prefill=6, G=4, R=8, head_dim=8,
                    layers=2, heads=2, batch=1, serving="rebuild",
                    async_flush=True, seed=0):
    torch.manual_seed(seed)
    cache = KiviCache(num_layers=layers, head_dim=head_dim,
                      num_kv_heads=heads, device="cpu",
                      group_size=G, residual_length=R,
                      serving=serving, async_flush=async_flush)
    model = FakeModel(layers, heads, head_dim, mode, batch=batch)

    cache.add_tokens(model.forward(prefill))
    for _ in range(num_steps):
        past = cache.get_full_cache()
        served_len = next(iter(_iter_kv_layers(past)))[0].shape[2]
        assert served_len == model.total_tokens, (
            f"stale cache served: {served_len} tokens, model has produced "
            f"{model.total_tokens} — generated tokens are being dropped")
        cache.add_tokens(model.forward(1, past))
    return cache, model


def _check_against_reference(cache, model, G):
    """Residual tail must be bit-exact; quantized history must equal the
    reference 2-bit round-trip applied in G-token groups (group alignment
    is invariant to how flushes were chunked — every chunk is a multiple
    of G)."""
    cache.harvest_pending()
    for i, (k, v) in enumerate(_iter_kv_layers(cache.get_full_cache())):
        n_hist = cache._quant_token_count[i]
        assert n_hist > 0, "test must run long enough to trigger flushes"
        assert k.shape[2] == model.total_tokens
        # Residual tail is served in full precision.
        assert torch.equal(k[:, :, n_hist:, :], model.truth[i][:, :, n_hist:, :]), \
            f"residual keys not exact, layer {i}"
        # History: keys per-channel (transpose then group along tokens),
        # values per-token (group along channels).
        for c0 in range(0, n_hist, G):
            chunk_k = model.truth[i][:, :, c0:c0 + G, :].transpose(2, 3).contiguous()
            expect_k = _roundtrip_2bit_reference(chunk_k, G).transpose(2, 3)
            assert torch.equal(k[:, :, c0:c0 + G, :], expect_k), (
                f"layer {i} key flush chunk at token {c0} does not match "
                f"the KIVI per-channel reference")
        # Values: per-token grouping is independent of flush chunking, so
        # the whole history can be checked in one shot.
        expect_v = _roundtrip_2bit_reference(
            model.truth_v[i][:, :, :n_hist, :].contiguous(), G)
        assert torch.equal(v[:, :, :n_hist, :], expect_v), (
            f"layer {i} value history does not match the KIVI "
            f"per-token reference")
        assert torch.equal(v[:, :, n_hist:, :], model.truth_v[i][:, :, n_hist:, :]), \
            f"residual values not exact, layer {i}"


@pytest.mark.parametrize("mode", ["in_place", "legacy_tuple"])
@pytest.mark.parametrize("async_flush", [True, False])
def test_history_matches_kivi_reference(mode, async_flush):
    G = 4
    cache, model = _run_generation(mode, G=G, async_flush=async_flush)
    _check_against_reference(cache, model, G)


def test_packed_store_is_resident_and_measured():
    """After harvest, the packed uint8 store must exist, cover every
    quantized token, and get_stats must report measured bytes for it."""
    cache, model = _run_generation("in_place", G=4, R=8)
    cache.harvest_pending()
    i = 0
    n_hist = cache._quant_token_count[i]
    B, H, D = cache.batch_size, cache.num_kv_heads, cache.head_dim
    assert cache._packed_k[i].shape == (B, H, D, n_hist // 4)
    assert cache._packed_k[i].dtype == torch.uint8
    assert cache._packed_v[i].shape == (B, H, n_hist, D // 4)
    assert cache._scales_k[i].shape == (B, H, D, n_hist // cache.G)
    assert cache._scales_v[i].shape == (B, H, n_hist, D // cache.G)

    s = cache.get_stats(layer=i)
    expected_packed = sum(
        t.numel() * t.element_size()
        for t in (cache._packed_k[i], cache._scales_k[i], cache._zeros_k[i],
                  cache._packed_v[i], cache._scales_v[i], cache._zeros_v[i]))
    assert s["packed_bytes"] == expected_packed
    assert s["serving_copy_bytes"] == 0          # rebuild mode
    assert s["kivi_bytes"] == expected_packed + s["residual_bytes"]
    assert s["quantized_tokens"] == n_hist


def test_rebuild_mode_retains_no_fp_history():
    """serving='rebuild': after add_tokens, no FP copy of the quantized
    history may stay resident (no serving copy, no persistent view)."""
    cache, _ = _run_generation("in_place", serving="rebuild")
    assert cache._cache_view is None
    assert all(d is None for d in cache._deq_serving)


def test_cached_mode_serves_identically_to_rebuild():
    """Both serving modes must produce bit-identical served tensors for
    the same token stream (cached only precomputes the dequant)."""
    cache_a, model_a = _run_generation("in_place", serving="rebuild", seed=7)
    cache_b, model_b = _run_generation("in_place", serving="cached", seed=7)
    assert torch.equal(model_a.truth[0], model_b.truth[0]), \
        "seeding must make both runs identical for the comparison to mean anything"
    cache_a.harvest_pending()
    cache_b.harvest_pending()
    for (ka, va), (kb, vb) in zip(_iter_kv_layers(cache_a.get_full_cache()),
                                  _iter_kv_layers(cache_b.get_full_cache())):
        assert torch.equal(ka, kb)
        assert torch.equal(va, vb)


def test_batch_of_two():
    """G6: batched KV (B=2) must flow through flush/serve identically."""
    G = 4
    cache, model = _run_generation("in_place", batch=2, G=G)
    assert cache.batch_size == 2
    _check_against_reference(cache, model, G)


def test_bulk_prefill_single_flush():
    """A long prompt must be flushed in ONE submission that leaves the
    residual in [R-G, R-1] tokens, not one flush per G tokens."""
    torch.manual_seed(0)
    G, R = 4, 8
    cache = KiviCache(num_layers=1, head_dim=8, num_kv_heads=2,
                      device="cpu", group_size=G, residual_length=R,
                      serving="rebuild", async_flush=False)
    model = FakeModel(1, 2, 8, "in_place")
    prompt_len = 50
    cache.add_tokens(model.forward(prompt_len))
    # One flush, sized to drain the residual to R-G..R-1.
    assert cache._flush_count[0] == 1
    res = cache.res_keys[0].shape[2]
    assert R - G <= res < R
    assert cache._quant_token_count[0] == prompt_len - res
    assert cache._quant_token_count[0] % G == 0


def test_async_flush_defers_trim_until_next_add():
    """With async_flush=True, flushed tokens must remain in the residual
    (still served in FP) until the next add_tokens harvests them."""
    torch.manual_seed(0)
    G, R = 4, 8
    cache = KiviCache(num_layers=1, head_dim=8, num_kv_heads=2,
                      device="cpu", group_size=G, residual_length=R,
                      serving="rebuild", async_flush=True)
    model = FakeModel(1, 2, 8, "in_place")
    cache.add_tokens(model.forward(R))          # triggers a flush submission
    assert cache._pending[0] is not None
    assert cache.res_keys[0].shape[2] == R      # not trimmed yet
    # Serving still covers all tokens, from the FP residual.
    served = next(iter(_iter_kv_layers(cache.get_full_cache())))[0]
    assert served.shape[2] == R
    assert torch.equal(served, model.truth[0])  # pending tokens still FP

    past = cache.get_full_cache()
    cache.add_tokens(model.forward(1, past))    # harvest happens here
    assert cache._pending[0] is None
    assert cache._quant_token_count[0] == G
    assert cache.res_keys[0].shape[2] == R + 1 - G


def test_double_add_tokens_fails_loudly():
    """Feeding the same forward output twice must raise, not silently
    no-op — a silent skip here is indistinguishable from dropped KV."""
    cache = KiviCache(num_layers=1, head_dim=8, num_kv_heads=2,
                      device="cpu", group_size=4, residual_length=8)
    model = FakeModel(1, 2, 8, "in_place")
    out = model.forward(5)
    cache.add_tokens(out)
    with pytest.raises(AssertionError, match="desync"):
        cache.add_tokens(out)


def test_reset_clears_packed_store():
    cache, _ = _run_generation("in_place")
    cache.reset()
    assert cache.batch_size is None
    for i in range(cache.num_layers):
        assert cache._packed_k[i] is None
        assert cache._packed_v[i] is None
        assert cache._pending[i] is None
        assert cache.res_keys[i] is None
        assert cache._deq_serving[i] is None
        assert cache._quant_token_count[i] == 0
