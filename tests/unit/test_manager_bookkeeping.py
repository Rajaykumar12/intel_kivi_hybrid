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
                    async_flush=True, seed=0, key_flush="drain"):
    torch.manual_seed(seed)
    cache = KiviCache(num_layers=layers, head_dim=head_dim,
                      num_kv_heads=heads, device="cpu",
                      group_size=G, residual_length=R,
                      serving=serving, async_flush=async_flush,
                      key_flush=key_flush)
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
        # Key and value histories are the same length under
        # key_flush="drain" and differ under "reset" — each side is checked
        # against its own boundary.
        n_k = cache._quant_token_count[i]
        n_v = cache._quant_token_count_v[i]
        assert n_k > 0 and n_v > 0, "test must run long enough to trigger flushes"
        assert k.shape[2] == model.total_tokens
        assert v.shape[2] == model.total_tokens
        # Residual tail is served in full precision.
        assert torch.equal(k[:, :, n_k:, :], model.truth[i][:, :, n_k:, :]), \
            f"residual keys not exact, layer {i}"
        # History: keys per-channel (transpose then group along tokens),
        # values per-token (group along channels).
        #
        # Every flush is a multiple of G and starts at a multiple of G, so
        # key groups land on global G boundaries regardless of how the
        # flushes were chunked — which is why this check is valid for both
        # the incremental drain and "reset"'s whole-buffer submissions.
        for c0 in range(0, n_k, G):
            chunk_k = model.truth[i][:, :, c0:c0 + G, :].transpose(2, 3).contiguous()
            expect_k = _roundtrip_2bit_reference(chunk_k, G).transpose(2, 3)
            assert torch.equal(k[:, :, c0:c0 + G, :], expect_k), (
                f"layer {i} key flush chunk at token {c0} does not match "
                f"the KIVI per-channel reference")
        # Values: per-token grouping is independent of flush chunking, so
        # the whole history can be checked in one shot.
        expect_v = _roundtrip_2bit_reference(
            model.truth_v[i][:, :, :n_v, :].contiguous(), G)
        assert torch.equal(v[:, :, :n_v, :], expect_v), (
            f"layer {i} value history does not match the KIVI "
            f"per-token reference")
        assert torch.equal(v[:, :, n_v:, :], model.truth_v[i][:, :, n_v:, :]), \
            f"residual values not exact, layer {i}"


@pytest.mark.parametrize("mode", ["in_place", "legacy_tuple"])
@pytest.mark.parametrize("async_flush", [True, False])
@pytest.mark.parametrize("key_flush", ["drain", "reset"])
def test_history_matches_kivi_reference(mode, async_flush, key_flush):
    G = 4
    cache, model = _run_generation(mode, G=G, async_flush=async_flush,
                                   key_flush=key_flush)
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


@pytest.mark.parametrize("key_flush", ["drain", "reset"])
def test_cached_mode_serves_identically_to_rebuild(key_flush):
    """Both serving modes must produce bit-identical served tensors for
    the same token stream (cached only precomputes the dequant).

    Covered for both flush policies because under "reset" a flush record
    often carries only one side, so the persistent dequantized history grows
    its key and value halves on different steps."""
    cache_a, model_a = _run_generation("in_place", serving="rebuild", seed=7,
                                       key_flush=key_flush)
    cache_b, model_b = _run_generation("in_place", serving="cached", seed=7,
                                       key_flush=key_flush)
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


# ---------------------------------------------------------------------- #
#  P1: key-residual flush policy                                          #
# ---------------------------------------------------------------------- #


def test_key_flush_drain_is_bit_identical_to_the_single_policy():
    """The default must not have moved. `drain` is the policy that produced
    every published number in this repo, so it is the regression baseline:
    splitting the key and value paths must be a no-op for it."""
    cache_a, _ = _run_generation("in_place", key_flush="drain", seed=3)
    cache_b, _ = _run_generation("in_place", seed=3)          # default
    cache_a.harvest_pending()
    cache_b.harvest_pending()
    for (ka, va), (kb, vb) in zip(_iter_kv_layers(cache_a.get_full_cache()),
                                  _iter_kv_layers(cache_b.get_full_cache())):
        assert torch.equal(ka, kb)
        assert torch.equal(va, vb)
    for i in range(cache_a.num_layers):
        # Under "drain" the split counters must stay locked together.
        assert cache_a._quant_token_count[i] == cache_a._quant_token_count_v[i]


def test_reset_requires_R_divisible_by_G():
    """The paper's §3.3 constraint, enforced only where it actually binds."""
    with pytest.raises(AssertionError, match="divisible by group_size"):
        KiviCache(num_layers=1, head_dim=8, num_kv_heads=2, device="cpu",
                  group_size=4, residual_length=10, key_flush="reset")
    # Same R is fine under the incremental drain, which is why this is not
    # a global constraint.
    KiviCache(num_layers=1, head_dim=8, num_kv_heads=2, device="cpu",
              group_size=4, residual_length=10, key_flush="drain")


def test_reset_empties_the_key_window_and_diverges_from_values():
    """Paper (§3.3, Fig. 3): at R tokens the whole key residual is quantized
    and the buffer is left empty. The value FIFO keeps draining a group at a
    time, so the two token counts must separate."""
    torch.manual_seed(0)
    G, R = 4, 8
    cache = KiviCache(num_layers=1, head_dim=8, num_kv_heads=2, device="cpu",
                      group_size=G, residual_length=R, async_flush=False,
                      key_flush="reset")
    model = FakeModel(1, 2, 8, "in_place")
    cache.add_tokens(model.forward(R))           # first flush fires here
    assert cache.res_keys[0].shape[2] == 0, "key window must reset to empty"
    assert cache._quant_token_count[0] == R
    # Values drained to [R-G, R-1], not to empty.
    assert cache.res_values[0].shape[2] == R - G
    assert cache._quant_token_count_v[0] == G
    assert cache._quant_token_count[0] != cache._quant_token_count_v[0]

    # Both sides still describe the same token stream.
    for _ in range(20):
        past = cache.get_full_cache()
        cache.add_tokens(model.forward(1, past))
    k, v = next(iter(_iter_kv_layers(cache.get_full_cache())))
    assert k.shape[2] == v.shape[2] == model.total_tokens


def test_reset_key_window_averages_half_R():
    """P1 acceptance: measured mean key occupancy under "reset" is within 5%
    of the paper's R/2, and "drain" holds roughly twice that.

    The exact means are (R-1)/2 = 63.5 for "reset" and R-G/2-1/2 = 111.5 for
    "drain"; the half-token offsets are the discrete-sampling artifact, which
    is why this needs a realistic R to land inside 5% rather than a toy one.

    Prefilling exactly R puts both policies in their steady-state cycle from
    the first sample, and the step count is a whole number of cycles for
    both. Otherwise the buffer's warm-up (filling from empty, which "drain"
    spends R steps doing) drags the mean down by ~9% and this measures
    sampling artifacts rather than the policy."""
    G, R = 32, 128
    stats = {}
    for policy in ("reset", "drain"):
        cache, _ = _run_generation("in_place", num_steps=5 * R - 1, prefill=R,
                                   G=G, R=R, head_dim=64, layers=1, heads=1,
                                   key_flush=policy)
        assert cache._res_occ_steps[0] % R == 0, "must sample whole cycles"
        stats[policy] = cache.get_stats(layer=0)

    mean_reset = stats["reset"]["mean_residual_tokens"]
    assert abs(mean_reset - R / 2) / (R / 2) < 0.05, (
        f"key window averaged {mean_reset:.1f}, expected ~{R/2} (paper R/2)")

    mean_drain = stats["drain"]["mean_residual_tokens"]
    assert abs(mean_drain - (R - G / 2)) / (R - G / 2) < 0.05, (
        f"drain key window averaged {mean_drain:.1f}, expected ~{R - G/2}")

    # The whole point of the flag: "drain" holds materially more FP keys.
    assert mean_drain > 1.5 * mean_reset

    # Values are untouched by the flag.
    assert stats["reset"]["mean_residual_tokens_v"] == \
        stats["drain"]["mean_residual_tokens_v"]

    # ...and holding fewer FP tokens on average must show up as a lower
    # time-averaged residual fraction. Note this is rho_mean, not rho: see
    # test_instantaneous_compression_is_not_a_policy_comparison.
    assert stats["reset"]["rho_mean"] < stats["drain"]["rho_mean"]


def test_instantaneous_compression_is_not_a_policy_comparison():
    """A reporting hazard, pinned so the G/key_flush ablation cannot walk
    into it.

    Under "reset" the key window sweeps the whole 0..R range, so the
    compression `get_stats` reports depends on where in the sawtooth the run
    happened to stop. Stop at the top of the ramp and "reset" looks exactly
    like "drain" — which is what a naive A/B at a single step length shows,
    despite "reset" holding half the full-precision keys on average.

    Compare policies with `rho_mean`, or at stopping points aligned to the
    same phase. Never with a single `compression` reading."""
    G, R = 32, 128
    seen = set()
    for extra in (0, R // 4, R // 2):
        cache, _ = _run_generation("in_place", num_steps=2 * R + extra,
                                   prefill=R, G=G, R=R, head_dim=64,
                                   layers=1, heads=1, key_flush="reset")
        s = cache.get_stats(layer=0)
        seen.add(round(s["compression"], 3))
    assert len(seen) == 3, (
        f"expected instantaneous compression to move with sawtooth phase, "
        f"got {seen}")


def test_mean_compression_separates_the_policies_at_every_run_length():
    """The fix for the hazard above.

    `mean_compression` does move with run length — correctly, because rho
    falls as context grows, and a longer run averages over more of the
    low-rho tail. What it must not do is invert or collapse the ranking of
    two policies the way a single instantaneous reading does. So the test is
    that "reset" beats "drain" at *every* length compared like-for-like,
    which is exactly how the ablation will use it."""
    G, R = 32, 128
    for steps in (2 * R, 3 * R, 4 * R + R // 2):
        got = {}
        for policy in ("drain", "reset"):
            cache, _ = _run_generation("in_place", num_steps=steps, prefill=R,
                                       G=G, R=R, head_dim=64, layers=1,
                                       heads=1, key_flush=policy)
            s = cache.get_stats(layer=0)
            assert s["compression_samples"] > 0
            got[policy] = s["mean_compression"]
        # "reset" holds half the FP keys on average, so it must compress
        # better — at every length, not just the one that flatters it.
        assert got["reset"] > got["drain"], (steps, got)


def test_compression_matches_the_law_at_measured_rho():
    """P1 acceptance: reported compression equals `C / (1 + rho(C-1))` using
    the *measured* rho, where the ceiling is `C = 2e / (0.5 + 4m/G)` for an
    element size e and metadata size m in bytes (both fp32 here).

    This is the check that keeps the headline number measured rather than
    analytic — the failure this repo already made once and recorded in
    status_and_roadmap.md §2.1."""
    G, R = 32, 128
    for policy in ("drain", "reset"):
        cache, _ = _run_generation("in_place", num_steps=400, prefill=1,
                                   G=G, R=R, head_dim=64, layers=1, heads=1,
                                   key_flush=policy)
        cache.harvest_pending()
        s = cache.get_stats(layer=0)
        e = m = 4.0                                   # fp32 cache and metadata
        C = 2 * e / (0.5 + 4 * m / G)                 # == 8.0 at G=32, fp32
        rho = s["rho"]
        predicted = C / (1 + rho * (C - 1))
        assert abs(predicted - s["compression"]) < 0.01, (
            f"{policy}: law predicts {predicted:.4f} at measured rho="
            f"{rho:.4f}, get_stats reports {s['compression']:.4f}")


# ---------------------------------------------------------------------- #
#  P9: peak vs resident memory                                            #
# ---------------------------------------------------------------------- #


def test_peak_bytes_is_at_least_resident():
    """Peak is a high-water mark over resident plus transients, so it can
    never fall below the resident figure measured at the same point."""
    cache, _ = _run_generation("in_place", G=4, R=8)
    cache.harvest_pending()
    s = cache.get_stats(layer=0)
    assert s["peak_bytes"] >= s["kivi_bytes"] > 0
    assert 0 < s["peak_compression"] <= s["compression"]
    assert s["peak_at_tokens"] > 0


def test_peak_captures_the_bulk_prefill_transient():
    """P9: during bulk prefill the flushed tokens exist in the FP residual,
    in the retained device-side kernel inputs, and in the packed store at
    once. Resident accounting cannot see that; peak must.

    The gap is what makes reporting resident against KIVI's peak-memory
    claim a metric mismatch rather than a rounding difference."""
    torch.manual_seed(0)
    G, R, D = 32, 128, 64
    cache = KiviCache(num_layers=1, head_dim=D, num_kv_heads=2, device="cpu",
                      group_size=G, residual_length=R, async_flush=True)
    model = FakeModel(1, 2, D, "in_place")
    cache.add_tokens(model.forward(2048))     # one big prefill flush

    s = cache.get_stats(layer=0)
    # The prompt's whole grouped part is still in the FP residual (pending,
    # untrimmed) *and* on the device *and* packed.
    assert cache._pending[0] is not None
    assert s["peak_bytes"] > s["kivi_bytes"], \
        "peak must exceed resident while a bulk flush is in flight"
    # Peak here is dominated by full-precision copies of the prompt, so it
    # lands near — or above — the FP baseline it is meant to beat. This is
    # the finding, not a bug: assert it is in that regime so a future change
    # that silently stops counting the transient fails here.
    assert s["peak_compression"] < 2.0, (
        f"peak compression {s['peak_compression']:.2f}x is suspiciously "
        f"good for a mid-prefill sample — is the transient still counted?")

    cache.harvest_pending()
    after = cache.get_stats(layer=0)
    assert after["peak_bytes"] == s["peak_bytes"], "peak must not decay"
    assert after["kivi_bytes"] < after["peak_bytes"]


@pytest.mark.parametrize("key_flush", ["drain", "reset"])
def test_chunked_prefill_is_bit_identical_and_lowers_peak(key_flush):
    """P9 mitigation: slicing a long prompt must change only *when* memory is
    held, never what is stored. Slice boundaries are multiples of G, so the
    quantization groups do not move and the packed output is bit-identical —
    which is the only reason this is a free win rather than a trade."""
    G, R, D, chunk = 32, 128, 64, 256
    out = {}
    for pc in (None, chunk):
        torch.manual_seed(11)
        cache = KiviCache(num_layers=1, head_dim=D, num_kv_heads=2,
                          device="cpu", group_size=G, residual_length=R,
                          async_flush=True, key_flush=key_flush,
                          prefill_chunk=pc)
        model = FakeModel(1, 2, D, "in_place")
        cache.add_tokens(model.forward(2048))
        for _ in range(3):
            past = cache.get_full_cache()
            cache.add_tokens(model.forward(1, past))
        cache.harvest_pending()
        k, v = next(iter(_iter_kv_layers(cache.get_full_cache())))
        out[pc] = (k, v, cache.get_stats(layer=0))

    k0, v0, s0 = out[None]
    k1, v1, s1 = out[chunk]
    assert torch.equal(k0, k1), "chunked prefill changed the served keys"
    assert torch.equal(v0, v1), "chunked prefill changed the served values"
    assert s0["kivi_bytes"] == s1["kivi_bytes"]
    assert s0["quantized_tokens"] == s1["quantized_tokens"]
    assert s0["quantized_tokens_v"] == s1["quantized_tokens_v"]
    # ...and the whole point: the transient is bounded by the slice.
    assert s1["peak_bytes"] < s0["peak_bytes"], (
        f"chunked peak {s1['peak_bytes']} not below unchunked "
        f"{s0['peak_bytes']}")


def test_prefill_chunk_must_preserve_group_alignment():
    """An unaligned slice would move the quantization group boundaries, so
    the packed output would silently stop matching the unchunked path.
    Reject it at construction rather than producing different numbers."""
    for bad in (100, 64):        # not a multiple of G / smaller than R
        with pytest.raises(AssertionError, match="prefill_chunk"):
            KiviCache(num_layers=1, head_dim=64, num_kv_heads=2, device="cpu",
                      group_size=32, residual_length=128, prefill_chunk=bad)


def test_peak_does_not_double_count_aliased_views():
    """_serve_layer returns the residual tensors themselves before anything
    has been quantized. Counting that as a fresh allocation would inflate
    peak by a full FP copy."""
    torch.manual_seed(0)
    cache = KiviCache(num_layers=1, head_dim=8, num_kv_heads=2, device="cpu",
                      group_size=4, residual_length=8)
    model = FakeModel(1, 2, 8, "in_place")
    cache.add_tokens(model.forward(4))        # below R: no flush, no history
    # Checked at the _serve_layer seam, not through the returned Cache:
    # DynamicCache.update() is free to copy, and that copy is transformers'
    # allocation, not one this accounting is responsible for.
    served_k, _ = cache._serve_layer(0)
    assert served_k is cache.res_keys[0], "precondition: the view is an alias"
    cache.get_full_cache()
    s = cache.get_stats(layer=0)
    assert s["peak_bytes"] == s["kivi_bytes"]


def test_reset_clears_peak_and_occupancy_counters():
    cache, _ = _run_generation("in_place")
    assert cache.get_stats(layer=0)["peak_bytes"] > 0
    cache.reset()
    for i in range(cache.num_layers):
        assert cache._peak_bytes[i] == 0
        assert cache._peak_at_tokens[i] == 0
        assert cache._res_occ_steps[i] == 0
        assert cache._quant_token_count_v[i] == 0


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
