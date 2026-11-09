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


_HALF_MAX = 65504.0
_HALF_SCALE_FLOOR = 2.0 ** -14   # smallest normal fp16


def _scale_zero_reference(flat, meta_dtype):
    """Mirror of compute_scale<M> / saturate_zero<M> / round_through_storage<M>
    in kernels/internal/quant_tile.hpp: returns the step and zero point the
    kernel actually packs with (already rounded through the metadata dtype),
    so packing and a later dequantize agree exactly.

    fp32 metadata: step = range/3 + 1e-10, unrounded.
    fp16 metadata: step = clamp(range/3, 2^-14, 65504), zero clamped to
    +/-65504, both rounded through fp16.
    """
    mn = flat.min(dim=1, keepdim=True).values
    mx = flat.max(dim=1, keepdim=True).values
    if meta_dtype == torch.float32:
        return (mx - mn) / 3.0 + 1e-10, mn
    scale = torch.clamp((mx - mn) / 3.0, _HALF_SCALE_FLOOR, _HALF_MAX)
    zero = torch.clamp(mn, -_HALF_MAX, _HALF_MAX)
    return scale.half().to(flat.dtype), zero.half().to(flat.dtype)


def _quantize_2bit_reference(inp, packed_out, scales_out, zeros_out, group_size):
    """Exact torch mirror of the SYCL quantize kernels.

    Both kernels reduce over contiguous runs of `group_size` elements, so
    a flat [-1, group_size] view reproduces their grouping for keys
    ([B,H,D,L] input, groups along tokens) and values ([B,H,L,D] input,
    groups along channels) alike. Packing is LSB-first:
    byte = q0 | q1<<2 | q2<<4 | q3<<6. The metadata dtype is taken from
    `scales_out`, as the real ops do.
    """
    flat = inp.reshape(-1, group_size)
    scale, mn = _scale_zero_reference(flat, scales_out.dtype)
    q = torch.clamp(torch.round((flat - mn) / scale), 0.0, 3.0).to(torch.uint8)
    qb = q.reshape(-1, group_size // 4, 4)
    byte = (qb[..., 0] | (qb[..., 1] << 2) | (qb[..., 2] << 4)
            | (qb[..., 3] << 6))
    packed_out.copy_(byte.reshape(packed_out.shape))
    scales_out.copy_(scale.reshape(scales_out.shape))
    zeros_out.copy_(mn.reshape(zeros_out.shape))


def _roundtrip_2bit_reference(inp, group_size, meta_dtype=torch.float16):
    """Quantize + dequantize in one go — the expected served value of a
    flushed chunk. Defaults to fp16 metadata, KiviCache's default."""
    flat = inp.reshape(-1, group_size)
    scale, mn = _scale_zero_reference(flat, meta_dtype)
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

    def alloc_usm_host(self, sizes, dtype, align=64):
        # Plain CPU memory stands in for USM host memory: the fake "kernels"
        # above are torch ops, which can write strided arena views directly.
        return torch.empty(sizes, dtype=dtype)


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
                    async_flush=True, seed=0, metadata_dtype="auto",
                    zero_copy="auto", reserve=None):
    torch.manual_seed(seed)
    cache = KiviCache(num_layers=layers, head_dim=head_dim,
                      num_kv_heads=heads, device="cpu",
                      group_size=G, residual_length=R,
                      serving=serving, async_flush=async_flush,
                      metadata_dtype=metadata_dtype, zero_copy=zero_copy)
    if reserve is not None:
        cache.reserve(reserve)
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


def _check_against_reference(cache, model, G, meta_dtype=torch.float16):
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
            expect_k = _roundtrip_2bit_reference(chunk_k, G, meta_dtype).transpose(2, 3)
            assert torch.equal(k[:, :, c0:c0 + G, :], expect_k), (
                f"layer {i} key flush chunk at token {c0} does not match "
                f"the KIVI per-channel reference")
        # Values: per-token grouping is independent of flush chunking, so
        # the whole history can be checked in one shot.
        expect_v = _roundtrip_2bit_reference(
            model.truth_v[i][:, :, :n_hist, :].contiguous(), G, meta_dtype)
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
    used_packed = sum(
        t.numel() * t.element_size()
        for t in (cache._packed_k[i], cache._scales_k[i], cache._zeros_k[i],
                  cache._packed_v[i], cache._scales_v[i], cache._zeros_v[i]))
    assert s["packed_used_bytes"] == used_packed
    # Resident = the whole arena allocation, never less than what is used.
    assert s["packed_bytes"] == cache._arena[i].allocated_bytes >= used_packed
    assert s["serving_copy_bytes"] == 0          # rebuild mode
    assert s["kivi_bytes"] == s["packed_bytes"] + s["residual_bytes"]
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


# ---------------------------------------------------------------------------
# Metadata dtype (fp16 scales/zeros by default)
# ---------------------------------------------------------------------------

def test_default_metadata_is_fp16_and_served_dtype_is_model_dtype():
    """fp16 metadata on an fp32 model: scales/zeros are stored as fp16, but
    the served cache must stay fp32 (a served-dtype regression would
    silently run the model's attention in fp16)."""
    cache, model = _run_generation("in_place", G=4, R=8)
    cache.harvest_pending()
    assert cache._scales_k[0].dtype == torch.float16
    assert cache._zeros_v[0].dtype == torch.float16
    for k, v in _iter_kv_layers(cache.get_full_cache()):
        assert k.dtype == v.dtype == torch.float32
    _check_against_reference(cache, model, 4, torch.float16)


def test_fp32_metadata_reproduces_previous_behaviour():
    cache, model = _run_generation("in_place", G=4, R=8,
                                   metadata_dtype=torch.float32)
    cache.harvest_pending()
    assert cache._scales_k[0].dtype == torch.float32
    _check_against_reference(cache, model, 4, torch.float32)


def test_fp16_metadata_shrinks_packed_store_by_metadata_half():
    """Same token stream, fp16 vs fp32 metadata: payload bytes are equal,
    metadata bytes halve, and nothing else changes."""
    a, _ = _run_generation("in_place", G=4, R=8, seed=3)
    b, _ = _run_generation("in_place", G=4, R=8, seed=3,
                           metadata_dtype=torch.float32)
    a.harvest_pending(); b.harvest_pending()
    nbytes = lambda *ts: sum(t.numel() * t.element_size() for t in ts)
    meta = lambda c: nbytes(c._scales_k[0], c._zeros_k[0], c._scales_v[0], c._zeros_v[0])
    payload = lambda c: nbytes(c._packed_k[0], c._packed_v[0])
    assert payload(a) == payload(b)
    assert meta(a) * 2 == meta(b)
    sa, sb = a.get_stats(0), b.get_stats(0)
    assert sb["packed_used_bytes"] - sa["packed_used_bytes"] == meta(b) - meta(a)
    assert sa["packed_bytes"] < sb["packed_bytes"]
    assert sa["compression"] > sb["compression"]


def test_fp32_metadata_rejected_for_fp16_model():
    cache = KiviCache(num_layers=1, head_dim=8, num_kv_heads=2, device="cpu",
                      group_size=4, residual_length=8,
                      metadata_dtype=torch.float32)
    k = torch.randn(1, 2, 3, 8, dtype=torch.float16)
    with pytest.raises(ValueError, match="fp32 model"):
        cache.add_tokens(((k, k.clone()),))


def test_invalid_metadata_dtype_rejected():
    with pytest.raises(AssertionError, match="metadata_dtype"):
        KiviCache(num_layers=1, head_dim=8, group_size=4, residual_length=8,
                  device="cpu", metadata_dtype=torch.bfloat16)


# ---------------------------------------------------------------------------
# Packed arena (Phase 1 step 10)
# ---------------------------------------------------------------------------

def test_reserve_sizes_arena_exactly_and_never_grows():
    """With the final length known up front, the arena is allocated once at
    the exact final quantized size: no growth copy, no spare capacity."""
    prefill, steps, G, R = 6, 40, 4, 8
    total = prefill + steps
    cache, model = _run_generation("in_place", num_steps=steps, prefill=prefill,
                                   G=G, R=R, reserve=total)
    cache.harvest_pending()
    arena = cache._arena[0]
    assert arena.used == cache._quant_token_count[0]
    assert arena.cap == arena.used, (arena.cap, arena.used)
    s = cache.get_stats(0)
    # Row padding is the only allocation beyond the used bytes.
    assert s["packed_capacity_tokens"] == arena.used
    _check_against_reference(cache, model, G)


def test_arena_grows_on_demand_and_history_stays_exact():
    """No reserve: the arena starts at the first flush's size and must grow
    repeatedly without corrupting the already-quantized history, while the
    spare tail stays within the growth policy (max(4 groups, 25%))."""
    G = 4
    cache, model = _run_generation("in_place", num_steps=300, prefill=6, G=G, R=8)
    cache.harvest_pending()
    arena = cache._arena[0]
    assert arena.used > 256 and arena.cap >= arena.used
    assert arena.cap - arena.used <= max(4 * G, arena.cap // 4)
    assert arena.cap % G == 0
    _check_against_reference(cache, model, G)


def test_flush_reuses_arena_until_capacity_runs_out():
    """Steady-state flushes append in place: the arena object (and its
    storage) must not be replaced while capacity remains."""
    cache, model = _run_generation("in_place", num_steps=10, prefill=6, G=4, R=8,
                                   reserve=6 + 10 + 40)
    cache.harvest_pending()
    arena = cache._arena[0]
    assert arena.cap > arena.used          # spare capacity to append into
    ptr = arena.packed_k.data_ptr()
    for _ in range(40):
        cache.add_tokens(model.forward(1, cache.get_full_cache()))
    cache.harvest_pending()
    assert cache._arena[0] is arena and arena.packed_k.data_ptr() == ptr


@pytest.mark.parametrize("serving", ["rebuild", "cached"])
def test_direct_and_staged_paths_store_identical_bytes(serving):
    """zero_copy=True (kernels write arena views) vs False (temporaries
    copied in at harvest) must leave bit-identical packed stores."""
    a, _ = _run_generation("in_place", seed=11, serving=serving, zero_copy=True)
    b, _ = _run_generation("in_place", seed=11, serving=serving, zero_copy=False)
    a.harvest_pending(); b.harvest_pending()
    for name in ("_packed_k", "_scales_k", "_zeros_k",
                 "_packed_v", "_scales_v", "_zeros_v"):
        for ta, tb in zip(getattr(a, name), getattr(b, name)):
            assert torch.equal(ta, tb), name
    for (ka, va), (kb, vb) in zip(_iter_kv_layers(a.get_full_cache()),
                                  _iter_kv_layers(b.get_full_cache())):
        assert torch.equal(ka, kb) and torch.equal(va, vb)


def test_reset_clears_arena_and_reserve():
    cache, _ = _run_generation("in_place", reserve=30)
    cache.reset()
    assert all(a is None for a in cache._arena)
    assert cache._reserve_tokens == 0


def test_row_padding_aligns_key_rows():
    """row_align pads each key row (one per channel) to a byte multiple; the
    used views and the served cache are unchanged by it."""
    from kivi_sycl.cache.arena import LayerArena
    a = LayerArena(1, 2, 8, 4, 12, torch.float16, lambda sh, dt: torch.empty(sh, dtype=dt),
                   row_align=64)
    assert a.packed_k.shape == (1, 2, 8, 3) and a.packed_k.stride(-2) == 64
    assert a.scales_k.shape == (1, 2, 8, 3) and a.scales_k.stride(-2) * 2 == 64
    b = LayerArena(1, 2, 8, 4, 12, torch.float16, lambda sh, dt: torch.empty(sh, dtype=dt))
    assert a.allocated_bytes > b.allocated_bytes


# ---------------------------------------------------------------------------
# Per-layer group size (Phase 3 step 5)
# ---------------------------------------------------------------------------

def test_per_layer_group_size_history_matches_reference():
    """group_size=[4, 8]: each layer flushes in multiples of its own G and its
    history equals the KIVI reference at that G."""
    Gs, R = [4, 8], 8
    torch.manual_seed(0)
    cache = KiviCache(num_layers=2, head_dim=8, num_kv_heads=2, device="cpu",
                      group_size=Gs, residual_length=R)
    assert cache.G is None and cache._G == Gs
    model = FakeModel(2, 2, 8, "in_place")
    cache.add_tokens(model.forward(6))
    for _ in range(40):
        cache.add_tokens(model.forward(1, cache.get_full_cache()))
    cache.harvest_pending()
    for i, (k, v) in enumerate(_iter_kv_layers(cache.get_full_cache())):
        G = Gs[i]
        n_hist = cache._quant_token_count[i]
        assert n_hist > 0 and n_hist % G == 0
        assert R - G <= k.shape[2] - n_hist <= R - 1      # residual window for this G
        for c0 in range(0, n_hist, G):
            chunk = model.truth[i][:, :, c0:c0 + G, :].transpose(2, 3).contiguous()
            assert torch.equal(k[:, :, c0:c0 + G, :],
                               _roundtrip_2bit_reference(chunk, G).transpose(2, 3))
        assert torch.equal(v[:, :, :n_hist, :], _roundtrip_2bit_reference(
            model.truth_v[i][:, :, :n_hist, :].contiguous(), G))
        assert cache.get_stats(i)["group_size"] == G


def test_per_layer_group_size_reserve_is_exact_per_layer():
    Gs, R, prefill, steps = [4, 8], 8, 6, 40
    cache = KiviCache(num_layers=2, head_dim=8, num_kv_heads=2, device="cpu",
                      group_size=Gs, residual_length=R)
    cache.reserve(prefill + steps)
    model = FakeModel(2, 2, 8, "in_place")
    cache.add_tokens(model.forward(prefill))
    for _ in range(steps - 1):
        cache.add_tokens(model.forward(1, cache.get_full_cache()))
    cache.add_tokens(model.forward(1, cache.get_full_cache()))
    cache.harvest_pending()
    for i in range(2):
        a = cache._arena[i]
        assert a.cap == a.used == cache._quant_token_count[i], (i, a.cap, a.used)


def test_per_layer_group_size_validation():
    with pytest.raises(AssertionError, match="entries for 3 layers"):
        KiviCache(num_layers=3, head_dim=8, group_size=[4, 8], residual_length=8, device="cpu")
    with pytest.raises(AssertionError, match="must be divisible by group_size"):
        KiviCache(num_layers=2, head_dim=8, group_size=[4, 12], residual_length=16, device="cpu")
