"""Shared driver for the native-engine differential tests (Phase 5 S1.4).

Feeds one token stream to the Python KiviCache and to the C++ _C.Engine and
compares their full state after every step. Two update orders:
  "hf"     KiviHFCache's per-layer update() (KiviLayer.update)
  "legacy" add_tokens(past) on the Python side; harvest_all + append +
           (sync: harvest_all) on the native side — what add_tokens does.
"""

import torch

from kivi_sycl.cache.manager import KiviCache
from kivi_sycl.hf.cache import KiviHFCache


def make_pair(_C, *, layers, heads, head_dim, G, R, device, async_flush,
              zero_copy="auto", meta="auto", reserve=None):
    py = KiviCache(num_layers=layers, head_dim=head_dim, num_kv_heads=heads,
                   device=device, group_size=G, residual_length=R,
                   async_flush=async_flush, zero_copy=zero_copy,
                   metadata_dtype=meta)
    gs = G if isinstance(G, list) else [G]
    zc = {"auto": "auto", True: "on", False: "off"}[zero_copy]
    md = {"auto": "auto", torch.float16: "float16", torch.float32: "float32"}[meta]
    nat = _C.Engine(num_layers=layers, head_dim=head_dim, residual_length=R,
                    group_size=gs, metadata_dtype=md, zero_copy=zc,
                    async_flush=async_flush, device=device)
    if reserve is not None:
        py.reserve(reserve)
        nat.reserve(reserve)
    return py, nat


def assert_same_state(py, nat, layers, where):
    for i in range(layers):
        pr = (py.res_keys[i], py.res_values[i])
        nr = nat.residual(i)
        if pr[0] is None:
            assert nr is None, f"{where} layer {i}: native has a residual, python not"
        else:
            assert torch.equal(pr[0], nr[0]) and torch.equal(pr[1], nr[1]), \
                f"{where} layer {i}: residual differs"
        ph = None if py._arena[i] is None or py._arena[i].used == 0 else \
            py._arena[i].used_keys() + py._arena[i].used_values()
        nh = nat.history(i)
        assert (ph is None) == (nh is None), f"{where} layer {i}: history presence differs"
        if ph is not None:
            for a, b, name in zip(ph, nh, ("pk", "sk", "zk", "pv", "sv", "zv")):
                assert torch.equal(a.cpu(), b.cpu()), f"{where} layer {i}: history {name} differs"
        assert py._quant_token_count[i] == nat.quant_tokens(i), f"{where} layer {i}: quant tokens"
        assert py._flush_count[i] == nat.flushes(i), f"{where} layer {i}: flush count"
        p_pending = py._pending[i]["n"] if py._pending[i] is not None else 0
        assert p_pending == nat.pending_tokens(i), f"{where} layer {i}: pending tokens"
        p_cap = py._arena[i].cap if py._arena[i] is not None else 0
        assert p_cap == nat.arena_capacity(i), f"{where} layer {i}: arena capacity"
        ps, ns = py.get_stats(i), nat.stats(i)
        for key in ("group_size", "total_tokens", "quantized_tokens", "residual_tokens",
                    "pending_tokens", "packed_bytes", "packed_used_bytes",
                    "packed_capacity_tokens", "residual_bytes", "kivi_bytes",
                    "fp_baseline_bytes"):
            assert ps[key] == ns[key], f"{where} layer {i}: stats[{key}] {ps[key]} != {ns[key]}"
        assert abs(ps["compression"] - ns["compression"]) < 1e-12


def run(_C, order, *, layers=2, heads=2, head_dim=8, G=4, R=8, device="cpu",
        async_flush=True, zero_copy="auto", meta="auto", reserve=None,
        prefill=11, steps=40, seed=0):
    torch.manual_seed(seed)
    py, nat = make_pair(_C, layers=layers, heads=heads, head_dim=head_dim, G=G, R=R,
                        device=device, async_flush=async_flush, zero_copy=zero_copy,
                        meta=meta, reserve=reserve)
    hf = KiviHFCache(py) if order == "hf" else None
    full = [None] * layers                      # cumulative KV for the legacy path
    for step, n in enumerate([prefill] + [1] * steps):
        new = [(torch.randn(1, heads, n, head_dim), torch.randn(1, heads, n, head_dim))
               for _ in range(layers)]
        if order == "hf":
            for i, (k, v) in enumerate(new):
                hf.layers[i].update(k, v)
                nat.update(i, k, v)
        else:
            for i, (k, v) in enumerate(new):
                full[i] = (k, v) if full[i] is None else \
                    (torch.cat([full[i][0], k], 2), torch.cat([full[i][1], v], 2))
            py.add_tokens(tuple(full))
            nat.harvest_all()
            for i, (k, v) in enumerate(new):
                nat.append(i, k, v)
            if not async_flush:
                nat.harvest_all()
        assert_same_state(py, nat, layers, f"{order} step {step}")
    py.harvest_pending()
    nat.harvest_all()
    assert_same_state(py, nat, layers, f"{order} final")
    assert py._quant_token_count[0] > 0, "test must exercise flushes"
    return py, nat
