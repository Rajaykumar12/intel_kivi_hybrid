"""KIVI as a native ``transformers.Cache`` (roadmap G9).

``KiviHFCache`` is a ``Cache`` whose per-layer ``update()`` feeds the
existing ``KiviCache`` engine (FP residual, async flush, packed arena) and
returns only the FP residual — never a full-length FP history. The packed
history is consumed by the ``"kivi"`` attention function (``attention.py``),
which finds this layer through a tag on the returned key tensor.

Compared with ``KiviCache.add_tokens()`` + ``get_full_cache()``, nothing
rebuilds a full [B, H, T, D] FP cache between steps; the reference attention
path still dequantizes inside attention, and the fused path does not at all.
"""

from typing import Optional

import torch
from transformers.cache_utils import Cache, CacheLayerMixin

from ..cache.manager import KiviCache

__all__ = ["KiviLayer", "KiviHFCache"]

# Attribute set on the key tensor returned by KiviLayer.update(), so the
# registered attention function can reach the packed history for that layer.
LAYER_TAG = "_kivi_layer"


class KiviLayer(CacheLayerMixin):
    """One model layer's view of the shared ``KiviCache`` engine."""

    is_sliding = False
    is_compileable = False

    def __init__(self, engine: KiviCache, layer_idx: int):
        super().__init__()
        self.engine = engine
        self.layer_idx = layer_idx

    def lazy_initialization(self, key_states, value_states) -> None:
        self.dtype, self.device = key_states.dtype, key_states.device
        self.is_initialized = True

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor,
               *args, **kwargs):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        e, i = self.engine, self.layer_idx
        e._init_session(key_states, i)
        # Same flush timing as KiviCache.add_tokens(), which runs after each
        # forward pass: (1) collect the flush submitted on the previous step
        # (it had a whole forward pass to run, so the wait is ~free); (2)
        # submit a flush for the residual as it stood *before* this step's
        # tokens; (3) append them. Flushed tokens stay in the residual, served
        # in FP, until harvested — so this step attends to them in FP and the
        # next step to their 2-bit form, exactly as on the add_tokens path.
        e._harvest_layer(i)
        e._maybe_flush(i)
        if not e.async_flush:
            e._harvest_layer(i)
        e._append_residual(i, key_states.to(e._cache_dtype), value_states.to(e._cache_dtype))
        keys, values = e.res_keys[i], e.res_values[i]
        setattr(keys, LAYER_TAG, self)
        self.keys, self.values = keys, values
        return keys, values

    # -- what the attention function needs -----------------------------------

    @property
    def num_history_tokens(self) -> int:
        a = self.engine._arena[self.layer_idx]
        return a.used if a is not None else 0

    def history(self):
        """(packed_k, scales_k, zeros_k, packed_v, scales_v, zeros_v) views of
        the quantized prefix, or None. Tokens of a pending flush are not in it
        yet — they are still part of the residual returned by update()."""
        a = self.engine._arena[self.layer_idx]
        if a is None or a.used == 0:
            return None
        return a.used_keys() + a.used_values()

    # -- Cache layer contract -------------------------------------------------

    def get_seq_length(self) -> int:
        e, i = self.engine, self.layer_idx
        res = e.res_keys[i]
        return e._quant_token_count[i] + (res.shape[2] if res is not None else 0)

    def get_mask_sizes(self, query_length: int):
        return self.get_seq_length() + query_length, 0

    def get_max_length(self) -> int:
        return -1

    def reset(self) -> None:
        # Cache.reset() calls this per layer; the engine resets all layers at
        # once, so only layer 0 forwards the call.
        if self.layer_idx == 0:
            self.engine.reset()
        self.keys = self.values = None
        self.is_initialized = False

    def reorder_cache(self, beam_idx):
        raise NotImplementedError("KIVI cache does not support beam search yet")

    def crop(self, max_length: int):
        raise NotImplementedError("KIVI cache does not support cropping")

    def batch_repeat_interleave(self, repeats: int):
        raise NotImplementedError("KIVI cache does not support batch expansion")

    def batch_select_indices(self, indices):
        raise NotImplementedError("KIVI cache does not support batch selection")

    def offload(self):
        pass        # the packed store already lives in host memory

    def prefetch(self):
        pass


class KiviHFCache(Cache):
    """``transformers.Cache`` backed by a ``KiviCache`` engine. Build it with
    ``kivi_sycl.enable_kivi(model)`` (which also selects the "kivi" attention)
    or ``KiviHFCache(KiviCache.from_model(model))``."""

    def __init__(self, engine: KiviCache):
        super().__init__(layers=[KiviLayer(engine, i) for i in range(engine.num_layers)])
        self.kivi = engine

    def get_stats(self, layer: Optional[int] = 0) -> dict:
        """Measured memory stats; see ``KiviCache.get_stats``."""
        return self.kivi.get_stats(layer=layer)

    def harvest_pending(self):
        self.kivi.harvest_pending()
