"""Op mode: KIVI as one stateful torch op per attention layer (Phase 5 S1.7).

In the default integration (``enable_kivi``) the cache's ``update()`` and the
"kivi" attention are Python that calls into the engine — fine eagerly, but
under ``torch.compile`` they force two graph breaks per layer (Phase 4
measured 48 per step on Qwen2.5-0.5B, and a 0.91–0.96× "speedup").

Op mode moves all KIVI work inside ``torch.ops.kivi.layer_attention``:

  * the cache layers pass k/v straight through (nothing to trace);
  * the "kivi_op" attention makes one op call: append this step's k/v to the
    C++ engine and attend over packed history + residual;
  * "kivi_op" is registered *without* a mask function, so transformers' mask
    builder exits early instead of asking the cache for its length (a Python
    int that would make Dynamo recompile every step); the op applies causality
    itself (bottom-right causal mask when earlier tokens exist);
  * the engine id (a compile-time constant) and a state token tensor (declared
    mutated, so calls are neither dropped nor reordered) reach the attention
    function as model kwargs — pass ``**cache.model_kwargs`` to every call.

The engine is the same kivi::Engine the default path uses, so results match
the default fused path. Batch size 1; no padding masks (the op cannot see
them) — use ``enable_kivi`` for those.
"""

import torch
from transformers.cache_utils import Cache, CacheLayerMixin
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from ..cache.manager import KiviCache
from ..extension.loader import kivi_native

__all__ = ["OP_ATTN_NAME", "KiviOpCache", "enable_kivi_op", "kivi_op_attention"]

OP_ATTN_NAME = "kivi_op"


class KiviOpLayer(CacheLayerMixin):
    """Pass-through cache layer: the engine sees k/v inside the op."""

    is_sliding = False
    is_compileable = True

    def __init__(self, cache: KiviCache, layer_idx: int):
        super().__init__()
        self.cache = cache
        self.layer_idx = layer_idx

    def lazy_initialization(self, key_states, value_states) -> None:
        self.dtype, self.device = key_states.dtype, key_states.device
        self.is_initialized = True

    def update(self, key_states, value_states, *args, **kwargs):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        return key_states, value_states

    # Read outside the compiled region (generate(), callers) — never in the graph.
    def get_seq_length(self) -> int:
        return self.cache._native.seq_length(self.layer_idx)

    def get_mask_sizes(self, query_length: int):
        return self.get_seq_length() + query_length, 0

    def get_max_length(self) -> int:
        return -1

    def reset(self) -> None:
        if self.layer_idx == 0:
            self.cache.reset()
        self.is_initialized = False

    def reorder_cache(self, beam_idx):
        raise NotImplementedError("KIVI cache does not support beam search yet")

    def crop(self, max_length: int):
        raise NotImplementedError("KIVI cache does not support cropping")


class KiviOpCache(Cache):
    """Cache for op mode. Pass ``**cache.model_kwargs`` to every model call."""

    def __init__(self, cache: KiviCache):
        if cache._native is None:
            raise ValueError("op mode needs the native engine (KiviCache(engine='native'))")
        super().__init__(layers=[KiviOpLayer(cache, i) for i in range(cache.num_layers)])
        self.kivi = cache
        self.engine_id = kivi_native.register_engine(cache._native)
        self.state = torch.zeros(1, dtype=torch.int64)

    @property
    def model_kwargs(self) -> dict:
        return {"kivi_engine_id": self.engine_id, "kivi_state": self.state}

    def get_stats(self, layer: int = 0) -> dict:
        return self.kivi.get_stats(layer)

    def harvest_pending(self):
        self.kivi.harvest_pending()

    def close(self):
        if self.engine_id is not None:
            kivi_native.unregister_engine(self.engine_id)
            self.engine_id = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def kivi_op_attention(module, query, key, value, attention_mask, scaling=None,
                      dropout=0.0, **kwargs):
    engine_id = kwargs.get("kivi_engine_id")
    if engine_id is None:
        # No KIVI cache in this call (e.g. a forward without past_key_values).
        return sdpa_attention_forward(module, query, key, value, attention_mask,
                                      scaling=scaling, dropout=dropout, **kwargs)
    if scaling is None:
        scaling = query.shape[-1] ** -0.5
    out = torch.ops.kivi.layer_attention(engine_id, module.layer_idx, query, key, value,
                                         float(scaling), kwargs["kivi_state"])
    return out, None


def enable_kivi_op(model, max_tokens: int = 0, **cache_kwargs) -> KiviOpCache:
    """Switch ``model`` to op mode and return a fresh cache. Call the model as
    ``model(ids, past_key_values=cache, position_ids=pos, **cache.model_kwargs)``
    with ``pos`` the absolute positions of ``ids`` ([1, T] int64). Without
    explicit ``position_ids`` transformers 5.x derives them from
    ``past_key_values.get_seq_length()`` — a call into the engine that breaks
    a compiled graph and changes every step (one recompile per token)."""
    # Deliberately no mask-function registration (see the module docstring).
    ALL_ATTENTION_FUNCTIONS.register(OP_ATTN_NAME, kivi_op_attention)
    model.set_attn_implementation(OP_ATTN_NAME)
    cache = KiviCache.from_model(model, engine="native", **cache_kwargs)
    if max_tokens:
        cache.reserve(max_tokens)
    return KiviOpCache(cache)
