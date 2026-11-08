"""Native transformers integration: KIVI as a Cache + registered attention.

    from kivi_sycl import enable_kivi
    cache = enable_kivi(model)                  # selects the "kivi" attention
    out = model.generate(input_ids, past_key_values=cache, max_new_tokens=200)
    print(cache.get_stats())
"""

from ..cache.manager import KiviCache
from .attention import ATTN_NAME, kivi_attention_forward, register_kivi_attention
from .cache import KiviHFCache, KiviLayer

__all__ = ["enable_kivi", "KiviHFCache", "KiviLayer", "kivi_attention_forward",
           "register_kivi_attention", "ATTN_NAME"]


def enable_kivi(model, max_tokens: int = 0, **cache_kwargs) -> KiviHFCache:
    """Switch ``model`` to the "kivi" attention and return a fresh KIVI cache
    to pass as ``past_key_values``. ``cache_kwargs`` go to
    ``KiviCache.from_model`` (group_size, residual_length, metadata_dtype,
    zero_copy, device, async_flush). ``max_tokens`` — prompt plus generated
    tokens, if known — sizes the packed store once (``KiviCache.reserve``).

    One cache per sequence: create a new one (or call ``.reset()``) for each
    generation."""
    register_kivi_attention()
    model.set_attn_implementation(ATTN_NAME)
    engine = KiviCache.from_model(model, **cache_kwargs)
    if max_tokens:
        engine.reserve(max_tokens)
    return KiviHFCache(engine)
