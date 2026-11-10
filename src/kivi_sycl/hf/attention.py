"""The "kivi" attention implementation for transformers' AttentionInterface.

Registered under the name "kivi"; select it with
``model.set_attn_implementation("kivi")`` (``enable_kivi`` does this). Each
attention module then calls ``kivi_attention_forward`` with the key/value
tensors returned by ``KiviLayer.update()`` — the FP residual only — and this
function brings in the layer's packed 2-bit history.

Two paths:
- fused (decode steps): ``torch.ops.kivi.fused_decode_attention`` reads the
  2-bit codes and per-group scales directly — the dequantized history never
  exists. ~5x faster than the reference path per layer (Phase 2 §B.4).
- reference (prefill, padded/sliding masks, or ``fused=False``): dequantize
  the history, concatenate the residual, run transformers' own SDPA. Exactly
  the computation of serving a full FP cache; the fused path is tested
  against it.
"""

import torch
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, sdpa_mask
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from ..extension.loader import kivi_native  # noqa: F401  (registers torch.ops.kivi)
from .cache import LAYER_TAG

__all__ = ["ATTN_NAME", "kivi_attention_forward", "register_kivi_attention",
           "dequantized_history"]

ATTN_NAME = "kivi"


def dequantized_history(layer):
    """[B, H, n_hist, D] FP keys/values of the layer's packed history, or
    (None, None) if nothing has been quantized yet."""
    hist = layer.history()
    if hist is None:
        return None, None
    e = layer.engine
    k = e._dequant_keys_cpu(*hist[:3])
    v = e._dequant_values_cpu(*hist[3:])
    return k, v


def _mask_is_trivial(mask) -> bool:
    """True if `mask` hides nothing (None, all-True bool, or all-zero additive)."""
    if mask is None:
        return True
    if mask.dtype == torch.bool:
        return bool(mask.all())
    return bool((mask == 0).all())


# kwargs that change the attention math in ways the fused kernel does not model.
_FUSED_UNSUPPORTED_KWARGS = ("sliding_window", "softcap", "s_aux", "sinks")


def _can_fuse(layer, query, attention_mask, dropout, kwargs) -> bool:
    return (layer.fused
            and query.shape[2] == 1
            and query.device.type == "cpu"
            and dropout == 0.0
            and layer.num_history_tokens > 0
            and hasattr(torch.ops.kivi, "fused_decode_attention")
            and not any(kwargs.get(k) is not None for k in _FUSED_UNSUPPORTED_KWARGS)
            and _mask_is_trivial(attention_mask))


def kivi_attention_forward(module, query, key, value, attention_mask,
                           scaling=None, dropout=0.0, **kwargs):
    layer = getattr(key, LAYER_TAG, None)
    if layer is None:
        # Not a KIVI-managed tensor (no cache passed, or cross-attention):
        # plain SDPA over exactly what we were given.
        return sdpa_attention_forward(module, query, key, value, attention_mask,
                                      scaling=scaling, dropout=dropout, **kwargs)

    if _can_fuse(layer, query, attention_mask, dropout, kwargs):
        if scaling is None:
            scaling = query.shape[-1] ** -0.5
        # Through the dispatcher (not the pybind function) so torch.compile /
        # torch.export see a real op rather than an opaque Python call.
        out = torch.ops.kivi.fused_decode_attention(
            query, key, value, *layer.history(), float(scaling),
            layer.engine._G[layer.layer_idx])
        layer.calls["fused"] += 1
        return out, None

    layer.calls["reference"] += 1
    hist_k, hist_v = dequantized_history(layer)
    if hist_k is not None:
        key = torch.cat([hist_k, key], dim=2)
        value = torch.cat([hist_v, value], dim=2)
    return sdpa_attention_forward(module, query, key, value, attention_mask,
                                  scaling=scaling, dropout=dropout, **kwargs)


def register_kivi_attention():
    """Register "kivi" with transformers' attention and mask interfaces.
    Masks are built exactly as for "sdpa". Idempotent."""
    ALL_ATTENTION_FUNCTIONS.register(ATTN_NAME, kivi_attention_forward)
    ALL_MASK_ATTENTION_FUNCTIONS.register(ATTN_NAME, sdpa_mask)
