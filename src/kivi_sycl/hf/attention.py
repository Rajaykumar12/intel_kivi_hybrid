"""The "kivi" attention implementation for transformers' AttentionInterface.

Registered under the name "kivi"; select it with
``model.set_attn_implementation("kivi")`` (``enable_kivi`` does this). Each
attention module then calls ``kivi_attention_forward`` with the key/value
tensors returned by ``KiviLayer.update()`` — the FP residual only — and this
function brings in the layer's packed 2-bit history.

Reference path (this module): dequantize the history, concatenate it with
the residual and run transformers' own SDPA attention. It is numerically the
same computation as serving a full FP cache, and is the oracle the fused
kernel path is tested against.
"""

import torch
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, sdpa_mask
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

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


def kivi_attention_forward(module, query, key, value, attention_mask,
                           scaling=None, dropout=0.0, **kwargs):
    layer = getattr(key, LAYER_TAG, None)
    if layer is None:
        # Not a KIVI-managed tensor (no cache passed, or cross-attention):
        # plain SDPA over exactly what we were given.
        return sdpa_attention_forward(module, query, key, value, attention_mask,
                                      scaling=scaling, dropout=dropout, **kwargs)

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
