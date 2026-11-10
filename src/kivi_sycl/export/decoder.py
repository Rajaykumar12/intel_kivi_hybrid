"""Exportable decoder: a model forward with every KIVI cache operation inside
torch.ops.kivi.layer_attention (Phase 5 A.1).

``KiviExportDecoder(model).forward(input_ids, position_ids, state) -> logits``
is a pure function of its inputs as far as tracing is concerned:

  * a pass-through stub cache stands in for past_key_values, so transformers
    calls the attention per layer but nothing reads cache state in the graph;
  * the "kivi_op" attention (no mask function) makes one layer_attention call
    per layer against engine id ``RUNTIME_ENGINE_ID`` (0), baked in as a
    constant — the C++ runtime registers its engine under that id;
  * ``state`` is the op's mutated token tensor; it keeps the side-effecting
    calls in order through export and compilation.

Measured on TinyLlama-1.1B (2026-11-10): decode and prefill (dynamic T)
programs export in ~2–3 s, contain exactly one layer_attention per layer, and
reproduce eager op mode bit for bit over 40 decode steps.
"""

import torch
from transformers.cache_utils import Cache, CacheLayerMixin
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from ..hf.compiled import OP_ATTN_NAME, kivi_op_attention

__all__ = ["RUNTIME_ENGINE_ID", "KiviExportDecoder", "export_programs"]

RUNTIME_ENGINE_ID = 0


class _PassThroughLayer(CacheLayerMixin):
    """Returns k/v unchanged; the engine sees them inside the op."""

    is_sliding = False

    def lazy_initialization(self, key_states, value_states):
        self.is_initialized = True

    def update(self, key_states, value_states, *args, **kwargs):
        return key_states, value_states

    def get_seq_length(self):
        raise RuntimeError("export stub cache: sequence length must not be read inside "
                           "the graph — pass position_ids explicitly")

    def get_mask_sizes(self, query_length):
        raise RuntimeError("export stub cache: masks must not be built inside the graph")

    def get_max_length(self):
        return -1


def _num_layers(config):
    for attr in ("num_hidden_layers", "n_layer", "num_layers"):
        if hasattr(config, attr):
            return getattr(config, attr)
    raise ValueError(f"cannot find the layer count in {type(config).__name__}")


class KiviExportDecoder(torch.nn.Module):
    """forward(input_ids [1,T], position_ids [1,T], state [1] int64) -> logits [1,T,V]."""

    def __init__(self, model, engine_id: int = RUNTIME_ENGINE_ID):
        super().__init__()
        ALL_ATTENTION_FUNCTIONS.register(OP_ATTN_NAME, kivi_op_attention)
        model.set_attn_implementation(OP_ATTN_NAME)
        self.model = model
        self.engine_id = engine_id
        self.num_layers = _num_layers(model.config)

    def forward(self, input_ids, position_ids, state):
        stub = Cache(layers=[_PassThroughLayer() for _ in range(self.num_layers)])
        return self.model(input_ids=input_ids, position_ids=position_ids,
                          past_key_values=stub, use_cache=True,
                          kivi_engine_id=self.engine_id, kivi_state=state).logits


def export_programs(model, max_prompt: int, example_prompt: int = 16):
    """Export (prefill, decode) ExportedPrograms for ``model``.

    prefill: input_ids [1, T] with T in [2, max_prompt] (dynamic);
    decode:  input_ids [1, 1].
    """
    dec = KiviExportDecoder(model).eval()
    state = torch.zeros(1, dtype=torch.int64)
    with torch.no_grad():
        decode = torch.export.export(
            dec, (torch.zeros(1, 1, dtype=torch.long), torch.tensor([[example_prompt]]), state),
            strict=False)
        T = torch.export.Dim("T", min=2, max=max_prompt)
        n = min(example_prompt, max_prompt)
        prefill = torch.export.export(
            dec, (torch.zeros(1, n, dtype=torch.long), torch.arange(n)[None], state),
            strict=False, dynamic_shapes=({1: T}, {1: T}, None))
    return prefill, decode
