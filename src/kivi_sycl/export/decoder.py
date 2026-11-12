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

Measured on TinyLlama-1.1B (2026-11-10): the program exports in a few
seconds, contains exactly one layer_attention per layer, and reproduces
eager op mode bit for bit over 40 decode steps.
"""

import torch
from transformers.cache_utils import Cache, CacheLayerMixin
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from ..hf.compiled import OP_ATTN_NAME, kivi_op_attention

__all__ = ["RUNTIME_ENGINE_ID", "KiviExportDecoder", "export_program", "compile_package",
           "exported_constants", "load_package"]

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
    """forward(input_ids [1,T], position_ids [1,T], state [1] int64) -> logits.

    ``last_logits_only`` (default): logits [1, 1, V] for the last position
    only (transformers' ``logits_to_keep=1``, what ``generate()`` uses) —
    all a generation loop reads. Otherwise [1, T, V]: at an 8k prefill of a
    151k-vocabulary model that is a 5 GB fp32 tensor plus ~270 GFLOP of
    lm_head work for positions nobody samples from."""

    def __init__(self, model, engine_id: int = RUNTIME_ENGINE_ID, last_logits_only: bool = True):
        super().__init__()
        self.logits_to_keep = 1 if last_logits_only else 0
        ALL_ATTENTION_FUNCTIONS.register(OP_ATTN_NAME, kivi_op_attention)
        model.set_attn_implementation(OP_ATTN_NAME)
        self.model = model
        self.engine_id = engine_id
        self.num_layers = _num_layers(model.config)

    def forward(self, input_ids, position_ids, state):
        stub = Cache(layers=[_PassThroughLayer() for _ in range(self.num_layers)])
        return self.model(input_ids=input_ids, position_ids=position_ids,
                          past_key_values=stub, use_cache=True,
                          logits_to_keep=self.logits_to_keep,
                          kivi_engine_id=self.engine_id, kivi_state=state).logits


def export_program(model, max_tokens: int, example_tokens: int = 8,
                   last_logits_only: bool = True):
    """Export ONE ExportedProgram serving both prefill and decode for ``model``:
    input_ids [1, T] with T dynamic in [1, max_tokens]. T == 1 is a decode
    step; larger T a (chunked) prefill. A single program means a single
    compiled package and a single copy of the weights for the runtime.
    (Dim.DYNAMIC would let export assume T >= 2; the explicit min=1 keeps the
    decode case inside the traced range.)"""
    dec = KiviExportDecoder(model, last_logits_only=last_logits_only).eval()
    state = torch.zeros(1, dtype=torch.int64)
    n = max(2, min(example_tokens, max_tokens))
    T = torch.export.Dim("T", min=1, max=max_tokens)
    with torch.no_grad():
        return torch.export.export(
            dec, (torch.zeros(1, n, dtype=torch.long), torch.arange(n)[None], state),
            strict=False, dynamic_shapes=({1: T}, {1: T}, None))


def compile_package(exported_program, package_path: str) -> str:
    """AOTInductor-compile ``exported_program`` into a *code-only* package
    (.pt2, a few MB): the weights are not serialized into it. Supply them at
    load time with ``load_constants(..., user_managed=True)`` — the runtime
    reads them from the model's own safetensors (see ``constant_sources``).

    Why code-only: serializing the weights into the package holds an extra
    full copy during compilation — on this 15 GB laptop that ran TinyLlama-
    1.1B (4.4 GB fp32) into a system OOM and even Qwen2.5-0.5B to the limit.
    (Inductor stages builds in TORCHINDUCTOR_CACHE_DIR, default /tmp — a RAM
    tmpfs here; point it at a disk path *without spaces*, Inductor does not
    quote paths in its g++ command.)"""
    return torch._inductor.aoti_compile_and_package(
        exported_program, package_path=package_path,
        inductor_configs={"aot_inductor.package_constants_in_so": False})


def exported_constants(exported_program) -> dict:
    """{fqn: tensor} for every constant a compiled package expects."""
    return {**exported_program.state_dict, **exported_program.constants}


def load_package(package_path: str, constants: dict):
    """Load a code-only package and hand it its weights (no copy)."""
    pkg = torch._inductor.aoti_load_package(package_path)
    need = pkg.get_constant_fqns()
    missing = [f for f in need if f not in constants]
    if missing:
        raise KeyError(f"constants missing for package: {missing[:5]}")
    pkg.load_constants({f: constants[f] for f in need}, check_full_update=True,
                       user_managed=True)
    return pkg
