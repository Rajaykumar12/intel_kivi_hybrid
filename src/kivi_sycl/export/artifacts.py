"""Artifact directory for the Python-free runtime (Phase 5 A.3).

``write_artifacts(model_id_or_dir, out_dir)`` exports a HuggingFace model once
(offline, in Python) into a self-contained directory:

    model.pt2                    code-only AOTInductor package (a few MB)
    model.safetensors            the model's own weights, copied
    extra_constants.safetensors  package constants absent from the weights
                                 (e.g. RoPE inv_freq buffers), fp32
    constants.json               package constant -> {file, tensor, dtype, shape}
    kivi_config.json             engine + model configuration
    tokenizer.json, tokenizer_config.json, generation_config.json (if present)

``load_artifacts(out_dir)`` is the Python mirror of what the C++ runtime does:
read the manifest, load each constant from its safetensors file (converting
e.g. bf16 -> fp32, which is exact), create a kivi::Engine from
kivi_config.json, register it as the runtime id, and return a callable
``step(input_ids, position_ids) -> logits``.
"""

import json
import os
import shutil

import torch

from ..config.model_config import detect_model_config
from .decoder import RUNTIME_ENGINE_ID, compile_package, export_program, exported_constants

__all__ = ["write_artifacts", "load_artifacts", "ARTIFACT_VERSION"]

ARTIFACT_VERSION = 1
_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
_COPY_FILES = ("tokenizer.json", "tokenizer_config.json", "generation_config.json",
               "special_tokens_map.json", "tokenizer.model")


def _snapshot_dir(model_id_or_dir: str) -> str:
    if os.path.isdir(model_id_or_dir):
        return model_id_or_dir
    from huggingface_hub import snapshot_download
    return snapshot_download(model_id_or_dir)


def _safetensors_index(snapshot: str) -> dict:
    """tensor name -> safetensors file (relative to `snapshot`)."""
    from safetensors import safe_open
    files = [f for f in os.listdir(snapshot) if f.endswith(".safetensors")]
    if not files:
        raise FileNotFoundError(f"no .safetensors weights in {snapshot}")
    index = {}
    for f in sorted(files):
        with safe_open(os.path.join(snapshot, f), framework="pt") as st:
            for k in st.keys():
                index[k] = f
    return index


def _dtype_name(dt: torch.dtype) -> str:
    return {v: k for k, v in _DTYPES.items()}[dt]


def write_artifacts(model_id_or_dir: str, out_dir: str, *, max_tokens: int = 2048,
                    group_size=32, residual_length: int = 128,
                    metadata_dtype: str = "float16") -> dict:
    """Export ``model_id_or_dir`` into ``out_dir``; returns kivi_config."""
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM

    snapshot = _snapshot_dir(model_id_or_dir)
    os.makedirs(out_dir, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(snapshot, dtype=torch.float32).eval()

    ep = export_program(model, max_tokens=max_tokens)
    package_path = compile_package(ep, os.path.join(out_dir, "model.pt2"))
    # Exactly the constants the compiled code reads — Inductor may fold some
    # exported buffers away (e.g. rotary_emb.original_inv_freq), and
    # load_constants rejects names the package does not know.
    needed = torch._inductor.aoti_load_package(package_path).get_constant_fqns()
    available = exported_constants(ep)
    constants = {f: available[f] for f in needed}

    # Map every package constant to where the runtime will read it from.
    index = _safetensors_index(snapshot)
    by_storage = {}          # data_ptr -> HF parameter name, to resolve tied weights
    for name, p in list(model.named_parameters()) + list(model.named_buffers()):
        if name in index:
            by_storage.setdefault(p.data_ptr(), name)
    manifest, extra = {}, {}
    for fqn, t in constants.items():
        hf = fqn.removeprefix("model.")
        src = hf if hf in index else by_storage.get(t.data_ptr())
        if src is not None:
            manifest[fqn] = {"file": index[src], "tensor": src}
        else:
            key = fqn.replace(".", "__")
            extra[key] = t.detach().contiguous().clone()
            manifest[fqn] = {"file": "extra_constants.safetensors", "tensor": key}
        manifest[fqn].update(dtype=_dtype_name(t.dtype), shape=list(t.shape))
    if extra:
        save_file(extra, os.path.join(out_dir, "extra_constants.safetensors"))
    for f in sorted(set(index.values())):
        shutil.copyfile(os.path.join(snapshot, f), os.path.join(out_dir, f))
    for f in _COPY_FILES:
        if os.path.exists(os.path.join(snapshot, f)):
            shutil.copyfile(os.path.join(snapshot, f), os.path.join(out_dir, f))
    with open(os.path.join(out_dir, "constants.json"), "w") as fh:
        json.dump(manifest, fh, indent=1, sort_keys=True)

    cfg = detect_model_config(model)
    gs = [group_size] * cfg["num_layers"] if isinstance(group_size, int) else list(group_size)
    eos = model.generation_config.eos_token_id if model.generation_config else None
    kivi_config = {
        "artifact_version": ARTIFACT_VERSION,
        "torch_version": torch.__version__,
        "engine_id": RUNTIME_ENGINE_ID,
        "num_layers": cfg["num_layers"], "num_kv_heads": cfg["num_kv_heads"],
        "head_dim": cfg["head_dim"],
        "num_attention_heads": getattr(model.config, "num_attention_heads", None),
        "residual_length": residual_length, "group_size": gs,
        "metadata_dtype": metadata_dtype, "model_dtype": "float32",
        "max_tokens": max_tokens, "vocab_size": model.config.vocab_size,
        "bos_token_id": getattr(model.config, "bos_token_id", None),
        "eos_token_ids": eos if isinstance(eos, list) else ([eos] if eos is not None else []),
        "model_type": model.config.model_type,
    }
    with open(os.path.join(out_dir, "kivi_config.json"), "w") as fh:
        json.dump(kivi_config, fh, indent=1)
    return kivi_config


class _LoadedArtifacts:
    """step(input_ids [1,T], position_ids [1,T]) -> logits [1,T,V]."""

    def __init__(self, out_dir, device):
        from safetensors import safe_open

        from .. import _C
        with open(os.path.join(out_dir, "kivi_config.json")) as fh:
            self.config = json.load(fh)
        with open(os.path.join(out_dir, "constants.json")) as fh:
            manifest = json.load(fh)
        self.package = torch._inductor.aoti_load_package(os.path.join(out_dir, "model.pt2"))
        needed = set(self.package.get_constant_fqns())
        if needed != set(manifest):
            raise ValueError(f"constants.json does not match model.pt2: missing "
                             f"{sorted(needed - set(manifest))[:5]}, unexpected "
                             f"{sorted(set(manifest) - needed)[:5]}")
        handles, constants = {}, {}
        for fqn, m in manifest.items():
            if m["file"] not in handles:
                handles[m["file"]] = safe_open(os.path.join(out_dir, m["file"]), framework="pt")
            t = handles[m["file"]].get_tensor(m["tensor"])
            constants[fqn] = t.to(_DTYPES[m["dtype"]]).contiguous()
        self.package.load_constants(constants, check_full_update=True, user_managed=True)
        self._constants = constants          # user-managed: keep them alive
        c = self.config
        self.engine = _C.Engine(num_layers=c["num_layers"], head_dim=c["head_dim"],
                                num_kv_heads=c["num_kv_heads"],
                                residual_length=c["residual_length"],
                                group_size=c["group_size"],
                                metadata_dtype=c["metadata_dtype"], device=device)
        self.engine_id = _C.register_engine(self.engine, c["engine_id"])
        self.state = torch.zeros(1, dtype=torch.int64)

    def reserve(self, total_tokens):
        self.engine.reserve(total_tokens)

    def __call__(self, input_ids, position_ids):
        return self.package(input_ids, position_ids, self.state)

    def close(self):
        from .. import _C
        if self.engine_id is not None:
            _C.unregister_engine(self.engine_id)
            self.engine_id = None


def load_artifacts(out_dir: str, device: str = "xpu") -> _LoadedArtifacts:
    """Load an artifact directory the way the C++ runtime does. Call
    ``.close()`` to release the runtime engine id."""
    return _LoadedArtifacts(out_dir, device)
