"""Auto-detection of KV-cache-relevant shape info from HuggingFace models."""

__all__ = ["detect_model_config"]


def detect_model_config(model) -> dict:
    """
    Auto-detect num_layers, num_heads, and head_dim from any HuggingFace model.
    Supports GPT-2, GPT-J, GPT-NeoX, LLaMA, Mistral, Phi, Qwen, Falcon,
    Gemma, StableLM, MPT, BLOOM, OPT, and more.
    """
    config = model.config

    # --- num_layers ---
    for attr in ("n_layer", "num_hidden_layers", "num_layers", "n_layers"):
        if hasattr(config, attr):
            num_layers = getattr(config, attr)
            break
    else:
        raise ValueError(
            f"Cannot detect num_layers from {type(config).__name__}. "
            f"Pass num_layers= explicitly.")

    # --- num_heads (for KV — may differ from query heads in GQA models) ---
    num_kv_heads = None
    for attr in ("num_key_value_heads", "num_kv_heads"):
        if hasattr(config, attr):
            num_kv_heads = getattr(config, attr)
            break
    if num_kv_heads is None:
        for attr in ("n_head", "num_attention_heads", "num_heads"):
            if hasattr(config, attr):
                num_kv_heads = getattr(config, attr)
                break
    if num_kv_heads is None:
        raise ValueError(
            f"Cannot detect num_heads from {type(config).__name__}. "
            f"Pass num_kv_heads= explicitly.")

    # --- head_dim ---
    if hasattr(config, "head_dim"):
        head_dim = config.head_dim
    else:
        hidden_size = getattr(config, "hidden_size",
                              getattr(config, "n_embd", None))
        num_q_heads = getattr(config, "num_attention_heads",
                              getattr(config, "n_head", None))
        if hidden_size and num_q_heads:
            head_dim = hidden_size // num_q_heads
        else:
            raise ValueError(
                f"Cannot detect head_dim from {type(config).__name__}. "
                f"Pass head_dim= explicitly.")

    return {
        "num_layers": num_layers,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
    }
