"""High-level generate() — one-liner plug-and-play text generation."""

from typing import Optional

import torch

from ..cache.manager import KiviCache
from .sampling import sample_token

__all__ = ["generate"]


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    group_size: int = 32,
    residual_length: int = 128,
    device: str = "xpu",
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    do_sample: bool = False,
    verbose: bool = False,
) -> str:
    """
    Generate text with KIVI 2-bit KV cache quantization.

    Works with any HuggingFace CausalLM model. The model stays on CPU;
    quantization kernels run on Intel XPU via SYCL.

    Args:
        model:            HuggingFace CausalLM (on CPU).
        tokenizer:        Matching tokenizer.
        prompt:           Input text.
        max_new_tokens:   Number of tokens to generate.
        group_size:       Quantization group size G (default 32).
        residual_length:  FP32 residual window R (default 128).
        device:           XPU device for quantization (default "xpu").
        temperature:      Sampling temperature (1.0 = greedy with do_sample=False).
        top_k:            Top-k filtering (None = disabled).
        do_sample:        Whether to sample (False = greedy argmax).
        verbose:          Print progress dots.

    Returns:
        Generated text (prompt + completion).

    Example:
        >>> from kivi_sycl import generate
        >>> text = generate(model, tokenizer, "Hello world", max_new_tokens=100)
    """
    cache = KiviCache.from_model(
        model, device=device, group_size=group_size,
        residual_length=residual_length,
    )

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    # Prefill
    out = model(input_ids, use_cache=True)
    cache.add_tokens(out.past_key_values)
    next_id = sample_token(out.logits[:, -1, :], temperature, top_k, do_sample)
    all_ids = [next_id]

    # Decode
    for step in range(max_new_tokens):
        past = cache.get_full_cache()
        out = model(next_id, past_key_values=past, use_cache=True)
        cache.add_tokens(out.past_key_values)
        next_id = sample_token(out.logits[:, -1, :], temperature, top_k, do_sample)
        all_ids.append(next_id)

        if verbose and (step + 1) % 10 == 0:
            print(".", end="", flush=True)

        # Stop on EOS
        if tokenizer.eos_token_id is not None and next_id.item() == tokenizer.eos_token_id:
            break

    if verbose:
        print()

    generated_ids = torch.cat(all_ids, dim=1)
    full_ids = torch.cat([input_ids, generated_ids], dim=1)
    return tokenizer.decode(full_ids[0], skip_special_tokens=True)
