"""
kivi_sycl — Plug-and-play KIVI 2-bit KV Cache for HuggingFace Models
======================================================================
Drop-in replacement for the standard KV cache in any HuggingFace
CausalLM. Works with GPT-2, LLaMA, Mistral, Phi, Qwen, Falcon, etc.

Usage:
    from kivi_sycl import KiviCache, generate

    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

    # Option 1: One-liner generation
    text = generate(model, tokenizer, "Once upon a time", max_new_tokens=200)

    # Option 2: Manual control
    cache = KiviCache.from_model(model)
    # ... use cache.add_tokens() / cache.get_full_cache() in your own loop

Paper: "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache"
       (arXiv:2402.02750)
"""

from ._version import __version__
from .cache.manager import KiviCache
from .generation.generate import generate

__all__ = ["KiviCache", "generate", "__version__"]
