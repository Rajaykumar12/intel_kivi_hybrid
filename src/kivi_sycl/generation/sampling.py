"""Token sampling strategies for the generate() loop."""

from typing import Optional

import torch

__all__ = ["sample_token"]


def sample_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: Optional[int],
    do_sample: bool,
) -> torch.Tensor:
    """Sample or argmax from logits."""
    if not do_sample:
        return logits.argmax(dim=-1).unsqueeze(1)

    logits = logits / max(temperature, 1e-8)
    if top_k is not None:
        topk_vals, _ = logits.topk(top_k, dim=-1)
        logits[logits < topk_vals[:, -1:]] = float("-inf")
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
