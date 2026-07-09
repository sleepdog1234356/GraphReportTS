from __future__ import annotations

import torch
import torch.nn.functional as F


def nt_xent_loss(a: torch.Tensor, b: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """Symmetric InfoNCE between time-series and text representations."""
    if a.size(0) <= 1:
        return torch.tensor(0.0, device=a.device)
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    logits = a @ b.t() / temperature
    labels = torch.arange(a.size(0), device=a.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
