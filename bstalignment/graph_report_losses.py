from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

try:
    from .losses import nt_xent_loss
except ImportError:
    from losses import nt_xent_loss


def masked_regression_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    loss_type: str = "smooth_l1",
) -> torch.Tensor:
    width = min(pred.size(1), target.size(1))
    pred = pred[:, :width]
    target = target[:, :width]
    mask = mask[:, :width]
    if target.ndim == 2 and pred.ndim == 3 and pred.size(-1) == 1:
        target = target.unsqueeze(-1)
    if mask.ndim == 2 and pred.ndim == 3:
        mask = mask.unsqueeze(-1).expand_as(pred)
    if loss_type == "mse":
        loss = (pred - target) ** 2
    elif loss_type == "mae":
        loss = torch.abs(pred - target)
    else:
        loss = F.smooth_l1_loss(pred, target, reduction="none")
    mask_f = mask.to(loss.dtype)
    return (loss * mask_f).sum() / mask_f.sum().clamp(min=1.0)


def graph_report_loss(
    out: Dict[str, torch.Tensor],
    y: torch.Tensor,
    mask: torch.Tensor,
    weights: Dict[str, float],
    loss_type: str = "smooth_l1",
) -> Dict[str, torch.Tensor]:
    pred = out["pred"]
    reg = masked_regression_loss(pred, y, mask, loss_type=loss_type)
    align = nt_xent_loss(out["context"], out["text_repr"], temperature=weights.get("temperature", 0.2))
    total = reg + weights.get("align", 0.01) * align
    return {"total": total, "reg": reg.detach(), "align": align.detach()}


@torch.no_grad()
def regression_metrics(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
    width = min(pred.size(1), target.size(1))
    pred = pred[:, :width]
    target = target[:, :width]
    mask = mask[:, :width]
    if target.ndim == 2 and pred.ndim == 3 and pred.size(-1) == 1:
        target = target.unsqueeze(-1)
    if mask.ndim == 2 and pred.ndim == 3:
        mask = mask.unsqueeze(-1).expand_as(pred)
    diff = pred - target
    valid = mask.bool()
    if not valid.any():
        return {"mae": 0.0, "rmse": 0.0, "mse": 0.0}
    vals = diff[valid]
    mse = float((vals**2).mean().cpu())
    mae = float(vals.abs().mean().cpu())
    return {"mse": mse, "mae": mae, "rmse": mse**0.5}
