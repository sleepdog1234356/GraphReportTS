from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

try:
    from .models import decomposition_independence_loss
except ImportError:
    from models import decomposition_independence_loss


def nt_xent_loss(a: torch.Tensor, b: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """Symmetric InfoNCE between time-series and text representations."""
    if a.size(0) <= 1:
        return torch.tensor(0.0, device=a.device)
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    logits = a @ b.t() / temperature
    labels = torch.arange(a.size(0), device=a.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def battery_multitask_loss(
    out: Dict[str, torch.Tensor],
    y: torch.Tensor,
    aging_stage: torch.Tensor,
    weights: Dict[str, float],
) -> Dict[str, torch.Tensor]:
    """Multi-task loss for scalar or multi-horizon SOH forecasting.

    out["soh"] and y should both be [B, H]. H can be 1.
    """
    pred = out["soh"]
    if y.ndim == 1:
        y = y.unsqueeze(-1)
    if pred.ndim == 1:
        pred = pred.unsqueeze(-1)
    reg = F.smooth_l1_loss(pred, y)

    stage = F.cross_entropy(out["stage_logits"], aging_stage)
    anchor = F.cross_entropy(out["anchor_logits"], aging_stage)
    align = nt_xent_loss(out["ts_repr"], out["text_repr"], temperature=weights.get("temperature", 0.2))
    decomp = decomposition_independence_loss(out["trend"], out["residual"])

    # Physical range constraint; optional weak smoothness penalty over the predicted trajectory.
    range_penalty = torch.relu(pred - 1.10).mean() + torch.relu(0.50 - pred).mean()
    if pred.size(1) > 1:
        # Battery SOH usually declines slowly. We penalize strong predicted upward jumps but keep it weak
        # to allow local regeneration/measurement rebound.
        upward_jump = torch.relu(pred[:, 1:] - pred[:, :-1] - 0.003).mean()
    else:
        upward_jump = torch.tensor(0.0, device=pred.device)
    phys = range_penalty + 0.5 * upward_jump

    total = (
        weights.get("soh", 1.0) * reg
        + weights.get("stage", 0.2) * stage
        + weights.get("anchor", 0.2) * anchor
        + weights.get("align", 0.05) * align
        + weights.get("decomp", 0.01) * decomp
        + weights.get("phys", 0.01) * phys
    )
    return {
        "total": total,
        "soh": reg.detach(),
        "stage": stage.detach(),
        "anchor": anchor.detach(),
        "align": align.detach(),
        "decomp": decomp.detach(),
        "phys": phys.detach(),
    }


def _masked_smooth_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = F.smooth_l1_loss(pred, target, reduction="none")
    mask_f = mask.to(dtype=loss.dtype)
    return (loss * mask_f).sum() / mask_f.sum().clamp(min=1.0)


def battery_cycle_forecast_loss(
    out: Dict[str, torch.Tensor],
    y_now: torch.Tensor,
    y_future: torch.Tensor,
    future_mask: torch.Tensor,
    weights: Dict[str, float],
) -> Dict[str, torch.Tensor]:
    """Loss for current-cycle SOH estimation and variable-horizon forecasting."""
    pred_now = out["soh_now"]
    pred_future = out["soh_future"]
    if y_now.ndim > 1:
        y_now = y_now.squeeze(-1)

    width = min(pred_future.size(1), y_future.size(1))
    pred_future = pred_future[:, :width]
    y_future = y_future[:, :width]
    future_mask = future_mask[:, :width] & out.get("future_mask", torch.ones_like(future_mask[:, :width])).bool()

    loss_now = F.smooth_l1_loss(pred_now, y_now)
    loss_future = _masked_smooth_l1(pred_future, y_future, future_mask)
    align = nt_xent_loss(out["numeric_repr"], out["text_repr"], temperature=weights.get("temperature", 0.2))

    pred_all = torch.cat([pred_now.unsqueeze(1), pred_future], dim=1)
    all_mask = torch.cat(
        [torch.ones(pred_now.size(0), 1, dtype=torch.bool, device=pred_now.device), future_mask],
        dim=1,
    )
    range_loss = (
        torch.relu(pred_all - 1.10) + torch.relu(0.50 - pred_all)
    )
    range_loss = (range_loss * all_mask.to(range_loss.dtype)).sum() / all_mask.to(range_loss.dtype).sum().clamp(min=1.0)

    pair_mask = all_mask[:, 1:] & all_mask[:, :-1]
    upward = torch.relu(pred_all[:, 1:] - pred_all[:, :-1] - 0.003)
    upward_loss = (upward * pair_mask.to(upward.dtype)).sum() / pair_mask.to(upward.dtype).sum().clamp(min=1.0)
    if pred_all.size(1) > 2:
        smooth_mask = all_mask[:, 2:] & all_mask[:, 1:-1] & all_mask[:, :-2]
        second_diff = torch.abs(pred_all[:, 2:] - 2 * pred_all[:, 1:-1] + pred_all[:, :-2])
        smooth_loss = (second_diff * smooth_mask.to(second_diff.dtype)).sum() / smooth_mask.to(second_diff.dtype).sum().clamp(min=1.0)
    else:
        smooth_loss = torch.tensor(0.0, device=pred_now.device)
    phys = range_loss + 0.5 * upward_loss + 0.1 * smooth_loss

    total = (
        weights.get("now", 0.5) * loss_now
        + weights.get("future", 1.0) * loss_future
        + weights.get("align", 0.01) * align
        + weights.get("phys", 0.01) * phys
    )
    return {
        "total": total,
        "now": loss_now.detach(),
        "future": loss_future.detach(),
        "align": align.detach(),
        "phys": phys.detach(),
    }
