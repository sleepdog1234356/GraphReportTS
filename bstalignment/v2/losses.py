from __future__ import annotations

import torch
import torch.nn.functional as F

from ..losses import nt_xent_loss


def masked_smooth_l1(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    loss = F.smooth_l1_loss(prediction, target, reduction="none", beta=beta)
    weight = mask.to(loss.dtype)
    return (loss * weight).sum() / weight.sum().clamp_min(1.0)


def alignment_loss(output: dict[str, object], weight: float, temperature: float = 0.2) -> torch.Tensor:
    prediction = output["pred"]
    if not torch.is_tensor(prediction):
        raise TypeError("output['pred'] must be a tensor")
    if weight <= 0 or output.get("text_enabled") is False:
        return prediction.new_tensor(0.0)
    align_graph = output["align_graph"]
    align_text = output["align_text"]
    if not torch.is_tensor(align_graph) or not torch.is_tensor(align_text):
        raise TypeError("alignment representations must be tensors")
    if align_graph.size(0) <= 1:
        return output["pred"].new_tensor(0.0)
    return nt_xent_loss(align_graph, align_text, temperature=temperature)


def general_v2_loss(
    output: dict[str, object], target: torch.Tensor, mask: torch.Tensor, align_weight: float
) -> dict[str, torch.Tensor]:
    prediction = output["pred"]
    if not torch.is_tensor(prediction):
        raise TypeError("output['pred'] must be a tensor")
    value = masked_smooth_l1(prediction, target, mask, beta=1.0)
    align = alignment_loss(output, align_weight)
    return {"total": value + align_weight * align, "value": value, "align": align}


def battery_v2_loss(
    output: dict[str, object], target: torch.Tensor, mask: torch.Tensor, align_weight: float
) -> dict[str, torch.Tensor]:
    prediction = output["pred"]
    if not torch.is_tensor(prediction):
        raise TypeError("output['pred'] must be a tensor")
    value = masked_smooth_l1(prediction, target, mask, beta=1.0)
    slope_mask = mask[:, 1:] & mask[:, :-1]
    slope = masked_smooth_l1(prediction[:, 1:] - prediction[:, :-1], target[:, 1:] - target[:, :-1], slope_mask)
    bound_error = torch.relu(-prediction).square() + torch.relu(prediction - 1.2).square()
    bound_weight = mask.to(bound_error.dtype)
    bound = (bound_error * bound_weight).sum() / bound_weight.sum().clamp_min(1.0)
    align = alignment_loss(output, align_weight)
    total = value + 0.2 * slope + 0.01 * bound + align_weight * align
    return {"total": total, "value": value, "slope": slope, "bound": bound, "align": align}


@torch.no_grad()
def regression_metrics(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    error = (prediction - target)[mask]
    if error.numel() == 0:
        return {"mse": 0.0, "mae": 0.0, "rmse": 0.0}
    mse = float(error.square().mean().cpu())
    return {"mse": mse, "mae": float(error.abs().mean().cpu()), "rmse": mse**0.5}
