"""Learned IC/DV residual channels appended to deterministic battery features."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


BASE_FEATURES = 50
RESIDUAL_FEATURES_PER_CURVE = 4
ADAPTED_FEATURES = 58


def _base_variable_types() -> torch.Tensor:
    # voltage, current, temperature, charge tail, process, IC, DV
    return torch.tensor(
        [0] * 4
        + [1] * 4
        + [2] * 4
        + [3] * 16
        + [4] * 6
        + [5] * 8
        + [6] * 8,
        dtype=torch.long,
    )


@dataclass
class AdaptedBatteryFeatures:
    values: torch.Tensor
    observed_mask: torch.Tensor
    reliability: torch.Tensor
    variable_type: torch.Tensor


class CurveResidualEncoder(nn.Module):
    """Compress one masked 128-point physical derivative curve to four residuals."""

    def __init__(self, points: int = 128, output_features: int = 4) -> None:
        super().__init__()
        self.points = int(points)
        self.output_features = int(output_features)
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(8, 8, kernel_size=5, padding=2, groups=8),
            nn.Conv1d(8, 8, kernel_size=1),
            nn.GELU(),
        )
        self.proj = nn.Linear(16, self.output_features)
        self.quality_gate = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        curve: torch.Tensor,
        curve_mask: torch.Tensor,
        quality: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if curve.ndim != 3 or curve.shape[-1] != self.points:
            raise ValueError(f"curve must be [B,L,{self.points}]")
        if curve_mask.shape != curve.shape or curve_mask.dtype != torch.bool:
            raise ValueError("curve_mask must be a boolean tensor matching curve")
        if quality.shape != curve.shape[:2]:
            raise ValueError("curve quality must be [B,L]")
        if not torch.isfinite(quality).all() or not ((quality >= 0) & (quality <= 1)).all():
            raise ValueError("curve quality must be finite and within 0..1")
        if not torch.isfinite(curve.masked_select(curve_mask)).all():
            raise ValueError("observed curve values must be finite")

        mask = curve_mask.to(curve.dtype)
        curve = torch.where(curve_mask, curve, torch.zeros_like(curve))
        count = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        mean = (curve * mask).sum(dim=-1, keepdim=True) / count
        variance = ((curve - mean).square() * mask).sum(dim=-1, keepdim=True) / count
        normalized = ((curve - mean) / variance.clamp_min(1e-8).sqrt()) * mask
        batch, cycles, points = curve.shape
        encoded = self.encoder(normalized.reshape(batch * cycles, 1, points))
        encoded_mask = mask.reshape(batch * cycles, 1, points)
        mean_pool = (encoded * encoded_mask).sum(-1) / encoded_mask.sum(-1).clamp_min(1.0)
        neg_inf = torch.finfo(encoded.dtype).min
        max_pool = encoded.masked_fill(encoded_mask == 0, neg_inf).amax(-1)
        has_curve = curve_mask.sum(-1) >= 8
        max_pool = torch.where(has_curve.reshape(-1, 1), max_pool, torch.zeros_like(max_pool))
        pooled = torch.cat((mean_pool, max_pool), dim=-1)
        gate = torch.sigmoid(self.quality_gate)
        residual = self.proj(pooled).reshape(batch, cycles, self.output_features)
        residual = residual * quality.unsqueeze(-1) * gate
        observed = has_curve & (quality > 0)
        residual = residual * observed.unsqueeze(-1).to(residual.dtype)
        residual_mask = observed.unsqueeze(-1).expand_as(residual)
        reliability = quality.unsqueeze(-1).expand_as(residual) * gate
        reliability = reliability * residual_mask.to(reliability.dtype)
        return residual, residual_mask, reliability


class BatteryFeatureAdapter(nn.Module):
    """Convert 50 fixed sensor features plus IC/DV curves to 58 variables."""

    def __init__(self, curve_points: int = 128) -> None:
        super().__init__()
        self.ic_encoder = CurveResidualEncoder(curve_points, RESIDUAL_FEATURES_PER_CURVE)
        self.dv_encoder = CurveResidualEncoder(curve_points, RESIDUAL_FEATURES_PER_CURVE)
        variable_type = torch.cat(
            (
                _base_variable_types(),
                torch.full((4,), 7, dtype=torch.long),
                torch.full((4,), 8, dtype=torch.long),
            )
        )
        self.register_buffer("variable_type", variable_type, persistent=True)

    def forward(
        self,
        base_values: torch.Tensor,
        base_observed_mask: torch.Tensor,
        base_reliability: torch.Tensor,
        ic_curve: torch.Tensor,
        ic_curve_mask: torch.Tensor,
        ic_quality: torch.Tensor,
        dv_curve: torch.Tensor,
        dv_curve_mask: torch.Tensor,
        dv_quality: torch.Tensor,
    ) -> AdaptedBatteryFeatures:
        if base_values.ndim != 3 or base_values.shape[-1] != BASE_FEATURES:
            raise ValueError("base_values must be [B,L,50]")
        if base_observed_mask.shape != base_values.shape or base_observed_mask.dtype != torch.bool:
            raise ValueError("base_observed_mask must be boolean and match base_values")
        if base_reliability.shape != base_values.shape:
            raise ValueError("base_reliability must match base_values")
        if not torch.isfinite(base_reliability).all() or not (
            (base_reliability >= 0) & (base_reliability <= 1)
        ).all():
            raise ValueError("base_reliability must be finite and within 0..1")
        observed_values = base_values.masked_select(base_observed_mask)
        if not torch.isfinite(observed_values).all():
            raise ValueError("observed base features must be finite")

        ic_values, ic_mask, ic_reliability = self.ic_encoder(ic_curve, ic_curve_mask, ic_quality)
        dv_values, dv_mask, dv_reliability = self.dv_encoder(dv_curve, dv_curve_mask, dv_quality)
        safe_base = torch.where(base_observed_mask, base_values, torch.zeros_like(base_values))
        values = torch.cat((safe_base, ic_values, dv_values), dim=-1)
        observed_mask = torch.cat((base_observed_mask, ic_mask, dv_mask), dim=-1)
        reliability = torch.cat(
            (
                base_reliability * base_observed_mask.to(base_reliability.dtype),
                ic_reliability,
                dv_reliability,
            ),
            dim=-1,
        ).clamp_(0.0, 1.0)
        if values.shape[-1] != ADAPTED_FEATURES:
            raise RuntimeError("battery feature adapter did not produce 58 channels")
        return AdaptedBatteryFeatures(
            values=values,
            observed_mask=observed_mask,
            reliability=reliability,
            variable_type=self.variable_type,
        )
