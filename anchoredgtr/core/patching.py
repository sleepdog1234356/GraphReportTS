from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PatchBatch:
    nodes: torch.Tensor
    node_mask: torch.Tensor
    variable_index: torch.Tensor
    scale_index: torch.Tensor
    patch_index: torch.Tensor
    start: torch.Tensor
    end: torch.Tensor
    center: torch.Tensor
    widths: tuple[int, ...]
    patches_per_scale: tuple[int, ...]
    variable_mask: torch.Tensor
    real_node_count: int


def patch_schedule(variable_count: int) -> tuple[int, ...]:
    return (2, 4, 8) if variable_count <= 64 else (4, 8, 16)


class AdaptivePatchifier(nn.Module):
    """Turn each variable's history into non-overlapping multi-scale patch nodes."""

    def __init__(
        self,
        d_model: int,
        max_variables: int = 512,
        max_types: int = 16,
        dropout: float = 0.1,
        history_len: int | None = None,
        embedding_variant: str = "patch",
    ) -> None:
        super().__init__()
        # Every raw point, its observation bit, and its reliability are retained.
        # Width-specific projections avoid reducing a patch to summary statistics.
        self.projectors = nn.ModuleDict({str(width): nn.Linear(width * 3, d_model) for width in (2, 4, 8, 16)})
        self.history_len = int(history_len) if history_len is not None else None
        self.embedding_variant = str(embedding_variant)
        allowed_variants = (
            "patch",
            "series_context",
            "series_context_diff",
            "series_context_decomp",
            "global_node",
            "global_node_diff",
            "global_node_raw",
        )
        if self.embedding_variant not in allowed_variants:
            raise ValueError(f"unsupported embedding_variant: {self.embedding_variant}")
        if self.embedding_variant != "patch" and self.history_len is None:
            raise ValueError("history_len is required for a history-conditioned graph embedding")
        self.variable_embedding = nn.Embedding(max_variables, d_model)
        self.scale_embedding = nn.Embedding(4 if self.embedding_variant.startswith("global_node") else 3, d_model)
        self.type_embedding = nn.Embedding(max_types, d_model)
        self.position_projection = nn.Linear(1, d_model)
        self.series_context = self._history_encoder(
            self.history_len
            if self.embedding_variant != "patch" and self.embedding_variant != "series_context_decomp"
            else None,
            d_model,
            dropout,
        )
        self.difference_context = (
            self._history_encoder(self.history_len, d_model, dropout)
            if self.embedding_variant.endswith("_diff")
            else None
        )
        self.global_raw_projection = (
            nn.Linear(self.history_len, d_model, bias=False)
            if self.embedding_variant == "global_node_raw" and self.history_len is not None
            else None
        )
        self.raw_gate_logit = (
            nn.Parameter(torch.tensor(math.log(0.1 / 0.9)))
            if self.global_raw_projection is not None
            else None
        )
        self.trend_context = (
            self._history_encoder(self.history_len, d_model, dropout)
            if self.embedding_variant == "series_context_decomp"
            else None
        )
        self.seasonal_context = (
            self._history_encoder(self.history_len, d_model, dropout)
            if self.embedding_variant == "series_context_decomp"
            else None
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _history_encoder(history_len: int | None, d_model: int, dropout: float) -> nn.Module | None:
        if history_len is None:
            return None
        return nn.Sequential(
            nn.LayerNorm(history_len),
            nn.Linear(history_len, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    @staticmethod
    def _patch_inputs(
        values: torch.Tensor,
        observed: torch.Tensor,
        reliability: torch.Tensor,
        width: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, length, variables = values.shape
        patches = math.ceil(length / width)
        padded = patches * width - length
        valid_time = torch.ones(length, dtype=values.dtype, device=values.device)
        if padded:
            values = torch.nn.functional.pad(values, (0, 0, 0, padded))
            observed = torch.nn.functional.pad(observed, (0, 0, 0, padded))
            reliability = torch.nn.functional.pad(reliability, (0, 0, 0, padded))
            valid_time = torch.nn.functional.pad(valid_time, (0, padded))
        x = values.reshape(batch, patches, width, variables).permute(0, 1, 3, 2)
        mask = observed.reshape(batch, patches, width, variables).permute(0, 1, 3, 2)
        rel = reliability.reshape(batch, patches, width, variables).permute(0, 1, 3, 2)
        possible = valid_time.reshape(patches, width).sum(-1).clamp_min(1.0)
        coverage = mask.float().sum(-1) / possible.view(1, patches, 1)
        encoded_input = torch.cat((x * mask, mask.to(x.dtype), rel * mask), dim=-1)
        return encoded_input, coverage >= 0.5

    def forward(
        self,
        values: torch.Tensor,
        observed_mask: torch.Tensor,
        reliability: torch.Tensor,
        variable_type: torch.Tensor,
        variable_mask: torch.Tensor,
        max_nodes: int = 6000,
    ) -> PatchBatch:
        if values.ndim != 3:
            raise ValueError("values must be [B,L,F]")
        batch, length, variables = values.shape
        if self.history_len is not None and length != self.history_len:
            raise ValueError(f"expected history length {self.history_len}, received {length}")
        if variables > self.variable_embedding.num_embeddings:
            raise ValueError(f"received {variables} variables, maximum is {self.variable_embedding.num_embeddings}")
        local_widths = patch_schedule(variables)
        local_counts = tuple(math.ceil(length / width) for width in local_widths)
        has_global_node = self.embedding_variant.startswith("global_node")
        widths = (*local_widths, length) if has_global_node else local_widths
        counts = (*local_counts, 1) if has_global_node else local_counts
        node_count = variables * sum(counts)
        if node_count > max_nodes:
            raise ValueError(f"adaptive graph requires {node_count} nodes, exceeding max_nodes={max_nodes}")
        if variable_type.ndim == 1:
            variable_type = variable_type.unsqueeze(0).expand(batch, -1)
        node_parts: list[torch.Tensor] = []
        mask_parts: list[torch.Tensor] = []
        variable_parts: list[torch.Tensor] = []
        scale_parts: list[torch.Tensor] = []
        patch_parts: list[torch.Tensor] = []
        start_parts: list[torch.Tensor] = []
        end_parts: list[torch.Tensor] = []
        center_parts: list[torch.Tensor] = []
        raw_residual_parts: list[torch.Tensor] = []
        variable_ids = torch.arange(variables, device=values.device)
        history_context = None
        patch_values = values
        if self.embedding_variant != "patch":
            masked_values = values * observed_mask.to(values.dtype)
            if self.embedding_variant == "series_context_decomp":
                if self.trend_context is None or self.seasonal_context is None:
                    raise RuntimeError("decomposition context encoders are unavailable")
                series = masked_values.transpose(1, 2).float()
                trend = F.avg_pool1d(F.pad(series, (12, 12), mode="replicate"), kernel_size=25, stride=1)
                seasonal = series - trend
                history_context = (
                    self.trend_context(trend) + self.seasonal_context(seasonal)
                ).to(values.dtype) / math.sqrt(2.0)
                count = observed_mask.sum(dim=1, keepdim=True).clamp_min(1).to(values.dtype)
                mean = masked_values.sum(dim=1, keepdim=True) / count
                centered = (values - mean) * observed_mask.to(values.dtype)
                variance = centered.square().sum(dim=1, keepdim=True) / count
                patch_values = centered / (variance + 1e-5).sqrt()
            else:
                if self.series_context is None:
                    raise RuntimeError("series context encoder is unavailable")
                history_context = self.series_context(masked_values.transpose(1, 2).float()).to(values.dtype)
            if self.difference_context is not None:
                differences = torch.diff(masked_values, dim=1, prepend=masked_values[:, :1])
                difference_context = self.difference_context(differences.transpose(1, 2).float()).to(values.dtype)
                history_context = (history_context + difference_context) / math.sqrt(2.0)
            history_context = history_context * variable_mask.unsqueeze(-1)
        for scale_id, width in enumerate(local_widths):
            patch_inputs, patch_observed = self._patch_inputs(patch_values, observed_mask, reliability, width)
            patches = patch_inputs.size(1)
            starts = torch.arange(patches, device=values.device) * width
            ends = (starts + width).clamp_max(length)
            centers = (starts.to(values.dtype) + ends.to(values.dtype) - 1.0) / (2.0 * max(length - 1, 1))
            position = centers.view(1, patches, 1, 1).expand(batch, patches, variables, 1)
            projected = self.projectors[str(width)](patch_inputs)
            projected = projected + self.variable_embedding(variable_ids).to(projected.dtype).view(1, 1, variables, -1)
            projected = projected + self.scale_embedding.weight[scale_id].to(projected.dtype).view(1, 1, 1, -1)
            projected = projected + self.position_projection(position)
            projected = projected + self.type_embedding(
                variable_type.clamp(0, self.type_embedding.num_embeddings - 1)
            ).to(projected.dtype).unsqueeze(1)
            if history_context is not None and self.embedding_variant.startswith("series_context"):
                projected = projected + history_context.to(projected.dtype).unsqueeze(1)
            valid = patch_observed & variable_mask.unsqueeze(1)
            node_parts.append(projected.reshape(batch, patches * variables, -1))
            if self.global_raw_projection is not None:
                raw_residual_parts.append(torch.zeros_like(node_parts[-1]))
            mask_parts.append(valid.reshape(batch, patches * variables))
            variable_parts.append(variable_ids.repeat(patches))
            scale_parts.append(torch.full((patches * variables,), scale_id, device=values.device, dtype=torch.long))
            patch_parts.append(torch.arange(patches, device=values.device).repeat_interleave(variables))
            start_parts.append(starts.repeat_interleave(variables))
            end_parts.append(ends.repeat_interleave(variables))
            center_parts.append(centers.repeat_interleave(variables))
        if has_global_node:
            if history_context is None:
                raise RuntimeError("global history node encoder is unavailable")
            global_scale = len(local_widths)
            position = values.new_full((batch, variables, 1), 0.5)
            projected = history_context.to(values.dtype)
            projected = projected + self.variable_embedding(variable_ids).to(projected.dtype).unsqueeze(0)
            projected = projected + self.scale_embedding.weight[global_scale].to(projected.dtype).view(1, 1, -1)
            projected = projected + self.position_projection(position)
            projected = projected + self.type_embedding(
                variable_type.clamp(0, self.type_embedding.num_embeddings - 1)
            ).to(projected.dtype)
            global_valid = variable_mask & observed_mask.float().mean(dim=1).ge(0.5)
            node_parts.append(projected)
            if self.global_raw_projection is not None:
                raw_residual_parts.append(
                    self.global_raw_projection(masked_values.transpose(1, 2).float()).to(projected.dtype)
                    * variable_mask.unsqueeze(-1)
                )
            mask_parts.append(global_valid)
            variable_parts.append(variable_ids)
            scale_parts.append(torch.full((variables,), global_scale, device=values.device, dtype=torch.long))
            patch_parts.append(torch.zeros(variables, device=values.device, dtype=torch.long))
            start_parts.append(torch.zeros(variables, device=values.device, dtype=torch.long))
            end_parts.append(torch.full((variables,), length, device=values.device, dtype=torch.long))
            center_parts.append(values.new_full((variables,), 0.5))
        nodes = self.norm(torch.cat(node_parts, dim=1))
        if self.raw_gate_logit is not None:
            nodes = nodes + torch.sigmoid(self.raw_gate_logit) * torch.cat(raw_residual_parts, dim=1)
        node_mask = torch.cat(mask_parts, dim=1)
        if not node_mask.any(dim=1).all():
            raise ValueError("each sample requires at least one patch with at least 50% observed coverage")
        nodes = self.dropout(nodes) * node_mask.unsqueeze(-1)
        return PatchBatch(
            nodes=nodes,
            node_mask=node_mask,
            variable_index=torch.cat(variable_parts),
            scale_index=torch.cat(scale_parts),
            patch_index=torch.cat(patch_parts),
            start=torch.cat(start_parts),
            end=torch.cat(end_parts),
            center=torch.cat(center_parts).to(values.dtype),
            widths=widths,
            patches_per_scale=counts,
            variable_mask=variable_mask,
            real_node_count=int(node_mask.sum(dim=1).max().item()),
        )
