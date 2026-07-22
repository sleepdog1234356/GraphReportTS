from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .patching import PatchBatch


RELATIONS = ("temporal", "variable", "cross_scale")
VARIABLE_NEIGHBOR_CAP = 16


@dataclass
class RelationEdges:
    edge_index: torch.Tensor
    prior: torch.Tensor


@dataclass
class SparseEdgeBatch:
    relations: dict[str, RelationEdges]
    total_nodes: int
    variable_score_mix: torch.Tensor


def _difference_correlation(values: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
    variables = values.size(-1)
    if values.size(-2) < 2:
        return values.new_zeros(*values.shape[:-2], variables, variables)
    differences = values[..., 1:, :] - values[..., :-1, :]
    diff_observed = observed[..., 1:, :] & observed[..., :-1, :]
    mask = diff_observed.to(values.dtype)
    count = torch.einsum("...lf,...lg->...fg", mask, mask)
    mean = (differences * mask).sum(-2) / mask.sum(-2).clamp_min(1.0)
    centered = (differences - mean.unsqueeze(-2)) * mask
    covariance = torch.einsum("...lf,...lg->...fg", centered, centered) / count.clamp_min(1.0)
    scale = torch.diagonal(covariance, dim1=-2, dim2=-1).clamp_min(1e-8).sqrt()
    return (covariance / (scale.unsqueeze(-1) * scale.unsqueeze(-2)).clamp_min(1e-8)).nan_to_num(0.0)


class SparseEdgeBuilder(nn.Module):
    """Construct relation edges while preserving every real variable node."""

    def __init__(
        self,
        dense_threshold: int = 64,
        max_neighbors: int = 16,
        max_variables: int = 512,
        relation_dim: int = 16,
        d_model: int = 128,
    ) -> None:
        super().__init__()
        # Retained only so existing serialized/configured constructors remain
        # loadable.  Variable topology no longer branches at a feature-count
        # threshold: every sample follows the same adaptive rule below.
        self.dense_threshold = int(dense_threshold)
        if not 8 <= int(max_neighbors) <= VARIABLE_NEIGHBOR_CAP:
            raise ValueError("max_neighbors must be within 8..16")
        self.max_neighbors = int(max_neighbors)
        self.pool_score = nn.Linear(d_model, 1, bias=False)
        self.dynamic_q = nn.Linear(d_model, relation_dim, bias=False)
        self.dynamic_k = nn.Linear(d_model, relation_dim, bias=False)
        self.static_embedding = nn.Embedding(max_variables, relation_dim)
        self.relation_mix_logits = nn.Parameter(torch.zeros(3))

    def _pool_patch_tokens(self, patches: PatchBatch) -> torch.Tensor:
        """Attention-pool valid multi-scale patch tokens for each variable."""

        nodes = patches.nodes
        batch, node_count, d_model = nodes.shape
        variables = patches.variable_mask.size(1)
        variable_index = patches.variable_index.view(1, node_count).expand(batch, -1)

        logits = self.pool_score(nodes).squeeze(-1)
        # Under BF16 autocast the linear score is BF16 while patch tokens can
        # remain FP32 after adding embeddings.  Scatter reductions require the
        # accumulator and source to share a dtype, so score-space reductions
        # must be allocated from ``logits`` rather than ``nodes``.
        maxima = logits.new_full((batch, variables), -torch.inf)
        maxima.scatter_reduce_(
            1,
            variable_index,
            logits.masked_fill(~patches.node_mask, -torch.inf),
            reduce="amax",
            include_self=True,
        )
        selected_maxima = maxima.gather(1, variable_index)
        shifted = torch.where(
            patches.node_mask,
            logits - selected_maxima,
            torch.zeros_like(logits),
        )
        unnormalized = torch.exp(shifted) * patches.node_mask.to(logits.dtype)
        # ``exp`` is an FP32 autocast op on CUDA, so its output can be wider
        # than the BF16 logits.  Allocate the sum accumulator from the actual
        # scatter source as well.
        denominator = unnormalized.new_zeros(batch, variables)
        denominator.scatter_add_(1, variable_index, unnormalized)
        weights = unnormalized / denominator.gather(1, variable_index).clamp_min(1e-8)

        summary = nodes.new_zeros(batch, variables, d_model)
        summary.scatter_add_(
            1,
            variable_index.unsqueeze(-1).expand(-1, -1, d_model),
            weights.unsqueeze(-1) * nodes,
        )
        return summary * patches.variable_mask.unsqueeze(-1)

    def _dynamic_scores(self, summary: torch.Tensor) -> torch.Tensor:
        """Return directed cosine scores from independent query/key maps."""

        dynamic_query = F.normalize(self.dynamic_q(summary), dim=-1)
        dynamic_key = F.normalize(self.dynamic_k(summary), dim=-1)
        return dynamic_query @ dynamic_key.transpose(-2, -1)

    def _variable_scores(
        self,
        patches: PatchBatch,
        history: torch.Tensor,
        observed: torch.Tensor,
    ) -> torch.Tensor:
        """Return score[batch, destination variable, source variable]."""

        summary = self._pool_patch_tokens(patches)
        dynamic = self._dynamic_scores(summary)
        difference = _difference_correlation(history, observed)
        variable_ids = torch.arange(history.size(-1), device=history.device)
        static_tokens = F.normalize(self.static_embedding(variable_ids), dim=-1)
        static = static_tokens @ static_tokens.transpose(0, 1)
        mix = torch.softmax(self.relation_mix_logits, dim=0)
        return mix[0] * dynamic + mix[1] * difference + mix[2] * static

    @staticmethod
    def _patch_groups(patches: PatchBatch, variables: int) -> tuple[torch.Tensor, ...]:
        """Map each (scale, patch, variable) tuple to its local node index."""

        nodes = patches.variable_index.numel()
        patch_base = max(patches.patches_per_scale)
        node_group_key = patches.scale_index * patch_base + patches.patch_index
        group_keys, group_index = torch.unique(node_group_key, sorted=True, return_inverse=True)
        groups = group_keys.numel()
        group_nodes = torch.full(
            (groups, variables), -1, dtype=torch.long, device=patches.variable_index.device
        )
        group_nodes[group_index, patches.variable_index] = torch.arange(
            nodes, device=patches.variable_index.device
        )

        representatives = torch.full(
            (groups,), nodes, dtype=torch.long, device=patches.variable_index.device
        )
        representatives.scatter_reduce_(
            0,
            group_index,
            torch.arange(nodes, device=patches.variable_index.device),
            reduce="amin",
            include_self=True,
        )
        return (
            group_nodes,
            patches.scale_index.index_select(0, representatives),
            patches.patch_index.index_select(0, representatives),
            patches.start.index_select(0, representatives),
            patches.end.index_select(0, representatives),
        )

    @staticmethod
    def _expand_group_pairs(
        group_nodes: torch.Tensor,
        destination_group: torch.Tensor,
        source_group: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Expand patch-group pairs over all variables and discard absent nodes."""

        variables = group_nodes.size(1)
        variable = torch.arange(variables, device=group_nodes.device).repeat(destination_group.numel())
        destination = group_nodes[
            destination_group.repeat_interleave(variables), variable
        ]
        source = group_nodes[source_group.repeat_interleave(variables), variable]
        valid = (source >= 0) & (destination >= 0)
        return source[valid], destination[valid], valid

    @staticmethod
    def _batch_template(
        source: torch.Tensor,
        destination: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply per-sample node validity and add flattened batch offsets."""

        batch, nodes = node_mask.shape
        valid = node_mask.index_select(1, source) & node_mask.index_select(1, destination)
        batch_index, template_index = torch.nonzero(valid, as_tuple=True)
        offsets = batch_index * nodes
        edge_index = torch.stack(
            (source.index_select(0, template_index) + offsets, destination.index_select(0, template_index) + offsets)
        )
        return edge_index, template_index

    def _static_edges(
        self,
        patches: PatchBatch,
        group_nodes: torch.Tensor,
        group_scale: torch.Tensor,
        group_patch: torch.Tensor,
        group_start: torch.Tensor,
        group_end: torch.Tensor,
        history: torch.Tensor,
    ) -> dict[str, RelationEdges]:
        """Build temporal and cross-scale templates once, then batch-filter them."""

        patch_delta = (group_patch[:, None] - group_patch[None, :]).abs()
        temporal_pair = (group_scale[:, None] == group_scale[None, :]) & (
            (patch_delta == 1) | (patch_delta == 2)
        )
        temporal_destination_group, temporal_source_group = torch.nonzero(temporal_pair, as_tuple=True)
        temporal_source, temporal_destination, temporal_node_valid = self._expand_group_pairs(
            group_nodes, temporal_destination_group, temporal_source_group
        )
        temporal_group_prior = torch.where(
            patch_delta[temporal_destination_group, temporal_source_group] == 1,
            history.new_tensor(0.35),
            history.new_tensor(0.15),
        )
        temporal_template_prior = temporal_group_prior.repeat_interleave(group_nodes.size(1))[temporal_node_valid]
        temporal_index, temporal_selection = self._batch_template(
            temporal_source, temporal_destination, patches.node_mask
        )

        cross_pair = (group_scale[:, None] != group_scale[None, :]) & (
            group_start[None, :] < group_end[:, None]
        ) & (group_end[None, :] > group_start[:, None])
        cross_destination_group, cross_source_group = torch.nonzero(cross_pair, as_tuple=True)
        cross_source, cross_destination, _ = self._expand_group_pairs(
            group_nodes, cross_destination_group, cross_source_group
        )
        cross_index, _ = self._batch_template(cross_source, cross_destination, patches.node_mask)

        return {
            "temporal": RelationEdges(
                edge_index=temporal_index,
                prior=temporal_template_prior.index_select(0, temporal_selection),
            ),
            "cross_scale": RelationEdges(
                edge_index=cross_index,
                prior=history.new_full((cross_index.size(1),), 0.25),
            ),
        }

    def _variable_edges(
        self,
        patches: PatchBatch,
        group_nodes: torch.Tensor,
        scores: torch.Tensor,
        variable_mask: torch.Tensor,
        history: torch.Tensor,
    ) -> RelationEdges:
        """Select batched variable neighbors and expand them over patch groups."""

        batch, variables = variable_mask.shape
        valid_count = variable_mask.sum(-1)
        diagonal = torch.eye(variables, dtype=torch.bool, device=history.device).unsqueeze(0)
        valid_pair = (
            variable_mask.unsqueeze(-1) & variable_mask.unsqueeze(-2) & ~diagonal
        )

        requested = torch.ceil(2.0 * torch.log2(valid_count.clamp_min(1).to(torch.float32))).to(torch.long)
        neighbor_count = torch.minimum(
            torch.minimum(
                torch.full_like(requested, self.max_neighbors),
                requested.clamp_min(8),
            ),
            (valid_count - 1).clamp_min(0),
        )
        # F_valid<=9 is naturally complete because the floor of eight is then
        # capped by F_valid-1.  Larger samples retain only their highest-scored
        # directed sources, capped at sixteen for bounded graph cost.
        max_count = min(self.max_neighbors, variables - 1)
        if max_count:
            selection_scores = scores.detach().masked_fill(~valid_pair, -torch.inf)
            top_source = selection_scores.topk(max_count, dim=-1).indices
            ranks = torch.arange(max_count, device=history.device).view(1, 1, -1)
            selected = (
                (neighbor_count > 0)[:, None, None]
                & variable_mask.unsqueeze(-1)
                & (ranks < neighbor_count[:, None, None])
            )
            sample, destination_variable, rank = torch.nonzero(selected, as_tuple=True)
            source_variable = top_source[sample, destination_variable, rank]
        else:
            sample = torch.empty(0, dtype=torch.long, device=history.device)
            destination_variable = sample
            source_variable = sample
        if sample.numel() == 0:
            return RelationEdges(
                edge_index=torch.empty((2, 0), dtype=torch.long, device=history.device),
                prior=history.new_empty(0),
            )

        groups = group_nodes.size(0)
        group = torch.arange(groups, device=history.device).repeat(sample.numel())
        expanded_sample = sample.repeat_interleave(groups)
        expanded_destination_variable = destination_variable.repeat_interleave(groups)
        expanded_source_variable = source_variable.repeat_interleave(groups)
        destination = group_nodes[group, expanded_destination_variable]
        source = group_nodes[group, expanded_source_variable]
        node_exists = (source >= 0) & (destination >= 0)
        node_valid = node_exists & patches.node_mask[
            expanded_sample,
            destination.clamp_min(0),
        ] & patches.node_mask[
            expanded_sample,
            source.clamp_min(0),
        ]
        expanded_sample = expanded_sample[node_valid]
        source = source[node_valid]
        destination = destination[node_valid]
        offsets = expanded_sample * patches.node_mask.size(1)
        edge_index = torch.stack((source + offsets, destination + offsets))

        pair_prior = scores[sample, destination_variable, source_variable]
        prior = pair_prior.repeat_interleave(groups)[node_valid]
        return RelationEdges(edge_index=edge_index, prior=prior)

    def forward(
        self,
        patches: PatchBatch,
        history: torch.Tensor,
        observed: torch.Tensor,
    ) -> SparseEdgeBatch:
        batch, nodes = patches.node_mask.shape
        variables = history.size(-1)
        if variables > self.static_embedding.num_embeddings:
            raise ValueError("variable count exceeds SparseEdgeBuilder.max_variables")
        if patches.node_mask.device != history.device:
            raise ValueError("patch metadata and history must be on the same device")

        group_nodes, group_scale, group_patch, group_start, group_end = self._patch_groups(
            patches, variables
        )
        static_relations = self._static_edges(
            patches,
            group_nodes,
            group_scale,
            group_patch,
            group_start,
            group_end,
            history,
        )
        scores = self._variable_scores(patches, history, observed)
        variable_edges = self._variable_edges(
            patches, group_nodes, scores, patches.variable_mask, history
        )
        relations = {
            "temporal": static_relations["temporal"],
            "variable": variable_edges,
            "cross_scale": static_relations["cross_scale"],
        }
        return SparseEdgeBatch(
            relations=relations,
            total_nodes=batch * nodes,
            variable_score_mix=torch.softmax(self.relation_mix_logits, dim=0),
        )
