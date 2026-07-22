from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from .edges import RELATIONS, RelationEdges, SparseEdgeBatch
from .patching import PatchBatch


MIXER_RELATIONS = (*RELATIONS, "router")


def segment_softmax(score: torch.Tensor, destination: torch.Tensor, segments: int) -> torch.Tensor:
    if score.numel() == 0:
        return score
    maximum = torch.full((segments, score.size(-1)), -torch.inf, dtype=score.dtype, device=score.device)
    maximum.scatter_reduce_(0, destination[:, None].expand_as(score), score, reduce="amax", include_self=True)
    exponent = torch.exp(score - maximum.index_select(0, destination))
    denominator = torch.zeros_like(maximum)
    denominator.index_add_(0, destination, exponent)
    return exponent / denominator.index_select(0, destination).clamp_min(1e-8)


class RelationAttention(nn.Module):
    def __init__(self, d_model: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = d_model // heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        # Bias-free so edgewise projection commutes with destination summation;
        # otherwise high-degree variables receive the output bias repeatedly.
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, nodes: torch.Tensor, edges: RelationEdges) -> torch.Tensor:
        if edges.edge_index.numel() == 0:
            # Under CUDA autocast the linear output is BF16 even though the
            # residual node tensor remains FP32.  Use a one-row projection to
            # create a correctly typed zero contribution for an empty relation.
            return self.out(nodes[:1]).new_zeros(nodes.shape)
        source, destination = edges.edge_index
        # Project every node once, then gather edge endpoints.  Projecting the
        # duplicated edge rows is mathematically identical but makes wide
        # graphs such as ECL repeat the same GEMM thousands of times.
        query_nodes = self.q(nodes).view(-1, self.heads, self.head_dim)
        key_nodes = self.k(nodes).view(-1, self.heads, self.head_dim)
        value_nodes = self.v(nodes).view(-1, self.heads, self.head_dim)
        query = query_nodes.index_select(0, destination)
        key = key_nodes.index_select(0, source)
        value = value_nodes.index_select(0, source)
        score = (query * key).sum(-1) / math.sqrt(self.head_dim)
        score = score + edges.prior.unsqueeze(-1)
        weight = segment_softmax(score, destination, nodes.size(0))
        message = (weight.unsqueeze(-1) * value).reshape(-1, self.heads * self.head_dim)
        projected = self.out(message)
        result = projected.new_zeros(nodes.shape)
        result.index_add_(0, destination, projected)
        return result


class ParallelGraphMixerBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float, ffn_expansion: int) -> None:
        super().__init__()
        self.relations = nn.ModuleDict({name: RelationAttention(d_model, heads) for name in RELATIONS})
        self.relation_gate = nn.Linear(d_model, len(MIXER_RELATIONS))
        self.nodes_from_routers = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.routers_from_nodes = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.node_norm1 = nn.LayerNorm(d_model)
        self.node_norm2 = nn.LayerNorm(d_model)
        self.router_norm1 = nn.LayerNorm(d_model)
        self.router_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.node_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_expansion, d_model),
        )
        self.router_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_expansion, d_model),
        )

    def forward(
        self,
        nodes: torch.Tensor,
        routers: torch.Tensor,
        edges: SparseEdgeBatch,
        node_mask: torch.Tensor,
        router_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, node_count, dim = nodes.shape
        normalized_nodes = self.node_norm1(nodes)
        normalized_routers = self.router_norm1(routers)
        flat_nodes = normalized_nodes.reshape(batch * node_count, dim)
        structural = [
            self.relations[name](flat_nodes, edges.relations[name]).reshape(batch, node_count, dim)
            for name in RELATIONS
        ]
        router_delta, _ = self.nodes_from_routers(
            normalized_nodes,
            normalized_routers,
            normalized_routers,
            key_padding_mask=~router_mask,
            need_weights=False,
        )
        deltas = torch.stack((*structural, router_delta), dim=-2)
        relation_gates = torch.softmax(self.relation_gate(normalized_nodes), dim=-1)
        node_delta = (relation_gates.unsqueeze(-1) * deltas).sum(dim=-2)
        mixed_nodes = nodes + self.dropout(node_delta)
        mixed_nodes = mixed_nodes + self.dropout(self.node_ffn(self.node_norm2(mixed_nodes)))
        mixed_nodes = mixed_nodes * node_mask.unsqueeze(-1)

        global_delta, _ = self.routers_from_nodes(
            normalized_routers,
            normalized_nodes,
            normalized_nodes,
            key_padding_mask=~node_mask,
            need_weights=False,
        )
        mixed_routers = routers + self.dropout(global_delta)
        mixed_routers = mixed_routers + self.dropout(self.router_ffn(self.router_norm2(mixed_routers)))
        mixed_routers = mixed_routers * router_mask.unsqueeze(-1)
        return mixed_nodes, mixed_routers, relation_gates * node_mask.unsqueeze(-1)


@dataclass
class GraphMixerOutput:
    patch_tokens: torch.Tensor
    variable_tokens: torch.Tensor
    router_tokens: torch.Tensor
    variable_mask: torch.Tensor
    router_mask: torch.Tensor
    relation_gates: torch.Tensor


class GraphMixer(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
        ffn_expansion: int = 2,
        max_routers: int = 16,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [ParallelGraphMixerBlock(d_model, heads, dropout, ffn_expansion) for _ in range(layers)]
        )
        self.pool_score = nn.Linear(d_model, 1)
        self.router_queries = nn.Parameter(torch.randn(max_routers, d_model) * 0.02)
        self.router_norm = nn.LayerNorm(d_model)
        self.max_routers = max_routers

    def _pool_variables(self, nodes: torch.Tensor, patches: PatchBatch) -> torch.Tensor:
        batch, _, dim = nodes.shape
        variables = patches.variable_mask.size(1)
        score = self.pool_score(nodes).squeeze(-1).masked_fill(~patches.node_mask, -1e4)
        output = torch.zeros(batch, variables, dim, device=nodes.device, dtype=nodes.dtype)
        for variable in range(variables):
            selected = patches.variable_index == variable
            valid = patches.node_mask[:, selected]
            weights = torch.softmax(score[:, selected], dim=-1) * valid
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
            output[:, variable] = (weights.unsqueeze(-1) * nodes[:, selected]).sum(1)
        return output * patches.variable_mask.unsqueeze(-1)

    def forward(self, patches: PatchBatch, edges: SparseEdgeBatch) -> GraphMixerOutput:
        batch = patches.nodes.size(0)
        variable_count = patches.variable_mask.sum(-1)
        per_sample_routers = torch.ceil(torch.sqrt(variable_count.float())).to(torch.long).clamp(min=4, max=self.max_routers)
        router_count = int(per_sample_routers.max().item())
        router_mask = torch.arange(router_count, device=patches.nodes.device).unsqueeze(0) < per_sample_routers.unsqueeze(1)
        router_seed = self.router_queries[:router_count].to(patches.nodes.dtype)
        routers = self.router_norm(router_seed).unsqueeze(0).expand(batch, -1, -1)
        routers = routers * router_mask.unsqueeze(-1)
        nodes = patches.nodes
        relation_gates = nodes.new_zeros(*nodes.shape[:2], len(MIXER_RELATIONS))
        for layer in self.layers:
            nodes, routers, relation_gates = layer(nodes, routers, edges, patches.node_mask, router_mask)
        variable_tokens = self._pool_variables(nodes, patches)
        return GraphMixerOutput(
            patch_tokens=nodes,
            variable_tokens=variable_tokens,
            router_tokens=routers,
            variable_mask=patches.variable_mask,
            router_mask=router_mask,
            relation_gates=relation_gates,
        )
