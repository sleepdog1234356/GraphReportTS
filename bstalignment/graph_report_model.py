from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .models import HFTextEncoder, SimpleTextEncoder
    from .raw_signal import maps_to_channel_patch_nodes
except ImportError:
    from models import HFTextEncoder, SimpleTextEncoder
    from raw_signal import maps_to_channel_patch_nodes


class DynamicGraphBlock(nn.Module):
    """Lightweight dynamic graph attention over 2D-map patch nodes."""

    def __init__(self, d_model: int, topk: int = 4, dropout: float = 0.1, use_dynamic_graph: bool = True):
        super().__init__()
        self.topk = int(topk)
        self.use_dynamic_graph = bool(use_dynamic_graph)
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.edge_proj = nn.Linear(d_model, d_model)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, nodes: torch.Tensor, structural_bias: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # nodes: [B,N,D]
        q = F.normalize(self.q(nodes), dim=-1)
        k = F.normalize(self.k(nodes), dim=-1)
        v = self.v(nodes)
        if self.use_dynamic_graph:
            scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.size(-1))
        else:
            scores = torch.zeros(nodes.size(0), nodes.size(1), nodes.size(1), dtype=nodes.dtype, device=nodes.device)
        if structural_bias is not None:
            scores = scores + structural_bias.unsqueeze(0)
        n = scores.size(-1)
        if self.topk > 0 and self.topk < n:
            vals, idx = torch.topk(scores, k=self.topk, dim=-1)
            mask = torch.full_like(scores, -1e4)
            scores = mask.scatter(-1, idx, vals)
        attn = torch.softmax(scores, dim=-1)
        msg = torch.matmul(attn, v)
        nodes = self.norm(nodes + self.dropout(self.edge_proj(msg)))
        nodes = nodes + self.dropout(self.ffn(nodes))
        return nodes, attn


class GraphMapEncoder(nn.Module):
    """Encode multi-view 2D maps as a dynamic patch graph."""

    def __init__(
        self,
        d_model: int = 128,
        patch_size: int = 8,
        patch_stride: int = 4,
        graph_layers: int = 2,
        topk_edges: int = 4,
        dropout: float = 0.1,
        max_map_channels: int = 128,
        use_domain_edges: bool = True,
        use_dynamic_graph: bool = True,
    ):
        super().__init__()
        self.patch_size = int(patch_size)
        self.patch_stride = int(patch_stride)
        self.input_proj = nn.LazyLinear(d_model)
        self.d_model = int(d_model)
        self.use_domain_edges = bool(use_domain_edges)
        self.channel_embed = nn.Embedding(max_map_channels, d_model)
        self.row_proj = nn.Linear(1, d_model)
        self.col_proj = nn.Linear(1, d_model)
        self.layers = nn.ModuleList(
            [DynamicGraphBlock(d_model, topk_edges, dropout, use_dynamic_graph) for _ in range(graph_layers)]
        )
        self.pool = nn.Sequential(nn.Linear(d_model, d_model), nn.Tanh(), nn.Linear(d_model, 1))
        self.norm = nn.LayerNorm(d_model)

    def _structural_bias(self, meta: torch.Tensor, grid: Tuple[int, int, int], dtype: torch.dtype) -> torch.Tensor:
        channels, rows, cols = grid
        ch = meta[:, 0]
        row = meta[:, 1]
        col = meta[:, 2]
        same_ch = ch[:, None] == ch[None, :]
        same_row = row[:, None] == row[None, :]
        same_col = col[:, None] == col[None, :]
        temporal = same_ch & same_row & (torch.abs(col[:, None] - col[None, :]) == 1)
        delay = same_ch & same_col & (torch.abs(row[:, None] - row[None, :]) == 1)
        variable = same_row & same_col & (ch[:, None] != ch[None, :])
        self_loop = torch.eye(meta.size(0), dtype=torch.bool, device=meta.device)
        bias = torch.zeros(meta.size(0), meta.size(0), dtype=dtype, device=meta.device)
        bias = bias + temporal.to(dtype) * 0.35
        bias = bias + delay.to(dtype) * 0.25
        bias = bias + variable.to(dtype) * 0.20
        if self.use_domain_edges:
            # Neighboring derived maps from the same raw channel are stored
            # consecutively: hankel, d1, d2, and optional IC/DV maps. This soft
            # prior connects each map to its adjacent derivative/domain views.
            domain = (torch.abs(ch[:, None] - ch[None, :]) == 1) & same_row & same_col
            bias = bias + domain.to(dtype) * 0.15
        return bias + self_loop.to(dtype) * 0.10

    def forward(self, maps: torch.Tensor) -> Dict[str, torch.Tensor]:
        patches, meta, grid = maps_to_channel_patch_nodes(maps, self.patch_size, self.patch_stride)
        nodes = self.input_proj(patches)
        ch = meta[:, 0].clamp(max=self.channel_embed.num_embeddings - 1)
        rows = max(grid[1] - 1, 1)
        cols = max(grid[2] - 1, 1)
        row_pos = (meta[:, 1].float() / rows).unsqueeze(-1)
        col_pos = (meta[:, 2].float() / cols).unsqueeze(-1)
        nodes = nodes + self.channel_embed(ch).unsqueeze(0) + self.row_proj(row_pos).unsqueeze(0) + self.col_proj(col_pos).unsqueeze(0)
        structural_bias = self._structural_bias(meta, grid, nodes.dtype)
        attn_last = None
        for layer in self.layers:
            nodes, attn_last = layer(nodes, structural_bias)
        pool_w = torch.softmax(self.pool(nodes).squeeze(-1), dim=-1)
        graph_repr = torch.sum(nodes * pool_w.unsqueeze(-1), dim=1)
        return {
            "tokens": self.norm(nodes),
            "repr": self.norm(graph_repr),
            "graph_attn": attn_last,
            "pool_weight": pool_w,
        }


class CrossModalFusion(nn.Module):
    """Fuse graph tokens with report prompt tokens."""

    def __init__(self, d_model: int = 128, dropout: float = 0.1):
        super().__init__()
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Sequential(nn.Linear(d_model, d_model), nn.Dropout(dropout), nn.LayerNorm(d_model))

    def forward(
        self,
        graph_tokens: torch.Tensor,
        graph_repr: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        q = self.q(graph_tokens)
        k = self.k(text_tokens)
        v = self.v(text_tokens)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.size(-1))
        if text_mask is not None:
            scores = scores.masked_fill(~text_mask.unsqueeze(1), -1e4)
        attn = torch.softmax(scores, dim=-1)
        retrieved = torch.matmul(attn, v)
        fused_tokens = graph_tokens + self.out(retrieved)
        context = fused_tokens.mean(dim=1) + graph_repr
        return {"tokens": fused_tokens, "context": context, "cross_attn": attn}


class UnifiedQueryDecoder(nn.Module):
    """Shared decoder for current estimation and arbitrary future horizon."""

    def __init__(self, d_model: int, out_dim: int = 1, max_steps: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.step_embed = nn.Embedding(max_steps + 1, d_model)
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, out_dim),
        )

    def forward(self, context: torch.Tensor, steps: torch.Tensor) -> torch.Tensor:
        # context: [B,D], steps: [S] or [B,S]
        if steps.ndim == 1:
            step = steps.unsqueeze(0).expand(context.size(0), -1)
        else:
            step = steps
        step = step.to(context.device).long().clamp(min=0, max=self.step_embed.num_embeddings - 1)
        query = context.unsqueeze(1) + self.step_embed(step)
        return self.mlp(query)


class SeparateNowFutureDecoder(nn.Module):
    """Ablation decoder with separate current and future heads."""

    def __init__(self, d_model: int, out_dim: int = 1, max_steps: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.step_embed = nn.Embedding(max_steps + 1, d_model)
        self.now_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, out_dim),
        )
        self.future_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, out_dim),
        )

    def forward(self, context: torch.Tensor, steps: torch.Tensor) -> torch.Tensor:
        if steps.ndim == 1:
            step = steps.unsqueeze(0).expand(context.size(0), -1)
        else:
            step = steps
        step = step.to(context.device).long().clamp(min=0, max=self.step_embed.num_embeddings - 1)
        query = context.unsqueeze(1) + self.step_embed(step)
        future = self.future_head(query)
        now = self.now_head(context).unsqueeze(1)
        return torch.where((step == 0).unsqueeze(-1), now.expand_as(future), future)


@dataclass
class GraphReportTSConfig:
    variant: str = "battery"  # battery or general
    d_model: int = 128
    output_dim: int = 1
    max_steps: int = 1024
    patch_size: int = 8
    patch_stride: int = 4
    graph_layers: int = 2
    topk_edges: int = 4
    dropout: float = 0.1
    use_domain_edges: bool = True
    text_model: str = "distilbert-base-uncased"
    use_hf_text_encoder: bool = True
    freeze_text: bool = True
    text_max_length: int = 192
    use_report_prompt: bool = True
    use_cross_modal_fusion: bool = True
    use_dynamic_graph: bool = True
    unified_decoder: bool = True


class GraphReportTS(nn.Module):
    """Unified backbone for Battery-GraphReportTS and General-GraphReportTS."""

    def __init__(self, cfg: GraphReportTSConfig):
        super().__init__()
        self.cfg = cfg
        self.graph_encoder = GraphMapEncoder(
            d_model=cfg.d_model,
            patch_size=cfg.patch_size,
            patch_stride=cfg.patch_stride,
            graph_layers=cfg.graph_layers,
            topk_edges=cfg.topk_edges,
            dropout=cfg.dropout,
            use_domain_edges=cfg.use_domain_edges,
            use_dynamic_graph=cfg.use_dynamic_graph,
        )
        if cfg.use_report_prompt and cfg.use_cross_modal_fusion and cfg.use_hf_text_encoder:
            self.text_encoder = HFTextEncoder(cfg.text_model, cfg.d_model, cfg.freeze_text, cfg.text_max_length)
        elif cfg.use_report_prompt and cfg.use_cross_modal_fusion:
            self.text_encoder = SimpleTextEncoder(cfg.d_model, max_length=cfg.text_max_length)
        else:
            self.text_encoder = None
        self.fusion = CrossModalFusion(cfg.d_model, cfg.dropout)
        self.context_norm = nn.LayerNorm(cfg.d_model)
        decoder_cls = UnifiedQueryDecoder if cfg.unified_decoder else SeparateNowFutureDecoder
        self.decoder = decoder_cls(cfg.d_model, cfg.output_dim, cfg.max_steps, cfg.dropout)

    def forward(
        self,
        maps: torch.Tensor,
        prompts: List[str],
        horizon: torch.Tensor | int,
        steps: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        graph = self.graph_encoder(maps)
        context = graph["repr"]
        text_repr = None
        cross_attn = None
        if self.text_encoder is not None and self.cfg.use_report_prompt and self.cfg.use_cross_modal_fusion:
            text_tokens, text_repr, text_mask = self.text_encoder(prompts)
            fused = self.fusion(graph["tokens"], graph["repr"], text_tokens, text_mask)
            context = fused["context"]
            cross_attn = fused["cross_attn"]
        context = self.context_norm(context)
        if steps is None:
            if not torch.is_tensor(horizon):
                max_h = int(horizon)
            else:
                max_h = int(horizon.max().item())
            start = 0 if self.cfg.variant == "battery" else 1
            steps = torch.arange(start, max_h + 1, device=maps.device)
        pred = self.decoder(context, steps)
        return {
            "pred": pred,
            "context": context,
            "graph_tokens": graph["tokens"],
            "graph_attn": graph["graph_attn"],
            "text_repr": text_repr if text_repr is not None else context.detach() * 0,
            "cross_attn": cross_attn,
        }


def build_graph_report_model(
    variant: str,
    output_dim: int,
    d_model: int = 128,
    **kwargs,
) -> GraphReportTS:
    cfg = GraphReportTSConfig(variant=variant, output_dim=output_dim, d_model=d_model, **kwargs)
    return GraphReportTS(cfg)
