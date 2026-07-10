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


def _valid_n_heads(d_model: int, requested: int) -> int:
    heads = max(1, min(int(requested), int(d_model)))
    while heads > 1 and d_model % heads != 0:
        heads -= 1
    return heads


class RawSequenceEncoder(nn.Module):
    """Encode each cycle's unpatched multivariate raw sequence."""

    def __init__(
        self,
        input_dim: int = 6,
        d_model: int = 128,
        max_length: int = 128,
        layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Embedding(max_length, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=_valid_n_heads(d_model, n_heads),
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.pool = nn.Sequential(nn.Linear(d_model, d_model), nn.Tanh(), nn.Linear(d_model, 1))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        b, t, length, channels = sequences.shape
        flat = sequences.reshape(b * t, length, channels)
        pos = torch.arange(length, device=sequences.device)
        tokens = self.encoder(self.input_proj(flat.float()) + self.pos_embed(pos).unsqueeze(0))
        weights = torch.softmax(self.pool(tokens).squeeze(-1), dim=-1)
        pooled = self.norm(torch.sum(tokens * weights.unsqueeze(-1), dim=1))
        return pooled.reshape(b, t, -1)


class InterCycleTemporalEncoder(nn.Module):
    """First-pass temporal encoder over per-cycle graph embeddings."""

    def __init__(
        self,
        d_model: int,
        max_history_len: int = 64,
        layers: int = 1,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pos_embed = nn.Embedding(max_history_len, d_model)
        heads = _valid_n_heads(d_model, n_heads)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=max(int(layers), 1))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, cycle_repr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # cycle_repr: [B,T,D]
        b, t, _ = cycle_repr.shape
        pos = torch.arange(t, device=cycle_repr.device).clamp(max=self.pos_embed.num_embeddings - 1)
        x = cycle_repr + self.pos_embed(pos).unsqueeze(0)
        tokens = self.norm(self.encoder(x))
        return tokens[:, -1], tokens


class NumericHistoryEncoder(nn.Module):
    """Encode direct cycle-level numeric history used by sequence baselines."""

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        max_history_len: int = 64,
        layers: int = 1,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.input_proj = nn.Linear(self.input_dim, d_model)
        self.pos_embed = nn.Embedding(max_history_len, d_model)
        heads = _valid_n_heads(d_model, n_heads)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=max(int(layers), 1))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, history_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if history_features.size(-1) < self.input_dim:
            pad = self.input_dim - history_features.size(-1)
            history_features = F.pad(history_features, (0, pad))
        elif history_features.size(-1) > self.input_dim:
            history_features = history_features[..., : self.input_dim]
        b, t, _ = history_features.shape
        pos = torch.arange(t, device=history_features.device).clamp(max=self.pos_embed.num_embeddings - 1)
        x = self.input_proj(history_features.float()) + self.pos_embed(pos).unsqueeze(0)
        tokens = self.norm(self.encoder(x))
        return tokens[:, -1], tokens


class GatedSemanticFusion(nn.Module):
    """Token-aware text retrieval followed by learnable prompt gating."""

    def __init__(self, d_model: int, dropout: float = 0.1, use_gate: bool = True):
        super().__init__()
        self.use_gate = bool(use_gate)
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.text_out = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.Dropout(dropout))
        self.gate = (
            nn.Sequential(
                nn.LayerNorm(d_model * 3),
                nn.Linear(d_model * 3, d_model),
                nn.GELU(),
                nn.Linear(d_model, 1),
            )
            if self.use_gate
            else None
        )

    def forward(
        self,
        base_context: torch.Tensor,
        query_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        q = self.q(query_tokens)
        k = self.k(text_tokens)
        v = self.v(text_tokens)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.size(-1))
        if text_mask is not None:
            scores = scores.masked_fill(~text_mask.unsqueeze(1), -1e4)
        attn = torch.softmax(scores, dim=-1)
        retrieved = torch.matmul(attn, v)
        text_context = self.text_out(retrieved.mean(dim=1))
        if self.gate is not None:
            gate = torch.sigmoid(self.gate(torch.cat([base_context, text_context, base_context * text_context], dim=-1)))
        else:
            gate = torch.ones(base_context.size(0), 1, dtype=base_context.dtype, device=base_context.device)
        context = base_context + gate * text_context
        return {
            "context": context,
            "gate": gate,
            "cross_attn": attn,
            "align_graph": query_tokens.mean(dim=1),
            "align_text": retrieved.mean(dim=1),
            "text_context": text_context,
        }


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
    battery_history_len: int = 32
    history_feature_dim: int = 8
    use_multi_cycle_raw: bool = True
    single_cycle_raw: bool = False
    use_numeric_history: bool = True
    use_text_gate: bool = True
    use_semantic_alignment: bool = True
    use_relative_steps: bool = True
    temporal_layers: int = 1
    temporal_heads: int = 4
    battery_input_mode: str = "hankel_graph"
    raw_sequence_len: int = 128
    raw_sequence_dim: int = 6


class GraphReportTS(nn.Module):
    """Unified backbone for Battery-GraphReportTS and General-GraphReportTS."""

    def __init__(self, cfg: GraphReportTSConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.battery_input_mode not in {"hankel_graph", "raw_sequence"}:
            raise ValueError(f"Unknown battery_input_mode: {cfg.battery_input_mode}")
        if cfg.battery_input_mode == "hankel_graph":
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
            self.raw_sequence_encoder = None
        else:
            self.graph_encoder = None
            self.raw_sequence_encoder = RawSequenceEncoder(
                input_dim=cfg.raw_sequence_dim,
                d_model=cfg.d_model,
                max_length=cfg.raw_sequence_len,
                layers=2,
                n_heads=cfg.temporal_heads,
                dropout=cfg.dropout,
            )
        if cfg.use_report_prompt and cfg.use_cross_modal_fusion and cfg.use_hf_text_encoder:
            self.text_encoder = HFTextEncoder(cfg.text_model, cfg.d_model, cfg.freeze_text, cfg.text_max_length)
        elif cfg.use_report_prompt and cfg.use_cross_modal_fusion:
            self.text_encoder = SimpleTextEncoder(cfg.d_model, max_length=cfg.text_max_length)
        else:
            self.text_encoder = None
        if cfg.use_report_prompt and cfg.use_cross_modal_fusion:
            self.fusion = CrossModalFusion(cfg.d_model, cfg.dropout)
            self.semantic_fusion = GatedSemanticFusion(cfg.d_model, cfg.dropout, use_gate=cfg.use_text_gate)
        else:
            self.fusion = None
            self.semantic_fusion = None
        self.temporal_encoder = InterCycleTemporalEncoder(
            cfg.d_model,
            max_history_len=cfg.battery_history_len,
            layers=cfg.temporal_layers,
            n_heads=cfg.temporal_heads,
            dropout=cfg.dropout,
        )
        self.numeric_history_encoder = NumericHistoryEncoder(
            cfg.history_feature_dim,
            cfg.d_model,
            max_history_len=cfg.battery_history_len,
            layers=cfg.temporal_layers,
            n_heads=cfg.temporal_heads,
            dropout=cfg.dropout,
        )
        self.context_fuser = nn.Sequential(
            nn.LayerNorm(cfg.d_model * 2),
            nn.Linear(cfg.d_model * 2, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.context_norm = nn.LayerNorm(cfg.d_model)
        decoder_cls = UnifiedQueryDecoder if cfg.unified_decoder else SeparateNowFutureDecoder
        self.decoder = decoder_cls(cfg.d_model, cfg.output_dim, cfg.max_steps, cfg.dropout)

    def _encode_graph_history(self, maps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if maps.ndim == 5:
            b, t, c, h, w = maps.shape
            if not self.cfg.use_multi_cycle_raw:
                zeros = torch.zeros(b, t, self.cfg.d_model, dtype=maps.dtype, device=maps.device)
                return zeros[:, -1], zeros, {"tokens": zeros, "graph_attn": None}
            if self.cfg.single_cycle_raw:
                graph = self.graph_encoder(maps[:, -1])
                tokens = graph["repr"].unsqueeze(1)
                return graph["repr"], tokens, graph
            if t > 1 and self.cfg.use_multi_cycle_raw:
                flat_maps = maps.reshape(b * t, c, h, w)
                graph = self.graph_encoder(flat_maps)
                cycle_repr = graph["repr"].reshape(b, t, -1)
                if t == 1:
                    return cycle_repr[:, -1], cycle_repr, graph
                context, tokens = self.temporal_encoder(cycle_repr)
                return context, tokens, graph
        graph = self.graph_encoder(maps if maps.ndim == 4 else maps[:, -1])
        tokens = graph["repr"].unsqueeze(1)
        return graph["repr"], tokens, graph

    def _encode_battery_history(
        self,
        maps: Optional[torch.Tensor],
        raw_sequences: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Optional[torch.Tensor]]]:
        if self.cfg.battery_input_mode == "raw_sequence":
            if maps is not None or raw_sequences is None:
                raise ValueError("raw_sequence mode requires raw_sequences and forbids maps")
            cycle_repr = self.raw_sequence_encoder(raw_sequences)
            context, tokens = self.temporal_encoder(cycle_repr)
            return context, tokens, {"tokens": tokens, "graph_attn": None}
        if maps is None or raw_sequences is not None:
            raise ValueError("hankel_graph mode requires maps and forbids raw_sequences")
        return self._encode_graph_history(maps)

    def forward(
        self,
        maps: Optional[torch.Tensor],
        prompts: List[str],
        horizon: torch.Tensor | int,
        steps: Optional[torch.Tensor] = None,
        history_features: Optional[torch.Tensor] = None,
        raw_sequences: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        graph_context, graph_cycle_tokens, graph = self._encode_battery_history(maps, raw_sequences)
        if history_features is not None and self.cfg.use_numeric_history:
            numeric_context, numeric_tokens = self.numeric_history_encoder(history_features)
        else:
            b = graph_context.size(0)
            numeric_context = torch.zeros(b, self.cfg.d_model, dtype=graph_context.dtype, device=graph_context.device)
            numeric_tokens = torch.zeros(b, 1, self.cfg.d_model, dtype=graph_context.dtype, device=graph_context.device)
        context = self.context_fuser(torch.cat([graph_context, numeric_context], dim=-1))
        query_tokens = torch.cat([graph_cycle_tokens, numeric_tokens], dim=1)
        text_repr = None
        cross_attn = None
        gate = torch.zeros(context.size(0), 1, dtype=context.dtype, device=context.device)
        align_graph = context
        align_text = context.detach()
        if self.text_encoder is not None and self.cfg.use_report_prompt and self.cfg.use_cross_modal_fusion:
            text_tokens, text_repr, text_mask = self.text_encoder(prompts)
            if self.cfg.battery_input_mode == "raw_sequence" or maps.ndim == 5 or self.cfg.variant == "battery":
                fused = self.semantic_fusion(context, query_tokens, text_tokens, text_mask)
            else:
                fused = self.fusion(graph["tokens"], graph["repr"], text_tokens, text_mask)
                fused["gate"] = torch.ones(context.size(0), 1, dtype=context.dtype, device=context.device)
                fused["align_graph"] = fused["context"]
                fused["align_text"] = text_repr
            context = fused["context"]
            cross_attn = fused["cross_attn"]
            gate = fused["gate"]
            align_graph = fused["align_graph"]
            align_text = fused["align_text"] if self.cfg.use_semantic_alignment else text_repr
        context = self.context_norm(context)
        if steps is None:
            if not torch.is_tensor(horizon):
                max_h = int(horizon)
            else:
                max_h = int(horizon.max().item())
            start = 1 if (self.cfg.variant == "battery" and self.cfg.use_relative_steps) else 1
            steps = torch.arange(start, max_h + start, device=graph_context.device)
        pred = self.decoder(context, steps)
        return {
            "pred": pred,
            "context": context,
            "graph_tokens": graph_cycle_tokens,
            "graph_attn": graph["graph_attn"],
            "text_repr": text_repr if text_repr is not None else context.detach() * 0,
            "cross_attn": cross_attn,
            "gate": gate,
            "align_graph": align_graph,
            "align_text": align_text,
        }


def build_graph_report_model(
    variant: str,
    output_dim: int,
    d_model: int = 128,
    **kwargs,
) -> GraphReportTS:
    cfg = GraphReportTSConfig(variant=variant, output_dim=output_dim, d_model=d_model, **kwargs)
    return GraphReportTS(cfg)
