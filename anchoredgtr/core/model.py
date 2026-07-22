from __future__ import annotations

from dataclasses import asdict

import torch
import torch.nn as nn

from ..models import SimpleTextEncoder
from .contracts import (
    BatteryRawBatch,
    ForecastBatch,
    GTRConfig,
    require_local_text_model,
    validate_batch,
    validate_battery_raw_batch,
)
from .edges import SparseEdgeBuilder
from .graph_mixer import GraphMixer
from .heads import BatterySOHHead, FixedLogitGate, GeneralResidualHead
from .patching import AdaptivePatchifier
from .semantic import FrozenDistilBERTEncoder, GatedSemanticFusion


class GraphTextResidualCore(nn.Module):
    """Shared graph/text body with domain-specific prediction heads only."""

    def __init__(self, config: GTRConfig) -> None:
        super().__init__()
        self.config = config
        self.patchifier = AdaptivePatchifier(
            config.d_model,
            config.max_variables,
            dropout=config.dropout,
            history_len=config.input_len,
            embedding_variant=config.graph_embedding_variant,
        )
        self.edge_builder = SparseEdgeBuilder(
            config.dense_variable_threshold,
            config.max_neighbors,
            max_variables=config.max_variables,
            d_model=config.d_model,
        )
        self.graph_mixer = GraphMixer(
            d_model=config.d_model,
            layers=config.graph_layers,
            heads=config.heads,
            dropout=config.dropout,
            ffn_expansion=config.ffn_expansion,
            max_routers=config.max_routers,
        )
        if config.use_text:
            if config.text_backend == "distilbert":
                local_path = require_local_text_model(config.text_model)
                self.text_encoder = FrozenDistilBERTEncoder(
                    str(local_path),
                    config.d_model,
                    config.text_max_length,
                    config.text_token_cache_size,
                    config.text_hidden_cache_size,
                    config.text_hidden_cache_max_bytes,
                )
            else:
                # This branch is an explicit smoke-test replacement, never an
                # automatic production fallback for missing DistilBERT files.
                self.text_encoder = SimpleTextEncoder(config.d_model, max_length=config.text_max_length)
            self.semantic_fusion = GatedSemanticFusion(config.d_model, config.dropout, initial_gate=0.4)
        else:
            self.text_encoder = None
            self.semantic_fusion = None
        if config.domain == "general":
            self.head = GeneralResidualHead(config.input_len, config.max_pred_len, config.d_model, config.dropout)
            if config.correction_gate_mode == "fixed_one":
                self.head.correction_gate = FixedLogitGate(1.0)
        else:
            self.head = BatterySOHHead(config.d_model, config.heads, config.pred_len, config.dropout)

    @staticmethod
    def _masked_context(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weight = mask.float() / mask.sum(-1, keepdim=True).clamp_min(1)
        return (tokens * weight.unsqueeze(-1)).sum(1)

    def forward(self, batch: ForecastBatch) -> dict[str, torch.Tensor | object]:
        validate_batch(
            batch,
            self.config.domain,
            input_len=self.config.input_len,
            pred_len=self.config.pred_len,
        )
        patches = self.patchifier(
            batch.values,
            batch.observed_mask,
            batch.reliability,
            batch.variable_type,
            batch.variable_mask,
            max_nodes=self.config.max_nodes,
        )
        edges = self.edge_builder(patches, batch.values, batch.observed_mask)
        graph = self.graph_mixer(patches, edges)
        if self.text_encoder is not None and self.semantic_fusion is not None:
            text_tokens, _, text_mask = self.text_encoder(batch.prompts, audit=True)
            semantic = self.semantic_fusion(
                graph.variable_tokens,
                graph.router_tokens,
                batch.variable_mask,
                graph.router_mask,
                text_tokens,
                text_mask,
                self.text_encoder.last_prompt_audit,
            )
            variable_tokens = semantic.variable_tokens
            context = semantic.context
            gate = semantic.gate
            cross_attention = semantic.cross_attention
            align_graph = semantic.align_graph
            align_text = semantic.align_text
            prompt_audit = semantic.prompt_audit
        else:
            variable_tokens = graph.variable_tokens
            context = self._masked_context(variable_tokens, batch.variable_mask)
            gate = context.new_zeros(context.size(0), 1)
            cross_attention = context.new_zeros(
                context.size(0), variable_tokens.size(1) + graph.router_tokens.size(1), 0
            )
            align_graph = context
            align_text = context.detach()
            prompt_audit = None
        if self.config.domain == "general":
            prediction, correction_gate = self.head(
                batch.values, variable_tokens, self.config.pred_len, batch.variable_mask
            )
        else:
            prediction = self.head(context, variable_tokens, batch.variable_mask)
            correction_gate = prediction.new_zeros(prediction.shape)
        return {
            "pred": prediction,
            "context": context,
            "variable_tokens": variable_tokens,
            "router_tokens": graph.router_tokens,
            "gate": gate,
            "correction_gate": correction_gate,
            "cross_attention": cross_attention,
            "align_graph": align_graph,
            "align_text": align_text,
            "text_enabled": self.text_encoder is not None,
            "prompt_audit": prompt_audit,
            "graph_diagnostics": {
                "real_nodes": patches.real_node_count,
                "edge_counts": {name: int(edge.edge_index.size(1)) for name, edge in edges.relations.items()},
                "widths": patches.widths,
                "variable_score_mix": edges.variable_score_mix.detach(),
                "relation_gate_mean": graph.relation_gates.detach().mean(dim=(0, 1)),
                "router_count": graph.router_tokens.size(1),
            },
        }

    def export_config(self) -> dict[str, object]:
        return asdict(self.config)


class BatteryGTRCore(nn.Module):
    """Defined here as an integration boundary; the adapter is imported lazily."""

    def __init__(self, config: GTRConfig) -> None:
        super().__init__()
        if config.domain != "battery":
            raise ValueError("BatteryGTRCore requires domain=battery")
        from .battery_adapter import BatteryFeatureAdapter

        self.feature_adapter = BatteryFeatureAdapter()
        self.shared = GraphTextResidualCore(config)

    def forward(self, raw_batch: BatteryRawBatch):
        validate_battery_raw_batch(raw_batch)
        adapted = self.feature_adapter(
            raw_batch.base_values,
            raw_batch.base_observed_mask,
            raw_batch.base_reliability,
            raw_batch.ic_curve,
            raw_batch.ic_curve_mask,
            raw_batch.ic_quality,
            raw_batch.dv_curve,
            raw_batch.dv_curve_mask,
            raw_batch.dv_quality,
        )
        variable_mask = adapted.observed_mask.any(dim=1)
        forecast_batch = ForecastBatch(
            values=adapted.values,
            observed_mask=adapted.observed_mask,
            reliability=adapted.reliability,
            variable_type=adapted.variable_type,
            variable_mask=variable_mask,
            prompts=raw_batch.prompts,
            target=raw_batch.target,
            target_mask=raw_batch.target_mask,
            metadata=raw_batch.metadata,
        )
        output = self.shared(forecast_batch)
        output["adapted_values"] = forecast_batch.values
        return output
