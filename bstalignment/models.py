from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MovingAverageDecomposition(nn.Module):
    """Differentiable moving-average decomposition.

    For battery SOH we interpret:
      trend ~= slow degradation component
      residual ~= regeneration/noise/transient protocol perturbation
    """

    def __init__(self, kernel_size: int = 5):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.pad = self.kernel_size // 2
        self.avg = nn.AvgPool1d(kernel_size=self.kernel_size, stride=1, padding=self.pad, count_include_pad=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, L, F]
        trend = self.avg(x.transpose(1, 2))[:, :, : x.size(1)].transpose(1, 2)
        residual = x - trend
        return trend, residual


def decomposition_independence_loss(trend: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    """Penalize covariance between trend and residual components.

    This is a lightweight analog of STEM-LTS decomposition regularization.
    """
    b, l, f = trend.shape
    z1 = trend.reshape(b, -1)
    z2 = residual.reshape(b, -1)
    z1 = z1 - z1.mean(dim=0, keepdim=True)
    z2 = z2 - z2.mean(dim=0, keepdim=True)
    cov = (z1 * z2).mean(dim=0)
    return (cov ** 2).mean()


class InvertedTimeSeriesEncoder(nn.Module):
    """TimeCMA/iTransformer-style inverted encoder.

    Treat each feature variable as a token; the whole seq_len history of that variable is projected
    into d_model. Transformer self-attention then models feature interactions.
    """

    def __init__(
        self,
        seq_len: int,
        num_features: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.num_features = num_features
        self.value_proj = nn.Linear(seq_len, d_model)
        self.feature_embed = nn.Parameter(torch.randn(num_features, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, F]
        tokens = self.value_proj(x.transpose(1, 2))  # [B, F, D]
        tokens = tokens + self.feature_embed.unsqueeze(0)
        tokens = self.encoder(tokens)
        return self.norm(tokens)


class HFTextEncoder(nn.Module):
    """Frozen HuggingFace encoder for prompts.

    Supports BERT-like and GPT-like models. It returns both token states and pooled states.
    """

    def __init__(self, model_name: str, d_model: int, freeze: bool = True, max_length: int = 192):
        super().__init__()
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:
            raise ImportError("Please install transformers to use HFTextEncoder.") from e
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.sep_token or self.tokenizer.cls_token
        self.backbone = AutoModel.from_pretrained(model_name)
        self.hidden_size = int(getattr(self.backbone.config, "hidden_size", getattr(self.backbone.config, "n_embd", 768)))
        self.proj = nn.Linear(self.hidden_size, d_model)
        self.max_length = max_length
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

    def forward(self, prompts: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        tok = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        tok = {k: v.to(device) for k, v in tok.items()}
        with torch.set_grad_enabled(any(p.requires_grad for p in self.backbone.parameters())):
            out = self.backbone(**tok)
        hidden = out.last_hidden_state  # [B, T, H]
        mask = tok.get("attention_mask", torch.ones(hidden.shape[:2], device=device)).bool()
        pooled = (hidden * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        return self.proj(hidden), self.proj(pooled), mask


class SimpleTextEncoder(nn.Module):
    """Fallback text encoder if no local HuggingFace model is available."""

    def __init__(self, d_model: int, vocab_size: int = 4096, max_length: int = 192):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.vocab_size = vocab_size
        self.max_length = max_length

    def _hash_tokens(self, text: str) -> List[int]:
        toks = text.lower().split()[: self.max_length]
        return [abs(hash(t)) % self.vocab_size for t in toks] or [0]

    def forward(self, prompts: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        ids = [self._hash_tokens(p) for p in prompts]
        max_len = min(max(len(x) for x in ids), self.max_length)
        arr = torch.zeros(len(ids), max_len, dtype=torch.long, device=device)
        mask = torch.zeros(len(ids), max_len, dtype=torch.bool, device=device)
        for i, row in enumerate(ids):
            row = row[:max_len]
            arr[i, : len(row)] = torch.tensor(row, device=device)
            mask[i, : len(row)] = True
        hidden = self.proj(self.emb(arr))
        pooled = (hidden * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        return hidden, pooled, mask


class CrossModalRetrieval(nn.Module):
    """Time-series queries retrieve useful semantic components from prompt token states."""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Sequential(nn.Linear(d_model, d_model), nn.Dropout(dropout), nn.LayerNorm(d_model))

    def forward(
        self, ts_tokens: torch.Tensor, text_tokens: torch.Tensor, text_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # ts_tokens: [B, F, D], text_tokens: [B, T, D]
        q = self.q(ts_tokens)
        k = self.k(text_tokens)
        v = self.v(text_tokens)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.size(-1))  # [B,F,T]
        if text_mask is not None:
            scores = scores.masked_fill(~text_mask.unsqueeze(1), -1e4)
        attn = torch.softmax(scores, dim=-1)
        retrieved = torch.matmul(attn, v)
        fused = ts_tokens + self.out(retrieved)
        return fused, attn


class SemanticAnchorHead(nn.Module):
    """Learned semantic anchors for aging stages."""

    def __init__(self, d_model: int, num_anchors: int = 3):
        super().__init__()
        self.anchors = nn.Parameter(torch.randn(num_anchors, d_model) * 0.02)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = F.normalize(z, dim=-1)
        anchors = F.normalize(self.anchors, dim=-1)
        return z @ anchors.t()


class CycleFeatureEncoder(nn.Module):
    """Encode one cycle's numeric features as feature tokens."""

    def __init__(
        self,
        num_features: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.value_proj = nn.Linear(1, d_model)
        self.feature_embed = nn.Parameter(torch.randn(num_features, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, F]
        tokens = self.value_proj(x.unsqueeze(-1))
        tokens = tokens + self.feature_embed.unsqueeze(0)
        return self.norm(self.encoder(tokens))


@dataclass
class BatteryCycleLLMAssistConfig:
    num_features: int
    max_horizon: int = 20
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    text_model: str = "distilbert-base-uncased"
    use_hf_text_encoder: bool = True
    freeze_text: bool = True
    text_max_length: int = 128


class BatteryCycleLLMAssist(nn.Module):
    """One-cycle LLM-assisted SOH estimator and horizon-aware forecaster."""

    def __init__(self, cfg: BatteryCycleLLMAssistConfig):
        super().__init__()
        self.cfg = cfg
        self.numeric_encoder = CycleFeatureEncoder(
            cfg.num_features, cfg.d_model, cfg.n_heads, cfg.n_layers, cfg.dropout
        )
        if cfg.use_hf_text_encoder:
            self.text_encoder = HFTextEncoder(cfg.text_model, cfg.d_model, cfg.freeze_text, cfg.text_max_length)
        else:
            self.text_encoder = SimpleTextEncoder(cfg.d_model, max_length=cfg.text_max_length)
        self.align = CrossModalRetrieval(cfg.d_model, cfg.dropout)
        self.horizon_embed = nn.Embedding(cfg.max_horizon + 1, cfg.d_model)
        self.step_embed = nn.Embedding(cfg.max_horizon + 1, cfg.d_model)
        self.gate = nn.Sequential(
            nn.Linear(cfg.d_model * 3, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.Sigmoid(),
        )
        self.context_norm = nn.LayerNorm(cfg.d_model)
        self.now_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, 1),
        )
        self.future_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, 1),
        )

    def forward(self, x: torch.Tensor, prompts: List[str], horizon: torch.Tensor | int) -> Dict[str, torch.Tensor]:
        numeric_tokens = self.numeric_encoder(x)
        numeric_repr = numeric_tokens.mean(dim=1)
        text_tokens, text_repr, text_mask = self.text_encoder(prompts)
        aligned_tokens, attn = self.align(numeric_tokens, text_tokens, text_mask)
        retrieved_repr = aligned_tokens.mean(dim=1)

        if not torch.is_tensor(horizon):
            horizon = torch.full((x.size(0),), int(horizon), device=x.device, dtype=torch.long)
        horizon = horizon.to(device=x.device, dtype=torch.long).clamp(min=1, max=self.cfg.max_horizon)
        horizon_repr = self.horizon_embed(horizon)
        gate = self.gate(torch.cat([numeric_repr, retrieved_repr, horizon_repr], dim=-1))
        context = self.context_norm(numeric_repr + gate * retrieved_repr + horizon_repr)

        soh_now = self.now_head(context).squeeze(-1)
        out_h = int(horizon.max().item())
        step_ids = torch.arange(1, out_h + 1, device=x.device, dtype=torch.long)
        step_repr = self.step_embed(step_ids).unsqueeze(0).expand(x.size(0), -1, -1)
        future_context = context.unsqueeze(1) + horizon_repr.unsqueeze(1) + step_repr
        soh_future = self.future_head(future_context).squeeze(-1)
        future_steps = step_ids.unsqueeze(0).expand(x.size(0), -1)
        future_valid = future_steps <= horizon.unsqueeze(1)
        return {
            "soh_now": soh_now,
            "soh_future": soh_future,
            "future_mask": future_valid,
            "soh_all": torch.cat([soh_now.unsqueeze(1), soh_future], dim=1),
            "numeric_repr": context,
            "text_repr": text_repr,
            "attn": attn,
            "gate": gate,
        }


@dataclass
class BatterySTAlignConfig:
    seq_len: int
    num_features: int
    forecast_horizon: int = 1
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    decomp_kernel: int = 5
    text_model: str = "distilbert-base-uncased"
    use_hf_text_encoder: bool = True
    freeze_text: bool = True
    text_max_length: int = 192


class BatterySTAlign(nn.Module):
    """Battery SOH semantic-temporal alignment model.

    For multi-horizon inference, out["soh"] has shape [B, forecast_horizon].
    """

    def __init__(self, cfg: BatterySTAlignConfig):
        super().__init__()
        self.cfg = cfg
        self.decomp = MovingAverageDecomposition(cfg.decomp_kernel)
        self.ts_raw = InvertedTimeSeriesEncoder(cfg.seq_len, cfg.num_features, cfg.d_model, cfg.n_heads, cfg.n_layers, cfg.dropout)
        self.ts_trend = InvertedTimeSeriesEncoder(cfg.seq_len, cfg.num_features, cfg.d_model, cfg.n_heads, 1, cfg.dropout)
        self.ts_res = InvertedTimeSeriesEncoder(cfg.seq_len, cfg.num_features, cfg.d_model, cfg.n_heads, 1, cfg.dropout)
        self.component_gate = nn.Sequential(nn.Linear(cfg.d_model * 3, cfg.d_model), nn.GELU(), nn.LayerNorm(cfg.d_model))
        if cfg.use_hf_text_encoder:
            self.text_encoder = HFTextEncoder(cfg.text_model, cfg.d_model, cfg.freeze_text, cfg.text_max_length)
        else:
            self.text_encoder = SimpleTextEncoder(cfg.d_model, max_length=cfg.text_max_length)
        self.align = CrossModalRetrieval(cfg.d_model, cfg.dropout)
        self.pool = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(), nn.LayerNorm(cfg.d_model))
        self.soh_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, cfg.forecast_horizon),
        )
        self.stage_head = nn.Linear(cfg.d_model, 3)
        self.anchor_head = SemanticAnchorHead(cfg.d_model, 3)

    def forward(self, x: torch.Tensor, prompts: List[str]) -> Dict[str, torch.Tensor]:
        trend, residual = self.decomp(x)
        raw_tok = self.ts_raw(x)
        trend_tok = self.ts_trend(trend)
        res_tok = self.ts_res(residual)
        ts_tokens = self.component_gate(torch.cat([raw_tok, trend_tok, res_tok], dim=-1))
        text_tokens, text_pooled, text_mask = self.text_encoder(prompts)
        fused_tokens, attn = self.align(ts_tokens, text_tokens, text_mask)
        z = self.pool(fused_tokens.mean(dim=1))
        soh = self.soh_head(z)  # [B, H]
        stage_logits = self.stage_head(z)
        anchor_logits = self.anchor_head(z)
        return {
            "soh": soh,
            "stage_logits": stage_logits,
            "anchor_logits": anchor_logits,
            "ts_repr": z,
            "text_repr": text_pooled,
            "attn": attn,
            "trend": trend,
            "residual": residual,
        }
