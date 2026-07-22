from __future__ import annotations

import math

import torch
import torch.nn as nn


class FixedLogitGate(nn.Module):
    """Parameter-free gate whose sigmoid is a registered constant."""

    def __init__(self, value: float) -> None:
        super().__init__()
        if not 0.0 < value <= 1.0:
            raise ValueError("FixedLogitGate requires a value within (0,1]")
        logit = math.inf if value == 1.0 else math.log(value / (1.0 - value))
        self.value = float(value)
        self.register_buffer("logit", torch.tensor(logit))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.logit.to(dtype=value.dtype).expand(*value.shape[:-1], 1)


class GeneralResidualHead(nn.Module):
    def __init__(self, input_len: int, max_pred_len: int, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear_anchor = nn.Linear(input_len, max_pred_len)
        self.step_embedding = nn.Embedding(max_pred_len, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.correction = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, 1))
        self.correction_gate = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        nn.init.zeros_(self.correction_gate[-1].weight)
        nn.init.constant_(self.correction_gate[-1].bias, math.log(0.1 / 0.9))

    def forward(
        self,
        history: torch.Tensor,
        variable_tokens: torch.Tensor,
        horizon: int,
        variable_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        base = self.linear_anchor(history.transpose(1, 2))[:, :, :horizon].transpose(1, 2)
        steps = self.step_embedding(torch.arange(horizon, device=history.device))
        query = self.norm(variable_tokens.unsqueeze(1) + steps.view(1, horizon, 1, -1))
        delta = self.correction(query).squeeze(-1)
        gate = torch.sigmoid(self.correction_gate(query)).squeeze(-1)
        prediction = (base + gate * delta) * variable_mask.unsqueeze(1)
        return prediction, gate


class BatterySOHHead(nn.Module):
    def __init__(self, d_model: int, heads: int, pred_len: int = 20, dropout: float = 0.1) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.step_embedding = nn.Embedding(pred_len, d_model)
        self.cross_attention = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model * 2, 1)
        )

    def forward(self, context: torch.Tensor, variable_tokens: torch.Tensor, variable_mask: torch.Tensor) -> torch.Tensor:
        steps = self.step_embedding(torch.arange(self.pred_len, device=context.device))
        query = context.unsqueeze(1) + steps.unsqueeze(0)
        retrieved, _ = self.cross_attention(
            query, variable_tokens, variable_tokens, key_padding_mask=~variable_mask, need_weights=False
        )
        return self.output(self.norm(query + retrieved))
