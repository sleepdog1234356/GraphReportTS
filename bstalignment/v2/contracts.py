from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Literal

import torch


Domain = Literal["general", "battery"]
TextBackend = Literal["distilbert", "simple"]
GraphEmbeddingVariant = Literal[
    "patch",
    "series_context",
    "series_context_diff",
    "series_context_decomp",
    "global_node",
    "global_node_diff",
    "global_node_raw",
]
CorrectionGateMode = Literal["trainable", "fixed_one"]


@dataclass(frozen=True)
class GraphReportTSv2Config:
    domain: Domain
    input_len: int
    pred_len: int
    max_pred_len: int = 60
    d_model: int = 128
    graph_layers: int = 3
    heads: int = 4
    dropout: float = 0.1
    ffn_expansion: int = 2
    dense_variable_threshold: int = 64
    max_neighbors: int = 16
    max_routers: int = 16
    max_nodes: int = 6000
    max_variables: int = 512
    text_model: str = "hf_models/distilbert-base-uncased"
    text_max_length: int = 128
    text_token_cache_size: int = 32_768
    text_hidden_cache_size: int = 4_096
    text_hidden_cache_max_bytes: int = 512 * 1024 * 1024
    text_backend: TextBackend = "distilbert"
    use_text: bool = True
    freeze_text: bool = True
    graph_embedding_variant: GraphEmbeddingVariant = "patch"
    correction_gate_mode: CorrectionGateMode = "trainable"

    def __post_init__(self) -> None:
        if self.domain == "general":
            expected_pred = {
                36: (24, 36, 48, 60),
                96: (96, 192, 336, 720),
            }.get(self.input_len)
            if expected_pred is None:
                raise ValueError("general input_len must be 36 or 96")
        else:
            if self.input_len != 32:
                raise ValueError("battery input_len must be 32")
            expected_pred = (20,)
        if self.pred_len not in expected_pred:
            raise ValueError(
                f"{self.domain} pred_len must be one of {expected_pred} for input_len={self.input_len}"
            )
        if self.max_pred_len < self.pred_len:
            raise ValueError("max_pred_len must cover pred_len")
        if self.d_model < 1 or self.graph_layers < 1 or self.heads < 1 or self.ffn_expansion < 1:
            raise ValueError("model dimensions and layer counts must be positive")
        if self.d_model % self.heads:
            raise ValueError("d_model must be divisible by heads")
        if not 8 <= self.max_neighbors <= 16:
            raise ValueError("max_neighbors must be within 8..16")
        if not 4 <= self.max_routers <= 16:
            raise ValueError("max_routers must be within 4..16")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be within [0,1)")
        if self.max_nodes < 1:
            raise ValueError("max_nodes must be positive")
        if self.domain == "general" and self.input_len == 96:
            if not 13_482 <= self.max_nodes <= 14_000:
                raise ValueError("general input_len=96 requires max_nodes within 13482..14000")
        elif self.max_nodes > 6000:
            raise ValueError("max_nodes must not exceed the approved ceiling of 6000")
        if self.max_variables < 1:
            raise ValueError("max_variables must be positive")
        if not 1 <= self.text_max_length <= 128:
            raise ValueError("text_max_length must be within 1..128")
        if self.text_token_cache_size < 0:
            raise ValueError("text_token_cache_size must be non-negative")
        if self.text_hidden_cache_size < 0:
            raise ValueError("text_hidden_cache_size must be non-negative")
        if self.text_hidden_cache_max_bytes < 0:
            raise ValueError("text_hidden_cache_max_bytes must be non-negative")
        if self.text_backend not in ("distilbert", "simple"):
            raise ValueError("text_backend must be distilbert or simple")
        if self.use_text and not self.text_model:
            raise ValueError("text_model is required when use_text=True")
        if self.text_backend == "distilbert" and not self.freeze_text:
            raise ValueError("the approved v2 DistilBERT backbone must remain frozen")
        if self.graph_embedding_variant not in (
            "patch",
            "series_context",
            "series_context_diff",
            "series_context_decomp",
            "global_node",
            "global_node_diff",
            "global_node_raw",
        ):
            raise ValueError("unsupported graph_embedding_variant")
        if self.correction_gate_mode not in ("trainable", "fixed_one"):
            raise ValueError("correction_gate_mode must be trainable or fixed_one")
        if self.domain != "general" and self.correction_gate_mode != "trainable":
            raise ValueError("fixed correction gates are defined only for the general residual head")


def _move_dataclass(instance, device: torch.device | str, non_blocking: bool = False):
    values = {}
    for item in fields(instance):
        value = getattr(instance, item.name)
        values[item.name] = value.to(device, non_blocking=non_blocking) if torch.is_tensor(value) else value
    return replace(instance, **values)


def _pin_dataclass(instance):
    values = {}
    for item in fields(instance):
        value = getattr(instance, item.name)
        values[item.name] = value.pin_memory() if torch.is_tensor(value) else value
    return replace(instance, **values)


@dataclass
class ForecastBatchV2:
    values: torch.Tensor
    observed_mask: torch.Tensor
    reliability: torch.Tensor
    variable_type: torch.Tensor
    variable_mask: torch.Tensor
    prompts: list[str]
    target: torch.Tensor
    target_mask: torch.Tensor
    metadata: list[dict[str, object]]

    def to(self, device: torch.device | str, non_blocking: bool = False) -> "ForecastBatchV2":
        return _move_dataclass(self, device, non_blocking)

    def pin_memory(self) -> "ForecastBatchV2":
        return _pin_dataclass(self)


@dataclass
class BatteryRawBatchV2:
    base_values: torch.Tensor
    base_observed_mask: torch.Tensor
    base_reliability: torch.Tensor
    ic_curve: torch.Tensor
    ic_curve_mask: torch.Tensor
    ic_quality: torch.Tensor
    dv_curve: torch.Tensor
    dv_curve_mask: torch.Tensor
    dv_quality: torch.Tensor
    prompts: list[str]
    target: torch.Tensor
    target_mask: torch.Tensor
    metadata: list[dict[str, object]]

    def to(self, device: torch.device | str, non_blocking: bool = False) -> "BatteryRawBatchV2":
        return _move_dataclass(self, device, non_blocking)

    def pin_memory(self) -> "BatteryRawBatchV2":
        return _pin_dataclass(self)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_bool_mask(mask: torch.Tensor, shape: tuple[int, ...], name: str) -> None:
    _require(tuple(mask.shape) == shape, f"{name} must have shape {shape}, got {tuple(mask.shape)}")
    _require(mask.dtype == torch.bool, f"{name} must be boolean")


def _validate_unit_interval(value: torch.Tensor, name: str) -> None:
    _require(torch.isfinite(value).all().item(), f"{name} must be finite")
    _require(((value >= 0) & (value <= 1)).all().item(), f"{name} must be within 0..1")


def validate_batch(
    batch: ForecastBatchV2,
    domain: Domain,
    *,
    input_len: int | None = None,
    pred_len: int | None = None,
) -> None:
    _require(batch.values.ndim == 3, "values must be [B,L,F]")
    batch_size, length, variables = batch.values.shape
    expected_length = int(input_len if input_len is not None else (36 if domain == "general" else 32))
    _require(length == expected_length, f"{domain} numeric history must have length {expected_length}")
    _validate_bool_mask(batch.observed_mask, (batch_size, length, variables), "observed_mask")
    _require(tuple(batch.reliability.shape) == (batch_size, length, variables), "reliability shape mismatch")
    _validate_unit_interval(batch.reliability, "reliability")
    _validate_bool_mask(batch.variable_mask, (batch_size, variables), "variable_mask")
    _require(batch.variable_mask.any(dim=1).all().item(), "each sample must contain at least one real variable")
    _require(batch.variable_type.ndim in (1, 2), "variable_type must be [F] or [B,F]")
    _require(batch.variable_type.shape[-1] == variables, "variable_type width mismatch")
    _require(batch.variable_type.dtype == torch.long, "variable_type must use torch.long")
    _require(len(batch.prompts) == batch_size, "prompt count must equal batch size")
    _require(len(batch.metadata) == batch_size, "metadata count must equal batch size")
    _require(batch.target.ndim == 3 and batch.target.shape[0] == batch_size, "target must be [B,H,F_out]")
    _validate_bool_mask(batch.target_mask, tuple(batch.target.shape), "target_mask")
    expected_output = variables if domain == "general" else 1
    _require(batch.target.shape[-1] == expected_output, f"{domain} output width must be {expected_output}")
    expected_horizons = (
        (int(pred_len),)
        if pred_len is not None
        else ((24, 36, 48, 60) if domain == "general" else (20,))
    )
    _require(batch.target.shape[1] in expected_horizons, f"{domain} target horizon must be one of {expected_horizons}")
    _require(torch.isfinite(batch.values).all().item(), "filled values must be finite")
    active_observed = batch.observed_mask & batch.variable_mask.unsqueeze(1)
    _require(active_observed.flatten(1).any(dim=1).all().item(), "each sample must contain an observed numeric value")
    valid_target = batch.target.masked_select(batch.target_mask)
    _require(torch.isfinite(valid_target).all().item(), "observed targets must be finite")


def validate_battery_raw_batch(batch: BatteryRawBatchV2) -> None:
    _require(batch.base_values.ndim == 3, "base_values must be [B,32,50]")
    b, length, features = batch.base_values.shape
    _require((length, features) == (32, 50), "base_values must be [B,32,50]")
    _validate_bool_mask(batch.base_observed_mask, (b, 32, 50), "base_observed_mask")
    _require(torch.isfinite(batch.base_values).all().item(), "filled base_values must be finite")
    _require(tuple(batch.base_reliability.shape) == (b, 32, 50), "base_reliability shape mismatch")
    _validate_unit_interval(batch.base_reliability, "base_reliability")
    for prefix in ("ic", "dv"):
        curve = getattr(batch, f"{prefix}_curve")
        curve_mask = getattr(batch, f"{prefix}_curve_mask")
        quality = getattr(batch, f"{prefix}_quality")
        _require(tuple(curve.shape) == (b, 32, 128), f"{prefix}_curve must be [B,32,128]")
        _require(torch.isfinite(curve).all().item(), f"filled {prefix}_curve must be finite")
        _validate_bool_mask(curve_mask, (b, 32, 128), f"{prefix}_curve_mask")
        _require(tuple(quality.shape) == (b, 32), f"{prefix}_quality must be [B,32]")
        _validate_unit_interval(quality, f"{prefix}_quality")
    _require(tuple(batch.target.shape) == (b, 20, 1), "battery target must be [B,20,1]")
    _validate_bool_mask(batch.target_mask, (b, 20, 1), "target_mask")
    _require(len(batch.prompts) == b and len(batch.metadata) == b, "batch text metadata count mismatch")
    valid_target = batch.target.masked_select(batch.target_mask)
    _require(torch.isfinite(valid_target).all().item(), "observed battery targets must be finite")


def require_local_text_model(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"local DistilBERT directory does not exist: {resolved}")
    if not (resolved / "config.json").exists():
        raise FileNotFoundError(f"local DistilBERT directory {resolved} is missing config.json")
    weight_files = (
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    )
    if not any((resolved / name).exists() for name in weight_files):
        raise FileNotFoundError(f"local DistilBERT directory {resolved} has no model weights")
    tokenizer_files = ("tokenizer.json", "vocab.txt")
    if not any((resolved / name).exists() for name in tokenizer_files):
        raise FileNotFoundError(f"local DistilBERT directory {resolved} has no tokenizer.json or vocab.txt")
    return resolved
