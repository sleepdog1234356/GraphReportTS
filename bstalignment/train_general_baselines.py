"""Shared, source-native contracts for formal general baseline runs.

The module deliberately does not import official source trees or text-model
dependencies. Model construction remains lazy in :mod:`baseline_adapters`.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


_MINUTE_DATASETS = {"ETTm1", "ETTm2", "Weather"}


def source_time_markers(dataset: str, timestamps: Any) -> torch.Tensor:
    """Build the official THUML ``timeF`` marker columns from Task 3 timestamps."""

    import pandas as pd

    if dataset not in {"ETTm1", "ETTm2", "ETTh1", "ETTh2", "ECL", "Weather"}:
        raise ValueError(f"unknown formal general dataset: {dataset}")
    index = pd.DatetimeIndex(pd.to_datetime(tuple(timestamps), errors="raise"))
    features = []
    if dataset in _MINUTE_DATASETS:
        features.append(index.minute.to_numpy(dtype=np.float32) / 59.0 - 0.5)
    features.extend(
        [
            index.hour.to_numpy(dtype=np.float32) / 23.0 - 0.5,
            index.dayofweek.to_numpy(dtype=np.float32) / 6.0 - 0.5,
            (index.day.to_numpy(dtype=np.float32) - 1.0) / 30.0 - 0.5,
            (index.dayofyear.to_numpy(dtype=np.float32) - 1.0) / 365.0 - 0.5,
        ]
    )
    return torch.from_numpy(np.stack(features, axis=1).astype(np.float32, copy=False))


def collate_general_baseline_batch(samples: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Collate Task 3 windows without dropping their official time markers."""

    if not samples:
        raise ValueError("general baseline collation requires at least one sample")
    datasets = {str(sample["series_id"]) for sample in samples}
    if len(datasets) != 1:
        raise ValueError("one general baseline batch cannot mix datasets")
    dataset = datasets.pop()
    return {
        "dataset": dataset,
        "x": torch.stack([torch.as_tensor(sample["history_scaled"]) for sample in samples]),
        "y": torch.stack([torch.as_tensor(sample["target_scaled"]) for sample in samples]),
        "x_mark": torch.stack(
            [source_time_markers(dataset, sample["timestamp_markers"]["history"]) for sample in samples]
        ),
        "y_mark": torch.stack(
            [source_time_markers(dataset, sample["timestamp_markers"]["target"]) for sample in samples]
        ),
        "start_index": torch.tensor([int(sample["start_index"]) for sample in samples], dtype=torch.long),
        "columns": tuple(samples[0]["columns"]),
        "timestamp_markers": [sample["timestamp_markers"] for sample in samples],
    }


def forward_general_baseline_batch(
    adapter: torch.nn.Module,
    batch: Mapping[str, Any],
    prompt_embeddings: torch.Tensor | None = None,
) -> torch.Tensor:
    """Forward a collated Task 3 batch through the shared adapter contract."""

    return adapter(
        batch["x"],
        time_mark=batch["x_mark"],
        decoder_time_mark=batch["y_mark"],
        prompt_embeddings=prompt_embeddings,
    )


def build_shared_general_datasets(dataset: str, data_root: str, pred_len: int):
    """Build all splits with the Task 3 train-fitted scaler instance."""

    try:
        from .data_general import GeneralForecastGraphDataset
    except ImportError:
        from data_general import GeneralForecastGraphDataset

    train = GeneralForecastGraphDataset(
        dataset, data_root=data_root, split="train", input_len=36, pred_len=pred_len, fit_scaler=True
    )
    validation = GeneralForecastGraphDataset(
        dataset, data_root=data_root, split="val", input_len=36, pred_len=pred_len,
        scaler=train.scaler, fit_scaler=False,
    )
    test = GeneralForecastGraphDataset(
        dataset, data_root=data_root, split="test", input_len=36, pred_len=pred_len,
        scaler=train.scaler, fit_scaler=False,
    )
    return train, validation, test


def scaler_checksum(scaler: Any) -> str:
    """Return a stable checksum for fitted column-wise Task 3 statistics."""

    digest = hashlib.sha256()
    for name in ("mean", "std"):
        value = getattr(scaler, name, None)
        if value is None:
            raise ValueError(f"general scaler must be fitted before checksumming: missing {name}")
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(name.encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(repr(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def build_general_optimizer(model: torch.nn.Module, profile: Any) -> torch.optim.Optimizer:
    mechanics = profile.training
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError(f"{profile.name} exposes no trainable parameters")
    if mechanics.optimizer == "adam":
        return torch.optim.Adam(parameters, lr=mechanics.lr, weight_decay=mechanics.weight_decay)
    if mechanics.optimizer == "adamw":
        return torch.optim.AdamW(parameters, lr=mechanics.lr, weight_decay=mechanics.weight_decay)
    raise ValueError(f"unsupported general baseline optimizer: {mechanics.optimizer}")


def build_general_scheduler(
    optimizer: torch.optim.Optimizer,
    profile: Any,
    steps_per_epoch: int,
):
    mechanics = profile.training
    if mechanics.scheduler == "one_cycle":
        if steps_per_epoch <= 0:
            raise ValueError("OneCycleLR requires a positive number of steps per epoch")
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=mechanics.lr,
            epochs=mechanics.max_epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=float(mechanics.pct_start),
        )
    if mechanics.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(mechanics.cosine_t_max),
            eta_min=mechanics.eta_min,
        )
    if mechanics.scheduler in {"type1", "type3"}:
        return None
    raise ValueError(f"unsupported general baseline scheduler: {mechanics.scheduler}")


def step_general_batch_scheduler(scheduler: Any, profile: Any) -> None:
    if profile.training.scheduler_step == "batch" and scheduler is not None:
        scheduler.step()


def clip_general_gradients(model: torch.nn.Module, profile: Any):
    max_norm = profile.training.gradient_clip
    if max_norm is None:
        return None
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    return torch.nn.utils.clip_grad_norm_(parameters, float(max_norm))


def step_general_optimizer(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    profile: Any,
    scheduler: Any = None,
):
    """Apply source clipping, optimizer step, then source batch scheduling."""

    total_norm = clip_general_gradients(model, profile)
    optimizer.step()
    step_general_batch_scheduler(scheduler, profile)
    return total_norm


def step_general_epoch_scheduler(
    scheduler: Any,
    optimizer: torch.optim.Optimizer,
    profile: Any,
    epoch: int,
) -> None:
    """Apply source formulas using the source scripts' one-based epoch number."""

    mechanics = profile.training
    if mechanics.scheduler_step != "epoch":
        return
    if epoch < 1:
        raise ValueError("source scheduler epoch numbers are one-based")
    if mechanics.scheduler == "type1":
        learning_rate = mechanics.lr * (0.5 ** (epoch - 1))
    elif mechanics.scheduler == "type3":
        learning_rate = mechanics.lr if epoch < 3 else mechanics.lr * (0.9 ** (epoch - 3))
    else:
        if scheduler is not None:
            scheduler.step()
        return
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


@dataclass(frozen=True)
class ValidationDecision:
    best_mse: float
    stale: int
    should_save: bool
    should_stop: bool


def validation_checkpoint_decision(
    *,
    best_mse: float,
    stale: int,
    val_mse: float,
    epoch: int,
    profile: Any,
) -> ValidationDecision:
    """Select and stop from validation MSE only, never test metrics."""

    mechanics = profile.training
    if val_mse < best_mse:
        return ValidationDecision(float(val_mse), 0, True, False)
    next_stale = int(stale) + 1
    return ValidationDecision(
        float(best_mse),
        next_stale,
        False,
        epoch >= mechanics.early_stop_start_epoch and next_stale >= mechanics.early_stop_patience,
    )


def general_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("general metrics require matching [batch, horizon, variables] tensors")
    difference = prediction.detach().float() - target.detach().float()
    return {
        "mse": float(torch.mean(difference.square()).cpu()),
        "mae": float(torch.mean(torch.abs(difference)).cpu()),
    }


def _mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): item for key, item in value.items()}


def general_result_record(
    *,
    profile: Any,
    seed: int,
    metrics: Mapping[str, float],
    scaler_checksum: str,
    best_epoch: int,
    best_val_mse: float,
    prompt_provenance: Mapping[str, Any] | None,
    runtime_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the common standardized-space result and audit record."""

    runtime = _validate_runtime_provenance(profile, runtime_provenance)

    return {
        "model": profile.name,
        "dataset": profile.dataset,
        "horizon": profile.pred_len,
        "seed": int(seed),
        "metrics_space": "standardized",
        "metrics": {"mse": float(metrics["mse"]), "mae": float(metrics["mae"])},
        "source": profile.source.as_dict(),
        "source_evidence": list(profile.source_evidence),
        "architecture": _mapping(profile.architecture),
        "training": profile.training.as_dict(),
        "protocol_overrides": _mapping(profile.protocol_overrides),
        "precision": profile.precision,
        "scaler_checksum": scaler_checksum,
        "prompt_provenance": dict(prompt_provenance) if prompt_provenance is not None else None,
        "runtime_provenance": runtime,
        "selection": {
            "metric": "validation_mse",
            "best_epoch": int(best_epoch),
            "best_val_mse": float(best_val_mse),
        },
    }


def _validate_runtime_provenance(profile: Any, runtime_provenance: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(runtime_provenance, Mapping):
        raise ValueError("formal result requires adapter runtime provenance")
    runtime = {str(key): value for key, value in runtime_provenance.items()}
    checkout = runtime.get("source_checkout")
    if not isinstance(checkout, Mapping):
        raise ValueError("runtime provenance requires source_checkout")
    full_sha = checkout.get("full_sha")
    if checkout.get("verified") is not True or not isinstance(full_sha, str) or len(full_sha) != 40:
        raise ValueError("runtime provenance requires a verified full source SHA")
    if checkout.get("manifest_revision") != profile.source.commit:
        raise ValueError("runtime provenance manifest revision does not match the resolved profile")

    if profile.name == "Time-LLM":
        text = runtime.get("time_llm")
        if not isinstance(text, Mapping):
            raise ValueError("Time-LLM result requires actual runtime provenance")
        placeholders = {"", "unknown", "placeholder", "required-at-runtime", "none"}
        required = (
            "model_path", "tokenizer_path", "model_revision", "tokenizer_revision", "precision", "backbone_dtype"
        )
        for field in required:
            value = text.get(field)
            if value is None or str(value).strip().lower() in placeholders:
                raise ValueError(f"Time-LLM runtime provenance has placeholder {field}")
        for field in ("model_path", "tokenizer_path"):
            path = Path(str(text[field]))
            if not path.is_absolute() or not path.exists():
                raise ValueError(f"Time-LLM runtime provenance requires existing absolute {field}")
        if text["precision"] not in {"bf16", "float16", "float32"}:
            raise ValueError("Time-LLM runtime provenance has unsupported precision")
    return runtime
