"""Shared, source-native contracts for formal general baseline runs.

The module deliberately does not import official source trees or text-model
dependencies. Model construction remains lazy in :mod:`baseline_adapters`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Any, Mapping

import numpy as np
import torch


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
    if epoch < mechanics.early_stop_start_epoch:
        return ValidationDecision(float(best_mse), 0, False, False)
    next_stale = int(stale) + 1
    return ValidationDecision(
        float(best_mse), next_stale, False, next_stale >= mechanics.early_stop_patience
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
) -> dict[str, Any]:
    """Build the common standardized-space result and audit record."""

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
        "selection": {
            "metric": "validation_mse",
            "best_epoch": int(best_epoch),
            "best_val_mse": float(best_val_mse),
        },
    }
