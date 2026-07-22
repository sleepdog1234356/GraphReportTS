from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class GTRTrainingConfig:
    epochs: int = 80
    core_lr: float = 1e-3
    semantic_lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    align_start_epoch: int = 6
    align_full_epoch: int = 15
    align_weight: float = 1e-3
    gradient_clip: float = 1.0
    patience: int = 20
    plateau_factor: float = 0.5
    plateau_patience: int = 4
    plateau_threshold: float = 1e-3
    plateau_cooldown: int = 1
    core_min_lr: float = 1e-5
    semantic_min_lr: float = 3e-6
    min_lr_reductions_before_stop: int = 2


def alignment_weight(epoch: int, config: GTRTrainingConfig) -> float:
    if epoch < config.align_start_epoch:
        return 0.0
    if epoch >= config.align_full_epoch:
        return config.align_weight
    progress = (epoch - config.align_start_epoch + 1) / max(
        config.align_full_epoch - config.align_start_epoch + 1,
        1,
    )
    return config.align_weight * progress


def build_optimizer(model: torch.nn.Module, config: GTRTrainingConfig) -> torch.optim.Optimizer:
    semantic, core = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (semantic if ("text_encoder.proj" in name or "semantic_fusion" in name) else core).append(parameter)
    groups = [{"params": core, "lr": config.core_lr, "role": "core"}]
    if semantic:
        groups.append({"params": semantic, "lr": config.semantic_lr, "role": "semantic"})
    return torch.optim.AdamW(groups, weight_decay=config.weight_decay)


def apply_warmup(optimizer: torch.optim.Optimizer, epoch: int, config: GTRTrainingConfig) -> None:
    if config.warmup_epochs <= 0 or epoch > config.warmup_epochs:
        return
    factor = min(1.0, max(epoch, 1) / max(config.warmup_epochs, 1))
    for group in optimizer.param_groups:
        base = config.semantic_lr if group["role"] == "semantic" else config.core_lr
        group["lr"] = base * factor


def build_plateau_scheduler(
    optimizer: torch.optim.Optimizer,
    config: GTRTrainingConfig,
) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
    minimum_lrs = [
        config.semantic_min_lr if group.get("role") == "semantic" else config.core_min_lr
        for group in optimizer.param_groups
    ]
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.plateau_factor,
        patience=config.plateau_patience,
        threshold=config.plateau_threshold,
        threshold_mode="rel",
        cooldown=config.plateau_cooldown,
        min_lr=minimum_lrs,
    )


def step_plateau_scheduler(
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    optimizer: torch.optim.Optimizer,
    validation_mse: float,
) -> bool:
    before = [float(group["lr"]) for group in optimizer.param_groups]
    scheduler.step(float(validation_mse))
    after = [float(group["lr"]) for group in optimizer.param_groups]
    return any(current < previous for previous, current in zip(before, after))


def optimizer_learning_rates(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    return {
        str(group.get("role", f"group_{index}")): float(group["lr"])
        for index, group in enumerate(optimizer.param_groups)
    }


def should_stop_gtr(stale: int, lr_reductions: int, config: GTRTrainingConfig) -> bool:
    return (
        int(stale) >= config.patience
        and int(lr_reductions) >= config.min_lr_reductions_before_stop
    )


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_mse: float,
    extra: dict[str, object] | None = None,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "val_mse": val_mse, **(extra or {})},
        path,
    )
