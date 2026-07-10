from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


TRAINING_STRATEGY_VERSION = "v3-source-profiles-main-adaptive"


@dataclass(frozen=True)
class BaselineTrainingProfile:
    optimizer: str
    loss: str
    lr: float
    weight_decay: float
    scheduler: str
    scheduler_step: str
    max_epochs: int
    early_stop_patience: int
    early_stop_start_epoch: int = 1
    pct_start: float | None = None
    cosine_t_max: int | None = None
    eta_min: float = 0.0
    gradient_clip: float | None = None


BASELINE_TRAINING_PROFILES = {
    "patchtst": BaselineTrainingProfile("adam", "mse", 1e-4, 0.0, "one_cycle", "batch", 100, 20, pct_start=0.3),
    "itransformer": BaselineTrainingProfile("adam", "mse", 1e-4, 0.0, "type1", "epoch", 10, 3),
    "timesnet": BaselineTrainingProfile("adam", "mse", 1e-4, 0.0, "type1", "epoch", 10, 3),
    "dlinear": BaselineTrainingProfile("adam", "mse", 1e-4, 0.0, "type1", "epoch", 10, 3),
    "time_llm": BaselineTrainingProfile("adam", "mse", 1e-3, 0.0, "one_cycle", "batch", 10, 10, pct_start=0.2),
    "timecma": BaselineTrainingProfile(
        "adamw", "mse", 1e-4, 1e-3, "cosine", "epoch", 100, 50,
        early_stop_start_epoch=50, cosine_t_max=50, eta_min=1e-6, gradient_clip=5.0,
    ),
}


def get_baseline_training_profile(name: str) -> BaselineTrainingProfile:
    try:
        return BASELINE_TRAINING_PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"No training profile for official baseline: {name}") from exc


def build_baseline_optimizer(model, profile):
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if profile.optimizer == "adam":
        return torch.optim.Adam(params, lr=profile.lr, weight_decay=profile.weight_decay)
    if profile.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=profile.lr, weight_decay=profile.weight_decay)
    raise ValueError(f"Unsupported optimizer: {profile.optimizer}")


def build_baseline_scheduler(optimizer, profile, steps_per_epoch):
    if profile.scheduler == "one_cycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=profile.lr,
            epochs=profile.max_epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=float(profile.pct_start),
        )
    if profile.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(profile.cosine_t_max),
            eta_min=profile.eta_min,
        )
    if profile.scheduler == "type1":
        return None
    raise ValueError(f"Unsupported scheduler: {profile.scheduler}")


def baseline_regression_loss(pred, target, profile):
    if profile.loss == "mse":
        return F.mse_loss(pred, target)
    raise ValueError(f"Unsupported baseline loss: {profile.loss}")


def step_baseline_batch_scheduler(scheduler, profile):
    if profile.scheduler_step == "batch" and scheduler is not None:
        scheduler.step()


def step_baseline_epoch_scheduler(scheduler, optimizer, profile, epoch):
    if profile.scheduler_step != "epoch":
        return
    if profile.scheduler == "type1":
        lr = profile.lr * (0.5 ** max(epoch - 1, 0))
        for group in optimizer.param_groups:
            group["lr"] = lr
    elif scheduler is not None:
        scheduler.step()


@dataclass(frozen=True)
class MainTrainingProfile:
    max_epochs: int = 80
    core_lr: float = 1e-3
    semantic_lr: float = 3e-4
    weight_decay: float = 1e-4
    lr_warmup_epochs: int = 5
    warmup_start_factor: float = 0.1
    plateau_factor: float = 0.5
    plateau_patience: int = 5
    core_min_lr: float = 1e-5
    semantic_min_lr: float = 3e-6
    align_start_epoch: int = 6
    align_full_epoch: int = 15
    align_weight: float = 1e-3
    early_stop_start_epoch: int = 20
    early_stop_patience: int = 20
    gradient_clip: float = 1.0


MAIN_TRAINING_PROFILE = MainTrainingProfile()


def _freeze_text_backbone(model) -> None:
    text_encoder = getattr(model, "text_encoder", None)
    backbone = getattr(text_encoder, "backbone", None)
    if backbone is not None:
        for parameter in backbone.parameters():
            parameter.requires_grad = False


def build_graph_report_optimizer(model, profile):
    _freeze_text_backbone(model)
    embedding_parameter_ids = {
        id(parameter)
        for module in model.modules()
        if isinstance(module, torch.nn.Embedding)
        for parameter in module.parameters(recurse=False)
    }
    groups = {
        "core_decay": {"params": [], "lr": profile.core_lr * profile.warmup_start_factor,
                       "weight_decay": profile.weight_decay, "role": "core"},
        "core_no_decay": {"params": [], "lr": profile.core_lr * profile.warmup_start_factor,
                          "weight_decay": 0.0, "role": "core"},
        "semantic_decay": {"params": [], "lr": profile.semantic_lr * profile.warmup_start_factor,
                            "weight_decay": profile.weight_decay, "role": "semantic"},
        "semantic_no_decay": {"params": [], "lr": profile.semantic_lr * profile.warmup_start_factor,
                               "weight_decay": 0.0, "role": "semantic"},
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        lower_name = name.lower()
        role = "semantic" if name.startswith(("text_encoder.proj", "semantic_fusion", "fusion")) else "core"
        uses_weight_decay = not (
            id(parameter) in embedding_parameter_ids
            or "norm" in lower_name
            or name.endswith(".bias")
            or "embed" in lower_name
        )
        group_name = f"{role}_{'decay' if uses_weight_decay else 'no_decay'}"
        groups[group_name]["params"].append(parameter)

    grouped_parameters = [parameter for group in groups.values() for parameter in group["params"]]
    grouped_ids = [id(parameter) for parameter in grouped_parameters]
    expected_ids = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if len(grouped_ids) != len(set(grouped_ids)):
        raise ValueError("GraphReportTS optimizer groups contain duplicate trainable parameters")
    if set(grouped_ids) != expected_ids:
        raise ValueError("GraphReportTS optimizer groups do not cover every trainable parameter")

    parameter_groups = []
    for name, group in groups.items():
        group["name"] = name
        parameter_groups.append(group)
    return torch.optim.AdamW(parameter_groups)


class GraphReportScheduler:
    def __init__(self, optimizer, profile):
        self.optimizer = optimizer
        self.profile = profile
        min_lrs = [
            profile.core_min_lr if group["role"] == "core" else profile.semantic_min_lr
            for group in optimizer.param_groups
        ]
        self.plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=profile.plateau_factor,
            patience=profile.plateau_patience,
            min_lr=min_lrs,
        )

    def _set_role_lrs(self, factor):
        for group in self.optimizer.param_groups:
            target_lr = self.profile.core_lr if group["role"] == "core" else self.profile.semantic_lr
            group["lr"] = target_lr * factor

    def start_epoch(self, epoch):
        if epoch <= self.profile.lr_warmup_epochs:
            progress = (epoch - 1) / max(self.profile.lr_warmup_epochs - 1, 1)
            factor = self.profile.warmup_start_factor + (1.0 - self.profile.warmup_start_factor) * progress
            self._set_role_lrs(factor)

    def step_validation(self, epoch, val_mse):
        if epoch >= self.profile.lr_warmup_epochs:
            self.plateau.step(val_mse)

    def state_dict(self):
        return {"plateau": self.plateau.state_dict()}

    def load_state_dict(self, state_dict):
        self.plateau.load_state_dict(state_dict["plateau"])


def graph_report_group_lrs(optimizer) -> dict[str, float]:
    lrs = {}
    for group in optimizer.param_groups:
        role = group["role"]
        lr = group["lr"]
        if role in lrs and lrs[role] != lr:
            raise ValueError(f"GraphReportTS optimizer groups disagree on {role} learning rate")
        lrs[role] = lr
    return lrs


def graph_report_align_weight(epoch, profile) -> float:
    if epoch < profile.align_start_epoch:
        return 0.0
    if epoch >= profile.align_full_epoch:
        return profile.align_weight
    initial_weight = profile.align_weight / (profile.align_full_epoch - profile.align_start_epoch + 1)
    progress = (epoch - profile.align_start_epoch) / (profile.align_full_epoch - profile.align_start_epoch)
    return initial_weight + (profile.align_weight - initial_weight) * progress
