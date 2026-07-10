from __future__ import annotations

from dataclasses import dataclass


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
