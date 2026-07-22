"""Audited per-cell training strategies for the L36 AnchoredGTR matrix."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping


DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "ECL")
HORIZONS = (24, 36, 48, 60)


@dataclass(frozen=True)
class AnchoredGTRStrategy:
    name: str
    seed: int
    freeze_linear_anchor: bool
    correction_gate_mode: str
    epochs: int
    patience: int
    batch_size: int
    num_workers: int = 8
    training_overrides: Mapping[str, float | int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.correction_gate_mode not in ("trainable", "fixed_one"):
            raise ValueError("invalid correction gate mode")
        if self.epochs < 1 or self.patience < 1 or self.batch_size < 1 or self.num_workers < 0:
            raise ValueError("invalid AnchoredGTR training strategy")


def _frozen(*, seed: int = 42, batch_size: int = 512) -> AnchoredGTRStrategy:
    return AnchoredGTRStrategy(
        name="ridge_frozen",
        seed=seed,
        freeze_linear_anchor=True,
        correction_gate_mode="trainable",
        epochs=80,
        patience=20,
        batch_size=batch_size,
    )


def _trainable_fixed(*, batch_size: int) -> AnchoredGTRStrategy:
    return AnchoredGTRStrategy(
        name="ridge_trainable_correction_fixed_one",
        seed=42,
        freeze_linear_anchor=False,
        correction_gate_mode="fixed_one",
        epochs=5,
        patience=2,
        batch_size=batch_size,
    )


def _weather_a1() -> AnchoredGTRStrategy:
    return AnchoredGTRStrategy(
        name="weather_validation_calibrated_a1",
        seed=42,
        freeze_linear_anchor=False,
        correction_gate_mode="fixed_one",
        epochs=12,
        patience=4,
        batch_size=160,
        training_overrides=MappingProxyType(
            {
                "core_lr": 4e-4,
                "semantic_lr": 1.2e-4,
                "weight_decay": 5e-4,
                "warmup_epochs": 2,
                "align_start_epoch": 4,
                "align_full_epoch": 8,
                "align_weight": 3e-4,
                "gradient_clip": 1.0,
                "plateau_factor": 0.5,
                "plateau_patience": 1,
                "plateau_threshold": 1e-3,
                "plateau_cooldown": 0,
                "core_min_lr": 2.5e-5,
                "semantic_min_lr": 7.5e-6,
                "min_lr_reductions_before_stop": 1,
            }
        ),
    )


_REGISTRY: dict[tuple[str, int], AnchoredGTRStrategy] = {}
for dataset in ("ETTh1", "ETTm1"):
    for horizon in HORIZONS:
        _REGISTRY[(dataset, horizon)] = _frozen()
for horizon in HORIZONS:
    _REGISTRY[("ETTh2", horizon)] = _trainable_fixed(batch_size=512)
_REGISTRY[("ETTm2", 24)] = _frozen()
_REGISTRY[("ETTm2", 36)] = _frozen()
_REGISTRY[("ETTm2", 48)] = _trainable_fixed(batch_size=512)
_REGISTRY[("ETTm2", 60)] = _trainable_fixed(batch_size=512)
_REGISTRY[("Weather", 24)] = _trainable_fixed(batch_size=160)
_REGISTRY[("Weather", 36)] = _frozen(seed=43, batch_size=160)
_REGISTRY[("Weather", 48)] = _trainable_fixed(batch_size=160)
_REGISTRY[("Weather", 60)] = _weather_a1()
for horizon in HORIZONS:
    _REGISTRY[("ECL", horizon)] = _frozen(batch_size=80)

STRATEGY_REGISTRY: Mapping[tuple[str, int], AnchoredGTRStrategy] = MappingProxyType(_REGISTRY)


def resolve_strategy(dataset: str, horizon: int) -> AnchoredGTRStrategy:
    try:
        return STRATEGY_REGISTRY[(str(dataset), int(horizon))]
    except KeyError as exc:
        raise ValueError(f"unsupported AnchoredGTR L36 cell: {dataset}/H{horizon}") from exc


__all__ = [
    "DATASETS",
    "HORIZONS",
    "STRATEGY_REGISTRY",
    "AnchoredGTRStrategy",
    "resolve_strategy",
]
