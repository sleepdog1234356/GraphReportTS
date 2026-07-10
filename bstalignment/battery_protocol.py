from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import numpy as np


BATTERY_INPUT_CYCLES = 32
BATTERY_PREDICTION_CYCLES = 20
BATTERY_TARGET_PROTOCOL = "32-observed-20-future-only-full-horizon"
BATTERY_CYCLE_SCALE_PROTOCOL = "train-split-max-cycle-id-no-clip"


def split_mit_items(
    items: Sequence[Any],
    *,
    seed: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Dict[str, list[Any]]:
    values = list(items)
    rng = np.random.default_rng(seed)
    order = np.arange(len(values))
    rng.shuffle(order)
    n_train = int(len(order) * train_ratio)
    n_val = int(len(order) * val_ratio)
    train = [values[int(index)] for index in order[:n_train]]
    val = [values[int(index)] for index in order[n_train : n_train + n_val]]
    test = [values[int(index)] for index in order[n_train + n_val :]]
    return {"train": train, "val": val, "test": test, "all": values}


def split_processed_items(items: Sequence[Any], *, seed: int) -> Dict[str, list[Any]]:
    values = list(items)
    rng = np.random.default_rng(seed)
    order = np.arange(len(values))
    rng.shuffle(order)
    if len(order) >= 3:
        n_train = max(1, int(len(order) * 0.7))
        n_val = max(1, int(len(order) * 0.15))
        if n_train + n_val >= len(order):
            n_train = max(1, len(order) - 2)
            n_val = 1
    elif len(order) == 2:
        n_train, n_val = 1, 0
    else:
        n_train, n_val = len(order), 0
    return {
        "train": [values[int(index)] for index in order[:n_train]],
        "val": [values[int(index)] for index in order[n_train : n_train + n_val]],
        "test": [values[int(index)] for index in order[n_train + n_val :]],
        "all": values,
    }


def fit_cycle_scale(cycle_id_arrays: Iterable[np.ndarray], max_cycles: int | None) -> float:
    maximum = 1.0
    for values in cycle_id_arrays:
        cycle_ids = np.asarray(values, dtype=np.float64).reshape(-1)
        if max_cycles is not None:
            cycle_ids = cycle_ids[: int(max_cycles)]
        finite = cycle_ids[np.isfinite(cycle_ids)]
        if finite.size:
            maximum = max(maximum, float(finite.max()))
    return maximum


def fit_processed_cycle_scale(train_paths: Iterable[str | Path], max_cycles: int | None) -> float:
    def cycle_arrays():
        for path in train_paths:
            with np.load(Path(path), allow_pickle=True) as data:
                if "cycle_id" not in data:
                    raise ValueError(f"{path} is missing required array: cycle_id")
                yield np.array(data["cycle_id"], copy=True)

    return fit_cycle_scale(cycle_arrays(), max_cycles)
