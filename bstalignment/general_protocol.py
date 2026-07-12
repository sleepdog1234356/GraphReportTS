from __future__ import annotations

from typing import Optional

import numpy as np


FORMAL_HISTORY = 36
_ETTH_BOUNDS = (8_640, 11_520, 14_400)
_ETTM_BOUNDS = (34_560, 46_080, 57_600)
_ETTH_DATASETS = frozenset({"ETTh1", "ETTh2"})
_ETTM_DATASETS = frozenset({"ETTm1", "ETTm2"})
_CHRONOLOGICAL_DATASETS = frozenset({"ECL", "Weather"})


class StandardScalerNP:
    """Column-wise standard scaler used by the general forecasting protocol."""

    def __init__(self):
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, values: np.ndarray) -> "StandardScalerNP":
        self.mean = np.nanmean(values, axis=0)
        self.std = np.nanstd(values, axis=0)
        self.std[self.std < 1e-6] = 1.0
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            return values.astype(np.float32)
        return ((values - self.mean) / self.std).astype(np.float32)


def split_bounds(dataset: str, n_rows: int, input_len: int) -> dict[str, tuple[int, int]]:
    """Return target intervals for one canonical general forecasting dataset."""

    if input_len != FORMAL_HISTORY:
        raise ValueError(f"formal general forecasting requires input_len={FORMAL_HISTORY}")
    if dataset in _ETTH_DATASETS:
        train_end, validation_end, test_end = _ETTH_BOUNDS
    elif dataset in _ETTM_DATASETS:
        train_end, validation_end, test_end = _ETTM_BOUNDS
    elif dataset in _CHRONOLOGICAL_DATASETS:
        train_end = int(n_rows * 0.7)
        validation_end = int(n_rows * 0.8)
        test_end = n_rows
    else:
        raise ValueError(f"unknown formal general dataset: {dataset}")
    if n_rows < test_end:
        raise ValueError(f"{dataset} requires at least {test_end} rows; received {n_rows}")
    return {
        "train": (0, train_end),
        "val": (train_end, validation_end),
        "test": (validation_end, test_end),
    }


def fit_train_scaler(
    values: np.ndarray,
    train_end: int,
    scaler: Optional[StandardScalerNP] = None,
) -> StandardScalerNP:
    """Fit *scaler* only to the formal training interval."""

    if not 0 < train_end <= len(values):
        raise ValueError(f"train_end must be in [1, {len(values)}], received {train_end}")
    return (scaler or StandardScalerNP()).fit(values[:train_end])


class GeneralForecastProtocol:
    """Canonical absolute window construction for one formal dataset."""

    def __init__(self, dataset: str, n_rows: int, input_len: int = FORMAL_HISTORY):
        self.input_len = int(input_len)
        self.bounds = split_bounds(dataset, n_rows, self.input_len)

    def window_index(self, split: str, pred_len: int) -> np.ndarray:
        """Return absolute history starts whose targets are inside *split*."""

        if split not in self.bounds:
            raise ValueError(f"unknown split: {split}")
        if pred_len <= 0:
            raise ValueError("pred_len must be positive")
        split_start, split_end = self.bounds[split]
        first_target_start = max(split_start, self.input_len)
        first_history_start = first_target_start - self.input_len
        last_history_start = split_end - pred_len - self.input_len
        if last_history_start < first_history_start:
            return np.empty(0, dtype=np.int64)
        return np.arange(first_history_start, last_history_start + 1, dtype=np.int64)
