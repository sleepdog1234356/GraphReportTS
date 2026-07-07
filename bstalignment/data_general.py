from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

try:
    from .experiment_config import GENERAL_DATASET_NOTES
    from .raw_signal import build_multiview_maps, build_report_from_array
except ImportError:
    from experiment_config import GENERAL_DATASET_NOTES
    from raw_signal import build_multiview_maps, build_report_from_array


TIMECMA_DATASETS = ["ETTm1", "ETTm2", "ETTh1", "ETTh2", "ECL", "FRED", "ILI", "Weather"]


def _find_csv(data_root: Path, dataset_name: str) -> Path:
    candidates = [
        data_root / "processed" / "general" / dataset_name / f"{dataset_name}.csv",
        data_root / "raw" / "general" / dataset_name / f"{dataset_name}.csv",
        data_root / "general" / dataset_name / f"{dataset_name}.csv",
        data_root / f"{dataset_name}.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    note = GENERAL_DATASET_NOTES.get(dataset_name, {})
    raise FileNotFoundError(
        f"No CSV found for {dataset_name}. Put raw data under {note.get('raw_dir')} "
        f"or processed CSV under {note.get('processed_dir')}. Expected numeric columns "
        "and an optional timestamp/date column."
    )


def _timecma_split_indices(n: int, dataset_name: str, input_len: int, pred_len: int) -> Dict[str, Tuple[int, int]]:
    # Standard long-term forecasting splits are commonly 7:1:2. For ETT, the
    # official scripts use date-based split; this ratio fallback keeps the
    # adapter usable until the exact TimeCMA preprocessing files are added.
    n_train = int(n * 0.7)
    n_val = int(n * 0.1)
    return {
        "train": (0, max(n_train, input_len + pred_len)),
        "val": (max(n_train - input_len, 0), max(n_train + n_val, input_len + pred_len)),
        "test": (max(n_train + n_val - input_len, 0), n),
        "all": (0, n),
    }


class StandardScalerNP:
    def __init__(self):
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray) -> "StandardScalerNP":
        self.mean = np.nanmean(x, axis=0)
        self.std = np.nanstd(x, axis=0)
        self.std[self.std < 1e-6] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            return x.astype(np.float32)
        return ((x - self.mean) / self.std).astype(np.float32)


class GeneralForecastGraphDataset(Dataset):
    """TimeCMA-aligned general forecasting dataset adapter.

    Expected CSV after preprocessing:
      date/timestamp column optional, remaining columns numeric.
    Use the same raw datasets and split settings as TimeCMA when reproducing
    paper-level comparisons.
    """

    def __init__(
        self,
        dataset_name: str,
        data_root: str = "bstalignment/data",
        split: str = "train",
        input_len: int = 96,
        pred_len: int = 96,
        resample_len: int = 128,
        delay_dim: int = 8,
        delay_lag: int = 1,
        include_derivatives: bool = True,
        include_hankel: bool = True,
        target_col: Optional[str] = None,
        scaler: Optional[StandardScalerNP] = None,
        fit_scaler: bool = False,
    ):
        self.dataset_name = dataset_name
        self.data_root = Path(data_root)
        self.split = split
        self.input_len = int(input_len)
        self.pred_len = int(pred_len)
        self.resample_len = int(resample_len)
        self.delay_dim = int(delay_dim)
        self.delay_lag = int(delay_lag)
        self.include_derivatives = bool(include_derivatives)
        self.include_hankel = bool(include_hankel)
        path = _find_csv(self.data_root, dataset_name)
        df = pd.read_csv(path)
        numeric = df.select_dtypes(include=[np.number]).copy()
        if numeric.empty:
            raise ValueError(f"{path} has no numeric columns.")
        self.columns = list(numeric.columns)
        values = numeric.to_numpy(dtype=np.float32)
        split_bounds = _timecma_split_indices(len(values), dataset_name, self.input_len, self.pred_len)
        train_start, train_end = split_bounds["train"]
        self.scaler = scaler or StandardScalerNP()
        if fit_scaler:
            self.scaler.fit(values[train_start:train_end])
        values = self.scaler.transform(values)
        start, end = split_bounds[split]
        self.values = values[start:end]
        self.offset = start
        self.target_idx = self.columns.index(target_col) if target_col in self.columns else None
        self.samples = list(range(0, max(len(self.values) - self.input_len - self.pred_len + 1, 0)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        start = int(self.samples[idx])
        x = self.values[start : start + self.input_len]
        y = self.values[start + self.input_len : start + self.input_len + self.pred_len]
        channels = {name: x[:, i] for i, name in enumerate(self.columns)}
        maps, _ = build_multiview_maps(
            channels,
            resample_len=self.resample_len,
            delay_dim=self.delay_dim,
            delay_lag=self.delay_lag,
            include_derivatives=self.include_derivatives,
            include_hankel=self.include_hankel,
            include_ic_dv=False,
        )
        prompt = build_report_from_array(x, domain=self.dataset_name, horizon=self.pred_len, variables=self.columns)
        return {
            "maps": torch.tensor(maps, dtype=torch.float32),
            "y": torch.tensor(y, dtype=torch.float32),
            "mask": torch.ones(self.pred_len, y.shape[-1], dtype=torch.bool),
            "horizon": torch.tensor(self.pred_len, dtype=torch.long),
            "prompt": prompt,
            "series_id": self.dataset_name,
            "start_index": self.offset + start,
            "target_steps": torch.arange(self.offset + start + self.input_len, self.offset + start + self.input_len + self.pred_len),
        }


def collate_general_graph_batch(batch: List[Dict[str, object]]) -> Dict[str, object]:
    max_c = max(b["maps"].shape[0] for b in batch)
    max_hmap = max(b["maps"].shape[1] for b in batch)
    max_wmap = max(b["maps"].shape[2] for b in batch)
    maps = torch.zeros(len(batch), max_c, max_hmap, max_wmap, dtype=torch.float32)
    for i, b in enumerate(batch):
        c, hm, wm = b["maps"].shape
        maps[i, :c, :hm, :wm] = b["maps"]
    return {
        "maps": maps,
        "y": torch.stack([b["y"] for b in batch], dim=0),
        "mask": torch.stack([b["mask"] for b in batch], dim=0),
        "horizon": torch.stack([b["horizon"] for b in batch], dim=0),
        "prompt": [b["prompt"] for b in batch],
        "series_id": [b["series_id"] for b in batch],
        "start_index": torch.tensor([b["start_index"] for b in batch], dtype=torch.long),
        "target_steps": torch.stack([b["target_steps"] for b in batch], dim=0),
    }
