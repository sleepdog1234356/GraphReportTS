from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

try:
    from .experiment_config import GENERAL_DATASET_NOTES
    from .general_data_schema import dataset_schema
    from .general_prompting import build_general_prompt_result
    from .general_protocol import FORMAL_HISTORY, GeneralForecastProtocol, StandardScalerNP, fit_train_scaler
    from .raw_signal import aggregate_variable_maps, build_variable_maps
except ImportError:
    from experiment_config import GENERAL_DATASET_NOTES
    from general_data_schema import dataset_schema
    from general_prompting import build_general_prompt_result
    from general_protocol import FORMAL_HISTORY, GeneralForecastProtocol, StandardScalerNP, fit_train_scaler
    from raw_signal import aggregate_variable_maps, build_variable_maps


TIMECMA_DATASETS = ["ETTm1", "ETTm2", "ETTh1", "ETTh2", "ECL", "Weather"]


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
        input_len: int = FORMAL_HISTORY,
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
        self.frequency = dataset_schema(dataset_name).frequency
        path = _find_csv(self.data_root, dataset_name)
        self.source_path = path.resolve()
        df = pd.read_csv(path)
        timestamp_column = next((column for column in ("date", "timestamp") if column in df.columns), None)
        numeric = df.drop(columns=[timestamp_column], errors="ignore").select_dtypes(include=[np.number]).copy()
        if numeric.empty:
            raise ValueError(f"{path} has no numeric columns.")
        self.columns = tuple(str(column) for column in numeric.columns)
        self.timestamps = (
            pd.to_datetime(df[timestamp_column], errors="raise").to_numpy()
            if timestamp_column is not None
            else np.arange(len(df), dtype=np.int64)
        )
        self.raw_values = numeric.to_numpy(dtype=np.float32)
        self.protocol = GeneralForecastProtocol(dataset_name, len(self.raw_values), self.input_len)
        self.split_bounds = self.protocol.bounds
        self.train_end = self.split_bounds["train"][1]
        self.scaler = scaler or StandardScalerNP()
        if fit_scaler:
            fit_train_scaler(self.raw_values, self.train_end, scaler=self.scaler)
        self.values = self.scaler.transform(self.raw_values)
        self.target_idx = self.columns.index(target_col) if target_col in self.columns else None
        self.samples = self.protocol.window_index(split, self.pred_len).tolist()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, object]:
        start = int(self.samples[idx])
        history_end = start + self.input_len
        target_end = history_end + self.pred_len
        x = self.values[start:history_end]
        y = self.values[history_end:target_end]
        maps = aggregate_variable_maps(
            build_variable_maps(
                x,
                resample_len=self.resample_len,
                delay_dim=self.delay_dim,
                delay_lag=self.delay_lag,
                include_derivatives=self.include_derivatives,
                include_hankel=self.include_hankel,
            )
        )
        prompt_result = build_general_prompt_result(x, self.columns, self.frequency, self.pred_len)
        return {
            "maps": torch.tensor(maps, dtype=torch.float32),
            "y": torch.tensor(y, dtype=torch.float32),
            "mask": torch.ones(self.pred_len, y.shape[-1], dtype=torch.bool),
            "horizon": torch.tensor(self.pred_len, dtype=torch.long),
            "prompt": prompt_result.prompt,
            "prompt_metadata": dict(prompt_result.metadata),
            "series_id": self.dataset_name,
            "start_index": start,
            "history_steps": torch.arange(start, history_end),
            "target_steps": torch.arange(history_end, target_end),
            "history_raw": torch.tensor(self.raw_values[start:history_end], dtype=torch.float32),
            "target_raw": torch.tensor(self.raw_values[history_end:target_end], dtype=torch.float32),
            "history_scaled": torch.tensor(x, dtype=torch.float32),
            "target_scaled": torch.tensor(y, dtype=torch.float32),
            "timestamp_markers": {
                "history": tuple(self.timestamps[start:history_end]),
                "target": tuple(self.timestamps[history_end:target_end]),
            },
            "columns": self.columns,
            "scaler_metadata": {
                "mean": self.scaler.mean.copy() if self.scaler.mean is not None else None,
                "std": self.scaler.std.copy() if self.scaler.std is not None else None,
                "train_end": self.train_end,
            },
        }


def collate_general_graph_batch(batch: List[Dict[str, object]]) -> Dict[str, object]:
    max_variables = max(b["y"].shape[-1] for b in batch)
    max_channels = max(b["maps"].shape[0] for b in batch)
    max_hmap = max(b["maps"].shape[1] for b in batch)
    max_wmap = max(b["maps"].shape[2] for b in batch)
    max_horizon = max(b["y"].shape[0] for b in batch)
    history_len = max(
        int(b.get("history_scaled", torch.empty(0, 0)).shape[0])
        for b in batch
    )
    maps = torch.zeros(len(batch), max_channels, max_hmap, max_wmap, dtype=torch.float32)
    histories = torch.zeros(len(batch), history_len, max_variables, dtype=torch.float32)
    targets = torch.zeros(len(batch), max_horizon, max_variables, dtype=torch.float32)
    target_mask = torch.zeros(len(batch), max_horizon, max_variables, dtype=torch.bool)
    variable_mask = torch.zeros(len(batch), max_variables, dtype=torch.bool)
    for i, b in enumerate(batch):
        channels, hm, wm = b["maps"].shape
        variables = b["y"].shape[-1]
        horizon = b["y"].shape[0]
        maps[i, :channels, :hm, :wm] = b["maps"]
        targets[i, :horizon, :variables] = b["y"]
        target_mask[i, :horizon, :variables] = b["mask"]
        if "history_scaled" in b:
            history = b["history_scaled"]
            histories[i, : history.shape[0], :variables] = history
        variable_mask[i, :variables] = True
    return {
        "maps": maps,
        "history_scaled": histories,
        "variable_mask": variable_mask,
        "y": targets,
        "mask": target_mask,
        "horizon": torch.stack([b["horizon"] for b in batch], dim=0),
        "prompt": [b["prompt"] for b in batch],
        "prompt_metadata": [b.get("prompt_metadata") for b in batch],
        "series_id": [b["series_id"] for b in batch],
        "start_index": torch.tensor([b["start_index"] for b in batch], dtype=torch.long),
        "target_steps": torch.stack([b["target_steps"] for b in batch], dim=0),
    }
