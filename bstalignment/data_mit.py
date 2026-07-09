from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

try:
    from .prompting import build_battery_prompt, build_cycle_prompt, aging_stage_from_soh, aging_stage_name
except ImportError:
    from prompting import build_battery_prompt, build_cycle_prompt, aging_stage_from_soh, aging_stage_name


DEFAULT_FEATURES = [
    "cycle_norm",
    "QD",
    "QC",
    "IR",
    "Tmax",
    "Tavg",
    "Tmin",
    "chargetime",
    "dQd_cycle",
    "QD_roll5",
]

CYCLE_FEATURES = [
    "cycle_norm",
    "QD",
    "QC",
    "IR",
    "Tmax",
    "Tavg",
    "Tmin",
    "chargetime",
    "dQd_cycle",
    "QD_roll5",
    "QD_slope5",
    "IR_roll5",
    "IR_slope5",
    "chargetime_roll5",
]


def _past_only_fill(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing cycle-level values without using later cycles."""
    return df.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)


@dataclass
class CellRecord:
    cell_id: str
    charge_policy: str
    cycle_life: float
    summary: pd.DataFrame
    raw: Dict[str, Any]


class StandardScalerTorch:
    """Tiny numpy/torch-compatible standardizer for sequence features."""

    def __init__(self, mean: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None):
        self.mean = mean
        self.std = std

    def fit(self, arrays: Sequence[np.ndarray]) -> "StandardScalerTorch":
        x = np.concatenate([a.reshape(-1, a.shape[-1]) for a in arrays], axis=0)
        self.mean = np.nanmean(x, axis=0)
        self.std = np.nanstd(x, axis=0)
        self.std[self.std < 1e-8] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            return x.astype(np.float32)
        return ((x - self.mean) / self.std).astype(np.float32)

    def state_dict(self) -> Dict[str, Any]:
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_state_dict(cls, state: Dict[str, Any]) -> "StandardScalerTorch":
        return cls(mean=np.asarray(state["mean"]), std=np.asarray(state["std"]))


def _safe_get(d: Dict[str, Any], keys: Iterable[str], default=None):
    for k in keys:
        if isinstance(d, dict) and k in d:
            return d[k]
    return default


def _to_1d_array(x: Any, dtype=float) -> np.ndarray:
    if x is None:
        return np.array([], dtype=dtype)
    arr = np.asarray(x, dtype=dtype)
    return arr.reshape(-1)


def _summary_to_frame(summary: Dict[str, Any]) -> pd.DataFrame:
    # MIT pkl files normally use keys like QD, QC, IR, Tmax, Tavg, Tmin, chargetime, cycle.
    candidates = {
        "cycle": ["cycle", "cycles", "Cycle"],
        "QD": ["QD", "QDischarge", "discharge_capacity", "Qd"],
        "QC": ["QC", "QCharge", "charge_capacity", "Qc"],
        "IR": ["IR", "internal_resistance", "Resistance"],
        "Tmax": ["Tmax", "T_max", "max_temperature"],
        "Tavg": ["Tavg", "T_avg", "avg_temperature"],
        "Tmin": ["Tmin", "T_min", "min_temperature"],
        "chargetime": ["chargetime", "charge_time", "ChargeTime"],
    }
    data = {}
    max_len = 0
    for out_key, possible in candidates.items():
        arr = _to_1d_array(_safe_get(summary, possible, None))
        data[out_key] = arr
        max_len = max(max_len, len(arr))
    if max_len == 0:
        raise ValueError("Cannot parse summary group: no known vector keys were found.")
    for k, arr in list(data.items()):
        if len(arr) == 0:
            data[k] = np.full(max_len, np.nan)
        elif len(arr) != max_len:
            # Pad or truncate conservatively.
            tmp = np.full(max_len, np.nan)
            tmp[: min(max_len, len(arr))] = arr[: min(max_len, len(arr))]
            data[k] = tmp
    df = pd.DataFrame(data)
    if df["cycle"].isna().all():
        df["cycle"] = np.arange(1, len(df) + 1)
    df = df.sort_values("cycle").reset_index(drop=True)
    return _past_only_fill(df)


def load_mit_battery_pkls(data_dir: str | Path, filenames: Optional[List[str]] = None) -> List[CellRecord]:
    """Load MIT/Stanford/TRI battery pkl files.

    Expected files are usually batch1.pkl/batch2.pkl/batch3.pkl. This loader is intentionally
    tolerant to minor key variations in community mirrors.
    """
    data_dir = Path(data_dir)
    if filenames is None:
        filenames = [p.name for p in sorted(data_dir.glob("*.pkl"))]
    if not filenames:
        raise FileNotFoundError(
            f"No .pkl files found under {data_dir}. Download batch pkl files from matr.io first."
        )

    records: List[CellRecord] = []
    for fname in filenames:
        path = data_dir / fname
        with open(path, "rb") as f:
            try:
                batch = pickle.load(f)
            except UnicodeDecodeError:
                f.seek(0)
                batch = pickle.load(f, encoding="latin1")
        if not isinstance(batch, dict):
            raise ValueError(f"{path} did not load to a dictionary.")
        for cell_id, cell in batch.items():
            if not isinstance(cell, dict):
                continue
            summary = _safe_get(cell, ["summary", "Summary"], None)
            if summary is None:
                continue
            try:
                df = _summary_to_frame(summary)
            except Exception:
                continue
            charge_policy = str(_safe_get(cell, ["charge_policy", "policy", "chargePolicy"], "unknown"))
            cycle_life = float(_safe_get(cell, ["cycle_life", "cyclelife", "life"], len(df)))
            records.append(
                CellRecord(
                    cell_id=f"{Path(fname).stem}_{cell_id}",
                    charge_policy=charge_policy,
                    cycle_life=cycle_life,
                    summary=df,
                    raw=cell,
                )
            )
    if not records:
        raise RuntimeError("No cells could be parsed from pkl files. Check file format.")
    return records


def add_cycle_features(df: pd.DataFrame) -> pd.DataFrame:
    out = _past_only_fill(df.copy())
    # MIT cells are A123 LFP/graphite nominally around 1.1Ah; for SOH, initial observed capacity is safer.
    qd = out["QD"].astype(float).to_numpy()
    valid = qd[np.isfinite(qd)]
    if len(valid) == 0:
        init_q = 1.1
    else:
        init_q = float(np.nanmedian(valid[: min(10, len(valid))]))
        if init_q <= 0:
            init_q = float(np.nanmax(valid))
    out["SOH"] = out["QD"] / max(init_q, 1e-6)
    out["cycle_norm"] = out["cycle"] / max(float(out["cycle"].max()), 1.0)
    out["dQd_cycle"] = out["QD"].diff().fillna(0.0)
    out["QD_roll5"] = out["QD"].rolling(5, min_periods=1).mean()
    out["IR_roll5"] = out["IR"].rolling(5, min_periods=1).mean()
    out["chargetime_roll5"] = out["chargetime"].rolling(5, min_periods=1).mean()
    out["QD_slope5"] = _rolling_slope(out["QD"], window=5)
    out["IR_slope5"] = _rolling_slope(out["IR"], window=5)
    out["aging_stage"] = out["SOH"].map(aging_stage_from_soh).astype(int)
    return _past_only_fill(out)


def _rolling_slope(series: pd.Series, window: int = 5) -> pd.Series:
    values = series.astype(float)

    def slope(x):
        if len(x) < 2:
            return 0.0
        idx = np.arange(len(x), dtype=float)
        try:
            return float(np.polyfit(idx, np.asarray(x, dtype=float), 1)[0])
        except Exception:
            return 0.0

    return values.rolling(window, min_periods=2).apply(slope, raw=False).fillna(0.0)


def split_cells(records: Sequence[CellRecord], train_ratio=0.7, val_ratio=0.15, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(records))
    rng.shuffle(idx)
    n_train = int(len(idx) * train_ratio)
    n_val = int(len(idx) * val_ratio)
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]
    return ([records[i] for i in train_idx], [records[i] for i in val_idx], [records[i] for i in test_idx])


class MITBatterySOHDataset(Dataset):
    """Cycle-level sliding-window SOH dataset.

    Each sample:
      x: [seq_len, num_features]
      y: [forecast_horizon] SOH values for cycles t+1 ... t+H
      prompt: dynamic battery prompt containing forecast_horizon
      aging_stage: 0/1/2 for early/middle/late aging at the last forecasted cycle
    """

    def __init__(
        self,
        records: Sequence[CellRecord],
        seq_len: int = 20,
        forecast_horizon: int = 1,
        pred_horizon: Optional[int] = None,
        features: Optional[List[str]] = None,
        scaler: Optional[StandardScalerTorch] = None,
        fit_scaler: bool = False,
        max_cycles: Optional[int] = None,
    ):
        self.records = list(records)
        self.seq_len = seq_len
        # Backward compatibility: if old code passes pred_horizon, treat it as forecast_horizon.
        self.forecast_horizon = int(pred_horizon if pred_horizon is not None else forecast_horizon)
        self.features = features or DEFAULT_FEATURES
        self.cells: Dict[str, Tuple[CellRecord, pd.DataFrame]] = {}
        self.samples: List[Dict[str, Any]] = []

        arrays_for_scaler = []
        for rec in self.records:
            df = add_cycle_features(rec.summary)
            if max_cycles is not None:
                df = df.iloc[:max_cycles].copy()
            if len(df) < seq_len + self.forecast_horizon:
                continue
            self.cells[rec.cell_id] = (rec, df)
            feat = df[self.features].to_numpy(dtype=np.float32)
            arrays_for_scaler.append(feat)

        self.scaler = scaler or StandardScalerTorch()
        if fit_scaler:
            self.scaler.fit(arrays_for_scaler)

        for rec, df in self.cells.values():
            # End index is the final observed cycle in the input window.
            # Targets are the next H cycles: end_idx+1 ... end_idx+H.
            for end_idx in range(seq_len - 1, len(df) - self.forecast_horizon):
                target_start_idx = end_idx + 1
                target_end_idx = end_idx + self.forecast_horizon
                target_slice = df.iloc[target_start_idx : target_end_idx + 1]
                x_raw = df.iloc[end_idx - seq_len + 1 : end_idx + 1][self.features].to_numpy(dtype=np.float32)
                y = target_slice["SOH"].to_numpy(dtype=np.float32)
                aging_stage = int(target_slice.iloc[-1]["aging_stage"])
                target_cycles = target_slice["cycle"].to_numpy(dtype=np.int64)
                hist = df.iloc[max(0, end_idx - seq_len + 1) : end_idx + 1]
                prompt = build_battery_prompt(
                    cell_id=rec.cell_id,
                    charge_policy=rec.charge_policy,
                    cycle_life=rec.cycle_life,
                    hist=hist,
                    target_cycle_start=int(target_cycles[0]),
                    target_cycle_end=int(target_cycles[-1]),
                    forecast_horizon=self.forecast_horizon,
                    chemistry="LFP/graphite",
                    nominal_capacity_ah=1.1,
                    ambient_temperature_c=30.0,
                )
                self.samples.append(
                    {
                        "cell_id": rec.cell_id,
                        "end_cycle": int(df.iloc[end_idx]["cycle"]),
                        "target_cycle": int(target_cycles[-1]),
                        "target_cycles": target_cycles,
                        "x": x_raw,
                        "y": y,
                        "aging_stage": aging_stage,
                        "prompt": prompt,
                        "forecast_horizon": self.forecast_horizon,
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        x = self.scaler.transform(s["x"])
        return {
            "x": torch.tensor(x, dtype=torch.float32),
            "y": torch.tensor(s["y"], dtype=torch.float32),
            "aging_stage": torch.tensor(s["aging_stage"], dtype=torch.long),
            "prompt": s["prompt"],
            "cell_id": s["cell_id"],
            "end_cycle": s["end_cycle"],
            "target_cycle": s["target_cycle"],
            "target_cycles": torch.tensor(s["target_cycles"], dtype=torch.long),
            "forecast_horizon": s["forecast_horizon"],
        }


def collate_battery_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "x": torch.stack([b["x"] for b in batch], dim=0),
        "y": torch.stack([b["y"] for b in batch], dim=0),
        "aging_stage": torch.stack([b["aging_stage"] for b in batch], dim=0),
        "prompt": [b["prompt"] for b in batch],
        "cell_id": [b["cell_id"] for b in batch],
        "end_cycle": torch.tensor([b["end_cycle"] for b in batch], dtype=torch.long),
        "target_cycle": torch.tensor([b["target_cycle"] for b in batch], dtype=torch.long),
        "target_cycles": torch.stack([b["target_cycles"] for b in batch], dim=0),
        "forecast_horizon": torch.tensor([b["forecast_horizon"] for b in batch], dtype=torch.long),
    }


class MITBatteryCycleDataset(Dataset):
    """One-cycle SOH estimation plus variable-horizon forecasting dataset.

    Each sample uses only the current cycle and historical rolling features:
      x: [num_features]
      y_now: scalar SOH for the current cycle
      y_future: [H] SOH for cycles t+1 ... t+H
      prompt: compact leakage-free cycle report containing H
    """

    def __init__(
        self,
        records: Sequence[CellRecord],
        max_horizon: int = 20,
        min_history: int = 5,
        features: Optional[List[str]] = None,
        scaler: Optional[StandardScalerTorch] = None,
        fit_scaler: bool = False,
        random_horizon: bool = True,
        max_cycles: Optional[int] = None,
        seed: int = 42,
    ):
        self.records = list(records)
        self.max_horizon = int(max_horizon)
        self.min_history = int(min_history)
        self.features = features or CYCLE_FEATURES
        self.random_horizon = bool(random_horizon)
        self.rng = np.random.default_rng(seed)
        self.cells: Dict[str, Tuple[CellRecord, pd.DataFrame]] = {}
        self.samples: List[Dict[str, Any]] = []

        arrays_for_scaler = []
        for rec in self.records:
            df = add_cycle_features(rec.summary)
            if max_cycles is not None:
                df = df.iloc[:max_cycles].copy()
            if len(df) <= self.min_history:
                continue
            self.cells[rec.cell_id] = (rec, df)
            arrays_for_scaler.append(df[self.features].to_numpy(dtype=np.float32))

        self.scaler = scaler or StandardScalerTorch()
        if fit_scaler:
            if not arrays_for_scaler:
                raise RuntimeError("No arrays available to fit scaler. Check data and max_horizon/min_history.")
            self.scaler.fit(arrays_for_scaler)

        for rec, df in self.cells.values():
            last_start = len(df) - 2
            for idx in range(self.min_history - 1, last_start + 1):
                available_h = min(self.max_horizon, len(df) - idx - 1)
                if available_h <= 0:
                    continue
                self.samples.append(
                    {
                        "cell_id": rec.cell_id,
                        "idx": idx,
                        "available_horizon": available_h,
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def _choose_horizon(self, available_horizon: int) -> int:
        if not self.random_horizon:
            return int(available_horizon)
        return int(self.rng.integers(1, available_horizon + 1))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        rec, df = self.cells[s["cell_id"]]
        row_idx = int(s["idx"])
        horizon = self._choose_horizon(int(s["available_horizon"]))
        row = df.iloc[row_idx]
        future = df.iloc[row_idx + 1 : row_idx + 1 + horizon]
        x_raw = row[self.features].to_numpy(dtype=np.float32)
        x = self.scaler.transform(x_raw.reshape(1, -1)).reshape(-1)
        trend_label = _trend_label_from_slope(float(row.get("QD_slope5", 0.0)))
        prompt = build_cycle_prompt(
            cycle=int(row.get("cycle", row_idx + 1)),
            forecast_horizon=horizon,
            charge_policy=rec.charge_policy,
            qd=float(row.get("QD", 0.0)),
            qc=float(row.get("QC", 0.0)),
            ir=float(row.get("IR", 0.0)),
            tmax=float(row.get("Tmax", 0.0)),
            tavg=float(row.get("Tavg", 0.0)),
            chargetime=float(row.get("chargetime", 0.0)),
            qd_roll5=float(row.get("QD_roll5", 0.0)),
            dqd_cycle=float(row.get("dQd_cycle", 0.0)),
            qd_slope5=float(row.get("QD_slope5", 0.0)),
            ir_slope5=float(row.get("IR_slope5", 0.0)),
            trend_label=trend_label,
            observed_stage=aging_stage_name(int(row.get("aging_stage", 0))),
        )
        return {
            "x": torch.tensor(x, dtype=torch.float32),
            "y_now": torch.tensor(float(row["SOH"]), dtype=torch.float32),
            "y_future": torch.tensor(future["SOH"].to_numpy(dtype=np.float32), dtype=torch.float32),
            "future_mask": torch.ones(horizon, dtype=torch.bool),
            "horizon": torch.tensor(horizon, dtype=torch.long),
            "prompt": prompt,
            "cell_id": rec.cell_id,
            "cycle": int(row.get("cycle", row_idx + 1)),
            "future_cycles": torch.tensor(future["cycle"].to_numpy(dtype=np.int64), dtype=torch.long),
        }


def _trend_label_from_slope(qd_slope: float) -> str:
    if qd_slope < -2e-3:
        return "fast_decrease"
    if qd_slope < -3e-4:
        return "slow_decrease"
    if qd_slope > 3e-4:
        return "local_rebound"
    return "stable"


def collate_cycle_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    max_h = max(int(b["horizon"]) for b in batch)
    y_future = torch.zeros(len(batch), max_h, dtype=torch.float32)
    future_mask = torch.zeros(len(batch), max_h, dtype=torch.bool)
    future_cycles = torch.zeros(len(batch), max_h, dtype=torch.long)
    for i, b in enumerate(batch):
        h = int(b["horizon"])
        y_future[i, :h] = b["y_future"]
        future_mask[i, :h] = True
        future_cycles[i, :h] = b["future_cycles"]
    return {
        "x": torch.stack([b["x"] for b in batch], dim=0),
        "y_now": torch.stack([b["y_now"] for b in batch], dim=0),
        "y_future": y_future,
        "future_mask": future_mask,
        "horizon": torch.stack([b["horizon"] for b in batch], dim=0),
        "prompt": [b["prompt"] for b in batch],
        "cell_id": [b["cell_id"] for b in batch],
        "cycle": torch.tensor([b["cycle"] for b in batch], dtype=torch.long),
        "future_cycles": future_cycles,
    }
