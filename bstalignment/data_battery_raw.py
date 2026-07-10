from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

try:
    from .battery_protocol import (
        BATTERY_CYCLE_SCALE_PROTOCOL,
        BATTERY_TARGET_PROTOCOL,
        fit_cycle_scale,
        fit_processed_cycle_scale,
        split_processed_items,
    )
    from .data_mit import CellRecord, add_cycle_features, load_mit_battery_pkls, split_cells
    from .experiment_config import BATTERY_DATASET_NOTES
    from .raw_signal import build_multiview_maps, build_report_from_array, current_to_capacity
except ImportError:
    from battery_protocol import (
        BATTERY_CYCLE_SCALE_PROTOCOL,
        BATTERY_TARGET_PROTOCOL,
        fit_cycle_scale,
        fit_processed_cycle_scale,
        split_processed_items,
    )
    from data_mit import CellRecord, add_cycle_features, load_mit_battery_pkls, split_cells
    from experiment_config import BATTERY_DATASET_NOTES
    from raw_signal import build_multiview_maps, build_report_from_array, current_to_capacity


RAW_BATTERY_CHANNELS = ["current", "voltage", "temperature", "capacity"]
BATTERY_GRAPH_CACHE_VERSION = 5
BATTERY_HISTORY_FEATURE_SCHEMA = [
    "capacity_value",
    "capacity_zscore",
    "internal_resistance_zscore",
    "charge_time_zscore",
    "cycle_ratio",
    "capacity_delta",
    "internal_resistance_delta",
    "charge_time_delta",
]
BATTERY_HISTORY_FEATURE_DIM = len(BATTERY_HISTORY_FEATURE_SCHEMA)


def battery_graph_cache_config(
    dataset_name: str,
    split: str,
    max_horizon: int,
    resample_len: int,
    delay_dim: int,
    delay_lag: int,
    include_derivatives: bool,
    include_hankel: bool,
    include_ic_dv: bool,
    allow_summary_fallback: bool,
    seed: int,
    max_cycles: Optional[int],
    history_len: int = 32,
) -> Dict[str, Any]:
    return {
        "version": BATTERY_GRAPH_CACHE_VERSION,
        "dataset": dataset_name.lower(),
        "split": split,
        "max_horizon": int(max_horizon),
        "resample_len": int(resample_len),
        "delay_dim": int(delay_dim),
        "delay_lag": int(delay_lag),
        "include_derivatives": bool(include_derivatives),
        "include_hankel": bool(include_hankel),
        "include_ic_dv": bool(include_ic_dv),
        "allow_summary_fallback": bool(allow_summary_fallback),
        "seed": int(seed),
        "max_cycles": None if max_cycles is None else int(max_cycles),
        "history_len": int(history_len),
        "history_feature_schema": list(BATTERY_HISTORY_FEATURE_SCHEMA),
        "target_protocol": BATTERY_TARGET_PROTOCOL,
        "cycle_scale_protocol": BATTERY_CYCLE_SCALE_PROTOCOL,
    }


def battery_graph_cache_hash(config: Dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]


def battery_graph_cache_path(cache_root: str | Path, config: Dict[str, Any]) -> Path:
    return Path(cache_root) / str(config["dataset"]) / str(config["split"]) / battery_graph_cache_hash(config)


def _safe_array(obj: Any) -> np.ndarray:
    if obj is None:
        return np.array([], dtype=np.float32)
    try:
        return np.asarray(obj, dtype=np.float32).reshape(-1)
    except Exception:
        return np.array([], dtype=np.float32)


def _lookup(d: Dict[str, Any], names: Sequence[str]) -> Any:
    if not isinstance(d, dict):
        return None
    for name in names:
        if name in d:
            return d[name]
    return None


def _extract_mit_cycle_channels(rec: CellRecord, cycle_id: int) -> Dict[str, np.ndarray]:
    """Best-effort raw cycle parser for common MIT pkl mirrors.

    Formal GraphReportTS experiments should use true raw cycle arrays. A
    separate summary fallback exists only for smoke tests.
    """
    cycle_group = _lookup(rec.raw, ["cycle", "cycles", "Cycle"])
    cyc = None
    if isinstance(cycle_group, dict):
        candidates = [str(cycle_id), f"cycle_{cycle_id}", cycle_id]
        for key in candidates:
            if key in cycle_group:
                cyc = cycle_group[key]
                break
        if cyc is None:
            keys = list(cycle_group.keys())
            if 0 <= cycle_id - 1 < len(keys):
                cyc = cycle_group[keys[cycle_id - 1]]
    channels = {}
    if isinstance(cyc, dict):
        time = _safe_array(_lookup(cyc, ["t", "time", "Time"]))
        current = _safe_array(_lookup(cyc, ["I", "current", "Current"]))
        voltage = _safe_array(_lookup(cyc, ["V", "voltage", "Voltage"]))
        temperature = _safe_array(_lookup(cyc, ["T", "temperature", "Temperature", "Temp"]))
        if len(current) and len(time):
            capacity = current_to_capacity(time, current)
        else:
            capacity = _safe_array(_lookup(cyc, ["Q", "capacity", "Capacity", "Qdlin"]))
        channels = {
            "current": current,
            "voltage": voltage,
            "temperature": temperature,
            "capacity": capacity,
        }
    return channels


def _summary_pseudo_channels(rec: CellRecord, cycle_id: int) -> Dict[str, np.ndarray]:
    """Smoke-test fallback, never used by default formal experiments."""
    df = add_cycle_features(rec.summary)
    row = df[df["cycle"].astype(int) == int(cycle_id)]
    if len(row) == 0:
        row = df.iloc[[min(max(cycle_id - 1, 0), len(df) - 1)]]
    r = row.iloc[0]
    x = np.linspace(0.0, 1.0, 32, dtype=np.float32)
    return {
        "current": np.full_like(x, float(r.get("QC", 1.0))),
        "voltage": 3.6 + 0.15 * x + 0.01 * float(r.get("QD", 1.0)),
        "temperature": np.full_like(x, float(r.get("Tavg", 30.0))),
        "capacity": x * float(r.get("QD", 1.0)),
    }


def _safe_col(df: pd.DataFrame, name: str, fallback: float = 0.0) -> np.ndarray:
    if name in df:
        return df[name].to_numpy(dtype=np.float32)
    return np.full(len(df), float(fallback), dtype=np.float32)


def _safe_zscore(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    arr = np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    std = float(np.nanstd(arr))
    if std < eps:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - float(np.nanmean(arr))) / std).astype(np.float32)


def _relative_to_first(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    arr = np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    base = float(arr[0]) if len(arr) else 1.0
    if abs(base) < eps:
        return arr.astype(np.float32)
    return (arr / base).astype(np.float32)


def _diff_prepend_zero(x: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if len(arr) == 0:
        return arr
    return np.diff(arr, prepend=arr[0]).astype(np.float32)


def _history_feature_matrix(
    capacity: np.ndarray,
    internal_resistance: np.ndarray,
    charge_time: np.ndarray,
    cycle_ids: np.ndarray,
    max_cycle_id: int,
) -> np.ndarray:
    capacity = np.nan_to_num(np.asarray(capacity, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    internal_resistance = np.nan_to_num(np.asarray(internal_resistance, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    charge_time = np.nan_to_num(np.asarray(charge_time, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    cycle_ids = np.asarray(cycle_ids, dtype=np.float32)
    denom = max(float(max_cycle_id), 1.0)
    feats = np.stack(
        [
            capacity.astype(np.float32),
            _safe_zscore(capacity),
            _safe_zscore(internal_resistance),
            _safe_zscore(charge_time),
            (cycle_ids / denom).astype(np.float32),
            _diff_prepend_zero(capacity),
            _diff_prepend_zero(internal_resistance),
            _diff_prepend_zero(charge_time),
        ],
        axis=-1,
    )
    return np.nan_to_num(feats.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


class BatteryRawGraphDataset(Dataset):
    """Battery SOH dataset for GraphReportTS.

    MIT can run from the current repository. CALCE and XJTU are intentionally
    documented placeholders until raw files are downloaded and preprocessed.

    Expected processed raw format for CALCE/XJTU:
      bstalignment/data/processed/battery/<dataset>/<cell_id>.npz
    with arrays:
      cycle_id [N], soh [N], current [N,L], voltage [N,L],
      temperature [N,L], optional capacity [N,L].
    """

    def __init__(
        self,
        dataset_name: str = "mit",
        data_root: str = "bstalignment/data",
        split: str = "train",
        max_horizon: int = 20,
        resample_len: int = 128,
        delay_dim: int = 8,
        delay_lag: int = 1,
        include_derivatives: bool = True,
        include_hankel: bool = True,
        include_ic_dv: bool = True,
        allow_summary_fallback: bool = False,
        cache_items: bool = False,
        cycle_cache_size: int = 4096,
        precomputed_cache_dir: Optional[str] = None,
        require_precomputed_cache: bool = False,
        seed: int = 42,
        max_cycles: Optional[int] = None,
        history_len: int = 32,
    ):
        self.dataset_name = dataset_name.lower()
        self.data_root = Path(data_root)
        self.split = split
        self.max_horizon = int(max_horizon)
        self.resample_len = int(resample_len)
        self.delay_dim = int(delay_dim)
        self.delay_lag = int(delay_lag)
        self.include_derivatives = bool(include_derivatives)
        self.include_hankel = bool(include_hankel)
        self.include_ic_dv = bool(include_ic_dv)
        self.allow_summary_fallback = bool(allow_summary_fallback)
        self.history_len = max(int(history_len), 1)
        self.cache_items = bool(cache_items)
        self._item_cache: Dict[int, Dict[str, Any]] = {}
        self.cycle_cache_size = max(int(cycle_cache_size), 0)
        self._cycle_map_cache: OrderedDict[Tuple[str, str, int], Tuple[np.ndarray, List[str]]] = OrderedDict()
        self.precomputed_cache_dir = Path(precomputed_cache_dir) if precomputed_cache_dir else None
        self.require_precomputed_cache = bool(require_precomputed_cache)
        self.cycle_scale = 1.0
        self.cache_config = battery_graph_cache_config(
            dataset_name=self.dataset_name,
            split=self.split,
            max_horizon=self.max_horizon,
            resample_len=self.resample_len,
            delay_dim=self.delay_dim,
            delay_lag=self.delay_lag,
            include_derivatives=self.include_derivatives,
            include_hankel=self.include_hankel,
            include_ic_dv=self.include_ic_dv,
            allow_summary_fallback=self.allow_summary_fallback,
            seed=seed,
            max_cycles=max_cycles,
            history_len=self.history_len,
        )
        self._precomputed = False
        self._cache_path: Optional[Path] = None
        self._cache_maps = None
        self._cache_cycle_maps = None
        self._cache_history_indices = None
        self._cache_y = None
        self._cache_mask = None
        self._cache_horizon = None
        self._cache_target_steps = None
        self._cache_history_features = None
        self._cache_history_cycles = None
        self._cache_layout = "sample_history"
        self._cache_meta: List[Dict[str, Any]] = []
        if self.precomputed_cache_dir is not None and self._try_load_precomputed_cache():
            return
        if self.require_precomputed_cache:
            expected = battery_graph_cache_path(self.precomputed_cache_dir or "", self.cache_config)
            raise FileNotFoundError(f"Required battery graph cache not found or invalid: {expected}")
        self.samples: List[Dict[str, Any]] = []
        self.records: List[CellRecord] = []
        self.processed_cells: List[Dict[str, Any]] = []
        if self.dataset_name == "mit":
            self._load_mit(seed=seed, max_cycles=max_cycles)
        else:
            self._load_processed(max_cycles=max_cycles, seed=seed)

    def _try_load_precomputed_cache(self) -> bool:
        if self.precomputed_cache_dir is None:
            return False
        cache_path = battery_graph_cache_path(self.precomputed_cache_dir, self.cache_config)
        manifest_path = cache_path / "manifest.json"
        if not manifest_path.exists():
            return False
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if manifest.get("config") != self.cache_config:
            return False
        self.cycle_scale = float(manifest.get("cycle_scale", 1.0))
        files = manifest.get("files", {})
        self._cache_layout = str(manifest.get("layout", "sample_history"))
        if self._cache_layout == "cycle_history":
            required = ["cycle_maps", "history_indices", "y", "mask", "horizon", "target_steps", "history_features", "history_cycles", "meta"]
        else:
            required = ["maps", "y", "mask", "horizon", "target_steps", "history_features", "history_cycles", "meta"]
        if any(not (cache_path / str(files.get(name, ""))).exists() for name in required):
            return False
        self._cache_path = cache_path
        if self._cache_layout == "cycle_history":
            self._cache_cycle_maps = np.load(cache_path / files["cycle_maps"], mmap_mode="r")
            self._cache_history_indices = np.load(cache_path / files["history_indices"], mmap_mode="r")
        else:
            self._cache_maps = np.load(cache_path / files["maps"], mmap_mode="r")
        self._cache_y = np.load(cache_path / files["y"], mmap_mode="r")
        self._cache_mask = np.load(cache_path / files["mask"], mmap_mode="r")
        self._cache_horizon = np.load(cache_path / files["horizon"], mmap_mode="r")
        self._cache_target_steps = np.load(cache_path / files["target_steps"], mmap_mode="r")
        self._cache_history_features = np.load(cache_path / files["history_features"], mmap_mode="r")
        self._cache_history_cycles = np.load(cache_path / files["history_cycles"], mmap_mode="r")
        with (cache_path / files["meta"]).open("r", encoding="utf-8") as f:
            self._cache_meta = [json.loads(line) for line in f if line.strip()]
        sample_count = int(manifest.get("sample_count", -1))
        if sample_count < 0 or sample_count != len(self._cache_meta) or sample_count != int(self._cache_y.shape[0]):
            return False
        self._precomputed = True
        self.samples = []
        self.records = []
        self.processed_cells = []
        return True

    def _load_mit(self, seed: int, max_cycles: Optional[int]) -> None:
        records = load_mit_battery_pkls(self.data_root / "mit")
        train, val, test = split_cells(records, seed=seed)
        self.cycle_scale = fit_cycle_scale(
            (record.summary["cycle"].to_numpy(dtype=np.float64) for record in train),
            max_cycles,
        )
        selected = {"train": train, "val": val, "test": test, "all": records}[self.split]
        self.records = list(selected)
        for rec_idx, rec in enumerate(self.records):
            df = add_cycle_features(rec.summary)
            if max_cycles is not None:
                df = df.iloc[:max_cycles].copy()
            for row_idx in range(self.history_len - 1, len(df) - self.max_horizon):
                self.samples.append(
                    {
                        "record_idx": rec_idx,
                        "row_idx": row_idx,
                        "cycle_id": int(df.iloc[row_idx]["cycle"]),
                        "horizon": self.max_horizon,
                    }
                )

    def _load_processed(self, max_cycles: Optional[int], seed: int) -> None:
        note = BATTERY_DATASET_NOTES.get(self.dataset_name, {})
        processed_dir = self.data_root / "processed" / "battery" / self.dataset_name
        if not processed_dir.exists() and note.get("processed_dir"):
            processed_dir = Path(note["processed_dir"])
        files = sorted(processed_dir.glob("*.npz"))
        if not files:
            required = "\n  - ".join(note.get("required", []))
            raise FileNotFoundError(
                f"No processed {self.dataset_name.upper()} files found under {processed_dir}.\n"
                f"Place raw data under {note.get('raw_dir')} and preprocess to .npz files with:\n  - {required}"
            )
        splits = split_processed_items(files, seed=seed)
        self.cycle_scale = fit_processed_cycle_scale(splits["train"], max_cycles)
        selected = splits[self.split]
        for cell_idx, path in enumerate(selected):
            data = np.load(path, allow_pickle=True)
            required = ["cycle_id", "soh", "current", "voltage", "temperature"]
            missing = [k for k in required if k not in data]
            if missing:
                raise ValueError(f"{path} is missing required arrays: {missing}")
            cell = {k: data[k] for k in data.files}
            cell["cell_id"] = path.stem
            if "capacity" not in cell:
                current = np.asarray(cell["current"], dtype=np.float32)
                time = np.asarray(cell["time"], dtype=np.float32) if "time" in cell else np.tile(np.arange(current.shape[1]), (current.shape[0], 1))
                cell["capacity"] = np.stack([current_to_capacity(time[i], current[i]) for i in range(current.shape[0])])
            self.processed_cells.append(cell)
            n = len(cell["cycle_id"])
            if max_cycles is not None:
                n = min(n, int(max_cycles))
            for row_idx in range(self.history_len - 1, n - self.max_horizon):
                self.samples.append(
                    {
                        "processed_idx": cell_idx,
                        "row_idx": row_idx,
                        "cycle_id": int(cell["cycle_id"][row_idx]),
                        "horizon": self.max_horizon,
                    }
                )

    def __len__(self) -> int:
        if self._precomputed:
            if self._cache_y is None:
                return 0
            return int(self._cache_y.shape[0])
        return len(self.samples)

    def _cached_cycle_maps(self, key: Tuple[str, str, int], builder: Callable[[], Tuple[np.ndarray, List[str]]]) -> Tuple[np.ndarray, List[str]]:
        if self.cycle_cache_size <= 0:
            return builder()
        if key in self._cycle_map_cache:
            maps, names = self._cycle_map_cache.pop(key)
            self._cycle_map_cache[key] = (maps, names)
            return maps, names
        maps, names = builder()
        self._cycle_map_cache[key] = (maps, names)
        while len(self._cycle_map_cache) > self.cycle_cache_size:
            self._cycle_map_cache.popitem(last=False)
        return maps, names

    def _build_maps_from_channels(self, channels: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[str]]:
        return build_multiview_maps(
            channels,
            resample_len=self.resample_len,
            delay_dim=self.delay_dim,
            delay_lag=self.delay_lag,
            include_derivatives=self.include_derivatives,
            include_hankel=self.include_hankel,
            include_ic_dv=self.include_ic_dv,
        )

    def _mit_cycle_maps(self, rec: CellRecord, cycle_id: int) -> Tuple[np.ndarray, List[str]]:
        def build() -> Tuple[np.ndarray, List[str]]:
            channels = _extract_mit_cycle_channels(rec, cycle_id)
            if not any(len(v) for v in channels.values()):
                if not self.allow_summary_fallback:
                    raise RuntimeError(
                        f"Raw MIT cycle arrays not found for {rec.cell_id} cycle {cycle_id}. "
                        "For formal GraphReportTS experiments, rebuild MIT pkl with raw cycles. "
                        "Use allow_summary_fallback only for smoke tests."
                    )
                channels = _summary_pseudo_channels(rec, cycle_id)
            return self._build_maps_from_channels(channels)

        return self._cached_cycle_maps(("mit", rec.cell_id, int(cycle_id)), build)

    def _processed_cycle_maps(self, cell: Dict[str, Any], row_idx: int) -> Tuple[np.ndarray, List[str]]:
        cycle_id = int(cell["cycle_id"][row_idx])

        def build() -> Tuple[np.ndarray, List[str]]:
            channels = {
                "current": np.asarray(cell["current"][row_idx], dtype=np.float32),
                "voltage": np.asarray(cell["voltage"][row_idx], dtype=np.float32),
                "temperature": np.asarray(cell["temperature"][row_idx], dtype=np.float32),
                "capacity": np.asarray(cell["capacity"][row_idx], dtype=np.float32),
            }
            return self._build_maps_from_channels(channels)

        return self._cached_cycle_maps((str(self.dataset_name), str(cell["cell_id"]), cycle_id), build)

    def _mit_history_features(self, df: pd.DataFrame, start: int, stop: int) -> np.ndarray:
        hist = df.iloc[start:stop]
        return _history_feature_matrix(
            capacity=_safe_col(hist, "QD"),
            internal_resistance=_safe_col(hist, "IR"),
            charge_time=_safe_col(hist, "chargetime"),
            cycle_ids=_safe_col(hist, "cycle"),
            max_cycle_id=self.cycle_scale,
        )

    def _processed_history_features(self, cell: Dict[str, Any], start: int, stop: int, n: int) -> np.ndarray:
        if "capacity_summary" in cell:
            capacity = np.asarray(cell["capacity_summary"][start:stop], dtype=np.float32)
        else:
            capacity_arr = np.asarray(cell["capacity"][start:stop], dtype=np.float32)
            capacity = capacity_arr[:, -1] if capacity_arr.ndim == 2 else capacity_arr.reshape(len(capacity_arr), -1)[:, -1]
        zeros = np.zeros_like(capacity, dtype=np.float32)
        return _history_feature_matrix(
            capacity=capacity,
            internal_resistance=np.asarray(cell.get("internal_resistance", zeros)[start:stop], dtype=np.float32)
            if "internal_resistance" in cell
            else zeros,
            charge_time=np.asarray(cell.get("charge_time", zeros)[start:stop], dtype=np.float32) if "charge_time" in cell else zeros,
            cycle_ids=np.asarray(cell["cycle_id"][start:stop], dtype=np.float32),
            max_cycle_id=self.cycle_scale,
        )

    def _prompt_from_history(
        self,
        summary_array: np.ndarray,
        variables: List[str],
        horizon: int,
        cell_id: str,
        cycle_id: int,
        map_names: List[str],
    ) -> str:
        prompt = build_report_from_array(
            summary_array,
            domain=f"battery-{self.dataset_name}",
            horizon=horizon,
            variables=variables,
        )
        prompt += (
            f" Recent {self.history_len} cycles are provided as direct numeric and raw-map input; "
            f"older history is summarized in this prompt. Battery adapter: cell_id={cell_id}; "
            f"cycle={cycle_id}; channels={', '.join(map_names[:10])}; target=future SOH."
        )
        return prompt

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self._precomputed:
            return self._getitem_precomputed(idx)
        if self.cache_items and idx in self._item_cache:
            return self._item_cache[idx]
        s = self.samples[idx]
        if "processed_idx" in s:
            item = self._getitem_processed(s)
            if self.cache_items:
                self._item_cache[idx] = item
            return item
        item = self._getitem_mit(s)
        if self.cache_items:
            self._item_cache[idx] = item
        return item

    def _getitem_precomputed(self, idx: int) -> Dict[str, Any]:
        if (
            self._cache_y is None
            or self._cache_mask is None
            or self._cache_horizon is None
            or self._cache_target_steps is None
            or self._cache_history_features is None
            or self._cache_history_cycles is None
        ):
            raise RuntimeError("Precomputed cache arrays are not loaded")
        horizon = int(self._cache_horizon[idx])
        width = horizon
        meta = self._cache_meta[idx]
        if self._cache_layout == "cycle_history":
            if self._cache_cycle_maps is None or self._cache_history_indices is None:
                raise RuntimeError("Precomputed cycle-history cache arrays are not loaded")
            hist_idx = np.asarray(self._cache_history_indices[idx], dtype=np.int64)
            maps_arr = np.array(self._cache_cycle_maps[hist_idx], copy=True)
        else:
            if self._cache_maps is None:
                raise RuntimeError("Precomputed sample-history map cache is not loaded")
            maps_arr = np.array(self._cache_maps[idx], copy=True)
        return {
            "maps": torch.tensor(maps_arr, dtype=torch.float32),
            "y": torch.tensor(np.array(self._cache_y[idx, :width], copy=True), dtype=torch.float32),
            "mask": torch.tensor(np.array(self._cache_mask[idx, :width], copy=True), dtype=torch.bool),
            "horizon": torch.tensor(horizon, dtype=torch.long),
            "prompt": str(meta["prompt"]),
            "cell_id": str(meta["cell_id"]),
            "cycle": int(meta["cycle"]),
            "target_steps": torch.tensor(np.array(self._cache_target_steps[idx, :width], copy=True), dtype=torch.long),
            "history_features": torch.tensor(np.array(self._cache_history_features[idx], copy=True), dtype=torch.float32),
            "history_cycles": torch.tensor(np.array(self._cache_history_cycles[idx], copy=True), dtype=torch.long),
        }

    def _getitem_mit(self, s: Dict[str, Any]) -> Dict[str, Any]:
        rec = self.records[int(s["record_idx"])]
        df = add_cycle_features(rec.summary)
        row_idx = int(s["row_idx"])
        horizon = int(s["horizon"])
        cycle_id = int(df.iloc[row_idx]["cycle"])
        start = row_idx - self.history_len + 1
        hist_df = df.iloc[start : row_idx + 1]
        future = df.iloc[row_idx + 1 : row_idx + 1 + horizon]
        map_rows: List[np.ndarray] = []
        map_names: List[str] = []
        for hist_cycle in hist_df["cycle"].to_numpy(dtype=np.int64):
            maps_i, names_i = self._mit_cycle_maps(rec, int(hist_cycle))
            map_rows.append(maps_i)
            if not map_names:
                map_names = names_i
        older = df.iloc[:start]
        if len(older):
            summary = older[["QD", "IR", "chargetime"]].to_numpy(dtype=np.float32)
        else:
            summary = hist_df[["QD", "IR", "chargetime"]].to_numpy(dtype=np.float32)
        prompt = self._prompt_from_history(summary, ["QD", "IR", "chargetime"], horizon, rec.cell_id, cycle_id, map_names)
        y = future["SOH"].to_numpy(dtype=np.float32)
        item = {
            "maps": torch.tensor(np.stack(map_rows, axis=0), dtype=torch.float32),
            "history_features": torch.tensor(self._mit_history_features(df, start, row_idx + 1), dtype=torch.float32),
            "history_cycles": torch.tensor(hist_df["cycle"].to_numpy(dtype=np.int64), dtype=torch.long),
            "y": torch.tensor(y, dtype=torch.float32),
            "mask": torch.ones(horizon, dtype=torch.bool),
            "horizon": torch.tensor(horizon, dtype=torch.long),
            "prompt": prompt,
            "cell_id": rec.cell_id,
            "cycle": cycle_id,
            "target_steps": torch.tensor(future["cycle"].to_numpy(dtype=np.int64), dtype=torch.long),
        }
        return item

    def _getitem_processed(self, s: Dict[str, Any]) -> Dict[str, Any]:
        cell = self.processed_cells[int(s["processed_idx"])]
        row_idx = int(s["row_idx"])
        horizon = int(s["horizon"])
        cycle_id = int(cell["cycle_id"][row_idx])
        future_slice = slice(row_idx + 1, row_idx + 1 + horizon)
        start = row_idx - self.history_len + 1
        map_rows: List[np.ndarray] = []
        map_names: List[str] = []
        for hist_row in range(start, row_idx + 1):
            maps_i, names_i = self._processed_cycle_maps(cell, hist_row)
            map_rows.append(maps_i)
            if not map_names:
                map_names = names_i
        older_stop = max(start, 1)
        hist_cols: List[np.ndarray] = []
        variables: List[str] = []
        if "capacity_summary" in cell:
            hist_cols.append(np.asarray(cell["capacity_summary"][:older_stop], dtype=np.float32))
            variables.append("capacity")
        elif "capacity" in cell:
            cap = np.asarray(cell["capacity"][:older_stop], dtype=np.float32)
            hist_cols.append(cap[:, -1] if cap.ndim == 2 else cap.reshape(older_stop, -1)[:, -1])
            variables.append("capacity")
        if "internal_resistance" in cell:
            hist_cols.append(np.asarray(cell["internal_resistance"][:older_stop], dtype=np.float32))
            variables.append("internal_resistance")
        if "charge_time" in cell:
            hist_cols.append(np.asarray(cell["charge_time"][:older_stop], dtype=np.float32))
            variables.append("charge_time")
        if not hist_cols:
            hist_cols.append(np.asarray(cell["cycle_id"][:older_stop], dtype=np.float32))
            variables.append("cycle_id")
        hist = np.stack(hist_cols, axis=-1)
        prompt = self._prompt_from_history(hist, variables, horizon, str(cell["cell_id"]), cycle_id, map_names)
        n = len(cell["cycle_id"])
        target_steps = np.asarray(cell["cycle_id"][future_slice], dtype=np.int64)
        y = np.asarray(cell["soh"][future_slice], dtype=np.float32)
        item = {
            "maps": torch.tensor(np.stack(map_rows, axis=0), dtype=torch.float32),
            "history_features": torch.tensor(self._processed_history_features(cell, start, row_idx + 1, n), dtype=torch.float32),
            "history_cycles": torch.tensor(np.asarray(cell["cycle_id"][start : row_idx + 1], dtype=np.int64), dtype=torch.long),
            "y": torch.tensor(y, dtype=torch.float32),
            "mask": torch.ones(horizon, dtype=torch.bool),
            "horizon": torch.tensor(horizon, dtype=torch.long),
            "prompt": prompt,
            "cell_id": str(cell["cell_id"]),
            "cycle": cycle_id,
            "target_steps": torch.tensor(target_steps, dtype=torch.long),
        }
        return item


def collate_graph_report_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    max_h = max(int(b["horizon"]) for b in batch)
    max_t = max(b["maps"].shape[0] for b in batch)
    max_c = max(b["maps"].shape[1] for b in batch)
    max_hmap = max(b["maps"].shape[2] for b in batch)
    max_wmap = max(b["maps"].shape[3] for b in batch)
    feat_dim = max(b["history_features"].shape[-1] for b in batch)
    maps = torch.zeros(len(batch), max_t, max_c, max_hmap, max_wmap, dtype=torch.float32)
    history_features = torch.zeros(len(batch), max_t, feat_dim, dtype=torch.float32)
    history_cycles = torch.zeros(len(batch), max_t, dtype=torch.long)
    y = torch.zeros(len(batch), max_h, dtype=torch.float32)
    mask = torch.zeros(len(batch), max_h, dtype=torch.bool)
    target_steps = torch.zeros(len(batch), max_h, dtype=torch.long)
    for i, b in enumerate(batch):
        t, c, hm, wm = b["maps"].shape
        steps = len(b["y"])
        maps[i, :t, :c, :hm, :wm] = b["maps"]
        history_features[i, : b["history_features"].shape[0], : b["history_features"].shape[1]] = b["history_features"]
        history_cycles[i, : len(b["history_cycles"])] = b["history_cycles"]
        y[i, :steps] = b["y"]
        mask[i, :steps] = True
        target_steps[i, :steps] = b["target_steps"]
    return {
        "maps": maps,
        "y": y,
        "mask": mask,
        "horizon": torch.stack([b["horizon"] for b in batch]),
        "prompt": [b["prompt"] for b in batch],
        "cell_id": [b["cell_id"] for b in batch],
        "cycle": torch.tensor([b["cycle"] for b in batch], dtype=torch.long),
        "target_steps": target_steps,
        "history_features": history_features,
        "history_cycles": history_cycles,
    }
