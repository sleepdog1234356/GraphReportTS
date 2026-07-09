from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

try:
    from .data_mit import CellRecord, add_cycle_features, load_mit_battery_pkls, split_cells
    from .experiment_config import BATTERY_DATASET_NOTES
    from .raw_signal import build_multiview_maps, build_report_from_array, current_to_capacity
except ImportError:
    from data_mit import CellRecord, add_cycle_features, load_mit_battery_pkls, split_cells
    from experiment_config import BATTERY_DATASET_NOTES
    from raw_signal import build_multiview_maps, build_report_from_array, current_to_capacity


RAW_BATTERY_CHANNELS = ["current", "voltage", "temperature", "capacity"]
BATTERY_GRAPH_CACHE_VERSION = 1


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
        precomputed_cache_dir: Optional[str] = None,
        require_precomputed_cache: bool = False,
        seed: int = 42,
        max_cycles: Optional[int] = None,
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
        self.cache_items = bool(cache_items)
        self._item_cache: Dict[int, Dict[str, Any]] = {}
        self.precomputed_cache_dir = Path(precomputed_cache_dir) if precomputed_cache_dir else None
        self.require_precomputed_cache = bool(require_precomputed_cache)
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
        )
        self._precomputed = False
        self._cache_path: Optional[Path] = None
        self._cache_maps = None
        self._cache_y = None
        self._cache_mask = None
        self._cache_horizon = None
        self._cache_target_steps = None
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
        files = manifest.get("files", {})
        required = ["maps", "y", "mask", "horizon", "target_steps", "meta"]
        if any(not (cache_path / str(files.get(name, ""))).exists() for name in required):
            return False
        self._cache_path = cache_path
        self._cache_maps = np.load(cache_path / files["maps"], mmap_mode="r")
        self._cache_y = np.load(cache_path / files["y"], mmap_mode="r")
        self._cache_mask = np.load(cache_path / files["mask"], mmap_mode="r")
        self._cache_horizon = np.load(cache_path / files["horizon"], mmap_mode="r")
        self._cache_target_steps = np.load(cache_path / files["target_steps"], mmap_mode="r")
        with (cache_path / files["meta"]).open("r", encoding="utf-8") as f:
            self._cache_meta = [json.loads(line) for line in f if line.strip()]
        sample_count = int(manifest.get("sample_count", -1))
        if sample_count < 0 or sample_count != len(self._cache_meta) or sample_count != int(self._cache_maps.shape[0]):
            return False
        self._precomputed = True
        self.samples = []
        self.records = []
        self.processed_cells = []
        return True

    def _load_mit(self, seed: int, max_cycles: Optional[int]) -> None:
        records = load_mit_battery_pkls(self.data_root / "mit")
        train, val, test = split_cells(records, seed=seed)
        selected = {"train": train, "val": val, "test": test, "all": records}[self.split]
        self.records = list(selected)
        for rec_idx, rec in enumerate(self.records):
            df = add_cycle_features(rec.summary)
            if max_cycles is not None:
                df = df.iloc[:max_cycles].copy()
            for row_idx in range(0, len(df) - 1):
                available = min(self.max_horizon, len(df) - row_idx - 1)
                if available <= 0:
                    continue
                self.samples.append(
                    {
                        "record_idx": rec_idx,
                        "row_idx": row_idx,
                        "cycle_id": int(df.iloc[row_idx]["cycle"]),
                        "horizon": available,
                    }
                )

    def _load_processed(self, max_cycles: Optional[int], seed: int) -> None:
        note = BATTERY_DATASET_NOTES.get(self.dataset_name, {})
        processed_dir = Path(note.get("processed_dir", self.data_root / "processed" / "battery" / self.dataset_name))
        files = sorted(processed_dir.glob("*.npz"))
        if not files:
            required = "\n  - ".join(note.get("required", []))
            raise FileNotFoundError(
                f"No processed {self.dataset_name.upper()} files found under {processed_dir}.\n"
                f"Place raw data under {note.get('raw_dir')} and preprocess to .npz files with:\n  - {required}"
            )
        rng = np.random.default_rng(seed)
        order = np.arange(len(files))
        rng.shuffle(order)
        if len(order) >= 3:
            n_train = max(1, int(len(order) * 0.7))
            n_val = max(1, int(len(order) * 0.15))
            if n_train + n_val >= len(order):
                n_train = max(1, len(order) - 2)
                n_val = 1
        elif len(order) == 2:
            n_train = 1
            n_val = 0
        else:
            n_train = len(order)
            n_val = 0
        split_ids = {
            "train": order[:n_train],
            "val": order[n_train : n_train + n_val],
            "test": order[n_train + n_val :],
            "all": order,
        }[self.split]
        selected = [files[i] for i in split_ids]
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
            for row_idx in range(0, n - 1):
                available = min(self.max_horizon, n - row_idx - 1)
                if available <= 0:
                    continue
                self.samples.append(
                    {
                        "processed_idx": cell_idx,
                        "row_idx": row_idx,
                        "cycle_id": int(cell["cycle_id"][row_idx]),
                        "horizon": available,
                    }
                )

    def __len__(self) -> int:
        if self._precomputed:
            return int(self._cache_maps.shape[0])
        return len(self.samples)

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
        rec = self.records[int(s["record_idx"])]
        df = add_cycle_features(rec.summary)
        row_idx = int(s["row_idx"])
        horizon = int(s["horizon"])
        cycle_id = int(df.iloc[row_idx]["cycle"])
        future = df.iloc[row_idx + 1 : row_idx + 1 + horizon]
        channels = _extract_mit_cycle_channels(rec, cycle_id)
        if not any(len(v) for v in channels.values()):
            if not self.allow_summary_fallback:
                raise RuntimeError(
                    f"Raw MIT cycle arrays not found for {rec.cell_id} cycle {cycle_id}. "
                    "For formal GraphReportTS experiments, rebuild MIT pkl with raw cycles. "
                    "Use allow_summary_fallback only for smoke tests."
                )
            channels = _summary_pseudo_channels(rec, cycle_id)
        maps, map_names = build_multiview_maps(
            channels,
            resample_len=self.resample_len,
            delay_dim=self.delay_dim,
            delay_lag=self.delay_lag,
            include_derivatives=self.include_derivatives,
            include_hankel=self.include_hankel,
            include_ic_dv=self.include_ic_dv,
        )
        hist = df.iloc[: row_idx + 1][["QD", "IR", "chargetime"]].to_numpy(dtype=np.float32)
        prompt = build_report_from_array(
            hist,
            domain=f"battery-{self.dataset_name}",
            horizon=horizon,
            variables=["QD", "IR", "chargetime"],
        )
        prompt += (
            f" Battery adapter: cell_id={rec.cell_id}; cycle={cycle_id}; "
            f"channels={', '.join(map_names[:10])}; target=SOH."
        )
        item = {
            "maps": torch.tensor(maps, dtype=torch.float32),
            "y": torch.tensor(np.concatenate([[float(df.iloc[row_idx]['SOH'])], future["SOH"].to_numpy(dtype=np.float32)]), dtype=torch.float32),
            "mask": torch.ones(horizon + 1, dtype=torch.bool),
            "horizon": torch.tensor(horizon, dtype=torch.long),
            "prompt": prompt,
            "cell_id": rec.cell_id,
            "cycle": cycle_id,
            "target_steps": torch.tensor(np.concatenate([[cycle_id], future["cycle"].to_numpy(dtype=np.int64)]), dtype=torch.long),
        }
        if self.cache_items:
            self._item_cache[idx] = item
        return item

    def _getitem_precomputed(self, idx: int) -> Dict[str, Any]:
        if self._cache_maps is None or self._cache_y is None or self._cache_mask is None or self._cache_horizon is None or self._cache_target_steps is None:
            raise RuntimeError("Precomputed cache arrays are not loaded")
        horizon = int(self._cache_horizon[idx])
        width = horizon + 1
        meta = self._cache_meta[idx]
        return {
            "maps": torch.tensor(np.array(self._cache_maps[idx], copy=True), dtype=torch.float32),
            "y": torch.tensor(np.array(self._cache_y[idx, :width], copy=True), dtype=torch.float32),
            "mask": torch.tensor(np.array(self._cache_mask[idx, :width], copy=True), dtype=torch.bool),
            "horizon": torch.tensor(horizon, dtype=torch.long),
            "prompt": str(meta["prompt"]),
            "cell_id": str(meta["cell_id"]),
            "cycle": int(meta["cycle"]),
            "target_steps": torch.tensor(np.array(self._cache_target_steps[idx, :width], copy=True), dtype=torch.long),
        }

    def _getitem_processed(self, s: Dict[str, Any]) -> Dict[str, Any]:
        cell = self.processed_cells[int(s["processed_idx"])]
        row_idx = int(s["row_idx"])
        horizon = int(s["horizon"])
        cycle_id = int(cell["cycle_id"][row_idx])
        future_slice = slice(row_idx + 1, row_idx + 1 + horizon)
        channels = {
            "current": np.asarray(cell["current"][row_idx], dtype=np.float32),
            "voltage": np.asarray(cell["voltage"][row_idx], dtype=np.float32),
            "temperature": np.asarray(cell["temperature"][row_idx], dtype=np.float32),
            "capacity": np.asarray(cell["capacity"][row_idx], dtype=np.float32),
        }
        maps, map_names = build_multiview_maps(
            channels,
            resample_len=self.resample_len,
            delay_dim=self.delay_dim,
            delay_lag=self.delay_lag,
            include_derivatives=self.include_derivatives,
            include_hankel=self.include_hankel,
            include_ic_dv=self.include_ic_dv,
        )
        hist_cols = [np.asarray(cell["soh"][: row_idx + 1], dtype=np.float32)]
        if "capacity_summary" in cell:
            hist_cols.append(np.asarray(cell["capacity_summary"][: row_idx + 1], dtype=np.float32))
        hist = np.stack(hist_cols, axis=-1)
        variables = ["SOH"] + (["capacity"] if len(hist_cols) > 1 else [])
        prompt = build_report_from_array(hist, domain=f"battery-{self.dataset_name}", horizon=horizon, variables=variables)
        prompt += (
            f" Battery adapter: cell_id={cell['cell_id']}; cycle={cycle_id}; "
            f"channels={', '.join(map_names[:10])}; target=SOH."
        )
        target_steps = np.concatenate([[cycle_id], np.asarray(cell["cycle_id"][future_slice], dtype=np.int64)])
        y = np.concatenate([[float(cell["soh"][row_idx])], np.asarray(cell["soh"][future_slice], dtype=np.float32)])
        item = {
            "maps": torch.tensor(maps, dtype=torch.float32),
            "y": torch.tensor(y, dtype=torch.float32),
            "mask": torch.ones(horizon + 1, dtype=torch.bool),
            "horizon": torch.tensor(horizon, dtype=torch.long),
            "prompt": prompt,
            "cell_id": str(cell["cell_id"]),
            "cycle": cycle_id,
            "target_steps": torch.tensor(target_steps, dtype=torch.long),
        }
        return item


def collate_graph_report_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    max_h = max(int(b["horizon"]) for b in batch) + 1
    max_c = max(b["maps"].shape[0] for b in batch)
    max_hmap = max(b["maps"].shape[1] for b in batch)
    max_wmap = max(b["maps"].shape[2] for b in batch)
    maps = torch.zeros(len(batch), max_c, max_hmap, max_wmap, dtype=torch.float32)
    y = torch.zeros(len(batch), max_h, dtype=torch.float32)
    mask = torch.zeros(len(batch), max_h, dtype=torch.bool)
    target_steps = torch.zeros(len(batch), max_h, dtype=torch.long)
    for i, b in enumerate(batch):
        c, hm, wm = b["maps"].shape
        steps = len(b["y"])
        maps[i, :c, :hm, :wm] = b["maps"]
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
    }
