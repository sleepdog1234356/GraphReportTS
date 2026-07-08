from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from scipy.io import loadmat

try:
    from .raw_signal import current_to_capacity, resample_1d
except ImportError:
    from raw_signal import current_to_capacity, resample_1d


CALCE_COLUMNS = [
    "Test_Time(s)",
    "Date_Time",
    "Step_Time(s)",
    "Step_Index",
    "Cycle_Index",
    "Current(A)",
    "Voltage(V)",
    "Charge_Capacity(Ah)",
    "Discharge_Capacity(Ah)",
]


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _finite_resample(values: Iterable[float], length: int, fill: float = 0.0) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float32).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.full(length, fill, dtype=np.float32)
    return resample_1d(arr, length)


def _soh_from_capacity(capacity: np.ndarray) -> np.ndarray:
    capacity = np.asarray(capacity, dtype=np.float32)
    valid = capacity[np.isfinite(capacity) & (capacity > 0)]
    if len(valid) == 0:
        return np.ones_like(capacity, dtype=np.float32)
    init = float(np.nanmedian(valid[: min(10, len(valid))]))
    if init <= 0:
        init = float(np.nanmax(valid))
    soh = capacity / max(init, 1e-6)
    return np.nan_to_num(soh, nan=1.0, posinf=1.0, neginf=0.0).astype(np.float32)


def _date_key(path: Path) -> tuple:
    nums = [int(x) for x in re.findall(r"\d+", path.stem)]
    return tuple(nums)


def _channel_sheets(path: Path) -> List[str]:
    import openpyxl

    wb = openpyxl.load_workbook(path, read_only=True, keep_links=False, data_only=True)
    try:
        return [name for name in wb.sheetnames if "Channel" in name]
    finally:
        wb.close()


def _read_calce_excel(path: Path) -> pd.DataFrame:
    frames = []
    for sheet in _channel_sheets(path):
        df = pd.read_excel(path, sheet_name=sheet, usecols=lambda c: c in CALCE_COLUMNS)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=CALCE_COLUMNS)
    out = pd.concat(frames, ignore_index=True)
    return out[[c for c in CALCE_COLUMNS if c in out.columns]]


def preprocess_calce_cell(cell_dir: Path, out_path: Path, resample_len: int = 128) -> Dict[str, object]:
    files = sorted(cell_dir.rglob("*.xlsx"), key=_date_key)
    if not files:
        raise FileNotFoundError(f"No CALCE xlsx files found under {cell_dir}")

    frames = []
    offset = 0
    for path in files:
        df = _read_calce_excel(path)
        if df.empty or "Cycle_Index" not in df:
            continue
        df = df.copy()
        df["Cycle_Index"] = pd.to_numeric(df["Cycle_Index"], errors="coerce").ffill().bfill()
        df = df.dropna(subset=["Cycle_Index"])
        if df.empty:
            continue
        local_cycle = df["Cycle_Index"].astype(int)
        df["Cycle_Index"] = local_cycle + offset
        offset = int(df["Cycle_Index"].max())
        frames.append(df)
    if not frames:
        raise RuntimeError(f"No readable CALCE channel data under {cell_dir}")

    data = pd.concat(frames, ignore_index=True)
    if "Date_Time" in data:
        data["Date_Time"] = pd.to_datetime(data["Date_Time"], errors="coerce")
        data = data.sort_values(["Date_Time", "Test_Time(s)"], na_position="last")

    cycles = []
    current = []
    voltage = []
    temperature = []
    capacity = []
    time = []
    cap_summary = []
    for cycle_id, group in data.groupby("Cycle_Index", sort=True):
        group = group.replace([np.inf, -np.inf], np.nan).dropna(subset=["Current(A)", "Voltage(V)"], how="all")
        if len(group) < 4:
            continue
        t = pd.to_numeric(group.get("Test_Time(s)", pd.Series(np.arange(len(group)))), errors="coerce").to_numpy(dtype=np.float32)
        if not np.isfinite(t).any():
            t = np.arange(len(group), dtype=np.float32)
        t = t - np.nanmin(t)
        cur = pd.to_numeric(group.get("Current(A)", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        volt = pd.to_numeric(group.get("Voltage(V)", 0.0), errors="coerce").interpolate(limit_direction="both").fillna(0.0).to_numpy(dtype=np.float32)
        chg = pd.to_numeric(group.get("Charge_Capacity(Ah)", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        dis = pd.to_numeric(group.get("Discharge_Capacity(Ah)", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        cap = np.where(dis > 0, dis, chg).astype(np.float32)
        cap_value = float(np.nanmax(dis)) if np.nanmax(dis) > 0 else float(np.nanmax(np.abs(cap)))
        if cap_value <= 0:
            cap_int = current_to_capacity(t, cur)
            cap_value = float(np.nanmax(np.abs(cap_int)))
            cap = cap_int
        cycles.append(int(cycle_id))
        time.append(_finite_resample(t, resample_len))
        current.append(_finite_resample(cur, resample_len))
        voltage.append(_finite_resample(volt, resample_len))
        temperature.append(np.full(resample_len, 25.0, dtype=np.float32))
        capacity.append(_finite_resample(cap, resample_len))
        cap_summary.append(cap_value)

    if not cycles:
        raise RuntimeError(f"No valid CALCE cycles found under {cell_dir}")

    cap_summary_arr = np.asarray(cap_summary, dtype=np.float32)
    payload = {
        "cycle_id": np.asarray(cycles, dtype=np.int64),
        "soh": _soh_from_capacity(cap_summary_arr),
        "current": np.stack(current).astype(np.float32),
        "voltage": np.stack(voltage).astype(np.float32),
        "temperature": np.stack(temperature).astype(np.float32),
        "capacity": np.stack(capacity).astype(np.float32),
        "time": np.stack(time).astype(np.float32),
        "capacity_summary": cap_summary_arr,
        "source_has_temperature": np.asarray(False),
    }
    _ensure_dir(out_path.parent)
    np.savez_compressed(out_path, **payload)
    return {"cell_id": out_path.stem, "cycles": len(cycles), "out": str(out_path)}


def _mat_cell_get(obj, index: int):
    try:
        return obj[0][index][index]
    except Exception:
        return obj[0][index]


def _xjtu_value(cycle, variable_index: int) -> np.ndarray:
    val = cycle[variable_index]
    if variable_index == 7:
        return np.array([], dtype=np.float32)
    return np.asarray(val, dtype=np.float32).reshape(-1)


def preprocess_xjtu_mat(path: Path, out_path: Path, resample_len: int = 128) -> Dict[str, object]:
    mat = loadmat(path)
    if "data" not in mat or "summary" not in mat:
        raise ValueError(f"{path} does not contain XJTU data/summary arrays")
    data = mat["data"]
    summary = mat["summary"]
    n_cycles = int(data.shape[1])
    try:
        cap_summary = np.asarray(summary[0][0][1], dtype=np.float32).reshape(-1)
    except Exception:
        cap_summary = np.full(n_cycles, np.nan, dtype=np.float32)

    cycles = []
    current = []
    voltage = []
    temperature = []
    capacity = []
    time = []
    cap_values = []
    for idx in range(n_cycles):
        cycle = data[0][idx]
        t = _xjtu_value(cycle, 1)
        volt = _xjtu_value(cycle, 2)
        cur = _xjtu_value(cycle, 3)
        cap = _xjtu_value(cycle, 4)
        temp = _xjtu_value(cycle, 6)
        if len(cur) < 4 or len(volt) < 4:
            continue
        if len(t) != len(cur):
            t = np.arange(len(cur), dtype=np.float32)
        if len(cap) != len(cur):
            cap = current_to_capacity(t * 60.0, cur)
        if len(temp) != len(cur):
            temp = np.full(len(cur), 25.0, dtype=np.float32)
        cap_value = cap_summary[idx] if idx < len(cap_summary) and np.isfinite(cap_summary[idx]) else np.nanmax(np.abs(cap))
        cycles.append(idx + 1)
        time.append(_finite_resample(t, resample_len))
        current.append(_finite_resample(cur, resample_len))
        voltage.append(_finite_resample(volt, resample_len))
        temperature.append(_finite_resample(temp, resample_len, fill=25.0))
        capacity.append(_finite_resample(cap, resample_len))
        cap_values.append(float(cap_value))

    if not cycles:
        raise RuntimeError(f"No valid XJTU cycles found in {path}")

    cap_values_arr = np.asarray(cap_values, dtype=np.float32)
    payload = {
        "cycle_id": np.asarray(cycles, dtype=np.int64),
        "soh": _soh_from_capacity(cap_values_arr),
        "current": np.stack(current).astype(np.float32),
        "voltage": np.stack(voltage).astype(np.float32),
        "temperature": np.stack(temperature).astype(np.float32),
        "capacity": np.stack(capacity).astype(np.float32),
        "time": np.stack(time).astype(np.float32),
        "capacity_summary": cap_values_arr,
        "source_has_temperature": np.asarray(True),
    }
    _ensure_dir(out_path.parent)
    np.savez_compressed(out_path, **payload)
    return {"cell_id": out_path.stem, "cycles": len(cycles), "out": str(out_path)}


def preprocess_calce(data_root: Path, resample_len: int) -> List[Dict[str, object]]:
    raw_root = data_root / "raw" / "battery" / "calce"
    out_root = _ensure_dir(data_root / "processed" / "battery" / "calce")
    cells = []
    for name in ["CS2_35", "CS2_36", "CS2_37", "CS2_38"]:
        candidates = [p for p in raw_root.rglob(name) if p.is_dir()]
        if not candidates:
            raise FileNotFoundError(f"Missing CALCE cell directory for {name} under {raw_root}")
        cell_dir = sorted(candidates, key=lambda p: len(str(p)))[-1]
        cells.append(preprocess_calce_cell(cell_dir, out_root / f"{name}.npz", resample_len=resample_len))
    return cells


def preprocess_xjtu(data_root: Path, resample_len: int, max_cells: Optional[int]) -> List[Dict[str, object]]:
    raw_root = data_root / "raw" / "battery" / "xjtu"
    out_root = _ensure_dir(data_root / "processed" / "battery" / "xjtu")
    mats = sorted(raw_root.rglob("*.mat"))
    if not mats:
        raise FileNotFoundError(f"No XJTU .mat files found under {raw_root}")
    if max_cells is not None:
        mats = mats[: int(max_cells)]
    cells = []
    for path in mats:
        cells.append(preprocess_xjtu_mat(path, out_root / f"{path.stem}.npz", resample_len=resample_len))
    return cells


def parse_args():
    p = argparse.ArgumentParser(description="Preprocess raw battery datasets to GraphReportTS .npz files")
    p.add_argument("--dataset", choices=["calce", "xjtu", "all"], default="all")
    p.add_argument("--data_root", type=str, default="bstalignment/data")
    p.add_argument("--resample_len", type=int, default=128)
    p.add_argument("--max_xjtu_cells", type=int, default=None)
    p.add_argument("--summary", type=str, default="runs/preprocess_battery_summary.json")
    return p.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    rows: Dict[str, List[Dict[str, object]]] = {}
    if args.dataset in {"calce", "all"}:
        rows["calce"] = preprocess_calce(data_root, args.resample_len)
    if args.dataset in {"xjtu", "all"}:
        rows["xjtu"] = preprocess_xjtu(data_root, args.resample_len, args.max_xjtu_cells)
    summary_path = Path(args.summary)
    _ensure_dir(summary_path.parent)
    summary_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
