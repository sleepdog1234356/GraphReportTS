from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import json
import multiprocessing
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence, Tuple

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

try:
    from .data_battery_raw import (
        BATTERY_HISTORY_FEATURE_DIM,
        BatteryRawGraphDataset,
        battery_graph_cache_config,
        battery_graph_cache_path,
    )
    from .data_mit import add_cycle_features
except ImportError:
    from data_battery_raw import (
        BATTERY_HISTORY_FEATURE_DIM,
        BatteryRawGraphDataset,
        battery_graph_cache_config,
        battery_graph_cache_path,
    )
    from data_mit import add_cycle_features


def parse_args():
    p = argparse.ArgumentParser(description="Precompute deterministic battery GraphReportTS map caches")
    p.add_argument("--dataset", choices=["mit", "calce", "xjtu"], required=True)
    p.add_argument("--data_root", type=str, default="bstalignment/data")
    p.add_argument("--cache_dir", type=str, default="runs/cache/battery_graph")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=["train", "val", "test", "all"])
    p.add_argument("--pred_len", type=int, default=20)
    p.add_argument("--history_len", type=int, default=32)
    p.add_argument("--resample_len", type=int, default=128)
    p.add_argument("--delay_dim", type=int, default=8)
    p.add_argument("--delay_lag", type=int, default=1)
    p.add_argument("--no_ic_dv", action="store_true")
    p.add_argument("--no_hankel_map", action="store_true")
    p.add_argument("--no_derivative_map", action="store_true")
    p.add_argument("--allow_summary_fallback", action="store_true")
    p.add_argument("--max_cycles", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=128, help="Maximum number of cycle maps scheduled in flight")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def _cache_is_valid(cache_path: Path, config: Dict[str, Any]) -> bool:
    manifest_path = cache_path / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if manifest.get("config") != config:
        return False
    files = manifest.get("files", {})
    layout = str(manifest.get("layout", "sample_history"))
    names = ["cycle_maps", "history_indices"] if layout == "cycle_history" else ["maps"]
    names += ["y", "mask", "horizon", "target_steps", "history_features", "history_cycles", "meta"]
    return all(
        (cache_path / str(files.get(name, ""))).exists()
        for name in names
    )


def _write_meta(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _flush_and_close_memmap(array: np.memmap) -> None:
    array.flush()
    mapping = getattr(array, "_mmap", None)
    if mapping is not None:
        mapping.close()


def _collect_cycle_entries(ds: BatteryRawGraphDataset) -> Dict[Tuple[str, int], Tuple[str, int, int]]:
    entries: Dict[Tuple[str, int], Tuple[str, int, int]] = {}
    mit_dfs = {}
    for sample in ds.samples:
        row_idx = int(sample["row_idx"])
        start = row_idx - ds.history_len + 1
        if "processed_idx" in sample:
            processed_idx = int(sample["processed_idx"])
            cell = ds.processed_cells[processed_idx]
            cell_id = str(cell["cell_id"])
            for hist_row in range(start, row_idx + 1):
                cycle_id = int(cell["cycle_id"][hist_row])
                entries.setdefault((cell_id, cycle_id), ("processed", processed_idx, hist_row))
        else:
            record_idx = int(sample["record_idx"])
            rec = ds.records[record_idx]
            if record_idx not in mit_dfs:
                mit_dfs[record_idx] = add_cycle_features(rec.summary)
            df = mit_dfs[record_idx]
            for hist_row in range(start, row_idx + 1):
                cycle_id = int(df.iloc[hist_row]["cycle"])
                entries.setdefault((rec.cell_id, cycle_id), ("mit", record_idx, cycle_id))
    return entries


def _entry_maps(ds: BatteryRawGraphDataset, entry: Tuple[str, int, int]) -> Tuple[np.ndarray, List[str]]:
    kind, owner_idx, row_or_cycle = entry
    if kind == "processed":
        return ds._processed_cycle_maps(ds.processed_cells[owner_idx], row_or_cycle)
    return ds._mit_cycle_maps(ds.records[owner_idx], row_or_cycle)


_WORKER_DATASET: BatteryRawGraphDataset | None = None
CycleItem = Tuple[int, Tuple[str, int], Tuple[str, int, int]]
CycleResult = Tuple[int, Tuple[str, int], np.ndarray, List[str]]


def _cycle_item_batches(items: Sequence[CycleItem], batch_size: int) -> Iterator[List[CycleItem]]:
    for start in range(0, len(items), batch_size):
        yield list(items[start : start + batch_size])


def _compute_cycle_map_batch(batch: Sequence[CycleItem]) -> List[CycleResult]:
    if _WORKER_DATASET is None:
        raise RuntimeError("Cycle-map worker dataset was not initialized")
    results = []
    for cycle_idx, key, entry in batch:
        maps, names = _entry_maps(_WORKER_DATASET, entry)
        results.append((cycle_idx, key, maps, names))
    return results


def _parallel_cycle_map_results(
    ds: BatteryRawGraphDataset,
    cycle_items: Sequence[CycleItem],
    *,
    num_workers: int,
    batch_size: int,
) -> Iterator[CycleResult]:
    if not cycle_items:
        return
    worker_count = min(int(num_workers), int(batch_size), len(cycle_items))
    chunk_size = max(1, int(batch_size) // worker_count)
    batches = iter(_cycle_item_batches(cycle_items, chunk_size))
    use_fork = os.name != "nt" and "fork" in multiprocessing.get_all_start_methods()

    global _WORKER_DATASET
    _WORKER_DATASET = ds
    try:
        if use_fork:
            executor = ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=multiprocessing.get_context("fork"),
            )
        else:
            executor = ThreadPoolExecutor(max_workers=worker_count)
        with executor:
            pending = deque()
            for _ in range(worker_count):
                try:
                    pending.append(executor.submit(_compute_cycle_map_batch, next(batches)))
                except StopIteration:
                    break
            while pending:
                future = pending.popleft()
                yield from future.result()
                try:
                    pending.append(executor.submit(_compute_cycle_map_batch, next(batches)))
                except StopIteration:
                    pass
    finally:
        _WORKER_DATASET = None


def _cycle_map_results(
    ds: BatteryRawGraphDataset,
    cycle_items: Sequence[CycleItem],
    *,
    num_workers: int,
    batch_size: int,
) -> Iterator[CycleResult]:
    if num_workers > 0:
        yield from _parallel_cycle_map_results(
            ds,
            cycle_items,
            num_workers=num_workers,
            batch_size=batch_size,
        )
        return
    for cycle_idx, key, entry in cycle_items:
        maps, names = _entry_maps(ds, entry)
        yield cycle_idx, key, maps, names


def _sample_payload(
    ds: BatteryRawGraphDataset,
    sample: Dict[str, Any],
    map_names: List[str],
    mit_dfs: Dict[int, Any],
) -> Dict[str, Any]:
    row_idx = int(sample["row_idx"])
    horizon = int(sample["horizon"])
    start = row_idx - ds.history_len + 1
    if "processed_idx" in sample:
        cell = ds.processed_cells[int(sample["processed_idx"])]
        cycle_id = int(cell["cycle_id"][row_idx])
        future_slice = slice(row_idx + 1, row_idx + 1 + horizon)
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
        return {
            "y": np.asarray(cell["soh"][future_slice], dtype=np.float32),
            "mask": np.ones(horizon, dtype=np.bool_),
            "horizon": horizon,
            "target_steps": np.asarray(cell["cycle_id"][future_slice], dtype=np.int64),
            "history_features": ds._processed_history_features(cell, start, row_idx + 1, len(cell["cycle_id"])),
            "history_cycles": np.asarray(cell["cycle_id"][start : row_idx + 1], dtype=np.int64),
            "prompt": ds._prompt_from_history(hist, variables, horizon, str(cell["cell_id"]), cycle_id, map_names),
            "cell_id": str(cell["cell_id"]),
            "cycle": cycle_id,
        }

    record_idx = int(sample["record_idx"])
    rec = ds.records[record_idx]
    if record_idx not in mit_dfs:
        mit_dfs[record_idx] = add_cycle_features(rec.summary)
    df = mit_dfs[record_idx]
    cycle_id = int(df.iloc[row_idx]["cycle"])
    hist_df = df.iloc[start : row_idx + 1]
    future = df.iloc[row_idx + 1 : row_idx + 1 + horizon]
    older = df.iloc[:start]
    if len(older):
        summary = older[["QD", "IR", "chargetime"]].to_numpy(dtype=np.float32)
    else:
        summary = hist_df[["QD", "IR", "chargetime"]].to_numpy(dtype=np.float32)
    return {
        "y": future["SOH"].to_numpy(dtype=np.float32),
        "mask": np.ones(horizon, dtype=np.bool_),
        "horizon": horizon,
        "target_steps": future["cycle"].to_numpy(dtype=np.int64),
        "history_features": ds._mit_history_features(df, start, row_idx + 1),
        "history_cycles": hist_df["cycle"].to_numpy(dtype=np.int64),
        "prompt": ds._prompt_from_history(summary, ["QD", "IR", "chargetime"], horizon, rec.cell_id, cycle_id, map_names),
        "cell_id": rec.cell_id,
        "cycle": cycle_id,
    }


def precompute_split(args, split: str) -> Path:
    if args.num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    config = battery_graph_cache_config(
        dataset_name=args.dataset,
        split=split,
        max_horizon=args.pred_len,
        resample_len=args.resample_len,
        delay_dim=args.delay_dim,
        delay_lag=args.delay_lag,
        include_derivatives=not args.no_derivative_map,
        include_hankel=not args.no_hankel_map,
        include_ic_dv=not args.no_ic_dv,
        allow_summary_fallback=args.allow_summary_fallback,
        seed=args.seed,
        max_cycles=args.max_cycles,
        history_len=args.history_len,
    )
    cache_path = battery_graph_cache_path(args.cache_dir, config)
    if not args.force and _cache_is_valid(cache_path, config):
        print(f"cache exists: {cache_path}")
        return cache_path

    ds = BatteryRawGraphDataset(
        dataset_name=args.dataset,
        data_root=args.data_root,
        split=split,
        max_horizon=args.pred_len,
        resample_len=args.resample_len,
        delay_dim=args.delay_dim,
        delay_lag=args.delay_lag,
        include_derivatives=not args.no_derivative_map,
        include_hankel=not args.no_hankel_map,
        include_ic_dv=not args.no_ic_dv,
        allow_summary_fallback=args.allow_summary_fallback,
        cache_items=False,
        cycle_cache_size=0,
        seed=args.seed,
        max_cycles=args.max_cycles,
        history_len=args.history_len,
    )
    sample_count = len(ds)
    if sample_count == 0:
        raise RuntimeError(f"Cannot precompute empty battery graph cache for dataset={args.dataset} split={split}")

    cycle_entries = _collect_cycle_entries(ds)
    if not cycle_entries:
        raise RuntimeError(f"Cannot precompute battery graph cache without cycle entries for dataset={args.dataset} split={split}")
    cycle_items = list(cycle_entries.items())
    first_maps, first_names = _entry_maps(ds, cycle_items[0][1])
    cycle_map_shape = tuple(int(v) for v in first_maps.shape)
    target_width = int(args.pred_len)
    history_shape = (int(args.history_len), int(BATTERY_HISTORY_FEATURE_DIM))
    history_cycles_shape = (int(args.history_len),)
    tmp_path = cache_path.parent / f".{cache_path.name}.tmp"
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=False)

    cycle_maps = np.lib.format.open_memmap(
        tmp_path / "cycle_maps.npy", mode="w+", dtype=np.float32, shape=(len(cycle_items), *cycle_map_shape)
    )
    history_indices = np.lib.format.open_memmap(
        tmp_path / "history_indices.npy", mode="w+", dtype=np.int64, shape=(sample_count, int(args.history_len))
    )
    y = np.lib.format.open_memmap(tmp_path / "y.npy", mode="w+", dtype=np.float32, shape=(sample_count, target_width))
    mask = np.lib.format.open_memmap(tmp_path / "mask.npy", mode="w+", dtype=np.bool_, shape=(sample_count, target_width))
    horizon = np.lib.format.open_memmap(tmp_path / "horizon.npy", mode="w+", dtype=np.int64, shape=(sample_count,))
    target_steps = np.lib.format.open_memmap(tmp_path / "target_steps.npy", mode="w+", dtype=np.int64, shape=(sample_count, target_width))
    history_features = np.lib.format.open_memmap(
        tmp_path / "history_features.npy", mode="w+", dtype=np.float32, shape=(sample_count, *history_shape)
    )
    history_cycles = np.lib.format.open_memmap(
        tmp_path / "history_cycles.npy", mode="w+", dtype=np.int64, shape=(sample_count, *history_cycles_shape)
    )
    y[:] = 0.0
    mask[:] = False
    target_steps[:] = 0
    history_indices[:] = 0
    meta_rows = []

    key_to_cycle_idx: Dict[Tuple[str, int], int] = {}
    cycle_maps[0] = first_maps
    key_to_cycle_idx[cycle_items[0][0]] = 0
    pending_cycle_items = [
        (cycle_idx, key, entry)
        for cycle_idx, (key, entry) in enumerate(cycle_items[1:], start=1)
    ]
    cycle_results = _cycle_map_results(
        ds,
        pending_cycle_items,
        num_workers=int(args.num_workers),
        batch_size=int(args.batch_size),
    )
    for cycle_idx, key, maps_i, _ in tqdm(
        cycle_results,
        total=len(cycle_items),
        initial=1,
        desc=f"cycle maps {args.dataset}/{split}",
    ):
        if tuple(maps_i.shape) != cycle_map_shape:
            raise ValueError(f"Cycle map shape changed for {key}: expected {cycle_map_shape}, got {tuple(maps_i.shape)}")
        cycle_maps[cycle_idx] = maps_i
        key_to_cycle_idx[key] = cycle_idx

    mit_dfs: Dict[int, Any] = {}
    for sample_idx, sample in enumerate(tqdm(ds.samples, desc=f"samples {args.dataset}/{split}")):
        item = _sample_payload(ds, sample, first_names, mit_dfs)
        width = min(len(item["y"]), target_width)
        y[sample_idx, :width] = np.asarray(item["y"], dtype=np.float32)[:width]
        mask[sample_idx, :width] = np.asarray(item["mask"], dtype=np.bool_)[:width]
        horizon[sample_idx] = int(item["horizon"])
        target_steps[sample_idx, :width] = np.asarray(item["target_steps"], dtype=np.int64)[:width]
        item_history = np.asarray(item["history_features"], dtype=np.float32)
        item_history_cycles = np.asarray(item["history_cycles"], dtype=np.int64)
        if tuple(item_history.shape) != history_shape:
            raise ValueError(f"History feature shape changed at index {sample_idx}: expected {history_shape}, got {tuple(item_history.shape)}")
        if tuple(item_history_cycles.shape) != history_cycles_shape:
            raise ValueError(
                f"History cycle shape changed at index {sample_idx}: expected {history_cycles_shape}, got {tuple(item_history_cycles.shape)}"
            )
        history_features[sample_idx] = item_history
        history_cycles[sample_idx] = item_history_cycles
        for hist_pos, cycle_id in enumerate(item_history_cycles):
            key = (str(item["cell_id"]), int(cycle_id))
            history_indices[sample_idx, hist_pos] = key_to_cycle_idx[key]
        meta_rows.append({"prompt": item["prompt"], "cell_id": item["cell_id"], "cycle": int(item["cycle"])})

    for array in (
        cycle_maps,
        history_indices,
        y,
        mask,
        horizon,
        target_steps,
        history_features,
        history_cycles,
    ):
        _flush_and_close_memmap(array)
    _write_meta(tmp_path / "meta.jsonl", meta_rows)
    manifest = {
        "layout": "cycle_history",
        "config": config,
        "cycle_scale": ds.cycle_scale,
        "sample_count": sample_count,
        "cycle_count": len(cycle_items),
        "cycle_map_shape": cycle_map_shape,
        "history_shape": history_shape,
        "history_cycles_shape": history_cycles_shape,
        "target_width": target_width,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": {
            "cycle_maps": "cycle_maps.npy",
            "history_indices": "history_indices.npy",
            "y": "y.npy",
            "mask": "mask.npy",
            "horizon": "horizon.npy",
            "target_steps": "target_steps.npy",
            "history_features": "history_features.npy",
            "history_cycles": "history_cycles.npy",
            "meta": "meta.jsonl",
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    if cache_path.exists():
        shutil.rmtree(cache_path)
    tmp_path.rename(cache_path)
    print(f"wrote cache: {cache_path}")
    return cache_path


def main():
    args = parse_args()
    for split in args.splits:
        precompute_split(args, split)


if __name__ == "__main__":
    main()
