from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
from typing import Any, Dict, Iterable

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable: Iterable[Any], **kwargs: Any) -> Iterable[Any]:
        return iterable

try:
    from .data_battery_raw import (
        BATTERY_HISTORY_FEATURE_DIM,
        BatteryRawGraphDataset,
        battery_sequence_cache_config,
        battery_sequence_cache_path,
    )
    from .precompute_battery_graph_cache import (
        _collect_cycle_entries,
        _flush_and_close_memmap,
        _sample_payload,
        _write_meta,
    )
    from .raw_signal import FULL_BATTERY_PROMPT_MAP_NAMES
except ImportError:
    from data_battery_raw import (
        BATTERY_HISTORY_FEATURE_DIM,
        BatteryRawGraphDataset,
        battery_sequence_cache_config,
        battery_sequence_cache_path,
    )
    from precompute_battery_graph_cache import (
        _collect_cycle_entries,
        _flush_and_close_memmap,
        _sample_payload,
        _write_meta,
    )
    from raw_signal import FULL_BATTERY_PROMPT_MAP_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute deterministic battery sequence caches")
    parser.add_argument("--dataset", choices=["mit", "calce", "xjtu"], required=True)
    parser.add_argument("--data_root", type=str, default="bstalignment/data")
    parser.add_argument("--cache_dir", type=str, default="runs/cache/battery_sequence")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        choices=["train", "val", "test", "all"],
    )
    parser.add_argument("--pred_len", type=int, default=20)
    parser.add_argument("--history_len", type=int, default=32)
    parser.add_argument("--resample_len", type=int, default=128)
    parser.add_argument("--allow_summary_fallback", action="store_true")
    parser.add_argument("--max_cycles", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _sequence_cache_is_valid(cache_path: Path, config: Dict[str, Any]) -> bool:
    manifest_path = cache_path / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if manifest.get("layout") != "cycle_sequence_history" or manifest.get("config") != config:
        return False
    files = manifest.get("files", {})
    required = [
        "cycle_sequences",
        "history_indices",
        "y",
        "mask",
        "horizon",
        "target_steps",
        "history_features",
        "history_cycles",
        "meta",
    ]
    return all((cache_path / str(files.get(name, ""))).exists() for name in required)


def _entry_sequence(
    ds: BatteryRawGraphDataset,
    entry: tuple[str, int, int],
) -> tuple[np.ndarray, list[str]]:
    kind, owner_idx, row_or_cycle = entry
    if kind == "processed":
        values, names = ds._processed_cycle_input(ds.processed_cells[owner_idx], row_or_cycle)
    else:
        values, names = ds._mit_cycle_input(ds.records[owner_idx], row_or_cycle)
    if ds.input_representation != "sequence":
        raise RuntimeError("Sequence cache requires input_representation=sequence")
    return values, names


def _cycle_item_batches(cycle_items, batch_size):
    for start in range(0, len(cycle_items), batch_size):
        yield cycle_items[start : start + batch_size]


def _sequence_results(ds, cycle_items, num_workers, batch_size):
    def compute(item):
        cycle_idx, key, entry = item
        values, names = _entry_sequence(ds, entry)
        return cycle_idx, key, values, names

    if num_workers == 0:
        for item in cycle_items:
            yield compute(item)
        return
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        for batch in _cycle_item_batches(cycle_items, batch_size):
            yield from pool.map(compute, batch)


def precompute_sequence_split(args: argparse.Namespace, split: str) -> Path:
    if args.num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    allow_summary_fallback = bool(getattr(args, "allow_summary_fallback", False))
    config = battery_sequence_cache_config(
        dataset_name=args.dataset,
        split=split,
        max_horizon=args.pred_len,
        resample_len=args.resample_len,
        allow_summary_fallback=allow_summary_fallback,
        seed=args.seed,
        max_cycles=args.max_cycles,
        history_len=args.history_len,
    )
    cache_path = battery_sequence_cache_path(args.cache_dir, config)
    if not args.force and _sequence_cache_is_valid(cache_path, config):
        print(f"cache exists: {cache_path}")
        return cache_path

    ds = BatteryRawGraphDataset(
        dataset_name=args.dataset,
        data_root=args.data_root,
        split=split,
        max_horizon=args.pred_len,
        resample_len=args.resample_len,
        allow_summary_fallback=allow_summary_fallback,
        cache_items=False,
        cycle_cache_size=0,
        seed=args.seed,
        max_cycles=args.max_cycles,
        history_len=args.history_len,
        input_representation="sequence",
    )
    sample_count = len(ds)
    if sample_count == 0:
        raise RuntimeError(
            f"Cannot precompute empty battery sequence cache for dataset={args.dataset} split={split}"
        )

    cycle_entries = _collect_cycle_entries(ds)
    if not cycle_entries:
        raise RuntimeError(
            f"Cannot precompute battery sequence cache without cycle entries for dataset={args.dataset} split={split}"
        )
    cycle_items = [
        (cycle_idx, key, entry)
        for cycle_idx, (key, entry) in enumerate(cycle_entries.items())
    ]
    target_width = int(args.pred_len)
    history_shape = (int(args.history_len), int(BATTERY_HISTORY_FEATURE_DIM))
    history_cycles_shape = (int(args.history_len),)
    tmp_path = cache_path.parent / f".{cache_path.name}.tmp"
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=False)

    cycle_sequences = np.lib.format.open_memmap(
        tmp_path / "cycle_sequences.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(cycle_items), int(args.resample_len), 6),
    )
    history_indices = np.lib.format.open_memmap(
        tmp_path / "history_indices.npy",
        mode="w+",
        dtype=np.int64,
        shape=(sample_count, int(args.history_len)),
    )
    y = np.lib.format.open_memmap(
        tmp_path / "y.npy", mode="w+", dtype=np.float32, shape=(sample_count, target_width)
    )
    mask = np.lib.format.open_memmap(
        tmp_path / "mask.npy", mode="w+", dtype=np.bool_, shape=(sample_count, target_width)
    )
    horizon = np.lib.format.open_memmap(
        tmp_path / "horizon.npy", mode="w+", dtype=np.int64, shape=(sample_count,)
    )
    target_steps = np.lib.format.open_memmap(
        tmp_path / "target_steps.npy",
        mode="w+",
        dtype=np.int64,
        shape=(sample_count, target_width),
    )
    history_features = np.lib.format.open_memmap(
        tmp_path / "history_features.npy",
        mode="w+",
        dtype=np.float32,
        shape=(sample_count, *history_shape),
    )
    history_cycles = np.lib.format.open_memmap(
        tmp_path / "history_cycles.npy",
        mode="w+",
        dtype=np.int64,
        shape=(sample_count, *history_cycles_shape),
    )
    y[:] = 0.0
    mask[:] = False
    target_steps[:] = 0
    history_indices[:] = 0
    meta_rows = []

    key_to_cycle_idx: Dict[tuple[str, int], int] = {}
    for cycle_idx, key, values, _ in tqdm(
        _sequence_results(
            ds,
            cycle_items,
            int(args.num_workers),
            int(args.batch_size),
        ),
        total=len(cycle_items),
        desc=f"cycle sequences {args.dataset}/{split}",
    ):
        expected_shape = (int(args.resample_len), 6)
        if tuple(values.shape) != expected_shape:
            raise ValueError(
                f"Cycle sequence shape changed for {key}: expected {expected_shape}, got {tuple(values.shape)}"
            )
        cycle_sequences[cycle_idx] = values
        key_to_cycle_idx[key] = cycle_idx

    mit_dfs: Dict[int, Any] = {}
    prompt_names = list(FULL_BATTERY_PROMPT_MAP_NAMES)
    for sample_idx, sample in enumerate(tqdm(ds.samples, desc=f"samples {args.dataset}/{split}")):
        item = _sample_payload(ds, sample, prompt_names, mit_dfs)
        width = min(len(item["y"]), target_width)
        y[sample_idx, :width] = np.asarray(item["y"], dtype=np.float32)[:width]
        mask[sample_idx, :width] = np.asarray(item["mask"], dtype=np.bool_)[:width]
        horizon[sample_idx] = int(item["horizon"])
        target_steps[sample_idx, :width] = np.asarray(
            item["target_steps"], dtype=np.int64
        )[:width]
        item_history = np.asarray(item["history_features"], dtype=np.float32)
        item_history_cycles = np.asarray(item["history_cycles"], dtype=np.int64)
        if tuple(item_history.shape) != history_shape:
            raise ValueError(
                f"History feature shape changed at index {sample_idx}: expected {history_shape}, got {tuple(item_history.shape)}"
            )
        if tuple(item_history_cycles.shape) != history_cycles_shape:
            raise ValueError(
                f"History cycle shape changed at index {sample_idx}: expected {history_cycles_shape}, got {tuple(item_history_cycles.shape)}"
            )
        history_features[sample_idx] = item_history
        history_cycles[sample_idx] = item_history_cycles
        for hist_pos, cycle_id in enumerate(item_history_cycles):
            key = (str(item["cell_id"]), int(cycle_id))
            history_indices[sample_idx, hist_pos] = key_to_cycle_idx[key]
        meta_rows.append(
            {"prompt": item["prompt"], "cell_id": item["cell_id"], "cycle": int(item["cycle"])}
        )

    for array in (
        cycle_sequences,
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
        "layout": "cycle_sequence_history",
        "config": config,
        "cycle_scale": ds.cycle_scale,
        "sample_count": sample_count,
        "cycle_count": len(cycle_items),
        "cycle_sequence_shape": [int(args.resample_len), 6],
        "files": {
            "cycle_sequences": "cycle_sequences.npy",
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
    (tmp_path / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    if cache_path.exists():
        shutil.rmtree(cache_path)
    tmp_path.rename(cache_path)
    print(f"wrote cache: {cache_path}")
    return cache_path


def main() -> None:
    args = parse_args()
    for split in args.splits:
        precompute_sequence_split(args, split)


if __name__ == "__main__":
    main()
