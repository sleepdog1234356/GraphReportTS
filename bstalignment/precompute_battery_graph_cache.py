from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

try:
    from .data_battery_raw import (
        BatteryRawGraphDataset,
        battery_graph_cache_config,
        battery_graph_cache_path,
    )
except ImportError:
    from data_battery_raw import (
        BatteryRawGraphDataset,
        battery_graph_cache_config,
        battery_graph_cache_path,
    )


def parse_args():
    p = argparse.ArgumentParser(description="Precompute deterministic battery GraphReportTS map caches")
    p.add_argument("--dataset", choices=["mit", "calce", "xjtu"], required=True)
    p.add_argument("--data_root", type=str, default="bstalignment/data")
    p.add_argument("--cache_dir", type=str, default="runs/cache/battery_graph")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=["train", "val", "test", "all"])
    p.add_argument("--pred_len", type=int, default=20)
    p.add_argument("--resample_len", type=int, default=128)
    p.add_argument("--delay_dim", type=int, default=8)
    p.add_argument("--delay_lag", type=int, default=1)
    p.add_argument("--no_ic_dv", action="store_true")
    p.add_argument("--no_hankel_map", action="store_true")
    p.add_argument("--no_derivative_map", action="store_true")
    p.add_argument("--allow_summary_fallback", action="store_true")
    p.add_argument("--max_cycles", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=128)
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
    return all((cache_path / str(files.get(name, ""))).exists() for name in ["maps", "y", "mask", "horizon", "target_steps", "meta"])


def _write_meta(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _identity_collate(batch):
    rows = []
    for item in batch:
        rows.append(
            {
                "maps": item["maps"].detach().cpu().numpy().astype(np.float32, copy=False),
                "y": item["y"].detach().cpu().numpy().astype(np.float32, copy=False),
                "mask": item["mask"].detach().cpu().numpy().astype(np.bool_, copy=False),
                "horizon": int(item["horizon"]),
                "prompt": item["prompt"],
                "cell_id": item["cell_id"],
                "cycle": int(item["cycle"]),
                "target_steps": item["target_steps"].detach().cpu().numpy().astype(np.int64, copy=False),
            }
        )
    return rows


def precompute_split(args, split: str) -> Path:
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
        seed=args.seed,
        max_cycles=args.max_cycles,
    )
    sample_count = len(ds)
    if sample_count == 0:
        raise RuntimeError(f"Cannot precompute empty battery graph cache for dataset={args.dataset} split={split}")

    first = ds[0]
    maps_shape = tuple(int(v) for v in first["maps"].shape)
    target_width = int(args.pred_len) + 1
    tmp_path = cache_path.parent / f".{cache_path.name}.tmp"
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=False)

    maps = np.lib.format.open_memmap(tmp_path / "maps.npy", mode="w+", dtype=np.float32, shape=(sample_count, *maps_shape))
    y = np.lib.format.open_memmap(tmp_path / "y.npy", mode="w+", dtype=np.float32, shape=(sample_count, target_width))
    mask = np.lib.format.open_memmap(tmp_path / "mask.npy", mode="w+", dtype=np.bool_, shape=(sample_count, target_width))
    horizon = np.lib.format.open_memmap(tmp_path / "horizon.npy", mode="w+", dtype=np.int64, shape=(sample_count,))
    target_steps = np.lib.format.open_memmap(tmp_path / "target_steps.npy", mode="w+", dtype=np.int64, shape=(sample_count, target_width))
    y[:] = 0.0
    mask[:] = False
    target_steps[:] = 0
    meta_rows = []

    loader_kwargs = {
        "batch_size": int(args.batch_size),
        "shuffle": False,
        "num_workers": int(args.num_workers),
        "collate_fn": _identity_collate,
    }
    if int(args.num_workers) > 0:
        loader_kwargs["prefetch_factor"] = 2
    loader = DataLoader(ds, **loader_kwargs)
    total_batches = (sample_count + int(args.batch_size) - 1) // int(args.batch_size)

    write_idx = 0
    for batch in tqdm(loader, total=total_batches, desc=f"precompute {args.dataset}/{split}"):
        for item in batch:
            idx = write_idx
            write_idx += 1
            item_maps = np.asarray(item["maps"], dtype=np.float32)
            if tuple(item_maps.shape) != maps_shape:
                raise ValueError(f"Map shape changed at index {idx}: expected {maps_shape}, got {tuple(item_maps.shape)}")
            width = min(len(item["y"]), target_width)
            maps[idx] = item_maps
            y[idx, :width] = np.asarray(item["y"], dtype=np.float32)[:width]
            mask[idx, :width] = np.asarray(item["mask"], dtype=np.bool_)[:width]
            horizon[idx] = int(item["horizon"])
            target_steps[idx, :width] = np.asarray(item["target_steps"], dtype=np.int64)[:width]
            meta_rows.append({"prompt": item["prompt"], "cell_id": item["cell_id"], "cycle": int(item["cycle"])})
    if write_idx != sample_count:
        raise RuntimeError(f"Precompute wrote {write_idx} samples, expected {sample_count}")

    maps.flush()
    y.flush()
    mask.flush()
    horizon.flush()
    target_steps.flush()
    _write_meta(tmp_path / "meta.jsonl", meta_rows)
    manifest = {
        "config": config,
        "sample_count": sample_count,
        "maps_shape": maps_shape,
        "target_width": target_width,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": {
            "maps": "maps.npy",
            "y": "y.npy",
            "mask": "mask.npy",
            "horizon": "horizon.npy",
            "target_steps": "target_steps.npy",
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
