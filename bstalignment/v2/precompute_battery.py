"""Precompute deterministic GraphReportTS-v2 battery features.

MIT raw pickle cycles are the primary supported source.  XJTU is accepted only
through an already processed NPZ boundary containing per-cycle time/V/I/T and
separate SOH targets.  CALCE is intentionally rejected by the approved design.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import json
from pathlib import Path
import pickle
import subprocess
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from bstalignment.data_mit import CellRecord, _summary_to_frame

from .battery_cache import (
    BatteryCellFeatures,
    BatteryFeatureCache,
    BatteryOperatingContext,
    file_sha256,
)
from .battery_features import (
    BASE_FEATURE_NAMES,
    CURVE_AXIS_SCHEMA_VERSION,
    CURVE_AXIS_SHA256,
    CURVE_POINTS,
    DV_NORMALIZED_Q_RANGE,
    FEATURE_SCHEMA_VERSION,
    IC_VOLTAGE_RANGE_V,
    CycleFeatureResult,
    canonicalize_cycle,
    extract_cycle_features,
)


@dataclass(frozen=True)
class RawSensorCycle:
    observation_id: int
    time: np.ndarray
    voltage: np.ndarray
    current: np.ndarray
    temperature: np.ndarray
    soh_label: float


@dataclass(frozen=True)
class BatteryCellInput:
    cell_id: str
    cycles: Sequence[RawSensorCycle]
    operating_context: BatteryOperatingContext | None = None

    def __iter__(self):
        """Retain two-value unpacking used by earlier read-only callers."""

        yield self.cell_id
        yield self.cycles

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int | slice):
        pair = (self.cell_id, self.cycles)
        return pair[index]


XJTU_COMMON_CONTEXT = {
    "manufacturer": "LISHEN",
    "chemistry": "NCM523",
    "form_factor": "18650",
    "nominal_capacity_ah": 2.0,
    "nominal_voltage_v": 3.6,
    "voltage_window_v": (2.5, 4.2),
}

MIT_COMMON_CONTEXT = {
    "manufacturer": "A123",
    "model": "APR18650M1A",
    "chemistry": "LFP/graphite",
    "form_factor": "18650",
    "nominal_capacity_ah": 1.1,
    "nominal_voltage_v": 3.3,
    "voltage_window_v": (2.0, 3.6),
}
MIT_DISCHARGE_PROTOCOL = "4C CC-CV to 2.0V with C/50 cutoff"

XJTU_PROTOCOLS: tuple[tuple[str, str, str], ...] = (
    ("2C_battery-", "2C CC-CV to 4.2V", "1C constant-current to 2.5V"),
    ("3C_battery-", "3C CC-CV to 4.2V", "1C constant-current to 2.5V"),
    ("R2.5_battery-", "2C CC-CV to 4.2V", "0.5/1/2/3/5C cyclic discharge to 2.5V"),
    ("R3_battery-", "2C CC-CV to 4.2V", "0.5/1/2/3/5C cyclic discharge to 3.0V"),
    (
        "RW_battery-",
        "staged 0.5/1/3C CC-CV to 4.2V",
        "random-walk 2-8A for 2-6min; 3.0V safety cutoff",
    ),
    (
        "Sim_satellite_battery-",
        "2C CC-CV to 4.2V",
        "0.667C GEO-shadow with scheduled variable duration and DOD below 80%",
    ),
)


def xjtu_operating_context(path_or_stem: str | Path) -> BatteryOperatingContext | None:
    """Resolve official XJTU protocol metadata without exposing batch/cell identity."""

    stem = Path(path_or_stem).stem
    for prefix, charge_protocol, discharge_protocol in XJTU_PROTOCOLS:
        if stem.startswith(prefix):
            return BatteryOperatingContext(
                **XJTU_COMMON_CONTEXT,
                charge_protocol=charge_protocol,
                discharge_protocol=discharge_protocol,
            )
    return None


def mit_operating_context(charge_policy: object) -> BatteryOperatingContext:
    policy = " ".join(str(charge_policy).split()).strip()
    if not policy or policy.lower() in {"unknown", "none", "nan"}:
        policy = None
    return BatteryOperatingContext(
        **MIT_COMMON_CONTEXT,
        charge_protocol=policy,
        discharge_protocol=MIT_DISCHARGE_PROTOCOL,
    )


def load_mit_operating_contexts_streaming(
    data_root: str | Path,
    *,
    max_cells: int | None = None,
) -> dict[str, BatteryOperatingContext]:
    """Read MIT charge-policy metadata one pickle at a time without retaining raw cycles."""

    root = Path(data_root).expanduser().resolve()
    contexts: dict[str, BatteryOperatingContext] = {}
    for path in sorted(root.glob("*.pkl")):
        with path.open("rb") as handle:
            try:
                batch = pickle.load(handle)
            except UnicodeDecodeError:
                handle.seek(0)
                batch = pickle.load(handle, encoding="latin1")
        if not isinstance(batch, Mapping):
            raise ValueError(f"MIT pickle {path} did not contain a cell mapping")
        for raw_cell_id, cell in batch.items():
            if not isinstance(cell, Mapping):
                continue
            summary = cell.get("summary", cell.get("Summary"))
            if not isinstance(summary, Mapping):
                continue
            policy = cell.get("charge_policy", cell.get("policy", "unknown"))
            contexts[f"{path.stem}_{raw_cell_id}"] = mit_operating_context(policy)
            if max_cells is not None and len(contexts) >= max_cells:
                break
        cell = None
        summary = None
        policy = None
        del batch
        if max_cells is not None and len(contexts) >= max_cells:
            break
    if not contexts:
        raise RuntimeError("no MIT operating context could be parsed from the selected pickle files")
    return contexts


def _array(mapping: Mapping[str, Any], keys: Sequence[str]) -> np.ndarray:
    for key in keys:
        if key in mapping:
            return np.asarray(mapping[key], dtype=np.float64).reshape(-1)
    return np.empty(0, dtype=np.float64)


def _extract_one(raw: RawSensorCycle) -> tuple[int, float, CycleFeatureResult] | None:
    try:
        signals = canonicalize_cycle(raw.time, raw.voltage, raw.current, raw.temperature)
        return raw.observation_id, raw.soh_label, extract_cycle_features(signals)
    except (ValueError, TypeError, FloatingPointError):
        return None


def build_cell_features(
    cell_id: str,
    cycles: Sequence[RawSensorCycle],
    *,
    workers: int = 0,
    executor: ProcessPoolExecutor | None = None,
    operating_context: BatteryOperatingContext | None = None,
) -> BatteryCellFeatures:
    """Extract and stack one cell; failed malformed cycles are skipped."""

    if executor is not None:
        extracted = list(executor.map(_extract_one, cycles, chunksize=max(len(cycles) // (workers * 8), 1)))
    elif workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            extracted = list(executor.map(_extract_one, cycles, chunksize=max(len(cycles) // (workers * 8), 1)))
    else:
        extracted = [_extract_one(cycle) for cycle in cycles]
    successful = [item for item in extracted if item is not None]
    successful.sort(key=lambda item: item[0])
    if not successful:
        raise ValueError(f"{cell_id}: no valid time/V/I cycles could be extracted")
    ids = np.asarray([item[0] for item in successful], dtype=np.int64)
    labels = np.asarray([item[1] for item in successful], dtype=np.float32)
    results = [item[2] for item in successful]
    return BatteryCellFeatures(
        cell_id=cell_id,
        observation_ids=ids,
        time_coverage=np.asarray([result.time_coverage for result in results], dtype=np.float32),
        base_values=np.stack([result.values for result in results]).astype(np.float32),
        base_observed_mask=np.stack([result.observed_mask for result in results]).astype(bool),
        base_reliability=np.stack([result.reliability for result in results]).astype(np.float32),
        ic_curve=np.stack([result.ic_curve.y for result in results]).astype(np.float32),
        ic_curve_axis=np.stack([result.ic_curve.x for result in results]).astype(np.float32),
        ic_curve_mask=np.stack([result.ic_curve.observed_mask for result in results]).astype(bool),
        ic_quality=np.asarray([result.ic_curve.quality for result in results], dtype=np.float32),
        dv_curve=np.stack([result.dv_curve.y for result in results]).astype(np.float32),
        dv_curve_axis=np.stack([result.dv_curve.x for result in results]).astype(np.float32),
        dv_curve_mask=np.stack([result.dv_curve.observed_mask for result in results]).astype(bool),
        dv_quality=np.asarray([result.dv_curve.quality for result in results], dtype=np.float32),
        soh_labels=labels,
        operating_context=operating_context,
    )


def _mit_soh_labels(record: CellRecord) -> np.ndarray:
    qd = record.summary["QD"].to_numpy(dtype=np.float64)
    valid = qd[np.isfinite(qd) & (qd > 0)]
    reference = float(np.median(valid[: min(10, len(valid))])) if len(valid) else np.nan
    if not np.isfinite(reference) or reference <= 0:
        return np.full(len(qd), np.nan, dtype=np.float32)
    return (qd / reference).astype(np.float32)


def _mit_raw_cycles(record: CellRecord, max_cycles: int | None) -> list[RawSensorCycle]:
    group = record.raw.get("cycles", record.raw.get("cycle", record.raw.get("Cycle", {})))
    if not isinstance(group, Mapping):
        raise ValueError(f"{record.cell_id}: MIT raw cycle dictionary is missing")
    labels = _mit_soh_labels(record)
    summary_cycles = record.summary["cycle"].to_numpy(dtype=np.float64)
    ordered: list[tuple[int, Any]] = []
    for key, cycle in group.items():
        try:
            row = int(key)
        except (TypeError, ValueError):
            continue
        if isinstance(cycle, Mapping):
            ordered.append((row, cycle))
    ordered.sort(key=lambda item: item[0])
    if max_cycles is not None:
        ordered = ordered[:max_cycles]
    cycles: list[RawSensorCycle] = []
    for row, cycle in ordered:
        if row < 0 or row >= len(labels):
            continue
        observation_id = int(summary_cycles[row]) if np.isfinite(summary_cycles[row]) else row + 1
        cycles.append(
            RawSensorCycle(
                observation_id=observation_id,
                time=_array(cycle, ("t", "time", "Time")),
                voltage=_array(cycle, ("V", "voltage", "Voltage")),
                current=_array(cycle, ("I", "current", "Current")),
                temperature=_array(cycle, ("T", "temperature", "Temperature", "Temp")),
                soh_label=float(labels[row]),
            )
        )
    return cycles


def load_mit_inputs(
    data_root: str | Path,
    *,
    max_cells: int | None = None,
    max_cycles: int | None = None,
) -> tuple[list[BatteryCellInput], list[Path]]:
    data_root = Path(data_root).expanduser().resolve()
    source_files = sorted(data_root.glob("*.pkl"))
    records, used_files = _load_mit_records_compatible(source_files, max_cells=max_cells)
    return [
        BatteryCellInput(
            record.cell_id,
            _mit_raw_cycles(record, max_cycles),
            mit_operating_context(record.charge_policy),
        )
        for record in records
    ], used_files


def _load_mit_records_compatible(
    source_files: Sequence[Path],
    *,
    max_cells: int | None = None,
) -> tuple[list[CellRecord], list[Path]]:
    """Parse the repository MIT pickles while accepting 1-element scalar arrays.

    The v1 loader calls ``float(array)`` for ``cycle_life``; NumPy 2 rejects
    that when the array is one-dimensional.  Keeping the compatibility shim in
    v2 avoids changing the established v1 data path.
    """

    records: list[CellRecord] = []
    used_files: list[Path] = []
    for path in source_files:
        used_files.append(path)
        with path.open("rb") as handle:
            try:
                batch = pickle.load(handle)
            except UnicodeDecodeError:
                handle.seek(0)
                batch = pickle.load(handle, encoding="latin1")
        if not isinstance(batch, Mapping):
            raise ValueError(f"MIT pickle {path} did not contain a cell mapping")
        for raw_cell_id, cell in batch.items():
            if not isinstance(cell, Mapping):
                continue
            summary = cell.get("summary", cell.get("Summary"))
            if not isinstance(summary, Mapping):
                continue
            frame = _summary_to_frame(dict(summary))
            raw_life = np.asarray(
                cell.get("cycle_life", cell.get("cyclelife", cell.get("life", len(frame))))
            ).reshape(-1)
            cycle_life = float(raw_life[0]) if raw_life.size else float(len(frame))
            charge_policy = str(cell.get("charge_policy", cell.get("policy", "unknown")))
            records.append(
                CellRecord(
                    cell_id=f"{path.stem}_{raw_cell_id}",
                    charge_policy=charge_policy,
                    cycle_life=cycle_life,
                    summary=frame,
                    raw=dict(cell),
                )
            )
            if max_cells is not None and len(records) >= max_cells:
                break
        if max_cells is not None and len(records) >= max_cells:
            break
    if not records:
        raise RuntimeError("no MIT cells could be parsed from the selected pickle files")
    return records, used_files


def _cycle_row(array: np.ndarray, index: int, count: int, name: str) -> np.ndarray:
    if array.ndim == 1:
        return array
    if array.ndim == 2 and array.shape[0] == count:
        return array[index]
    raise ValueError(f"processed XJTU {name} must be [L] or [N,L]")


def load_xjtu_inputs(
    processed_root: str | Path,
    *,
    max_cells: int | None = None,
    max_cycles: int | None = None,
) -> tuple[list[BatteryCellInput], list[Path]]:
    root = Path(processed_root).expanduser().resolve()
    files = [root] if root.is_file() else sorted(root.glob("*.npz"))
    if max_cells is not None:
        files = files[:max_cells]
    cells: list[BatteryCellInput] = []
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            required = {"cycle_id", "soh", "time", "voltage", "current", "temperature"}
            missing = sorted(required - set(data.files))
            if missing:
                raise ValueError(f"processed XJTU NPZ {path} is missing generic V/I/T/time keys: {missing}")
            ids = np.asarray(data["cycle_id"]).reshape(-1)
            labels = np.asarray(data["soh"], dtype=np.float64).reshape(-1)
            count = min(len(ids), len(labels))
            if max_cycles is not None:
                count = min(count, max_cycles)
            arrays = {name: np.asarray(data[name], dtype=np.float64) for name in ("time", "voltage", "current", "temperature")}
            cycles = [
                RawSensorCycle(
                    observation_id=int(ids[index]),
                    time=_cycle_row(arrays["time"], index, len(ids), "time"),
                    voltage=_cycle_row(arrays["voltage"], index, len(ids), "voltage"),
                    current=_cycle_row(arrays["current"], index, len(ids), "current"),
                    temperature=_cycle_row(arrays["temperature"], index, len(ids), "temperature"),
                    soh_label=float(labels[index]),
                )
                for index in range(count)
            ]
        cells.append(BatteryCellInput(path.stem, cycles, xjtu_operating_context(path)))
    if not cells:
        raise FileNotFoundError(f"no processed XJTU NPZ files found under {root}")
    return cells, files


def synthetic_inputs(
    cycles: int = 96,
    cells: int = 3,
) -> tuple[list[BatteryCellInput], list[Path]]:
    """Create a multi-cell cache source large enough for 32+32+20 windows."""

    if cycles < 1 or cells < 1:
        raise ValueError("synthetic cycles and cells must be positive")
    synthetic_cells: list[BatteryCellInput] = []
    for cell_index in range(cells):
        raw_cycles: list[RawSensorCycle] = []
        current_scale = 1.0 + 0.08 * cell_index
        temperature_offset = 0.7 * cell_index
        degradation = 0.0013 + 0.00015 * cell_index
        for cycle_index in range(cycles):
            charge_t = np.linspace(0.0, 3000.0, 180)
            discharge_t = np.linspace(3010.0, 5000.0, 100)
            time = np.concatenate((charge_t, discharge_t))
            charge_v = 3.0 + (1.18 - 0.00025 * cycle_index) * (1.0 - np.exp(-charge_t / 850.0))
            discharge_v = np.linspace(charge_v[-1], 3.0 - 0.0001 * cycle_index, len(discharge_t))
            voltage = np.concatenate((charge_v, discharge_v))
            current = current_scale * np.concatenate(
                (
                    np.where(
                        charge_t < 2200.0,
                        1.5,
                        1.5 * np.exp(-(charge_t - 2200.0) / 420.0),
                    ),
                    np.full(len(discharge_t), -1.0),
                )
            )
            temperature = (
                25.0
                + temperature_offset
                + (3.0 + 0.002 * cycle_index) * (time / time[-1])
                + 0.1 * np.sin(time / 300.0)
            )
            raw_cycles.append(
                RawSensorCycle(
                    observation_id=cycle_index + 1,
                    time=time,
                    voltage=voltage,
                    current=current,
                    temperature=temperature,
                    soh_label=1.0 - cycle_index * degradation,
                )
            )
        synthetic_cells.append(BatteryCellInput(f"synthetic_cell_{cell_index + 1}", raw_cycles))
    return synthetic_cells, []


def _repository_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def precompute(
    dataset: str,
    data_root: str | Path,
    output: str | Path,
    *,
    workers: int = 0,
    max_cells: int | None = None,
    max_cycles: int | None = None,
    split_seed: int = 42,
    compressed: bool = True,
    overwrite: bool = False,
) -> BatteryFeatureCache:
    if workers < 0:
        raise ValueError("workers must be non-negative")
    if max_cells is not None and max_cells <= 0:
        raise ValueError("max_cells must be positive")
    if max_cycles is not None and max_cycles <= 0:
        raise ValueError("max_cycles must be positive")
    name = dataset.strip().lower()
    if name == "calce":
        raise ValueError("CALCE is excluded from the approved GraphReportTS-v2 design")
    if name == "mit":
        inputs, source_files = load_mit_inputs(data_root, max_cells=max_cells, max_cycles=max_cycles)
    elif name == "xjtu":
        inputs, source_files = load_xjtu_inputs(data_root, max_cells=max_cells, max_cycles=max_cycles)
    elif name == "synthetic":
        inputs, source_files = synthetic_inputs(cycles=max_cycles or 96, cells=max_cells or 3)
    else:
        raise ValueError("dataset must be mit, xjtu, or synthetic; CALCE is not supported")
    if workers > 1:
        # Keep one worker pool alive across all cells; repeatedly spawning a pool
        # is expensive on Windows and defeats the purpose of preprocessing once.
        with ProcessPoolExecutor(max_workers=workers) as executor:
            cells = {
                source.cell_id: build_cell_features(
                    source.cell_id,
                    source.cycles,
                    workers=workers,
                    executor=executor,
                    operating_context=source.operating_context,
                )
                for source in inputs
            }
    else:
        cells = {
            source.cell_id: build_cell_features(
                source.cell_id,
                source.cycles,
                workers=workers,
                operating_context=source.operating_context,
            )
            for source in inputs
        }
    provenance = {
        "dataset": name,
        "source_file_checksums": {str(path): file_sha256(path) for path in source_files},
        "source_boundary": "single-cycle time/voltage/current/temperature; SOH target separate",
    }
    algorithm_params = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "curve_points": CURVE_POINTS,
        "curve_axis_schema_version": CURVE_AXIS_SCHEMA_VERSION,
        "curve_axis_sha256": CURVE_AXIS_SHA256,
        "ic_voltage_range_v": list(IC_VOLTAGE_RANGE_V),
        "dv_normalized_q_range": list(DV_NORMALIZED_Q_RANGE),
        "base_feature_count": len(BASE_FEATURE_NAMES),
        "minimum_cycle_samples": 4,
        "ic_dv_savgol_window": 11,
        "ic_dv_savgol_polyorder": 3,
    }
    return BatteryFeatureCache.create(
        output,
        cells,
        provenance,
        algorithm_params=algorithm_params,
        split_seed=split_seed,
        repository_commit=_repository_commit(),
        compressed=compressed,
        overwrite=overwrite,
    )


def source_operating_contexts(
    dataset: str,
    data_root: str | Path,
    *,
    max_cells: int | None = None,
) -> dict[str, BatteryOperatingContext | None]:
    """Read only lightweight source metadata for an existing feature cache."""

    name = dataset.strip().lower()
    root = Path(data_root).expanduser().resolve()
    if name == "mit":
        return load_mit_operating_contexts_streaming(root, max_cells=max_cells)
    if name == "xjtu":
        files = sorted(root.glob("*.npz"))
        if max_cells is not None:
            files = files[:max_cells]
        return {path.stem: xjtu_operating_context(path) for path in files}
    raise ValueError("metadata-only enrichment supports mit or xjtu")


def enrich_existing_cache_operating_context(
    cache_dir: str | Path,
    dataset: str,
    data_root: str | Path,
    *,
    max_cells: int | None = None,
) -> BatteryFeatureCache:
    cache = BatteryFeatureCache.open(cache_dir)
    contexts = source_operating_contexts(dataset, data_root, max_cells=max_cells)
    selected = {cell_id: contexts[cell_id] for cell_id in cache.cell_ids if cell_id in contexts}
    if not selected:
        raise ValueError("source metadata did not match any cells in the existing cache")
    missing = sorted(set(cache.cell_ids) - set(selected))
    if missing:
        raise ValueError(f"source metadata is missing cache cells: {missing}")
    return cache.enrich_operating_context(selected)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="mit, xjtu, or synthetic; calce is rejected")
    parser.add_argument("--data-root", default="data/battery/mit")
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=0, help="parallel per-cycle CPU workers; 0/1 is serial")
    parser.add_argument("--max-cells", type=int)
    parser.add_argument("--max-cycles", type=int)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--no-compression", action="store_true", help="trade disk space for faster preprocessing/load")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="atomically enrich an existing cache manifest without recomputing NPZ features",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.workers < 0:
        raise ValueError("workers must be non-negative")
    if args.metadata_only:
        cache = enrich_existing_cache_operating_context(
            args.output,
            args.dataset,
            args.data_root,
            max_cells=args.max_cells,
        )
    else:
        cache = precompute(
            args.dataset,
            args.data_root,
            args.output,
            workers=args.workers,
            max_cells=args.max_cells,
            max_cycles=args.max_cycles,
            split_seed=args.split_seed,
            compressed=not args.no_compression,
            overwrite=args.overwrite,
        )
    print(
        json.dumps(
            {
                "cache": str(cache.root),
                "manifest_sha256": cache.manifest_hash,
                "cells": len(cache),
                "observations": sum(cache.manifest["cells"][cell]["observations"] for cell in cache.cell_ids),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
