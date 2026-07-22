"""Versioned on-disk cache for deterministic GTR battery inputs."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterator, Mapping

import numpy as np

from .battery_features import (
    BASE_FEATURE_NAMES,
    CURVE_AXIS_SCHEMA_VERSION,
    CURVE_AXIS_SHA256,
    CURVE_POINTS,
    DV_NORMALIZED_Q_AXIS,
    DV_NORMALIZED_Q_RANGE,
    FEATURE_SCHEMA_VERSION,
    IC_VOLTAGE_AXIS,
    IC_VOLTAGE_RANGE_V,
)


CACHE_SCHEMA = "battery_features"
CACHE_FORMAT_VERSION = 2
OPERATING_CONTEXT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class BatteryOperatingContext:
    """JSON-safe, non-identifying cell specification and test protocol."""

    manufacturer: str | None = None
    model: str | None = None
    chemistry: str | None = None
    form_factor: str | None = None
    nominal_capacity_ah: float | None = None
    nominal_voltage_v: float | None = None
    voltage_window_v: tuple[float, float] | None = None
    charge_protocol: str | None = None
    discharge_protocol: str | None = None
    source: str = "declared"

    _FIELDS = frozenset(
        {
            "manufacturer",
            "model",
            "chemistry",
            "form_factor",
            "nominal_capacity_ah",
            "nominal_voltage_v",
            "voltage_window_v",
            "charge_protocol",
            "discharge_protocol",
            "source",
        }
    )

    def __post_init__(self) -> None:
        for name in (
            "manufacturer",
            "model",
            "chemistry",
            "form_factor",
            "charge_protocol",
            "discharge_protocol",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            if not isinstance(value, str):
                raise TypeError(f"operating_context.{name} must be a string or None")
            cleaned = " ".join(value.split()).strip()
            object.__setattr__(self, name, cleaned or None)
        if self.source not in {"declared", "sensor_inferred"}:
            raise ValueError("operating_context.source must be declared or sensor_inferred")
        for name in ("nominal_capacity_ah", "nominal_voltage_v"):
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or not np.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"operating_context.{name} must be finite and positive")
            object.__setattr__(self, name, float(value))
        if self.voltage_window_v is not None:
            if len(self.voltage_window_v) != 2:
                raise ValueError("operating_context.voltage_window_v must contain [low, high]")
            low, high = (float(value) for value in self.voltage_window_v)
            if not np.isfinite((low, high)).all() or low >= high:
                raise ValueError("operating_context.voltage_window_v must be finite and increasing")
            object.__setattr__(self, "voltage_window_v", (low, high))

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for name in sorted(self._FIELDS):
            value = getattr(self, name)
            if value is None:
                continue
            payload[name] = list(value) if name == "voltage_window_v" else value
        return payload

    @classmethod
    def from_json(cls, payload: Mapping[str, Any] | None) -> "BatteryOperatingContext | None":
        if payload is None:
            return None
        if not isinstance(payload, Mapping):
            raise TypeError("operating_context must be a JSON object or None")
        unsupported = sorted(set(payload) - cls._FIELDS)
        if unsupported:
            raise ValueError(f"unsupported operating-context fields: {unsupported}")
        return cls(**dict(payload))


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class BatteryCellFeatures:
    cell_id: str
    observation_ids: np.ndarray
    time_coverage: np.ndarray
    base_values: np.ndarray
    base_observed_mask: np.ndarray
    base_reliability: np.ndarray
    ic_curve: np.ndarray
    ic_curve_axis: np.ndarray
    ic_curve_mask: np.ndarray
    ic_quality: np.ndarray
    dv_curve: np.ndarray
    dv_curve_axis: np.ndarray
    dv_curve_mask: np.ndarray
    dv_quality: np.ndarray
    soh_labels: np.ndarray
    operating_context: BatteryOperatingContext | None = None

    @property
    def base_features(self) -> np.ndarray:
        """Compatibility alias used by cache consumers and provenance checks."""

        return self.base_values

    def validate(self) -> None:
        if self.operating_context is not None and not isinstance(self.operating_context, BatteryOperatingContext):
            raise TypeError(f"{self.cell_id}: operating_context must be BatteryOperatingContext or None")
        n = int(np.asarray(self.observation_ids).reshape(-1).size)
        expected = {
            "time_coverage": (n,),
            "base_values": (n, 50),
            "base_observed_mask": (n, 50),
            "base_reliability": (n, 50),
            "ic_curve": (n, CURVE_POINTS),
            "ic_curve_axis": (n, CURVE_POINTS),
            "ic_curve_mask": (n, CURVE_POINTS),
            "ic_quality": (n,),
            "dv_curve": (n, CURVE_POINTS),
            "dv_curve_axis": (n, CURVE_POINTS),
            "dv_curve_mask": (n, CURVE_POINTS),
            "dv_quality": (n,),
            "soh_labels": (n,),
        }
        for name, shape in expected.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape:
                raise ValueError(f"{self.cell_id}: {name} must have shape {shape}, got {value.shape}")
        if np.asarray(self.base_observed_mask).dtype != np.bool_:
            raise ValueError(f"{self.cell_id}: base_observed_mask must be boolean")
        if np.asarray(self.ic_curve_mask).dtype != np.bool_ or np.asarray(self.dv_curve_mask).dtype != np.bool_:
            raise ValueError(f"{self.cell_id}: curve masks must be boolean")
        for name in ("base_reliability", "ic_quality", "dv_quality"):
            value = np.asarray(getattr(self, name))
            if not np.isfinite(value).all() or np.any((value < 0) | (value > 1)):
                raise ValueError(f"{self.cell_id}: {name} must be finite and within 0..1")
        observed = np.asarray(self.base_values)[np.asarray(self.base_observed_mask)]
        if not np.isfinite(observed).all():
            raise ValueError(f"{self.cell_id}: observed base features must be finite")
        for prefix in ("ic", "dv"):
            curve_mask = np.asarray(getattr(self, f"{prefix}_curve_mask"))
            curve = np.asarray(getattr(self, f"{prefix}_curve"))
            axis = np.asarray(getattr(self, f"{prefix}_curve_axis"))
            if not np.isfinite(curve[curve_mask]).all() or not np.isfinite(axis[curve_mask]).all():
                raise ValueError(f"{self.cell_id}: observed {prefix} curve and physical axis must be finite")
            for row in range(n):
                selected_axis = axis[row, curve_mask[row]]
                if len(selected_axis) > 1 and np.any(np.diff(selected_axis) <= 0):
                    raise ValueError(f"{self.cell_id}: {prefix} physical axis must increase")
            expected_axis = IC_VOLTAGE_AXIS if prefix == "ic" else DV_NORMALIZED_Q_AXIS
            if not np.allclose(axis, expected_axis[None, :], rtol=0.0, atol=1e-7):
                axis_name = "2.0--5.0 V" if prefix == "ic" else "normalized-Q 0.0--1.0"
                raise ValueError(f"{self.cell_id}: {prefix} curve axis must use fixed {axis_name} grid")
        if n > 1 and np.any(np.diff(np.asarray(self.observation_ids, dtype=np.int64)) <= 0):
            raise ValueError(f"{self.cell_id}: observation_ids must be strictly increasing")
        coverage = np.asarray(self.time_coverage)
        if not np.isfinite(coverage).all() or np.any(coverage < 0):
            raise ValueError(f"{self.cell_id}: time_coverage must be finite and non-negative")


class BatteryFeatureCache:
    """Lazy per-cell reader/writer for the deterministic battery-gtr cache."""

    def __init__(self, root: Path, manifest: dict[str, Any]) -> None:
        self.root = root
        self.manifest = manifest
        self.manifest_hash = str(manifest["manifest_sha256"])

    @property
    def cell_ids(self) -> tuple[str, ...]:
        return tuple(self.manifest["cells"].keys())

    def __len__(self) -> int:
        return len(self.cell_ids)

    def __iter__(self) -> Iterator[str]:
        return iter(self.cell_ids)

    @staticmethod
    def expected_config(**overrides: Any) -> dict[str, Any]:
        config: dict[str, Any] = {
            "schema": CACHE_SCHEMA,
            "format_version": CACHE_FORMAT_VERSION,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "base_feature_names": list(BASE_FEATURE_NAMES),
            "curve_points": CURVE_POINTS,
            "curve_axis_schema_version": CURVE_AXIS_SCHEMA_VERSION,
            "curve_axis_sha256": CURVE_AXIS_SHA256,
            "ic_axis": {
                "coordinate": "voltage",
                "unit": "V",
                "range": list(IC_VOLTAGE_RANGE_V),
                "values": [float(value) for value in IC_VOLTAGE_AXIS],
            },
            "dv_axis": {
                "coordinate": "normalized_q",
                "unit": "fraction",
                "range": list(DV_NORMALIZED_Q_RANGE),
                "values": [float(value) for value in DV_NORMALIZED_Q_AXIS],
            },
        }
        config.update(overrides)
        return config

    @classmethod
    def open(
        cls,
        root: str | Path,
        expected_config: Mapping[str, Any] | None = None,
    ) -> "BatteryFeatureCache":
        root = Path(root).expanduser().resolve()
        manifest_path = root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"BatteryGTR cache manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        stored_hash = manifest.get("manifest_sha256")
        unhashed = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
        actual_hash = sha256(_canonical_json(unhashed)).hexdigest()
        if stored_hash != actual_hash:
            raise ValueError("BatteryGTR cache manifest checksum mismatch")
        expected = cls.expected_config()
        if expected_config:
            expected.update(dict(expected_config))
        for key, expected_value in expected.items():
            if manifest.get(key) != expected_value:
                raise ValueError(
                    f"BatteryGTR feature schema mismatch: {key} expected {expected_value!r}, "
                    f"got {manifest.get(key)!r}"
                )
        cells = manifest.get("cells")
        if not isinstance(cells, dict) or not cells:
            raise ValueError("BatteryGTR cache has no cells")
        for cell_id, info in cells.items():
            if not isinstance(info, dict) or "file" not in info:
                raise ValueError(f"BatteryGTR cache cell entry is invalid: {cell_id}")
            path = (root / info["file"]).resolve()
            if path.parent != root or not path.exists():
                raise ValueError(f"BatteryGTR cache cell file is missing or unsafe: {cell_id}")
            BatteryOperatingContext.from_json(info.get("operating_context"))
        return cls(root, manifest)

    @classmethod
    def create(
        cls,
        root: str | Path,
        cells: Mapping[str, BatteryCellFeatures],
        provenance: Mapping[str, Any],
        *,
        algorithm_params: Mapping[str, Any] | None = None,
        split_seed: int = 42,
        repository_commit: str = "unknown",
        compressed: bool = True,
        overwrite: bool = False,
    ) -> "BatteryFeatureCache":
        root = Path(root).expanduser().resolve()
        if not cells:
            raise ValueError("cannot create an empty BatteryGTR cache")
        if root.exists() and any(root.iterdir()) and not overwrite:
            raise FileExistsError(f"cache directory is not empty: {root}")
        root.parent.mkdir(parents=True, exist_ok=True)
        temp_root = Path(tempfile.mkdtemp(prefix=f".{root.name}.tmp-", dir=root.parent))
        try:
            cell_manifest: dict[str, Any] = {}
            saver = np.savez_compressed if compressed else np.savez
            for index, (cell_id, cell) in enumerate(cells.items()):
                if cell.cell_id != cell_id:
                    raise ValueError(f"cache key {cell_id!r} differs from payload cell_id {cell.cell_id!r}")
                cell.validate()
                filename = f"cell-{index:05d}-{sha256(cell_id.encode('utf-8')).hexdigest()[:12]}.npz"
                saver(
                    temp_root / filename,
                    observation_ids=np.asarray(cell.observation_ids, dtype=np.int64),
                    time_coverage=np.asarray(cell.time_coverage, dtype=np.float32),
                    base_values=np.asarray(cell.base_values, dtype=np.float32),
                    base_observed_mask=np.asarray(cell.base_observed_mask, dtype=bool),
                    base_reliability=np.asarray(cell.base_reliability, dtype=np.float32),
                    ic_curve=np.asarray(cell.ic_curve, dtype=np.float32),
                    ic_curve_axis=np.asarray(cell.ic_curve_axis, dtype=np.float32),
                    ic_curve_mask=np.asarray(cell.ic_curve_mask, dtype=bool),
                    ic_quality=np.asarray(cell.ic_quality, dtype=np.float32),
                    dv_curve=np.asarray(cell.dv_curve, dtype=np.float32),
                    dv_curve_axis=np.asarray(cell.dv_curve_axis, dtype=np.float32),
                    dv_curve_mask=np.asarray(cell.dv_curve_mask, dtype=bool),
                    dv_quality=np.asarray(cell.dv_quality, dtype=np.float32),
                    soh_labels=np.asarray(cell.soh_labels, dtype=np.float32),
                )
                cell_manifest[cell_id] = {
                    "file": filename,
                    "observations": len(cell.observation_ids),
                }
                if cell.operating_context is not None:
                    cell_manifest[cell_id]["operating_context"] = cell.operating_context.to_json()
            manifest: dict[str, Any] = {
                **cls.expected_config(),
                "algorithm_params": dict(algorithm_params or {}),
                "split_seed": int(split_seed),
                "repository_commit": str(repository_commit),
                "provenance": dict(provenance),
                "compressed": bool(compressed),
                "learned_curve_residuals_cached": False,
                "operating_context_schema_version": OPERATING_CONTEXT_SCHEMA_VERSION,
                "cells": cell_manifest,
            }
            if any("soh" in key.lower() for key in manifest if key != "cells"):
                raise RuntimeError("SOH must not appear in the cache manifest predictor schema")
            manifest["manifest_sha256"] = sha256(_canonical_json(manifest)).hexdigest()
            (temp_root / "manifest.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if root.exists():
                if not overwrite and any(root.iterdir()):
                    raise FileExistsError(f"cache directory is not empty: {root}")
                shutil.rmtree(root)
            os.replace(temp_root, root)
        except Exception:
            shutil.rmtree(temp_root, ignore_errors=True)
            raise
        return cls.open(root)

    def enrich_operating_context(
        self,
        contexts: Mapping[str, BatteryOperatingContext | Mapping[str, Any] | None],
    ) -> "BatteryFeatureCache":
        """Atomically update only manifest metadata, preserving cached NPZ arrays."""

        unknown = sorted(set(contexts) - set(self.cell_ids))
        if unknown:
            raise KeyError(f"operating context supplied for unknown cache cells: {unknown}")
        manifest = json.loads(json.dumps(self.manifest, ensure_ascii=False))
        manifest.pop("manifest_sha256", None)
        for cell_id, raw_context in contexts.items():
            context = (
                raw_context
                if isinstance(raw_context, BatteryOperatingContext)
                else BatteryOperatingContext.from_json(raw_context)
            )
            info = manifest["cells"][cell_id]
            if context is None:
                info.pop("operating_context", None)
            else:
                info["operating_context"] = context.to_json()
        manifest["operating_context_schema_version"] = OPERATING_CONTEXT_SCHEMA_VERSION
        manifest["manifest_sha256"] = sha256(_canonical_json(manifest)).hexdigest()
        temporary = self.root / f".manifest-{os.getpid()}.tmp"
        try:
            temporary.write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, self.root / "manifest.json")
        finally:
            temporary.unlink(missing_ok=True)
        return type(self).open(self.root)

    def load_cell(self, cell_id: str, mmap_mode: str | None = None) -> BatteryCellFeatures:
        if cell_id not in self.manifest["cells"]:
            raise KeyError(f"unknown battery cache cell: {cell_id}")
        path = self.root / self.manifest["cells"][cell_id]["file"]
        with np.load(path, allow_pickle=False, mmap_mode=mmap_mode) as arrays:
            cell = BatteryCellFeatures(
                cell_id=cell_id,
                observation_ids=np.asarray(arrays["observation_ids"]),
                time_coverage=np.asarray(arrays["time_coverage"]),
                base_values=np.asarray(arrays["base_values"]),
                base_observed_mask=np.asarray(arrays["base_observed_mask"]),
                base_reliability=np.asarray(arrays["base_reliability"]),
                ic_curve=np.asarray(arrays["ic_curve"]),
                ic_curve_axis=np.asarray(arrays["ic_curve_axis"]),
                ic_curve_mask=np.asarray(arrays["ic_curve_mask"]),
                ic_quality=np.asarray(arrays["ic_quality"]),
                dv_curve=np.asarray(arrays["dv_curve"]),
                dv_curve_axis=np.asarray(arrays["dv_curve_axis"]),
                dv_curve_mask=np.asarray(arrays["dv_curve_mask"]),
                dv_quality=np.asarray(arrays["dv_quality"]),
                soh_labels=np.asarray(arrays["soh_labels"]),
                operating_context=BatteryOperatingContext.from_json(
                    self.manifest["cells"][cell_id].get("operating_context")
                ),
            )
        cell.validate()
        return cell
