from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from ..battery_protocol import split_processed_items
from .battery_cache import BatteryFeatureCache
from .contracts import BatteryRawBatch, validate_battery_raw_batch
from .prompts import BATTERY_PROMPT_METRICS, build_battery_prompt, fit_battery_prompt_thresholds


BATTERY_PROMPT_CYCLES = 32
BATTERY_NUMERIC_CYCLES = 32
BATTERY_PREDICTION_CYCLES = 20
BATTERY_MIN_VALID_NUMERIC_CYCLES = 24


def battery_window_key(
    *,
    cache_hash: str,
    split: str,
    cell_id: str,
    origin_row: int,
    forecast_observation_id: int,
) -> str:
    """Return a stable cache key for one split-local forecast window."""

    payload = (
        "battery-gtr-window-v1",
        str(cache_hash),
        str(split),
        str(cell_id),
        int(origin_row),
        int(forecast_observation_id),
    )
    digest = hashlib.sha256("\0".join(str(value) for value in payload).encode("utf-8")).hexdigest()
    return f"{split}:{cell_id}:{int(forecast_observation_id)}:{digest[:24]}"


def _base_values(cell: Any) -> np.ndarray:
    value = getattr(cell, "base_values", None)
    if value is None:
        value = getattr(cell, "base_features", None)
    if value is None:
        raise ValueError("battery cache cell does not expose 50 deterministic features")
    return np.asarray(value, dtype=np.float32)


@dataclass(frozen=True)
class BatteryFeatureScaler:
    median: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    schema_hash: str

    @classmethod
    def fit(cls, cells: Iterable[Any]) -> "BatteryFeatureScaler":
        values: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        for cell in cells:
            value = _base_values(cell)
            mask = np.asarray(cell.base_observed_mask, dtype=bool)
            if value.ndim != 2 or value.shape[1] != 50 or mask.shape != value.shape:
                raise ValueError("battery scaler requires cell arrays [N,50]")
            values.append(value)
            masks.append(mask & np.isfinite(value))
        if not values:
            raise ValueError("cannot fit battery feature scaler without training cells")
        joined = np.concatenate(values, axis=0).astype(np.float64)
        observed = np.concatenate(masks, axis=0)
        median = np.zeros(joined.shape[1], dtype=np.float64)
        for feature_index in range(joined.shape[1]):
            selected = joined[observed[:, feature_index], feature_index]
            if selected.size:
                median[feature_index] = float(np.median(selected))
        filled = np.where(observed, joined, median[None, :])
        mean = filled.mean(axis=0)
        scale = filled.std(axis=0)
        scale = np.where(np.isfinite(scale) & (scale >= 1e-6), scale, 1.0)
        payload = np.concatenate((median, mean, scale)).astype("<f8", copy=False).tobytes()
        return cls(
            median=median.astype(np.float32),
            mean=mean.astype(np.float32),
            scale=scale.astype(np.float32),
            schema_hash=hashlib.sha256(payload).hexdigest(),
        )

    def transform(self, values: np.ndarray, observed_mask: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        mask = np.asarray(observed_mask, dtype=bool)
        if array.shape != mask.shape or array.shape[-1] != 50:
            raise ValueError("battery feature scaler expects matching [...,50] arrays")
        filled = np.where(mask & np.isfinite(array), array, self.median)
        return ((filled - self.mean) / self.scale).astype(np.float32)

    def state_dict(self) -> dict[str, Any]:
        return {
            "median": self.median.tolist(),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "schema_hash": self.schema_hash,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "BatteryFeatureScaler":
        return cls(
            median=np.asarray(state["median"], dtype=np.float32),
            mean=np.asarray(state["mean"], dtype=np.float32),
            scale=np.asarray(state["scale"], dtype=np.float32),
            schema_hash=str(state["schema_hash"]),
        )


@dataclass(frozen=True)
class BatterySplit:
    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]

    @classmethod
    def from_cell_ids(cls, cell_ids: Sequence[str], seed: int) -> "BatterySplit":
        splits = split_processed_items(sorted(str(value) for value in cell_ids), seed=int(seed))
        return cls(tuple(splits["train"]), tuple(splits["val"]), tuple(splits["test"]))

    @classmethod
    def from_cache(cls, cache: BatteryFeatureCache, seed: int) -> "BatterySplit":
        """Build the legacy split, except for protocol-stratified XJTU cells."""

        dataset = str(cache.manifest.get("provenance", {}).get("dataset", "")).strip().lower()
        if dataset != "xjtu":
            return cls.from_cell_ids(cache.cell_ids, seed=seed)

        groups: dict[tuple[str, str], list[str]] = {}
        for cell_id in cache.cell_ids:
            payload = cache.manifest["cells"][cell_id].get("operating_context")
            if payload is None:
                raise ValueError(f"XJTU protocol-stratified split requires operating context: {cell_id}")
            charge = str(payload.get("charge_protocol") or "").strip()
            discharge = str(payload.get("discharge_protocol") or "").strip()
            if not charge or not discharge:
                raise ValueError(
                    f"XJTU protocol-stratified split requires charge and discharge protocols: {cell_id}"
                )
            groups.setdefault((charge, discharge), []).append(str(cell_id))

        group_sizes = sorted(len(cell_ids) for cell_ids in groups.values())
        expected_sizes = [8, 8, 8, 8, 8, 15]
        if group_sizes != expected_sizes:
            raise ValueError(
                "XJTU protocol-stratified split requires six canonical protocol groups with "
                f"sizes {expected_sizes}; got {group_sizes}"
            )

        partitions: dict[str, list[str]] = {"train": [], "val": [], "test": []}
        for (charge, discharge), group_cell_ids in sorted(groups.items()):
            seed_payload = f"{int(seed)}\0{charge}\0{discharge}".encode("utf-8")
            group_seed = int.from_bytes(
                hashlib.blake2b(seed_payload, digest_size=8).digest(),
                byteorder="little",
                signed=False,
            )
            shuffled = np.asarray(sorted(group_cell_ids), dtype=object)
            np.random.default_rng(group_seed).shuffle(shuffled)
            train_count, val_count, test_count = (6, 1, 1) if len(shuffled) == 8 else (8, 3, 4)
            partitions["train"].extend(str(value) for value in shuffled[:train_count])
            partitions["val"].extend(
                str(value) for value in shuffled[train_count : train_count + val_count]
            )
            partitions["test"].extend(str(value) for value in shuffled[-test_count:])

        split = cls(
            tuple(sorted(partitions["train"])),
            tuple(sorted(partitions["val"])),
            tuple(sorted(partitions["test"])),
        )
        split_sets = [set(split.train), set(split.val), set(split.test)]
        if [len(values) for values in split_sets] != [38, 8, 9]:
            raise RuntimeError("XJTU protocol-stratified split produced invalid 38/8/9 totals")
        if any(split_sets[left] & split_sets[right] for left in range(3) for right in range(left + 1, 3)):
            raise RuntimeError("XJTU protocol-stratified split produced overlapping partitions")
        if set().union(*split_sets) != set(cache.cell_ids):
            raise RuntimeError("XJTU protocol-stratified split does not cover every cache cell")
        return split

    def ids(self, split: str) -> tuple[str, ...]:
        if split not in {"train", "val", "test"}:
            raise ValueError("battery split must be train, val, or test")
        return getattr(self, split)


@dataclass(frozen=True)
class BatteryWindow:
    cell_id: str
    origin: int
    segment_index: int
    segment_start: int
    segment_end: int


def _consecutive_segments(observation_ids: np.ndarray) -> list[tuple[int, int]]:
    """Return maximal [start,end) runs whose observation IDs differ by one."""

    ids = np.asarray(observation_ids, dtype=np.int64).reshape(-1)
    if ids.size == 0:
        return []
    boundaries = np.flatnonzero(np.diff(ids) != 1) + 1
    starts = np.concatenate((np.asarray([0]), boundaries))
    ends = np.concatenate((boundaries, np.asarray([len(ids)])))
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def _training_prompt_windows(cells: Sequence[Any], max_windows: int = 4096) -> Iterable[np.ndarray]:
    """Yield balanced valid prompt windows from training cells only."""

    if not cells:
        return
    per_cell = max(1, int(max_windows) // len(cells))
    minimum = BATTERY_PROMPT_CYCLES + BATTERY_NUMERIC_CYCLES + BATTERY_PREDICTION_CYCLES
    for cell in cells:
        observations = np.asarray(cell.observation_ids, dtype=np.int64).reshape(-1)
        raw_base = _base_values(cell)
        base_mask = np.asarray(cell.base_observed_mask, dtype=bool)
        targets = np.asarray(cell.soh_labels, dtype=np.float32).reshape(-1)
        origins: list[int] = []
        for segment_start, segment_end in _consecutive_segments(observations):
            if segment_end - segment_start < minimum:
                continue
            first = segment_start + BATTERY_PROMPT_CYCLES + BATTERY_NUMERIC_CYCLES
            final = segment_end - BATTERY_PREDICTION_CYCLES
            for origin in range(first, final + 1):
                numeric = slice(origin - BATTERY_NUMERIC_CYCLES, origin)
                future = targets[origin : origin + BATTERY_PREDICTION_CYCLES]
                valid_v_i = base_mask[numeric, 0] & base_mask[numeric, 4]
                if int(valid_v_i.sum()) >= BATTERY_MIN_VALID_NUMERIC_CYCLES and np.isfinite(future).all():
                    origins.append(origin)
        if len(origins) > per_cell:
            selected = np.linspace(0, len(origins) - 1, per_cell, dtype=np.int64)
            origins = [origins[int(index)] for index in selected]
        for origin in origins:
            prompt_slice = slice(
                origin - BATTERY_NUMERIC_CYCLES - BATTERY_PROMPT_CYCLES,
                origin - BATTERY_NUMERIC_CYCLES,
            )
            values = raw_base[prompt_slice].copy()
            mask = base_mask[prompt_slice]
            values[~mask | ~np.isfinite(values)] = np.nan
            yield values


class BatteryForecastDataset(Dataset):
    """Dual-window battery samples with no numerical SOH or cycle-index leakage."""

    def __init__(
        self,
        cache: BatteryFeatureCache,
        *,
        split: str,
        seed: int = 42,
        prompt_mode: str = "sensor_only",
        scaler: BatteryFeatureScaler | None = None,
        prompt_thresholds: Mapping[str, Sequence[float]] | None = None,
        training: bool | None = None,
        soh_context_dropout: float = 0.5,
        max_samples: int | None = None,
    ):
        if prompt_mode not in {"sensor_only", "soh_assisted"}:
            raise ValueError("prompt_mode must be sensor_only or soh_assisted")
        if not 0.0 <= float(soh_context_dropout) <= 1.0:
            raise ValueError("soh_context_dropout must be within 0..1")
        self.cache = cache
        self.split = split
        self.seed = int(seed)
        self.prompt_mode = prompt_mode
        self.training = split == "train" if training is None else bool(training)
        self.soh_context_dropout = float(soh_context_dropout)
        self.epoch = 0
        self.split_definition = BatterySplit.from_cache(cache, seed=self.seed)
        train_cells = [cache.load_cell(cell_id) for cell_id in self.split_definition.train]
        self.scaler = scaler or BatteryFeatureScaler.fit(train_cells)
        if prompt_thresholds is None:
            prompt_thresholds = fit_battery_prompt_thresholds(_training_prompt_windows(train_cells))
        missing_thresholds = sorted(set(BATTERY_PROMPT_METRICS) - set(prompt_thresholds))
        if missing_thresholds:
            raise ValueError(f"BatteryGTR prompt thresholds are missing: {missing_thresholds}")
        self.prompt_thresholds = {
            name: tuple(float(value) for value in prompt_thresholds[name])
            for name in BATTERY_PROMPT_METRICS
        }
        if any(len(values) != 2 for values in self.prompt_thresholds.values()):
            raise ValueError("each BatteryGTR prompt threshold requires two train-fitted boundaries")
        self.cells = {cell_id: cache.load_cell(cell_id) for cell_id in self.split_definition.ids(split)}
        self.samples: list[BatteryWindow] = []
        minimum = BATTERY_PROMPT_CYCLES + BATTERY_NUMERIC_CYCLES + BATTERY_PREDICTION_CYCLES
        for cell_id, cell in self.cells.items():
            observations = np.asarray(cell.observation_ids, dtype=np.int64).reshape(-1)
            base_mask = np.asarray(cell.base_observed_mask, dtype=bool)
            targets = np.asarray(cell.soh_labels, dtype=np.float32).reshape(-1)
            for segment_index, (segment_start, segment_end) in enumerate(_consecutive_segments(observations)):
                if segment_end - segment_start < minimum:
                    continue
                first_origin = segment_start + BATTERY_PROMPT_CYCLES + BATTERY_NUMERIC_CYCLES
                final_origin = segment_end - BATTERY_PREDICTION_CYCLES
                for origin in range(first_origin, final_origin + 1):
                    numeric = slice(origin - BATTERY_NUMERIC_CYCLES, origin)
                    valid_v_i = base_mask[numeric, 0] & base_mask[numeric, 4]
                    future = targets[origin : origin + BATTERY_PREDICTION_CYCLES]
                    if int(valid_v_i.sum()) < BATTERY_MIN_VALID_NUMERIC_CYCLES or not np.isfinite(future).all():
                        continue
                    self.samples.append(
                        BatteryWindow(cell_id, origin, segment_index, segment_start, segment_end)
                    )
                    if max_samples is not None and len(self.samples) >= int(max_samples):
                        break
                if max_samples is not None and len(self.samples) >= int(max_samples):
                    break
            if max_samples is not None and len(self.samples) >= int(max_samples):
                break
        self.sample_keys = tuple(
            battery_window_key(
                cache_hash=self.cache.manifest_hash,
                split=self.split,
                cell_id=window.cell_id,
                origin_row=window.origin,
                forecast_observation_id=int(
                    np.asarray(self.cells[window.cell_id].observation_ids).reshape(-1)[window.origin]
                ),
            )
            for window in self.samples
        )
        if len(self.sample_keys) != len(set(self.sample_keys)):
            raise RuntimeError("BatteryGTR forecast windows produced duplicate stable keys")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.samples)

    def _effective_prompt_mode(self, cell_id: str, origin: int) -> str:
        if self.prompt_mode == "sensor_only" or not self.training:
            return self.prompt_mode
        key = f"{self.seed}:{self.epoch}:{cell_id}:{origin}".encode("utf-8")
        draw = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little") / float(2**64)
        return "sensor_only" if draw < self.soh_context_dropout else "soh_assisted"

    def fixed_timecma_input(self, index: int) -> dict[str, Any]:
        """Read only the fixed recent-window inputs needed for cache population."""

        window = self.samples[index]
        cell = self.cells[window.cell_id]
        numeric_slice = slice(window.origin - BATTERY_NUMERIC_CYCLES, window.origin)
        raw_base = _base_values(cell)
        base_mask = np.asarray(cell.base_observed_mask, dtype=bool)
        base_values = self.scaler.transform(raw_base[numeric_slice], base_mask[numeric_slice])
        return {
            "base_values": torch.from_numpy(base_values),
            "base_observed_mask": torch.from_numpy(base_mask[numeric_slice].copy()),
            "metadata": {
                "dataset_split": self.split,
                "sample_index": int(index),
                "window_key": self.sample_keys[index],
            },
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        window = self.samples[index]
        cell_id, origin = window.cell_id, window.origin
        cell = self.cells[cell_id]
        prompt_slice = slice(origin - BATTERY_NUMERIC_CYCLES - BATTERY_PROMPT_CYCLES, origin - BATTERY_NUMERIC_CYCLES)
        numeric_slice = slice(origin - BATTERY_NUMERIC_CYCLES, origin)
        target_slice = slice(origin, origin + BATTERY_PREDICTION_CYCLES)
        raw_base = _base_values(cell)
        base_mask = np.asarray(cell.base_observed_mask, dtype=bool)
        base_values = self.scaler.transform(raw_base[numeric_slice], base_mask[numeric_slice])
        effective_mode = self._effective_prompt_mode(cell_id, origin)
        prior_soh = np.asarray(cell.soh_labels, dtype=np.float32)[prompt_slice] if effective_mode == "soh_assisted" else None
        prompt_values = raw_base[prompt_slice].copy()
        prompt_mask = base_mask[prompt_slice]
        prompt_values[~prompt_mask | ~np.isfinite(prompt_values)] = np.nan
        prompt = build_battery_prompt(
            prompt_values,
            prior_soh,
            effective_mode,
            operating_context=cell.operating_context,
            thresholds=self.prompt_thresholds,
        )
        observations = np.asarray(cell.observation_ids).reshape(-1)
        target = np.asarray(cell.soh_labels, dtype=np.float32)[target_slice]
        return {
            "base_values": torch.from_numpy(base_values),
            "base_observed_mask": torch.from_numpy(base_mask[numeric_slice].copy()),
            "base_reliability": torch.from_numpy(np.asarray(cell.base_reliability, dtype=np.float32)[numeric_slice].copy()),
            "ic_curve": torch.from_numpy(np.asarray(cell.ic_curve, dtype=np.float32)[numeric_slice].copy()),
            "ic_curve_mask": torch.from_numpy(np.asarray(cell.ic_curve_mask, dtype=bool)[numeric_slice].copy()),
            "ic_quality": torch.from_numpy(np.asarray(cell.ic_quality, dtype=np.float32)[numeric_slice].copy()),
            "dv_curve": torch.from_numpy(np.asarray(cell.dv_curve, dtype=np.float32)[numeric_slice].copy()),
            "dv_curve_mask": torch.from_numpy(np.asarray(cell.dv_curve_mask, dtype=bool)[numeric_slice].copy()),
            "dv_quality": torch.from_numpy(np.asarray(cell.dv_quality, dtype=np.float32)[numeric_slice].copy()),
            "prompt": prompt.text,
            "target": torch.from_numpy(target[:, None].copy()),
            "target_mask": torch.from_numpy(np.isfinite(target[:, None])),
            "metadata": {
                "cell_id": cell_id,
                "dataset_split": self.split,
                "sample_index": int(index),
                "window_key": self.sample_keys[index],
                "prompt_mode": effective_mode,
                "prompt_observation_ids": observations[prompt_slice].tolist(),
                "numeric_observation_ids": observations[numeric_slice].tolist(),
                "target_observation_ids": observations[target_slice].tolist(),
                "segment_id": f"{cell_id}:{window.segment_index}",
                "segment_index": window.segment_index,
                "segment_row_bounds": [window.segment_start, window.segment_end],
                "segment_observation_bounds": [
                    int(observations[window.segment_start]),
                    int(observations[window.segment_end - 1]),
                ],
                "cache_hash": self.cache.manifest_hash,
                "scaler_hash": self.scaler.schema_hash,
                "prompt_thresholds": {name: list(values) for name, values in self.prompt_thresholds.items()},
            },
        }


def collate_battery(items: Sequence[dict[str, Any]]) -> BatteryRawBatch:
    if not items:
        raise ValueError("cannot collate an empty BatteryGTR batch")
    batch = BatteryRawBatch(
        base_values=torch.stack([item["base_values"] for item in items]),
        base_observed_mask=torch.stack([item["base_observed_mask"] for item in items]).bool(),
        base_reliability=torch.stack([item["base_reliability"] for item in items]),
        ic_curve=torch.stack([item["ic_curve"] for item in items]),
        ic_curve_mask=torch.stack([item["ic_curve_mask"] for item in items]).bool(),
        ic_quality=torch.stack([item["ic_quality"] for item in items]),
        dv_curve=torch.stack([item["dv_curve"] for item in items]),
        dv_curve_mask=torch.stack([item["dv_curve_mask"] for item in items]).bool(),
        dv_quality=torch.stack([item["dv_quality"] for item in items]),
        prompts=[str(item["prompt"]) for item in items],
        target=torch.stack([item["target"] for item in items]),
        target_mask=torch.stack([item["target_mask"] for item in items]).bool(),
        metadata=[dict(item["metadata"]) for item in items],
    )
    validate_battery_raw_batch(batch)
    return batch
