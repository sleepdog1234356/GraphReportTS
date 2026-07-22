from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..data_general import _find_csv
from ..general_data_schema import dataset_schema
from ..general_protocol import FORMAL_HISTORY, GeneralForecastProtocol, StandardScalerNP, fit_train_scaler
from .contracts import ForecastBatchV2
from .prompts import (
    GENERAL_PROMPT_METRICS,
    LEVELS,
    PromptResultV2,
    build_general_prompt,
    fit_general_prompt_thresholds,
)


GENERAL_PREPROCESSING_SCHEMA = "graph-report-ts-v2-general-preprocessing-v2"
GENERAL_PROMPT_SCHEMA = "graph-report-ts-v2-general-prompt-v2"


def _stable_digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class GeneralForecastV2Dataset(Dataset):
    """Main-model dataset with disjoint causal prompt and numeric windows."""

    def __init__(
        self,
        dataset_name: str,
        data_root: str = "data/general",
        split: str = "train",
        pred_len: int = 24,
        scaler: StandardScalerNP | None = None,
        fit_scaler: bool = False,
        fill_values: np.ndarray | None = None,
        prompt_thresholds: Mapping[str, Sequence[float]] | None = None,
        fit_prompt_thresholds: bool = False,
        cache_prompts: bool = True,
        input_len: int = FORMAL_HISTORY,
        prompt_len: int = FORMAL_HISTORY,
    ) -> None:
        self.dataset_name = dataset_name
        self.split = split
        self.pred_len = int(pred_len)
        self.input_len = int(input_len)
        self.prompt_len = int(prompt_len)
        source = _find_csv(Path(data_root), dataset_name)
        self.source_csv = source.resolve()
        self._source_csv_sha256: str | None = None
        frame = pd.read_csv(self.source_csv)
        timestamp_column = next((c for c in ("date", "timestamp") if c in frame.columns), None)
        numeric = frame.drop(columns=[timestamp_column], errors="ignore").select_dtypes(include=[np.number])
        if numeric.empty:
            raise ValueError(f"{source} has no numeric columns")
        self.columns = tuple(str(c) for c in numeric.columns)
        self.raw_values = numeric.to_numpy(np.float32)
        self.frequency = dataset_schema(dataset_name).frequency
        self.protocol = GeneralForecastProtocol(
            dataset_name,
            len(self.raw_values),
            self.input_len,
            self.prompt_len,
        )
        self.scaler = scaler or StandardScalerNP()
        train_end = self.protocol.bounds["train"][1]
        if fit_scaler:
            fit_train_scaler(self.raw_values, train_end, self.scaler)
        if self.scaler.mean is None or self.scaler.std is None:
            raise ValueError("general v2 requires a train-fitted scaler")
        if fill_values is None:
            if split != "train":
                raise ValueError("validation/test datasets require train-only fill_values")
            raw_fill = np.nanmedian(self.raw_values[:train_end], axis=0)
            raw_fill = np.where(np.isfinite(raw_fill), raw_fill, self.scaler.mean)
            fill_values = self.scaler.transform(raw_fill[None, :])[0]
        self.fill_values = np.asarray(fill_values, dtype=np.float32)
        if self.fill_values.shape != (self.raw_values.shape[1],):
            raise ValueError("fill_values must match the general variable count")
        transformed = self.scaler.transform(self.raw_values)
        self.values = np.where(np.isfinite(transformed), transformed, self.fill_values[None, :]).astype(np.float32)
        self.samples = [
            int(start)
            for start in self.protocol.window_index(split, self.pred_len)
            if int(start) >= self.prompt_len
        ]
        if fit_prompt_thresholds or (prompt_thresholds is None and split == "train"):
            if split != "train":
                raise ValueError("prompt thresholds may only be fitted by the training split")
            prompt_thresholds = fit_general_prompt_thresholds(
                self.raw_values,
                self.samples,
                prompt_len=self.prompt_len,
            )
        if prompt_thresholds is None:
            raise ValueError("general v2 requires train-fitted prompt_thresholds")
        self.prompt_thresholds = {name: tuple(float(x) for x in boundaries) for name, boundaries in prompt_thresholds.items()}
        self.cache_prompts = bool(cache_prompts)
        self._prompt_cache: dict[int, PromptResultV2] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def _prompt_result_at(self, index: int) -> PromptResultV2:
        numeric_start = self.samples[index]
        prompt_len = int(getattr(self, "prompt_len", FORMAL_HISTORY))
        prompt = self._prompt_cache.get(numeric_start)
        if prompt is None:
            prompt = build_general_prompt(
                self.raw_values[numeric_start - prompt_len : numeric_start],
                self.columns,
                self.frequency,
                self.pred_len,
                self.prompt_thresholds,
            )
            if self.cache_prompts:
                self._prompt_cache[numeric_start] = prompt
        return prompt

    def prompt_at(self, index: int) -> str:
        """Build only the causal prompt context, without reading its target."""

        return self._prompt_result_at(index).text

    def __getitem__(self, index: int) -> dict[str, object]:
        numeric_start = self.samples[index]
        prompt_len = int(getattr(self, "prompt_len", FORMAL_HISTORY))
        input_len = int(getattr(self, "input_len", FORMAL_HISTORY))
        prompt_start = numeric_start - prompt_len
        numeric_end = numeric_start + input_len
        target_end = numeric_end + self.pred_len
        values = self.values[numeric_start:numeric_end]
        target = self.values[numeric_end:target_end]
        observed = np.isfinite(self.raw_values[numeric_start:numeric_end])
        target_observed = np.isfinite(self.raw_values[numeric_end:target_end])
        prompt = self._prompt_result_at(index)
        return {
            "values": torch.from_numpy(values).float(),
            "observed_mask": torch.from_numpy(observed),
            "reliability": torch.from_numpy(observed.astype(np.float32)),
            "target": torch.from_numpy(target).float(),
            "target_mask": torch.from_numpy(target_observed),
            "prompt": prompt.text,
            "metadata": {
                "dataset": self.dataset_name,
                "prompt_start": prompt_start,
                "numeric_start": numeric_start,
                "numeric_end": numeric_end,
                "target_start": numeric_end,
                "target_end": target_end,
                "input_len": input_len,
                "prompt_len": prompt_len,
                "columns": self.columns,
                "prompt_metadata": prompt.metadata,
            },
        }

    def preprocessing_state(self) -> dict[str, object]:
        # Only the training dataset is asked for this state by the trainer.  Keep
        # the potentially expensive file digest lazy and memoized so a repeated
        # checkpoint/config write never rereads a large ECL CSV.
        if self._source_csv_sha256 is None:
            self._source_csv_sha256 = _file_sha256(self.source_csv)
        scaler_schema = {
            "kind": "standard_scaler",
            "fit_split": "train",
            "variable_count": len(self.columns),
            "columns": list(self.columns),
            "mean": self.scaler.mean.tolist(),
            "std": self.scaler.std.tolist(),
            "fill_values": self.fill_values.tolist(),
        }
        prompt_len = int(getattr(self, "prompt_len", FORMAL_HISTORY))
        input_len = int(getattr(self, "input_len", FORMAL_HISTORY))
        prompt_schema = {
            "schema": GENERAL_PROMPT_SCHEMA,
            "context_length": prompt_len,
            "numeric_history_length": input_len,
            "metrics": list(GENERAL_PROMPT_METRICS),
            "levels": list(LEVELS),
            "threshold_fit_split": "train",
            "thresholds": {name: list(values) for name, values in self.prompt_thresholds.items()},
        }
        dataset_identity = {
            "name": self.dataset_name,
            "source_csv": {
                "path": str(self.source_csv),
                "sha256": self._source_csv_sha256,
            },
            "row_count": int(self.raw_values.shape[0]),
            "variable_count": int(self.raw_values.shape[1]),
            "columns_sha256": _stable_digest(list(self.columns)),
        }
        return {
            "schema": GENERAL_PREPROCESSING_SCHEMA,
            "dataset_identity": dataset_identity,
            "scaler": {"summary": scaler_schema, "sha256": _stable_digest(scaler_schema)},
            "prompt_schema": {"summary": prompt_schema, "sha256": _stable_digest(prompt_schema)},
        }


class SyntheticGeneralV2Dataset(Dataset):
    def __init__(
        self,
        size: int = 32,
        variables: int = 7,
        pred_len: int = 24,
        seed: int = 0,
        input_len: int = FORMAL_HISTORY,
        prompt_len: int = FORMAL_HISTORY,
    ) -> None:
        self.size = int(size)
        self.variables = int(variables)
        self.pred_len = int(pred_len)
        self.seed = int(seed)
        self.input_len = int(input_len)
        self.prompt_len = int(prompt_len)
        rng = np.random.default_rng(seed)
        self.items = []
        for item_id in range(size):
            total = self.prompt_len + self.input_len + self.pred_len
            series = np.cumsum(rng.normal(size=(total, variables)), axis=0).astype(np.float32) / 10
            thresholds = {name: (0.0, 0.1, 0.3, 0.6) for name in GENERAL_PROMPT_METRICS}
            prompt = build_general_prompt(
                series[: self.prompt_len],
                [f"v{i}" for i in range(variables)],
                "steps",
                pred_len,
                thresholds,
            )
            numeric_end = self.prompt_len + self.input_len
            self.items.append({
                "values": torch.from_numpy(series[self.prompt_len : numeric_end]),
                "observed_mask": torch.ones(self.input_len, variables, dtype=torch.bool),
                "reliability": torch.ones(self.input_len, variables),
                "target": torch.from_numpy(series[numeric_end:]),
                "target_mask": torch.ones(pred_len, variables, dtype=torch.bool),
                "prompt": prompt.text,
                "metadata": {
                    "dataset": "synthetic",
                    "item_id": item_id,
                    "prompt_start": 0,
                    "numeric_start": self.prompt_len,
                    "numeric_end": numeric_end,
                    "target_start": numeric_end,
                    "target_end": total,
                    "input_len": self.input_len,
                    "prompt_len": self.prompt_len,
                },
            })

    def __len__(self) -> int:
        return len(self.items)

    def prompt_at(self, index: int) -> str:
        return str(self.items[index]["prompt"])

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.items[index]

    def preprocessing_state(self) -> dict[str, object]:
        dataset_identity = {
            "name": "synthetic",
            "generator": "numpy.default_rng-cumulative-normal-v1",
            "size": self.size,
            "variable_count": self.variables,
            "pred_len": self.pred_len,
            "seed": self.seed,
            "input_len": self.input_len,
            "prompt_len": self.prompt_len,
        }
        scaler_schema = {"kind": "identity", "fit_split": "synthetic"}
        prompt_schema = {
            "schema": GENERAL_PROMPT_SCHEMA,
            "context_length": self.prompt_len,
            "numeric_history_length": self.input_len,
            "metrics": list(GENERAL_PROMPT_METRICS),
            "levels": list(LEVELS),
            "threshold_fit_split": "synthetic-fixed",
        }
        return {
            "schema": GENERAL_PREPROCESSING_SCHEMA,
            "dataset_identity": dataset_identity,
            "scaler": {"summary": scaler_schema, "sha256": _stable_digest(scaler_schema)},
            "prompt_schema": {"summary": prompt_schema, "sha256": _stable_digest(prompt_schema)},
        }


def collate_general_v2(items: Sequence[dict[str, object]]) -> ForecastBatchV2:
    if not items:
        raise ValueError("cannot collate an empty general batch")
    batch = len(items)
    variables = max(int(item["values"].shape[-1]) for item in items)
    horizon = max(int(item["target"].shape[0]) for item in items)
    history = int(items[0]["values"].shape[0])
    if any(int(item["values"].shape[0]) != history for item in items):
        raise ValueError("general batch mixes numeric history lengths")
    values = torch.zeros(batch, history, variables)
    observed = torch.zeros(batch, history, variables, dtype=torch.bool)
    reliability = torch.zeros(batch, history, variables)
    target = torch.zeros(batch, horizon, variables)
    target_mask = torch.zeros(batch, horizon, variables, dtype=torch.bool)
    variable_mask = torch.zeros(batch, variables, dtype=torch.bool)
    for row, item in enumerate(items):
        width = int(item["values"].shape[-1])
        steps = int(item["target"].shape[0])
        values[row, :, :width] = item["values"]
        observed[row, :, :width] = item["observed_mask"]
        reliability[row, :, :width] = item["reliability"]
        target[row, :steps, :width] = item["target"]
        target_mask[row, :steps, :width] = item["target_mask"]
        variable_mask[row, :width] = True
    return ForecastBatchV2(
        values=values,
        observed_mask=observed,
        reliability=reliability,
        variable_type=torch.zeros(variables, dtype=torch.long),
        variable_mask=variable_mask,
        prompts=[str(item["prompt"]) for item in items],
        target=target,
        target_mask=target_mask,
        metadata=[dict(item["metadata"]) for item in items],
    )
