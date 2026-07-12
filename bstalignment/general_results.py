"""Atomic, comparable result bundles for formal general forecasting runs.

The contract deliberately works on standardized tensors only.  Trainers own
model execution; this module owns the evidence needed to compare completed
runs without allowing the test split to influence checkpoint selection.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any, Iterable, Mapping

import numpy as np
import torch


REQUIRED_ARTIFACTS = (
    "run_config.json",
    "metrics.json",
    "history.csv",
    "predictions.npz",
    "best.pt",
    "environment.json",
)
FORMAL_DATASETS = ("ETTm1", "ETTm2", "ETTh1", "ETTh2", "ECL", "Weather")
FORMAL_HORIZONS = (96, 192, 336, 720)


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    errors: tuple[str, ...] = ()


def standardized_metrics(prediction: Any, target: Any) -> dict[str, float | int | str]:
    """Return element-weighted MSE/MAE for matching standardized forecasts."""

    prediction_array = np.asarray(prediction, dtype=np.float64)
    target_array = np.asarray(target, dtype=np.float64)
    if prediction_array.ndim != 3 or prediction_array.shape != target_array.shape:
        raise ValueError("standardized metrics require matching [sample, step, variable] arrays")
    if prediction_array.shape[0] == 0 or prediction_array.size == 0:
        raise ValueError("standardized metrics require at least one forecast element")
    difference = prediction_array - target_array
    return {
        "space": "standardized",
        "aggregation": "sample_element_weighted",
        "sample_count": int(prediction_array.shape[0]),
        "element_count": int(prediction_array.size),
        "mse": float(np.mean(np.square(difference))),
        "mae": float(np.mean(np.abs(difference))),
    }


def select_validation_checkpoint(best_mse: float, validation_mse: float) -> bool:
    """Return whether a checkpoint improves validation MSE; test metrics are absent by design."""

    if not np.isfinite(validation_mse):
        raise ValueError("validation MSE must be finite")
    return float(validation_mse) < float(best_mse)


def _json_write(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _plain(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


class GeneralRunWriter:
    """Build one formal result bundle in ``<run_dir>.partial`` then rename it atomically."""

    def __init__(self, run_dir: str | Path, expected_spec: Mapping[str, Any]):
        self.run_dir = Path(run_dir)
        self.partial_dir = self.run_dir.with_name(f"{self.run_dir.name}.partial")
        self.expected_spec = _plain(dict(expected_spec))
        if self.run_dir.exists():
            raise FileExistsError(f"completed run already exists: {self.run_dir}")
        if self.partial_dir.exists():
            raise FileExistsError(f"incomplete run already exists: {self.partial_dir}")
        self.partial_dir.mkdir(parents=True)
        self._config: dict[str, Any] | None = None
        self._history: list[dict[str, Any]] = []
        self._best_epoch: int | None = None
        self._best_mse = float("inf")
        self._test_metrics: dict[str, Any] | None = None
        self._test_count = 0
        self._environment_written = False
        self._completed = False

    @property
    def path(self) -> Path:
        return self.partial_dir

    def write_run_config(self, config: Mapping[str, Any]) -> None:
        if self._config is not None:
            raise RuntimeError("run_config may be written only once")
        self._config = _plain(dict(config))
        if self._config.get("dataset") != self.expected_spec.get("dataset"):
            raise ValueError("run config dataset does not match formal expected spec")
        if self._config.get("metrics_space") != "standardized":
            raise ValueError("general result contract requires standardized metrics")
        _json_write(self.partial_dir / "run_config.json", self._config)

    def append_history(self, row: Mapping[str, Any]) -> None:
        if self._completed:
            raise RuntimeError("cannot append history after completion")
        clean = _plain(dict(row))
        if "epoch" not in clean:
            raise ValueError("history rows require epoch")
        self._history.append(clean)
        fields = sorted({key for item in self._history for key in item})
        with (self.partial_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self._history)

    def record_validation(self, *, epoch: int, mse: float, checkpoint: Mapping[str, Any]) -> bool:
        if self._test_count:
            raise RuntimeError("validation checkpoint selection must finish before test evaluation")
        if epoch < 1:
            raise ValueError("checkpoint epoch must be positive")
        if any("test" in str(field).lower() for field in checkpoint):
            raise ValueError("validation checkpoint payload cannot include test metrics")
        if not select_validation_checkpoint(self._best_mse, mse):
            return False
        self._best_mse = float(mse)
        self._best_epoch = int(epoch)
        payload = dict(checkpoint)
        payload.update({"epoch": self._best_epoch, "validation_mse": self._best_mse, "selection_metric": "validation_mse"})
        torch.save(payload, self.partial_dir / "best.pt")
        return True

    def record_test(
        self,
        prediction: Any,
        target: Any,
        *,
        sample_indices: Iterable[int],
        step_indices: Iterable[int],
        variable_indices: Iterable[int],
    ) -> dict[str, Any]:
        if self._best_epoch is None:
            raise RuntimeError("test evaluation requires a validation-selected best checkpoint")
        if self._test_count:
            raise RuntimeError("formal result contract permits exactly one test evaluation")
        prediction_array = np.asarray(prediction, dtype=np.float32)
        target_array = np.asarray(target, dtype=np.float32)
        metrics = standardized_metrics(prediction_array, target_array)
        sample_index = np.asarray(list(sample_indices), dtype=np.int64)
        step_index = np.asarray(list(step_indices), dtype=np.int64)
        variable_index = np.asarray(list(variable_indices), dtype=np.int64)
        samples, steps, variables = prediction_array.shape
        valid_step_shape = step_index.shape in {(steps,), (samples, steps)}
        if sample_index.shape != (samples,) or not valid_step_shape or variable_index.shape != (variables,):
            raise ValueError("prediction indices must match sample, step, and variable dimensions")
        np.savez_compressed(
            self.partial_dir / "predictions.npz",
            prediction=prediction_array,
            target=target_array,
            sample_index=sample_index,
            step_index=step_index,
            variable_index=variable_index,
        )
        self._test_metrics = metrics
        self._test_count = 1
        return dict(metrics)

    def write_environment(self, environment: Mapping[str, Any]) -> None:
        snapshot = {
            "python": sys.version,
            "platform": platform.platform(),
            "pid": os.getpid(),
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            **_plain(dict(environment)),
        }
        _json_write(self.partial_dir / "environment.json", snapshot)
        self._environment_written = True

    def complete(self, provenance: Mapping[str, Any]) -> Path:
        if self._completed:
            raise RuntimeError("run has already been completed")
        if self._config is None or not self._history or self._best_epoch is None or self._test_metrics is None or not self._environment_written:
            raise RuntimeError("cannot complete run before config, history, validation checkpoint, one test, and environment are recorded")
        clean_provenance = _plain(dict(provenance))
        _validate_provenance(clean_provenance, self.expected_spec, self._config)
        environment_path = self.partial_dir / "environment.json"
        environment = json.loads(environment_path.read_text(encoding="utf-8"))
        environment["provenance"] = {
            "dataset_checksum": clean_provenance["dataset_checksum"],
            "source_commit": clean_provenance["source_commit"],
            "protocol": clean_provenance["protocol"],
            "runtime": clean_provenance["runtime"],
        }
        _json_write(environment_path, environment)
        self._config["provenance"] = clean_provenance
        self._config["selection"] = {"metric": "validation_mse", "best_epoch": self._best_epoch, "best_mse": self._best_mse}
        self._config["test_evaluations"] = self._test_count
        _json_write(self.partial_dir / "run_config.json", self._config)
        metrics = {"space": "standardized", "aggregation": "sample_element_weighted", "selection": self._config["selection"], "test_evaluations": self._test_count, "test": self._test_metrics}
        _json_write(self.partial_dir / "metrics.json", metrics)
        validation = _validate_run(self.partial_dir, self.expected_spec, allow_partial=True)
        if not validation.valid:
            raise ValueError("cannot finalize invalid run: " + "; ".join(validation.errors))
        self.partial_dir.replace(self.run_dir)
        self._completed = True
        return self.run_dir


def _validate_provenance(provenance: Mapping[str, Any], expected_spec: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    for field in ("dataset_checksum", "source_commit", "protocol", "source", "runtime"):
        if field not in provenance:
            raise ValueError(f"provenance requires {field}")
    for field in ("dataset_checksum", "source_commit", "protocol"):
        if provenance[field] != expected_spec.get(field):
            raise ValueError(f"provenance {field} does not match expected formal spec")
    source = provenance["source"]
    if not isinstance(source, Mapping) or source.get("commit") != provenance["source_commit"]:
        raise ValueError("provenance source commit must match source_commit")
    runtime = provenance["runtime"]
    if not isinstance(runtime, Mapping):
        raise ValueError("provenance runtime must be a mapping")
    for field in ("wall_time_seconds", "peak_gpu_memory_bytes", "trainable_parameters"):
        if field not in runtime:
            raise ValueError(f"provenance runtime requires {field}")
    if config.get("model") == "Time-LLM":
        if not isinstance(runtime.get("time_llm"), Mapping):
            raise ValueError("Time-LLM runtime provenance is required")
        if not isinstance(provenance.get("prompt_audit"), Mapping):
            raise ValueError("Time-LLM prompt audit is required")


def _validate_run(run_dir: Path, expected_spec: Mapping[str, Any], *, allow_partial: bool) -> ValidationResult:
    errors: list[str] = []
    if not run_dir.is_dir():
        return ValidationResult(False, (f"run directory does not exist: {run_dir}",))
    if run_dir.name.endswith(".partial") and not allow_partial:
        errors.append("partial run directories are not completed runs")
    for artifact in REQUIRED_ARTIFACTS:
        if not (run_dir / artifact).is_file():
            errors.append(f"missing required artifact: {artifact}")
    if errors:
        return ValidationResult(False, tuple(errors))
    try:
        config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        _validate_provenance(config.get("provenance", {}), expected_spec, config)
        selection = metrics.get("selection")
        if selection != config.get("selection") or selection.get("metric") != "validation_mse":
            errors.append("selection must be validation_mse and match run config")
        if metrics.get("test_evaluations") != 1 or config.get("test_evaluations") != 1:
            errors.append("completed formal run requires exactly one test evaluation")
        if metrics.get("space") != "standardized" or metrics.get("aggregation") != "sample_element_weighted":
            errors.append("metrics must be standardized sample_element_weighted")
        with np.load(run_dir / "predictions.npz") as values:
            required_keys = {"prediction", "target", "sample_index", "step_index", "variable_index"}
            if set(values.files) != required_keys:
                errors.append("predictions archive has an invalid schema")
            elif values["prediction"].shape != values["target"].shape or values["prediction"].ndim != 3:
                errors.append("predictions must contain matching [sample, step, variable] arrays")
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        errors.append(str(exc))
    return ValidationResult(not errors, tuple(errors))


def validate_completed_run(run_dir: str | Path, expected_spec: Mapping[str, Any]) -> ValidationResult:
    """Reject incomplete, stale, or provenance-mismatched formal run directories."""

    return _validate_run(Path(run_dir), _plain(dict(expected_spec)), allow_partial=False)
