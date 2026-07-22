from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch


RESULT_SCHEMA_VERSION = "gtr-results-v1"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def stable_digest(value: Mapping[str, Any]) -> str:
    payload = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def masked_regression_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, float]:
    if prediction.shape != target.shape or mask.shape != target.shape:
        raise ValueError("prediction, target, and mask must have identical shapes")
    valid = mask.bool()
    if not bool(valid.any()):
        raise ValueError("cannot compute regression metrics without valid targets")
    difference = prediction.detach()[valid].float() - target.detach()[valid].float()
    mse = float(difference.square().mean().cpu())
    mae = float(difference.abs().mean().cpu())
    return {"mse": mse, "mae": mae, "rmse": mse**0.5}


@dataclass(frozen=True)
class GTRResultRecord:
    model: str
    domain: str
    task: str
    dataset: str
    horizon: int
    seed: int
    best_epoch: int
    best_validation_mse: float
    test_mse: float
    test_mae: float
    test_rmse: float
    parameter_count: int
    prompt_policy: str
    source_commit: str
    adapter_schema_hash: str
    cache_hash: str
    optimizer_profile: Mapping[str, Any]
    runtime: Mapping[str, Any]
    schema_version: str = RESULT_SCHEMA_VERSION

    def validate(self) -> None:
        if self.domain not in {"general", "battery"}:
            raise ValueError(f"unsupported result domain: {self.domain}")
        if self.domain == "battery" and self.task != "battery_M_to_1":
            raise ValueError("BatteryGTR results must use task=battery_M_to_1")
        if self.horizon <= 0:
            raise ValueError("result horizon must be positive")
        if self.best_epoch < 1:
            raise ValueError("best_epoch must be positive")
        for name in ("best_validation_mse", "test_mse", "test_mae", "test_rmse"):
            value = float(getattr(self, name))
            if not (value >= 0.0 and value < float("inf")):
                raise ValueError(f"{name} must be finite and non-negative")
        if self.parameter_count <= 0:
            raise ValueError("parameter_count must be positive")

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        return _jsonable(asdict(self))


def make_result_record(
    *,
    model_name: str,
    domain: str,
    dataset: str,
    seed: int,
    best_epoch: int,
    validation_mse: float,
    test_metrics: Mapping[str, float],
    parameter_count: int,
    prompt_policy: str,
    source_commit: str,
    horizon: int | None = None,
    adapter_schema_hash: str = "not-applicable",
    cache_hash: str = "not-applicable",
    optimizer_profile: Mapping[str, Any] | None = None,
    runtime: Mapping[str, Any] | None = None,
) -> GTRResultRecord:
    task = "battery_M_to_1" if domain == "battery" else "general_M_to_M"
    return GTRResultRecord(
        model=model_name,
        domain=domain,
        task=task,
        dataset=dataset,
        horizon=20 if domain == "battery" else int(horizon or 0),
        seed=int(seed),
        best_epoch=int(best_epoch),
        best_validation_mse=float(validation_mse),
        test_mse=float(test_metrics["mse"]),
        test_mae=float(test_metrics["mae"]),
        test_rmse=float(test_metrics["rmse"]),
        parameter_count=int(parameter_count),
        prompt_policy=prompt_policy,
        source_commit=source_commit,
        adapter_schema_hash=adapter_schema_hash,
        cache_hash=cache_hash,
        optimizer_profile=dict(optimizer_profile or {}),
        runtime=dict(runtime or {}),
    )


def write_result(record: GTRResultRecord, run_dir: str | Path) -> Path:
    destination = Path(run_dir)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "result.json"
    temporary = destination / "result.json.tmp"
    temporary.write_text(json.dumps(record.as_dict(), indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)
    return path


def load_result(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise ValueError(f"unsupported GTR result schema in {path}")
    return payload


def aggregate_results(result_paths: Iterable[str | Path], output_csv: str | Path) -> Path:
    rows = [load_result(path) for path in result_paths]
    if not rows:
        raise ValueError("at least one GTR result is required for aggregation")
    scalar_fields = (
        "schema_version",
        "model",
        "domain",
        "task",
        "dataset",
        "horizon",
        "seed",
        "best_epoch",
        "best_validation_mse",
        "test_mse",
        "test_mae",
        "test_rmse",
        "parameter_count",
        "prompt_policy",
        "source_commit",
        "adapter_schema_hash",
        "cache_hash",
    )
    destination = Path(output_csv)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item["domain"], item["dataset"], item["model"], item["seed"])):
            writer.writerow({field: row[field] for field in scalar_fields})
    return destination
