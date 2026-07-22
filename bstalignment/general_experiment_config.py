"""Validated, immutable configuration for formal general forecasting runs."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


EXPECTED_DATASETS = ("ETTm1", "ETTm2", "ETTh1", "ETTh2", "ECL", "Weather")
EXPECTED_MODELS = (
    "GraphReportTS", "PatchTST", "iTransformer", "TimeCMA", "TimesNet", "DLinear", "Time-LLM"
)
EXPECTED_HORIZONS = (24, 36, 48, 60)
FORMAL_SEEDS = (2021, 2022, 2023)
SMOKE_SEED = 42
FEATURES = "M"
SUPPORTED_DATASETS = frozenset(EXPECTED_DATASETS)
SUPPORTED_MODELS = frozenset(EXPECTED_MODELS)
BASELINE_MODELS = SUPPORTED_MODELS - {"GraphReportTS"}
SUPPORTED_HORIZONS = frozenset(EXPECTED_HORIZONS)
AUDITED_SOURCES = MappingProxyType(
    {
        "PatchTST": ("https://github.com/yuqinie98/PatchTST", "204c21e"),
        "iTransformer": ("https://github.com/thuml/iTransformer", "c2426e6"),
        "TimeCMA": ("https://github.com/ChenxiLiu-HNU/TimeCMA", "223e4ae"),
        "TimesNet": ("https://github.com/thuml/Time-Series-Library", "4e938a1"),
        "DLinear": ("https://github.com/cure-lab/LTSF-Linear", "0c11366"),
        "Time-LLM": ("https://github.com/KimMeen/Time-LLM", "b13e881"),
    }
)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    raw_path: str
    raw_sha256: str


@dataclass(frozen=True)
class ModelSpec:
    name: str


@dataclass(frozen=True)
class SourceCommit:
    name: str
    url: str
    commit: str


@dataclass(frozen=True)
class HorizonSpec:
    values: tuple[int, ...]


@dataclass(frozen=True)
class SeedSpec:
    smoke: int
    values: tuple[int, ...]


@dataclass(frozen=True)
class ExperimentPaths:
    datasets: str
    models: str
    output_root: str


@dataclass(frozen=True)
class GeneralExperimentSpec:
    datasets: tuple[DatasetSpec, ...]
    models: tuple[ModelSpec, ...]
    input_len: int
    features: str
    horizon_spec: HorizonSpec
    seed_spec: SeedSpec
    paths: ExperimentPaths
    source_commits: tuple[SourceCommit, ...]

    @property
    def horizons(self) -> tuple[int, ...]:
        return self.horizon_spec.values

    @property
    def formal_seeds(self) -> tuple[int, ...]:
        return self.seed_spec.values

    @property
    def smoke_seed(self) -> int:
        return self.seed_spec.smoke

    @property
    def run_ids(self) -> tuple[str, ...]:
        return tuple(
            f"{model.name}__{dataset.name}__h{horizon}__seed{seed}"
            for model in self.models
            for dataset in self.datasets
            for horizon in self.horizons
            for seed in self.formal_seeds
        )


def _read_mapping(path: Path) -> Mapping[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, Mapping):
        raise ValueError(f"configuration at {path} must be a mapping")
    return value


def _required_string(record: Mapping[str, Any], field: str, description: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{description} requires {field}")
    return value


def _require_integer(value: Any, field: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{field} must be an integer")
    return value


def _load_datasets(config: Mapping[str, Any]) -> tuple[DatasetSpec, ...]:
    records = config.get("datasets")
    if not isinstance(records, list):
        raise ValueError("datasets configuration requires a datasets list")
    datasets = tuple(
        DatasetSpec(
            name=_required_string(record, "name", "dataset"),
            raw_path=_required_string(record, "raw_path", "dataset"),
            raw_sha256=_required_string(record, "raw_sha256", "dataset"),
        )
        for record in records
        if isinstance(record, Mapping)
    )
    if len(datasets) != len(records):
        raise ValueError("dataset records must be mappings")
    for dataset in datasets:
        if dataset.name not in SUPPORTED_DATASETS:
            raise ValueError(f"unknown dataset: {dataset.name}")
        if len(dataset.raw_sha256) != 64 or any(char not in "0123456789abcdef" for char in dataset.raw_sha256):
            raise ValueError(f"dataset {dataset.name} requires a lowercase SHA-256 checksum")
    if tuple(dataset.name for dataset in datasets) != EXPECTED_DATASETS:
        raise ValueError("dataset catalog must contain the complete formal datasets in canonical order")
    return datasets


def _load_models(config: Mapping[str, Any]) -> tuple[ModelSpec, ...]:
    records = config.get("models")
    if not isinstance(records, list):
        raise ValueError("models configuration requires a models list")
    models = tuple(
        ModelSpec(name=_required_string(record, "name", "model"))
        for record in records
        if isinstance(record, Mapping)
    )
    if len(models) != len(records):
        raise ValueError("model records must be mappings")
    unknown_models = {model.name for model in models} - SUPPORTED_MODELS
    if unknown_models:
        raise ValueError(f"unknown model: {sorted(unknown_models)[0]}")
    if tuple(model.name for model in models) != EXPECTED_MODELS:
        raise ValueError("model catalog must contain the complete formal models in canonical order")
    return models


def _load_sources(config: Mapping[str, Any]) -> tuple[SourceCommit, ...]:
    records = config.get("sources")
    if not isinstance(records, Mapping):
        raise ValueError("models configuration requires source commits")
    sources = tuple(
        SourceCommit(
            name=name,
            url=_required_string(record, "url", "source"),
            commit=_required_string(record, "commit", "source commit"),
        )
        for name, record in records.items()
        if isinstance(name, str) and isinstance(record, Mapping)
    )
    if len(sources) != len(records):
        raise ValueError("source commit records must be mappings")
    source_names = {source.name for source in sources}
    missing_sources = BASELINE_MODELS - source_names
    if missing_sources:
        raise ValueError(f"missing source commit for {sorted(missing_sources)[0]}")
    for source in sources:
        if source.name not in BASELINE_MODELS:
            raise ValueError(f"unknown source model: {source.name}")
        expected_url, expected_commit = AUDITED_SOURCES[source.name]
        if source.url != expected_url:
            raise ValueError(f"audited source URL mismatch for {source.name}")
        if source.commit != expected_commit:
            raise ValueError(f"audited source commit mismatch for {source.name}")
    return sources


def _load_paths(config: Mapping[str, Any]) -> ExperimentPaths:
    paths = config.get("paths")
    if not isinstance(paths, Mapping):
        raise ValueError("experiment matrix requires paths")
    return ExperimentPaths(
        datasets=_required_string(paths, "datasets", "paths"),
        models=_required_string(paths, "models", "paths"),
        output_root=_required_string(paths, "output_root", "paths"),
    )


def load_general_experiment_spec(path: Path) -> GeneralExperimentSpec:
    """Load and validate the frozen formal general-forecasting matrix."""

    path = Path(path)
    matrix = _read_mapping(path)
    paths = _load_paths(matrix)
    datasets = _load_datasets(_read_mapping(path.parent / paths.datasets))
    models_config = _read_mapping(path.parent / paths.models)
    models = _load_models(models_config)
    sources = _load_sources(models_config)

    input_len = _require_integer(matrix.get("input_len"), "input_len")
    if input_len != 36:
        raise ValueError("input_len must be 36 for formal general forecasting")
    raw_horizons = matrix.get("horizons")
    if not isinstance(raw_horizons, list):
        raise ValueError("horizons must be a list of integers")
    horizons = tuple(_require_integer(horizon, "horizons") for horizon in raw_horizons)
    if not horizons or any(horizon not in SUPPORTED_HORIZONS for horizon in horizons):
        raise ValueError("horizons must be selected from 24, 36, 48, and 60")
    smoke_seed = _require_integer(matrix.get("smoke_seed"), "smoke_seed")
    if smoke_seed != SMOKE_SEED:
        raise ValueError("smoke_seed must be 42")
    raw_formal_seeds = matrix.get("formal_seeds")
    if not isinstance(raw_formal_seeds, list):
        raise ValueError("formal_seeds must be a list of integers")
    formal_seeds = tuple(_require_integer(seed, "formal_seeds") for seed in raw_formal_seeds)
    if formal_seeds != FORMAL_SEEDS:
        raise ValueError("formal_seeds must be 2021, 2022, and 2023")
    features = matrix.get("features")
    if features != FEATURES:
        raise ValueError("features must be M for multivariate-to-multivariate forecasting")

    selected_datasets = tuple(matrix.get("datasets", ()))
    selected_models = tuple(matrix.get("models", ()))
    configured_datasets = {dataset.name: dataset for dataset in datasets}
    configured_models = {model.name: model for model in models}
    unknown_datasets = set(selected_datasets) - set(configured_datasets)
    if unknown_datasets:
        raise ValueError(f"unknown dataset: {sorted(unknown_datasets)[0]}")
    unknown_models = set(selected_models) - set(configured_models)
    if unknown_models:
        raise ValueError(f"unknown model: {sorted(unknown_models)[0]}")
    selected_dataset_specs = tuple(configured_datasets[name] for name in selected_datasets)
    selected_model_specs = tuple(configured_models[name] for name in selected_models)

    spec = GeneralExperimentSpec(
        datasets=selected_dataset_specs,
        models=selected_model_specs,
        input_len=input_len,
        features=features,
        horizon_spec=HorizonSpec(horizons),
        seed_spec=SeedSpec(smoke_seed, formal_seeds),
        paths=paths,
        source_commits=sources,
    )
    if len(spec.run_ids) != len(set(spec.run_ids)):
        raise ValueError("duplicate run ID in experiment matrix")
    if tuple(dataset.name for dataset in spec.datasets) != EXPECTED_DATASETS:
        raise ValueError("formal matrix must contain the complete datasets in canonical order")
    if tuple(model.name for model in spec.models) != EXPECTED_MODELS:
        raise ValueError("formal matrix must contain the complete models in canonical order")
    if spec.horizons != EXPECTED_HORIZONS:
        raise ValueError("formal matrix must contain the complete horizons in canonical order")
    return spec
