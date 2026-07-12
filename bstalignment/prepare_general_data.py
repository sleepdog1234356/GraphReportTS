"""Prepare checksum-verified canonical CSVs for formal general forecasting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from .general_data_schema import DatasetManifest, dataset_schema, sha256_file, validate_frame
from .general_experiment_config import DatasetSpec, load_general_experiment_spec


def _impute_values(values: pd.DataFrame, train_rows: int) -> tuple[pd.DataFrame, dict[str, object]]:
    nonfinite = ~np.isfinite(values.to_numpy(dtype=float, copy=False))
    cleaned = values.mask(nonfinite)
    original_missing = cleaned.isna()
    forward_filled = cleaned.ffill()
    forward_fill_cells = int((original_missing & forward_filled.notna()).to_numpy().sum())
    medians = forward_filled.iloc[:train_rows].median(axis=0)
    if medians.isna().any():
        column = str(medians[medians.isna()].index[0])
        raise ValueError(f"training values for {column} cannot supply a fallback median")
    imputed = forward_filled.fillna(medians)
    median_fill_cells = int((forward_filled.isna() & imputed.notna()).to_numpy().sum())
    if not np.isfinite(imputed.to_numpy(dtype=float, copy=False)).all():
        raise ValueError("imputation did not produce finite values")
    return imputed, {
        "method": "causal_forward_fill_then_train_median",
        "training_rows": train_rows,
        "nonfinite_values_replaced": int((~np.isfinite(values.to_numpy(dtype=float, copy=False))).sum()),
        "forward_fill_cells": forward_fill_cells,
        "median_fill_cells": median_fill_cells,
        "median_values": {str(column): float(value) for column, value in medians.items()},
    }


def _atomic_write(path: Path, contents: str) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(contents, encoding="utf-8")
    temporary.replace(path)


def prepare_dataset(spec: DatasetSpec, raw_path: Path, output_root: Path) -> DatasetManifest:
    """Validate *raw_path* and emit a canonical CSV plus immutable manifest."""

    raw_path = Path(raw_path)
    raw_sha256 = sha256_file(raw_path)
    if raw_sha256 != spec.raw_sha256:
        raise ValueError(f"raw checksum mismatch for {spec.name}")
    schema = dataset_schema(spec.name)
    validated = validate_frame(schema, pd.read_csv(raw_path))
    train_rows = schema.training_rows(len(validated.frame))
    values, imputation = _impute_values(validated.frame.iloc[:, 1:], train_rows)
    canonical = values.copy()
    canonical.insert(0, "date", validated.frame["date"])

    dataset_dir = Path(output_root) / spec.name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    processed_path = dataset_dir / f"{spec.name}.csv"
    csv_contents = canonical.to_csv(index=False, lineterminator="\n")
    _atomic_write(processed_path, csv_contents)
    manifest = DatasetManifest(
        name=spec.name,
        raw_path=str(raw_path),
        raw_sha256=raw_sha256,
        processed_path=str(processed_path),
        processed_sha256=sha256_file(processed_path),
        row_count=len(canonical),
        feature_count=len(validated.value_columns),
        timestamp_column=validated.timestamp_column,
        value_columns=validated.value_columns,
        expected_frequency=str(schema.frequency),
        timestamp_exceptions=validated.timestamp_exceptions,
        imputation=imputation,
    )
    _atomic_write(dataset_dir / "manifest.json", json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/general_forecasting/experiment_matrix.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("bstalignment/data/processed/general"))
    args = parser.parse_args()
    spec = load_general_experiment_spec(args.config)
    for dataset in spec.datasets:
        manifest = prepare_dataset(dataset, Path(dataset.raw_path), args.output_root)
        print(f"prepared {manifest.name}: {manifest.feature_count} features, {manifest.processed_sha256}")


if __name__ == "__main__":
    main()
