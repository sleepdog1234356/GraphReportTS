"""Schema and provenance records for canonical general forecasting data."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Mapping

import pandas as pd


LEAKAGE_COLUMN_NAMES = frozenset(
    {
        "forecast",
        "forecasts",
        "future",
        "future_value",
        "future_values",
        "label",
        "labels",
        "prediction",
        "predictions",
        "target",
        "target_value",
        "target_values",
        "y",
    }
)


@dataclass(frozen=True)
class DatasetSchema:
    name: str
    frequency: pd.Timedelta
    expected_feature_count: int
    train_rows: int | None = None
    train_ratio: float | None = None
    allowed_nonstandard_intervals: frozenset[pd.Timedelta] = frozenset()
    allows_duplicate_timestamps: bool = False

    def training_rows(self, row_count: int) -> int:
        if self.train_rows is not None:
            return min(self.train_rows, row_count)
        if self.train_ratio is None:
            raise ValueError(f"{self.name} has no formal training boundary")
        return int(row_count * self.train_ratio)


@dataclass(frozen=True)
class ValidatedFrame:
    frame: pd.DataFrame
    timestamp_column: str
    value_columns: tuple[str, ...]
    timestamp_exceptions: Mapping[str, object]


@dataclass(frozen=True)
class DatasetManifest:
    name: str
    raw_path: str
    raw_sha256: str
    processed_path: str
    processed_sha256: str
    row_count: int
    feature_count: int
    timestamp_column: str
    value_columns: tuple[str, ...]
    expected_frequency: str
    timestamp_exceptions: Mapping[str, object]
    imputation: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


_SCHEMAS = {
    "ETTh1": DatasetSchema("ETTh1", pd.Timedelta(hours=1), 7, train_rows=8_640),
    "ETTh2": DatasetSchema("ETTh2", pd.Timedelta(hours=1), 7, train_rows=8_640),
    "ETTm1": DatasetSchema("ETTm1", pd.Timedelta(minutes=15), 7, train_rows=34_560),
    "ETTm2": DatasetSchema("ETTm2", pd.Timedelta(minutes=15), 7, train_rows=34_560),
    "ECL": DatasetSchema("ECL", pd.Timedelta(hours=1), 321, train_ratio=0.7),
    "Weather": DatasetSchema(
        "Weather",
        pd.Timedelta(minutes=10),
        21,
        train_ratio=0.7,
        allowed_nonstandard_intervals=frozenset({pd.Timedelta(minutes=100)}),
        allows_duplicate_timestamps=True,
    ),
}


def dataset_schema(name: str) -> DatasetSchema:
    """Return the frozen schema for a formal general forecasting dataset."""

    try:
        return _SCHEMAS[name]
    except KeyError as error:
        raise ValueError(f"unknown general dataset: {name}") from error


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 fingerprint of *path*."""

    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_frame(schema: DatasetSchema, frame: pd.DataFrame) -> ValidatedFrame:
    """Validate raw CSV structure without changing value-column ordering."""

    if frame.empty:
        raise ValueError(f"{schema.name} raw CSV must contain rows")
    if len(frame.columns) < 2:
        raise ValueError(f"{schema.name} raw CSV requires a timestamp and numeric values")
    timestamp_column = str(frame.columns[0])
    parsed_timestamps = pd.to_datetime(frame.iloc[:, 0], errors="coerce")
    if parsed_timestamps.isna().any():
        raise ValueError(f"{schema.name} contains an unparseable timestamp")

    timestamp_deltas = parsed_timestamps.diff().iloc[1:]
    if (timestamp_deltas < pd.Timedelta(0)).any():
        raise ValueError(f"{schema.name} timestamps must be monotonic")
    duplicate_count = int((timestamp_deltas == pd.Timedelta(0)).sum())
    if duplicate_count and not schema.allows_duplicate_timestamps:
        raise ValueError(f"{schema.name} contains duplicate timestamps")

    nonstandard = timestamp_deltas[(timestamp_deltas != schema.frequency) & (timestamp_deltas != pd.Timedelta(0))]
    unexpected = set(nonstandard.unique()) - set(schema.allowed_nonstandard_intervals)
    if unexpected:
        raise ValueError(f"{schema.name} timestamps violate expected frequency {schema.frequency}")

    value_columns = tuple(str(column) for column in frame.columns[1:])
    leaked = sorted(column for column in value_columns if column.strip().lower() in LEAKAGE_COLUMN_NAMES)
    if leaked:
        raise ValueError(f"{schema.name} contains target leakage column: {leaked[0]}")
    numeric = frame.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    invalid = numeric.isna() & frame.iloc[:, 1:].notna()
    if invalid.any().any():
        column = str(invalid.any()[invalid.any()].index[0])
        raise ValueError(f"{schema.name} has non-numeric values in {column}")

    validated = numeric.copy()
    validated.columns = value_columns
    validated.insert(0, "date", parsed_timestamps)
    return ValidatedFrame(
        frame=validated,
        timestamp_column=timestamp_column,
        value_columns=value_columns,
        timestamp_exceptions={
            "duplicate_timestamps": duplicate_count,
            "nonstandard_intervals": {
                str(interval): int(count) for interval, count in nonstandard.value_counts().sort_index().items()
            },
        },
    )
