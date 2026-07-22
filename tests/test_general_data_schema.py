from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from anchoredgtr.general_data_schema import dataset_schema, sha256_file, validate_frame
from anchoredgtr.general_experiment_config import DatasetSpec
from anchoredgtr.prepare_general_data import prepare_dataset


class GeneralDataSchemaTests(unittest.TestCase):
    def write_csv(self, directory: Path, name: str, frame: pd.DataFrame) -> Path:
        path = directory / name
        frame.to_csv(path, index=False)
        return path

    def spec_for(self, name: str, path: Path) -> DatasetSpec:
        return DatasetSpec(name=name, raw_path=str(path), raw_sha256=sha256_file(path))

    def hourly_frame(self, values: dict[str, list[object]] | None = None) -> pd.DataFrame:
        values = dict(values or {"first": [1.0, 2.0, 3.0], "second": [4.0, 5.0, 6.0]})
        while len(values) < dataset_schema("ETTh1").expected_feature_count:
            values[f"filler_{len(values)}"] = [0.0, 0.0, 0.0]
        return pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=3, freq="h"), **values})

    def test_schema_defines_formal_frequency_and_feature_count(self):
        self.assertEqual(dataset_schema("ETTh1").frequency, pd.Timedelta(hours=1))
        self.assertEqual(dataset_schema("ETTm1").frequency, pd.Timedelta(minutes=15))
        self.assertEqual(dataset_schema("ECL").expected_feature_count, 321)
        self.assertEqual(dataset_schema("Weather").expected_feature_count, 21)

    def test_validation_rejects_wrong_numeric_feature_count(self):
        frame = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=3, freq="h"),
                **{f"feature_{index}": [1.0, 2.0, 3.0] for index in range(6)},
            }
        )
        with self.assertRaisesRegex(ValueError, "feature count"):
            validate_frame(dataset_schema("ETTh1"), frame)

    def test_validation_rejects_unparseable_timestamp(self):
        frame = self.hourly_frame()
        frame["timestamp"] = frame["timestamp"].astype(object)
        frame.loc[1, "timestamp"] = "not-a-time"
        with self.assertRaisesRegex(ValueError, "timestamp"):
            validate_frame(dataset_schema("ETTh1"), frame)

    def test_validation_rejects_non_monotonic_timestamp(self):
        frame = self.hourly_frame()
        frame.loc[2, "timestamp"] = "2024-01-01 00:30:00"
        with self.assertRaisesRegex(ValueError, "monotonic"):
            validate_frame(dataset_schema("ETTh1"), frame)

    def test_validation_rejects_duplicate_timestamp(self):
        frame = self.hourly_frame()
        frame.loc[2, "timestamp"] = frame.loc[1, "timestamp"]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_frame(dataset_schema("ETTh1"), frame)

    def test_validation_rejects_unexpected_frequency(self):
        frame = self.hourly_frame()
        frame.loc[2, "timestamp"] = "2024-01-01 03:00:00"
        with self.assertRaisesRegex(ValueError, "frequency"):
            validate_frame(dataset_schema("ETTh1"), frame)

    def test_validation_rejects_target_leakage_column(self):
        frame = self.hourly_frame({"observed": [1.0, 2.0, 3.0], "target": [4.0, 5.0, 6.0]})
        with self.assertRaisesRegex(ValueError, "target leakage"):
            validate_frame(dataset_schema("ETTh1"), frame)

    def test_weather_validation_records_documented_source_exceptions(self):
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2020-01-01 00:00", "2020-01-01 00:10", "2020-01-01 00:10", "2020-01-01 01:50"]
                ),
                "value": [1.0, 2.0, 3.0, 4.0],
                **{f"filler_{index}": [0.0, 0.0, 0.0, 0.0] for index in range(20)},
            }
        )
        validated = validate_frame(dataset_schema("Weather"), frame)
        self.assertEqual(validated.timestamp_exceptions["duplicate_timestamps"], 1)
        self.assertEqual(validated.timestamp_exceptions["nonstandard_intervals"], {"0 days 01:40:00": 1})

    def test_prepare_preserves_numeric_column_order_and_canonicalizes_date(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = self.write_csv(root, "raw.csv", self.hourly_frame({"z_last": [1.0, 2.0, 3.0], "a_first": [4.0, 5.0, 6.0]}))
            manifest = prepare_dataset(self.spec_for("ETTh1", raw), raw, root / "processed")
            processed = pd.read_csv(root / "processed" / "ETTh1" / "ETTh1.csv")
            self.assertEqual(list(processed.columns), ["date", "z_last", "a_first", "filler_2", "filler_3", "filler_4", "filler_5", "filler_6"])
            self.assertEqual(manifest.feature_count, 7)

    def test_prepare_uses_causal_fill_then_train_median_for_leading_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = 8_640
            frame = pd.DataFrame(
                {
                    "date": pd.date_range("2020-01-01", periods=rows, freq="h"),
                    "leading": [float("nan")] + [2.0] * (rows - 1),
                    "causal": [1.0, float("nan")] + [3.0] * (rows - 2),
                    **{f"filler_{index}": [0.0] * rows for index in range(5)},
                }
            )
            raw = self.write_csv(root, "raw.csv", frame)
            manifest = prepare_dataset(self.spec_for("ETTh1", raw), raw, root / "processed")
            processed = pd.read_csv(root / "processed" / "ETTh1" / "ETTh1.csv")
            self.assertEqual(processed.loc[0, "leading"], 2.0)
            self.assertEqual(processed.loc[1, "causal"], 1.0)
            self.assertEqual(manifest.imputation["forward_fill_cells"], 1)
            self.assertEqual(manifest.imputation["median_fill_cells"], 1)
            self.assertTrue(pd.notna(processed.iloc[:, 1:]).all().all())

    def test_prepare_rejects_raw_checksum_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = self.write_csv(root, "raw.csv", self.hourly_frame())
            spec = DatasetSpec(name="ETTh1", raw_path=str(raw), raw_sha256="0" * 64)
            with self.assertRaisesRegex(ValueError, "checksum"):
                prepare_dataset(spec, raw, root / "processed")

    def test_manifest_records_raw_and_processed_sha256(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = self.write_csv(root, "raw.csv", self.hourly_frame())
            manifest = prepare_dataset(self.spec_for("ETTh1", raw), raw, root / "processed")
            csv_path = root / "processed" / "ETTh1" / "ETTh1.csv"
            manifest_path = root / "processed" / "ETTh1" / "manifest.json"
            self.assertEqual(manifest.raw_sha256, sha256_file(raw))
            self.assertEqual(manifest.processed_sha256, sha256_file(csv_path))
            self.assertEqual(json.loads(manifest_path.read_text(encoding="utf-8"))["processed_sha256"], manifest.processed_sha256)


if __name__ == "__main__":
    unittest.main()
