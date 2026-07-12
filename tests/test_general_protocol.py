from __future__ import annotations

import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

try:
    from bstalignment.general_protocol import GeneralForecastProtocol, fit_train_scaler, split_bounds
except ModuleNotFoundError:
    GeneralForecastProtocol = None
    fit_train_scaler = None
    split_bounds = None


class GeneralProtocolTests(unittest.TestCase):
    def test_etth_uses_fixed_official_borders_and_ignores_trailing_rows(self):
        self.assertIsNotNone(split_bounds, "general protocol module must define split_bounds")
        self.assertEqual(
            split_bounds("ETTh1", n_rows=20_000, input_len=36),
            {"train": (0, 8_640), "val": (8_640, 11_520), "test": (11_520, 14_400)},
        )

    def test_ettm_uses_fixed_official_borders_and_ignores_trailing_rows(self):
        self.assertIsNotNone(split_bounds, "general protocol module must define split_bounds")
        self.assertEqual(
            split_bounds("ETTm2", n_rows=60_000, input_len=36),
            {"train": (0, 34_560), "val": (34_560, 46_080), "test": (46_080, 57_600)},
        )

    def test_non_ett_datasets_use_chronological_seventy_ten_twenty_borders(self):
        self.assertIsNotNone(split_bounds, "general protocol module must define split_bounds")
        self.assertEqual(
            split_bounds("ECL", n_rows=1_001, input_len=36),
            {"train": (0, 700), "val": (700, 800), "test": (800, 1_001)},
        )

    def test_protocol_rejects_nonformal_datasets(self):
        self.assertIsNotNone(split_bounds, "general protocol module must define split_bounds")
        for dataset in ("Traffic", "ETTh3", "ETTm3", "FRED"):
            with self.subTest(dataset=dataset), self.assertRaisesRegex(ValueError, "unknown formal general dataset"):
                split_bounds(dataset, n_rows=60_000, input_len=36)

    def test_validation_and_test_targets_start_at_their_boundaries_and_end_inside_them(self):
        self.assertIsNotNone(GeneralForecastProtocol, "general protocol module must define GeneralForecastProtocol")
        protocol = GeneralForecastProtocol("ECL", n_rows=10_000, input_len=36)
        for pred_len in (24, 36, 48, 60):
            for split in ("val", "test"):
                starts = protocol.window_index(split, pred_len=pred_len)
                target_start = starts + 36
                target_end = target_start + pred_len
                boundary_start, boundary_end = protocol.bounds[split]
                self.assertEqual(target_start[0], boundary_start)
                self.assertLessEqual(target_end[-1], boundary_end)

    def test_scaler_ignores_validation_and_test_outliers(self):
        self.assertIsNotNone(fit_train_scaler, "general protocol module must define fit_train_scaler")
        values = np.array([[0.0, 2.0], [2.0, 4.0], [10_000.0, -10_000.0], [-10_000.0, 10_000.0]])
        scaler = fit_train_scaler(values, train_end=2)
        np.testing.assert_allclose(scaler.mean, [1.0, 3.0])
        np.testing.assert_allclose(scaler.std, [1.0, 1.0])

    def test_training_targets_never_cross_the_validation_boundary(self):
        self.assertIsNotNone(GeneralForecastProtocol, "general protocol module must define GeneralForecastProtocol")
        protocol = GeneralForecastProtocol("ECL", n_rows=2_000, input_len=36)
        starts = protocol.window_index("train", pred_len=24)
        train_end = protocol.bounds["train"][1]
        values = np.zeros((2_000, 1), dtype=np.float32)
        values[train_end, 0] = 999_999.0
        training_targets = np.concatenate([values[start + 36 : start + 36 + 24, 0] for start in starts])
        self.assertLessEqual((starts[-1] + 36 + 24), train_end)
        self.assertNotIn(999_999.0, training_targets)

    def test_dataset_shares_history_but_keeps_validation_target_at_boundary(self):
        try:
            from bstalignment.data_general import GeneralForecastGraphDataset
        except ImportError as error:
            self.fail(f"general graph dataset must use the canonical protocol: {error}")
        with TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "processed" / "general" / "ECL"
            data_dir.mkdir(parents=True)
            frame = pd.DataFrame(
                {
                    "date": pd.date_range("2020-01-01", periods=1_000, freq="h"),
                    "load": np.arange(1_000, dtype=float),
                    "temperature": np.arange(1_000, dtype=float) * 2,
                }
            )
            frame.to_csv(data_dir / "ECL.csv", index=False)
            dataset = GeneralForecastGraphDataset("ECL", data_root=str(root), split="val", pred_len=24, fit_scaler=True)
            sample = dataset[0]
        self.assertEqual(dataset.input_len, 36)
        self.assertEqual(sample["start_index"], 700 - 36)
        self.assertEqual(sample["target_steps"][0].item(), 700)
        self.assertEqual(sample["target_steps"][-1].item(), 723)
        np.testing.assert_allclose(sample["history_raw"].numpy()[0], [664.0, 1_328.0])
        self.assertEqual(sample["columns"], ("load", "temperature"))
        self.assertEqual(sample["scaler_metadata"]["train_end"], 700)

    def test_general_trainer_defaults_to_the_formal_history_length(self):
        from bstalignment.train_graph_report import parse_args

        with patch("sys.argv", ["train_graph_report", "--variant", "general", "--pred_len", "24"]):
            args = parse_args()
        self.assertEqual(args.input_len, 36)

    def test_general_trainer_rejects_the_shared_battery_horizon_default(self):
        from bstalignment.train_graph_report import parse_args

        stderr = io.StringIO()
        with patch("sys.argv", ["train_graph_report", "--variant", "general"]), patch("sys.stderr", stderr):
            with self.assertRaises(SystemExit):
                parse_args()
        self.assertIn("general forecasting requires --pred_len", stderr.getvalue())

    def test_general_trainer_rejects_a_nonformal_history_length_before_loading_data(self):
        from bstalignment.train_graph_report import parse_args

        stderr = io.StringIO()
        with patch(
            "sys.argv",
            ["train_graph_report", "--variant", "general", "--input_len", "35", "--pred_len", "24"],
        ), patch("sys.stderr", stderr):
            with self.assertRaises(SystemExit):
                parse_args()
        self.assertIn("general forecasting requires --input_len 36", stderr.getvalue())

    def test_battery_parser_retains_its_shared_twenty_step_default(self):
        from bstalignment.train_graph_report import parse_args

        with patch("sys.argv", ["train_graph_report", "--variant", "battery"]):
            args = parse_args()
        self.assertEqual(args.pred_len, 20)
        self.assertEqual(args.input_len, 96)

    def test_window_index_rejects_unsupported_formal_horizons(self):
        protocol = GeneralForecastProtocol("ECL", n_rows=10_000, input_len=36)

        with self.assertRaisesRegex(ValueError, "unsupported formal prediction length"):
            protocol.window_index("train", pred_len=23)

    def test_timestamp_column_is_preserved_as_timestamp_metadata(self):
        from bstalignment.data_general import GeneralForecastGraphDataset

        with TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "processed" / "general" / "ECL"
            data_dir.mkdir(parents=True)
            timestamps = pd.date_range("2020-01-01", periods=1_000, freq="h")
            pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "load": np.arange(1_000, dtype=float),
                    "temperature": np.arange(1_000, dtype=float) * 2,
                }
            ).to_csv(data_dir / "ECL.csv", index=False)
            sample = GeneralForecastGraphDataset("ECL", data_root=str(root), split="val", pred_len=24)[0]
        self.assertEqual(sample["timestamp_markers"]["history"][0], timestamps[664].to_datetime64())


if __name__ == "__main__":
    unittest.main()
