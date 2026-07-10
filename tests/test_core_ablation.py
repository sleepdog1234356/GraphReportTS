from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

import bstalignment.data_battery_raw as battery_data
from bstalignment.data_battery_raw import BatteryRawGraphDataset, collate_graph_report_batch
from bstalignment.raw_signal import (
    BATTERY_SEQUENCE_CHANNELS,
    FULL_BATTERY_PROMPT_MAP_NAMES,
    build_battery_sequence,
    build_multiview_maps,
)


class ResampledBatterySequenceTests(unittest.TestCase):
    def channels(self):
        return {
            "current": np.linspace(0.0, 1.0, 17, dtype=np.float32),
            "voltage": np.linspace(3.0, 4.2, 17, dtype=np.float32),
            "temperature": np.linspace(25.0, 35.0, 17, dtype=np.float32),
            "capacity": np.linspace(0.0, 1.1, 17, dtype=np.float32),
        }

    def test_sequence_has_fixed_six_channel_contract(self):
        values, names = build_battery_sequence(self.channels(), resample_len=16)
        self.assertEqual(tuple(names), BATTERY_SEQUENCE_CHANNELS)
        self.assertEqual(values.shape, (16, 6))
        self.assertEqual(values.dtype, np.float32)
        self.assertTrue(np.isfinite(values).all())

    def test_full_prompt_names_match_current_full_map_order(self):
        _, names = build_multiview_maps(
            self.channels(),
            resample_len=16,
            delay_dim=2,
            delay_lag=1,
            include_derivatives=True,
            include_hankel=True,
            include_ic_dv=True,
        )
        self.assertEqual(tuple(names[:10]), FULL_BATTERY_PROMPT_MAP_NAMES)

    def test_missing_formal_channel_is_rejected(self):
        channels = self.channels()
        channels.pop("temperature")
        with self.assertRaisesRegex(ValueError, "temperature"):
            build_battery_sequence(channels, resample_len=16)


class SequenceDatasetTests(unittest.TestCase):
    @staticmethod
    def _batch_item(input_key: str):
        item = {
            "history_features": torch.ones(32, 8),
            "history_cycles": torch.arange(32),
            "y": torch.ones(20),
            "mask": torch.ones(20, dtype=torch.bool),
            "horizon": torch.tensor(20),
            "prompt": "prompt",
            "cell_id": "cell",
            "cycle": 32,
            "target_steps": torch.arange(1, 21),
        }
        item[input_key] = torch.ones(32, 16, 6) if input_key == "raw_sequences" else torch.ones(32, 18, 8, 9)
        return item

    def test_sequence_mode_returns_no_maps(self):
        from tests.test_training_strategy import BatteryDataFixtureMixin

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            BatteryDataFixtureMixin.write_processed_data(root, [1, 1, 1, 1])
            ds = BatteryRawGraphDataset(
                dataset_name="calce",
                data_root=root,
                split="train",
                history_len=32,
                max_horizon=20,
                resample_len=16,
                input_representation="sequence",
                include_ic_dv=True,
            )
            item = ds[0]
            self.assertNotIn("maps", item)
            self.assertEqual(item["raw_sequences"].shape, (32, 16, 6))

    def test_sequence_collation_preserves_formal_shapes(self):
        item = {
            "raw_sequences": torch.ones(32, 16, 6),
            "history_features": torch.ones(32, 8),
            "history_cycles": torch.arange(32),
            "y": torch.ones(20),
            "mask": torch.ones(20, dtype=torch.bool),
            "horizon": torch.tensor(20),
            "prompt": "prompt",
            "cell_id": "cell",
            "cycle": 32,
            "target_steps": torch.arange(1, 21),
        }
        batch = collate_graph_report_batch([item, item])
        self.assertNotIn("maps", batch)
        self.assertEqual(batch["raw_sequences"].shape, (2, 32, 16, 6))
        self.assertEqual(batch["y"].shape, (2, 20))

    def test_graph_and_sequence_samples_preserve_prompt_and_target_identity(self):
        from tests.test_training_strategy import BatteryDataFixtureMixin

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            BatteryDataFixtureMixin.write_processed_data(root, [1, 1, 1, 1])
            graph = BatteryRawGraphDataset(
                dataset_name="calce",
                data_root=root,
                split="train",
                history_len=32,
                max_horizon=20,
                resample_len=16,
                input_representation="graph",
                include_ic_dv=True,
            )
            sequence = BatteryRawGraphDataset(
                dataset_name="calce",
                data_root=root,
                split="train",
                history_len=32,
                max_horizon=20,
                resample_len=16,
                input_representation="sequence",
                include_ic_dv=True,
            )
            graph_item = graph[0]
            sequence_item = sequence[0]
            self.assertEqual(graph_item["prompt"], sequence_item["prompt"])
            self.assertEqual(graph_item["cell_id"], sequence_item["cell_id"])
            self.assertEqual(graph_item["cycle"], sequence_item["cycle"])
            torch.testing.assert_close(graph_item["y"], sequence_item["y"], rtol=0.0, atol=0.0)

    def test_unknown_representation_is_rejected(self):
        from tests.test_training_strategy import BatteryDataFixtureMixin

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            BatteryDataFixtureMixin.write_processed_data(root, [1, 1, 1, 1])
            with self.assertRaisesRegex(ValueError, "Unknown battery input_representation: invalid"):
                BatteryRawGraphDataset(dataset_name="calce", data_root=root, input_representation="invalid")

    def test_formal_sequence_requires_ic_dv(self):
        from tests.test_training_strategy import BatteryDataFixtureMixin

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            BatteryDataFixtureMixin.write_processed_data(root, [1, 1, 1, 1])
            with self.assertRaisesRegex(ValueError, "Formal sequence representation requires IC/DV"):
                BatteryRawGraphDataset(
                    dataset_name="calce",
                    data_root=root,
                    input_representation="sequence",
                    include_ic_dv=False,
                )

    def test_mixed_representation_collation_is_rejected(self):
        graph_item = self._batch_item("maps")
        sequence_item = self._batch_item("raw_sequences")
        with self.assertRaisesRegex(ValueError, "Cannot collate mixed graph and sequence battery samples"):
            collate_graph_report_batch([graph_item, sequence_item])

    def test_graph_only_cycle_wrapper_rejects_sequence_dataset(self):
        from tests.test_training_strategy import BatteryDataFixtureMixin

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            BatteryDataFixtureMixin.write_processed_data(root, [1, 1, 1, 1])
            ds = BatteryRawGraphDataset(
                dataset_name="calce",
                data_root=root,
                input_representation="sequence",
                resample_len=16,
            )
            with self.assertRaisesRegex(RuntimeError, "Graph cycle maps require input_representation=graph"):
                ds._processed_cycle_maps(ds.processed_cells[0], 0)

    def test_sequence_cache_config_and_path_encode_formal_input_identity(self):
        config_factory = getattr(battery_data, "battery_sequence_cache_config", None)
        path_factory = getattr(battery_data, "battery_sequence_cache_path", None)
        self.assertIsNotNone(config_factory, "battery_sequence_cache_config is missing")
        self.assertIsNotNone(path_factory, "battery_sequence_cache_path is missing")
        config = config_factory(
            dataset_name="CALCE",
            split="train",
            max_horizon=20,
            resample_len=16,
            allow_summary_fallback=False,
            seed=42,
            max_cycles=None,
            history_len=32,
        )
        self.assertEqual(config["version"], "battery-sequence-cycle-history-v1")
        self.assertEqual(config["dataset"], "calce")
        self.assertEqual(config["channel_order"], list(BATTERY_SEQUENCE_CHANNELS))
        self.assertEqual(config["ic_dv_formula_version"], "robust-scaled-smoothed-gradient-v1")
        cache_path = path_factory("cache", config)
        self.assertEqual(cache_path.parent, Path("cache") / "calce" / "train")
        self.assertEqual(len(cache_path.name), 12)
        self.assertEqual(cache_path, path_factory(Path("cache"), dict(config)))
