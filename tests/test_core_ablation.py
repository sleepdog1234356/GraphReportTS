from __future__ import annotations

from argparse import Namespace
from dataclasses import asdict
from hashlib import sha256
import json
from multiprocessing.reduction import ForkingPickler
from pathlib import Path
import pickle
import re
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

import bstalignment.data_battery_raw as battery_data
import bstalignment.precompute_battery_sequence_cache as sequence_cache_precompute
from bstalignment.data_battery_raw import BatteryRawGraphDataset, collate_graph_report_batch
from bstalignment.graph_report_model import GraphReportTS, GraphReportTSConfig
from bstalignment.graph_report_losses import masked_regression_loss
from bstalignment.precompute_battery_sequence_cache import precompute_sequence_split
from bstalignment.raw_signal import (
    BATTERY_SEQUENCE_CHANNELS,
    FULL_BATTERY_PROMPT_MAP_NAMES,
    build_battery_sequence,
    build_multiview_maps,
)
from bstalignment.training_strategy import MAIN_TRAINING_PROFILE, build_graph_report_optimizer


class CoreAblationModelTests(unittest.TestCase):
    def config(self, **updates):
        values = dict(
            variant="battery", d_model=8, output_dim=1, graph_layers=1,
            patch_size=2, patch_stride=1, topk_edges=1,
            use_hf_text_encoder=False, temporal_heads=2,
            raw_sequence_len=16, raw_sequence_dim=6,
        )
        values.update(updates)
        return GraphReportTSConfig(**values)

    def test_raw_sequence_model_has_no_graph_encoder(self):
        model = GraphReportTS(self.config(battery_input_mode="raw_sequence"))
        self.assertIsNone(model.graph_encoder)
        self.assertIsNotNone(model.raw_sequence_encoder)
        out = model(
            None, ["battery prompt", "battery prompt"], torch.tensor([20, 20]),
            history_features=torch.randn(2, 32, 8),
            raw_sequences=torch.randn(2, 32, 16, 6),
        )
        self.assertEqual(out["pred"].shape, (2, 20, 1))

    def test_scalar_prediction_keeps_dimension_used_by_inference_consumer(self):
        model = GraphReportTS(self.config(use_report_prompt=False))
        out = model(torch.randn(1, 32, 3, 2, 3), ["p"], 3)
        self.assertEqual(out["pred"].ndim, 3)
        self.assertIsInstance(float(out["pred"][0, 0, 0]), float)

    def test_scalar_prediction_accepts_two_dimensional_battery_loss_targets(self):
        model = GraphReportTS(self.config(use_report_prompt=False))
        out = model(torch.randn(2, 32, 3, 2, 3), ["p", "p"], 3)
        self.assertEqual(out["pred"].shape, (2, 3, 1))
        loss = masked_regression_loss(
            out["pred"],
            torch.zeros(2, 3),
            torch.ones(2, 3, dtype=torch.bool),
        )
        self.assertTrue(torch.isfinite(loss))

    def test_no_gate_has_constant_one_and_no_gate_parameters(self):
        model = GraphReportTS(self.config(use_text_gate=False))
        self.assertIsNone(model.semantic_fusion.gate)
        out = model(
            torch.randn(2, 32, 3, 2, 3),
            ["p", "p"],
            torch.tensor([20, 20]),
            history_features=torch.randn(2, 32, 8),
        )
        torch.testing.assert_close(out["gate"], torch.ones_like(out["gate"]))

    def test_no_prompt_constructs_no_semantic_modules(self):
        model = GraphReportTS(self.config(use_report_prompt=False))
        self.assertIsNone(model.text_encoder)
        self.assertIsNone(model.semantic_fusion)
        self.assertFalse(
            any(
                name.startswith(("text_encoder", "semantic_fusion", "fusion"))
                for name, _ in model.named_parameters()
            )
        )

    def test_graph_model_has_no_raw_sequence_encoder(self):
        model = GraphReportTS(self.config())
        self.assertIsNotNone(model.graph_encoder)
        self.assertIsNone(model.raw_sequence_encoder)

    def test_raw_sequence_encoder_is_unpatched_two_layer_transformer(self):
        model = GraphReportTS(self.config(battery_input_mode="raw_sequence"))
        encoder = model.raw_sequence_encoder
        self.assertEqual(encoder.input_proj.in_features, 6)
        self.assertEqual(encoder.pos_embed.num_embeddings, 16)
        self.assertEqual(len(encoder.encoder.layers), 2)
        self.assertFalse(hasattr(encoder, "patch_size"))

    def test_input_modes_reject_the_other_path_payload(self):
        graph_model = GraphReportTS(self.config(use_report_prompt=False))
        with self.assertRaisesRegex(ValueError, "hankel_graph mode requires maps and forbids raw_sequences"):
            graph_model(
                torch.randn(1, 32, 3, 2, 3),
                ["p"],
                20,
                raw_sequences=torch.randn(1, 32, 16, 6),
            )
        raw_model = GraphReportTS(
            self.config(battery_input_mode="raw_sequence", use_report_prompt=False)
        )
        with self.assertRaisesRegex(ValueError, "raw_sequence mode requires raw_sequences and forbids maps"):
            raw_model(torch.randn(1, 32, 3, 2, 3), ["p"], 20)

    def test_unknown_battery_input_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown battery_input_mode: invalid"):
            GraphReportTS(self.config(battery_input_mode="invalid"))

    def test_raw_encoder_optimizer_parameters_are_all_core(self):
        model = GraphReportTS(self.config(battery_input_mode="raw_sequence"))
        optimizer = build_graph_report_optimizer(model, MAIN_TRAINING_PROFILE)
        parameter_roles = {
            id(parameter): group["role"]
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        raw_parameters = dict(model.raw_sequence_encoder.named_parameters())
        self.assertTrue(raw_parameters)
        self.assertEqual(
            {parameter_roles[id(parameter)] for parameter in raw_parameters.values()},
            {"core"},
        )

    def test_no_prompt_forward_keeps_semantic_outputs_inactive(self):
        model = GraphReportTS(self.config(use_report_prompt=False))
        out = model(
            torch.randn(2, 32, 3, 2, 3),
            ["ignored", "ignored"],
            torch.tensor([20, 20]),
            history_features=torch.randn(2, 32, 8),
        )
        torch.testing.assert_close(out["gate"], torch.zeros_like(out["gate"]))
        self.assertIsNone(out["cross_attn"])

    def test_default_full_state_dict_matches_c2ba958_golden_after_serialization(self):
        cfg = self.config()
        original = GraphReportTS(cfg)
        restored = GraphReportTS(GraphReportTSConfig(**asdict(cfg)))
        maps = torch.randn(1, 32, 3, 2, 3)
        history = torch.randn(1, 32, 8)
        original(maps, ["p"], 2, history_features=history)
        restored(maps, ["p"], 2, history_features=history)
        original_state = original.state_dict()
        restored_state = restored.state_dict()
        expected_signature = "22fd1d49bb51d837eecc1155a3d14ae1c82bde464464f929de73f72213e4ba9f"
        for state in (original_state, restored_state):
            contract = "\n".join(
                f"{name}:{tuple(value.shape)}" for name, value in state.items()
            )
            self.assertEqual(len(state), 105)
            self.assertEqual(sha256(contract.encode()).hexdigest(), expected_signature)


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

    def test_sequence_representation_rejects_graph_cache_arguments(self):
        from tests.test_training_strategy import BatteryDataFixtureMixin

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            BatteryDataFixtureMixin.write_processed_data(root, [1, 1, 1, 1])
            cases = {
                "cache directory": {"precomputed_cache_dir": str(root / "graph_cache")},
                "required cache": {"require_precomputed_cache": True},
            }
            for label, cache_kwargs in cases.items():
                with self.subTest(label=label):
                    with self.assertRaisesRegex(ValueError, "(?i)graph cache.*sequence representation"):
                        BatteryRawGraphDataset(
                            dataset_name="calce",
                            data_root=root,
                            input_representation="sequence",
                            **cache_kwargs,
                        )

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


class SequenceCacheTests(unittest.TestCase):
    @staticmethod
    def _args(
        root: Path,
        *,
        batch_size: int = 4,
        num_workers: int = 0,
        resample_len: int = 16,
    ) -> Namespace:
        return Namespace(
            dataset="calce",
            data_root=str(root),
            cache_dir=str(root / "sequence_cache"),
            pred_len=20,
            history_len=32,
            resample_len=resample_len,
            seed=42,
            max_cycles=None,
            batch_size=batch_size,
            num_workers=num_workers,
            force=True,
        )

    @staticmethod
    def _cached_kwargs(root: Path, *, resample_len: int = 16) -> dict:
        return {
            "dataset_name": "calce",
            "data_root": root,
            "split": "train",
            "history_len": 32,
            "max_horizon": 20,
            "resample_len": resample_len,
            "input_representation": "sequence",
            "precomputed_sequence_cache_dir": str(root / "sequence_cache"),
            "require_precomputed_sequence_cache": True,
        }

    def test_sequence_cache_matches_direct_dataset_without_map_calls(self):
        from tests.test_training_strategy import BatteryDataFixtureMixin

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            BatteryDataFixtureMixin.write_processed_data(root, [1, 1, 1, 1])
            args = self._args(root)
            with patch(
                "bstalignment.data_battery_raw.build_multiview_maps",
                side_effect=AssertionError("graph path called"),
            ):
                cache_path = precompute_sequence_split(args, "train")
            direct = BatteryRawGraphDataset(
                dataset_name="calce",
                data_root=root,
                split="train",
                history_len=32,
                max_horizon=20,
                resample_len=16,
                input_representation="sequence",
            )
            self.assertTrue(cache_path.exists())
            with BatteryRawGraphDataset(**self._cached_kwargs(root)) as cached:
                self.assertEqual(len(cached), len(direct))
                self.assertEqual(cached.cycle_scale, direct.cycle_scale)
                cached_arrays = (
                    cached._cache_cycle_sequences,
                    cached._cache_history_indices,
                    cached._cache_y,
                    cached._cache_mask,
                    cached._cache_horizon,
                    cached._cache_target_steps,
                    cached._cache_history_features,
                    cached._cache_history_cycles,
                )
                self.assertTrue(all(isinstance(array, np.memmap) for array in cached_arrays))
                for index in range(len(direct)):
                    from_cache = cached[index]
                    direct_item = direct[index]
                    for key in (
                        "raw_sequences",
                        "y",
                        "mask",
                        "target_steps",
                        "history_features",
                        "history_cycles",
                    ):
                        torch.testing.assert_close(
                            from_cache[key], direct_item[key], rtol=0.0, atol=0.0
                        )
                    self.assertEqual(from_cache["prompt"], direct_item["prompt"])
            self.assertTrue(all(array._mmap.closed for array in cached_arrays))

    def test_forking_pickler_reopens_memmaps_without_serializing_array_payloads(self):
        from tests.test_training_strategy import BatteryDataFixtureMixin

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            BatteryDataFixtureMixin.write_processed_data(root, [1, 1, 1, 1])
            payload_sizes = []
            array_sizes = []
            datasets = []
            try:
                for resample_len in (16, 128):
                    precompute_sequence_split(
                        self._args(root, resample_len=resample_len), "train"
                    )
                    original = BatteryRawGraphDataset(
                        **self._cached_kwargs(root, resample_len=resample_len)
                    )
                    payload = bytes(ForkingPickler.dumps(original))
                    restored = pickle.loads(payload)
                    datasets.extend((original, restored))
                    arrays = [
                        getattr(restored, name)
                        for name in battery_data.SEQUENCE_CACHE_ARRAY_ATTRS
                    ]
                    self.assertTrue(all(isinstance(array, np.memmap) for array in arrays))
                    self.assertTrue(all(array.filename is not None for array in arrays))
                    self.assertTrue(all(array._mmap is not None and not array._mmap.closed for array in arrays))
                    for key in (
                        "raw_sequences",
                        "y",
                        "mask",
                        "target_steps",
                        "history_features",
                        "history_cycles",
                    ):
                        torch.testing.assert_close(
                            restored[0][key], original[0][key], rtol=0.0, atol=0.0
                        )
                    payload_sizes.append(len(payload))
                    array_sizes.append(sum(array.nbytes for array in arrays))
            finally:
                for dataset in datasets:
                    dataset.close()
            self.assertGreater(array_sizes[1], array_sizes[0] * 3)
            self.assertLess(abs(payload_sizes[1] - payload_sizes[0]), 4096)

    def test_invalid_cycle_scale_is_rejected_before_array_loading(self):
        from tests.test_training_strategy import BatteryDataFixtureMixin

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            BatteryDataFixtureMixin.write_processed_data(root, [1, 1, 1, 1])
            cache_path = precompute_sequence_split(self._args(root), "train")
            manifest_path = cache_path / "manifest.json"
            original_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            cases = {
                "missing": None,
                "nan": float("nan"),
                "positive infinity": float("inf"),
                "negative infinity": float("-inf"),
                "zero": 0.0,
                "negative": -1.0,
                "non-numeric": "invalid",
            }
            for label, value in cases.items():
                with self.subTest(label=label):
                    manifest = json.loads(json.dumps(original_manifest))
                    if value is None:
                        manifest.pop("cycle_scale")
                    else:
                        manifest["cycle_scale"] = value
                    manifest_path.write_text(
                        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
                    )
                    with patch(
                        "bstalignment.data_battery_raw.np.load",
                        side_effect=AssertionError("arrays loaded before cycle_scale validation"),
                    ):
                        with self.assertRaisesRegex(
                            ValueError,
                            rf"{re.escape(str(cache_path))}.*cycle_scale",
                        ):
                            BatteryRawGraphDataset(**self._cached_kwargs(root))

    def test_manifest_missing_file_mapping_key_names_key_and_cache(self):
        from tests.test_training_strategy import BatteryDataFixtureMixin

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            BatteryDataFixtureMixin.write_processed_data(root, [1, 1, 1, 1])
            cache_path = precompute_sequence_split(self._args(root), "train")
            manifest_path = cache_path / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["files"].pop("y")
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                ValueError,
                rf"{re.escape(str(cache_path))}.*files.*y",
            ):
                BatteryRawGraphDataset(**self._cached_kwargs(root))

    def test_missing_required_sequence_cache_names_expected_path(self):
        from tests.test_training_strategy import BatteryDataFixtureMixin

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            BatteryDataFixtureMixin.write_processed_data(root, [1, 1, 1, 1])
            with self.assertRaisesRegex(
                FileNotFoundError,
                rf"Required battery sequence cache.*{re.escape(str(root / 'sequence_cache'))}",
            ):
                BatteryRawGraphDataset(**self._cached_kwargs(root))

    def test_corrupted_sequence_cache_is_rejected_before_activation(self):
        from tests.test_training_strategy import BatteryDataFixtureMixin

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            BatteryDataFixtureMixin.write_processed_data(root, [1, 1, 1, 1])
            cache_path = precompute_sequence_split(self._args(root), "train")
            manifest_path = cache_path / "manifest.json"
            original_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            meta_path = cache_path / original_manifest["files"]["meta"]
            original_meta = meta_path.read_text(encoding="utf-8")
            array_files = {
                name: cache_path / filename
                for name, filename in original_manifest["files"].items()
                if name != "meta"
            }
            original_arrays = {name: np.load(path).copy() for name, path in array_files.items()}

            def save_wrong_shape(name):
                array = original_arrays[name]
                corrupted = array[:, :-1] if name == "cycle_sequences" else array[:-1]
                np.save(array_files[name], corrupted)

            cases = [
                ("manifest sample_count", "sample_count", lambda manifest, arrays: manifest.__setitem__("sample_count", manifest["sample_count"] + 1)),
                ("manifest cycle_count", "cycle_count", lambda manifest, arrays: manifest.__setitem__("cycle_count", manifest["cycle_count"] + 1)),
                ("manifest cycle_sequence_shape", "cycle_sequence_shape", lambda manifest, arrays: manifest.__setitem__("cycle_sequence_shape", [15, 6])),
            ]
            for name in array_files:
                cases.append(
                    (
                        f"{name} shape",
                        rf"{name}.*shape",
                        lambda manifest, arrays, key=name: save_wrong_shape(key),
                    )
                )
                wrong_dtype = {
                    np.dtype(np.float32): np.float64,
                    np.dtype(np.int64): np.int32,
                    np.dtype(np.bool_): np.uint8,
                }[original_arrays[name].dtype]
                cases.append(
                    (
                        f"{name} dtype",
                        rf"{name}.*dtype",
                        lambda manifest, arrays, key=name, dtype=wrong_dtype: np.save(
                            array_files[key], arrays[key].astype(dtype)
                        ),
                    )
                )
            cases.extend(
                [
                    (
                        "history index bounds",
                        "history_indices.*bounds",
                        lambda manifest, arrays: np.save(
                            array_files["history_indices"],
                            np.full_like(arrays["history_indices"], manifest["cycle_count"]),
                        ),
                    ),
                    (
                        "formal horizon",
                        "horizon.*formal",
                        lambda manifest, arrays: np.save(
                            array_files["horizon"],
                            np.full_like(arrays["horizon"], 19),
                        ),
                    ),
                    (
                        "unreadable array",
                        "y.*load",
                        lambda manifest, arrays: array_files["y"].write_bytes(b"not a numpy file"),
                    ),
                    (
                        "malformed meta",
                        "meta",
                        lambda manifest, arrays: meta_path.write_text("{", encoding="utf-8"),
                    ),
                ]
            )

            for label, mismatch, corrupt in cases:
                with self.subTest(label=label):
                    manifest = json.loads(json.dumps(original_manifest))
                    for name, path in array_files.items():
                        np.save(path, original_arrays[name])
                    meta_path.write_text(original_meta, encoding="utf-8")
                    corrupt(manifest, original_arrays)
                    manifest_path.write_text(
                        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
                    )
                    with self.assertRaisesRegex(
                        ValueError,
                        rf"{re.escape(str(cache_path))}.*{mismatch}",
                    ):
                        BatteryRawGraphDataset(**self._cached_kwargs(root))

    def test_parallel_sequence_results_submit_only_batch_sized_chunks(self):
        class FakeSequenceDataset:
            input_representation = "sequence"
            processed_cells = [object()]

            @staticmethod
            def _processed_cycle_input(cell, row_idx):
                return np.full((2, 6), row_idx, dtype=np.float32), ["channel"]

        submitted_sizes = []
        real_executor = sequence_cache_precompute.ThreadPoolExecutor

        class RecordingExecutor(real_executor):
            def map(self, fn, *iterables, **kwargs):
                items = list(iterables[0])
                submitted_sizes.append(len(items))
                return super().map(fn, items, **kwargs)

        cycle_items = [
            (index, ("cell", index), ("processed", 0, index))
            for index in range(7)
        ]
        with patch.object(sequence_cache_precompute, "ThreadPoolExecutor", RecordingExecutor):
            results = list(
                sequence_cache_precompute._sequence_results(
                    FakeSequenceDataset(), cycle_items, num_workers=2, batch_size=3
                )
            )
        self.assertEqual([result[0] for result in results], list(range(7)))
        self.assertEqual(submitted_sizes, [3, 3, 1])
