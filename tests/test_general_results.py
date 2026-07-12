from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np


class GeneralResultContractTests(unittest.TestCase):
    @staticmethod
    def contract():
        from bstalignment.general_results import GeneralRunWriter

        return GeneralRunWriter

    @staticmethod
    def expected_spec():
        return {
            "dataset": "ECL",
            "dataset_checksum": "a" * 64,
            "source_commit": "1234567",
            "protocol": {"input_len": 36, "features": "M", "horizons": [96, 192, 336, 720]},
        }

    @staticmethod
    def provenance():
        return {
            "dataset_checksum": "a" * 64,
            "source_commit": "1234567",
            "protocol": {"input_len": 36, "features": "M", "horizons": [96, 192, 336, 720]},
            "source": {"url": "https://example.invalid/source", "commit": "1234567"},
            "runtime": {"wall_time_seconds": 1.25, "peak_gpu_memory_bytes": 0, "trainable_parameters": 12},
        }

    def write_complete_run(self, root: Path, *, model: str = "PatchTST") -> Path:
        Writer = self.contract()
        run_dir = root / "run"
        writer = Writer(run_dir, self.expected_spec())
        writer.write_run_config({"model": model, "dataset": "ECL", "seed": 2021, "metrics_space": "standardized"})
        writer.append_history({"epoch": 1, "train_mse": 0.9, "val_mse": 0.4})
        writer.record_validation(epoch=1, mse=0.4, checkpoint={"epoch": 1})
        writer.record_validation(epoch=2, mse=0.5, checkpoint={"epoch": 2})
        prediction = np.array([[[1.0], [3.0]], [[2.0], [6.0]]], dtype=np.float32)
        target = np.zeros_like(prediction)
        writer.record_test(prediction, target, sample_indices=[7, 8], step_indices=[100, 101], variable_indices=[0])
        writer.write_environment({"python": "test", "cuda_available": False})
        writer.complete(self.provenance())
        return run_dir

    def test_standardized_metrics_are_element_weighted_and_predictions_are_indexed(self):
        with TemporaryDirectory() as directory:
            run_dir = self.write_complete_run(Path(directory))
            metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["space"], "standardized")
            self.assertEqual(metrics["aggregation"], "sample_element_weighted")
            self.assertEqual(metrics["test"]["sample_count"], 2)
            self.assertEqual(metrics["test"]["element_count"], 4)
            self.assertAlmostEqual(metrics["test"]["mse"], 12.5)
            self.assertAlmostEqual(metrics["test"]["mae"], 3.0)
            with np.load(run_dir / "predictions.npz") as values:
                self.assertEqual(set(values.files), {"prediction", "target", "sample_index", "step_index", "variable_index"})
                self.assertEqual(values["prediction"].shape, (2, 2, 1))
                np.testing.assert_array_equal(values["sample_index"], [7, 8])
                np.testing.assert_array_equal(values["step_index"], [100, 101])

    def test_best_checkpoint_uses_validation_mse_and_test_is_recorded_once_after_selection(self):
        with TemporaryDirectory() as directory:
            run_dir = self.write_complete_run(Path(directory))
            metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["selection"], {"metric": "validation_mse", "best_epoch": 1, "best_mse": 0.4})
            self.assertEqual(metrics["test_evaluations"], 1)
            self.assertTrue((run_dir / "best.pt").is_file())
            self.assertFalse((run_dir / "last.pt").exists())

    def test_checkpoint_payload_cannot_include_test_metrics(self):
        with TemporaryDirectory() as directory:
            writer = self.contract()(Path(directory) / "run", self.expected_spec())
            writer.write_run_config({"model": "PatchTST", "dataset": "ECL", "seed": 2021, "metrics_space": "standardized"})
            writer.append_history({"epoch": 1, "val_mse": 0.4})
            with self.assertRaisesRegex(ValueError, "test metrics"):
                writer.record_validation(epoch=1, mse=0.4, checkpoint={"test_mse": 0.0})

    def test_complete_run_has_required_artifacts_and_atomic_completion(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = self.write_complete_run(root)
            self.assertFalse((root / "run.partial").exists())
            self.assertTrue(run_dir.is_dir())
            for name in ("run_config.json", "metrics.json", "history.csv", "predictions.npz", "best.pt", "environment.json"):
                self.assertTrue((run_dir / name).is_file(), name)
            from bstalignment.general_results import validate_completed_run

            result = validate_completed_run(run_dir, self.expected_spec())
            self.assertTrue(result.valid, result.errors)

    def test_partial_or_mismatched_provenance_is_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = self.write_complete_run(root)
            from bstalignment.general_results import validate_completed_run

            self.assertFalse(validate_completed_run(root / "run.partial", self.expected_spec()).valid)
            for field, value in (("dataset_checksum", "b" * 64), ("source_commit", "7654321"), ("protocol", {"input_len": 12})):
                provenance_path = run_dir / "run_config.json"
                config = json.loads(provenance_path.read_text(encoding="utf-8"))
                config["provenance"][field] = value
                provenance_path.write_text(json.dumps(config), encoding="utf-8")
                result = validate_completed_run(run_dir, self.expected_spec())
                self.assertFalse(result.valid)
                self.assertIn(field, " ".join(result.errors))
                config["provenance"][field] = self.provenance()[field]
                provenance_path.write_text(json.dumps(config), encoding="utf-8")
            metrics_path = run_dir / "metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics["selection"] = None
            metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
            self.assertFalse(validate_completed_run(run_dir, self.expected_spec()).valid)

    def test_timellm_requires_persisted_runtime_and_prompt_audit(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            Writer = self.contract()
            writer = Writer(root / "timellm", self.expected_spec())
            writer.write_run_config({"model": "Time-LLM", "dataset": "ECL", "seed": 2021, "metrics_space": "standardized"})
            writer.append_history({"epoch": 1, "val_mse": 0.1})
            writer.record_validation(epoch=1, mse=0.1, checkpoint={"epoch": 1})
            values = np.zeros((1, 96, 1), dtype=np.float32)
            writer.record_test(values, values, sample_indices=[1], step_indices=range(96), variable_indices=[0])
            writer.write_environment({"python": "test"})
            missing_prompt_audit = self.provenance()
            missing_prompt_audit["runtime"]["time_llm"] = {"model_revision": "a", "tokenizer_revision": "b"}
            with self.assertRaisesRegex(ValueError, "Time-LLM.*prompt audit"):
                writer.complete(missing_prompt_audit)

            writer = Writer(root / "timellm-ready", self.expected_spec())
            writer.write_run_config({"model": "Time-LLM", "dataset": "ECL", "seed": 2021, "metrics_space": "standardized"})
            writer.append_history({"epoch": 1, "val_mse": 0.1})
            writer.record_validation(epoch=1, mse=0.1, checkpoint={"epoch": 1})
            writer.record_test(values, values, sample_indices=[1], step_indices=range(96), variable_indices=[0])
            writer.write_environment({"python": "test"})
            provenance = self.provenance()
            provenance["runtime"]["time_llm"] = {"model_revision": "a", "tokenizer_revision": "b"}
            provenance["prompt_audit"] = {"prompt_mode": "source_native", "token_count": 5}
            writer.complete(provenance)
            self.assertTrue((root / "timellm-ready" / "run_config.json").is_file())

    def test_general_trainers_expose_the_shared_result_contract_without_touching_battery_mode(self):
        from bstalignment.train_general_baselines import begin_general_result_run
        from bstalignment.train_graph_report import build_general_result_spec, prepare_result_output_dir

        with TemporaryDirectory() as directory:
            expected = self.expected_spec()
            writer = begin_general_result_run(Path(directory) / "baseline", expected)
            self.assertEqual(writer.expected_spec, expected)
        with patch("bstalignment.train_graph_report._git_source_commit", return_value="1234567"):
            spec = build_general_result_spec("ECL", "a" * 64)
        self.assertEqual(spec, self.expected_spec())
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(prepare_result_output_dir(root / "general", "general"), root / "general")
            self.assertFalse((root / "general").exists())
            self.assertTrue(prepare_result_output_dir(root / "battery", "battery").is_dir())


if __name__ == "__main__":
    unittest.main()
