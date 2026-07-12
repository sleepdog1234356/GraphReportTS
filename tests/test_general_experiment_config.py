from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "general_forecasting" / "experiment_matrix.yaml"


class GeneralExperimentConfigTests(unittest.TestCase):
    @staticmethod
    def load_spec(path=CONFIG_PATH):
        try:
            from bstalignment.general_experiment_config import load_general_experiment_spec
        except ModuleNotFoundError as exc:
            raise AssertionError("general experiment configuration loader does not exist") from exc
        return load_general_experiment_spec(path)

    def write_mutated_manifest(self, mutate):
        with TemporaryDirectory() as temporary_directory:
            config_dir = Path(temporary_directory) / "general_forecasting"
            config_dir.mkdir()
            manifest_path = config_dir / "experiment_matrix.yaml"
            datasets = {
                "datasets": [
                    {"name": name, "raw_path": f"raw/{name}.csv", "raw_sha256": "0" * 64}
                    for name in ("ETTm1", "ETTm2", "ETTh1", "ETTh2", "ECL", "Weather")
                ]
            }
            models = {
                "models": [
                    {"name": name}
                    for name in ("GraphReportTS", "PatchTST", "iTransformer", "TimeCMA", "TimesNet", "DLinear", "Time-LLM")
                ],
                "sources": {
                    name: {"url": f"https://github.com/example/{name}", "commit": "abc1234"}
                    for name in ("PatchTST", "iTransformer", "TimeCMA", "TimesNet", "DLinear", "Time-LLM")
                },
            }
            manifest = {
                "input_len": 36,
                "datasets": [dataset["name"] for dataset in datasets["datasets"]],
                "models": [model["name"] for model in models["models"]],
                "horizons": [96, 192, 336, 720],
                "formal_seeds": [2021, 2022, 2023],
                "paths": {"datasets": "datasets.yaml", "models": "models.yaml", "output_root": "runs/general_forecasting"},
            }
            mutate(manifest, models)
            (config_dir / "datasets.yaml").write_text(json.dumps(datasets, indent=2), encoding="utf-8")
            (config_dir / "models.yaml").write_text(json.dumps(models, indent=2), encoding="utf-8")
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            return self.load_spec(manifest_path)

    def test_manifest_freezes_the_formal_general_forecasting_matrix(self):
        spec = self.load_spec()

        self.assertEqual(
            {dataset.name for dataset in spec.datasets},
            {"ETTm1", "ETTm2", "ETTh1", "ETTh2", "ECL", "Weather"},
        )
        self.assertEqual(
            {model.name for model in spec.models},
            {"GraphReportTS", "PatchTST", "iTransformer", "TimeCMA", "TimesNet", "DLinear", "Time-LLM"},
        )
        self.assertEqual(spec.input_len, 36)
        self.assertEqual(spec.horizons, (96, 192, 336, 720))
        self.assertEqual(spec.formal_seeds, (2021, 2022, 2023))
        self.assertEqual(len(spec.run_ids), 504)
        self.assertEqual(len(spec.run_ids), len(set(spec.run_ids)))
        self.assertEqual(
            {dataset.name: dataset.raw_sha256 for dataset in spec.datasets},
            {
                "ETTh1": "f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066",
                "ETTh2": "a3dc2c597b9218c7ce1cd55eb77b283fd459a1d09d753063f944967dd6b9218b",
                "ETTm1": "6ce1759b1a18e3328421d5d75fadcb316c449fcd7cec32820c8dafda71986c9e",
                "ETTm2": "db973ca252c6410a30d0469b13d696cf919648d0f3fd588c60f03fdbdbadd1fd",
                "ECL": "7e45845d54c5219bad0ae6bc1b5316cf8ff9cead5d33fa998a5a51c2e4a497ad",
                "Weather": "34ee981d07313e51da2a50bb600072c8ae4a69cb4b0651f4cb93a069d7a2ba63",
            },
        )

    def test_loaded_records_are_immutable(self):
        spec = self.load_spec()

        with self.assertRaises(FrozenInstanceError):
            spec.datasets[0].name = "Traffic"
        self.assertIsNotNone(getattr(spec, "horizon_spec", None))
        self.assertIsNotNone(getattr(spec, "seed_spec", None))
        with self.assertRaises(FrozenInstanceError):
            spec.horizon_spec.values = (48,)
        with self.assertRaises(FrozenInstanceError):
            spec.seed_spec.values = (42,)

    def test_manifest_freezes_audited_official_source_commits(self):
        spec = self.load_spec()

        self.assertEqual(
            {source.name: source.commit for source in spec.source_commits},
            {
                "PatchTST": "204c21e",
                "iTransformer": "c2426e6",
                "TimeCMA": "223e4ae",
                "TimesNet": "4e938a1",
                "DLinear": "0c11366",
                "Time-LLM": "b13e881",
            },
        )
        self.assertTrue(all(source.url.startswith("https://github.com/") for source in spec.source_commits))
        self.assertTrue(all(source.commit for source in spec.source_commits))

    def test_loader_rejects_an_unknown_dataset(self):
        with self.assertRaisesRegex(ValueError, "unknown dataset"):
            self.write_mutated_manifest(lambda manifest, _: manifest.update(datasets=["ETTm1", "Traffic"]))

    def test_loader_rejects_a_non_36_step_input(self):
        with self.assertRaisesRegex(ValueError, "input_len"):
            self.write_mutated_manifest(lambda manifest, _: manifest.update(input_len=35))

    def test_loader_rejects_an_unsupported_horizon(self):
        with self.assertRaisesRegex(ValueError, "horizons"):
            self.write_mutated_manifest(lambda manifest, _: manifest.update(horizons=[96, 48]))

    def test_loader_rejects_duplicate_run_ids(self):
        with self.assertRaisesRegex(ValueError, "duplicate run ID"):
            self.write_mutated_manifest(lambda manifest, _: manifest.update(datasets=manifest["datasets"] + ["ETTm1"]))

    def test_loader_rejects_a_missing_source_commit(self):
        def remove_commit(manifest, models):
            models["sources"]["PatchTST"].pop("commit")

        with self.assertRaisesRegex(ValueError, "source commit"):
            self.write_mutated_manifest(remove_commit)


if __name__ == "__main__":
    unittest.main()
