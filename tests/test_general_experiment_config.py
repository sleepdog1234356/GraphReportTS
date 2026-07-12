from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "general_forecasting" / "experiment_matrix.yaml"
AUDITED_SOURCES = {
    "PatchTST": ("https://github.com/yuqinie98/PatchTST", "204c21e"),
    "iTransformer": ("https://github.com/thuml/iTransformer", "c2426e6"),
    "TimeCMA": ("https://github.com/ChenxiLiu-HNU/TimeCMA", "223e4ae"),
    "TimesNet": ("https://github.com/thuml/Time-Series-Library", "4e938a1"),
    "DLinear": ("https://github.com/cure-lab/LTSF-Linear", "0c11366"),
    "Time-LLM": ("https://github.com/KimMeen/Time-LLM", "b13e881"),
}


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
                    name: {"url": url, "commit": commit}
                    for name, (url, commit) in AUDITED_SOURCES.items()
                },
            }
            manifest = {
                "input_len": 36,
                "features": "M",
                "datasets": [dataset["name"] for dataset in datasets["datasets"]],
                "models": [model["name"] for model in models["models"]],
                "horizons": [24, 36, 48, 60],
                "smoke_seed": 42,
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
        self.assertEqual(getattr(spec, "features", None), "M")
        self.assertEqual(spec.horizons, (24, 36, 48, 60))
        self.assertEqual(getattr(spec, "smoke_seed", None), 42)
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
            {source.name: (source.url, source.commit) for source in spec.source_commits},
            AUDITED_SOURCES,
        )

    def test_loader_requires_the_complete_formal_matrix(self):
        mutations = {
            "datasets": lambda manifest, _: manifest.update(datasets=manifest["datasets"][:-1]),
            "models": lambda manifest, _: manifest.update(models=manifest["models"][:-1]),
            "horizons": lambda manifest, _: manifest.update(horizons=manifest["horizons"][:-1]),
        }
        for field, mutation in mutations.items():
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, f"complete.*{field}"):
                self.write_mutated_manifest(mutation)

    def test_loader_rejects_changed_audited_source_identity(self):
        def change_url(manifest, models):
            models["sources"]["PatchTST"]["url"] = "https://github.com/example/PatchTST"

        def change_commit(manifest, models):
            models["sources"]["PatchTST"]["commit"] = "abcdef0"

        for field, mutation in (("URL", change_url), ("commit", change_commit)):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, f"audited source {field}"):
                self.write_mutated_manifest(mutation)

    def test_loader_requires_smoke_seed_42(self):
        with self.assertRaisesRegex(ValueError, "smoke_seed"):
            self.write_mutated_manifest(lambda manifest, _: manifest.update(smoke_seed=41))

    def test_loader_requires_multivariate_to_multivariate_features(self):
        with self.assertRaisesRegex(ValueError, "features"):
            self.write_mutated_manifest(lambda manifest, _: manifest.update(features="S"))

    def test_loader_rejects_non_integer_numeric_fields(self):
        mutations = {
            "input_len": lambda manifest, _: manifest.update(input_len=36.0),
            "horizons": lambda manifest, _: manifest.update(horizons=[24.0, 36, 48, 60]),
            "smoke_seed": lambda manifest, _: manifest.update(smoke_seed=True),
            "formal_seeds": lambda manifest, _: manifest.update(formal_seeds=[2021.0, 2022, 2023]),
        }
        for field, mutation in mutations.items():
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "integer"):
                self.write_mutated_manifest(mutation)

    def test_loader_rejects_an_unknown_dataset(self):
        with self.assertRaisesRegex(ValueError, "unknown dataset"):
            self.write_mutated_manifest(lambda manifest, _: manifest.update(datasets=["ETTm1", "Traffic"]))

    def test_loader_rejects_a_non_36_step_input(self):
        with self.assertRaisesRegex(ValueError, "input_len"):
            self.write_mutated_manifest(lambda manifest, _: manifest.update(input_len=35))

    def test_loader_rejects_an_unsupported_horizon(self):
        with self.assertRaisesRegex(ValueError, "horizons"):
            self.write_mutated_manifest(lambda manifest, _: manifest.update(horizons=[24, 48]))

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
