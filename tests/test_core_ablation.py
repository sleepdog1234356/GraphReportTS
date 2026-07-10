from __future__ import annotations

from argparse import Namespace
from contextlib import redirect_stdout
from dataclasses import asdict
from hashlib import sha256
import io
import json
from multiprocessing.reduction import ForkingPickler
from pathlib import Path
import pickle
import re
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

import bstalignment.data_battery_raw as battery_data
import bstalignment.precompute_battery_sequence_cache as sequence_cache_precompute
import bstalignment.train_graph_report as graph_report_trainer
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
from bstalignment.run_core_ablation_suite import (
    CORE_ABLATION_SUITE_VERSION,
    CORE_BATTERY_ABLATIONS,
    core_run_config_matches,
    main as run_core_ablation_main,
    require_reusable_full_reference,
    verify_prompt_cache_identity,
)
from bstalignment.training_strategy import MAIN_TRAINING_PROFILE, build_graph_report_optimizer
from bstalignment.training_strategy import TRAINING_STRATEGY_VERSION


class CoreAblationRunnerTests(unittest.TestCase):
    FULL_ARGUMENTS = {
        "variant": "battery",
        "dataset": "mit",
        "history_len": 32,
        "pred_len": 20,
        "batch_size": 64,
        "seed": 42,
        "no_ic_dv": False,
        "no_hankel_map": False,
        "no_derivative_map": False,
        "no_report_prompt": False,
        "no_cross_modal": False,
        "no_text_gate": False,
        "no_semantic_alignment": False,
        "no_align_loss": False,
    }
    FULL_MODEL_CONFIG = {
        "variant": "battery",
        "freeze_text": True,
        "use_hf_text_encoder": True,
        "use_report_prompt": True,
        "use_cross_modal_fusion": True,
        "use_dynamic_graph": True,
        "use_domain_edges": True,
        "unified_decoder": True,
        "battery_history_len": 32,
        "history_feature_dim": 8,
        "use_multi_cycle_raw": True,
        "single_cycle_raw": False,
        "use_numeric_history": True,
        "use_text_gate": True,
        "use_semantic_alignment": True,
        "use_relative_steps": True,
    }

    def _write_full_result(self, result: Path, **config_updates) -> None:
        result.mkdir(parents=True, exist_ok=True)
        (result / "best.pt").write_bytes(b"checkpoint")
        (result / "test_metrics.json").write_text(
            '{"mse": 0.1, "mae": 0.2, "rmse": 0.316}', encoding="utf-8"
        )
        config = {
            "training_strategy_version": TRAINING_STRATEGY_VERSION,
            "args": dict(self.FULL_ARGUMENTS),
            "model_cfg": dict(self.FULL_MODEL_CONFIG),
        }
        for dotted_name, value in config_updates.items():
            container, name = dotted_name.split("__", 1)
            if container == "root":
                config[name] = value
            else:
                config[container][name] = value
        (result / "run_config.json").write_text(json.dumps(config), encoding="utf-8")

    @staticmethod
    def _write_core_checkpoint(
        path: Path,
        *,
        strategy: str = TRAINING_STRATEGY_VERSION,
        suite: str = "core-v1",
        profile: dict | None = None,
    ) -> None:
        torch.save(
            {
                "training_strategy_version": strategy,
                "ablation_suite_version": suite,
                "training_profile": (
                    dict(MAIN_TRAINING_PROFILE.__dict__) if profile is None else profile
                ),
            },
            path,
        )

    @staticmethod
    def _core_config(dataset: str, ablation: str) -> dict:
        args = dict(CoreAblationRunnerTests.FULL_ARGUMENTS)
        args.update({
            "no_dynamic_graph": False,
            "no_domain_edges": False,
            "separate_heads": False,
            "no_numeric_history": False,
            "no_multi_cycle_raw": False,
            "single_cycle_raw": False,
            "absolute_step_decoder": False,
        })
        args["dataset"] = dataset
        args["protocol_stage"] = "ablation"
        args["ablation_suite_version"] = "core-v1"
        args["battery_input_mode"] = "raw_sequence" if ablation == "no_hankel_graph" else "hankel_graph"
        if ablation != "no_hankel_graph":
            args[ablation] = True
        model_cfg = dict(CoreAblationRunnerTests.FULL_MODEL_CONFIG)
        model_cfg["battery_input_mode"] = args["battery_input_mode"]
        if ablation == "no_report_prompt":
            model_cfg["use_report_prompt"] = False
        elif ablation == "no_text_gate":
            model_cfg["use_text_gate"] = False
        return {
            "training_strategy_version": TRAINING_STRATEGY_VERSION,
            "protocol_stage": "ablation",
            "ablation_suite_version": "core-v1",
            "training_profile": dict(MAIN_TRAINING_PROFILE.__dict__),
            "args": args,
            "model_cfg": model_cfg,
        }

    def test_formal_matrix_contains_only_four_single_factor_variants(self):
        self.assertEqual(CORE_ABLATION_SUITE_VERSION, "core-v1")
        self.assertEqual(
            list(CORE_BATTERY_ABLATIONS),
            ["no_hankel_graph", "no_report_prompt", "no_ic_dv", "no_text_gate"],
        )
        self.assertTrue(all(isinstance(tokens, tuple) for tokens in CORE_BATTERY_ABLATIONS.values()))

    def test_full_reference_requires_complete_matching_artifacts(self):
        with TemporaryDirectory() as tmp:
            result = Path(tmp)
            self._write_full_result(result)
            row = require_reusable_full_reference(result, "mit", TRAINING_STRATEGY_VERSION)
            self.assertEqual(row["result_source"], "reused_main")
            self.assertEqual(row["mse"], 0.1)

    def test_present_full_timing_manifest_must_be_complete_typed_and_identity_matched(self):
        valid = {
            "best_epoch": 7,
            "stopped_epoch": 25,
            "mean_epoch_seconds": 1.5,
            "total_train_seconds": 37.5,
            "trainable_parameter_count": 100,
            "training_strategy_version": TRAINING_STRATEGY_VERSION,
            "ablation_suite_version": None,
        }
        cases = {
            "missing timing": lambda row: row.pop("best_epoch"),
            "wrong timing type": lambda row: row.update(best_epoch=7.5),
            "wrong strategy": lambda row: row.update(training_strategy_version="legacy"),
            "missing strategy": lambda row: row.pop("training_strategy_version"),
            "wrong suite identity": lambda row: row.update(ablation_suite_version="core-v1"),
            "missing suite identity": lambda row: row.pop("ablation_suite_version"),
        }
        with TemporaryDirectory() as tmp:
            result = Path(tmp) / "valid"
            self._write_full_result(result)
            (result / "run_summary.json").write_text(json.dumps(valid), encoding="utf-8")
            row = require_reusable_full_reference(result, "mit", TRAINING_STRATEGY_VERSION)
            self.assertEqual(row["best_epoch"], 7)
            self.assertEqual(row["total_train_seconds"], 37.5)
        for name, mutate in cases.items():
            with self.subTest(name=name), TemporaryDirectory() as tmp:
                result = Path(tmp) / "invalid"
                self._write_full_result(result)
                manifest = dict(valid)
                mutate(manifest)
                (result / "run_summary.json").write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, r"dataset=mit.*expected=.*observed=.*run_summary.json"):
                    require_reusable_full_reference(result, "mit", TRAINING_STRATEGY_VERSION)
                self.assertTrue(result.is_dir())

    def test_full_reference_rejects_mismatches_without_modifying_reference(self):
        cases = {
            "dataset": ("args__dataset", "calce"),
            "seed": ("args__seed", 7),
            "batch": ("args__batch_size", 32),
            "model flag": ("model_cfg__use_text_gate", False),
            "strategy": ("root__training_strategy_version", "legacy"),
            "explicit sequence mode": ("args__battery_input_mode", "raw_sequence"),
        }
        for name, (field, value) in cases.items():
            with self.subTest(name=name), TemporaryDirectory() as tmp:
                result = Path(tmp) / "full"
                self._write_full_result(result, **{field: value})
                before = {path.relative_to(result): path.read_bytes() for path in result.iterdir()}
                with self.assertRaisesRegex(RuntimeError, r"dataset=mit.*expected=.*observed=.*path="):
                    require_reusable_full_reference(result, "mit", TRAINING_STRATEGY_VERSION)
                self.assertTrue(result.is_dir())
                self.assertEqual(
                    before,
                    {path.relative_to(result): path.read_bytes() for path in result.iterdir()},
                )

    def test_full_reference_rejects_missing_and_malformed_artifacts_non_destructively(self):
        cases = ("best.pt", "test_metrics.json", "run_config.json")
        for missing in cases:
            with self.subTest(missing=missing), TemporaryDirectory() as tmp:
                result = Path(tmp) / "full"
                self._write_full_result(result)
                (result / missing).unlink()
                with self.assertRaisesRegex(RuntimeError, r"dataset=mit.*path="):
                    require_reusable_full_reference(result, "mit", TRAINING_STRATEGY_VERSION)
                self.assertTrue(result.is_dir())
        with TemporaryDirectory() as tmp:
            result = Path(tmp) / "full"
            self._write_full_result(result)
            (result / "run_config.json").write_text("{broken", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, r"dataset=mit.*expected=.*observed=.*path="):
                require_reusable_full_reference(result, "mit", TRAINING_STRATEGY_VERSION)
            self.assertTrue(result.is_dir())

    def test_prompt_cache_identity_checks_sample_count_order_identity_and_prompt(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = root / "reference"
            candidate = root / "candidate"
            reference.mkdir()
            candidate.mkdir()
            rows = [
                {"cell_id": "a", "cycle": 32, "prompt": "same"},
                {"cell_id": "b", "cycle": 33, "prompt": "same again"},
            ]
            text = "\n".join(json.dumps(row) for row in rows) + "\n"
            (reference / "meta.jsonl").write_text(text, encoding="utf-8")
            (candidate / "meta.jsonl").write_text(text, encoding="utf-8")
            verify_prompt_cache_identity(reference, candidate)
            (candidate / "meta.jsonl").write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Prompt sample count mismatch"):
                verify_prompt_cache_identity(reference, candidate)
            changed = [dict(rows[0]), dict(rows[1], prompt="different")]
            (candidate / "meta.jsonl").write_text(
                "\n".join(json.dumps(row) for row in changed), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "Prompt mismatch dataset sample=1"):
                verify_prompt_cache_identity(reference, candidate)

    def test_core_config_matching_is_variant_exact_and_rejects_legacy_suite(self):
        with TemporaryDirectory() as tmp:
            result = Path(tmp)
            config = self._core_config("mit", "no_text_gate")
            (result / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
            self.assertTrue(core_run_config_matches(result, "mit", "no_text_gate", TRAINING_STRATEGY_VERSION))
            config["args"]["no_ic_dv"] = True
            (result / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
            self.assertFalse(core_run_config_matches(result, "mit", "no_text_gate", TRAINING_STRATEGY_VERSION))
            config = self._core_config("mit", "no_text_gate")
            config["args"]["no_dynamic_graph"] = True
            (result / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
            self.assertFalse(core_run_config_matches(result, "mit", "no_text_gate", TRAINING_STRATEGY_VERSION))
            config = self._core_config("mit", "no_text_gate")
            config.pop("ablation_suite_version")
            (result / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
            self.assertFalse(core_run_config_matches(result, "mit", "no_text_gate", TRAINING_STRATEGY_VERSION))
            config = self._core_config("mit", "no_text_gate")
            config.pop("training_profile")
            (result / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
            self.assertFalse(core_run_config_matches(result, "mit", "no_text_gate", TRAINING_STRATEGY_VERSION))
            config = self._core_config("mit", "no_text_gate")
            config["training_profile"]["max_epochs"] = 79
            (result / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
            self.assertFalse(core_run_config_matches(result, "mit", "no_text_gate", TRAINING_STRATEGY_VERSION))
            config = self._core_config("mit", "no_text_gate")
            config["training_profile"]["gradient_clip"] = True
            (result / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
            self.assertFalse(core_run_config_matches(result, "mit", "no_text_gate", TRAINING_STRATEGY_VERSION))

    def test_default_dry_run_emits_twelve_train_commands_without_artifacts(self):
        with TemporaryDirectory() as tmp, patch(
            "bstalignment.run_core_ablation_suite.subprocess.run"
        ) as run:
            run.return_value.stdout = "source-commit\n"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                run_core_ablation_main([
                    "--full_result_root", str(Path(tmp) / "missing-full"),
                    "--graph_cache_dir", str(Path(tmp) / "missing-graph"),
                    "--sequence_cache_dir", str(Path(tmp) / "missing-sequence"),
                    "--out_root", str(Path(tmp) / "out"),
                    "--text_model", str(Path(tmp) / "missing-model"),
                    "--full_reference_commit", "full-commit",
                    "--dry_run",
                ])
            train_lines = [
                line for line in stdout.getvalue().splitlines()
                if "bstalignment.train_graph_report" in line
            ]
            self.assertEqual(len(train_lines), 12)
            self.assertTrue(all("--run_dir" in line and "--out_dir" not in line for line in train_lines))

    def test_matching_incomplete_result_resumes_complete_result_skips_and_mismatch_is_preserved(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = root / "no_text_gate"
            result.mkdir()
            config = self._core_config("mit", "no_text_gate")
            (result / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
            (result / "last.pt").write_bytes(b"last")
            self.assertTrue(core_run_config_matches(result, "mit", "no_text_gate", TRAINING_STRATEGY_VERSION))
            config["args"]["seed"] = 99
            (result / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
            before = (result / "last.pt").read_bytes()
            self.assertFalse(core_run_config_matches(result, "mit", "no_text_gate", TRAINING_STRATEGY_VERSION))
            self.assertEqual((result / "last.pt").read_bytes(), before)

    def test_normal_execution_writes_full_plus_four_variant_rows(self):
        with TemporaryDirectory() as tmp, patch(
            "bstalignment.run_core_ablation_suite.verify_prompt_cache_identity"
        ), patch("bstalignment.run_core_ablation_suite.subprocess.run") as run:
            root = Path(tmp)
            full = root / "full" / "mit"
            self._write_full_result(full)
            (full / "run_summary.json").write_text(json.dumps({
                "best_epoch": 7, "stopped_epoch": 25, "mean_epoch_seconds": 1.5,
                "total_train_seconds": 37.5, "trainable_parameter_count": 100,
                "training_strategy_version": TRAINING_STRATEGY_VERSION,
                "ablation_suite_version": None,
            }), encoding="utf-8")

            def fake_run(command, **kwargs):
                completed = Namespace(stdout="source-commit\n", returncode=0)
                if "bstalignment.train_graph_report" in command:
                    run_dir = Path(command[command.index("--run_dir") + 1])
                    ablation = run_dir.name
                    run_dir.mkdir(parents=True, exist_ok=True)
                    (run_dir / "run_config.json").write_text(
                        json.dumps(self._core_config("mit", ablation)), encoding="utf-8"
                    )
                    self._write_core_checkpoint(run_dir / "best.pt")
                    (run_dir / "test_metrics.json").write_text(
                        '{"mse": 0.2, "mae": 0.3, "rmse": 0.447}', encoding="utf-8"
                    )
                    (run_dir / "run_summary.json").write_text(json.dumps({
                        "best_epoch": 8, "stopped_epoch": 30, "mean_epoch_seconds": 2.0,
                        "total_train_seconds": 60.0, "trainable_parameter_count": 90,
                    }), encoding="utf-8")
                return completed

            run.side_effect = fake_run
            run_core_ablation_main([
                "--datasets", "mit",
                "--full_result_root", str(root / "full"),
                "--graph_cache_dir", str(root / "graph"),
                "--sequence_cache_dir", str(root / "sequence"),
                "--out_root", str(root / "out"),
                "--text_model", str(root / "text-model"),
                "--full_reference_commit", "full-commit",
            ])
            summary_path = root / "out" / "battery" / "mit" / "core_ablation_summary.csv"
            rows = summary_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(rows), 6)
            self.assertIn("dataset", rows[0].split(","))
            self.assertIn("full", rows[1])
            self.assertTrue((root / "out" / "battery" / "core_ablation_summary.csv").is_file())
            train_commands = [
                call.args[0] for call in run.call_args_list
                if "bstalignment.train_graph_report" in call.args[0]
            ]
            self.assertEqual(len(train_commands), 4)
            self.assertTrue(all("--no_resume" in command for command in train_commands))

    def test_runner_resumes_matching_last_skips_complete_and_preserves_mismatch(self):
        with TemporaryDirectory() as tmp, patch(
            "bstalignment.run_core_ablation_suite.verify_prompt_cache_identity"
        ), patch("bstalignment.run_core_ablation_suite.subprocess.run") as run:
            root = Path(tmp)
            self._write_full_result(root / "full" / "mit")
            result = root / "out" / "battery" / "mit" / "no_text_gate"
            result.mkdir(parents=True)
            (result / "run_config.json").write_text(
                json.dumps(self._core_config("mit", "no_text_gate")), encoding="utf-8"
            )
            self._write_core_checkpoint(result / "last.pt")
            original_last = (result / "last.pt").read_bytes()

            def fake_run(command, **kwargs):
                if command[:3] == ["git", "rev-parse", "HEAD"]:
                    return Namespace(stdout="source-commit\n", returncode=0)
                if "bstalignment.train_graph_report" in command:
                    self._write_core_checkpoint(result / "best.pt")
                    (result / "test_metrics.json").write_text(
                        '{"mse": 0.2, "mae": 0.3, "rmse": 0.447}', encoding="utf-8"
                    )
                    (result / "run_summary.json").write_text(json.dumps({
                        "best_epoch": 8, "stopped_epoch": 30, "mean_epoch_seconds": 2.0,
                        "total_train_seconds": 60.0, "trainable_parameter_count": 90,
                    }), encoding="utf-8")
                return Namespace(stdout="", returncode=0)

            run.side_effect = fake_run
            argv = [
                "--datasets", "mit", "--ablations", "no_text_gate",
                "--full_result_root", str(root / "full"),
                "--graph_cache_dir", str(root / "graph"),
                "--sequence_cache_dir", str(root / "sequence"),
                "--out_root", str(root / "out"),
                "--text_model", str(root / "text-model"),
                "--full_reference_commit", "full-commit",
            ]
            run_core_ablation_main(argv)
            train_commands = [
                call.args[0] for call in run.call_args_list
                if "bstalignment.train_graph_report" in call.args[0]
            ]
            self.assertEqual(len(train_commands), 1)
            self.assertNotIn("--no_resume", train_commands[0])
            self.assertEqual((result / "last.pt").read_bytes(), original_last)
            precompute_commands = [
                call.args[0] for call in run.call_args_list
                if any("precompute_battery" in str(token) for token in call.args[0])
            ]
            self.assertEqual(len(precompute_commands), 1)
            self.assertIn("bstalignment.precompute_battery_graph_cache", precompute_commands[0])
            self.assertNotIn("--no_ic_dv", precompute_commands[0])

            run.reset_mock()
            run_core_ablation_main(argv)
            self.assertFalse(any(
                "bstalignment.train_graph_report" in call.args[0]
                for call in run.call_args_list
            ))

            config = self._core_config("mit", "no_text_gate")
            config["args"]["seed"] = 99
            (result / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
            before = {path.name: path.read_bytes() for path in result.iterdir()}
            with self.assertRaisesRegex(RuntimeError, "mismatched metadata"):
                run_core_ablation_main(argv)
            self.assertEqual(before, {path.name: path.read_bytes() for path in result.iterdir()})

    def test_existing_matching_output_has_explicit_complete_resumable_or_invalid_state(self):
        cases = {
            "complete": ({"best.pt", "test_metrics.json", "run_summary.json"}, "skip"),
            "missing best with last": ({"last.pt", "test_metrics.json", "run_summary.json"}, "resume"),
            "missing summary with best": ({"best.pt", "test_metrics.json"}, "resume"),
            "checkpoint only": ({"best.pt"}, "resume"),
            "no checkpoint": (set(), "invalid"),
            "results without checkpoint": ({"test_metrics.json", "run_summary.json"}, "invalid"),
        }
        summary = {
            "best_epoch": 8,
            "stopped_epoch": 30,
            "mean_epoch_seconds": 2.0,
            "total_train_seconds": 60.0,
            "trainable_parameter_count": 90,
            "training_strategy_version": TRAINING_STRATEGY_VERSION,
            "ablation_suite_version": "core-v1",
        }
        for name, (artifacts, expected_state) in cases.items():
            with self.subTest(name=name), TemporaryDirectory() as tmp, patch(
                "bstalignment.run_core_ablation_suite.verify_prompt_cache_identity"
            ), patch("bstalignment.run_core_ablation_suite.subprocess.run") as run:
                root = Path(tmp)
                self._write_full_result(root / "full" / "mit")
                result = root / "out" / "battery" / "mit" / "no_text_gate"
                result.mkdir(parents=True)
                (result / "run_config.json").write_text(
                    json.dumps(self._core_config("mit", "no_text_gate")), encoding="utf-8"
                )
                if "best.pt" in artifacts:
                    self._write_core_checkpoint(result / "best.pt")
                if "last.pt" in artifacts:
                    self._write_core_checkpoint(result / "last.pt")
                if "test_metrics.json" in artifacts:
                    (result / "test_metrics.json").write_text(
                        '{"mse": 0.2, "mae": 0.3, "rmse": 0.447}', encoding="utf-8"
                    )
                if "run_summary.json" in artifacts:
                    (result / "run_summary.json").write_text(json.dumps(summary), encoding="utf-8")

                def fake_run(command, **kwargs):
                    if command[:3] == ["git", "rev-parse", "HEAD"]:
                        return Namespace(stdout="source-commit\n", returncode=0)
                    if "bstalignment.train_graph_report" in command:
                        self._write_core_checkpoint(result / "best.pt")
                        (result / "test_metrics.json").write_text(
                            '{"mse": 0.2, "mae": 0.3, "rmse": 0.447}', encoding="utf-8"
                        )
                        (result / "run_summary.json").write_text(json.dumps(summary), encoding="utf-8")
                    return Namespace(stdout="", returncode=0)

                run.side_effect = fake_run
                argv = [
                    "--datasets", "mit", "--ablations", "no_text_gate",
                    "--full_result_root", str(root / "full"),
                    "--graph_cache_dir", str(root / "graph"),
                    "--sequence_cache_dir", str(root / "sequence"),
                    "--out_root", str(root / "out"),
                    "--text_model", str(root / "text-model"),
                    "--full_reference_commit", "full-commit",
                ]
                if expected_state == "invalid":
                    before = {path.name: path.read_bytes() for path in result.iterdir()}
                    with self.assertRaisesRegex(RuntimeError, "checkpoint"):
                        run_core_ablation_main(argv)
                    self.assertEqual(before, {path.name: path.read_bytes() for path in result.iterdir()})
                    continue
                run_core_ablation_main(argv)
                train_commands = [
                    call.args[0] for call in run.call_args_list
                    if "bstalignment.train_graph_report" in call.args[0]
                ]
                if expected_state == "skip":
                    self.assertEqual(train_commands, [])
                else:
                    self.assertEqual(len(train_commands), 1)
                    self.assertNotIn("--no_resume", train_commands[0])

    def test_bad_core_checkpoint_identity_is_invalid_before_skip_or_resume(self):
        cases = {
            "complete wrong strategy": ("complete", {"strategy": "legacy"}),
            "complete wrong suite": ("complete", {"suite": "legacy-suite"}),
            "resumable wrong profile": (
                "resumable",
                {"profile": dict(MAIN_TRAINING_PROFILE.__dict__, max_epochs=79)},
            ),
            "complete corrupt payload": ("complete", None),
        }
        summary = {
            "best_epoch": 8,
            "stopped_epoch": 30,
            "mean_epoch_seconds": 2.0,
            "total_train_seconds": 60.0,
            "trainable_parameter_count": 90,
        }
        for name, (state, checkpoint_updates) in cases.items():
            with self.subTest(name=name), TemporaryDirectory() as tmp, patch(
                "bstalignment.run_core_ablation_suite.verify_prompt_cache_identity"
            ), patch("bstalignment.run_core_ablation_suite.subprocess.run") as run:
                root = Path(tmp)
                self._write_full_result(root / "full" / "mit")
                result = root / "out" / "battery" / "mit" / "no_text_gate"
                result.mkdir(parents=True)
                (result / "run_config.json").write_text(
                    json.dumps(self._core_config("mit", "no_text_gate")), encoding="utf-8"
                )
                if checkpoint_updates is None:
                    (result / "best.pt").write_bytes(b"not a torch checkpoint")
                else:
                    self._write_core_checkpoint(result / "best.pt", **checkpoint_updates)
                if state == "complete":
                    (result / "test_metrics.json").write_text(
                        '{"mse": 0.2, "mae": 0.3, "rmse": 0.447}', encoding="utf-8"
                    )
                    (result / "run_summary.json").write_text(json.dumps(summary), encoding="utf-8")
                run.side_effect = lambda command, **kwargs: Namespace(
                    stdout="source-commit\n" if command[:3] == ["git", "rev-parse", "HEAD"] else "",
                    returncode=0,
                )
                argv = [
                    "--datasets", "mit", "--ablations", "no_text_gate",
                    "--full_result_root", str(root / "full"),
                    "--graph_cache_dir", str(root / "graph"),
                    "--sequence_cache_dir", str(root / "sequence"),
                    "--out_root", str(root / "out"),
                    "--text_model", str(root / "text-model"),
                    "--full_reference_commit", "full-commit",
                ]
                before = {path.name: path.read_bytes() for path in result.iterdir()}
                with self.assertRaisesRegex(RuntimeError, r"dataset=mit.*best.pt"):
                    run_core_ablation_main(argv)
                self.assertEqual(before, {path.name: path.read_bytes() for path in result.iterdir()})
                self.assertFalse(any(
                    "bstalignment.train_graph_report" in call.args[0]
                    for call in run.call_args_list
                ))


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

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_all_core_variants_complete_one_optimizer_step_on_cuda(self):
        device = torch.device("cuda")
        variants = {
            "no_hankel_graph": dict(battery_input_mode="raw_sequence"),
            "no_report_prompt": dict(use_report_prompt=False),
            "no_ic_dv": {},
            "no_text_gate": dict(use_text_gate=False),
        }
        for name, updates in variants.items():
            with self.subTest(name=name):
                cfg = GraphReportTSConfig(
                    variant="battery", d_model=8, output_dim=1, graph_layers=1,
                    patch_size=2, patch_stride=1, topk_edges=1,
                    use_hf_text_encoder=False, battery_history_len=2,
                    temporal_heads=2, raw_sequence_len=16, raw_sequence_dim=6,
                    **updates,
                )
                model = GraphReportTS(cfg).to(device)
                optimizer = build_graph_report_optimizer(model, MAIN_TRAINING_PROFILE)
                batch = {
                    "prompt": ["battery prompt", "battery prompt"],
                    "horizon": torch.tensor([20, 20], device=device),
                    "history_features": torch.randn(2, 2, 8, device=device),
                }
                if name == "no_hankel_graph":
                    batch["raw_sequences"] = torch.randn(2, 2, 16, 6, device=device)
                else:
                    map_channels = 12 if name == "no_ic_dv" else 18
                    batch["maps"] = torch.randn(2, 2, map_channels, 2, 3, device=device)
                output = graph_report_trainer._model_forward(model, batch)
                self.assertEqual(output["pred"].shape, (2, 20, 1))
                loss = output["pred"].square().mean()
                self.assertTrue(torch.isfinite(loss))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()


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


class TrainerIntegrationTests(unittest.TestCase):
    def test_resume_checkpoint_prefers_last_and_falls_back_to_best(self):
        with TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            self.assertIsNone(graph_report_trainer._resume_checkpoint_path(out_dir))
            best_path = out_dir / "best.pt"
            best_path.write_bytes(b"best")
            self.assertEqual(graph_report_trainer._resume_checkpoint_path(out_dir), best_path)
            last_path = out_dir / "last.pt"
            last_path.write_bytes(b"last")
            self.assertEqual(graph_report_trainer._resume_checkpoint_path(out_dir), last_path)

    def test_formal_resume_checkpoint_requires_shared_strategy_suite_and_profile_identity(self):
        validator = getattr(graph_report_trainer, "require_graph_report_checkpoint_identity", None)
        self.assertIsNotNone(
            validator,
            "formal trainer must use the shared checkpoint identity validator",
        )
        if validator is None:
            return
        valid = {
            "training_strategy_version": TRAINING_STRATEGY_VERSION,
            "ablation_suite_version": "core-v1",
            "training_profile": dict(MAIN_TRAINING_PROFILE.__dict__),
        }
        validator(
            valid,
            training_strategy_version=TRAINING_STRATEGY_VERSION,
            ablation_suite_version="core-v1",
            context="test checkpoint",
        )
        cases = {
            "missing strategy": dict(valid, training_strategy_version=None),
            "wrong suite": dict(valid, ablation_suite_version="legacy-suite"),
            "missing profile": {**valid, "training_profile": None},
            "wrong max epochs": {
                **valid,
                "training_profile": dict(MAIN_TRAINING_PROFILE.__dict__, max_epochs=79)
            },
            "incomplete profile": {**valid, "training_profile": {"max_epochs": 80}},
            "boolean numeric field": {
                **valid,
                "training_profile": dict(MAIN_TRAINING_PROFILE.__dict__, gradient_clip=True)
            },
        }
        for name, checkpoint in cases.items():
            with self.subTest(name=name), self.assertRaises(RuntimeError):
                validator(
                    checkpoint,
                    training_strategy_version=TRAINING_STRATEGY_VERSION,
                    ablation_suite_version="core-v1",
                    context="test checkpoint",
                )

    def test_raw_sequence_parser_flags_keep_formal_batch_default(self):
        with patch.object(
            sys,
            "argv",
            [
                "train_graph_report",
                "--variant", "battery",
                "--battery_input_mode", "raw_sequence",
                "--precomputed_sequence_cache_dir", "sequence-cache",
                "--require_precomputed_sequence_cache",
                "--protocol_stage", "ablation",
                "--ablation_suite_version", "core-v1",
                "--run_dir", "runs/core/battery/mit/no_hankel_graph",
            ],
        ):
            args = graph_report_trainer.parse_args()

        self.assertEqual(args.battery_input_mode, "raw_sequence")
        self.assertEqual(args.precomputed_sequence_cache_dir, "sequence-cache")
        self.assertTrue(args.require_precomputed_sequence_cache)
        self.assertEqual(args.protocol_stage, "ablation")
        self.assertEqual(args.ablation_suite_version, "core-v1")
        self.assertEqual(args.run_dir, "runs/core/battery/mit/no_hankel_graph")
        self.assertEqual(args.batch_size, 64)

    def test_raw_sequence_forwarding_uses_representation_aware_payload(self):
        raw_model = GraphReportTS(
            GraphReportTSConfig(
                variant="battery", d_model=8, output_dim=1, graph_layers=1,
                patch_size=2, patch_stride=1, topk_edges=1,
                use_hf_text_encoder=False, battery_history_len=32,
                temporal_heads=2, raw_sequence_len=16, raw_sequence_dim=6,
                battery_input_mode="raw_sequence",
            )
        )
        batch = {
            "raw_sequences": torch.randn(2, 32, 16, 6),
            "prompt": ["p", "p"],
            "horizon": torch.tensor([20, 20]),
            "history_features": torch.randn(2, 32, 8),
        }

        out = graph_report_trainer._model_forward(raw_model, batch)

        self.assertEqual(out["pred"].shape, (2, 20, 1))

    def test_battery_loader_passes_only_the_selected_cache_payload(self):
        cases = (
            (
                "hankel_graph",
                {"precomputed_cache_dir": "graph-cache", "require_precomputed_cache": True},
                ("input_representation", "precomputed_sequence_cache_dir", "require_precomputed_sequence_cache"),
            ),
            (
                "raw_sequence",
                {
                    "input_representation": "sequence",
                    "precomputed_sequence_cache_dir": "sequence-cache",
                    "require_precomputed_sequence_cache": True,
                },
                ("precomputed_cache_dir", "require_precomputed_cache"),
            ),
        )
        for input_mode, expected, forbidden in cases:
            with self.subTest(input_mode=input_mode), patch.object(
                sys,
                "argv",
                [
                    "train_graph_report",
                    "--battery_input_mode", input_mode,
                    "--precomputed_cache_dir", "graph-cache",
                    "--require_precomputed_cache",
                    "--precomputed_sequence_cache_dir", "sequence-cache",
                    "--require_precomputed_sequence_cache",
                ],
            ), patch.object(
                graph_report_trainer,
                "BatteryRawGraphDataset",
                side_effect=lambda **kwargs: [kwargs],
            ) as dataset, patch.object(
                graph_report_trainer,
                "DataLoader",
                side_effect=lambda data, **kwargs: data,
            ):
                graph_report_trainer.build_loaders(graph_report_trainer.parse_args())

            self.assertEqual(dataset.call_count, 3)
            for call in dataset.call_args_list:
                for key, value in expected.items():
                    self.assertEqual(call.kwargs[key], value)
                for key in forbidden:
                    self.assertNotIn(key, call.kwargs)

    def test_output_directory_defaults_to_legacy_layout_and_run_dir_is_direct(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy_args = Namespace(run_dir=None, out_dir=str(root / "legacy"), variant="battery", dataset="mit")
            direct_args = Namespace(
                run_dir=str(root / "core" / "battery" / "mit" / "no_hankel_graph"),
                out_dir=str(root / "ignored"),
                variant="battery",
                dataset="mit",
            )

            self.assertEqual(
                graph_report_trainer._resolve_out_dir(legacy_args),
                root / "legacy" / "battery" / "mit",
            )
            self.assertEqual(
                graph_report_trainer._resolve_out_dir(direct_args),
                root / "core" / "battery" / "mit" / "no_hankel_graph",
            )

    def test_best_checkpoint_fallback_truncates_history_before_resumed_append(self):
        with TemporaryDirectory() as tmp:
            history_path = Path(tmp) / "epoch_history.jsonl"
            original_lines = [
                json.dumps({"epoch": epoch, "marker": f"row-{epoch}", "epoch_seconds": float(epoch)}) + "\n"
                for epoch in range(1, 6)
            ]
            history_path.write_text("".join(original_lines), encoding="utf-8")

            seconds = graph_report_trainer._reconcile_epoch_history(
                history_path,
                checkpoint_epoch=3,
                checkpoint_epoch_seconds=[1.0, 2.0, 3.0],
            )

            self.assertEqual(seconds, [1.0, 2.0, 3.0])
            self.assertEqual(history_path.read_text(encoding="utf-8"), "".join(original_lines[:3]))
            with history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"epoch": 4, "epoch_seconds": 4.0}) + "\n")
            seconds.append(4.0)
            rows = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["epoch"] for row in rows], [1, 2, 3, 4])
            summary = graph_report_trainer._run_summary_payload(
                best_epoch=3,
                stopped_epoch=4,
                epoch_seconds=seconds,
                trainable_parameter_count=123,
                training_strategy_version="strategy-v3",
                ablation_suite_version="core-v1",
            )
            self.assertEqual(summary["total_train_seconds"], 10.0)
            self.assertEqual(summary["mean_epoch_seconds"], 2.5)

    def test_legacy_checkpoint_without_timing_uses_only_retained_available_history(self):
        with TemporaryDirectory() as tmp:
            history_path = Path(tmp) / "epoch_history.jsonl"
            history_path.write_text(
                '{"epoch": 1, "legacy": true}\n'
                '{"epoch": 2, "legacy": true}\n'
                '{"epoch": 3, "epoch_seconds": 99.0}\n',
                encoding="utf-8",
            )

            seconds = graph_report_trainer._reconcile_epoch_history(
                history_path,
                checkpoint_epoch=2,
                checkpoint_epoch_seconds=None,
            )

            self.assertEqual(seconds, [])
            self.assertEqual(
                [json.loads(line)["epoch"] for line in history_path.read_text(encoding="utf-8").splitlines()],
                [1, 2],
            )
            with history_path.open("a", encoding="utf-8") as handle:
                handle.write('{"epoch": 3, "epoch_seconds": 3.5}\n')
            self.assertEqual(
                graph_report_trainer._reconcile_epoch_history(
                    history_path,
                    checkpoint_epoch=3,
                    checkpoint_epoch_seconds=[3.5],
                ),
                [3.5],
            )

    def test_checkpoint_only_resume_allows_a_later_contiguous_history_suffix(self):
        with TemporaryDirectory() as tmp:
            history_path = Path(tmp) / "epoch_history.jsonl"
            self.assertEqual(
                graph_report_trainer._reconcile_epoch_history(
                    history_path,
                    checkpoint_epoch=2,
                    checkpoint_epoch_seconds=[1.0, 2.0],
                ),
                [1.0, 2.0],
            )
            history_path.write_text('{"epoch": 3, "epoch_seconds": 3.0}\n', encoding="utf-8")

            seconds = graph_report_trainer._reconcile_epoch_history(
                history_path,
                checkpoint_epoch=3,
                checkpoint_epoch_seconds=[1.0, 2.0, 3.0],
            )

            self.assertEqual(seconds, [1.0, 2.0, 3.0])
            self.assertEqual(json.loads(history_path.read_text(encoding="utf-8"))["epoch"], 3)

    def test_malformed_tail_after_checkpoint_epoch_is_truncated(self):
        with TemporaryDirectory() as tmp:
            history_path = Path(tmp) / "epoch_history.jsonl"
            history_path.write_text(
                '{"epoch": 1, "epoch_seconds": 1.0}\n'
                '{"epoch": 2, "epoch_seconds": 2.0}\n'
                '{"epoch": 3, "epoch_seconds": 3.0}\n'
                '{"epoch": 4,',
                encoding="utf-8",
            )

            seconds = graph_report_trainer._reconcile_epoch_history(
                history_path,
                checkpoint_epoch=3,
                checkpoint_epoch_seconds=[1.0, 2.0, 3.0],
            )

            self.assertEqual(seconds, [1.0, 2.0, 3.0])
            self.assertEqual(
                [json.loads(line)["epoch"] for line in history_path.read_text(encoding="utf-8").splitlines()],
                [1, 2, 3],
            )

    def test_reconcile_normalizes_one_trailing_newline_before_append(self):
        with TemporaryDirectory() as tmp:
            history_path = Path(tmp) / "epoch_history.jsonl"
            first_row = b'{"epoch": 1, "epoch_seconds": 1.0}'
            history_path.write_bytes(first_row)

            graph_report_trainer._reconcile_epoch_history(
                history_path,
                checkpoint_epoch=1,
                checkpoint_epoch_seconds=[1.0],
            )

            self.assertEqual(history_path.read_bytes(), first_row + b"\n")
            with history_path.open("a", encoding="utf-8", newline="") as handle:
                handle.write('{"epoch": 2, "epoch_seconds": 2.0}\n')
            self.assertEqual(
                [json.loads(line)["epoch"] for line in history_path.read_text(encoding="utf-8").splitlines()],
                [1, 2],
            )

    def test_resume_reconciliation_rejects_zero_and_negative_history_epochs(self):
        cases = (
            ('{"epoch": 0}\n', 0),
            ('{"epoch": -1}\n{"epoch": 0}\n', 0),
        )
        with TemporaryDirectory() as tmp:
            history_path = Path(tmp) / "epoch_history.jsonl"
            for contents, checkpoint_epoch in cases:
                with self.subTest(contents=contents):
                    history_path.write_text(contents, encoding="utf-8")
                    with self.assertRaisesRegex(RuntimeError, "epoch must be a positive integer"):
                        graph_report_trainer._reconcile_epoch_history(
                            history_path,
                            checkpoint_epoch=checkpoint_epoch,
                            checkpoint_epoch_seconds=None,
                        )

    def test_resume_reconciliation_rejects_history_and_checkpoint_timing_anomalies(self):
        with TemporaryDirectory() as tmp:
            history_path = Path(tmp) / "epoch_history.jsonl"
            history_path.write_text(
                '{"epoch": 1, "epoch_seconds": 1.0}\n'
                '{"epoch": 2, "epoch_seconds": 2.0}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "checkpoint epoch_seconds.*history"):
                graph_report_trainer._reconcile_epoch_history(
                    history_path,
                    checkpoint_epoch=2,
                    checkpoint_epoch_seconds=[1.0],
                )

            history_path.write_text(
                '{"epoch": 1, "epoch_seconds": 1.0}\n'
                '{"epoch": 1, "epoch_seconds": 1.0}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "expected epoch 2"):
                graph_report_trainer._reconcile_epoch_history(
                    history_path,
                    checkpoint_epoch=2,
                    checkpoint_epoch_seconds=[1.0, 1.0],
                )

            history_path.write_text(
                '{"epoch": 1, "epoch_seconds": 1.0}\n'
                '{"epoch": 2, "legacy": true}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "timing gap"):
                graph_report_trainer._reconcile_epoch_history(
                    history_path,
                    checkpoint_epoch=2,
                    checkpoint_epoch_seconds=[1.0],
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
