from argparse import Namespace
import gc
import json
from pathlib import Path
import pickle
import shlex
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import torch

import bstalignment.battery_protocol as battery_protocol
from bstalignment.data_battery_raw import (
    BATTERY_GRAPH_CACHE_VERSION,
    BatteryRawGraphDataset,
    battery_graph_cache_config,
)
from bstalignment.graph_report_model import GraphReportTS, GraphReportTSConfig
import bstalignment.precompute_battery_graph_cache as cache_precompute
from bstalignment.precompute_battery_graph_cache import precompute_split
import bstalignment.training_strategy as training_strategy
from bstalignment.run_ablation_suite import (
    has_matching_strategy_version,
    remove_ablation_output_if_forced,
    remove_ablation_output_if_fresh,
    should_skip_ablation,
)
from bstalignment.train_battery_baselines import BatterySequenceDataset
from bstalignment.train_battery_official_baselines import resolve_baseline_profile
from bstalignment.training_strategy import (
    BASELINE_TRAINING_PROFILES,
    MAIN_TRAINING_PROFILE,
    TRAINING_STRATEGY_VERSION,
    GraphReportScheduler,
    baseline_regression_loss,
    build_baseline_optimizer,
    build_baseline_scheduler,
    build_graph_report_optimizer,
    get_baseline_training_profile,
    graph_report_align_weight,
    graph_report_group_lrs,
    should_stop_graph_report,
    step_baseline_batch_scheduler,
    step_baseline_epoch_scheduler,
    update_graph_report_stale,
)


class BatteryDataFixtureMixin:
    @staticmethod
    def write_mit_data(root: Path, multipliers):
        mit_dir = root / "mit"
        mit_dir.mkdir(parents=True)
        batch = {}
        for index, multiplier in enumerate(multipliers):
            n = 60
            cycle = np.arange(1, n + 1, dtype=np.float32) * float(multiplier)
            batch[f"cell{index}"] = {
                "summary": {
                    "cycle": cycle,
                    "QD": np.linspace(1.1, 0.8, n, dtype=np.float32),
                    "QC": np.linspace(1.15, 0.85, n, dtype=np.float32),
                    "IR": np.linspace(0.01, 0.03, n, dtype=np.float32),
                    "Tmax": np.full(n, 35.0, dtype=np.float32),
                    "Tavg": np.full(n, 30.0, dtype=np.float32),
                    "Tmin": np.full(n, 25.0, dtype=np.float32),
                    "chargetime": np.linspace(10.0, 15.0, n, dtype=np.float32),
                },
                "cycle_life": float(cycle[-1]),
                "charge_policy": "fixture",
            }
        with (mit_dir / "batch1.pkl").open("wb") as handle:
            pickle.dump(batch, handle)

    @staticmethod
    def write_processed_data(root: Path, multipliers):
        processed_dir = root / "processed" / "battery" / "calce"
        processed_dir.mkdir(parents=True)
        for index, multiplier in enumerate(multipliers):
            n = 60
            cycle = np.arange(1, n + 1, dtype=np.int64) * int(multiplier)
            current = np.tile(np.array([0.0, 1.0, 0.0], dtype=np.float32), (n, 1))
            voltage = np.tile(np.array([3.0, 3.5, 4.0], dtype=np.float32), (n, 1))
            temperature = np.tile(np.array([25.0, 30.0, 28.0], dtype=np.float32), (n, 1))
            capacity = np.tile(np.array([0.0, 0.5, 1.0], dtype=np.float32), (n, 1))
            np.savez(
                processed_dir / f"cell{index}.npz",
                cycle_id=cycle,
                soh=np.linspace(1.0, 0.75, n, dtype=np.float32),
                current=current,
                voltage=voltage,
                temperature=temperature,
                capacity=capacity,
                capacity_summary=capacity[:, -1],
                internal_resistance=np.linspace(0.01, 0.03, n, dtype=np.float32),
                charge_time=np.linspace(10.0, 15.0, n, dtype=np.float32),
            )


class BatteryWindowProtocolTests(BatteryDataFixtureMixin, unittest.TestCase):
    def test_mit_graph_and_sequence_windows_have_only_full_twenty_step_targets(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_mit_data(root, [1, 1, 1, 1])

            graph = BatteryRawGraphDataset(
                dataset_name="mit",
                data_root=root,
                split="train",
                history_len=32,
                max_horizon=20,
            )
            sequence = BatterySequenceDataset(
                dataset_name="mit",
                data_root=root,
                split="train",
                input_len=32,
                pred_len=20,
            )

            self.assertGreater(len(graph), 0)
            self.assertEqual({sample["horizon"] for sample in graph.samples}, {20})
            self.assertTrue(all(sequence[index]["y"].shape == (20,) for index in range(len(sequence))))
            prompt = graph._prompt_from_history(
                np.ones((3, 1), dtype=np.float32), ["capacity"], graph.samples[-1]["horizon"], "cell", 40, []
            )
            self.assertIn("predict next 20 steps", prompt)

    def test_processed_graph_and_sequence_windows_have_only_full_twenty_step_targets(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_processed_data(root, [1, 1, 1, 1, 1, 1])

            graph = BatteryRawGraphDataset(
                dataset_name="calce",
                data_root=root,
                split="train",
                history_len=32,
                max_horizon=20,
            )
            sequence = BatterySequenceDataset(
                dataset_name="calce",
                data_root=root,
                split="train",
                input_len=32,
                pred_len=20,
            )

            self.assertGreater(len(graph), 0)
            self.assertEqual({sample["horizon"] for sample in graph.samples}, {20})
            self.assertTrue(all(sequence[index]["y"].shape == (20,) for index in range(len(sequence))))

    def test_processed_cycle_scale_is_train_only_shared_and_not_clipped(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_processed_data(root, [10, 8, 2, 1, 4, 3])
            graph_sets = {
                split: BatteryRawGraphDataset(
                    dataset_name="calce",
                    data_root=root,
                    split=split,
                    history_len=32,
                    max_horizon=20,
                    max_cycles=55,
                )
                for split in ("train", "val", "test")
            }
            sequence_sets = {
                split: BatterySequenceDataset(
                    dataset_name="calce",
                    data_root=root,
                    split=split,
                    input_len=32,
                    pred_len=20,
                    max_cycles=55,
                )
                for split in ("train", "val", "test")
            }

            expected_scale = 4 * 55
            for dataset in (*graph_sets.values(), *sequence_sets.values()):
                self.assertEqual(dataset.cycle_scale, expected_scale)
            val_sequence = sequence_sets["val"]
            val_values = next(iter(val_sequence.series.values()))
            self.assertGreater(float(val_values[-1, 4]), 1.0)
            val_graph = graph_sets["val"]
            val_cell = val_graph.processed_cells[0]
            val_history = val_graph._processed_history_features(val_cell, 0, 32, 55)
            self.assertGreater(float(val_history[-1, 4]), 1.0)

    def test_mit_cycle_scale_is_train_only_shared_and_respects_max_cycles(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_mit_data(root, [1, 10, 2, 3, 4, 5, 6, 7, 9, 8])
            graph_sets = {
                split: BatteryRawGraphDataset(
                    dataset_name="mit",
                    data_root=root,
                    split=split,
                    history_len=32,
                    max_horizon=20,
                    max_cycles=55,
                )
                for split in ("train", "val", "test")
            }
            sequence_sets = {
                split: BatterySequenceDataset(
                    dataset_name="mit",
                    data_root=root,
                    split=split,
                    input_len=32,
                    pred_len=20,
                    max_cycles=55,
                )
                for split in ("train", "val", "test")
            }

            expected_scale = 7 * 55
            for dataset in (*graph_sets.values(), *sequence_sets.values()):
                self.assertEqual(dataset.cycle_scale, expected_scale)
            test_values = np.concatenate(list(sequence_sets["test"].series.values()), axis=0)
            self.assertGreater(float(test_values[:, 4].max()), 1.0)

    def test_graph_cache_config_identifies_fixed_horizon_train_scale_protocol(self):
        config = battery_graph_cache_config(
            dataset_name="mit",
            split="train",
            max_horizon=20,
            resample_len=128,
            delay_dim=8,
            delay_lag=1,
            include_derivatives=True,
            include_hankel=True,
            include_ic_dv=True,
            allow_summary_fallback=False,
            seed=42,
            max_cycles=None,
            history_len=32,
        )

        self.assertGreater(BATTERY_GRAPH_CACHE_VERSION, 4)
        self.assertEqual(config["target_protocol"], "32-observed-20-future-only-full-horizon")
        self.assertEqual(config["cycle_scale_protocol"], "train-split-max-cycle-id-no-clip")

    def test_precomputed_graph_cache_matches_uncached_full_horizon_samples(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_mit_data(root, [1, 1, 1, 1])
            common = {
                "dataset_name": "mit",
                "data_root": root,
                "split": "train",
                "history_len": 32,
                "max_horizon": 20,
                "resample_len": 16,
                "delay_dim": 2,
                "include_derivatives": False,
                "include_hankel": False,
                "include_ic_dv": False,
                "allow_summary_fallback": True,
            }
            uncached = BatteryRawGraphDataset(**common)
            precompute_split(
                Namespace(
                    dataset="mit",
                    data_root=str(root),
                    cache_dir=str(root / "cache"),
                    pred_len=20,
                    history_len=32,
                    resample_len=16,
                    delay_dim=2,
                    delay_lag=1,
                    no_derivative_map=True,
                    no_hankel_map=True,
                    no_ic_dv=True,
                    allow_summary_fallback=True,
                    max_cycles=None,
                    seed=42,
                    batch_size=1,
                    num_workers=0,
                    force=True,
                ),
                "train",
            )
            cached = BatteryRawGraphDataset(
                **common,
                precomputed_cache_dir=str(root / "cache"),
                require_precomputed_cache=True,
            )

            self.assertEqual(len(cached), len(uncached))
            self.assertEqual(cached.cycle_scale, uncached.cycle_scale)
            for index in range(len(uncached)):
                direct = uncached[index]
                from_cache = cached[index]
                for key in ("maps", "y", "mask", "horizon", "target_steps", "history_features", "history_cycles"):
                    torch.testing.assert_close(from_cache[key], direct[key], rtol=0.0, atol=0.0)
                self.assertEqual(from_cache["prompt"], direct["prompt"])
                self.assertEqual(from_cache["cell_id"], direct["cell_id"])
                self.assertEqual(from_cache["cycle"], direct["cycle"])
            del cached
            gc.collect()

    def test_parallel_precompute_uses_parallel_path_and_matches_serial_cache(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_mit_data(root, [1, 1, 1, 1])

            def args(cache_dir, num_workers):
                return Namespace(
                    dataset="mit",
                    data_root=str(root),
                    cache_dir=str(cache_dir),
                    pred_len=20,
                    history_len=32,
                    resample_len=16,
                    delay_dim=2,
                    delay_lag=1,
                    no_derivative_map=True,
                    no_hankel_map=True,
                    no_ic_dv=True,
                    allow_summary_fallback=True,
                    max_cycles=None,
                    seed=42,
                    batch_size=4,
                    num_workers=num_workers,
                    force=True,
                )

            serial_path = precompute_split(args(root / "serial_cache", 0), "train")
            with patch.object(
                cache_precompute,
                "_parallel_cycle_map_results",
                wraps=cache_precompute._parallel_cycle_map_results,
            ) as parallel_path_spy, patch.object(
                cache_precompute,
                "BatteryRawGraphDataset",
                wraps=BatteryRawGraphDataset,
            ) as dataset_spy:
                parallel_path = precompute_split(args(root / "parallel_cache", 2), "train")

            parallel_path_spy.assert_called_once()
            self.assertEqual(dataset_spy.call_args.kwargs["cycle_cache_size"], 0)
            for filename in (
                "cycle_maps.npy",
                "history_indices.npy",
                "y.npy",
                "mask.npy",
                "horizon.npy",
                "target_steps.npy",
                "history_features.npy",
                "history_cycles.npy",
            ):
                np.testing.assert_array_equal(np.load(serial_path / filename), np.load(parallel_path / filename))
            self.assertEqual(
                (serial_path / "meta.jsonl").read_text(encoding="utf-8"),
                (parallel_path / "meta.jsonl").read_text(encoding="utf-8"),
            )


class FormalBatteryProtocolTests(unittest.TestCase):
    def test_exact_formal_protocol_is_accepted_and_nonconforming_lengths_are_rejected(self):
        battery_protocol.require_formal_battery_protocol(
            observed_cycles=battery_protocol.BATTERY_INPUT_CYCLES,
            prediction_cycles=battery_protocol.BATTERY_PREDICTION_CYCLES,
            context="test",
        )
        for observed_cycles, prediction_cycles in ((31, 20), (32, 19), (33, 20), (32, 21)):
            with self.subTest(observed_cycles=observed_cycles, prediction_cycles=prediction_cycles):
                with self.assertRaisesRegex(ValueError, "exactly 32 observed cycles and 20 future-only targets"):
                    battery_protocol.require_formal_battery_protocol(
                        observed_cycles=observed_cycles,
                        prediction_cycles=prediction_cycles,
                        context="test",
                    )

    def test_direct_battery_entrypoints_reject_nonconforming_formal_v3_lengths(self):
        commands = (
            [
                sys.executable,
                "-B",
                "-m",
                "bstalignment.train_graph_report",
                "--variant",
                "battery",
                "--history_len",
                "31",
                "--pred_len",
                "20",
                "--device",
                "cpu",
            ],
            [
                sys.executable,
                "-B",
                "-m",
                "bstalignment.train_battery_official_baselines",
                "--model",
                "patchtst",
                "--dataset",
                "mit",
                "--input_len",
                "32",
                "--pred_len",
                "19",
                "--device",
                "cpu",
            ],
            [
                sys.executable,
                "-B",
                "-m",
                "bstalignment.run_ablation_suite",
                "--variant",
                "battery",
                "--history_len",
                "31",
                "--pred_len",
                "20",
                "--dry_run",
            ],
        )
        for command in commands:
            with self.subTest(module=command[3]):
                result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("exactly 32 observed cycles and 20 future-only targets", result.stdout + result.stderr)


class RunMetadataTests(unittest.TestCase):
    strategy_version = TRAINING_STRATEGY_VERSION

    @staticmethod
    def valid_config(stage):
        args = {"pred_len": 20}
        args["input_len" if stage == "baseline" else "history_len"] = 32
        return {"training_strategy_version": TRAINING_STRATEGY_VERSION, "args": args}

    def test_valid_typed_root_metadata_matches_every_formal_stage(self):
        with TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "run_config.json"
            for stage in ("main", "baseline", "ablation"):
                with self.subTest(stage=stage):
                    config_path.write_text(json.dumps(self.valid_config(stage)), encoding="utf-8")
                    self.assertTrue(
                        battery_protocol.run_config_matches(
                            config_path,
                            training_strategy_version=self.strategy_version,
                            stage=stage,
                        )
                    )

    def test_malformed_truncated_nested_decoy_and_wrong_protocol_metadata_do_not_match(self):
        valid = self.valid_config("main")
        cases = {
            "malformed": "{not-json",
            "truncated": '{"training_strategy_version": "' + self.strategy_version + '",',
            "root_array": json.dumps([valid]),
            "nested_decoy": json.dumps(
                {
                    "training_strategy_version": "v2-stale",
                    "args": valid["args"],
                    "decoy": {"training_strategy_version": self.strategy_version},
                }
            ),
            "missing_args": json.dumps({"training_strategy_version": self.strategy_version}),
            "wrong_value": json.dumps(
                {"training_strategy_version": self.strategy_version, "args": {"history_len": 31, "pred_len": 20}}
            ),
            "wrong_type": json.dumps(
                {"training_strategy_version": self.strategy_version, "args": {"history_len": "32", "pred_len": True}}
            ),
        }
        with TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "run_config.json"
            for name, contents in cases.items():
                with self.subTest(case=name):
                    config_path.write_text(contents, encoding="utf-8")
                    self.assertFalse(
                        battery_protocol.run_config_matches(
                            config_path,
                            training_strategy_version=self.strategy_version,
                            stage="main",
                        )
                    )


class BaselineTrainerIntegrationTests(unittest.TestCase):
    def test_profile_is_not_overridden_when_cli_values_are_absent(self):
        args = Namespace(model="timecma", epochs=None, lr=None, weight_decay=None, early_stop_patience=None)
        profile = resolve_baseline_profile(args)
        self.assertEqual(profile.max_epochs, 100)
        self.assertEqual(profile.weight_decay, 1e-3)
        self.assertEqual(profile.early_stop_start_epoch, 50)

    def test_explicit_debug_override_is_visible(self):
        args = Namespace(model="patchtst", epochs=2, lr=None, weight_decay=None, early_stop_patience=1)
        profile = resolve_baseline_profile(args)
        self.assertEqual(profile.max_epochs, 2)
        self.assertEqual(profile.early_stop_patience, 1)


class PipelineScriptTests(unittest.TestCase):
    @staticmethod
    def formal_config(stage, version=TRAINING_STRATEGY_VERSION):
        args = {"pred_len": 20}
        args["input_len" if stage == "baseline" else "history_len"] = 32
        return {"training_strategy_version": version, "args": args}

    @staticmethod
    def run_formal_script(script, out_root, **overrides):
        executable = Path(sys.executable).as_posix()
        control_python = f"/mnt/{executable[0].lower()}/{executable[3:]}"
        settings = {
            "OUT_ROOT": out_root.as_posix(),
            "FORCE_RETRAIN": "0",
            "USE_BATTERY_GRAPH_CACHE": "0",
            "BASELINE_MODELS": "patchtst",
            "PY": "true",
            "CONTROL_PY": control_python,
        }
        settings.update(overrides)
        assignments = " ".join(f"{key}={shlex.quote(str(value))}" for key, value in settings.items())
        return subprocess.run(
            ["bash", "-c", f"{assignments} bash {shlex.quote(script)} ."],
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

    def test_formal_pipeline_is_main_first_and_uses_v3_root(self):
        text = Path("scripts/run_battery_v3_training_strategy_pipeline.sh").read_text(encoding="utf-8")
        self.assertIn("runs/full_hf_v3_training_strategy_nosoh", text)
        main = text.index("run_battery_main_full_hf.sh")
        baselines = text.index("run_battery_official_baselines.sh")
        ablations = text.index("run_battery_ablations_full_hf.sh")
        self.assertLess(main, baselines)
        self.assertLess(baselines, ablations)

    def test_pipeline_uses_approved_hardware_settings(self):
        text = Path("scripts/run_battery_v3_training_strategy_pipeline.sh").read_text(encoding="utf-8")
        self.assertIn('NUM_WORKERS="${NUM_WORKERS:-16}"', text)
        self.assertIn('BASELINE_NUM_WORKERS="${BASELINE_NUM_WORKERS:-8}"', text)
        self.assertIn('BATCH_SIZE="${BATCH_SIZE:-128}"', text)
        self.assertIn('BATTERY_GRAPH_CACHE_DIR:-runs/cache/battery_graph', text)
        for setting in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "TOKENIZERS_PARALLELISM",
            "PYTORCH_CUDA_ALLOC_CONF",
        ):
            self.assertIn(setting, text)

    def test_formal_scripts_use_shared_cache_and_versioned_completion(self):
        for path in (
            Path("scripts/run_battery_main_full_hf.sh"),
            Path("scripts/run_battery_official_baselines.sh"),
        ):
            text = path.read_text(encoding="utf-8")
            self.assertIn("run_config.json", text, path)
            self.assertIn("test_metrics.json", text, path)
            self.assertIn("v3-source-profiles-main-adaptive", text, path)
            self.assertIn("run-config-matches", text, path)
            self.assertNotIn("grep", text, path)
            self.assertNotIn("${OUT_ROOT}/cache/battery_graph", text, path)
        self.assertIn("runs/cache/battery_graph", Path("scripts/run_battery_main_full_hf.sh").read_text(encoding="utf-8"))
        self.assertIn("runs/cache/battery_graph", Path("scripts/run_battery_ablations_full_hf.sh").read_text(encoding="utf-8"))

    def test_ablation_shell_passes_force_and_version_controls(self):
        text = Path("scripts/run_battery_ablations_full_hf.sh").read_text(encoding="utf-8")
        self.assertIn('--training_strategy_version "$TRAINING_STRATEGY_VERSION"', text)
        self.assertIn("--force_retrain", text)

    def test_main_and_ablation_use_approved_workers_and_batch_size(self):
        for path in (
            Path("scripts/run_battery_main_full_hf.sh"),
            Path("scripts/run_battery_ablations_full_hf.sh"),
        ):
            text = path.read_text(encoding="utf-8")
            self.assertIn('BATCH_SIZE="${', text, path)
            self.assertIn(':-128}"', text, path)
            self.assertIn('NUM_WORKERS="${NUM_WORKERS:-16}"', text, path)
        baseline = Path("scripts/run_battery_official_baselines.sh").read_text(encoding="utf-8")
        self.assertIn('BATCH_SIZE="${BASELINE_BATCH_SIZE:-128}"', baseline)
        self.assertIn('NUM_WORKERS="${BASELINE_NUM_WORKERS:-8}"', baseline)

    def test_baseline_script_does_not_force_profile_budgets(self):
        text = Path("scripts/run_battery_official_baselines.sh").read_text(encoding="utf-8")
        self.assertNotIn("BASELINE_EPOCHS", text)
        self.assertNotIn("BASELINE_LR", text)
        self.assertNotIn("BASELINE_EARLY_STOP_PATIENCE", text)
        self.assertNotIn('--epochs "$EPOCHS"', text)
        self.assertNotIn('--lr "$LR"', text)
        self.assertNotIn('--early_stop_patience "$EARLY_STOP_PATIENCE"', text)

    def test_force_and_version_mismatch_remove_main_and_baseline_variant_outputs(self):
        scripts_and_outputs = (
            ("scripts/run_battery_main_full_hf.sh", Path("graph_report_ts/battery/mit")),
            ("scripts/run_battery_official_baselines.sh", Path("baselines/mit/patchtst")),
        )
        with TemporaryDirectory(dir=Path.cwd()) as tmp:
            temp_root = Path(tmp)
            relative_root = temp_root.relative_to(Path.cwd()) / "runs"
            for script, variant_suffix in scripts_and_outputs:
                for force_retrain, config_version in (("1", TRAINING_STRATEGY_VERSION), ("0", "v2-stale")):
                    variant_dir = Path(relative_root) / variant_suffix
                    variant_dir.mkdir(parents=True, exist_ok=True)
                    (variant_dir / "stale.pt").write_text("stale", encoding="utf-8")
                    stage = "baseline" if "official_baselines" in script else "main"
                    (variant_dir / "run_config.json").write_text(
                        json.dumps(self.formal_config(stage, config_version)),
                        encoding="utf-8",
                    )
                    result = self.run_formal_script(
                        script,
                        relative_root,
                        FORCE_RETRAIN=force_retrain,
                    )
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

                    self.assertFalse(variant_dir.exists(), (script, force_retrain, config_version))

    def test_invalid_metadata_removes_main_and_baseline_outputs_before_fresh_runs(self):
        scripts_and_outputs = (
            ("scripts/run_battery_main_full_hf.sh", Path("graph_report_ts/battery/mit"), "main"),
            ("scripts/run_battery_official_baselines.sh", Path("baselines/mit/patchtst"), "baseline"),
        )
        invalid_configs = {
            "malformed": "{not-json",
            "truncated": '{"training_strategy_version": "' + TRAINING_STRATEGY_VERSION + '",',
            "nested_decoy": json.dumps(
                {
                    "training_strategy_version": "v2-stale",
                    "args": {"history_len": 32, "input_len": 32, "pred_len": 20},
                    "decoy": {"training_strategy_version": TRAINING_STRATEGY_VERSION},
                }
            ),
            "wrong_protocol": json.dumps(
                {
                    "training_strategy_version": TRAINING_STRATEGY_VERSION,
                    "args": {"history_len": 31, "input_len": 31, "pred_len": 20},
                }
            ),
            "wrong_type": json.dumps(
                {
                    "training_strategy_version": TRAINING_STRATEGY_VERSION,
                    "args": {"history_len": "32", "input_len": "32", "pred_len": True},
                }
            ),
        }
        with TemporaryDirectory(dir=Path.cwd()) as tmp:
            relative_root = Path(tmp).relative_to(Path.cwd()) / "runs"
            for script, variant_suffix, _stage in scripts_and_outputs:
                for case, contents in invalid_configs.items():
                    with self.subTest(script=script, case=case):
                        variant_dir = relative_root / variant_suffix
                        variant_dir.mkdir(parents=True, exist_ok=True)
                        (variant_dir / "stale.pt").write_text("stale", encoding="utf-8")
                        (variant_dir / "test_metrics.json").write_text("{}", encoding="utf-8")
                        (variant_dir / "run_config.json").write_text(contents, encoding="utf-8")

                        result = self.run_formal_script(script, relative_root)

                        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                        self.assertFalse(variant_dir.exists())

    def test_matching_complete_outputs_skip_and_matching_incomplete_outputs_resume(self):
        scripts_and_outputs = (
            ("scripts/run_battery_main_full_hf.sh", Path("graph_report_ts/battery/mit"), "main"),
            ("scripts/run_battery_official_baselines.sh", Path("baselines/mit/patchtst"), "baseline"),
        )
        with TemporaryDirectory(dir=Path.cwd()) as tmp:
            relative_root = Path(tmp).relative_to(Path.cwd()) / "runs"
            for script, variant_suffix, stage in scripts_and_outputs:
                variant_dir = relative_root / variant_suffix
                variant_dir.mkdir(parents=True, exist_ok=True)
                marker = variant_dir / "marker.txt"
                marker.write_text("keep", encoding="utf-8")
                (variant_dir / "test_metrics.json").write_text("{}", encoding="utf-8")
                (variant_dir / "run_config.json").write_text(
                    json.dumps(self.formal_config(stage)),
                    encoding="utf-8",
                )

                result = self.run_formal_script(script, relative_root)

                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertTrue(marker.exists())
                self.assertIn("skip completed", result.stdout)

                (variant_dir / "test_metrics.json").unlink()
                resume_checkpoint = variant_dir / "last.pt"
                resume_checkpoint.write_text("resume", encoding="utf-8")
                result = self.run_formal_script(script, relative_root)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertTrue(resume_checkpoint.exists())

    def test_formal_shell_entrypoints_reject_environment_length_overrides(self):
        cases = (
            ("scripts/run_battery_main_full_hf.sh", {"HISTORY_LEN": "31"}),
            ("scripts/run_battery_official_baselines.sh", {"INPUT_LEN": "31"}),
            ("scripts/run_battery_ablations_full_hf.sh", {"PRED_LEN": "19"}),
            ("scripts/run_battery_v3_training_strategy_pipeline.sh", {"HISTORY_LEN": "31"}),
        )
        with TemporaryDirectory(dir=Path.cwd()) as tmp:
            relative_root = Path(tmp).relative_to(Path.cwd()) / "runs"
            for script, overrides in cases:
                with self.subTest(script=script):
                    result = self.run_formal_script(script, relative_root, **overrides)
                    self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn("exactly 32 observed cycles and 20 future-only targets", result.stdout + result.stderr)

    def test_matching_incomplete_outputs_are_preserved_for_resume(self):
        scripts_and_outputs = (
            ("scripts/run_battery_main_full_hf.sh", Path("graph_report_ts/battery/mit")),
            ("scripts/run_battery_official_baselines.sh", Path("baselines/mit/patchtst")),
        )
        with TemporaryDirectory(dir=Path.cwd()) as tmp:
            temp_root = Path(tmp)
            relative_root = temp_root.relative_to(Path.cwd()) / "runs"
            for script, variant_suffix in scripts_and_outputs:
                variant_dir = Path(relative_root) / variant_suffix
                variant_dir.mkdir(parents=True, exist_ok=True)
                stale_checkpoint = variant_dir / "last.pt"
                stale_checkpoint.write_text("resume", encoding="utf-8")
                stage = "baseline" if "official_baselines" in script else "main"
                (variant_dir / "run_config.json").write_text(
                    json.dumps(self.formal_config(stage)),
                    encoding="utf-8",
                )
                result = self.run_formal_script(script, relative_root)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

                self.assertTrue(stale_checkpoint.exists(), script)
                text = Path(script).read_text(encoding="utf-8")
                self.assertIn('if [ "$FRESH_RUN" = "1" ]; then', text)
                self.assertIn("RESUME_ARGS=(--no_resume)", text)

    def test_public_docs_describe_the_v3_training_protocol(self):
        readme = Path("README.md").read_text(encoding="utf-8")
        report = Path("docs/work_report.md").read_text(encoding="utf-8")
        workflow = Path("docs/cloud_training_workflow.md").read_text(encoding="utf-8")

        for text in (readme, report):
            self.assertIn("scripts/run_battery_v3_training_strategy_pipeline.sh", text)
            self.assertIn("runs/full_hf_v3_training_strategy_nosoh", text)
            self.assertIn("main -> baselines -> ablations", text)
            self.assertIn("DistilBERT", text)
            self.assertIn("5-epoch LR warmup", text)
            self.assertIn("delayed/ramped alignment", text)
            self.assertIn("plateau scheduler and early stopping", text)
            self.assertIn("source-native", text)
            self.assertIn("32 observed cycles", text)
            self.assertIn("20 future-only", text)
            self.assertIn("train-only dataset-global cycle scaling", text)
            self.assertIn("no clipping", text)

        self.assertIn("fixed AdamW/SmoothL1/no scheduler", report)
        self.assertIn("73/79/54/77/72", report)
        self.assertIn("[B, 32, N_patch, D]", report)
        self.assertIn("RTX4090 48GiB", workflow)
        self.assertIn("208 CPU threads", workflow)
        self.assertIn("FORCE_RETRAIN=0", workflow)
        self.assertIn("no AMP", workflow)
        mkdir_command = "mkdir -p runs/full_hf_v3_training_strategy_nosoh/logs"
        tee_command = "tee runs/full_hf_v3_training_strategy_nosoh/logs/v3_start.log"
        self.assertIn(mkdir_command, workflow)
        self.assertLess(workflow.index(mkdir_command), workflow.index(tee_command))


class AblationCompletionPolicyTests(unittest.TestCase):
    strategy_version = "v3-source-profiles-main-adaptive"

    @staticmethod
    def write_json(path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    def formal_config(self, **args):
        protocol_args = {"history_len": 32, "pred_len": 20}
        protocol_args.update(args)
        return {"training_strategy_version": self.strategy_version, "args": protocol_args}

    def test_matching_version_with_both_result_files_skips(self):
        with TemporaryDirectory() as tmp:
            result_dir = Path(tmp)
            self.write_json(result_dir / "test_metrics.json", {"mse": 0.1})
            self.write_json(
                result_dir / "run_config.json",
                self.formal_config(),
            )

            self.assertTrue(should_skip_ablation(result_dir, self.strategy_version, force_retrain=False))

    def test_missing_result_or_config_retrains(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics_missing = root / "metrics_missing"
            metrics_missing.mkdir()
            self.write_json(
                metrics_missing / "run_config.json",
                self.formal_config(),
            )
            config_missing = root / "config_missing"
            config_missing.mkdir()
            self.write_json(config_missing / "test_metrics.json", {"mse": 0.1})

            self.assertFalse(should_skip_ablation(metrics_missing, self.strategy_version, force_retrain=False))
            self.assertFalse(should_skip_ablation(config_missing, self.strategy_version, force_retrain=False))

    def test_general_ablation_matching_does_not_require_battery_protocol_fields(self):
        with TemporaryDirectory() as tmp:
            result_dir = Path(tmp)
            self.write_json(result_dir / "test_metrics.json", {"mse": 0.1})
            self.write_json(
                result_dir / "run_config.json",
                {
                    "training_strategy_version": self.strategy_version,
                    "args": {"input_len": 96, "pred_len": 96},
                },
            )

            self.assertTrue(
                should_skip_ablation(
                    result_dir,
                    self.strategy_version,
                    force_retrain=False,
                    protocol_stage=None,
                )
            )

    def test_malformed_or_mismatched_config_retrains(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            malformed = root / "malformed"
            malformed.mkdir()
            self.write_json(malformed / "test_metrics.json", {"mse": 0.1})
            (malformed / "run_config.json").write_text("{not-json", encoding="utf-8")
            invalid_encoding = root / "invalid_encoding"
            invalid_encoding.mkdir()
            self.write_json(invalid_encoding / "test_metrics.json", {"mse": 0.1})
            (invalid_encoding / "run_config.json").write_bytes(b"\xff\xfe")
            mismatched = root / "mismatched"
            mismatched.mkdir()
            self.write_json(mismatched / "test_metrics.json", {"mse": 0.1})
            self.write_json(
                mismatched / "run_config.json",
                self.formal_config() | {"training_strategy_version": "v2-legacy"},
            )

            wrong_protocol = root / "wrong_protocol"
            wrong_protocol.mkdir()
            self.write_json(wrong_protocol / "test_metrics.json", {"mse": 0.1})
            self.write_json(wrong_protocol / "run_config.json", self.formal_config(history_len=31))

            nested_decoy = root / "nested_decoy"
            nested_decoy.mkdir()
            self.write_json(nested_decoy / "test_metrics.json", {"mse": 0.1})
            self.write_json(
                nested_decoy / "run_config.json",
                {
                    "training_strategy_version": "v2-legacy",
                    "args": {"history_len": 32, "pred_len": 20},
                    "decoy": {"training_strategy_version": self.strategy_version},
                },
            )

            self.assertFalse(should_skip_ablation(malformed, self.strategy_version, force_retrain=False))
            self.assertFalse(should_skip_ablation(invalid_encoding, self.strategy_version, force_retrain=False))
            self.assertFalse(should_skip_ablation(mismatched, self.strategy_version, force_retrain=False))
            self.assertFalse(should_skip_ablation(wrong_protocol, self.strategy_version, force_retrain=False))
            self.assertFalse(should_skip_ablation(nested_decoy, self.strategy_version, force_retrain=False))

    def test_force_retrain_never_skips_and_removes_variant_output(self):
        with TemporaryDirectory() as tmp:
            variant_dir = Path(tmp) / "full"
            result_dir = variant_dir / "battery" / "mit"
            result_dir.mkdir(parents=True)
            self.write_json(result_dir / "test_metrics.json", {"mse": 0.1})
            self.write_json(
                result_dir / "run_config.json",
                self.formal_config(),
            )

            self.assertFalse(should_skip_ablation(result_dir, self.strategy_version, force_retrain=True))
            remove_ablation_output_if_forced(variant_dir, force_retrain=True)
            self.assertFalse(variant_dir.exists())

    def test_version_mismatch_removes_variant_output_before_a_fresh_ablation(self):
        with TemporaryDirectory() as tmp:
            variant_dir = Path(tmp) / "full"
            result_dir = variant_dir / "battery" / "mit"
            result_dir.mkdir(parents=True)
            self.write_json(result_dir / "run_config.json", {"training_strategy_version": "v2-legacy"})
            (result_dir / "last.pt").write_text("stale", encoding="utf-8")

            start_fresh = not has_matching_strategy_version(result_dir, self.strategy_version)
            remove_ablation_output_if_fresh(variant_dir, start_fresh)

            self.assertTrue(start_fresh)
            self.assertFalse(variant_dir.exists())


class TrainingProfileTests(unittest.TestCase):
    def test_all_official_baselines_have_explicit_profiles(self):
        self.assertEqual(
            set(BASELINE_TRAINING_PROFILES),
            {"patchtst", "itransformer", "timecma", "timesnet", "dlinear", "time_llm"},
        )
        for profile in BASELINE_TRAINING_PROFILES.values():
            self.assertGreater(profile.max_epochs, 0)
            self.assertGreater(profile.early_stop_patience, 0)
            self.assertIn(profile.loss, {"mse"})

    def test_source_native_profile_values(self):
        patch = BASELINE_TRAINING_PROFILES["patchtst"]
        self.assertEqual((patch.optimizer, patch.scheduler, patch.max_epochs), ("adam", "one_cycle", 100))
        self.assertEqual(patch.pct_start, 0.3)
        self.assertEqual(BASELINE_TRAINING_PROFILES["itransformer"].max_epochs, 10)
        self.assertEqual(BASELINE_TRAINING_PROFILES["timesnet"].scheduler, "type1")
        self.assertEqual(BASELINE_TRAINING_PROFILES["dlinear"].early_stop_patience, 3)
        self.assertEqual(BASELINE_TRAINING_PROFILES["time_llm"].pct_start, 0.2)
        timecma = BASELINE_TRAINING_PROFILES["timecma"]
        self.assertEqual((timecma.optimizer, timecma.scheduler), ("adamw", "cosine"))
        self.assertEqual((timecma.weight_decay, timecma.gradient_clip), (1e-3, 5.0))
        self.assertEqual(timecma.early_stop_start_epoch, 50)

    def test_main_profile_matches_approved_design(self):
        self.assertEqual(MAIN_TRAINING_PROFILE.core_lr, 1e-3)
        self.assertEqual(MAIN_TRAINING_PROFILE.semantic_lr, 3e-4)
        self.assertEqual(MAIN_TRAINING_PROFILE.lr_warmup_epochs, 5)
        self.assertEqual(MAIN_TRAINING_PROFILE.align_start_epoch, 6)
        self.assertEqual(MAIN_TRAINING_PROFILE.align_full_epoch, 15)
        self.assertEqual(MAIN_TRAINING_PROFILE.early_stop_start_epoch, 20)
        self.assertEqual(MAIN_TRAINING_PROFILE.early_stop_patience, 20)
        self.assertTrue(TRAINING_STRATEGY_VERSION.startswith("v3-"))


class BaselineMechanicsTests(unittest.TestCase):
    def test_one_cycle_steps_per_batch(self):
        model = torch.nn.Linear(2, 1)
        profile = get_baseline_training_profile("patchtst")
        optimizer = build_baseline_optimizer(model, profile)
        scheduler = build_baseline_scheduler(optimizer, profile, steps_per_epoch=4)
        before = scheduler.last_epoch
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        step_baseline_batch_scheduler(scheduler, profile)
        self.assertEqual(scheduler.last_epoch, before + 1)

    def test_type1_sets_epoch_two_to_half_lr_after_epoch_one_completes(self):
        model = torch.nn.Linear(2, 1)
        profile = get_baseline_training_profile("itransformer")
        optimizer = build_baseline_optimizer(model, profile)
        scheduler = build_baseline_scheduler(optimizer, profile, steps_per_epoch=4)
        step_baseline_epoch_scheduler(scheduler, optimizer, profile, epoch=1)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 5e-5)

    def test_formal_time_llm_path_has_no_autocast_or_amp(self):
        text = Path("bstalignment/train_battery_official_baselines.py").read_text(encoding="utf-8").lower()
        self.assertNotIn("autocast", text)
        self.assertNotIn("gradscaler", text)

    def test_timecma_cosine_and_mse(self):
        model = torch.nn.Linear(2, 1)
        profile = get_baseline_training_profile("timecma")
        optimizer = build_baseline_optimizer(model, profile)
        scheduler = build_baseline_scheduler(optimizer, profile, steps_per_epoch=4)
        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
        pred = torch.tensor([[1.0, 3.0]])
        target = torch.tensor([[0.0, 1.0]])
        self.assertEqual(float(baseline_regression_loss(pred, target, profile)), 2.5)


class TinyGraphReport(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.graph_encoder = torch.nn.Linear(2, 2)
        self.context_norm = torch.nn.LayerNorm(2)
        self.decoder = torch.nn.Embedding(4, 2)
        self.text_encoder = torch.nn.Module()
        self.text_encoder.backbone = torch.nn.Linear(2, 2)
        self.text_encoder.emb = torch.nn.Embedding(8, 2)
        self.text_encoder.proj = torch.nn.Linear(2, 2)
        self.semantic_fusion = torch.nn.Linear(2, 2)
        for parameter in self.text_encoder.backbone.parameters():
            parameter.requires_grad = False


class MainStrategyTests(unittest.TestCase):
    def test_parameter_groups_cover_trainable_parameters_once(self):
        model = TinyGraphReport()
        optimizer = build_graph_report_optimizer(model, MAIN_TRAINING_PROFILE)
        optimized = [parameter for group in optimizer.param_groups for parameter in group["params"]]
        expected = [parameter for parameter in model.parameters() if parameter.requires_grad]
        self.assertEqual({id(parameter) for parameter in optimized}, {id(parameter) for parameter in expected})
        self.assertEqual(len(optimized), len({id(parameter) for parameter in optimized}))
        self.assertFalse(any(parameter.requires_grad for parameter in model.text_encoder.backbone.parameters()))

    def test_embedding_parameters_are_excluded_from_weight_decay(self):
        model = TinyGraphReport()
        optimizer = build_graph_report_optimizer(model, MAIN_TRAINING_PROFILE)
        for embedding_parameter in (model.decoder.weight, model.text_encoder.emb.weight):
            matching_groups = [
                group
                for group in optimizer.param_groups
                if any(parameter is embedding_parameter for parameter in group["params"])
            ]
            self.assertEqual(len(matching_groups), 1)
            self.assertEqual(matching_groups[0]["weight_decay"], 0.0)

    def test_actual_graph_report_layernorm_parameters_are_excluded_from_weight_decay(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="enable_nested_tensor is True")
            model = GraphReportTS(
                GraphReportTSConfig(
                    d_model=8,
                    patch_size=2,
                    patch_stride=1,
                    graph_layers=1,
                    topk_edges=1,
                    use_hf_text_encoder=False,
                    temporal_heads=2,
                )
            )
        optimizer = build_graph_report_optimizer(model, MAIN_TRAINING_PROFILE)

        for parameter in (model.context_fuser[0].weight, model.context_fuser[0].bias):
            groups = [
                group
                for group in optimizer.param_groups
                if any(candidate is parameter for candidate in group["params"])
            ]
            self.assertEqual(len(groups), 1)
            self.assertEqual(groups[0]["weight_decay"], 0.0)

    def test_lr_warmup_reaches_role_targets(self):
        model = TinyGraphReport()
        optimizer = build_graph_report_optimizer(model, MAIN_TRAINING_PROFILE)
        scheduler = GraphReportScheduler(optimizer, MAIN_TRAINING_PROFILE)
        scheduler.start_epoch(1)
        first = graph_report_group_lrs(optimizer)
        scheduler.start_epoch(5)
        full = graph_report_group_lrs(optimizer)
        self.assertAlmostEqual(first["core"], 1e-4)
        self.assertAlmostEqual(first["semantic"], 3e-5)
        self.assertAlmostEqual(full["core"], 1e-3)
        self.assertAlmostEqual(full["semantic"], 3e-4)

    def test_plateau_reduces_both_roles_and_respects_minimum_lrs(self):
        model = TinyGraphReport()
        optimizer = build_graph_report_optimizer(model, MAIN_TRAINING_PROFILE)
        scheduler = GraphReportScheduler(optimizer, MAIN_TRAINING_PROFILE)
        scheduler.start_epoch(MAIN_TRAINING_PROFILE.lr_warmup_epochs)
        epoch = MAIN_TRAINING_PROFILE.lr_warmup_epochs
        reduction_interval = MAIN_TRAINING_PROFILE.plateau_patience + 1
        for _ in range(1 + reduction_interval):
            scheduler.step_validation(epoch, 1.0)
            epoch += 1
        reduced = graph_report_group_lrs(optimizer)
        self.assertAlmostEqual(reduced["core"], MAIN_TRAINING_PROFILE.core_lr * 0.5)
        self.assertAlmostEqual(reduced["semantic"], MAIN_TRAINING_PROFILE.semantic_lr * 0.5)

        for _ in range(reduction_interval * 10):
            scheduler.step_validation(epoch, 1.0)
            epoch += 1
        floored = graph_report_group_lrs(optimizer)
        self.assertAlmostEqual(floored["core"], MAIN_TRAINING_PROFILE.core_min_lr)
        self.assertAlmostEqual(floored["semantic"], MAIN_TRAINING_PROFILE.semantic_min_lr)

    def test_advanced_plateau_state_restores_and_continues(self):
        model = TinyGraphReport()
        optimizer = build_graph_report_optimizer(model, MAIN_TRAINING_PROFILE)
        scheduler = GraphReportScheduler(optimizer, MAIN_TRAINING_PROFILE)
        scheduler.start_epoch(MAIN_TRAINING_PROFILE.lr_warmup_epochs)
        epoch = MAIN_TRAINING_PROFILE.lr_warmup_epochs
        reduction_interval = MAIN_TRAINING_PROFILE.plateau_patience + 1
        for _ in range(1 + reduction_interval + 3):
            scheduler.step_validation(epoch, 1.0)
            epoch += 1

        optimizer_state = optimizer.state_dict()
        state = scheduler.state_dict()
        restored_optimizer = build_graph_report_optimizer(TinyGraphReport(), MAIN_TRAINING_PROFILE)
        restored = GraphReportScheduler(restored_optimizer, MAIN_TRAINING_PROFILE)
        restored_optimizer.load_state_dict(optimizer_state)
        restored.load_state_dict(state)
        self.assertEqual(restored.state_dict(), state)

        for _ in range(3):
            scheduler.step_validation(epoch, 1.0)
            restored.step_validation(epoch, 1.0)
            epoch += 1
        self.assertEqual(restored.state_dict(), scheduler.state_dict())
        self.assertEqual(graph_report_group_lrs(restored_optimizer), graph_report_group_lrs(optimizer))

    def test_align_weight_is_delayed_and_ramped(self):
        self.assertEqual(graph_report_align_weight(5, MAIN_TRAINING_PROFILE), 0.0)
        self.assertAlmostEqual(graph_report_align_weight(6, MAIN_TRAINING_PROFILE), 1e-4)
        self.assertAlmostEqual(graph_report_align_weight(15, MAIN_TRAINING_PROFILE), 1e-3)


class MainTrainerPolicyTests(unittest.TestCase):
    def test_empty_validation_is_rejected_without_test_fallback(self):
        with self.assertRaisesRegex(RuntimeError, "validation"):
            training_strategy.require_nonempty_splits([1], [], [1], "fixture trainer")
        baseline_source = Path("bstalignment/train_battery_official_baselines.py").read_text(encoding="utf-8")
        graph_source = Path("bstalignment/train_graph_report.py").read_text(encoding="utf-8")
        self.assertNotIn("val_eval_ds", baseline_source)
        self.assertIn("require_nonempty_splits", baseline_source)
        self.assertIn("require_nonempty_splits", graph_source)

    def test_checkpoint_strategy_version_mismatch_is_rejected_by_both_trainers(self):
        with self.assertRaisesRegex(RuntimeError, "training strategy version"):
            training_strategy.require_checkpoint_strategy_version(
                {"training_strategy_version": "v2-stale"},
                "fixture trainer",
            )
        for path in (
            Path("bstalignment/train_battery_official_baselines.py"),
            Path("bstalignment/train_graph_report.py"),
        ):
            self.assertIn("require_checkpoint_strategy_version", path.read_text(encoding="utf-8"), path)

    def test_stale_count_starts_at_epoch_20_and_stops_after_20_failures(self):
        stale = update_graph_report_stale(
            epoch=19,
            stale=7,
            improved=False,
            profile=MAIN_TRAINING_PROFILE,
        )
        self.assertEqual(stale, 0)

        stale = update_graph_report_stale(
            epoch=20,
            stale=stale,
            improved=False,
            profile=MAIN_TRAINING_PROFILE,
        )
        self.assertEqual(stale, 1)

        for epoch in range(21, 40):
            stale = update_graph_report_stale(epoch, stale, improved=False, profile=MAIN_TRAINING_PROFILE)
        self.assertEqual(stale, 20)
        self.assertTrue(should_stop_graph_report(epoch=39, stale=stale, profile=MAIN_TRAINING_PROFILE))

    def test_early_stop_counter_is_inactive_before_epoch_20(self):
        self.assertFalse(should_stop_graph_report(epoch=19, stale=100, profile=MAIN_TRAINING_PROFILE))
        self.assertFalse(should_stop_graph_report(epoch=38, stale=19, profile=MAIN_TRAINING_PROFILE))
        self.assertTrue(should_stop_graph_report(epoch=39, stale=20, profile=MAIN_TRAINING_PROFILE))
