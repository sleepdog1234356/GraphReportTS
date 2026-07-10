from argparse import Namespace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from bstalignment.run_ablation_suite import remove_ablation_output_if_forced, should_skip_ablation
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

        self.assertIn("fixed AdamW/SmoothL1/no scheduler", report)
        self.assertIn("73/79/54/77/72", report)
        self.assertIn("[B, 32, N_patch, D]", report)
        self.assertIn("RTX4090 48GiB", workflow)
        self.assertIn("208 CPU threads", workflow)
        self.assertIn("FORCE_RETRAIN=0", workflow)
        self.assertIn("no AMP", workflow)


class AblationCompletionPolicyTests(unittest.TestCase):
    strategy_version = "v3-source-profiles-main-adaptive"

    @staticmethod
    def write_json(path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    def test_matching_version_with_both_result_files_skips(self):
        with TemporaryDirectory() as tmp:
            result_dir = Path(tmp)
            self.write_json(result_dir / "test_metrics.json", {"mse": 0.1})
            self.write_json(
                result_dir / "run_config.json",
                {"training_strategy_version": self.strategy_version},
            )

            self.assertTrue(should_skip_ablation(result_dir, self.strategy_version, force_retrain=False))

    def test_missing_result_or_config_retrains(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics_missing = root / "metrics_missing"
            metrics_missing.mkdir()
            self.write_json(
                metrics_missing / "run_config.json",
                {"training_strategy_version": self.strategy_version},
            )
            config_missing = root / "config_missing"
            config_missing.mkdir()
            self.write_json(config_missing / "test_metrics.json", {"mse": 0.1})

            self.assertFalse(should_skip_ablation(metrics_missing, self.strategy_version, force_retrain=False))
            self.assertFalse(should_skip_ablation(config_missing, self.strategy_version, force_retrain=False))

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
                {"training_strategy_version": "v2-legacy"},
            )

            self.assertFalse(should_skip_ablation(malformed, self.strategy_version, force_retrain=False))
            self.assertFalse(should_skip_ablation(invalid_encoding, self.strategy_version, force_retrain=False))
            self.assertFalse(should_skip_ablation(mismatched, self.strategy_version, force_retrain=False))

    def test_force_retrain_never_skips_and_removes_variant_output(self):
        with TemporaryDirectory() as tmp:
            variant_dir = Path(tmp) / "full"
            result_dir = variant_dir / "battery" / "mit"
            result_dir.mkdir(parents=True)
            self.write_json(result_dir / "test_metrics.json", {"mse": 0.1})
            self.write_json(
                result_dir / "run_config.json",
                {"training_strategy_version": self.strategy_version},
            )

            self.assertFalse(should_skip_ablation(result_dir, self.strategy_version, force_retrain=True))
            remove_ablation_output_if_forced(variant_dir, force_retrain=True)
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

    def test_type1_halves_lr_each_epoch(self):
        model = torch.nn.Linear(2, 1)
        profile = get_baseline_training_profile("itransformer")
        optimizer = build_baseline_optimizer(model, profile)
        scheduler = build_baseline_scheduler(optimizer, profile, steps_per_epoch=4)
        step_baseline_epoch_scheduler(scheduler, optimizer, profile, epoch=2)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 5e-5)

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
