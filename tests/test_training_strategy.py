from argparse import Namespace
import unittest

import torch

from bstalignment.train_battery_official_baselines import resolve_baseline_profile
from bstalignment.training_strategy import (
    BASELINE_TRAINING_PROFILES,
    MAIN_TRAINING_PROFILE,
    TRAINING_STRATEGY_VERSION,
    baseline_regression_loss,
    build_baseline_optimizer,
    build_baseline_scheduler,
    get_baseline_training_profile,
    step_baseline_batch_scheduler,
    step_baseline_epoch_scheduler,
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
