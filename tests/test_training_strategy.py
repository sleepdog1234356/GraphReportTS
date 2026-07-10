import unittest

from bstalignment.training_strategy import (
    BASELINE_TRAINING_PROFILES,
    MAIN_TRAINING_PROFILE,
    TRAINING_STRATEGY_VERSION,
)


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
