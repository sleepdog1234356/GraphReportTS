from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np
import torch

from bstalignment.v2.train_general import (
    fit_shared_ridge_anchor,
    freeze_linear_anchor,
    linear_anchor_settings,
    validate_resume_linear_anchor,
    write_linear_anchor,
)
from bstalignment.v2.training import (
    V2TrainingConfig,
    apply_warmup,
    build_optimizer,
    build_plateau_scheduler,
    optimizer_learning_rates,
    should_stop_v2,
    step_plateau_scheduler,
)


def _role_optimizer(config: V2TrainingConfig) -> torch.optim.Optimizer:
    core = torch.nn.Parameter(torch.tensor(1.0))
    semantic = torch.nn.Parameter(torch.tensor(2.0))
    return torch.optim.AdamW(
        [
            {"params": [core], "lr": config.core_lr, "role": "core"},
            {"params": [semantic], "lr": config.semantic_lr, "role": "semantic"},
        ]
    )


def _known_linear_series(
    *,
    samples: int = 64,
    variables: int = 3,
    input_len: int = 36,
    horizon: int = 4,
) -> tuple[np.ndarray, list[int], np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    weight = rng.normal(0.0, 0.2, size=(horizon, input_len)).astype(np.float32)
    bias = rng.normal(0.0, 0.1, size=(horizon,)).astype(np.float32)
    block = input_len + horizon
    values = np.zeros((samples * block, variables), dtype=np.float32)
    starts: list[int] = []
    for index in range(samples):
        start = index * block
        history = rng.normal(size=(input_len, variables)).astype(np.float32)
        target = (history.T @ weight.T + bias).T
        values[start : start + input_len] = history
        values[start + input_len : start + block] = target
        starts.append(start)
    return values, starts, weight, bias


class _AnchorModel(torch.nn.Module):
    def __init__(self, input_len: int = 36, max_horizon: int = 60) -> None:
        super().__init__()
        self.head = torch.nn.Module()
        self.head.linear_anchor = torch.nn.Linear(input_len, max_horizon)
        self.other = torch.nn.Linear(2, 2)


class LinearAnchorInitializationTests(unittest.TestCase):
    def test_ridge_recovers_known_shared_linear_mapping(self):
        values, samples, expected_weight, expected_bias = _known_linear_series()

        weight, bias = fit_shared_ridge_anchor(
            values,
            samples,
            input_len=36,
            horizon=4,
            ridge=1e-8,
            chunk_size=11,
        )

        np.testing.assert_allclose(weight, expected_weight, atol=2e-5, rtol=2e-5)
        np.testing.assert_allclose(bias, expected_bias, atol=2e-5, rtol=2e-5)

    def test_ridge_coefficients_are_written_to_active_anchor_rows(self):
        model = _AnchorModel()
        weight = np.arange(4 * 36, dtype=np.float32).reshape(4, 36) / 100.0
        bias = np.arange(4, dtype=np.float32) / 10.0
        untouched = model.head.linear_anchor.weight[4:].detach().clone()

        write_linear_anchor(model.head.linear_anchor, weight, bias, horizon=4)

        torch.testing.assert_close(model.head.linear_anchor.weight[:4], torch.from_numpy(weight))
        torch.testing.assert_close(model.head.linear_anchor.bias[:4], torch.from_numpy(bias))
        torch.testing.assert_close(model.head.linear_anchor.weight[4:], untouched)

    def test_frozen_anchor_parameters_are_excluded_from_optimizer(self):
        model = _AnchorModel()
        freeze_linear_anchor(model.head.linear_anchor, frozen=True)

        optimizer = build_optimizer(model, V2TrainingConfig())
        optimized = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}

        self.assertNotIn(id(model.head.linear_anchor.weight), optimized)
        self.assertNotIn(id(model.head.linear_anchor.bias), optimized)
        self.assertIn(id(model.other.weight), optimized)

    def test_legacy_resume_is_only_compatible_with_original_anchor_behavior(self):
        validate_resume_linear_anchor(None, linear_anchor_settings("random", 1e-4, False))

        with self.assertRaisesRegex(ValueError, "predates linear-anchor settings"):
            validate_resume_linear_anchor(None, linear_anchor_settings("ridge", 1e-4, False))

    def test_random_anchor_cannot_be_frozen(self):
        with self.assertRaisesRegex(ValueError, "requires ridge initialization"):
            linear_anchor_settings("random", 1e-4, True)


class V2WarmupAndPlateauTests(unittest.TestCase):
    def test_warmup_does_not_reset_reduced_lrs_after_epoch_five(self):
        config = V2TrainingConfig()
        optimizer = _role_optimizer(config)
        apply_warmup(optimizer, 3, config)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 6e-4)
        self.assertAlmostEqual(optimizer.param_groups[1]["lr"], 1.8e-4)
        apply_warmup(optimizer, 5, config)
        optimizer.param_groups[0]["lr"] = 5e-4
        optimizer.param_groups[1]["lr"] = 1.5e-4
        apply_warmup(optimizer, 6, config)
        self.assertEqual(
            optimizer_learning_rates(optimizer),
            {"core": 5e-4, "semantic": 1.5e-4},
        )

    def test_plateau_reduces_both_roles_and_reports_actual_reductions(self):
        config = replace(V2TrainingConfig(), plateau_patience=0, plateau_cooldown=0)
        optimizer = _role_optimizer(config)
        scheduler = build_plateau_scheduler(optimizer, config)
        self.assertFalse(step_plateau_scheduler(scheduler, optimizer, 1.0))
        self.assertTrue(step_plateau_scheduler(scheduler, optimizer, 1.1))
        self.assertEqual(
            optimizer_learning_rates(optimizer),
            {"core": 5e-4, "semantic": 1.5e-4},
        )

    def test_plateau_respects_per_role_minimum_lrs(self):
        config = replace(V2TrainingConfig(), plateau_patience=0, plateau_cooldown=0)
        optimizer = _role_optimizer(config)
        scheduler = build_plateau_scheduler(optimizer, config)
        scheduler.step(1.0)
        for metric in range(1, 20):
            step_plateau_scheduler(scheduler, optimizer, 1.0 + metric)
        rates = optimizer_learning_rates(optimizer)
        self.assertEqual(rates["core"], config.core_min_lr)
        self.assertEqual(rates["semantic"], config.semantic_min_lr)

    def test_scheduler_state_round_trip_preserves_bad_epoch_counter(self):
        config = V2TrainingConfig()
        optimizer = _role_optimizer(config)
        scheduler = build_plateau_scheduler(optimizer, config)
        scheduler.step(1.0)
        scheduler.step(1.1)
        state = scheduler.state_dict()
        restored_optimizer = _role_optimizer(config)
        restored = build_plateau_scheduler(restored_optimizer, config)
        restored.load_state_dict(state)
        self.assertEqual(restored.state_dict()["num_bad_epochs"], 1)


class V2GuardedEarlyStopTests(unittest.TestCase):
    def test_stale_patience_alone_cannot_stop_before_two_lr_reductions(self):
        config = V2TrainingConfig(patience=20, min_lr_reductions_before_stop=2)
        self.assertFalse(should_stop_v2(stale=20, lr_reductions=0, config=config))
        self.assertFalse(should_stop_v2(stale=40, lr_reductions=1, config=config))
        self.assertTrue(should_stop_v2(stale=20, lr_reductions=2, config=config))

    def test_fewer_than_twenty_stale_epochs_never_stops(self):
        config = V2TrainingConfig(patience=20, min_lr_reductions_before_stop=2)
        self.assertFalse(should_stop_v2(stale=19, lr_reductions=9, config=config))


if __name__ == "__main__":
    unittest.main()
