# Training Strategy Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore source-consistent optimization for all official baselines, add a structure-aware GraphReportTS training policy, and run the formal pipeline in main-model-first order.

**Architecture:** Add one focused `training_strategy` module that owns immutable training profiles, optimizer construction, scheduler stepping, parameter grouping, LR/align warmup, and strategy versioning. The existing baseline and GraphReportTS trainers remain responsible for data loading, forward/backward passes, metrics, and checkpoints, but consume this shared strategy API. Shell scripts only orchestrate stages and never override model-native profile values.

**Tech Stack:** Python 3, PyTorch, `unittest`, Bash, Hugging Face Transformers, Git, remote Linux/RTX 4090.

## Global Constraints

- Keep the leak-free, no-historical-SOH input schema unchanged.
- Keep input length 32 cycles, prediction length 20 steps, batch size 128, and seed 42 for formal battery comparisons.
- Keep the DistilBERT backbone frozen; train only its projection and downstream semantic modules.
- Select checkpoints only with validation MSE; evaluate the test split only after training.
- Use `runs/full_hf_v3_training_strategy_nosoh` and never mix v2 metrics into v3 summaries.
- Run formal stages in this exact order: GraphReportTS main models, official baselines, battery ablations.
- Preserve model architecture and full data; model-native epoch budgets are source protocol, not lightweight substitutes.
- Keep shell files LF-only.

---

### Task 1: Define Versioned Training Profiles

**Files:**
- Create: `bstalignment/training_strategy.py`
- Create: `tests/__init__.py`
- Create: `tests/test_training_strategy.py`

**Interfaces:**
- Produces: `TRAINING_STRATEGY_VERSION: str`
- Produces: `BaselineTrainingProfile`
- Produces: `BASELINE_TRAINING_PROFILES: dict[str, BaselineTrainingProfile]`
- Produces: `get_baseline_training_profile(name: str) -> BaselineTrainingProfile`
- Produces: `MainTrainingProfile` and `MAIN_TRAINING_PROFILE`

- [ ] **Step 1: Write failing profile tests**

```python
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
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m unittest tests.test_training_strategy.TrainingProfileTests -v`

Expected: import failure for `bstalignment.training_strategy`.

- [ ] **Step 3: Implement immutable profiles**

```python
from __future__ import annotations

from dataclasses import dataclass


TRAINING_STRATEGY_VERSION = "v3-source-profiles-main-adaptive"


@dataclass(frozen=True)
class BaselineTrainingProfile:
    optimizer: str
    loss: str
    lr: float
    weight_decay: float
    scheduler: str
    scheduler_step: str
    max_epochs: int
    early_stop_patience: int
    early_stop_start_epoch: int = 1
    pct_start: float | None = None
    cosine_t_max: int | None = None
    eta_min: float = 0.0
    gradient_clip: float | None = None


BASELINE_TRAINING_PROFILES = {
    "patchtst": BaselineTrainingProfile("adam", "mse", 1e-4, 0.0, "one_cycle", "batch", 100, 20, pct_start=0.3),
    "itransformer": BaselineTrainingProfile("adam", "mse", 1e-4, 0.0, "type1", "epoch", 10, 3),
    "timesnet": BaselineTrainingProfile("adam", "mse", 1e-4, 0.0, "type1", "epoch", 10, 3),
    "dlinear": BaselineTrainingProfile("adam", "mse", 1e-4, 0.0, "type1", "epoch", 10, 3),
    "time_llm": BaselineTrainingProfile("adam", "mse", 1e-3, 0.0, "one_cycle", "batch", 10, 10, pct_start=0.2),
    "timecma": BaselineTrainingProfile(
        "adamw", "mse", 1e-4, 1e-3, "cosine", "epoch", 100, 50,
        early_stop_start_epoch=50, cosine_t_max=50, eta_min=1e-6, gradient_clip=5.0,
    ),
}


def get_baseline_training_profile(name: str) -> BaselineTrainingProfile:
    try:
        return BASELINE_TRAINING_PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"No training profile for official baseline: {name}") from exc


@dataclass(frozen=True)
class MainTrainingProfile:
    max_epochs: int = 80
    core_lr: float = 1e-3
    semantic_lr: float = 3e-4
    weight_decay: float = 1e-4
    lr_warmup_epochs: int = 5
    warmup_start_factor: float = 0.1
    plateau_factor: float = 0.5
    plateau_patience: int = 5
    core_min_lr: float = 1e-5
    semantic_min_lr: float = 3e-6
    align_start_epoch: int = 6
    align_full_epoch: int = 15
    align_weight: float = 1e-3
    early_stop_start_epoch: int = 20
    early_stop_patience: int = 20
    gradient_clip: float = 1.0


MAIN_TRAINING_PROFILE = MainTrainingProfile()
```

- [ ] **Step 4: Run tests and verify GREEN**

Run: `python -m unittest tests.test_training_strategy.TrainingProfileTests -v`

Expected: all three tests pass.

- [ ] **Step 5: Commit**

```bash
git add bstalignment/training_strategy.py tests/__init__.py tests/test_training_strategy.py
git commit -m "Add versioned battery training profiles"
```

### Task 2: Implement Baseline Optimizer and Scheduler Mechanics

**Files:**
- Modify: `bstalignment/training_strategy.py`
- Modify: `tests/test_training_strategy.py`

**Interfaces:**
- Consumes: `BaselineTrainingProfile`
- Produces: `build_baseline_optimizer(model, profile) -> torch.optim.Optimizer`
- Produces: `build_baseline_scheduler(optimizer, profile, steps_per_epoch) -> object | None`
- Produces: `baseline_regression_loss(pred, target, profile) -> torch.Tensor`
- Produces: `step_baseline_batch_scheduler(scheduler, profile) -> None`
- Produces: `step_baseline_epoch_scheduler(scheduler, optimizer, profile, epoch) -> None`

- [ ] **Step 1: Add failing scheduler and loss tests**

```python
import torch

from bstalignment.training_strategy import (
    baseline_regression_loss,
    build_baseline_optimizer,
    build_baseline_scheduler,
    get_baseline_training_profile,
    step_baseline_batch_scheduler,
    step_baseline_epoch_scheduler,
)


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
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m unittest tests.test_training_strategy.BaselineMechanicsTests -v`

Expected: import failures for the five new helper functions.

- [ ] **Step 3: Implement optimizer, scheduler, and loss helpers**

```python
import torch
import torch.nn.functional as F


def build_baseline_optimizer(model, profile):
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if profile.optimizer == "adam":
        return torch.optim.Adam(params, lr=profile.lr, weight_decay=profile.weight_decay)
    if profile.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=profile.lr, weight_decay=profile.weight_decay)
    raise ValueError(f"Unsupported optimizer: {profile.optimizer}")


def build_baseline_scheduler(optimizer, profile, steps_per_epoch):
    if profile.scheduler == "one_cycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=profile.lr,
            epochs=profile.max_epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=float(profile.pct_start),
        )
    if profile.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(profile.cosine_t_max),
            eta_min=profile.eta_min,
        )
    if profile.scheduler == "type1":
        return None
    raise ValueError(f"Unsupported scheduler: {profile.scheduler}")


def baseline_regression_loss(pred, target, profile):
    if profile.loss == "mse":
        return F.mse_loss(pred, target)
    raise ValueError(f"Unsupported baseline loss: {profile.loss}")


def step_baseline_batch_scheduler(scheduler, profile):
    if profile.scheduler_step == "batch" and scheduler is not None:
        scheduler.step()


def step_baseline_epoch_scheduler(scheduler, optimizer, profile, epoch):
    if profile.scheduler_step != "epoch":
        return
    if profile.scheduler == "type1":
        lr = profile.lr * (0.5 ** max(epoch - 1, 0))
        for group in optimizer.param_groups:
            group["lr"] = lr
    elif scheduler is not None:
        scheduler.step()
```

- [ ] **Step 4: Run the strategy test module**

Run: `python -m unittest tests.test_training_strategy -v`

Expected: all profile and mechanics tests pass.

- [ ] **Step 5: Commit**

```bash
git add bstalignment/training_strategy.py tests/test_training_strategy.py
git commit -m "Implement source-consistent baseline schedules"
```

### Task 3: Integrate Profiles into the Official Baseline Trainer

**Files:**
- Modify: `bstalignment/train_battery_official_baselines.py`
- Modify: `tests/test_training_strategy.py`

**Interfaces:**
- Consumes: all baseline helpers from Task 2
- Produces: `resolve_baseline_profile(args) -> BaselineTrainingProfile`
- Produces: strategy-versioned `run_config.json`, `last.pt`, and `best.pt`

- [ ] **Step 1: Write failing trainer integration tests**

```python
from argparse import Namespace

from bstalignment.train_battery_official_baselines import resolve_baseline_profile


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
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m unittest tests.test_training_strategy.BaselineTrainerIntegrationTests -v`

Expected: import failure for `resolve_baseline_profile`.

- [ ] **Step 3: Replace unified defaults and training mechanics**

Implement these exact changes:

```python
from dataclasses import replace

from .training_strategy import (
    TRAINING_STRATEGY_VERSION,
    baseline_regression_loss,
    build_baseline_optimizer,
    build_baseline_scheduler,
    get_baseline_training_profile,
    step_baseline_batch_scheduler,
    step_baseline_epoch_scheduler,
)


def resolve_baseline_profile(args):
    profile = get_baseline_training_profile(args.model)
    updates = {}
    if args.epochs is not None:
        updates["max_epochs"] = args.epochs
    if args.lr is not None:
        updates["lr"] = args.lr
    if args.weight_decay is not None:
        updates["weight_decay"] = args.weight_decay
    if args.early_stop_patience is not None:
        updates["early_stop_patience"] = args.early_stop_patience
    return replace(profile, **updates)
```

Change the four CLI defaults to `None`. Pass `profile`, optional batch scheduler, and
`profile.gradient_clip` into `run_epoch`; compute MSE with
`baseline_regression_loss`; clip only when the profile declares a value; step OneCycle
after each optimizer update. Step type1/cosine after each completed epoch.

Use `for epoch in range(start_epoch, profile.max_epochs + 1)` and stop only when:

```python
if epoch >= profile.early_stop_start_epoch and stale >= profile.early_stop_patience:
    break
```

Persist `training_strategy_version`, `training_profile`, current LR, scheduler state,
optimizer state, epoch, best MSE, stale counter, and validation metrics. Restore all of
them on resume. Append one JSON object per epoch to `epoch_history.jsonl`.

- [ ] **Step 4: Run integration and strategy tests**

Run: `python -m unittest tests.test_training_strategy -v`

Expected: all tests pass.

- [ ] **Step 5: Compile the modified trainer**

Run: `python -m compileall -q bstalignment`

Expected: exit code 0 and no output.

- [ ] **Step 6: Commit**

```bash
git add bstalignment/train_battery_official_baselines.py tests/test_training_strategy.py
git commit -m "Apply native training profiles to baselines"
```

### Task 4: Implement GraphReportTS Parameter Groups and Adaptive Schedule

**Files:**
- Modify: `bstalignment/training_strategy.py`
- Modify: `tests/test_training_strategy.py`

**Interfaces:**
- Consumes: `MAIN_TRAINING_PROFILE`
- Produces: `build_graph_report_optimizer(model, profile) -> torch.optim.AdamW`
- Produces: `GraphReportScheduler`
- Produces: `graph_report_group_lrs(optimizer) -> dict[str, float]`
- Produces: `graph_report_align_weight(epoch, profile) -> float`

- [ ] **Step 1: Add failing main-strategy tests**

```python
from bstalignment.training_strategy import (
    GraphReportScheduler,
    build_graph_report_optimizer,
    graph_report_align_weight,
    graph_report_group_lrs,
)


class TinyGraphReport(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.graph_encoder = torch.nn.Linear(2, 2)
        self.context_norm = torch.nn.LayerNorm(2)
        self.decoder = torch.nn.Embedding(4, 2)
        self.text_encoder = torch.nn.Module()
        self.text_encoder.backbone = torch.nn.Linear(2, 2)
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

    def test_lr_warmup_plateau_and_state_restore(self):
        model = TinyGraphReport()
        optimizer = build_graph_report_optimizer(model, MAIN_TRAINING_PROFILE)
        scheduler = GraphReportScheduler(optimizer, MAIN_TRAINING_PROFILE)
        scheduler.start_epoch(1)
        first = graph_report_group_lrs(optimizer)
        scheduler.start_epoch(5)
        full = graph_report_group_lrs(optimizer)
        self.assertAlmostEqual(first["core"], 1e-4)
        self.assertAlmostEqual(full["core"], 1e-3)
        state = scheduler.state_dict()
        restored = GraphReportScheduler(optimizer, MAIN_TRAINING_PROFILE)
        restored.load_state_dict(state)
        self.assertEqual(restored.state_dict(), state)

    def test_align_weight_is_delayed_and_ramped(self):
        self.assertEqual(graph_report_align_weight(5, MAIN_TRAINING_PROFILE), 0.0)
        self.assertAlmostEqual(graph_report_align_weight(6, MAIN_TRAINING_PROFILE), 1e-4)
        self.assertAlmostEqual(graph_report_align_weight(15, MAIN_TRAINING_PROFILE), 1e-3)
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m unittest tests.test_training_strategy.MainStrategyTests -v`

Expected: import failures for the four main-strategy helpers.

- [ ] **Step 3: Implement grouping and scheduler**

Classify names beginning with `text_encoder.proj`, `semantic_fusion`, or `fusion` as
semantic. Exclude names containing `norm`, ending in `.bias`, or containing `embed`
from weight decay. Reject duplicate or uncovered trainable parameter IDs.

Create four AdamW groups (`core_decay`, `core_no_decay`, `semantic_decay`,
`semantic_no_decay`) with a `role` field. Initialize each LR at 10% of its target.

Implement `GraphReportScheduler` with:

```python
def start_epoch(self, epoch):
    if epoch <= self.profile.lr_warmup_epochs:
        progress = (epoch - 1) / max(self.profile.lr_warmup_epochs - 1, 1)
        factor = self.profile.warmup_start_factor + (1.0 - self.profile.warmup_start_factor) * progress
        self._set_role_lrs(factor)

def step_validation(self, epoch, val_mse):
    if epoch >= self.profile.lr_warmup_epochs:
        self.plateau.step(val_mse)
```

Use `ReduceLROnPlateau(mode="min", factor=0.5, patience=5,
min_lr=[group-specific minimum for each optimizer group])`. Include complete
`state_dict` and `load_state_dict` methods.

Implement align weight as 0 through epoch 5 and a ten-point linear sequence from
0.0001 at epoch 6 to 0.001 at epoch 15.

- [ ] **Step 4: Run all strategy tests**

Run: `python -m unittest tests.test_training_strategy -v`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add bstalignment/training_strategy.py tests/test_training_strategy.py
git commit -m "Add adaptive GraphReportTS training strategy"
```

### Task 5: Integrate the Main Strategy and Resume State

**Files:**
- Modify: `bstalignment/train_graph_report.py`
- Modify: `bstalignment/run_ablation_suite.py`
- Modify: `tests/test_training_strategy.py`

**Interfaces:**
- Consumes: Task 4 main-strategy helpers
- Produces: strategy-versioned main and ablation checkpoints/history

- [ ] **Step 1: Add failing schedule-state and early-stop tests**

```python
from bstalignment.training_strategy import should_stop_graph_report


class MainTrainerPolicyTests(unittest.TestCase):
    def test_early_stop_counter_is_inactive_before_epoch_20(self):
        self.assertFalse(should_stop_graph_report(epoch=19, stale=100, profile=MAIN_TRAINING_PROFILE))
        self.assertFalse(should_stop_graph_report(epoch=38, stale=19, profile=MAIN_TRAINING_PROFILE))
        self.assertTrue(should_stop_graph_report(epoch=39, stale=20, profile=MAIN_TRAINING_PROFILE))
```

- [ ] **Step 2: Run the policy test and verify RED**

Run: `python -m unittest tests.test_training_strategy.MainTrainerPolicyTests -v`

Expected: import failure for `should_stop_graph_report`.

- [ ] **Step 3: Implement the policy helper and trainer integration**

```python
def should_stop_graph_report(epoch, stale, profile):
    return epoch >= profile.early_stop_start_epoch and stale >= profile.early_stop_patience
```

In `train_graph_report.py`:

- Build the optimizer with `build_graph_report_optimizer` after LazyLinear initialization.
- Assert every DistilBERT backbone parameter has `requires_grad=False` in the formal frozen mode.
- Call `scheduler.start_epoch(epoch)` before the training loader.
- Compute align weight with `graph_report_align_weight`; force it to zero for the existing no-align variants.
- Call `scheduler.step_validation(epoch, val["mse"])` after validation.
- Reset stale to zero at epoch 20; do not count epochs 1-19.
- Save the true best checkpoint on every strict MSE reduction.
- Save and restore scheduler state, optimizer state, strategy version, stale, epoch, and both group LRs.
- Add `core_lr`, `semantic_lr`, and `training_strategy_version` to each history row and console line.
- Retain SmoothL1, validation-MSE checkpoint selection, gradient clip 1.0, gate logging, and final one-time test evaluation.

In `run_ablation_suite.py`, remove independent early-stop defaults and pass the same main
profile arguments to every variant. Keep variant flags as the only strategy differences.

- [ ] **Step 4: Run tests and compile**

Run: `python -m unittest tests.test_training_strategy -v`

Expected: all tests pass.

Run: `python -m compileall -q bstalignment`

Expected: exit code 0.

- [ ] **Step 5: Commit**

```bash
git add bstalignment/train_graph_report.py bstalignment/run_ablation_suite.py bstalignment/training_strategy.py tests/test_training_strategy.py
git commit -m "Apply adaptive training to main and ablations"
```

### Task 6: Reorder and Version the Formal Pipeline

**Files:**
- Create: `scripts/run_battery_v3_training_strategy_pipeline.sh`
- Delete: `scripts/run_battery_v2_full_hf_pipeline.sh`
- Modify: `scripts/run_battery_main_full_hf.sh`
- Modify: `scripts/run_battery_official_baselines.sh`
- Modify: `scripts/run_battery_ablations_full_hf.sh`
- Modify: `tests/test_training_strategy.py`

**Interfaces:**
- Consumes: `TRAINING_STRATEGY_VERSION = "v3-source-profiles-main-adaptive"`
- Produces: main -> baselines -> ablations orchestration under the v3 output root

- [ ] **Step 1: Add failing script-contract tests**

```python
from pathlib import Path


class PipelineScriptTests(unittest.TestCase):
    def test_formal_pipeline_is_main_first_and_uses_v3_root(self):
        text = Path("scripts/run_battery_v3_training_strategy_pipeline.sh").read_text(encoding="utf-8")
        self.assertIn("runs/full_hf_v3_training_strategy_nosoh", text)
        main = text.index("run_battery_main_full_hf.sh")
        baselines = text.index("run_battery_official_baselines.sh")
        ablations = text.index("run_battery_ablations_full_hf.sh")
        self.assertLess(main, baselines)
        self.assertLess(baselines, ablations)

    def test_baseline_script_does_not_force_one_epoch_budget(self):
        text = Path("scripts/run_battery_official_baselines.sh").read_text(encoding="utf-8")
        self.assertNotIn("--epochs \"$EPOCHS\"", text)
        self.assertNotIn("--early_stop_patience \"$EARLY_STOP_PATIENCE\"", text)
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m unittest tests.test_training_strategy.PipelineScriptTests -v`

Expected: assertions fail because the current pipeline is baseline-first and uses v2.

- [ ] **Step 3: Update orchestration scripts**

Set the top-level default output root to
`runs/full_hf_v3_training_strategy_nosoh`. Invoke scripts in this exact order:

```bash
bash scripts/run_battery_main_full_hf.sh "$ROOT" 2>&1 | tee "$OUT_ROOT/logs/pipeline_main.out"
bash scripts/run_battery_official_baselines.sh "$ROOT" 2>&1 | tee "$OUT_ROOT/logs/pipeline_baselines.out"
bash scripts/run_battery_ablations_full_hf.sh "$ROOT" 2>&1 | tee "$OUT_ROOT/logs/pipeline_ablation.out"
```

Do not export one shared baseline epoch, LR, or patience. Let each Python profile own
those values. Main and ablation scripts pass the approved v3 settings or rely on the
immutable main profile without overriding it.

Add a shell helper that skips a result only when both JSON files exist and
`run_config.json` contains `"training_strategy_version":
"v3-source-profiles-main-adaptive"`. A mismatched version must be retrained.

Keep `FORCE_RETRAIN=1` for a new formal launch and `FORCE_RETRAIN=0` for safe resume.

- [ ] **Step 4: Normalize shell line endings and run tests**

Run: `git add --renormalize scripts/*.sh`

Run: `python -m unittest tests.test_training_strategy -v`

Expected: all tests pass.

Run: `bash -n scripts/run_battery_v3_training_strategy_pipeline.sh scripts/run_battery_main_full_hf.sh scripts/run_battery_official_baselines.sh scripts/run_battery_ablations_full_hf.sh`

Expected: exit code 0 on Linux/WSL or the remote server.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_battery_v3_training_strategy_pipeline.sh scripts/run_battery_v2_full_hf_pipeline.sh scripts/run_battery_main_full_hf.sh scripts/run_battery_official_baselines.sh scripts/run_battery_ablations_full_hf.sh tests/test_training_strategy.py
git commit -m "Run main-first versioned training pipeline"
```

### Task 7: Update Public Documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/work_report.md`
- Modify: `docs/cloud_training_workflow.md`

**Interfaces:**
- Consumes: approved design and implemented v3 commands
- Produces: user-facing training protocol and operational instructions

- [ ] **Step 1: Add documentation contract checks**

Extend `PipelineScriptTests` with assertions that README and work report contain:

```python
for path in [Path("README.md"), Path("docs/work_report.md")]:
    text = path.read_text(encoding="utf-8")
    self.assertIn("full_hf_v3_training_strategy_nosoh", text)
    self.assertIn("main -> baselines -> ablations", text)
    self.assertIn("DistilBERT", text)
```

- [ ] **Step 2: Run the documentation checks and verify RED**

Run: `python -m unittest tests.test_training_strategy.PipelineScriptTests -v`

Expected: assertions fail because v3 is not documented.

- [ ] **Step 3: Document the implemented protocol**

README must describe the formal command, input feature schema, main-first order, frozen
DistilBERT, main LR/align schedule, and source-native baseline profiles.

The work report must record why v2 results are legacy-only, the five late best epochs,
the missing scheduler diagnosis, the approved v3 strategy, and future work on full-patch
cross-cycle attention.

The cloud workflow must use the v3 output/log paths and show status commands for main,
baseline, and ablation processes.

- [ ] **Step 4: Run tests and Markdown checks**

Run: `python -m unittest tests.test_training_strategy -v`

Expected: all tests pass.

Run: `git diff --check`

Expected: no whitespace errors.

- [ ] **Step 5: Commit**

```bash
git add README.md docs/work_report.md docs/cloud_training_workflow.md tests/test_training_strategy.py
git commit -m "Document v3 training protocol"
```

### Task 8: Verify, Publish, Deploy, and Restart

**Files:**
- Verify only; no new source files

**Interfaces:**
- Consumes: all previous tasks
- Produces: GitHub commit and a running remote v3 main-model process

- [ ] **Step 1: Run the complete local verification set**

Run: `python -m unittest tests.test_training_strategy -v`

Expected: all tests pass.

Run: `python -m compileall -q bstalignment`

Expected: exit code 0.

Run: `git diff --check && git status --short`

Expected: no uncommitted implementation changes.

- [ ] **Step 2: Push the implementation commits**

Run: `git push origin main`

Expected: remote `main` advances to the local HEAD.

- [ ] **Step 3: Synchronize the remote repository**

Run:

```bash
ssh connect.westc.seetacloud.com "cd /root/autodl-tmp/GraphReportTS && git pull --ff-only origin main"
```

Expected: remote HEAD matches local HEAD; ignored data, external repositories, HF weights,
and v2 runs remain present.

- [ ] **Step 4: Run remote strategy verification**

Run:

```bash
ssh connect.westc.seetacloud.com "cd /root/autodl-tmp/GraphReportTS && source /root/miniconda3/etc/profile.d/conda.sh && conda activate graphreport && python -m unittest tests.test_training_strategy -v && python -m compileall -q bstalignment"
```

Expected: tests pass and compilation succeeds.

- [ ] **Step 5: Start the v3 main-first pipeline**

Run:

```bash
ssh connect.westc.seetacloud.com "cd /root/autodl-tmp/GraphReportTS; mkdir -p runs/full_hf_v3_training_strategy_nosoh/logs; nohup bash -lc 'source /root/miniconda3/etc/profile.d/conda.sh && conda activate graphreport && FORCE_RETRAIN=1 bash scripts/run_battery_v3_training_strategy_pipeline.sh /root/autodl-tmp/GraphReportTS' > runs/full_hf_v3_training_strategy_nosoh/logs/pipeline.nohup 2>&1 < /dev/null & echo \$! > runs/full_hf_v3_training_strategy_nosoh/pipeline.pid"
```

Expected: the first Python training process is GraphReportTS on MIT, not a baseline.

- [ ] **Step 6: Verify first-process invariants**

Check process command, GPU use, v3 run config, first history row, and logs. Confirm:

- DistilBERT backbone is frozen.
- Core LR begins at 1e-4 and semantic LR begins at 3e-5.
- Align weight is 0 through epoch 5.
- Output root is v3 and v2 files are untouched.
- The next pipeline stages remain baselines then ablations.

- [ ] **Step 7: Commit any verification-only documentation correction**

Only if an exact command or path in documentation was wrong, patch that documentation,
rerun `git diff --check`, commit with `git commit -m "Fix v3 training instructions"`, push,
and fast-forward the remote checkout. Do not modify model behavior during this step.
