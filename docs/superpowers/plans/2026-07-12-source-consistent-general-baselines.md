# Source-Consistent General Baselines Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build source-audited, lazy official adapters and training contracts for six formal general-forecasting baselines.

**Architecture:** An immutable profile resolver separates pinned source defaults from formal protocol overrides. A lazy import-isolated adapter wraps official classes behind one M2M output contract, while focused prompt/cache utilities preserve TimeCMA and Time-LLM behavior without loading text models during unit tests. The trainer module consumes Task 3 data and resolved mechanics but does not execute training in this task.

**Tech Stack:** Python 3, PyTorch, unittest, temporary fake Python source trees, JSON provenance, Git/SSH source audit.

## Global Constraints

- Formal datasets are exactly ETTm1, ETTm2, ETTh1, ETTh2, ECL, and Weather; Traffic is excluded.
- Formal history is 36 and horizons are exactly 96, 192, 336, and 720.
- Forecasting is M2M: input `[B,36,C]`, output `[B,H,C]`.
- Use Task 3 splits and train-only scaler; checkpoint selection is validation-MSE-only.
- Validate each module/class path and training/profile value against its pinned server commit before encoding it.
- Do not train, use CUDA, generate real embeddings, download weights, or alter/copy server repositories.
- Preserve every battery adapter API and unrelated edit.

---

### Task 1: Lock profile contracts with RED tests

**Files:**
- Create: `tests/test_general_baselines.py`
- Create: `bstalignment/general_baseline_profiles.py`

**Interfaces:**
- Produces: `SourceIdentity`, `TrainingMechanics`, `GeneralBaselineProfile`
- Produces: `resolve_general_profile(name: str, dataset: str, pred_len: int) -> GeneralBaselineProfile`

- [ ] **Step 1: Write failing profile tests**

Add table-driven assertions covering all 144 combinations. Assert source URL,
commit, source module/class, prompt policy, `seq_len=36`, horizon, and the
directly audited dataset/horizon fields. Include sentinel cases that distinguish
source mechanics: PatchTST ETTm1 OneCycle/pct_start 0.4 versus ETTh1 type3 and
patience 100; TimesNet ETTm2-H192 one epoch; DLinear ETTm2-H336 lr 0.01;
TimeCMA ETTm1-H96 999 epochs versus Weather-H96 20; Time-LLM ETTh1-H336 cosine
versus ETTm1-H96 OneCycle.

- [ ] **Step 2: Run RED**

Run: `python -m unittest tests.test_general_baselines.GeneralBaselineProfileTests -v`

Expected: FAIL because `bstalignment.general_baseline_profiles` does not exist.

- [ ] **Step 3: Implement immutable profiles**

Define frozen dataclasses with architecture stored as immutable key/value tuples.
Normalize aliases (`time_llm`, `Time-LLM`) without changing canonical display
names. Encode only audited source values; reject unknown datasets/horizons/models.
Record protocol overrides and source evidence strings separately.

- [ ] **Step 4: Run GREEN**

Run: `python -m unittest tests.test_general_baselines.GeneralBaselineProfileTests -v`

Expected: PASS for all source/profile cases.

### Task 2: Build lazy official adapters with fake source trees

**Files:**
- Modify: `tests/test_general_baselines.py`
- Modify: `bstalignment/baseline_adapters.py`

**Interfaces:**
- Produces: `build_general_baseline(name, dataset_meta, args) -> torch.nn.Module`
- Produces: adapter metadata attributes `profile`, `source_identity`, and `prompt_policy`

- [ ] **Step 1: Write failing fake-source adapter tests**

Create temporary package layouts matching each audited official path. Fake
classes capture their constructor config and return `[B,H,C]`. Assert ordinary
module import does not import `transformers` or source packages; all non-text
adapters construct the exact class, receive `enc_in=c_out=C`, and return all
channels. Assert sequential builds from different roots do not reuse stale
`models`/`layers` modules. TimeCMA accepts official `[B,768,C]` embeddings;
Time-LLM freezes its fake `llm_model` parameters.

- [ ] **Step 2: Run RED**

Run: `python -m unittest tests.test_general_baselines.GeneralBaselineAdapterTests -v`

Expected: FAIL because `build_general_baseline` is missing.

- [ ] **Step 3: Implement minimal lazy wrappers**

Add a temporary external import scope that snapshots/restores `sys.path` and
conflicting source module names. Validate the checkout commit unless
`verify_source_commit=False` is explicitly supplied for fake tests. Construct a
source-shaped `SimpleNamespace` from the resolved profile plus dataset channel
count. Dispatch source forward signatures without slicing channels and validate
the exact `[B,H,C]` result. Keep `list_baseline_specs`, clone setup, and battery
callers unchanged.

- [ ] **Step 4: Run GREEN and battery regression tests**

Run: `python -m unittest tests.test_general_baselines.GeneralBaselineAdapterTests tests.test_training_strategy -v`

Expected: PASS with existing battery profile behavior unchanged.

### Task 3: Add TimeCMA and Time-LLM prompt contracts

**Files:**
- Modify: `tests/test_general_baselines.py`
- Modify: `bstalignment/baseline_adapters.py`

**Interfaces:**
- Produces: `TimeCMACacheProvenance`
- Produces: `TimeCMAPromptCache.get_or_create(...) -> torch.Tensor`
- Produces: `build_timecma_prompt(...) -> str`
- Produces: `build_time_llm_prompts(history, description, pred_len) -> tuple[str, ...]`

- [ ] **Step 1: Write failing prompt/cache tests**

Use source examples to assert exact cadence, timestamp, integer rendering, and
summed-difference trend for TimeCMA. Use a fake frozen encoder to prove cache
reuse and final-token shape. Assert cache paths vary by dataset/split/input
length/commit/scaler/model/tokenizer/revision/precision and by absolute sample
and variable index. Reject a 37th value, a mismatched provenance record, or an
observed end at/after the forecast origin. For Time-LLM, compare exact prompt
text for min/max/median, tied/downward trend, and `torch.topk` FFT lags; assert
the official ETT/ECL/Weather descriptions and frozen backbone.

- [ ] **Step 2: Run RED**

Run: `python -m unittest tests.test_general_baselines.GeneralPromptContractTests -v`

Expected: FAIL because prompt/cache APIs do not exist.

- [ ] **Step 3: Implement exact pure prompt functions and disk cache**

Keep Transformers imports inside an optional encoder constructor. Store tensor
and canonical JSON metadata beneath a provenance digest. Validate metadata and
history boundaries before cache reads. Build Time-LLM prompts from per-variable
normalized history with the exact official wording and lag computation. Require
local backbone paths/revisions for formal Time-LLM construction and force local
loading behavior.

- [ ] **Step 4: Run GREEN**

Run: `python -m unittest tests.test_general_baselines.GeneralPromptContractTests -v`

Expected: PASS without network, CUDA, or real embedding generation.

### Task 4: Add source-native orchestration and result schema

**Files:**
- Modify: `tests/test_general_baselines.py`
- Create: `bstalignment/train_general_baselines.py`

**Interfaces:**
- Produces: `build_general_optimizer(model, profile)`
- Produces: `build_general_scheduler(optimizer, profile, steps_per_epoch)`
- Produces: scheduler stepping helpers and validation-only stale tracking
- Produces: `general_result_record(...) -> dict[str, object]`

- [ ] **Step 1: Write failing orchestration tests**

Assert optimizer class, source lr/weight decay, OneCycle batch stepping, cosine
epoch stepping, type1/type3 formulas, TimeCMA gradient clipping metadata and
delayed early stopping, and validation-only stale/checkpoint decisions. Assert
the result record contains dataset/model/horizon/seed, standardized MSE/MAE,
source identity, architecture, training mechanics, protocol overrides, scaler
checksum, prompt provenance, and validation-best epoch/MSE.

- [ ] **Step 2: Run RED**

Run: `python -m unittest tests.test_general_baselines.GeneralTrainingContractTests -v`

Expected: FAIL because `train_general_baselines` does not exist.

- [ ] **Step 3: Implement import-safe orchestration helpers**

Reuse `GeneralForecastGraphDataset` and its fitted scaler rather than defining a
new split. Build only trainable parameters into Adam/AdamW. Implement source
scheduler semantics and standardized MSE/MAE. Keep execution under `main()` and
require explicit local external/model paths so importing the module is safe.

- [ ] **Step 4: Run GREEN**

Run: `python -m unittest tests.test_general_baselines.GeneralTrainingContractTests -v`

Expected: PASS on CPU.

### Task 5: Audit documentation, integration skip, and final verification

**Files:**
- Modify: `tests/test_general_baselines.py`
- Modify: `docs/general_forecasting_source_audit.md`
- Create: `.superpowers/sdd/task-6-report.md`

- [ ] **Step 1: Add optional local-repository integration checks**

Skip unless all six local roots exist. When available, verify commit and class
construction only on CPU with no Time-LLM/TimeCMA text-weight loading.

- [ ] **Step 2: Correct the audit from direct pinned evidence**

Replace earlier generalized mechanics with dataset/horizon-sensitive facts,
including the directly observed PatchTST, TimeCMA, TimesNet, DLinear, and
Time-LLM exceptions. Cite server repo commit, source path, and exact lines.

- [ ] **Step 3: Run focused and full suites**

Run: `python -m unittest tests.test_general_baselines -v`

Run: `python -m unittest discover -s tests -v`

Expected: all tests PASS; only the documented real-repository integration test
may SKIP.

- [ ] **Step 4: Self-review and write report**

Run `git diff --check`, inspect every changed file, verify no battery behavior
changed, scan for heavyweight top-level imports and future leakage, and record
RED/GREEN output, source evidence, tests, files, commits, and concerns in the
Task 6 report.

- [ ] **Step 5: Commit**

Run:

```bash
git add bstalignment/general_baseline_profiles.py bstalignment/baseline_adapters.py \
  bstalignment/train_general_baselines.py tests/test_general_baselines.py \
  docs/general_forecasting_source_audit.md .superpowers/sdd/task-6-report.md \
  docs/superpowers/plans/2026-07-12-source-consistent-general-baselines.md
git commit -m "Add official general forecasting baselines"
```
