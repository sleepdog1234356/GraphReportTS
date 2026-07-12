# Task 6 Final Frequency Correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make pinned iTransformer and TimesNet formal runs retain source-default `freq='h'` and four-column `timeF` markers for all six datasets.

**Architecture:** Keep factual dataset cadence in Task 3 schemas and text prompts, but treat marker generation as a pinned iTransformer/TimesNet source contract. Remove the adapter's dataset-derived `t` overwrite, make the shared executable marker helper follow the pinned `h` profile, and document that this is source-native behavior.

**Tech Stack:** Python 3, unittest, PyTorch, NumPy, fake official source modules, Markdown source audit.

## Global Constraints

- Preserve pinned iTransformer `c2426e6` and TimesNet `4e938a1` behavior.
- Do not change factual cadence in TimeCMA or Time-LLM prompts.
- NumPy/PyTorch eager imports are permitted; official repositories, Transformers, weights, and CUDA initialization remain lazy.
- Do not train, initialize CUDA, access weights, or modify the read-only server repositories.

---

### Task 1: Add source-default frequency RED tests

**Files:**
- Modify: `tests/test_general_baselines.py`

**Interfaces:**
- Consumes: `resolve_general_profile`, `build_general_baseline`, `collate_general_baseline_batch`, `forward_general_baseline_batch`
- Produces: regression coverage requiring `freq='h'` and four marker columns for ETTm1, ETTm2, and Weather

- [ ] **Step 1: Add profile assertions**

Add a test iterating `iTransformer` and `TimesNet` over ETTm1, ETTm2, and Weather and asserting `profile.architecture['freq'] == 'h'`, the source evidence cites the default and absent script override, and `freq` is not a project protocol override.

- [ ] **Step 2: Add executable-path assertions**

Build both fake official models for each minute-cadence dataset, collate Task 3 timestamps, assert encoder and decoder markers have four columns, forward them, and assert the fake official model receives those exact tensors while its config remains `h`.

- [ ] **Step 3: Run RED tests**

Run:

```powershell
C:\Python313\python.exe -m unittest tests.test_general_baselines.GeneralBaselineProfileTests.test_itransformer_and_timesnet_preserve_pinned_hourly_timef_default_for_all_datasets tests.test_general_baselines.GeneralTrainingContractTests.test_minute_cadence_datasets_execute_with_pinned_four_column_markers -v
```

Expected: failures showing five marker columns and/or the unsupported `t` runtime overwrite.

### Task 2: Implement the minimal source-native correction

**Files:**
- Modify: `bstalignment/baseline_adapters.py`
- Modify: `bstalignment/train_general_baselines.py`
- Modify: `bstalignment/general_baseline_profiles.py`
- Test: `tests/test_general_baselines.py`

**Interfaces:**
- Consumes: pinned profile `architecture['freq']`
- Produces: four-column `source_time_markers(dataset, timestamps)` for the iTransformer/TimesNet shared path and unmodified `configs.freq == 'h'`

- [ ] **Step 1: Remove the unsupported adapter override**

Delete the assignment that derives `freq='t'` from ETTm1, ETTm2, or Weather after applying profile architecture.

- [ ] **Step 2: Match pinned hourly marker generation**

Remove the dataset-specific minute feature from `source_time_markers`; retain hour, day-of-week, day-of-month, and day-of-year in official order.

- [ ] **Step 3: Strengthen profile evidence**

Record the relevant `run.py` default, absent formal-script override, loader propagation, and `utils/timefeatures.py` evidence in `source_evidence`. Do not add `freq` to `protocol_override_items`.

- [ ] **Step 4: Run focused GREEN tests**

Run the two RED tests, then:

```powershell
C:\Python313\python.exe -m unittest tests.test_general_baselines -v
```

Expected: all focused tests pass, with only the optional real-repository integration skipped.

### Task 3: Correct audits and verify the repository

**Files:**
- Modify: `docs/general_forecasting_source_audit.md`
- Modify: `docs/superpowers/specs/2026-07-13-task6-review-fixes-design.md`
- Modify: `.superpowers/sdd/task-6-report.md`

**Interfaces:**
- Consumes: RED/GREEN command output and pinned server line evidence
- Produces: corrected source-native audit and final Task 6 evidence

- [ ] **Step 1: Correct prior five-column claims**

State that iTransformer/TimesNet formal scripts preserve `freq='h'` for all six datasets and therefore produce four marker columns. Explicitly distinguish this from factual cadence retained by schemas and TimeCMA/Time-LLM prompts.

- [ ] **Step 2: Clarify import scope**

State only that NumPy/PyTorch are intentionally eager in the executable trainer while official repos, Transformers, weights, and CUDA initialization remain lazy.

- [ ] **Step 3: Run full verification**

Run:

```powershell
C:\Python313\python.exe -m unittest discover -s tests -v
C:\Python313\python.exe -m compileall -q bstalignment tests
git diff --check
```

Expected: full suite passes with existing optional skips; compile and diff checks succeed.

- [ ] **Step 4: Self-review and commit**

Review the diff for battery/unrelated changes, stage only Task 6 files, and commit the implementation and report evidence.
