# Task 6 Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix every Task 6 review finding with source-format timestamps, scoped text loading, exact source identity, runtime provenance, corrected stopping, and executable training helpers.

**Architecture:** Task 3 samples remain canonical; pure trainer helpers collate official `timeF` markers and forward batches to lazy adapters. Adapter construction scopes all external mutation and returns complete checkout/text runtime provenance. Training helpers remain small source-mechanics primitives with explicit tests.

**Tech Stack:** Python 3.13, unittest, PyTorch, NumPy, pandas, temporary Git repositories, fake official source trees.

## Global Constraints

- Preserve all battery adapters and unrelated edits.
- Use exactly six datasets, M2M channels, input 36, and horizons 96/192/336/720.
- Use Task 3 loader, splits, timestamps, and train-only scaler.
- Do not train, initialize CUDA, load weights, generate embeddings, or use network access.
- Write and verify RED tests before each production fix.
- Append exact RED/GREEN commands and outcomes to `.superpowers/sdd/task-6-report.md`.

---

### Task 1: Source-format time markers and batch forwarding

**Files:**
- Modify: `tests/test_general_baselines.py`
- Modify: `bstalignment/baseline_adapters.py`
- Modify: `bstalignment/train_general_baselines.py`

**Interfaces:**
- Produces: `source_time_markers(dataset, timestamps) -> torch.Tensor`
- Produces: `collate_general_baseline_batch(samples) -> dict`
- Produces: `forward_general_baseline_batch(adapter, batch, prompt_embeddings=None) -> torch.Tensor`

- [ ] Write fake iTransformer/TimesNet source classes that assert non-null encoder markers and capture decoder markers. Add exact 4-column hourly and 5-column minute cadence assertions plus a Task 3 sample-to-collator-to-forward test.
- [ ] Run the marker tests and verify failure because adapters pass `None` and collation helpers are absent.
- [ ] Implement the audited THUML normalized time features, baseline collator, forward helper, adapter marker validation, and dataset-specific config frequency.
- [ ] Re-run the marker tests and verify all pass.

### Task 2: Scoped Time-LLM local loading and runtime provenance

**Files:**
- Modify: `tests/test_general_baselines.py`
- Modify: `bstalignment/baseline_adapters.py`
- Modify: `bstalignment/train_general_baselines.py`

**Interfaces:**
- Produces: `_scoped_timellm_local_sources(module, args, formal)` context manager
- Produces: adapter `runtime_provenance` containing text paths/revisions/precision/dtype

- [ ] Add fake Llama config/model/tokenizer descriptors and two sequential builds with distinct paths/revisions. Assert exact original descriptors are restored after each build and calls never cross-contaminate. Add result-record rejection tests for missing/placeholder Time-LLM runtime provenance and acceptance tests for actual values.
- [ ] Run focused tests and verify permanent mutation/provenance failures.
- [ ] Replace permanent monkey-patching with descriptor-preserving `try/finally`, validate formal local runtime fields, attach observed runtime metadata, and require it in Time-LLM result records.
- [ ] Re-run focused tests and verify all pass.

### Task 3: Full checkout identity and tracked-clean validation

**Files:**
- Modify: `tests/test_general_baselines.py`
- Modify: `bstalignment/baseline_adapters.py`

**Interfaces:**
- Produces: `SourceCheckoutProvenance`
- Produces: `validate_source_checkout(external_root, source) -> SourceCheckoutProvenance`

- [ ] Create a temporary committed Git repository and test short-manifest-revision resolution to full SHA, wrong revision rejection, tracked-dirty rejection, and untracked-file acceptance.
- [ ] Run focused tests and verify current abbreviated validator fails the new contract.
- [ ] Resolve full `HEAD` and `<pinned>^{commit}`, compare exact SHAs, check tracked porcelain status, return full provenance, and attach it to adapters.
- [ ] Re-run source validation tests and verify all pass.

### Task 4: Delayed stopping and executable mechanics

**Files:**
- Modify: `tests/test_general_baselines.py`
- Modify: `bstalignment/train_general_baselines.py`

**Interfaces:**
- Produces: `clip_general_gradients(model, profile)`
- Produces: `step_general_optimizer(model, optimizer, profile, scheduler=None)`

- [ ] Replace the isolated delayed-stop assertion with a chained pre-gate failure sequence and add gradient clipping/optimizer/batch-scheduler ordering tests.
- [ ] Run focused tests and verify stale reset and missing helper failures.
- [ ] Accumulate stale failures before the gate while gating only `should_stop`; add clipping and optimizer-step helpers using profile mechanics.
- [ ] Re-run training-contract tests and verify all pass.

### Task 5: Import-safety, documentation, full verification, and report

**Files:**
- Modify: `tests/test_general_baselines.py`
- Modify: `docs/superpowers/specs/2026-07-12-source-consistent-general-baselines-design.md`
- Modify: `docs/general_forecasting_source_audit.md`
- Modify: `.superpowers/sdd/task-6-report.md`

- [ ] Add a subprocess import test proving `train_general_baselines` imports NumPy/PyTorch but not Transformers, official `models`/`model`/`layers`, weights, or initialized CUDA.
- [ ] Correct design/audit/report wording to state the precise executable-core import boundary and append each RED/GREEN result.
- [ ] Run `C:\Python313\python.exe -m unittest tests.test_general_baselines -v`, then `C:\Python313\python.exe -m unittest discover -s tests -v`.
- [ ] Run `compileall`, `git diff --check`, inspect the full diff, verify a clean battery regression, and commit with `Fix Task 6 source adapter contracts`.
