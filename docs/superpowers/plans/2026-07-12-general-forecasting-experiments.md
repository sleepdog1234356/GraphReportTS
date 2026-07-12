# General Time-Series Forecasting Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate a leakage-free, source-consistent experiment pipeline for General-GraphReportTS and the same six battery-study baselines on ETTm1, ETTm2, ETTh1, ETTh2, ECL, and Weather with input length 36 and prediction lengths 96, 192, 336, and 720.

**Architecture:** One canonical data layer owns schema validation, official split boundaries, timestamp features, and train-only scaling. All models consume the same variables and split manifests; source-model adapters preserve each baseline architecture, prompt, optimizer, scheduler, and early-stopping semantics. General-GraphReportTS uses a bounded-memory variable path so ECL (321 variables) remains tractable without dropping input variables.

**Tech Stack:** Python 3.11, PyTorch, pandas, NumPy, scikit-learn, Hugging Face Transformers, pytest/unittest, Bash, official PatchTST/iTransformer/TimeCMA/TimesNet/DLinear/Time-LLM repositories.

## Global Constraints

- Datasets: `ETTm1`, `ETTm2`, `ETTh1`, `ETTh2`, `ECL`, `Weather`.
- Models: `GraphReportTS`, `PatchTST`, `iTransformer`, `TimeCMA`, `TimesNet`, `DLinear`, `Time-LLM`.
- Every model uses exactly 36 observed time steps and predicts exactly one of 96, 192, 336, or 720 future steps.
- Task mode is multivariate-to-multivariate (`features=M`); every model sees every numeric variable in the canonical dataset.
- No PCA, feature selection, target-only conversion, interpolation across split boundaries, or future-derived prompt fields.
- ETT uses official fixed 12/4/4-month train/validation/test boundaries; ECL and Weather use chronological 70/10/20 boundaries with a 36-step history overlap at validation/test starts.
- Scaling statistics are fitted on the train interval only, independently per variable, and reused unchanged for validation and test.
- Main-model prompts are domain-general and computed only from the 36-step observed window. TimeCMA and Time-LLM use their official source prompt construction. Non-text baselines receive no prompt.
- Formal metrics are MSE and MAE in standardized space, aggregated over all forecast steps and variables. Optional inverse-scale metrics must be reported separately.
- Seed policy: smoke tests use seed 42; formal tables use seeds 2021, 2022, and 2023 and report mean ± standard deviation.
- Published baseline table values are not directly comparable because the required lookback is 36 rather than the usual 96; comparisons are valid only among runs under this shared protocol.
- Existing battery code paths and the active battery training run must remain unchanged.

---

## Research Findings and Frozen Decisions

### Dataset contract

| Dataset | Frequency | Variables | Canonical split | Timestamp features |
|---|---:|---:|---|---|
| ETTh1, ETTh2 | 1 hour | 7 | 12/4/4 months = 8640/2880/2880 rows | month, day, weekday, hour |
| ETTm1, ETTm2 | 15 minutes | 7 | 12/4/4 months = 34560/11520/11520 rows | month, day, weekday, hour, quarter-hour |
| ECL | 1 hour | 321 | chronological 70/10/20 | month, day, weekday, hour |
| Weather | 10 minutes | 21 | chronological 70/10/20 | month, day, weekday, hour, minute bucket |

The exact variable counts are validated from downloaded CSV headers rather than silently assumed. `date` is metadata, not a regression target. Numeric values must be finite after parsing; missing values are handled by forward fill using past values within each series, followed by train-set medians for leading gaps. The imputation parameters and missing counts are recorded in the manifest.

For a raw boundary `[a, b)`, train windows begin at `a`; validation and test arrays begin at `a - 36` so their first target begins exactly at the official boundary. A sample is admitted only if its full 36-step input and full prediction horizon lie in the intended interval.

### Prompt policy

General-GraphReportTS uses one deterministic, dataset-agnostic template:

```text
Task: multivariate time-series forecasting.
Observation: 36 past steps sampled every {frequency}; {num_variables} variables are observed.
Window summary: aggregate mean={...}, standard deviation={...}, mean absolute change={...},
trend balance={up_count} increasing/{down_count} decreasing/{flat_count} approximately flat.
Variable summaries: {bounded deterministic summaries of variable name, last value, trend, volatility}.
Instruction: predict all {num_variables} variables for the next {pred_len} steps.
Use only the observed window and do not assume future measurements.
```

Values are computed after train-fitted standardization. To bound token length, include aggregate statistics over all variables plus at most 12 variable summaries selected deterministically: first include named variables for datasets with at most 12 variables; otherwise choose six largest and six smallest absolute standardized trends, breaking ties by canonical column index. The prompt must not contain target-window values, split labels, dataset-specific forecasting hints, or manually supplied future seasonality.

Time-LLM keeps its official per-variable prompt: dataset description, next/past lengths, min, max, median, trend direction, and top-five lags. Set the official domain description for each dataset and change only `seq_len=36` and `pred_len`. TimeCMA keeps its official per-variable template containing observed values, timestamp range, sampling interval, and total observed trend. PatchTST, iTransformer, TimesNet, and DLinear receive numeric/time-marker tensors only.

### Experiment matrix and run budget

- Phase A, data QA: 6 datasets × 4 horizons, no optimization.
- Phase B, one-batch model QA: 7 models × 6 datasets × 4 horizons = 168 forward/backward checks.
- Phase C, convergence pilot: all 7 models on `ETTm1/96` and `ECL/96`, seed 42 = 14 runs.
- Phase D, single-seed sweep: 168 runs with seed 42.
- Phase E, formal replication: repeat the 168-run matrix with seeds 2021, 2022, and 2023 = 504 runs.

Do not start Phase D until all Phase C runs produce finite losses, save a best validation checkpoint, and complete test evaluation. Do not start Phase E until single-seed tables pass completeness and fairness audits.

---

### Task 1: Freeze Upstream Sources and Experiment Manifest

**Files:**
- Create: `configs/general_forecasting/datasets.yaml`
- Create: `configs/general_forecasting/models.yaml`
- Create: `configs/general_forecasting/experiment_matrix.yaml`
- Create: `docs/general_forecasting_source_audit.md`
- Test: `tests/test_general_experiment_config.py`

**Interfaces:**
- Produces: `load_general_experiment_spec(path: Path) -> GeneralExperimentSpec`
- Produces: immutable source URL/commit records for all six baselines and checksums for every raw dataset.

- [ ] **Step 1: Write configuration tests**

Test exact dataset/model/horizon sets, `input_len == 36`, three formal seeds, unique run IDs, and source commits currently audited on the server: PatchTST `204c21e`, iTransformer `c2426e6`, TimeCMA `223e4ae`, TimesNet `4e938a1`, DLinear `0c11366`, and Time-LLM `b13e881`.

- [ ] **Step 2: Run the tests and verify failure**

Run: `python -m unittest tests.test_general_experiment_config -v`

Expected: FAIL because the general experiment configuration loader and YAML files do not exist.

- [ ] **Step 3: Implement typed configuration loading**

Add frozen dataclasses for datasets, models, horizons, seeds, paths, and source commits. Reject unknown datasets, duplicate run IDs, non-36 input lengths, horizons outside `{96,192,336,720}`, and missing source commits.

- [ ] **Step 4: Record source methodology**

Document for each baseline: official repository/commit, model file, data loader, optimizer/loss/scheduler, early stopping, and prompt implementation if any. Explicitly distinguish source-preserved fields from protocol-overridden fields (`seq_len`, `pred_len`, paths, feature counts, seed, output path).

- [ ] **Step 5: Run tests and commit**

Run: `python -m unittest tests.test_general_experiment_config -v`

Expected: PASS.

Commit: `git commit -m "Define general forecasting experiment manifest"`

### Task 2: Download, Canonicalize, and Fingerprint Data

**Files:**
- Create: `bstalignment/general_data_schema.py`
- Create: `bstalignment/prepare_general_data.py`
- Create: `scripts/prepare_general_data.sh`
- Create: `tests/test_general_data_schema.py`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `prepare_dataset(spec: DatasetSpec, raw_path: Path, output_root: Path) -> DatasetManifest`
- Produces: `bstalignment/data/processed/general/<dataset>/<dataset>.csv`
- Produces: `bstalignment/data/processed/general/<dataset>/manifest.json`

- [ ] **Step 1: Write schema and corruption tests**

Cover timestamp parsing, monotonic timestamps, expected frequency, duplicate timestamps, numeric variable ordering, finite-value imputation, header-derived feature count, raw SHA-256, processed SHA-256, and rejection of target leakage columns.

- [ ] **Step 2: Verify tests fail**

Run: `python -m unittest tests.test_general_data_schema -v`

Expected: FAIL because schema utilities are absent.

- [ ] **Step 3: Implement canonical conversion**

Preserve the official CSV value columns and order. Rename only the timestamp column to `date`; never reorder numeric columns by alphabet. Apply causal forward fill, then train-median fill for leading missing values, and record every changed cell count. Do not standardize the persisted canonical CSV.

- [ ] **Step 4: Add data acquisition commands**

Use the dataset links supplied by the official Time-Series-Library/ETDataset sources. Downloads must be resumable and checksum-verified. The script exits before modification when a raw file checksum differs from the manifest.

- [ ] **Step 5: Prepare and inspect all datasets**

Run: `bash scripts/prepare_general_data.sh /root/autodl-tmp/GraphReportTS`

Expected: six processed CSV files and six manifests; feature counts match the table above; no timestamp gaps beyond documented source behavior.

- [ ] **Step 6: Run tests and commit**

Run: `python -m unittest tests.test_general_data_schema -v`

Expected: PASS.

Commit: `git commit -m "Add canonical general forecasting datasets"`

### Task 3: Implement Official Splits, Scaling, and Shared Windows

**Files:**
- Modify: `bstalignment/data_general.py`
- Create: `bstalignment/general_protocol.py`
- Create: `tests/test_general_protocol.py`

**Interfaces:**
- Produces: `split_bounds(dataset: str, n_rows: int, input_len: int) -> dict[str, tuple[int, int]]`
- Produces: `fit_train_scaler(values: np.ndarray, train_end: int) -> StandardScalerNP`
- Produces: `window_index(split: str, pred_len: int) -> np.ndarray`

- [ ] **Step 1: Write exact-boundary tests**

Assert ETTh boundaries `0/8640/11520/14400`, ETTm boundaries `0/34560/46080/57600`, and generic chronological boundaries. Assert the first validation/test target index equals the split boundary and the final target remains inside its split.

- [ ] **Step 2: Write leakage tests**

Construct validation/test outliers and prove they do not change scaler mean/std. Construct a boundary sentinel and prove no training target crosses into validation.

- [ ] **Step 3: Verify tests fail**

Run: `python -m unittest tests.test_general_protocol -v`

Expected: FAIL because `_timecma_split_indices` currently applies 70/10/20 to ETT and does not expose canonical indices.

- [ ] **Step 4: Implement the protocol**

Replace the ratio fallback with explicit ETT classes and a generic custom class. Return raw/scaled history, future target, timestamp markers, canonical column names, absolute indices, and scaler metadata. Set `input_len=36`; remove the old default of 96 from formal general runs.

- [ ] **Step 5: Verify tests and commit**

Run: `python -m unittest tests.test_general_protocol -v`

Expected: PASS.

Commit: `git commit -m "Implement leakage-free general data protocol"`

### Task 4: Validate and Bound General-GraphReportTS for 321 Variables

**Files:**
- Modify: `bstalignment/raw_signal.py`
- Modify: `bstalignment/graph_report_model.py`
- Modify: `bstalignment/data_general.py`
- Modify: `bstalignment/train_graph_report.py`
- Create: `tests/test_general_graph_scaling.py`

**Interfaces:**
- Produces: `build_variable_maps(x: np.ndarray, ...) -> np.ndarray` shaped `[variables, views, height, width]`
- Produces: `VariableGraphEncoder.forward(maps, variable_mask) -> dict[str, Tensor]`
- Produces: bounded-memory forward for `[B, 321, views, H, W]` without discarding variables.

- [ ] **Step 1: Write shape, invariance, and memory tests**

Test 7, 21, and 321 variables; output must remain `[B, pred_len, variables]`. Test that padding masks remove padded variables and add a CUDA smoke test whose peak allocated memory stays below a documented 4090-safe threshold at batch size 1 for ECL.

- [ ] **Step 2: Verify the current model against the ECL memory test**

Run: `python -m unittest tests.test_general_graph_scaling -v`

Expected: FAIL if the current path has incompatible shape or excessive graph memory; otherwise record the passing bound and avoid unnecessary architecture changes.

- [ ] **Step 3: Implement shared per-variable encoding**

If the memory test fails, build Hankel/first-derivative/second-derivative views per variable, apply one shared map encoder to variable chunks, and pool variable representations with bounded memory. Decode all variables without discarding any input channel. If the existing path passes, retain it and document the measured bound.

- [ ] **Step 4: Preserve low-dimensional graph behavior**

Keep the existing battery branch unchanged. For general datasets, add a compatibility flag allowing the legacy global graph only for diagnostic comparisons; the formal protocol always uses the scalable variable path.

- [ ] **Step 5: Verify tests and commit**

Run: `python -m unittest tests.test_general_graph_scaling -v`

Expected: PASS on CPU shape tests and the available CUDA ECL smoke test.

Commit: `git commit -m "Scale general graph forecasting to high-dimensional data"`

### Task 5: Implement Leakage-Free General Prompting

**Files:**
- Create: `bstalignment/general_prompting.py`
- Modify: `bstalignment/data_general.py`
- Create: `tests/test_general_prompting.py`

**Interfaces:**
- Produces: `build_general_prompt(history, columns, frequency, pred_len) -> str`
- Produces: stable prompt snapshots for 7-, 21-, and 321-variable inputs.

- [ ] **Step 1: Write prompt snapshot and leakage tests**

Assert the exact generic template, 36-step observation statement, requested horizon, all-variable count, maximum 12 variable summaries, deterministic selection, and stable output under changes to future targets. Assert no battery terms (`SOH`, capacity, cycle, chemistry, degradation) occur.

- [ ] **Step 2: Verify tests fail**

Run: `python -m unittest tests.test_general_prompting -v`

Expected: FAIL because the current `build_report_from_array` truncates names without a formal high-dimensional selection policy.

- [ ] **Step 3: Implement the generic report**

Compute only observed-window standardized statistics. Keep exact variable names from the canonical manifest. Format floats consistently and cap tokenizer length deterministically; log token counts and truncation status.

- [ ] **Step 4: Verify tests and commit**

Run: `python -m unittest tests.test_general_prompting -v`

Expected: PASS.

Commit: `git commit -m "Add general leakage-free forecasting prompts"`

### Task 6: Add Source-Consistent General Baseline Adapters

**Files:**
- Create: `bstalignment/train_general_baselines.py`
- Create: `bstalignment/general_baseline_profiles.py`
- Modify: `bstalignment/baseline_adapters.py`
- Create: `tests/test_general_baselines.py`

**Interfaces:**
- Produces: `build_general_baseline(name, dataset_meta, args) -> nn.Module`
- Produces: `resolve_general_profile(name, dataset, pred_len) -> BaselineTrainingProfile`
- Produces: one shared evaluation/result schema while preserving source training mechanics.

- [ ] **Step 1: Write adapter-contract tests**

For every model/dataset/horizon, assert input `[B,36,C]`, prediction `[B,H,C]`, all-channel output, source commit, source model class, and exact prompt policy (`none`, `timecma`, or `time_llm`).

- [ ] **Step 2: Verify tests fail**

Run: `python -m unittest tests.test_general_baselines -v`

Expected: FAIL because only battery-specific baseline adapters exist.

- [ ] **Step 3: Implement non-text adapters**

Use official model classes for PatchTST, iTransformer, TimesNet, and DLinear. Set `enc_in=c_out=C`, `seq_len=36`, requested `pred_len`, and `label_len=18` only where an encoder-decoder API requires it. Preserve each official dataset/horizon architecture and optimizer settings except parameters made invalid by `seq_len=36`; document every necessary patch-length adjustment.

- [ ] **Step 4: Implement TimeCMA adapter**

Generate/cache official prompt embeddings per dataset/split/input length/source commit/scaler checksum. Cache keys include absolute sample index and variable index. Use the official timestamp/value/trend template and frozen GPT-2 encoder, and verify no future values enter prompt storage.

- [ ] **Step 5: Implement Time-LLM adapter**

Use official per-variable statistics and top-five lag prompt with dataset-specific factual domain descriptions. Freeze the selected official LLM backbone. Do not substitute the main-model prompt. Record tokenizer/model revision and precision.

- [ ] **Step 6: Verify tests and commit**

Run: `python -m unittest tests.test_general_baselines -v`

Expected: PASS.

Commit: `git commit -m "Add official general forecasting baselines"`

### Task 7: Add Unified Metrics, Checkpoints, and Provenance

**Files:**
- Create: `bstalignment/general_results.py`
- Modify: `bstalignment/train_graph_report.py`
- Modify: `bstalignment/train_general_baselines.py`
- Create: `tests/test_general_results.py`

**Interfaces:**
- Produces: `run_config.json`, `metrics.json`, `history.csv`, `predictions.npz`, `best.pt`, and `environment.json` per run.
- Produces: `validate_completed_run(run_dir, expected_spec) -> ValidationResult`.

- [ ] **Step 1: Write result-contract tests**

Assert sample-weighted MSE/MAE over identical standardized targets, best-checkpoint selection by validation MSE, no test-set checkpoint selection, complete provenance, atomic completion markers, and rejection of mismatched dataset checksum/protocol/source commit.

- [ ] **Step 2: Verify tests fail**

Run: `python -m unittest tests.test_general_results -v`

Expected: FAIL because no unified general result contract exists.

- [ ] **Step 3: Implement metrics and atomic output**

Write runs to `<run_dir>.partial`; rename to the final directory only after best-checkpoint test evaluation and artifact validation. Save predictions and targets in standardized space with sample/step/variable indices. Record wall time, peak GPU memory, trainable parameter count, epochs, and early-stop reason.

- [ ] **Step 4: Verify tests and commit**

Run: `python -m unittest tests.test_general_results -v`

Expected: PASS.

Commit: `git commit -m "Standardize general forecasting results"`

### Task 8: Build Dry-Run, Pilot, and Formal Orchestration

**Files:**
- Create: `scripts/run_general_experiments.sh`
- Create: `scripts/run_general_pilot.sh`
- Create: `scripts/audit_general_runs.sh`
- Create: `tests/test_general_pipeline.py`

**Interfaces:**
- Produces: deterministic run commands from `experiment_matrix.yaml`.
- Produces: restart-safe execution with per-run logs and completion validation.

- [ ] **Step 1: Write orchestration tests**

Assert exactly 168 commands per seed, unique output paths, main plus six baseline names, all six datasets, all four horizons, `seq_len=36`, no battery flags, no test leakage, and skip behavior only after artifact validation.

- [ ] **Step 2: Verify tests fail**

Run: `python -m unittest tests.test_general_pipeline -v`

Expected: FAIL because general orchestration scripts do not exist.

- [ ] **Step 3: Implement dry-run generation**

Output path format:

```text
runs/general_v1/<dataset>/<model>/L36_H<pred_len>/seed_<seed>/
```

Support `--phase qa|pilot|sweep|formal`, `--models`, `--datasets`, `--horizons`, `--seeds`, `--dry-run`, and `--resume`. Never delete a valid run automatically; quarantine invalid partial outputs with a timestamp.

- [ ] **Step 4: Implement resource-aware ordering**

Order each horizon by DLinear → iTransformer → TimesNet → PatchTST → GraphReportTS → TimeCMA → Time-LLM. Run ETT/Weather before ECL. Default to one GPU job at a time on the RTX 4090. Allow gradient accumulation but keep effective batch size and its calculation in metadata; choose the largest per-model batch that passes the memory pilot.

- [ ] **Step 5: Run syntax and matrix tests**

Run: `bash -n scripts/run_general_experiments.sh scripts/run_general_pilot.sh scripts/audit_general_runs.sh`

Run: `python -m unittest tests.test_general_pipeline -v`

Expected: all checks PASS and dry-run emits exactly 168 commands for one seed.

- [ ] **Step 6: Commit**

Commit: `git commit -m "Add staged general experiment pipeline"`

### Task 9: Execute Training Readiness Gates

**Files:**
- Create: `docs/general_forecasting_readiness_report.md`
- Modify: `README.md`

**Interfaces:**
- Produces: signed-off readiness report; no formal training may begin without every gate passing.

- [ ] **Step 1: Run the complete test suite**

Run: `python -m unittest discover -s tests -v`

Expected: all battery and general tests PASS.

- [ ] **Step 2: Audit canonical data**

Run: `bash scripts/audit_general_runs.sh --data-only`

Expected: six checksum-valid datasets, exact columns/frequencies/splits, zero unreported imputations, and nonempty windows for all four horizons.

- [ ] **Step 3: Run all 168 one-batch checks**

Run: `bash scripts/run_general_experiments.sh --phase qa --seed 42`

Expected: finite forward loss, finite gradients, correct output shapes, and recorded peak memory for every matrix cell.

- [ ] **Step 4: Run the 14-run convergence pilot**

Run: `bash scripts/run_general_pilot.sh --seed 42`

Expected: every run lowers training loss from its initial epoch, saves a best validation checkpoint, and produces finite test MSE/MAE. ECL peak memory must fit the RTX 4090 with at least 10% headroom.

- [ ] **Step 5: Freeze hyperparameters**

Use pilot data only to fix batch size, gradient accumulation, precision, and OOM-safe chunk size. Do not tune architecture or learning rate against test metrics. Record all final profiles and their rationale in the readiness report.

- [ ] **Step 6: Commit readiness evidence**

Commit: `git commit -m "Document general experiment readiness"`

### Task 10: Run Single-Seed Sweep, Audit, Then Formal Replicates

**Files:**
- Create: `bstalignment/summarize_general_results.py`
- Create: `tests/test_general_summary.py`
- Create: `docs/general_forecasting_runbook.md`

**Interfaces:**
- Produces: `runs/general_v1/summary_seed42.csv`
- Produces: `runs/general_v1/summary_formal.csv` with mean, standard deviation, run count, and completeness flags.

- [ ] **Step 1: Write summary completeness tests**

Require one row per dataset/model/horizon/seed, identical protocol/checksums across compared rows, finite MSE/MAE, and exactly three seeds for every formal aggregate.

- [ ] **Step 2: Run the single-seed sweep**

Run: `bash scripts/run_general_experiments.sh --phase sweep --seeds 42 --resume`

Expected: 168 validated completed runs.

- [ ] **Step 3: Audit the single-seed results**

Run: `python -m bstalignment.summarize_general_results --root runs/general_v1 --seeds 42 --strict`

Expected: complete 6 × 7 × 4 table; no protocol mismatches or invalid artifacts.

- [ ] **Step 4: Run formal seeds**

Run: `bash scripts/run_general_experiments.sh --phase formal --seeds 2021,2022,2023 --resume`

Expected: 504 validated completed runs.

- [ ] **Step 5: Produce the formal table**

Run: `python -m bstalignment.summarize_general_results --root runs/general_v1 --seeds 2021,2022,2023 --strict`

Expected: MSE/MAE mean ± standard deviation for every model/dataset/horizon, plus average rank by horizon and dataset; missing or mismatched runs cause a nonzero exit.

- [ ] **Step 6: Commit code and runbook, not generated artifacts**

Commit: `git commit -m "Add general forecasting result aggregation"`

---

## Training Profiles and Fairness Rules

1. Use each baseline's official model implementation and dataset/horizon-specific hyperparameters as the starting profile. Override only the shared protocol fields and parameters mathematically incompatible with a 36-step input.
2. Preserve source loss, optimizer, scheduler, warmup, early stopping, and checkpoint semantics. Record the resolved profile in every run.
3. GraphReportTS uses its own validation-only training profile chosen during Phase C. It must not inherit the battery epoch budget blindly.
4. Batch size may differ by model because memory costs differ. Report both microbatch and effective batch size; use gradient accumulation to target effective batch 32 where feasible.
5. Use mixed precision consistently where the official source supports it. Time-LLM backbone precision must match its official configuration and be logged.
6. Validation selects checkpoints and may set early stopping. Test is evaluated once from the selected checkpoint and never influences prompt, preprocessing, hyperparameters, or retries.
7. A failed seed is rerun with the same configuration. Any emergency numerical or memory change creates a new protocol revision and invalidates direct aggregation with prior runs.

## Required Final Tables

- Per dataset: rows are seven models; columns are MSE/MAE for horizons 96/192/336/720 plus horizon average.
- Overall: average rank and mean normalized error across six datasets, reported separately for MSE and MAE.
- Efficiency: trainable parameters, total parameters, peak GPU memory, training wall time, inference samples/second.
- Prompt audit: token length and truncation rates for GraphReportTS, TimeCMA, and Time-LLM; prompt-disabled models marked N/A.
- Reproducibility appendix: source commits, dataset checksums, resolved hyperparameters, seeds, software/CUDA versions, and failed/retried run log.

## Primary Sources Consulted

- ETT dataset and Informer preprocessing: https://github.com/zhouhaoyi/ETDataset and https://github.com/zhouhaoyi/Informer2020
- Standard dataset loaders and TimesNet: https://github.com/thuml/Time-Series-Library
- PatchTST: https://github.com/yuqinie98/PatchTST
- iTransformer: https://github.com/thuml/iTransformer
- DLinear: https://github.com/cure-lab/LTSF-Linear
- Time-LLM prompt and implementation: https://github.com/KimMeen/Time-LLM
- TimeCMA prompt and implementation: https://github.com/ChenxiLiu-HNU/TimeCMA

## Self-Review Result

- Spec coverage: all requested datasets, all existing six baselines, input 36, four horizons, generic main prompt, source-consistent baseline prompts, heterogeneous feature handling, preprocessing, staged training, and pre-training readiness gates are covered.
- Placeholder scan: no deferred implementation placeholders are present.
- Type consistency: dataset manifest, shared protocol, baseline profiles, result contract, and orchestration interfaces use consistent names across tasks.
- Critical pre-training blockers: official data are not yet present on the server; ETT split logic is currently incorrect; the current general graph path must be validated for ECL's 321 variables; general baseline runners and formal orchestration do not yet exist.
