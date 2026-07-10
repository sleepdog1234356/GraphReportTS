# Core Battery Ablation Redesign

Date: 2026-07-10
Status: Approved design, pending implementation plan

## 1. Purpose

Replace the current 16-variant battery ablation suite with a smaller, scientifically focused suite that measures the four contributions central to the project report:

1. the Hankel dynamic-graph representation relative to direct sequence encoding;
2. the additional IC/DV signal channels;
3. the learnable semantic gate;
4. the report prompt.

The redesign must reduce the number and cost of formal ablation runs without changing the formal optimization protocol. It must also be deployed into the already-running remote pipeline without interrupting or changing the main-model or baseline jobs that are in progress.

## 2. Scope and Experimental Matrix

The suite uses one-factor-at-a-time ablations. Every dataset has five result rows, of which four require new training:

| Result row | Definition | Execution source |
| --- | --- | --- |
| `full` | Unmodified formal GraphReportTS | Reuse the completed formal main-model result |
| `no_hankel_graph` | Replace all Hankel/map/graph processing with direct resampled-sequence encoding | New training |
| `no_ic_dv` | Keep Hankel, derivative maps, and graph processing, but remove the additional IC/DV channels and all maps derived from them | New training |
| `no_text_gate` | Keep prompt retrieval and semantic alignment, but replace the learned gate with a constant value of one | New training |
| `no_report_prompt` | Remove the text encoder, semantic fusion, gate, and alignment loss | New training |

The matrix is executed in full on MIT, CALCE, and XJTU. This produces 12 new training jobs rather than the current 48 jobs. No factorial combinations are included.

The following legacy ablations are removed from the formal battery suite:

- `no_numeric_history`
- `no_multi_cycle_raw`
- `single_cycle_raw`
- `no_semantic_alignment`
- `no_align_loss`
- `absolute_step_decoder`
- `no_derivative_map`
- `static_graph`
- `no_domain_edges`
- `no_cross_modal`
- `separate_heads`
- the old `no_hankel_map` definition

They may remain available as non-formal developer switches, but the formal core suite must not schedule them or include them in its summary.

## 3. Controlled Variables

All new variants retain the formal battery protocol:

- seed 42 and the identical train/validation/test cell split;
- 32 observed cycles and 20 future-only targets;
- batch size 64;
- no historical SOH in any model input;
- FP32 execution;
- the current GraphReportTS optimizer parameter groups and learning rates;
- SmoothL1 regression loss;
- five-epoch learning-rate warmup;
- validation-MSE plateau scheduling;
- semantic-alignment ramp from epochs 6 through 15 where alignment exists;
- maximum 80 epochs;
- early stopping beginning at epoch 20 with patience 20;
- checkpoint selection by validation MSE only;
- one test evaluation after training, using `best.pt`.

AMP, TF32 changes, reduced epochs, reduced data, altered validation frequency, and changed batch size are excluded from the formal redesign because the reused `full` result was produced under the existing FP32 protocol.

## 4. Model Architecture

### 4.1 Full graph path

The full path remains unchanged:

```text
raw cycle signals
  -> resampled base channels and IC/DV
  -> Hankel and first/second-derivative maps
  -> channel-preserving 2D patch nodes
  -> GraphMapEncoder
  -> per-cycle representation
  -> InterCycleTemporalEncoder
```

The implementation must preserve the full-model parameter names, shapes, forward behavior, and configuration semantics so the previously trained full result remains a valid reference.

### 4.2 `no_hankel_graph` direct sequence path

This is a structural ablation, not a reuse of the existing `--no_hankel_map` switch. It removes every operation specific to the 2D map and graph representation:

- Hankel construction;
- first- and second-derivative map construction;
- 2D map caching and loading;
- patch extraction;
- graph nodes and graph attention;
- structural graph bias and domain edges;
- `GraphMapEncoder` construction and execution.

For each cycle, the dataset provides six aligned sequence channels in this fixed order:

```text
current, voltage, temperature, capacity, IC(dQ/dV), DV(dV/dQ)
```

After resampling, the history tensor has shape `[B, 32, 128, 6]`. The direct encoder is:

```text
[B, 32, 128, 6]
  -> reshape [B*32, 128, 6]
  -> Linear(6, d_model=128)
  -> learned positional embedding over 128 positions
  -> two TransformerEncoder layers
       n_heads=4
       feed-forward width=4*d_model
       dropout=0.1
  -> learned attention pooling over the 128 sequence positions
  -> [B, 32, 128]
  -> existing InterCycleTemporalEncoder
```

No patching is performed. Each resampled position is one Transformer token. `RawSequenceEncoder` belongs to the core optimizer group and replaces `GraphMapEncoder`; the two encoders must never coexist in a `no_hankel_graph` model instance.

### 4.3 `no_ic_dv`

This variant retains the full graph path for the four original channels. It omits IC and DV before map construction, so no Hankel or derivative maps are produced from IC/DV. All graph, temporal, numeric-history, semantic, and decoder modules otherwise remain identical to the full model.

### 4.4 `no_text_gate`

This variant retains the report prompt, frozen text backbone, trainable text projection, token-aware retrieval, text context, and semantic alignment. It removes the learnable gate network and uses:

```text
gate = 1
context = base_context + text_context
```

The constant gate is logged as mean/min/max 1 and standard deviation 0. No unused trainable gate parameters may remain in the model or optimizer.

### 4.5 `no_report_prompt`

This variant retains the complete graph and numeric-history branches. It does not instantiate or execute the text encoder, semantic fusion, or gate. Its alignment weight is always zero. No unused semantic parameters may remain in the optimizer.

## 5. Data and Cache Design

### 5.1 Shared resampled sequence builder

Extract one deterministic sequence-building function from the existing raw-signal path. It must reuse the current resampling, robust scaling, and smoothed-gradient definitions so that the sequences used by the graph and direct-sequence paths originate from identical numerical inputs.

For formal runs, missing required raw channels are errors. The formal path must not synthesize zero channels or use the summary-derived smoke-test fallback.

### 5.2 Cache allocation

- `no_text_gate` and `no_report_prompt` reuse the complete full graph cache.
- `no_ic_dv` uses one separate cycle-level graph cache per dataset with `include_ic_dv=false`.
- `no_hankel_graph` uses a new cycle-level sequence cache. Each `(dataset, cell_id, cycle)` is stored once as `[128, 6]`, and each sample stores 32 cycle indices.
- Targets, masks, history features, history cycle IDs, split metadata, and cycle scaling remain identical across variants.

The sequence cache manifest includes dataset, split, resampling length, channel order, IC/DV formula version, seed, maximum-cycle setting, history length, and cache schema version. A manifest mismatch is a hard failure when the formal runner requires a cache.

### 5.3 Prompt invariance

Every text-enabled ablation must receive exactly the same prompt string that the formal full model received for the same sample. Prompt content must not be regenerated from the ablated variant's active channel list.

Implementation must compare generated or loaded prompts against the full graph-cache metadata. Any mismatch in prompt text, sample identity, or ordering is a hard failure. This preserves the existing full result and ensures that `no_ic_dv` and `no_hankel_graph` change numerical architecture inputs only.

## 6. Runner and Result Provenance

Create a separately versioned formal suite, for example:

```text
ablation_suite_version = core-v1
```

This version is distinct from the existing training-strategy version. The new suite writes to an isolated root:

```text
runs/full_hf_v3_training_strategy_nosoh/
  graph_report_core_ablation/
    battery/
      mit/
        no_hankel_graph/
        no_ic_dv/
        no_text_gate/
        no_report_prompt/
      calce/
      xjtu/
```

Recommended scheduling order within each dataset is:

```text
reuse full
  -> no_hankel_graph
  -> no_report_prompt
  -> no_ic_dv
  -> no_text_gate
```

The runner supports selecting datasets and variants for verification and recovery, but its default formal dry-run must resolve to exactly 3 datasets by 4 trained variants.

Completed results are skipped only when both `test_metrics.json` and matching run metadata exist. Incomplete matching runs resume from `last.pt`. The ablation stage uses an independent `ABLATION_FORCE_RETRAIN` control and must not inherit the top-level pipeline's default `FORCE_RETRAIN=1`.

### 6.1 Full-result reuse

The `full` row is imported from the formal main-model output only after validating:

- dataset identity;
- split/seed identity;
- history length 32 and prediction length 20;
- batch size 64;
- no-SOH input protocol;
- training-strategy version;
- complete full-model flags and model configuration;
- existence of `best.pt`, `run_config.json`, and `test_metrics.json`.

The protocol stage may differ (`main` versus `ablation`) and is not itself a mismatch. Any substantive mismatch fails the suite rather than silently retraining or importing an invalid reference.

### 6.2 Summary output

Each dataset summary contains five rows and at least these columns:

- ablation name;
- MSE, MAE, and RMSE;
- best epoch and stopped epoch;
- mean epoch seconds and total training seconds where available;
- trainable parameter count;
- result source (`reused_main` or `trained_ablation`);
- training-strategy version;
- ablation-suite version;
- source Git commit;
- full-reference Git commit.

The summary must preserve the two commits separately because the full result predates the ablation-only implementation commit.
Timing columns are nullable only for a reused full result whose existing epoch history predates timing instrumentation; every newly trained ablation must populate them.

## 7. Failure Handling

The suite fails before scheduling formal training when:

- the full reference is missing or incompatible;
- a required graph or sequence cache is absent or has incompatible metadata;
- prompt strings or sample ordering differ from the full reference;
- required formal raw channels are missing;
- a model variant contains modules forbidden by its definition;
- optimizer groups omit a trainable parameter or include a forbidden/unused parameter;
- checkpoint, training-strategy, or ablation-suite versions do not match.

Failures must name the dataset, variant, expected value, observed value, and affected path. Existing valid results are never deleted as a recovery side effect.

## 8. Verification Plan

### 8.1 Unit tests

- Verify resampled base and IC/DV sequences against the inputs used immediately before full map construction.
- Verify fixed channel order, tensor shapes, and deterministic IC/DV computation.
- Verify direct and cached sequence samples are exactly equal.
- Verify cached and uncached graph samples remain exactly equal.
- Verify text-enabled variants receive prompts identical to full-cache prompts.
- Verify `no_hankel_graph` has no graph encoder and fails if any graph function is invoked.
- Verify `no_text_gate` produces a constant-one gate and has no trainable gate parameters.
- Verify `no_report_prompt` has no text/fusion/gate parameters and zero alignment weight.
- Verify optimizer groups are disjoint and cover every trainable parameter for all variants.
- Verify the unchanged full configuration preserves its parameter names, shapes, and forward output contract.

### 8.2 Integration tests

For every new variant, run one batch through:

```text
load -> forward -> loss -> backward -> gradient clip -> optimizer step -> validation
```

Check output shape `[B, 20]`, finite losses, expected gradient presence/absence, checkpoint serialization, and bounded GPU memory.

### 8.3 Runner tests

- Dry-run emits exactly 12 new formal training commands.
- Completed compatible variants skip.
- Incomplete compatible variants resume.
- Incompatible metadata fails without deletion.
- Full-reference reuse succeeds for the current formal outputs and fails for deliberate mismatches.
- Legacy 16-item summaries cannot be read as `core-v1` output.

## 9. Remote No-Interruption Handoff

The current remote pipeline and active worktree must remain untouched by the new Python implementation while main models or baselines are running.

The deployment sequence is:

1. Complete and commit implementation locally.
2. Create a separate remote Git worktree pinned to the new commit, for example `/root/autodl-tmp/GraphReportTS-core-ablation`.
3. Access the active worktree's datasets, HuggingFace weights, caches, and output root through explicit absolute asset paths; do not copy or mutate them during preflight.
4. In the new worktree, run compilation, unit tests, one-batch tests, cache checks, full-reference checks, and the suite dry-run.
5. Record the active training process tree, PID, elapsed time, current epoch, latest checkpoint timestamp, and GPU utilization.
6. Atomically replace only the future-stage `scripts/run_battery_ablations_full_hf.sh` entry in the active worktree with a delegator pinned to the new worktree and commit. Do not change the top-level pipeline, main runner, baseline runner, or any Python file that a future main/baseline subprocess may import.
7. Recheck that the active PID is unchanged, elapsed time increased, GPU utilization remains normal, and epoch/checkpoint progress continues.
8. When the existing pipeline reaches the ablation stage, the delegator launches the new suite and writes to the new isolated result root.

The delegator is installed only after the new worktree passes every preflight check. If the delegated ablation entry fails, the top-level pipeline remains fail-fast at the ablation boundary; completed main-model and baseline outputs are preserved, and the legacy 16-item suite is not started.

## 10. Acceptance Criteria

- Formal scheduling contains only four trained variants per dataset.
- MIT, CALCE, and XJTU each produce a five-row summary including the reused full result.
- `no_hankel_graph` performs no Hankel, derivative-map, patch, or graph operation.
- `no_ic_dv` changes only IC/DV numerical inputs and their derived maps.
- `no_text_gate` changes only adaptive gating.
- `no_report_prompt` removes the complete text path and alignment.
- All text-enabled variants use prompts identical to the reused full run.
- All new trainings use the unchanged formal FP32 optimization protocol.
- Restarting the pipeline does not delete or retrain compatible completed ablations.
- Remote handoff leaves the active training PID and progress uninterrupted.
- The legacy formal outputs and legacy 16-item ablation outputs remain isolated and unmodified.

## 11. Out of Scope

- mixed-precision conversion of the formal suite;
- multi-seed statistical studies;
- factorial interactions among the four factors;
- changes to official baseline training;
- changes to the full GraphReportTS architecture or its completed checkpoints;
- deployment to multiple GPUs or additional servers;
- removal of legacy developer switches from the command-line interface.
