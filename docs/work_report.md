# GraphReportTS Work Report

Updated: 2026-07-10

## 1. Project Goal

This project builds GraphReportTS for battery SOH forecasting and general time-series forecasting. The battery setting is the current focus: train the main model on MIT, CALCE, and XJTU; compare against strong time-series baselines; and run ablations that measure the contribution of each model component.

As of 2026-07-10, earlier full-HF comparisons and the stopped `runs/full_hf_v2` attempt are legacy references. The earlier input contract included historical SOH in baseline sequence inputs and in the first v2 numeric-history branch; later v2 no-SOH runs still used a unified fixed AdamW/SmoothL1/no scheduler baseline loop. Their metrics are not formal v3 results. The formal no-historical-SOH rerun uses `runs/full_hf_v3_training_strategy_nosoh` and the full v3 protocol: main -> baselines -> ablations, with source-native baseline profiles and adaptive GraphReportTS training.

## 2. Current Main Model

GraphReportTS uses a raw-signal pipeline instead of only low-dimensional cycle summaries.

1. Raw cycle channels are resampled and converted into multi-view 2D maps.
2. Map views include Hankel delay maps, first/second derivative maps, and optional battery IC/DV maps.
3. 2D maps are split into channel-preserving patch nodes.
4. A graph temporal encoder applies dynamic attention with structural biases.
5. A statistical report prompt is encoded by a text encoder.
6. Cross-modal fusion combines graph context and report text.
7. A unified query decoder predicts the future SOH horizon.

The battery model is implemented in `bstalignment/graph_report_model.py`, with raw-signal mapping in `bstalignment/raw_signal.py` and training in `bstalignment/train_graph_report.py`.

## 2.1 Battery-GraphReportTS Architecture Retained In V3

The multi-cycle design addresses the MIT/XJTU gap against PatchTST, iTransformer, TimesNet, and other sequence baselines. It is retained in the v3 protocol and should not be simplified.

Implementation notes:

- The dataset now emits future-only targets and masks with width `pred_len`.
- Raw maps are emitted as `multi_cycle_maps [B, 32, C_map, H, W]`.
- Numeric history is emitted as `history_features [B, 32, 8]`.
- The v2 cache is cycle-level: `cycle_maps.npy` stores each `(cell_id, cycle)` once and `history_indices.npy` stores the 32-cycle index window per sample. This avoids the 32x disk blow-up of sample-level multi-cycle caching.
- Training logs `gate_mean`, `gate_std`, `gate_min`, and `gate_max` to `epoch_history.jsonl`, and writes validation/test sample gates to CSV files.

Required v2 architecture:

1. **Baseline-aligned recent history input**
   - Use the recent `32` cycles as direct model input.
   - Predict the future `20` cycles.
   - Test metrics must be future-only; current-cycle estimation must not enter MSE/MAE/RMSE.
   - Older history is summarized into prompt statistics, not fed as a raw tensor.

2. **Multi-cycle raw-map branch**
   - Input shape: `[B, 32, C_map, H, W]`.
   - Each cycle is encoded by the existing shared `GraphMapEncoder`.
   - The first implementation then pools each cycle to `cycle_graph_repr [B, 32, D]`.
   - These cycle embeddings are passed into a new `InterCycleTemporalEncoder`.

3. **InterCycleTemporalEncoder first version**
   - This is a first-pass cycle-level temporal encoder, not the final possible spatio-temporal design.
   - Recommended implementation: a small Transformer encoder over `[B, 32, D]` with relative position embeddings.
   - It is designed to fit the current `GraphMapEncoder` output cleanly and avoid excessive memory use.
   - Future work, if training resources allow: preserve all patch tokens and apply cross-cycle attention on `[B, 32, N_patch, D]`.

4. **Numeric history branch**
    - Add `NumericHistoryEncoder` over `[B, 32, F]`.
   - Exclude historical SOH, SOH deltas, and SOH-derived aging-stage labels.
   - Current schema: raw capacity/QD value, capacity/QD z-score, IR z-score, charge-time z-score, cycle ratio, capacity/QD delta, IR delta, and charge-time delta.
   - This branch is required so the model has the same direct observable sequence information available to the baselines.

5. **Relative-step decoder**
    - Decode relative future steps `1..20`.
    - Do not use absolute `cycle_id` as the decoder step id.
   - Absolute aging information may enter as continuous covariates such as cycle ratio or recent capacity slope, but not as current/historical SOH.

6. **Learnable text gate**
   - Restore the learnable gate used in the earliest local MIT prototype.
   - Fuse text as `context = temporal_numeric_graph_context + gate * semantic_text_context`.
   - Log per-epoch `gate_mean`, `gate_std`, `gate_min`, and `gate_max`.
   - Save sample-level gate values during validation/test to measure prompt contribution.

7. **Weak and optional semantic alignment**
   - Earlier `w_align=0.01` can dominate small regression losses on MIT/XJTU.
   - Default should be `0.001` or lower, with support for `w_align=0` and warmup.
   - Alignment should be token-aware and closer to TimeCMA-style cross-modality alignment than a single global context/text contrastive loss.

## 3. Data Pipeline

Battery datasets:

- MIT: loaded from local MIT battery files.
- CALCE: raw files can be downloaded and converted into processed `.npz` files.
- XJTU: raw `.mat` files are converted into processed `.npz` files.

Formal training uses raw current, voltage, temperature, and capacity sequences. The summary-derived fallback path remains only for smoke tests through `--allow_summary_fallback`.

To reduce repeated CPU-side map construction, `bstalignment/precompute_battery_graph_cache.py` precomputes deterministic graph caches for each split. The v3 cache is cycle-level rather than sample-level: it stores `cycle_maps.npy` and per-sample `history_indices.npy`. The structural raw-sequence ablation uses a separate deterministic sequence cache under `BATTERY_SEQUENCE_CACHE_DIR` (default `runs/cache/battery_sequence`); the core runner validates prompt and target identity between graph and sequence representations before training.

## 4. V3 Training Strategy

The formal entrypoint is `scripts/run_battery_v3_training_strategy_pipeline.sh`, which writes to `runs/full_hf_v3_training_strategy_nosoh` in the exact order `main -> baselines -> ablations`. The main stage covers MIT, CALCE, and XJTU before the official baseline and ablation stages. The pipeline defaults to `FORCE_RETRAIN=1` for main/baseline work, while the separately exported `ABLATION_FORCE_RETRAIN=0` preserves valid core-ablation results and resumes compatible checkpoints. The top-level force value is not routed into ablations.

The approved RTX 4090 configuration uses batch 64 for GraphReportTS main/ablations, batch 128 for baselines, independent `CACHE_TASK_BATCH_SIZE=128` CPU cache scheduling, 16 workers for main/cache/ablations, and 8 workers for baselines. A full-model 32/20 preflight measured 33.116 GiB peak allocated CUDA memory at batch 64; batch 128 exhausted the 48 GiB device during the graph encoder forward pass.

The battery data contract is exactly 32 observed cycles followed by 20 future-only labels, with no terminal partial horizons. Historical SOH remains excluded from every model input. `cycle_ratio` uses train-only dataset-global cycle scaling: the maximum cycle ID from the seeded training-cell split is reused unchanged for train, validation, and test, with no clipping above 1.0.

GraphReportTS freezes the DistilBERT backbone in evaluation mode. AdamW separates main/core (`1e-3`) and semantic (`3e-4`) learning rates, uses a 5-epoch LR warmup from 10% to target rates, then applies a validation-MSE plateau scheduler and early stopping. SmoothL1 remains the regression loss, and `best.pt` remains validation-MSE selected. The delayed/ramped alignment is zero for epochs 1-5, linearly ramps during epochs 6-15 to `w_align=0.001`, then remains constant; early stopping begins at epoch 20 with patience 20.

The six official baselines use source-native profiles, not identical epoch budgets: PatchTST is Adam/MSE with batch-stepped OneCycleLR (100 epochs, patience 20); iTransformer, TimesNet, and DLinear are Adam/MSE with source-style type1 epoch decay (10 epochs, patience 3); Time-LLM is Adam/MSE with batch-stepped OneCycleLR (10 epochs, patience 10); and TimeCMA is AdamW/MSE with cosine epoch decay, gradient clipping 5.0, 100 epochs, patience 50, and early stopping only from epoch 50. All adapters select checkpoints by validation MSE, not test metrics.

The old v2 results are legacy because the unified fixed AdamW/SmoothL1/no scheduler loop did not retain source-native optimizer, loss, scheduler, or early-stop semantics. The late best validation epochs `73/79/54/77/72` for PatchTST, iTransformer, TimeCMA, TimesNet, and DLinear respectively demonstrate why a single shortened or identical baseline budget is invalid.

## 5. Baselines

The repository supports two comparison tracks:

- Compact in-repository baselines in `bstalignment/train_battery_baselines.py` for quick checks.
- Official baseline adapters in `bstalignment/train_battery_official_baselines.py` for PatchTST, iTransformer, TimeCMA, TimesNet, DLinear, and Time-LLM.

Official baseline source code is not vendored. It should be cloned under ignored `external/` by `scripts/clone_battery_baselines.sh`.

Time-LLM does not require an external API in this setup. It uses local HuggingFace model weights. Downloaded weights should be stored outside Git, such as under ignored `hf_models/`.

For the no-historical-SOH formal comparison, baseline inputs are `capacity_summary`, `capacity_delta`, `internal_resistance`, `charge_time`, and `cycle_ratio`; targets remain future SOH. TimeCMA prompt embeddings and Time-LLM config text must describe these observable features rather than historical SOH.

## 6. Ablation Design

The formal runner is `bstalignment/run_core_ablation_suite.py` with suite identity `core-v1`. Each dataset summary contains this exact five-row matrix:

| Row | Execution policy | Representation or controlled change |
| --- | --- | --- |
| `full` | reused from the completed main result | full `hankel_graph` model |
| `no_hankel_graph` | newly trained | raw cycle sequences; no map/graph encoder |
| `no_report_prompt` | newly trained | graph model without the report prompt |
| `no_ic_dv` | newly trained | graph model without IC/DV map channels |
| `no_text_gate` | newly trained | graph model with the learned text gate disabled |

The three reused `full` rows are imported, not retrained. Four variants on each of MIT, CALCE, and XJTU therefore produce exactly 12 new jobs. This replaces the legacy formal 16-item entry. In particular, the legacy `no_hankel_map` removed Hankel channels but kept the graph/map encoder; the new structural `no_hankel_graph` switches to the unpatched raw-sequence encoder and its sequence cache.

The formal shell is `scripts/run_battery_ablations_full_hf.sh ACTIVE_ASSET_ROOT`. It resolves its implementation location with `readlink -f`; `ABLATION_CODE_ROOT` can override that code root, while the positional asset root continues to supply data, existing main results, caches, and outputs. Graph variants use `BATTERY_GRAPH_CACHE_DIR`, and `no_hankel_graph` uses `BATTERY_SEQUENCE_CACHE_DIR`. The versioned provenance default is `FULL_REFERENCE_COMMIT=1d6a8f975fd3225cc087af90f03a00414ce84591`, the deployed full v3 source snapshot, not the active asset-tree HEAD. Explicit overrides must be non-empty 40-hex commits; a different valid commit is accepted only with `ALLOW_FULL_REFERENCE_COMMIT_OVERRIDE=1`, preventing accidental result mislabeling. `ABLATION_FORCE_RETRAIN=0` is the recovery-safe default: matching complete variants skip and matching incomplete variants resume; `ABLATION_FORCE_RETRAIN=1` explicitly retrains only the four core variants, never `full`.

Dry-run does not require full artifacts: the core runner validates CLI/protocol values and returns after printing exactly 12 training commands, before full-result or cache validation. This permits Task 8 preflight while formal full training is incomplete. Actual non-dry execution still requires completed full artifacts for MIT, CALCE, and XJTU and applies the strict reuse validation for all three datasets.

This code-root/asset-root split supports an atomic replacement of only the future ablation shell entry. Such a replacement does not restart or signal the current main/baseline process and does not alter its working tree; the reviewed core implementation is consumed only after the parent pipeline reaches `ablations`. On validation failure, preserve current training plus valid main/baseline/ablation artifacts, verify a corrected implementation commit, and atomically repoint the future entry. Do not restore the legacy 16-item runner.

## 7. Training Workflow

Typical battery workflow:

1. Place or download datasets under `bstalignment/data/`.
2. Preprocess CALCE/XJTU into processed `.npz` files.
3. Optionally precompute the battery graph cache.
4. Train GraphReportTS on MIT, CALCE, and XJTU.
5. Train official baselines on the same datasets and horizon.
6. Run ablations for each dataset.
7. Use generated metrics and figures under `runs/` for analysis.

`runs/` is ignored because it contains logs, checkpoints, figures, cached tensors, and metrics generated by experiments.

Formal v3 no-historical-SOH workflow:

1. Verify the no-SOH leakage tests locally or in the remote `graphreport` environment.
2. Prepare and preflight a reviewed detached code worktree; atomically repoint only the future ablation entry without changing the active worktree or current training process.
3. Run GraphReportTS main models on MIT, CALCE, and XJTU first.
4. Train official baselines with their source-native profiles.
5. Reuse the three compatible full results and train the 12 `core-v1` ablation jobs with the same main-model adaptive strategy.
6. Compare future-only metrics across the no-SOH baselines, main model, and ablations.

The formal script is `scripts/run_battery_v3_training_strategy_pipeline.sh`. `FORCE_RETRAIN=1` may be used for a new formal main/baseline run, independently of the recovery-safe `ABLATION_FORCE_RETRAIN=0` default. Baseline profiles intentionally retain their source-native, non-identical epoch budgets.

Remote status at the moment this report was updated:

- The previous official baseline comparison is complete for all 18 dataset/model combinations, but those results used historical SOH inputs and are now treated as an SOH-available upper-bound/reference rather than the formal no-SOH comparison.
- The first v2 pipeline under `runs/full_hf_v2` was stopped after discovering historical SOH in `history_features` and baseline inputs.
- The corrected v3 pipeline runs under `runs/full_hf_v3_training_strategy_nosoh` with local HF weights from `hf_models/distilbert-base-uncased`, `hf_models/openai-community__gpt2`, and `hf_models/google-bert__bert-base-uncased`.
- If training resources allow after the no-SOH run, add an optional strict `no_capacity_or_QD` sensitivity experiment to separate raw-map/text contributions from the capacity proxy.

## 8. Cleanup Status

Early local prototype code based on one-cycle MIT summary features has been removed from the active project. The remaining code is organized around GraphReportTS, official baseline adapters, data preprocessing, inference, visualization, and ablation experiments.

## 9. Repository Boundary

The GitHub repository should contain:

- source code;
- public documentation;
- reusable training, preprocessing, baseline, and ablation scripts;
- dependency and ignore configuration.

The repository should not contain:

- model checkpoints or downloaded LLM weights;
- raw or processed datasets;
- external baseline repositories;
- experiment outputs under `runs/`;
- local Codex handoff notes or machine-specific nohup orchestration files.

## 10. Memory-Safety Notes For Future Codex Sessions

Do not collapse the v3 multi-cycle design into a simpler single-cycle model. The following modules are required for the formal implementation:

- `multi_cycle_maps [B, 32, C_map, H, W]`
- shared `GraphMapEncoder` applied per cycle
- first-version `InterCycleTemporalEncoder` over `[B, 32, D]`
- `NumericHistoryEncoder` over `[B, 32, F]`
- future-only metric computation
- relative-step decoder
- learnable text gate with gate logging
- weak or disabled alignment option
- TimeCMA-style token-aware semantic retrieval/alignment
- v3 ablation suite

The first `InterCycleTemporalEncoder` is deliberately modest. It is not the final ceiling. If resources allow after the first v3 run, test full-patch cross-cycle attention over `[B, 32, N_patch, D]`, or a factorized design with intra-cycle patch graph attention followed by inter-cycle patch/channel attention.
