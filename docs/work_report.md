# GraphReportTS Work Report

Updated: 2026-07-10

## 1. Project Goal

This project builds GraphReportTS for battery SOH forecasting and general time-series forecasting. The battery setting is the current focus: train the main model on MIT, CALCE, and XJTU; compare against strong time-series baselines; and run ablations that measure the contribution of each model component.

As of 2026-07-10, the full-HF baseline comparison had completed on the remote server under the earlier input contract. That contract included historical SOH in the baseline sequence inputs and in the first v2 numeric-history branch, so the first `runs/full_hf_v2` attempt was stopped. The formal rerun now uses the no-historical-SOH contract under `runs/full_hf_v2_nosoh`, and official baselines must be retrained because their input features changed.

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

## 2.1 Planned Battery-GraphReportTS V2

The v2 design is intended to address the MIT/XJTU gap against PatchTST, iTransformer, TimesNet, and other sequence baselines. It should not be simplified during implementation.

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

To reduce repeated CPU-side map construction, `bstalignment/precompute_battery_graph_cache.py` precomputes deterministic graph caches for each split. For v2, this cache is cycle-level rather than sample-level: it stores `cycle_maps.npy` and per-sample `history_indices.npy`.

## 4. Baselines

The repository supports two comparison tracks:

- Compact in-repository baselines in `bstalignment/train_battery_baselines.py` for quick checks.
- Official baseline adapters in `bstalignment/train_battery_official_baselines.py` for PatchTST, iTransformer, TimeCMA, TimesNet, DLinear, and Time-LLM.

Official baseline source code is not vendored. It should be cloned under ignored `external/` by `scripts/clone_battery_baselines.sh`.

Time-LLM does not require an external API in this setup. It uses local HuggingFace model weights. Downloaded weights should be stored outside Git, such as under ignored `hf_models/`.

For the no-historical-SOH formal comparison, baseline inputs are `capacity_summary`, `capacity_delta`, `internal_resistance`, `charge_time`, and `cycle_ratio`; targets remain future SOH. TimeCMA prompt embeddings and Time-LLM config text must describe these observable features rather than historical SOH.

## 5. Ablation Design

The original ablation suite toggles the main design choices:

- IC/DV battery maps
- Hankel maps
- derivative maps
- dynamic graph attention
- domain structural edges
- report prompt
- cross-modal fusion
- unified decoder versus separate heads

The runner is `bstalignment/run_ablation_suite.py`. It writes one run directory per ablation and an `ablation_summary.csv` table.

For v2, the ablation suite must be updated. Keep the original map/graph ablations, but add:

- `no_numeric_history`
- `no_multi_cycle_raw`
- `single_cycle_raw`
- `no_text_gate`
- `no_semantic_alignment`
- `no_align_loss`
- `absolute_step_decoder`

These ablations should evaluate the new architecture under the no-historical-SOH input contract, not the stopped old pipeline. The next full run should write to `runs/full_hf_v2_nosoh` to avoid overwriting old results and to avoid reusing cache files generated with historical SOH.

## 6. Training Workflow

Typical battery workflow:

1. Place or download datasets under `bstalignment/data/`.
2. Preprocess CALCE/XJTU into processed `.npz` files.
3. Optionally precompute the battery graph cache.
4. Train GraphReportTS on MIT, CALCE, and XJTU.
5. Train official baselines on the same datasets and horizon.
6. Run ablations for each dataset.
7. Use generated metrics and figures under `runs/` for analysis.

`runs/` is ignored because it contains logs, checkpoints, figures, cached tensors, and metrics generated by experiments.

Next v2 no-historical-SOH workflow:

1. Verify the no-SOH leakage tests locally or in the remote `graphreport` environment.
2. Sync code to the remote server after the stopped old pipeline is no longer running.
3. Retrain official baselines because their sequence inputs no longer contain historical SOH.
4. Train the new main model on MIT, CALCE, and XJTU.
5. Run v2-specific ablations on the new main model.
6. Compare future-only metrics across the no-SOH baselines, main model, and ablations.

Remote status at the moment this report was updated:

- The previous official baseline comparison is complete for all 18 dataset/model combinations, but those results used historical SOH inputs and are now treated as an SOH-available upper-bound/reference rather than the formal no-SOH comparison.
- The first v2 pipeline under `runs/full_hf_v2` was stopped after discovering historical SOH in `history_features` and baseline inputs.
- The corrected pipeline should run under `runs/full_hf_v2_nosoh` with local HF weights from `hf_models/distilbert-base-uncased`, `hf_models/openai-community__gpt2`, and `hf_models/google-bert__bert-base-uncased`.
- If training resources allow after the no-SOH run, add an optional strict `no_capacity_or_QD` sensitivity experiment to separate raw-map/text contributions from the capacity proxy.

## 7. Cleanup Status

Early local prototype code based on one-cycle MIT summary features has been removed from the active project. The remaining code is organized around GraphReportTS, official baseline adapters, data preprocessing, inference, visualization, and ablation experiments.

## 8. Repository Boundary

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

## 9. Memory-Safety Notes For Future Codex Sessions

Do not collapse the v2 design into a simpler single-cycle model. The following modules are required for the next implementation:

- `multi_cycle_maps [B, 32, C_map, H, W]`
- shared `GraphMapEncoder` applied per cycle
- first-version `InterCycleTemporalEncoder` over `[B, 32, D]`
- `NumericHistoryEncoder` over `[B, 32, F]`
- future-only metric computation
- relative-step decoder
- learnable text gate with gate logging
- weak or disabled alignment option
- TimeCMA-style token-aware semantic retrieval/alignment
- v2-specific ablation suite

The first `InterCycleTemporalEncoder` is deliberately modest. It is not the final ceiling. If resources allow after the first v2 run, test full patch-level cross-cycle attention over `[B, 32, N_patch, D]`, or a factorized design with intra-cycle patch graph attention followed by inter-cycle patch/channel attention.
