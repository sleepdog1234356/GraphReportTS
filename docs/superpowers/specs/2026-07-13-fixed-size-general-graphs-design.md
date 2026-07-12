# Fixed-Size General Forecasting Graphs Design

## Goal

Make General-GraphReportTS accept a graph representation with a fixed shape for any input feature count while retaining all variables for multivariate-to-multivariate forecasting. Freeze the common general protocol at history length 36 and prediction lengths 24, 36, 48, and 60.

## Scope

- General datasets only: ETTm1, ETTm2, ETTh1, ETTh2, ECL, and Weather.
- Keep every raw numeric variable as an input and forecast every variable.
- Keep battery data, battery graph construction, battery prompt construction, and battery baselines unchanged.
- Update GraphReportTS, all six general baseline adapters, prompts, profiles, scripts, configs, and tests to the new horizon set.

## Data Flow

```text
scaled history [B, 36, C]
  ├─ per-variable map construction
  │    [B, C, views, map_h, map_w]
  │          ↓ aggregate across C
  │    [B, 6 × views, map_h, map_w]  (fixed graph input)
  │          ↓
  │      global graph context
  │
  └─ shared per-variable temporal encoder
       [B, C, 36] → [B, C, D]
                 ↓
global graph context + variable token → shared variable query decoder → [B, H, C]
```

For every map view and every graph pixel, aggregate the variable axis with six deterministic statistics: mean, standard deviation, minimum, maximum, 25th percentile, and 75th percentile. Therefore graph input channel count is `6 × number_of_map_views` and does not depend on `C`. No variables are dropped or projected with PCA.

The direct numeric branch is required because the aggregated graph intentionally no longer preserves the identity of individual variables. It encodes each variable's full 36-step scaled history with shared weights and supplies one token per output variable to the decoder. This makes an input-variable permutation produce the corresponding output-variable permutation.

## Prediction Protocol

- `input_len = 36` for GraphReportTS and all baselines.
- `pred_len ∈ {24, 36, 48, 60}`.
- Task mode is `M → M`: `enc_in = dec_in = c_out = C` for all baselines.
- Metrics remain standardized-space, element-weighted MSE and MAE over `[B, H, C]`.
- Train-only per-variable standardization, official temporal split boundaries, and all-variable input policy remain unchanged.

The existing source-specific baseline optimizer/scheduler mechanisms remain unchanged. Since the original official scripts generally provide horizon-specific settings for 96 and above rather than 24/36/48/60, each new short-horizon profile will inherit that dataset/model's source `H=96` architecture and optimization setting, change only `pred_len`, and record this explicitly as a protocol override. No test-set tuning is allowed.

## Prompt Policy

The generic prompt remains based only on the scaled 36-step observed window: frequency, variable count, aggregate statistics, bounded variable summaries, and requested horizon.

The battery-specific `build_battery_prompt` structure is not copied verbatim: it contains chemistry, capacity-degradation, observed SOH, and aging-stage concepts that do not apply to general forecasting and conflict with the no-historical-SOH battery protocol. The transferable principle is structured, observed-history-only reporting. Optional additions are history-window timestamps, lag/periodicity inferred from the observed window, and missingness summaries; no future-derived quantities may be used.

## Validation

- Fixed graph shape is identical for C=1, 7, 21, and 321.
- Aggregated graph values equal independently calculated statistics.
- Variable temporal tokens retain C entries and outputs have `[B, H, C]` for every formal horizon.
- Input-variable permutation produces an equivalently permuted output.
- All six baseline profiles resolve for the four new horizons and explicitly state the inherited H=96 source setting.
- Battery tests remain unchanged and pass.
