# GraphReportTS Reconstruction Audit

This document records how the code matches the agreed reconstruction plan.

## Implemented Without Placeholder Shortcuts

- Battery and general models share one `GraphReportTS` backbone.
- Raw 1D signals are converted into multi-view 2D maps:
  - Hankel delay maps;
  - first/second derivative maps;
  - optional battery IC/DV maps.
- 2D maps are converted into channel-preserving patch nodes.
- The graph encoder uses:
  - dynamic similarity attention;
  - temporal structural bias;
  - delay structural bias;
  - variable-position structural bias;
  - optional domain-view structural bias.
- The decoder uses a unified step-query head by default.
- Separate current/future heads are available only as an ablation.
- Forecast horizon is not injected into the fusion gate; output length is controlled by decoder step queries.
- Report prompt, cross-modal fusion, Hankel maps, derivative maps, IC/DV maps, dynamic graph, domain edges, and decoder style are all real runnable ablation switches.
- Visualization uses a shared AAAI-style plotting module and saves paper-friendly PNG/PDF figures.

## Data Status

MIT:

- Current code can parse common MIT raw cycle dictionaries.
- Formal GraphReportTS experiments require true raw cycle arrays.
- Summary-derived pseudo-curves are disabled by default and available only through `--allow_summary_fallback` for smoke tests.

CALCE and XJTU:

- Raw data are not included in this repository.
- Processed `.npz` loading is implemented.
- Required processed fields:
  - `cycle_id [N]`
  - `soh [N]`
  - `current [N, L]`
  - `voltage [N, L]`
  - `temperature [N, L]`
  - `capacity [N, L]` or enough `time/current` data to compute capacity.

General datasets:

- TimeCMA-aligned CSV loading is implemented for:
  - `ETTm1`
  - `ETTm2`
  - `ETTh1`
  - `ETTh2`
  - `ECL`
  - `FRED`
  - `ILI`
  - `Weather`

## External Baselines

Official baseline source code is not vendored. `bstalignment.baseline_adapters` provides official repository URLs and can print or execute clone commands for:

- PatchTST
- iTransformer
- TimeCMA
- TimesNet
- DLinear
- Time-LLM

`bstalignment.train_battery_official_baselines` provides adapters that instantiate model definitions from those official repositories after they are cloned under `external/`.

## Local Verification

Completed:

- `python -m compileall bstalignment`
- baseline setup dry-run
- ablation command dry-run

Environment-dependent:

- Full training requires a Python environment with PyTorch and the relevant external baseline repositories.
- full experiment execution, because CALCE/XJTU and general datasets are not yet downloaded.

Cloud execution instructions are in `docs/cloud_training_workflow.md`.

