# Battery Cloud Complete Comparison Design

## Scope

Deploy the GitHub project `sleepdog1234356/GraphReportTS.git` to the connected cloud server and run the battery complete-comparison workflow from the data disk only.

The workflow includes:

- Battery datasets: MIT, CALCE, and XJTU.
- Main model: GraphReportTS.
- Ablation suite: all battery ablations exposed by `bstalignment.run_ablation_suite`.
- Baselines: PatchTST, iTransformer, TimeCMA, TimesNet, and DLinear.
- Optional API-heavy baseline handling: avoid paid or remote LLM APIs; use a local HuggingFace model only when an official baseline requires a language model and can be adapted safely.

The workflow excludes the general long-term forecasting datasets for this phase.

## Remote Layout

All remote work happens under the cloud data disk:

```text
/root/autodl-tmp/GraphReportTS
```

The repository owns code, scripts, data, external baseline repositories, logs, and results below that root:

```text
/root/autodl-tmp/GraphReportTS/
  bstalignment/data/
    mit/
    raw/battery/calce/
    raw/battery/xjtu/
    processed/battery/calce/
    processed/battery/xjtu/
  external/
    patchtst/
    itransformer/
    timecma/
    timesnet/
    dlinear/
  scripts/
  runs/
```

No training, preprocessing, or baseline output should intentionally write to `/root` except for conda package caches or unavoidable tool metadata.

## Environment

Use the existing server Miniconda installation at `/root/miniconda3` and create a project-specific environment:

```text
conda environment: graphreport
python: 3.10
torch: CUDA 12.x wheel compatible with the RTX 4090 and driver 580.105.08
```

The first preference is a PyTorch CUDA 12 wheel. If a specific wheel channel fails, choose the nearest officially available CUDA 12 wheel supported by the installed driver. Validate the environment before any data work:

```text
python imports torch
torch.cuda.is_available() is true
GPU name resolves to RTX 4090
project imports succeed
```

## Data Acquisition

Use official or paper-linked sources first:

- MIT/Stanford/TRI battery data from the official matr.io release or an equivalent paper-linked mirror.
- CALCE battery data from the UMD CALCE Battery Research Group release.
- XJTU battery aging data from its paper-linked Zenodo, author repository, or official mirror.

Downloaded archives and extracted raw files stay under:

```text
bstalignment/data/mit
bstalignment/data/raw/battery/calce
bstalignment/data/raw/battery/xjtu
```

If a dataset source requires manual login or a license gate, record the exact URL, expected file names, and destination path, then continue with datasets that can be fetched non-interactively.

## Preprocessing

Implement preprocessing scripts for every raw battery dataset used by formal training.

MIT:

- Prefer existing `batch1.pkl`, `batch2.pkl`, and `batch3.pkl` files when available.
- If only MATLAB files are available, rebuild pkl files with the existing MIT build scripts, after verifying they preserve raw per-cycle current, voltage, temperature, time, and summary fields.
- Formal GraphReportTS training must not rely on `--allow_summary_fallback`.

CALCE and XJTU:

- Convert raw files into one `.npz` file per cell under:

```text
bstalignment/data/processed/battery/calce/<cell_id>.npz
bstalignment/data/processed/battery/xjtu/<cell_id>.npz
```

- Each processed file must contain:

```text
cycle_id [N]
soh [N]
current [N, L]
voltage [N, L]
temperature [N, L]
capacity [N, L] or time [N, L] for capacity integration
```

- Resampling length should match the project default unless the raw dataset forces a different stable length.
- Preprocessing should be deterministic and re-runnable.
- Add lightweight validation that loads each processed dataset through `BatteryRawGraphDataset` for train, validation, and test splits.

## Baseline Integration

Clone official baseline repositories into `external/` when available:

- PatchTST from its official repository.
- iTransformer from THUML's official repository.
- TimeCMA from the official paper repository.
- TimesNet through THUML Time-Series-Library.
- DLinear through LTSF-Linear or an equivalent official implementation.

Use official training entry points where they can consume the generated battery tabular or sequence data with minimal adaptation. Add thin project-side adapters when necessary. The adapters should:

- Reuse the same train, validation, and test cell splits as GraphReportTS where practical.
- Read from processed battery artifacts rather than duplicating raw parsing logic.
- Write metrics to `runs/baselines/<dataset>/<model>/`.
- Avoid changing third-party source code unless the change is small, documented, and isolated.

## Local Model Substitute For API Baselines

No external LLM API key is required for this phase.

If a baseline offers an API-backed LLM path, disable it or replace it with a local HuggingFace model. The preferred fallback is a small local model already compatible with the baseline. If adapting an API-heavy baseline would become a separate research project, mark that baseline path as skipped with a clear log entry and keep the non-API baselines running.

GraphReportTS itself uses either `distilbert-base-uncased` through HuggingFace or the existing `--no_hf_text` fallback.

## Verification

Before launching long jobs:

1. Clone or update the repository under `/root/autodl-tmp/GraphReportTS`.
2. Build and activate the `graphreport` conda environment.
3. Verify CUDA PyTorch works.
4. Verify MIT, CALCE, and XJTU dataset objects can instantiate.
5. Run one short GraphReportTS smoke command per dataset.
6. Run one short baseline smoke command per enabled baseline.
7. Run one short ablation command or dry-run command if supported.

The long workflow may start only after these checks pass or a skipped component is explicitly recorded as non-blocking.

## Long-Running Execution

Create an ordered shell runner:

```text
scripts/run_all_battery_nohup.sh
```

It runs:

1. Main GraphReportTS training for MIT, CALCE, and XJTU.
2. Baseline training for PatchTST, iTransformer, TimeCMA, TimesNet, and DLinear on the same battery datasets where supported.
3. Battery ablation suite for the selected GraphReportTS datasets.

Each command appends to a timestamped log under `runs/logs/`. The top-level invocation is:

```bash
cd /root/autodl-tmp/GraphReportTS
nohup bash scripts/run_all_battery_nohup.sh > runs/full_battery_pipeline.nohup.log 2>&1 &
```

The script must be restart-aware enough to skip completed outputs when a run config and final metrics already exist.

## Risks And Handling

The server data disk is 50 GB. Raw MIT files, processed files, baseline repositories, and logs can approach that limit. Large archives should be deleted after verified extraction unless they are required for reproducibility.

Some official datasets may require web login or manual license acceptance. Those steps cannot be bypassed. When that happens, record exact instructions and continue with accessible datasets.

Third-party baseline dependencies can conflict. Prefer separate lightweight setup scripts or isolated installation steps for baselines that cannot share the main `graphreport` environment.

Long full training may exceed a single interactive Codex session. The final approved workflow uses `nohup` and writes logs so the server continues after the local client closes.
