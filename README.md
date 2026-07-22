# AnchoredGTR and BatteryGTR

This repository contains the two paper-facing main models of the project:

- **AnchoredGTR** is the general multivariate forecasting model. It uses a decomposition-aware sparse graph encoder, frozen DistilBERT prompt encoder, graph-text semantic alignment, and a Ridge-initialized frozen lightweight anchor. The released protocol uses history length 36 and horizons 24/36/48/60.
- **BatteryGTR** is the battery SOH transfer model. It converts 32 consecutive V/I/T cycles into 50 deterministic sensor features plus eight IC/DV residual channels, builds the original multi-scale patch graph, and predicts 20 future SOH values with a direct `BatterySOHHead`. It has no linear/Ridge anchor and does not use historical SOH or absolute cycle index.

## Architecture

AnchoredGTR separates the easily fitted trend from the nonlinear graph-text correction:

```text
multivariate history [B, 36, M]
  ├─ Ridge-initialized frozen lightweight anchor ──────────┐
  ├─ series/context decomposition → sparse graph encoder ┐ │
  └─ sample-specific prompt → frozen DistilBERT → gate ──┴─┴→ residual forecast
```

BatteryGTR keeps the graph-text transfer mechanism but uses the battery-specialized direct head:

```text
32 V/I/T cycles
  → 50 sensor-derived features + 4 IC residual + 4 DV residual
  → [B, 32, 58]
  → multi-scale patch graph + battery operating-context prompt
  → direct BatterySOHHead [B, 20, 1]
```

The detailed AnchoredGTR equations and module definitions are in [`docs/anchored_gtr_method_2026-07-19.md`](docs/anchored_gtr_method_2026-07-19.md).

## Repository layout

```text
anchoredgtr/
  general/                     AnchoredGTR identity, strategy registry, trainer
  battery/                     BatteryGTR identity and trainer
  core/                        shared graph, text, data, loss, and training runtime
projects/
  general/anchored_gtr/        L36 multi-dataset launcher and data preparation
  battery/battery_gtr/         MIT/XJTU launcher and feature-cache preparation
configs/
  general_forecasting/         dataset manifests used by general-data preparation
  gtr/                         shared and battery-main configuration
docs/                          Method description and architecture figure
tests/                         focused main-model and shared-runtime tests
```

Baseline, ablation, optimization-search, result-archive, maintenance, and server-audit code is intentionally excluded from this public main-model tree.

## Installation

Use Python 3.12 and install a PyTorch build matching the local CUDA runtime. A reproducible CUDA 12.8 environment is provided in `environment.yml`:

```bash
conda env create -f environment.yml
conda activate anchoredgtr-py312-cu128
```

Alternatively:

```bash
pip install -r requirements.txt
```

Formal runs require a local DistilBERT checkpoint, supplied either at `hf_models/distilbert-base-uncased` or through the corresponding CLI option. Model weights, datasets, and caches are not stored in Git.

## Data layout

```text
data/general/raw/ETT-small/{ETTh1,ETTh2,ETTm1,ETTm2}.csv
data/general/raw/electricity/electricity.csv
data/general/raw/weather/weather.csv
data/battery/mit/{batch1,batch2,batch3}.pkl
data/battery/raw/xjtu/
data/battery/processed/xjtu/
```

Prepare general data with:

```bash
bash projects/general/anchored_gtr/scripts/prepare_general_data.sh
```

Download/preprocess supported battery inputs with:

```bash
bash projects/battery/battery_gtr/scripts/download_battery_data.sh
bash projects/battery/battery_gtr/scripts/preprocess_battery_data.sh
```

CALCE is outside the current project scope.

## Training

Run one AnchoredGTR cell:

```bash
python -m anchoredgtr.general.train_anchored_gtr \
  --dataset ETTh2 \
  --horizon 24 \
  --provenance-manifest /path/to/provenance.json
```

Run the complete AnchoredGTR L36 matrix on two GPUs:

```bash
bash projects/general/anchored_gtr/run_matrix.sh /path/to/provenance.json
```

Run one BatteryGTR dataset:

```bash
python -m anchoredgtr.battery.train_battery_gtr \
  --dataset mit \
  --cache_dir data/battery/cache/features/mit \
  --output artifacts/battery/battery_gtr/runs
```

Run MIT and XJTU in parallel:

```bash
bash projects/battery/battery_gtr/run_matrix.sh /path/to/provenance.json
```

Both launchers derive the repository root from their own locations and allow data, model, cache, device, and output paths to be overridden.

## Checkpoint compatibility

`BatteryGTR` subclasses the original internal `BatteryGTRCore` implementation without adding parameterized modules. Therefore historical battery-main checkpoints keep the same `state_dict` key layout and load with `strict=True`. Historical result JSON files should keep their original model strings; only new runs emit `model=BatteryGTR`.

## Verification

```bash
python -m pytest tests -q
```

The publication contract verifies model identities, graph variants, anchor presence/absence, launcher portability, prompt/cache behavior, training safeguards, and strict BatteryGTR checkpoint compatibility.
