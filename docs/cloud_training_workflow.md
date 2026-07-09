# Local Editing and Cloud Training Workflow

This guide describes a reproducible workflow for editing the project locally and running heavy experiments on a cloud GPU server.

## 1. Local Setup

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPO_URL>
cd GraphReportTS
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install the PyTorch build that matches the target machine. For example:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Before pushing, check that large generated files are not staged:

```bash
git status
```

Do not commit raw datasets, processed datasets, checkpoints, downloaded HuggingFace weights, external baseline repositories, or `runs/`.

## 2. Cloud Server Setup

On the cloud GPU server:

```bash
git clone <YOUR_GITHUB_REPO_URL>
cd GraphReportTS
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Install a CUDA-compatible PyTorch build for the server GPU.

If the server cannot access HuggingFace reliably, download model weights once into an ignored directory such as `hf_models/`, then pass the local path:

```bash
--text_model hf_models/distilbert-base-uncased
```

For fully offline transformer loading:

```bash
export TRANSFORMERS_OFFLINE=1
```

## 3. Data Placement

Place battery data under:

```text
bstalignment/data/mit
bstalignment/data/raw/battery/calce
bstalignment/data/raw/battery/xjtu
bstalignment/data/processed/battery/calce
bstalignment/data/processed/battery/xjtu
```

The helper script can download public CALCE/XJTU data when network access is available:

```bash
bash scripts/download_battery_data.sh "$(pwd)"
```

Preprocess CALCE/XJTU:

```bash
bash scripts/preprocess_battery_data.sh "$(pwd)"
```

## 4. Running Experiments

Main GraphReportTS battery experiments:

```bash
bash scripts/run_battery_main_full_hf.sh "$(pwd)"
```

Official baselines:

```bash
bash scripts/run_battery_official_baselines.sh "$(pwd)"
```

Ablations:

```bash
bash scripts/run_battery_ablations_full_hf.sh "$(pwd)"
```

You can override runtime settings through environment variables:

```bash
PY="python -u" \
OUT_ROOT=runs/full_hf \
TEXT_MODEL=hf_models/distilbert-base-uncased \
EPOCHS=80 \
BATCH_SIZE=128 \
NUM_WORKERS=8 \
bash scripts/run_battery_main_full_hf.sh "$(pwd)"
```

## 5. Results

Experiment outputs are written under `runs/`, including:

- `test_metrics.json`
- `val_metrics.json`
- `ablation_summary.csv`
- prediction CSV files
- paper-style figures
- checkpoints and logs

These files are ignored by Git. Copy selected metrics or figures into a separate report only when they are ready to publish.
