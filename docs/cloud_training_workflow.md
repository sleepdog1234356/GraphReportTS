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

## 4. Running The Formal V3 Protocol

The formal v3 entrypoint runs `main -> baselines -> ablations` and writes to `runs/full_hf_v3_training_strategy_nosoh`:

The formal battery input protocol is exactly 32 observed cycles and 20 future-only labels, with historical SOH excluded from inputs. `cycle_ratio` uses train-only dataset-global cycle scaling from the selected training cells, is shared unchanged by all splits and models, and uses no clipping above 1.0.

```bash
mkdir -p runs/full_hf_v3_training_strategy_nosoh/logs
FORCE_RETRAIN=1 bash scripts/run_battery_v3_training_strategy_pipeline.sh "$(pwd)" \
  2>&1 | tee runs/full_hf_v3_training_strategy_nosoh/logs/v3_start.log
```

Run it on the approved server configuration: RTX4090 48GiB, 208 CPU threads, batch size 128, 16 workers for main/cache/ablations, and 8 workers for baselines. Cache workers construct unique cycle maps in bounded parallel batches while the parent process owns deterministic memmap writes and publication. Main, cache, and ablation stages share `runs/cache/battery_graph`; all stages share the v3 output root. This protocol adds no AMP.

Check the pipeline and stage logs:

```bash
tail -n 100 -f runs/full_hf_v3_training_strategy_nosoh/logs/v3_start.log
tail -n 100 -f runs/full_hf_v3_training_strategy_nosoh/logs/pipeline_main.out
tail -n 100 -f runs/full_hf_v3_training_strategy_nosoh/logs/pipeline_baselines.out
tail -n 100 -f runs/full_hf_v3_training_strategy_nosoh/logs/pipeline_ablation.out
```

Check completed runs and live training processes:

```bash
find runs/full_hf_v3_training_strategy_nosoh -name test_metrics.json -print
ps -ef | grep -E 'run_battery_v3|train_graph_report|train_battery_official_baselines|run_ablation_suite' | grep -v grep
```

To resume safely after an interrupted v3 run, retain the same output root and use `FORCE_RETRAIN=0`. The scripts skip only runs with both `test_metrics.json` and a matching v3 `run_config.json`; incomplete runs resume when their current-strategy checkpoint is available.

```bash
FORCE_RETRAIN=0 bash scripts/run_battery_v3_training_strategy_pipeline.sh "$(pwd)" \
  2>&1 | tee -a runs/full_hf_v3_training_strategy_nosoh/logs/v3_resume.log
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
