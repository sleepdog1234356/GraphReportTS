# Editing Locally with Codex and Training on a Cloud Server

This guide describes the intended workflow after the project is uploaded to GitHub:

1. edit and review code on the local PC with Codex;
2. push changes to GitHub;
3. pull the latest code on a rented cloud GPU server;
4. run heavy GraphReportTS training on the server;
5. copy metrics, figures, and checkpoints back as needed.

## 1. Local PC Setup

Clone the GitHub repository on the PC:

```bash
git clone <YOUR_GITHUB_REPO_URL>
cd battery_stalign_code
```

Open this folder in Codex Desktop. Continue editing code in the same project folder.

Before each cloud training run:

```bash
git status
git add README.md bstalignment docs .gitignore requirements.txt
git commit -m "Update GraphReportTS experiment code"
git push origin main
```

Do not commit raw datasets, checkpoints, `runs/`, `external/`, or downloaded baseline repositories. They are ignored by `.gitignore`.

## 2. Cloud Server Setup

On the cloud GPU server:

```bash
git clone <YOUR_GITHUB_REPO_URL>
cd battery_stalign_code
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Install the PyTorch build that matches the server CUDA version. For example:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

If the server cannot access HuggingFace reliably, use:

```bash
python -m bstalignment.train_graph_report ... --no_hf_text
```

or download the text model once and pass its local path:

```bash
--text_model /path/to/distilbert-base-uncased
```

## 3. Data Placement on the Server

Place raw and processed data under the ignored data folders:

```text
bstalignment/data/mit
bstalignment/data/raw/battery/calce
bstalignment/data/raw/battery/xjtu
bstalignment/data/processed/battery/calce
bstalignment/data/processed/battery/xjtu
bstalignment/data/raw/general/ETTm1
...
```

For CALCE/XJTU, preprocess raw files into `.npz` files under:

```text
bstalignment/data/processed/battery/<dataset>/<cell_id>.npz
```

Required arrays:

```text
cycle_id [N]
soh [N]
current [N, L]
voltage [N, L]
temperature [N, L]
capacity [N, L] optional if time/current are available
time [N, L] optional
```

For general datasets, place CSV files such as:

```text
bstalignment/data/raw/general/ETTm1/ETTm1.csv
```

with one optional timestamp column and numeric variable columns.

## 4. Training Commands

Battery MIT smoke test, only if raw arrays are not available:

```bash
python -m bstalignment.train_graph_report \
  --variant battery \
  --dataset mit \
  --pred_len 20 \
  --allow_summary_fallback \
  --no_hf_text
```

Formal battery training should use raw cycle arrays and should not use `--allow_summary_fallback`:

```bash
python -m bstalignment.train_graph_report \
  --variant battery \
  --dataset mit \
  --pred_len 20 \
  --batch_size 32
```

CALCE and XJTU after preprocessing:

```bash
python -m bstalignment.train_graph_report --variant battery --dataset calce --pred_len 20
python -m bstalignment.train_graph_report --variant battery --dataset xjtu --pred_len 20
```

General TimeCMA-aligned experiments:

```bash
python -m bstalignment.train_graph_report \
  --variant general \
  --dataset ETTm1 \
  --input_len 96 \
  --pred_len 96 \
  --batch_size 32
```

Ablations:

```bash
python -m bstalignment.run_ablation_suite \
  --variant battery \
  --dataset mit \
  --pred_len 20
```

## 5. Running Long Jobs

Use `tmux` so jobs continue after the SSH session disconnects:

```bash
tmux new -s graphreport
python -m bstalignment.train_graph_report --variant battery --dataset mit --pred_len 20
```

Detach:

```text
Ctrl-b d
```

Resume:

```bash
tmux attach -t graphreport
```

## 6. Pulling New Local Changes on the Server

After editing with Codex and pushing:

```bash
git pull origin main
```

If dependencies changed:

```bash
pip install -r requirements.txt
```

## 7. Copying Results Back

From the local PC:

```bash
scp -r user@server:/path/to/battery_stalign_code/runs ./runs_from_server
```

Only commit selected small result tables or paper-ready figures if needed. Keep checkpoints and full run folders out of GitHub.

