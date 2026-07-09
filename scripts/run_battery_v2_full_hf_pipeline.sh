#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$(pwd)}"
OUT_ROOT="${OUT_ROOT:-runs/full_hf_v2_nosoh}"
EPOCHS="${EPOCHS:-80}"
ABLATION_EPOCHS="${ABLATION_EPOCHS:-30}"
BASELINE_EPOCHS="${BASELINE_EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-128}"
ABLATION_BATCH_SIZE="${ABLATION_BATCH_SIZE:-128}"
BASELINE_BATCH_SIZE="${BASELINE_BATCH_SIZE:-64}"
PRED_LEN="${PRED_LEN:-20}"
HISTORY_LEN="${HISTORY_LEN:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BASELINE_NUM_WORKERS="${BASELINE_NUM_WORKERS:-4}"
USE_GRAPH_CACHE="${USE_BATTERY_GRAPH_CACHE:-1}"
GRAPH_CACHE_DIR="${BATTERY_GRAPH_CACHE_DIR:-${OUT_ROOT}/cache/battery_graph}"
TEXT_MODEL="${TEXT_MODEL:-hf_models/distilbert-base-uncased}"
HF_GPT2_MODEL="${HF_GPT2_MODEL:-hf_models/openai-community__gpt2}"
HF_BERT_MODEL="${HF_BERT_MODEL:-hf_models/google-bert__bert-base-uncased}"
W_ALIGN="${W_ALIGN:-0.001}"
ALIGN_WARMUP_EPOCHS="${ALIGN_WARMUP_EPOCHS:-0}"
cd "$ROOT"
mkdir -p "$OUT_ROOT/logs"

export OUT_ROOT
export EPOCHS
export ABLATION_EPOCHS
export BASELINE_EPOCHS
export BATCH_SIZE
export ABLATION_BATCH_SIZE
export BASELINE_BATCH_SIZE
export PRED_LEN
export HISTORY_LEN
export NUM_WORKERS
export BASELINE_NUM_WORKERS
export USE_BATTERY_GRAPH_CACHE="$USE_GRAPH_CACHE"
export BATTERY_GRAPH_CACHE_DIR="$GRAPH_CACHE_DIR"
export TEXT_MODEL
export HF_GPT2_MODEL
export HF_BERT_MODEL
export W_ALIGN
export ALIGN_WARMUP_EPOCHS

bash scripts/run_battery_official_baselines.sh "$ROOT" 2>&1 | tee "$OUT_ROOT/logs/pipeline_baselines.out"
bash scripts/run_battery_main_full_hf.sh "$ROOT" 2>&1 | tee "$OUT_ROOT/logs/pipeline_main.out"
bash scripts/run_battery_ablations_full_hf.sh "$ROOT" 2>&1 | tee "$OUT_ROOT/logs/pipeline_ablation.out"
