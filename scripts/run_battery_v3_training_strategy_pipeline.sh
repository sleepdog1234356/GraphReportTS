#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$(pwd)}"
OUT_ROOT="${OUT_ROOT:-runs/full_hf_v3_training_strategy_nosoh}"
BATCH_SIZE="${BATCH_SIZE:-64}"
ABLATION_BATCH_SIZE="${ABLATION_BATCH_SIZE:-64}"
BASELINE_BATCH_SIZE="${BASELINE_BATCH_SIZE:-128}"
CACHE_TASK_BATCH_SIZE="${CACHE_TASK_BATCH_SIZE:-128}"
PRED_LEN="${PRED_LEN:-20}"
HISTORY_LEN="${HISTORY_LEN:-32}"
INPUT_LEN="${INPUT_LEN:-32}"
NUM_WORKERS="${NUM_WORKERS:-16}"
BASELINE_NUM_WORKERS="${BASELINE_NUM_WORKERS:-8}"
USE_GRAPH_CACHE="${USE_BATTERY_GRAPH_CACHE:-1}"
GRAPH_CACHE_DIR="${BATTERY_GRAPH_CACHE_DIR:-runs/cache/battery_graph}"
TEXT_MODEL="${TEXT_MODEL:-hf_models/distilbert-base-uncased}"
HF_GPT2_MODEL="${HF_GPT2_MODEL:-hf_models/openai-community__gpt2}"
HF_BERT_MODEL="${HF_BERT_MODEL:-hf_models/google-bert__bert-base-uncased}"
FORCE_RETRAIN="${FORCE_RETRAIN:-1}"
cd "$ROOT"
CONTROL_PY="${CONTROL_PY:-python}"
$CONTROL_PY -m bstalignment.battery_protocol validate-formal-protocol \
  --observed-cycles "$HISTORY_LEN" \
  --prediction-cycles "$PRED_LEN" \
  --batch-size "$BATCH_SIZE" \
  --stage main \
  --cache-task-batch-size "$CACHE_TASK_BATCH_SIZE" \
  --context "Formal v3 pipeline main stage"
$CONTROL_PY -m bstalignment.battery_protocol validate-formal-protocol \
  --observed-cycles "$HISTORY_LEN" \
  --prediction-cycles "$PRED_LEN" \
  --batch-size "$ABLATION_BATCH_SIZE" \
  --stage ablation \
  --cache-task-batch-size "$CACHE_TASK_BATCH_SIZE" \
  --context "Formal v3 pipeline ablation stage"
$CONTROL_PY -m bstalignment.battery_protocol validate-formal-protocol \
  --observed-cycles "$INPUT_LEN" \
  --prediction-cycles "$PRED_LEN" \
  --batch-size "$BASELINE_BATCH_SIZE" \
  --stage baseline \
  --context "Formal v3 pipeline baseline stage"
mkdir -p "$OUT_ROOT/logs"

export OUT_ROOT
export BATCH_SIZE
export ABLATION_BATCH_SIZE
export BASELINE_BATCH_SIZE
export CACHE_TASK_BATCH_SIZE
export PRED_LEN
export HISTORY_LEN
export INPUT_LEN
export NUM_WORKERS
export BASELINE_NUM_WORKERS
export USE_BATTERY_GRAPH_CACHE="$USE_GRAPH_CACHE"
export BATTERY_GRAPH_CACHE_DIR="$GRAPH_CACHE_DIR"
export TEXT_MODEL
export HF_GPT2_MODEL
export HF_BERT_MODEL
export FORCE_RETRAIN
export ABLATION_FORCE_RETRAIN="${ABLATION_FORCE_RETRAIN:-0}"
export CONTROL_PY
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

bash scripts/run_battery_main_full_hf.sh "$ROOT" 2>&1 | tee "$OUT_ROOT/logs/pipeline_main.out"
bash scripts/run_battery_official_baselines.sh "$ROOT" 2>&1 | tee "$OUT_ROOT/logs/pipeline_baselines.out"
bash scripts/run_battery_ablations_full_hf.sh "$ROOT" 2>&1 | tee "$OUT_ROOT/logs/pipeline_ablation.out"
