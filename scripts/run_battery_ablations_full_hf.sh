#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$(pwd)}"
OUT_ROOT="${OUT_ROOT:-runs/full_hf_v3_training_strategy_nosoh}"
BATCH_SIZE="${ABLATION_BATCH_SIZE:-128}"
PRED_LEN="${PRED_LEN:-20}"
HISTORY_LEN="${HISTORY_LEN:-32}"
NUM_WORKERS="${NUM_WORKERS:-16}"
FORCE_RETRAIN="${FORCE_RETRAIN:-0}"
USE_GRAPH_CACHE="${USE_BATTERY_GRAPH_CACHE:-1}"
GRAPH_CACHE_DIR="${BATTERY_GRAPH_CACHE_DIR:-runs/cache/battery_graph}"
TEXT_MODEL="${TEXT_MODEL:-hf_models/distilbert-base-uncased}"
TRAINING_STRATEGY_VERSION="v3-source-profiles-main-adaptive-fixed-horizon-train-scale"
cd "$ROOT"
mkdir -p "$OUT_ROOT/logs"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
PY="${PY:-python -u}"

CONTROL_ARGS=(--training_strategy_version "$TRAINING_STRATEGY_VERSION")
if [ "$FORCE_RETRAIN" = "1" ]; then
  CONTROL_ARGS+=(--force_retrain)
fi

for dataset in mit calce xjtu; do
  CACHE_ARGS=()
  if [ "$USE_GRAPH_CACHE" = "1" ]; then
    CACHE_ARGS=(--precomputed_cache_dir "$GRAPH_CACHE_DIR" --require_precomputed_cache)
  fi
  $PY -m bstalignment.run_ablation_suite \
    --variant battery \
    --dataset "$dataset" \
    --data_root bstalignment/data \
    --out_root "${OUT_ROOT}/graph_report_ablation" \
    --pred_len "$PRED_LEN" \
    --history_len "$HISTORY_LEN" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --device cuda \
    --text_model "$TEXT_MODEL" \
    "${CONTROL_ARGS[@]}" \
    "${CACHE_ARGS[@]}" 2>&1 | tee "${OUT_ROOT}/logs/ablation_${dataset}.log"
done
