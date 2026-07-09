#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$(pwd)}"
OUT_ROOT="${OUT_ROOT:-runs/full_hf}"
EPOCHS="${EPOCHS:-80}"
BATCH_SIZE="${BATCH_SIZE:-128}"
PRED_LEN="${PRED_LEN:-20}"
HISTORY_LEN="${HISTORY_LEN:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
W_ALIGN="${W_ALIGN:-0.001}"
ALIGN_WARMUP_EPOCHS="${ALIGN_WARMUP_EPOCHS:-0}"
USE_GRAPH_CACHE="${USE_BATTERY_GRAPH_CACHE:-1}"
GRAPH_CACHE_DIR="${BATTERY_GRAPH_CACHE_DIR:-${OUT_ROOT}/cache/battery_graph}"
TEXT_MODEL="${TEXT_MODEL:-hf_models/distilbert-base-uncased}"
cd "$ROOT"
mkdir -p "$OUT_ROOT/logs"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
PY="${PY:-python -u}"

for dataset in mit calce xjtu; do
  out="${OUT_ROOT}/graph_report_ts/battery/${dataset}"
  if [ -f "$out/test_metrics.json" ]; then
    echo "skip completed full-HF GraphReportTS $dataset"
    continue
  fi
  CACHE_ARGS=()
  if [ "$USE_GRAPH_CACHE" = "1" ]; then
    $PY -m bstalignment.precompute_battery_graph_cache \
      --dataset "$dataset" \
      --data_root bstalignment/data \
      --cache_dir "$GRAPH_CACHE_DIR" \
      --pred_len "$PRED_LEN" \
      --history_len "$HISTORY_LEN" \
      --batch_size "$BATCH_SIZE" \
      --num_workers "$NUM_WORKERS" \
      --splits train val test
    CACHE_ARGS=(--precomputed_cache_dir "$GRAPH_CACHE_DIR" --require_precomputed_cache)
  fi
  $PY -m bstalignment.train_graph_report \
    --variant battery \
    --dataset "$dataset" \
    --data_root bstalignment/data \
    --out_dir "${OUT_ROOT}/graph_report_ts" \
    --pred_len "$PRED_LEN" \
    --history_len "$HISTORY_LEN" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --device cuda \
    --w_align "$W_ALIGN" \
    --align_warmup_epochs "$ALIGN_WARMUP_EPOCHS" \
    --text_model "$TEXT_MODEL" \
    "${CACHE_ARGS[@]}" 2>&1 | tee "${OUT_ROOT}/logs/main_${dataset}.log"
done
