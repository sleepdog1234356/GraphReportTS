#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$(pwd)}"
OUT_ROOT="${OUT_ROOT:-runs/full_hf}"
EPOCHS="${ABLATION_EPOCHS:-80}"
BATCH_SIZE="${ABLATION_BATCH_SIZE:-128}"
PRED_LEN="${PRED_LEN:-20}"
HISTORY_LEN="${HISTORY_LEN:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
W_ALIGN="${W_ALIGN:-0.001}"
ALIGN_WARMUP_EPOCHS="${ALIGN_WARMUP_EPOCHS:-0}"
FORCE_RETRAIN="${FORCE_RETRAIN:-0}"
EARLY_STOP_PATIENCE="${ABLATION_EARLY_STOP_PATIENCE:-$EPOCHS}"
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
  summary="${OUT_ROOT}/graph_report_ablation/battery/${dataset}/ablation_summary.csv"
  if [ "$FORCE_RETRAIN" = "1" ]; then
    rm -rf "${OUT_ROOT}/graph_report_ablation/battery/${dataset}"
  elif [ -f "$summary" ]; then
    echo "skip completed full-HF ablation $dataset"
    continue
  fi
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
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --device cuda \
    --w_align "$W_ALIGN" \
    --align_warmup_epochs "$ALIGN_WARMUP_EPOCHS" \
    --early_stop_patience "$EARLY_STOP_PATIENCE" \
    --early_stop_min_delta 0 \
    --text_model "$TEXT_MODEL" \
    "${CACHE_ARGS[@]}" 2>&1 | tee "${OUT_ROOT}/logs/ablation_${dataset}.log"
done
