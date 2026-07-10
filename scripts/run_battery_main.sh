#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$(pwd)}"
EPOCHS="${EPOCHS:-80}"
BATCH_SIZE="${BATCH_SIZE:-64}"
CACHE_TASK_BATCH_SIZE="${CACHE_TASK_BATCH_SIZE:-128}"
PRED_LEN="${PRED_LEN:-20}"
HISTORY_LEN="${HISTORY_LEN:-32}"
NUM_WORKERS="${NUM_WORKERS:-16}"
W_ALIGN="${W_ALIGN:-0.001}"
ALIGN_WARMUP_EPOCHS="${ALIGN_WARMUP_EPOCHS:-0}"
USE_GRAPH_CACHE="${USE_BATTERY_GRAPH_CACHE:-1}"
GRAPH_CACHE_DIR="${BATTERY_GRAPH_CACHE_DIR:-runs/cache/battery_graph}"
cd "$ROOT"
CONTROL_PY="${CONTROL_PY:-python}"
$CONTROL_PY -m bstalignment.battery_protocol validate-formal-protocol \
  --observed-cycles "$HISTORY_LEN" \
  --prediction-cycles "$PRED_LEN" \
  --batch-size "$BATCH_SIZE" \
  --stage main \
  --cache-task-batch-size "$CACHE_TASK_BATCH_SIZE" \
  --context "GraphReportTS main runner"
mkdir -p runs/logs
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
PY="${PY:-python -u}"

for dataset in mit calce xjtu; do
  out="runs/graph_report_ts/battery/${dataset}"
  if [ -f "$out/test_metrics.json" ]; then
    echo "skip completed GraphReportTS $dataset"
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
      --batch_size "$CACHE_TASK_BATCH_SIZE" \
      --num_workers "$NUM_WORKERS" \
      --splits train val test
    CACHE_ARGS=(--precomputed_cache_dir "$GRAPH_CACHE_DIR" --require_precomputed_cache)
  fi
  $PY -m bstalignment.train_graph_report \
    --variant battery \
    --dataset "$dataset" \
    --data_root bstalignment/data \
    --out_dir runs/graph_report_ts \
    --pred_len "$PRED_LEN" \
    --history_len "$HISTORY_LEN" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --device cuda \
    --w_align "$W_ALIGN" \
    --align_warmup_epochs "$ALIGN_WARMUP_EPOCHS" \
    "${CACHE_ARGS[@]}" \
    --no_hf_text 2>&1 | tee "runs/logs/main_${dataset}.log"
done
