#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/root/autodl-tmp/GraphReportTS}"
EPOCHS="${BASELINE_EPOCHS:-50}"
BATCH_SIZE="${BASELINE_BATCH_SIZE:-64}"
INPUT_LEN="${INPUT_LEN:-32}"
PRED_LEN="${PRED_LEN:-20}"
cd "$ROOT"
mkdir -p runs/logs
PY="/root/miniconda3/bin/conda run -n graphreport python"

for dataset in mit calce xjtu; do
  for model in patchtst itransformer timecma timesnet dlinear; do
    out="runs/baselines/${dataset}/${model}"
    if [ -f "$out/test_metrics.json" ]; then
      echo "skip completed baseline $dataset $model"
      continue
    fi
    $PY -m bstalignment.train_battery_baselines \
      --model "$model" \
      --dataset "$dataset" \
      --data_root bstalignment/data \
      --out_dir runs/baselines \
      --input_len "$INPUT_LEN" \
      --pred_len "$PRED_LEN" \
      --epochs "$EPOCHS" \
      --batch_size "$BATCH_SIZE" \
      --device cuda 2>&1 | tee "runs/logs/baseline_${dataset}_${model}.log"
  done
done
