#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/root/autodl-tmp/GraphReportTS}"
EPOCHS="${BASELINE_EPOCHS:-50}"
BATCH_SIZE="${BASELINE_BATCH_SIZE:-64}"
NUM_WORKERS="${BASELINE_NUM_WORKERS:-4}"
INPUT_LEN="${INPUT_LEN:-32}"
PRED_LEN="${PRED_LEN:-20}"
cd "$ROOT"
mkdir -p runs/logs
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
PY="/root/miniconda3/bin/conda run --no-capture-output -n graphreport python -u"

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
      --num_workers "$NUM_WORKERS" \
      --device cuda 2>&1 | tee "runs/logs/baseline_${dataset}_${model}.log"
  done
done
