#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/root/autodl-tmp/GraphReportTS}"
EPOCHS="${EPOCHS:-80}"
BATCH_SIZE="${BATCH_SIZE:-32}"
PRED_LEN="${PRED_LEN:-20}"
cd "$ROOT"
mkdir -p runs/logs
PY="/root/miniconda3/bin/conda run -n graphreport python"

for dataset in mit calce xjtu; do
  out="runs/graph_report_ts/battery/${dataset}"
  if [ -f "$out/test_metrics.json" ]; then
    echo "skip completed GraphReportTS $dataset"
    continue
  fi
  $PY -m bstalignment.train_graph_report \
    --variant battery \
    --dataset "$dataset" \
    --data_root bstalignment/data \
    --out_dir runs/graph_report_ts \
    --pred_len "$PRED_LEN" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --device cuda \
    --no_hf_text 2>&1 | tee "runs/logs/main_${dataset}.log"
done
