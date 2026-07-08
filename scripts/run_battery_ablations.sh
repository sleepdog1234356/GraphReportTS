#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/root/autodl-tmp/GraphReportTS}"
EPOCHS="${ABLATION_EPOCHS:-30}"
BATCH_SIZE="${ABLATION_BATCH_SIZE:-32}"
PRED_LEN="${PRED_LEN:-20}"
cd "$ROOT"
mkdir -p runs/logs
PY="/root/miniconda3/bin/conda run -n graphreport python"

for dataset in mit calce xjtu; do
  summary="runs/graph_report_ablation/battery/${dataset}/ablation_summary.csv"
  if [ -f "$summary" ]; then
    echo "skip completed ablation $dataset"
    continue
  fi
  $PY -m bstalignment.run_ablation_suite \
    --variant battery \
    --dataset "$dataset" \
    --data_root bstalignment/data \
    --out_root runs/graph_report_ablation \
    --pred_len "$PRED_LEN" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --device cuda \
    --no_hf_text 2>&1 | tee "runs/logs/ablation_${dataset}.log"
done
