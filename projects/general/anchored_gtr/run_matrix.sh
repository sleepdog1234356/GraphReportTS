#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ANCHOREDGTR_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data/general}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/artifacts/general/anchored_gtr/runs}"
TEXT_MODEL="${TEXT_MODEL:-$ROOT/hf_models/distilbert-base-uncased}"
TEXT_CACHE_ROOT="${TEXT_CACHE_ROOT:-$ROOT/data/general/cache/distilbert_hidden}"
PROVENANCE="${PROVENANCE:-$OUTPUT_ROOT/provenance.json}"
HORIZONS="${HORIZONS:-24 36 48 60}"

cd "$ROOT"
mkdir -p "$OUTPUT_ROOT/logs" "$TEXT_CACHE_ROOT"
if [[ ! -f "$PROVENANCE" ]]; then
  "$PYTHON_BIN" -m anchoredgtr.core.provenance \
    --project_root "$ROOT" --external_root "$ROOT/external" --output "$PROVENANCE"
fi

run_cell() {
  local gpu="$1" dataset="$2" horizon="$3"
  local result="$OUTPUT_ROOT/$dataset/H$horizon"
  if find "$result" -path '*/seed-*/result.json' -type f -print -quit 2>/dev/null | grep -q .; then
    printf 'skip completed AnchoredGTR %s H%s\n' "$dataset" "$horizon"
    return 0
  fi
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u -m anchoredgtr.general.train_anchored_gtr \
    --dataset "$dataset" --horizon "$horizon" \
    --data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT" \
    --provenance-manifest "$PROVENANCE" --text-model "$TEXT_MODEL" \
    --text-cache-root "$TEXT_CACHE_ROOT" --device cuda:0 \
    2>&1 | tee "$OUTPUT_ROOT/logs/${dataset}_H${horizon}.log"
}

worker() {
  local gpu="$1"; shift
  local dataset horizon
  for dataset in "$@"; do
    for horizon in $HORIZONS; do
      run_cell "$gpu" "$dataset" "$horizon"
    done
  done
}

worker 0 ETTh1 ETTh2 Weather &
pid0=$!
worker 1 ETTm1 ETTm2 ECL &
pid1=$!
wait "$pid0"
wait "$pid1"
