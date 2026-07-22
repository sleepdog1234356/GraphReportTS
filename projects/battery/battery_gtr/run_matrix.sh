#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${GRAPHREPORTTS_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/artifacts/battery/battery_gtr/runs}"
TEXT_MODEL="${TEXT_MODEL:-$ROOT/hf_models/distilbert-base-uncased}"
TEXT_CACHE_ROOT="${TEXT_CACHE_ROOT:-$ROOT/data/battery/cache/distilbert_hidden}"
PROVENANCE="${PROVENANCE:-$OUTPUT_ROOT/provenance.json}"

cd "$ROOT"
mkdir -p "$OUTPUT_ROOT/logs" "$TEXT_CACHE_ROOT"
if [[ ! -f "$PROVENANCE" ]]; then
  "$PYTHON_BIN" -m bstalignment.v2.provenance \
    --project_root "$ROOT" --external_root "$ROOT/external" --output "$PROVENANCE"
fi

run_dataset() {
  local gpu="$1" dataset="$2"
  local cache="$ROOT/data/battery/cache/features/$dataset"
  local result="$OUTPUT_ROOT/$dataset/seed-42/result.json"
  if [[ -f "$result" ]]; then
    printf 'skip completed BatteryGTR %s\n' "$dataset"
    return 0
  fi
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u -m bstalignment.battery.train_battery_gtr \
    --dataset "$dataset" --cache_dir "$cache" --output "$OUTPUT_ROOT" \
    --text_model "$TEXT_MODEL" --provenance_manifest "$PROVENANCE" \
    --prompt_mode sensor_only --epochs 80 --patience 20 \
    --batch_size 256 --num_workers 8 --prefetch_factor 2 \
    --core_lr 1e-3 --semantic_lr 3e-4 --weight_decay 1e-4 \
    --amp bf16 --device cuda:0 --precompute_text \
    --text_cache_root "$TEXT_CACHE_ROOT/$dataset" \
    2>&1 | tee "$OUTPUT_ROOT/logs/${dataset}.log"
}

run_dataset 0 mit &
pid0=$!
run_dataset 1 xjtu &
pid1=$!
wait "$pid0"
wait "$pid1"
