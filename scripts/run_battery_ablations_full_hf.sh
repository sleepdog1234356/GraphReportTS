#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
CODE_ROOT="${ABLATION_CODE_ROOT:-$(cd "$(dirname "$SCRIPT_PATH")/.." && pwd)}"
ASSET_ROOT="${1:-$CODE_ROOT}"

asset_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s\n' "$ASSET_ROOT/$1" ;;
  esac
}

OUT_ROOT="$(asset_path "${OUT_ROOT:-runs/full_hf_v3_training_strategy_nosoh}")"
GRAPH_CACHE_DIR="$(asset_path "${BATTERY_GRAPH_CACHE_DIR:-runs/cache/battery_graph}")"
SEQUENCE_CACHE_DIR="$(asset_path "${BATTERY_SEQUENCE_CACHE_DIR:-runs/cache/battery_sequence}")"
TEXT_MODEL="$(asset_path "${TEXT_MODEL:-hf_models/distilbert-base-uncased}")"
FULL_REFERENCE_COMMIT="${FULL_REFERENCE_COMMIT:-$(git -C "$ASSET_ROOT" rev-parse HEAD)}"

FORCE_ARGS=()
case "${ABLATION_FORCE_RETRAIN:-0}" in
  0) ;;
  1) FORCE_ARGS=(--force_retrain) ;;
  *) echo "ABLATION_FORCE_RETRAIN must be 0 or 1" >&2; exit 2 ;;
esac

cd "$CODE_ROOT"
python -u -m bstalignment.run_core_ablation_suite \
  --datasets mit calce xjtu \
  --data_root "$ASSET_ROOT/bstalignment/data" \
  --full_result_root "$OUT_ROOT/graph_report_ts/battery" \
  --out_root "$OUT_ROOT/graph_report_core_ablation" \
  --graph_cache_dir "$GRAPH_CACHE_DIR" \
  --sequence_cache_dir "$SEQUENCE_CACHE_DIR" \
  --text_model "$TEXT_MODEL" \
  --batch_size 64 \
  --cache_task_batch_size 128 \
  --num_workers 16 \
  --device cuda \
  --full_reference_commit "$FULL_REFERENCE_COMMIT" \
  "${FORCE_ARGS[@]}"
