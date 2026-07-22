#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${1:-${GRAPHREPORTTS_ROOT:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
WORKERS="${WORKERS:-16}"
cd "$ROOT"

build_cache() {
  local dataset="$1" source="$2" output="$3"
  if [[ -f "$output/manifest.json" && "${FORCE_PREPROCESS:-0}" != "1" ]]; then
    printf 'reuse battery feature cache %s: %s\n' "$dataset" "$output"
    return 0
  fi
  args=(
    -m bstalignment.v2.precompute_battery
    --dataset "$dataset"
    --data-root "$source"
    --output "$output"
    --workers "$WORKERS"
  )
  if [[ "${FORCE_PREPROCESS:-0}" == "1" ]]; then
    args+=(--overwrite)
  fi
  "$PYTHON_BIN" "${args[@]}"
}

build_cache mit data/battery/mit data/battery/cache/features/mit
build_cache xjtu data/battery/processed/xjtu data/battery/cache/features/xjtu

"$PYTHON_BIN" - <<'PY'
from bstalignment.v2.battery_cache import BatteryFeatureCache

for dataset in ("mit", "xjtu"):
    cache = BatteryFeatureCache.load(f"data/battery/cache/features/{dataset}")
    print(dataset, len(cache.cell_ids), cache.manifest_hash)
PY
