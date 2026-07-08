#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/root/autodl-tmp/GraphReportTS}"
cd "$ROOT"
PY="/root/miniconda3/bin/conda run --no-capture-output -n graphreport python -u"

calce_count="$(find bstalignment/data/processed/battery/calce -name "*.npz" 2>/dev/null | wc -l)"
xjtu_count="$(find bstalignment/data/processed/battery/xjtu -name "*.npz" 2>/dev/null | wc -l)"
if [ "${FORCE_PREPROCESS:-0}" = "1" ] || [ "$calce_count" -lt 4 ] || [ "$xjtu_count" -lt 55 ]; then
  $PY -m bstalignment.preprocess_battery_data \
    --dataset all \
    --data_root bstalignment/data \
    --summary runs/preprocess_battery_summary.json
else
  echo "Found processed CALCE=$calce_count and XJTU=$xjtu_count; skip preprocessing."
fi

$PY - <<'PY'
from bstalignment.data_battery_raw import BatteryRawGraphDataset
for dataset in ["mit", "calce", "xjtu"]:
    for split in ["train", "val", "test"]:
        ds = BatteryRawGraphDataset(dataset_name=dataset, data_root="bstalignment/data", split=split, max_horizon=2, max_cycles=5)
        print(dataset, split, len(ds))
        if len(ds):
            item = ds[0]
            print(" ", item["maps"].shape, item["y"].shape, item["cell_id"])
PY
