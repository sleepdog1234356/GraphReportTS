#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/root/autodl-tmp/GraphReportTS}"
cd "$ROOT"
PY="/root/miniconda3/bin/conda run -n graphreport python"

$PY -m bstalignment.preprocess_battery_data \
  --dataset all \
  --data_root bstalignment/data \
  --summary runs/preprocess_battery_summary.json

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
