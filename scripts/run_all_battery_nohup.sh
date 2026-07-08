#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/GraphReportTS}"
cd "$ROOT"
mkdir -p runs/logs

echo "[$(date -Is)] clone official baseline sources"
bash scripts/clone_battery_baselines.sh "$ROOT" 2>&1 | tee runs/logs/00_clone_baselines.log

echo "[$(date -Is)] download battery data"
bash scripts/download_battery_data.sh "$ROOT" 2>&1 | tee runs/logs/00_download_data.log

echo "[$(date -Is)] preprocess battery data"
bash scripts/preprocess_battery_data.sh "$ROOT" 2>&1 | tee runs/logs/01_preprocess.log

echo "[$(date -Is)] train GraphReportTS battery models"
bash scripts/run_battery_main.sh "$ROOT" 2>&1 | tee runs/logs/02_main.log

echo "[$(date -Is)] train battery baselines"
bash scripts/run_battery_baselines.sh "$ROOT" 2>&1 | tee runs/logs/03_baselines.log

echo "[$(date -Is)] run battery ablations"
bash scripts/run_battery_ablations.sh "$ROOT" 2>&1 | tee runs/logs/04_ablations.log

echo "[$(date -Is)] battery pipeline complete"
