#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$(pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
cd "$ROOT"

declare -A urls=(
  [ETTm1]="https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/ETT-small/ETTm1.csv"
  [ETTm2]="https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/ETT-small/ETTm2.csv"
  [ETTh1]="https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/ETT-small/ETTh1.csv"
  [ETTh2]="https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/ETT-small/ETTh2.csv"
  [ECL]="https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/electricity/electricity.csv"
  [Weather]="https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/weather/weather.csv"
)
records=()
while IFS=$'\t' read -r name destination checksum; do
  records+=("$name"$'\t'"$destination"$'\t'"$checksum")
done < <("$PYTHON_BIN" -c '
import json
with open("configs/general_forecasting/datasets.yaml", encoding="utf-8") as stream:
    datasets = json.load(stream)["datasets"]
for item in datasets:
    print(item["name"], item["raw_path"], item["raw_sha256"], sep="\t")
')

actual_sha256() {
  sha256sum "$1" | awk '{print $1}'
}

# Validate every pre-existing raw file before changing any local data.
for record in "${records[@]}"; do
  IFS=$'\t' read -r name destination checksum <<< "$record"
  if [[ -f "$destination" ]] && [[ "$(actual_sha256 "$destination")" != "$checksum" ]]; then
    echo "Raw checksum mismatch for $name: $destination" >&2
    exit 1
  fi
done

for record in "${records[@]}"; do
  IFS=$'\t' read -r name destination checksum <<< "$record"
  if [[ -f "$destination" ]]; then
    continue
  fi
  if [[ -z "${urls[$name]:-}" ]]; then
    echo "No official download URL configured for $name" >&2
    exit 1
  fi
  mkdir -p "$(dirname "$destination")"
  partial="${destination}.part"
  curl --fail --location --continue-at - --output "$partial" "${urls[$name]}"
  if [[ "$(actual_sha256 "$partial")" != "$checksum" ]]; then
    echo "Downloaded checksum mismatch for $name" >&2
    exit 1
  fi
  mv "$partial" "$destination"
done

"$PYTHON_BIN" -m bstalignment.prepare_general_data \
  --config configs/general_forecasting/experiment_matrix.yaml \
  --output-root bstalignment/data/processed/general
