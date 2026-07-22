#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${1:-${ANCHOREDGTR_ROOT:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}}"
cd "$ROOT"

mkdir -p \
  data/battery/mit \
  data/battery/raw/xjtu

cd "$ROOT/data/battery/raw/xjtu"
if find . -type f -name "*.mat" | grep -q .; then
  echo "Found local XJTU .mat files under $(pwd); skip Zenodo download."
  exit 0
fi
expected_xjtu_bytes=2438769934
actual_xjtu_bytes=0
if [ -f Battery_Dataset.zip ]; then
  actual_xjtu_bytes="$(stat -c%s Battery_Dataset.zip)"
fi
if [ "$actual_xjtu_bytes" -lt "$expected_xjtu_bytes" ]; then
  wget -c "https://zenodo.org/api/records/10963339/files/Battery%20Dataset.zip/content" -O Battery_Dataset.zip
fi
actual_xjtu_bytes="$(stat -c%s Battery_Dataset.zip)"
if [ "$actual_xjtu_bytes" -lt "$expected_xjtu_bytes" ]; then
  echo "XJTU download is incomplete: $actual_xjtu_bytes / $expected_xjtu_bytes bytes" >&2
  exit 1
fi
if [ ! -d Battery_Dataset ]; then
  mkdir -p Battery_Dataset
  unzip -q Battery_Dataset.zip -d Battery_Dataset
fi

echo "MIT pkl files are expected under $ROOT/data/battery/mit."
echo "If matr.io access is gated, copy batch1.pkl batch2.pkl batch3.pkl manually or from the local workstation."
