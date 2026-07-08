#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/root/autodl-tmp/GraphReportTS}"
cd "$ROOT"

mkdir -p \
  bstalignment/data/mit \
  bstalignment/data/raw/battery/calce \
  bstalignment/data/raw/battery/xjtu

for name in CS2_35 CS2_36 CS2_37 CS2_38; do
  cd "$ROOT/bstalignment/data/raw/battery/calce"
  if [ ! -f "${name}.zip" ]; then
    wget -c "https://web.calce.umd.edu/batteries/data/${name}.zip" -O "${name}.zip"
  fi
  if [ ! -d "${name}/${name}" ]; then
    rm -rf "${name}"
    unzip -q "${name}.zip" -d "${name}"
  fi
done

cd "$ROOT/bstalignment/data/raw/battery/xjtu"
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

echo "MIT pkl files are expected under $ROOT/bstalignment/data/mit."
echo "If matr.io access is gated, copy batch1.pkl batch2.pkl batch3.pkl manually or from the local workstation."
