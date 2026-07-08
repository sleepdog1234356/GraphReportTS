#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/root/autodl-tmp/GraphReportTS}"
cd "$ROOT"
mkdir -p external

clone_or_update() {
  local name="$1"
  local url="$2"
  local dir="external/$name"
  if [ -d "$dir/.git" ]; then
    if [ "${UPDATE_BASELINES:-0}" = "1" ]; then
      git -C "$dir" fetch --all --prune || echo "warn: could not update existing $name" >&2
    else
      echo "skip existing $dir"
    fi
  else
    local ok=0
    for attempt in 1 2 3; do
      if git clone --depth 1 "$url" "$dir"; then
        ok=1
        break
      fi
      rm -rf "$dir"
      echo "retry clone $name attempt $attempt" >&2
      sleep 5
    done
    if [ "$ok" != "1" ]; then
      echo "warn: could not clone $name from $url" >&2
    fi
  fi
}

clone_or_update patchtst https://github.com/yuqinie98/PatchTST.git
clone_or_update itransformer https://github.com/thuml/iTransformer.git
clone_or_update timecma https://github.com/ChenxiLiu-HNU/TimeCMA.git
clone_or_update timesnet https://github.com/thuml/Time-Series-Library.git
clone_or_update dlinear https://github.com/cure-lab/LTSF-Linear.git

PYTHON="${PYTHON:-/root/miniconda3/envs/graphreport/bin/python}"
"$PYTHON" -m bstalignment.baseline_adapters --root external
