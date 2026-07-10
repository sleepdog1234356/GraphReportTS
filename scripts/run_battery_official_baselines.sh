#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$(pwd)}"
OUT_ROOT="${OUT_ROOT:-runs/full_hf_v3_training_strategy_nosoh}"
BATCH_SIZE="${BASELINE_BATCH_SIZE:-128}"
NUM_WORKERS="${BASELINE_NUM_WORKERS:-8}"
INPUT_LEN="${INPUT_LEN:-32}"
PRED_LEN="${PRED_LEN:-20}"
HF_GPT2_MODEL="${HF_GPT2_MODEL:-hf_models/openai-community__gpt2}"
HF_BERT_MODEL="${HF_BERT_MODEL:-hf_models/google-bert__bert-base-uncased}"
BASELINE_MODELS="${BASELINE_MODELS:-patchtst itransformer timecma timesnet dlinear time_llm}"
FORCE_RETRAIN="${FORCE_RETRAIN:-0}"
TRAINING_STRATEGY_VERSION="v3-source-profiles-main-adaptive"
cd "$ROOT"
mkdir -p "$OUT_ROOT/logs"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
PY="${PY:-python -u}"

has_current_strategy_version() {
  local out="$1"
  [ -f "$out/run_config.json" ] && \
    grep -Eq "\"training_strategy_version\"[[:space:]]*:[[:space:]]*\"${TRAINING_STRATEGY_VERSION}\"" "$out/run_config.json"
}

is_completed_current_strategy() {
  local out="$1"
  [ -f "$out/test_metrics.json" ] && has_current_strategy_version "$out"
}

for dataset in mit calce xjtu; do
  for model in $BASELINE_MODELS; do
    out="${OUT_ROOT}/baselines/${dataset}/${model}"
    if [ "$FORCE_RETRAIN" != "1" ] && is_completed_current_strategy "$out"; then
      echo "skip completed official baseline $dataset $model"
      continue
    fi
    RESUME_ARGS=()
    if [ "$FORCE_RETRAIN" = "1" ] || ! has_current_strategy_version "$out"; then
      RESUME_ARGS=(--no_resume)
    fi
    $PY -m bstalignment.train_battery_official_baselines \
      --model "$model" \
      --dataset "$dataset" \
      --data_root bstalignment/data \
      --out_dir "${OUT_ROOT}/baselines" \
      --external_root external \
      --input_len "$INPUT_LEN" \
      --pred_len "$PRED_LEN" \
      --batch_size "$BATCH_SIZE" \
      --num_workers "$NUM_WORKERS" \
      --hf_gpt2_model "$HF_GPT2_MODEL" \
      --hf_bert_model "$HF_BERT_MODEL" \
      --device cuda \
      "${RESUME_ARGS[@]}" 2>&1 | tee "${OUT_ROOT}/logs/baseline_${dataset}_${model}.log"
  done
done
