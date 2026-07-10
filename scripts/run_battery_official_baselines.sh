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
TRAINING_STRATEGY_VERSION="v3-source-profiles-main-adaptive-fixed-horizon-train-scale-batch64"
cd "$ROOT"
CONTROL_PY="${CONTROL_PY:-python}"
$CONTROL_PY -m bstalignment.battery_protocol validate-formal-protocol \
  --observed-cycles "$INPUT_LEN" \
  --prediction-cycles "$PRED_LEN" \
  --batch-size "$BATCH_SIZE" \
  --stage baseline \
  --context "Formal official baseline runner"
mkdir -p "$OUT_ROOT/logs"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
PY="${PY:-python -u}"

has_matching_run_metadata() {
  local out="$1"
  $CONTROL_PY -m bstalignment.battery_protocol run-config-matches \
    --config "$out/run_config.json" \
    --training-strategy-version "$TRAINING_STRATEGY_VERSION" \
    --stage baseline
}

is_completed_current_strategy() {
  local out="$1"
  [ -f "$out/test_metrics.json" ] && has_matching_run_metadata "$out"
}

for dataset in mit calce xjtu; do
  for model in $BASELINE_MODELS; do
    out="${OUT_ROOT}/baselines/${dataset}/${model}"
    if [ "$FORCE_RETRAIN" != "1" ] && is_completed_current_strategy "$out"; then
      echo "skip completed official baseline $dataset $model"
      continue
    fi
    FRESH_RUN=0
    if [ "$FORCE_RETRAIN" = "1" ]; then
      rm -rf -- "$out"
      FRESH_RUN=1
    elif [ ! -e "$out" ]; then
      FRESH_RUN=1
    elif ! has_matching_run_metadata "$out"; then
      rm -rf -- "$out"
      FRESH_RUN=1
    fi
    RESUME_ARGS=()
    if [ "$FRESH_RUN" = "1" ]; then
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
