#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$(pwd)}"
OUT_ROOT="${OUT_ROOT:-runs/full_hf}"
EPOCHS="${BASELINE_EPOCHS:-80}"
BATCH_SIZE="${BASELINE_BATCH_SIZE:-64}"
NUM_WORKERS="${BASELINE_NUM_WORKERS:-4}"
INPUT_LEN="${INPUT_LEN:-32}"
PRED_LEN="${PRED_LEN:-20}"
HF_GPT2_MODEL="${HF_GPT2_MODEL:-hf_models/openai-community__gpt2}"
HF_BERT_MODEL="${HF_BERT_MODEL:-hf_models/google-bert__bert-base-uncased}"
BASELINE_MODELS="${BASELINE_MODELS:-patchtst itransformer timecma timesnet dlinear time_llm}"
FORCE_RETRAIN="${FORCE_RETRAIN:-0}"
EARLY_STOP_PATIENCE="${BASELINE_EARLY_STOP_PATIENCE:-$EPOCHS}"
cd "$ROOT"
mkdir -p "$OUT_ROOT/logs"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
PY="${PY:-python -u}"

for dataset in mit calce xjtu; do
  for model in $BASELINE_MODELS; do
    out="${OUT_ROOT}/baselines/${dataset}/${model}"
    if [ "$FORCE_RETRAIN" = "1" ]; then
      rm -rf "$out"
    elif [ -f "$out/test_metrics.json" ]; then
      echo "skip completed official baseline $dataset $model"
      continue
    fi
    RUN_BATCH_SIZE="$BATCH_SIZE"
    if [ "$model" = "time_llm" ]; then
      RUN_BATCH_SIZE="${TIME_LLM_BATCH_SIZE:-$BATCH_SIZE}"
    fi
    $PY -m bstalignment.train_battery_official_baselines \
      --model "$model" \
      --dataset "$dataset" \
      --data_root bstalignment/data \
      --out_dir "${OUT_ROOT}/baselines" \
      --external_root external \
      --input_len "$INPUT_LEN" \
      --pred_len "$PRED_LEN" \
      --epochs "$EPOCHS" \
      --batch_size "$RUN_BATCH_SIZE" \
      --num_workers "$NUM_WORKERS" \
      --early_stop_patience "$EARLY_STOP_PATIENCE" \
      --hf_gpt2_model "$HF_GPT2_MODEL" \
      --hf_bert_model "$HF_BERT_MODEL" \
      --device cuda \
      --no_resume 2>&1 | tee "${OUT_ROOT}/logs/baseline_${dataset}_${model}.log"
  done
done
