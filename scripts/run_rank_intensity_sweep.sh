#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${CONFIG_PATH:-Configs/config_emo_intensity_rank.yml}"
EPOCH="${EPOCH:-00025}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-Models/exp3_emo_balanced_holdout_intensity_rank2/epoch_${EPOCH}.pth}"
MODEL_VARIANT="${MODEL_VARIANT:-rank_intensity_epoch_${EPOCH}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-experiments/MODEL_native_intensity_ranker/rank_intensity_sweep_${EPOCH}}"
ALPHAS="${ALPHAS:-0.0 0.2 0.4 0.6 0.8 1.0}"
DEVICE="${DEVICE:-cuda}"
LIMIT="${LIMIT:-}"
SAVE_AUDIO="${SAVE_AUDIO:-0}"
RANK_SEARCH_STEPS="${RANK_SEARCH_STEPS:-101}"

read -r -a ALPHAS_ARRAY <<< "${ALPHAS}"

for alpha in "${ALPHAS_ARRAY[@]}"; do
  out_dir="${OUTPUT_ROOT}/rank_eval_alpha_${alpha}"
  cmd=(
    python scripts/run_balanced_benchmark_eval.py
    --config-path "${CONFIG_PATH}"
    --checkpoint-path "${CHECKPOINT_PATH}"
    --output-dir "${out_dir}"
    --model-variant "${MODEL_VARIANT}"
    --intensity-mode native
    --intensity-control rank
    --rank-search-steps "${RANK_SEARCH_STEPS}"
    --alpha "${alpha}"
    --device "${DEVICE}"
  )
  if [[ -n "${LIMIT}" ]]; then
    cmd+=(--limit "${LIMIT}")
  fi
  if [[ "${SAVE_AUDIO}" == "1" ]]; then
    cmd+=(--save-audio)
  fi
  echo "[rank-eval] alpha=${alpha} out=${out_dir}"
  "${cmd[@]}"
  
done
