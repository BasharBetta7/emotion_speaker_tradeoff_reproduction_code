#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-experiments/MODEL_native_intensity_ranker/ablation/speaker_contrastive_loss/lambda_0.1/ESD/DSST/full_set_alpha_variants}"
CONFIG_PATH="${CONFIG_PATH:-Configs/config_emo_native_intensity_contrastive_loss.yml}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-Models/exp10_native_contrastive_loss_0.1}"
MANIFEST_PATH="${MANIFEST_PATH:-Data/esd_balanced_benchmark_manifest_different_speaker_same_text_3texts.csv}"
DEVICE="${DEVICE:-}"
LIMIT="${LIMIT:-}"
SAMPLE_ID="${SAMPLE_ID:-}"
INTENSITY_MODE="${INTENSITY_MODE:-native}"
EPOCH="${EPOCH:-50}"
ALPHAS="${ALPHAS:-0.0 0.2 0.4 0.6 0.8 1.0}"
SAVE_AUDIO="${SAVE_AUDIO:-0}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_COMPARE="${RUN_COMPARE:-1}"
RUN_STYLE_VIZ="${RUN_STYLE_VIZ:-1}"
STYLE_VIZ_INPUT_CSV="${STYLE_VIZ_INPUT_CSV:-Data/esd_balanced_train.csv}"
STYLE_VIZ_OUTPUT_DIR="${STYLE_VIZ_OUTPUT_DIR:-${OUTPUT_ROOT}/style_embedding_viz_epoch_$(printf "%05d" "${EPOCH}")}"
STYLE_VIZ_METHODS="${STYLE_VIZ_METHODS:-pca}"
STYLE_VIZ_DIMS="${STYLE_VIZ_DIMS:-2}"
STYLE_VIZ_MAX_SAMPLES="${STYLE_VIZ_MAX_SAMPLES:-}"
STYLE_VIZ_SAMPLES_PER_EMOTION="${STYLE_VIZ_SAMPLES_PER_EMOTION:-}"
STYLE_VIZ_SAMPLES_PER_SPEAKER="${STYLE_VIZ_SAMPLES_PER_SPEAKER:-}"
STYLE_VIZ_BATCH_SIZE="${STYLE_VIZ_BATCH_SIZE:-1}"
STYLE_VIZ_RANDOM_SEED="${STYLE_VIZ_RANDOM_SEED:-42}"

mkdir -p "${OUTPUT_ROOT}"

EPOCH_PADDED="$(printf "%05d" "${EPOCH}")"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${CHECKPOINT_ROOT}/epoch_${EPOCH_PADDED}.pth}"
MODEL_VARIANT="${MODEL_VARIANT:-epoch_${EPOCH_PADDED}}"

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "ERROR: checkpoint not found: ${CHECKPOINT_PATH}" >&2
  exit 1
fi

run_evaluation() {
  local alpha="$1"
  local output_dir="${OUTPUT_ROOT}/full_eval_epoch_${EPOCH_PADDED}_alpha_${alpha}_$(date +%Y%m%d_%H%M%S)"

  echo "Running evaluation for epoch ${EPOCH_PADDED}, alpha ${alpha}..."

  PYTHON_BIN="${PYTHON_BIN}" \
    CONFIG_PATH="${CONFIG_PATH}" \
    CHECKPOINT_PATH="${CHECKPOINT_PATH}" \
    MANIFEST_PATH="${MANIFEST_PATH}" \
    MODEL_VARIANT="${MODEL_VARIANT}" \
    OUTPUT_DIR="${output_dir}" \
    DEVICE="${DEVICE}" \
    LIMIT="${LIMIT}" \
    SAMPLE_ID="${SAMPLE_ID}" \
    INTENSITY_MODE="${INTENSITY_MODE}" \
    ALPHA="${alpha}" \
    EPOCH="${EPOCH}" \
    SAVE_AUDIO="${SAVE_AUDIO}" \
    bash scripts/run_full_evaluation_pipeline.sh
}

if [[ "${RUN_EVAL}" == "1" ]]; then
  read -r -a ALPHA_ARRAY <<< "${ALPHAS}"
  for alpha in "${ALPHA_ARRAY[@]}"; do
    run_evaluation "${alpha}"
  done
fi

run_comparisons() {
  "${PYTHON_BIN}" scripts/compare_alpha_variants.py \
    --exp-dir "${OUTPUT_ROOT}" \
    --out-dir "${OUTPUT_ROOT}"

  "${PYTHON_BIN}" scripts/compare_model_variants.py \
    --stargan-alpha-root "${OUTPUT_ROOT}" \
    --output-dir "${OUTPUT_ROOT}"
}

if [[ "${RUN_COMPARE}" == "1" ]]; then
  run_comparisons
fi

run_style_viz() {
  local style_cmd=(
    "${PYTHON_BIN}" scripts/visualize_style_embeddings.py
    --config-path "${CONFIG_PATH}"
    --checkpoint-path "${CHECKPOINT_PATH}"
    --input-csv "${STYLE_VIZ_INPUT_CSV}"
    --output-dir "${STYLE_VIZ_OUTPUT_DIR}"
    --methods ${STYLE_VIZ_METHODS}
    --dims ${STYLE_VIZ_DIMS}
    --random-seed "${STYLE_VIZ_RANDOM_SEED}"
  )

  if [[ -n "${STYLE_VIZ_MAX_SAMPLES}" ]]; then
    style_cmd+=(--max-samples "${STYLE_VIZ_MAX_SAMPLES}")
  fi
  if [[ -n "${STYLE_VIZ_SAMPLES_PER_EMOTION}" ]]; then
    style_cmd+=(--samples-per-emotion "${STYLE_VIZ_SAMPLES_PER_EMOTION}")
  fi
  if [[ -n "${STYLE_VIZ_SAMPLES_PER_SPEAKER}" ]]; then
    style_cmd+=(--samples-per-speaker "${STYLE_VIZ_SAMPLES_PER_SPEAKER}")
  fi
  if [[ -n "${STYLE_VIZ_BATCH_SIZE}" ]]; then
    style_cmd+=(--batch-size "${STYLE_VIZ_BATCH_SIZE}")
  fi
  if [[ -n "${DEVICE}" ]]; then
    style_cmd+=(--device "${DEVICE}")
  fi

  "${style_cmd[@]}"
}

if [[ "${RUN_STYLE_VIZ}" == "1" ]]; then
  run_style_viz
fi
