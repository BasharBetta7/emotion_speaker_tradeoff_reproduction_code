#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-experiments/MODEL_native_intensity_ranker/CREMA_D/from_scratch/full_set_alpha_variants}"
CONFIG_PATH="${CONFIG_PATH:-Configs/config_emo_intensity_rank_cremad.yml}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-Models/exp7_emo_balanced_holdout_native_intensity_cremad_from_scratch}"
MANIFEST_PATH="${MANIFEST_PATH:-Data/cremad_speaker_disjoint_benchmark_manifest.csv}"
DEVICE="${DEVICE:-}"
LIMIT="${LIMIT:-}"
SAMPLE_ID="${SAMPLE_ID:-}"
INTENSITY_MODE="${INTENSITY_MODE:-native}"
EPOCH="${EPOCH:-25}"
ALPHAS="${ALPHAS:-0.0 0.2 0.4 0.6 0.8 1.0}"

mkdir -p "${OUTPUT_ROOT}"

EPOCH_PADDED="$(printf "%05d" "${EPOCH}")"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${CHECKPOINT_ROOT}/epoch_${EPOCH_PADDED}.pth}"
MODEL_VARIANT="${MODEL_VARIANT:-epoch_${EPOCH_PADDED}}"

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "ERROR: checkpoint not found: ${CHECKPOINT_PATH}" >&2
  exit 1
fi

read -r -a ALPHA_ARRAY <<< "${ALPHAS}"
for alpha in "${ALPHA_ARRAY[@]}"; do
  OUTPUT_DIR="${OUTPUT_ROOT}/full_eval_epoch_${EPOCH_PADDED}_alpha_${alpha}_$(date +%Y%m%d_%H%M%S)"

  echo "Running evaluation for epoch ${EPOCH_PADDED}, alpha ${alpha}..."

  PYTHON_BIN="${PYTHON_BIN}" \
    CONFIG_PATH="${CONFIG_PATH}" \
    CHECKPOINT_PATH="${CHECKPOINT_PATH}" \
    MANIFEST_PATH="${MANIFEST_PATH}" \
    MODEL_VARIANT="${MODEL_VARIANT}" \
    OUTPUT_DIR="${OUTPUT_DIR}" \
    DEVICE="${DEVICE}" \
    LIMIT="${LIMIT}" \
    SAMPLE_ID="${SAMPLE_ID}" \
    INTENSITY_MODE="${INTENSITY_MODE}" \
    ALPHA="${alpha}" \
    EPOCH="${EPOCH}" \
    bash scripts/run_full_evaluation_pipeline.sh
done

"${PYTHON_BIN}" scripts/compare_alpha_variants.py \
  --exp-dir "${OUTPUT_ROOT}" \
  --out-dir "${OUTPUT_ROOT}"

"${PYTHON_BIN}" scripts/compare_model_variants.py \
  --stargan-alpha-root "${OUTPUT_ROOT}" \
  --output-dir "${OUTPUT_ROOT}"
