#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-experiments/MODEL_native_intensity_ranker/CREMA_D/finetuned/full_set_epoch_variants}"
CONFIG_PATH="${CONFIG_PATH:-Configs/config_emo_intensity_rank_cremad.yml}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-Models/exp6_emo_balanced_holdout_native_intensity_cremad_finetuned}"
MANIFEST_PATH="${MANIFEST_PATH:-Data/cremad_speaker_disjoint_benchmark_manifest.csv}"
DEVICE="${DEVICE:-}"
LIMIT="${LIMIT:-}"
SAMPLE_ID="${SAMPLE_ID:-}"
INTENSITY_MODE="${INTENSITY_MODE:-native}"
ALPHA="${ALPHA:-1.0}"
EPOCHS="${EPOCHS:-25 30 35 40 45 50}"

mkdir -p "${OUTPUT_ROOT}"

read -r -a EPOCH_ARRAY <<< "${EPOCHS}"
for epoch in "${EPOCH_ARRAY[@]}"; do
  EPOCH_PADDED="$(printf "%05d" "${epoch}")"
  CHECKPOINT_PATH="${CHECKPOINT_ROOT}/epoch_${EPOCH_PADDED}.pth"
  MODEL_VARIANT="${MODEL_VARIANT_PREFIX:-epoch}_${EPOCH_PADDED}"
  OUTPUT_DIR="${OUTPUT_ROOT}/full_eval_epoch_${EPOCH_PADDED}_alpha_${ALPHA}_$(date +%Y%m%d_%H%M%S)"

  if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
    echo "WARNING: checkpoint not found for epoch ${EPOCH_PADDED}: ${CHECKPOINT_PATH}; continuing..." >&2
    continue
  fi

  echo "Running evaluation for epoch ${EPOCH_PADDED}, alpha ${ALPHA}..."

  if ! PYTHON_BIN="${PYTHON_BIN}" \
      CONFIG_PATH="${CONFIG_PATH}" \
      CHECKPOINT_PATH="${CHECKPOINT_PATH}" \
      MANIFEST_PATH="${MANIFEST_PATH}" \
      MODEL_VARIANT="${MODEL_VARIANT}" \
      OUTPUT_DIR="${OUTPUT_DIR}" \
      DEVICE="${DEVICE}" \
      LIMIT="${LIMIT}" \
      SAMPLE_ID="${SAMPLE_ID}" \
      INTENSITY_MODE="${INTENSITY_MODE}" \
      ALPHA="${ALPHA}" \
      EPOCH="${epoch}" \
      bash scripts/run_full_evaluation_pipeline.sh; then
    echo "WARNING: evaluation failed for epoch ${EPOCH_PADDED}; continuing..." >&2
    continue
  fi
done

"${PYTHON_BIN}" scripts/compare_variants.py \
  --exp-dir "${OUTPUT_ROOT}" \
  --out-dir "${OUTPUT_ROOT}"

"${PYTHON_BIN}" scripts/compare_model_variants.py \
  --stargan-alpha-root "${OUTPUT_ROOT}" \
  --output-dir "${OUTPUT_ROOT}"
