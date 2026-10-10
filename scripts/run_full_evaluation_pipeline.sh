#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"


EPOCH="${EPOCH:-25}"
echo "EPOCH: ${EPOCH}"
EPOCH_PADDED="$(printf "%05d" "${EPOCH}")"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_PATH="${CONFIG_PATH:-Configs/config_emo.yml}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-Models/exp3_emo_balanced_holdout_intensity_rank2/epoch_"${EPOCH_PADDED}".pth}"
MANIFEST_PATH="${MANIFEST_PATH:-Data/esd_balanced_benchmark_manifest.csv}"
MODEL_VARIANT="${MODEL_VARIANT:-$(basename "${CHECKPOINT_PATH}" .pth)}"
ALPHA="${ALPHA:-1.0}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/MODEL_inferred_intensity/ESD/DSST/full_eval_epoch_${EPOCH_PADDED}_alpha_${ALPHA}_$(date +%Y%m%d_%H%M%S)}"
DEVICE="${DEVICE:-}"
LIMIT="${LIMIT:-}"
SAMPLE_ID="${SAMPLE_ID:-}"
INTENSITY_MODE="${INTENSITY_MODE:-inferred}"
SAVE_AUDIO="${SAVE_AUDIO:-0}"


export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# for epoch in 55 60 65 70 75; do
#   CHECKPOINT_PATH="Models/exp3_emo_balanced_holdout_from_scratch/epoch_$(printf "%05d" "${epoch}").pth"
#   if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
#     echo "Warning: Checkpoint not found: ${CHECKPOINT_PATH}" >&2
#     continue
#   fi
  # MODEL_VARIANT="epoch_$(printf "%05d" "${epoch}")"
  # OUTPUT_DIR="experiments/full_eval_${MODEL_VARIANT}_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "${OUTPUT_DIR}"
  echo "Running evaluation for alpha: ${ALPHA}, epoch: ${EPOCH_PADDED}"
  echo "Checkpoint path: ${CHECKPOINT_PATH}"
  echo "Output directory: ${OUTPUT_DIR}"
  echo "manifest path: ${MANIFEST_PATH}"
  echo "Config path: ${CONFIG_PATH}"
  echo "Model variant: ${MODEL_VARIANT}"
  EVAL_CMD=(
    "${PYTHON_BIN}" "scripts/run_balanced_benchmark_eval.py"
    --config-path "${CONFIG_PATH}"
    --checkpoint-path "${CHECKPOINT_PATH}"
    --manifest-path "${MANIFEST_PATH}"
    --output-dir "${OUTPUT_DIR}"
    --model-variant "${MODEL_VARIANT}"
    --alpha "${ALPHA}"
    --intensity-mode "${INTENSITY_MODE}"
  )

  if [[ -n "${DEVICE}" ]]; then
    EVAL_CMD+=(--device "${DEVICE}")
  fi

  if [[ -n "${LIMIT}" ]]; then
    EVAL_CMD+=(--limit "${LIMIT}")
  fi

  if [[ "${SAVE_AUDIO}" == "1" ]]; then
    EVAL_CMD+=(--save-audio)
  fi

  echo "[1/3] Running benchmark evaluation..."
  "${EVAL_CMD[@]}"

  echo "[2/3] Generating aggregate plots..."
  "${PYTHON_BIN}" "scripts/visualize_results.py" --experiment-dir "${OUTPUT_DIR}"

  METRICS_CSV="${OUTPUT_DIR}/per_sample_metrics.csv"
  if [[ ! -f "${METRICS_CSV}" ]]; then
    echo "Missing metrics file: ${METRICS_CSV}" >&2
    exit 1
  fi

  if [[ -z "${SAMPLE_ID}" ]]; then
    SAMPLE_ID="$(tail -n +2 "${METRICS_CSV}" | shuf -n 1 | cut -d, -f1)"
  fi

  echo "${SAMPLE_ID}" > "${OUTPUT_DIR}/random_sample_id.txt"
  echo
  echo "Pipeline completed."
  echo "Output directory: ${OUTPUT_DIR}"
  echo "Random sample ID: ${SAMPLE_ID}"
  echo "Aggregate plots: ${OUTPUT_DIR}/plots"
  echo "Mel triplet: ${OUTPUT_DIR}/random_sample_mel_triplet.png"
