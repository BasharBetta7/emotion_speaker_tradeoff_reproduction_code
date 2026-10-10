#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_PATH="${CONFIG_PATH:-Configs/config_emo_balanced_holdout.yml}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
VARIANT_NAME="${VARIANT_NAME:-intensity_ranked}"
MANIFEST_PATH="${MANIFEST_PATH:-Data/esd_balanced_benchmark_manifest.csv}"
ALPHAS="${ALPHAS:-0.0 0.2 0.4 0.6 0.8 1.0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
DEVICE="${DEVICE:-}"
LIMIT="${LIMIT:-}"
SAVE_AUDIO="${SAVE_AUDIO:-0}"
SKIP_IF_EXISTS="${SKIP_IF_EXISTS:-1}"

if [[ -z "${CHECKPOINT_PATH}" ]]; then
  echo "CHECKPOINT_PATH is required." >&2
  exit 1
fi

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT_PATH}" >&2
  exit 1
fi

if [[ -z "${VARIANT_NAME}" ]]; then
  VARIANT_NAME="$(basename "${CHECKPOINT_PATH}" .pth)"
fi

if [[ -z "${OUTPUT_ROOT}" ]]; then
  OUTPUT_ROOT="experiments/native_variants/${VARIANT_NAME}_alpha_sweep"
fi

mkdir -p "${OUTPUT_ROOT}"
read -r -a ALPHAS_ARRAY <<< "${ALPHAS}"

for alpha in "${ALPHAS_ARRAY[@]}"; do
  out_dir="${OUTPUT_ROOT}/full_eval_${VARIANT_NAME}_alpha_${alpha}"

  if [[ "${SKIP_IF_EXISTS}" == "1" && -f "${out_dir}/overall_metrics.csv" ]]; then
    echo "[skip] ${out_dir}/overall_metrics.csv already exists"
    continue
  fi

  mkdir -p "${out_dir}"

  eval_cmd=(
    "${PYTHON_BIN}" scripts/run_balanced_benchmark_eval.py
    --config-path "${CONFIG_PATH}"
    --checkpoint-path "${CHECKPOINT_PATH}"
    --manifest-path "${MANIFEST_PATH}"
    --output-dir "${out_dir}"
    --model-variant "${VARIANT_NAME}"
    --intensity-mode native
    --alpha "${alpha}"
  )

  if [[ -n "${DEVICE}" ]]; then
    eval_cmd+=(--device "${DEVICE}")
  fi

  if [[ -n "${LIMIT}" ]]; then
    eval_cmd+=(--limit "${LIMIT}")
  fi

  if [[ "${SAVE_AUDIO}" == "1" ]]; then
    eval_cmd+=(--save-audio)
  fi

  echo "[eval] variant=${VARIANT_NAME} alpha=${alpha}"
  "${eval_cmd[@]}"

  echo "Generating aggregate plots..."
  "${PYTHON_BIN}" "scripts/visualize_results.py" --experiment-dir "${out_dir}"
done

echo "Native intensity sweep completed."
echo "Variant: ${VARIANT_NAME}"
echo "Output root: ${OUTPUT_ROOT}"
