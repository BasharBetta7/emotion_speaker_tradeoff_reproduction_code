#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
PRETRAINED_MODE="${PRETRAINED_MODE:-fixed}"   # native | fixed
ALLOW_PROXY_PRETRAINED="${ALLOW_PROXY_PRETRAINED:-0}"
MODEL_VARIANT="${MODEL_VARIANT:-pretrained_emo_stargan}"
BENCHMARK_MANIFEST="${BENCHMARK_MANIFEST:-Data/esd_balanced_benchmark_manifest.csv}"
ALPHAS="${ALPHAS:-0.0 0.2 0.4 0.6 0.8 1.0}"
LIMIT="${LIMIT:-}"

# Native-mode inputs
CONFIG_PATH="${CONFIG_PATH:-Configs/config_emo_balanced_holdout.yml}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
NATIVE_OUTPUT_ROOT="${NATIVE_OUTPUT_ROOT:-experiments/native_variants/${MODEL_VARIANT}_alpha_sweep}"

# Fixed-mode inputs
GENERATED_MANIFEST="${GENERATED_MANIFEST:-}"
FIXED_EVAL_DIR="${FIXED_EVAL_DIR:-experiments/baselines/${MODEL_VARIANT}_fixed_eval}"
INFERRED_EVAL_DIR="${INFERRED_EVAL_DIR:-experiments/baselines/${MODEL_VARIANT}_inferred_eval}"

# Comparison inputs
STARGAN_ALPHA_ROOT="${STARGAN_ALPHA_ROOT:-experiments/full_set_alpha_variants}"
STARGAN_ALPHA_ROOTS="${STARGAN_ALPHA_ROOTS:-}"
STARGAN_LABELS="${STARGAN_LABELS:-}"
PRESERVE_STARGAN_VARIANT_NAMES="${PRESERVE_STARGAN_VARIANT_NAMES:-0}"
COMPARE_OUT_DIR="${COMPARE_OUT_DIR:-paper/artifacts/baseline_comparison_${MODEL_VARIANT}}"

if [[ "${PRETRAINED_MODE}" != "native" && "${PRETRAINED_MODE}" != "fixed" ]]; then
  echo "Unsupported PRETRAINED_MODE='${PRETRAINED_MODE}'. Expected native or fixed." >&2
  exit 1
fi

if [[ "${PRETRAINED_MODE}" == "fixed" && "${ALLOW_PROXY_PRETRAINED}" != "1" ]]; then
  echo "Refusing fixed-output pretrained baseline: set ALLOW_PROXY_PRETRAINED=1 to allow proxy inferred-intensity evaluation." >&2
  exit 1
fi

run_compare() {
  local compare_cmd=(
    "${PYTHON_BIN}" scripts/compare_model_variants.py
    --baseline-eval-dirs "$@"
    --output-dir "${COMPARE_OUT_DIR}"
  )

  if [[ -n "${STARGAN_ALPHA_ROOTS}" ]]; then
    read -r -a roots_array <<< "${STARGAN_ALPHA_ROOTS}"
    compare_cmd+=(--stargan-alpha-roots "${roots_array[@]}")
  else
    compare_cmd+=(--stargan-alpha-root "${STARGAN_ALPHA_ROOT}")
  fi

  if [[ -n "${STARGAN_LABELS}" ]]; then
    read -r -a labels_array <<< "${STARGAN_LABELS}"
    compare_cmd+=(--stargan-labels "${labels_array[@]}")
  fi

  if [[ "${PRESERVE_STARGAN_VARIANT_NAMES}" == "1" ]]; then
    compare_cmd+=(--preserve-stargan-variant-names)
  fi

  "${compare_cmd[@]}"
}

if [[ "${PRETRAINED_MODE}" == "native" ]]; then
  if [[ -z "${CHECKPOINT_PATH}" ]]; then
    echo "CHECKPOINT_PATH is required when PRETRAINED_MODE=native." >&2
    exit 1
  fi

  native_cmd=(
    scripts/run_native_intensity_variant_sweep.sh
    )

  env \
    PYTHON_BIN="${PYTHON_BIN}" \
    CONFIG_PATH="${CONFIG_PATH}" \
    CHECKPOINT_PATH="${CHECKPOINT_PATH}" \
    VARIANT_NAME="${MODEL_VARIANT}" \
    MANIFEST_PATH="${BENCHMARK_MANIFEST}" \
    ALPHAS="${ALPHAS}" \
    OUTPUT_ROOT="${NATIVE_OUTPUT_ROOT}" \
    LIMIT="${LIMIT}" \
    "${native_cmd[@]}"

  run_compare "${NATIVE_OUTPUT_ROOT}"
  echo "Native pretrained baseline completed."
  echo "Native output: ${NATIVE_OUTPUT_ROOT}"
  echo "Comparison output: ${COMPARE_OUT_DIR}"
  exit 0
fi

if [[ -z "${GENERATED_MANIFEST}" ]]; then
  echo "GENERATED_MANIFEST is required when PRETRAINED_MODE=fixed." >&2
  exit 1
fi

mkdir -p "${FIXED_EVAL_DIR}" "${INFERRED_EVAL_DIR}"

fixed_cmd=(
  "${PYTHON_BIN}" scripts/eval_generated_audio_manifest.py
  --generated-manifest "${GENERATED_MANIFEST}"
  --benchmark-manifest "${BENCHMARK_MANIFEST}"
  --output-dir "${FIXED_EVAL_DIR}"
  --model-variant "${MODEL_VARIANT}"
  --intensity-mode fixed
)
if [[ -n "${LIMIT}" ]]; then
  fixed_cmd+=(--limit "${LIMIT}")
fi
"${fixed_cmd[@]}"

read -r -a ALPHAS_ARRAY <<< "${ALPHAS}"
inferred_cmd=(
  "${PYTHON_BIN}" scripts/eval_baseline_intensity_sweep.py
  --generated-manifest "${GENERATED_MANIFEST}"
  --benchmark-manifest "${BENCHMARK_MANIFEST}"
  --output-dir "${INFERRED_EVAL_DIR}"
  --model-variant "${MODEL_VARIANT}"
  --intensity-mode inferred
  --alphas "${ALPHAS_ARRAY[@]}"
)
if [[ -n "${LIMIT}" ]]; then
  inferred_cmd+=(--limit "${LIMIT}")
fi
"${inferred_cmd[@]}"

run_compare "${FIXED_EVAL_DIR}" "${INFERRED_EVAL_DIR}"

echo "Fixed-output pretrained baseline completed under proxy mode."
echo "Fixed eval output: ${FIXED_EVAL_DIR}"
echo "Inferred eval output: ${INFERRED_EVAL_DIR}"
echo "Comparison output: ${COMPARE_OUT_DIR}"
