#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
GENERATED_MANIFEST="${GENERATED_MANIFEST:-experiments/baselines/emostargan_full_infer/generated_manifest.csv}"
BENCHMARK_MANIFEST="${BENCHMARK_MANIFEST:-Data/esd_balanced_benchmark_manifest.csv}"
MODEL_VARIANT="${MODEL_VARIANT:-emostargan_pretrained}"
ALPHAS="${ALPHAS:-0.0 0.2 0.4 0.6 0.8 1.0}"
EVAL_LIMIT="${EVAL_LIMIT:-}"

STARGAN_ALPHA_ROOT="${STARGAN_ALPHA_ROOT:-experiments/full_set_alpha_variants}"
STARGAN_ALPHA_ROOTS="${STARGAN_ALPHA_ROOTS:-}"
STARGAN_LABELS="${STARGAN_LABELS:-}"
PRESERVE_STARGAN_VARIANT_NAMES="${PRESERVE_STARGAN_VARIANT_NAMES:-0}"

FIXED_EVAL_DIR="${FIXED_EVAL_DIR:-experiments/baselines/emostargan_eval_full}"
INFERRED_EVAL_DIR="${INFERRED_EVAL_DIR:-experiments/baselines/emostargan_inferred_full}"

FIXED_COMPARE_DIR="${FIXED_COMPARE_DIR:-paper/artifacts/baseline_comparison/emostargan_full_fixed}"
INFERRED_COMPARE_DIR="${INFERRED_COMPARE_DIR:-paper/artifacts/baseline_comparison/emostargan_full_inferred}"
COMBINED_COMPARE_DIR="${COMBINED_COMPARE_DIR:-paper/artifacts/baseline_comparison/emostargan_full_combined}"

RUN_FIXED="${RUN_FIXED:-1}"
RUN_INFERRED="${RUN_INFERRED:-1}"
RUN_COMBINED_COMPARE="${RUN_COMBINED_COMPARE:-1}"

if [[ ! -f "${GENERATED_MANIFEST}" ]]; then
  echo "Missing GENERATED_MANIFEST: ${GENERATED_MANIFEST}" >&2
  exit 1
fi

if [[ "${RUN_FIXED}" == "1" ]]; then
  echo "[1/3] Emo-StarGAN fixed-point evaluation"
  fixed_cmd=(
    "${PYTHON_BIN}" scripts/eval_generated_audio_manifest.py
    --generated-manifest "${GENERATED_MANIFEST}"
    --benchmark-manifest "${BENCHMARK_MANIFEST}"
    --output-dir "${FIXED_EVAL_DIR}"
    --model-variant "${MODEL_VARIANT}"
    --intensity-mode fixed
  )
  if [[ -n "${EVAL_LIMIT}" ]]; then
    fixed_cmd+=(--limit "${EVAL_LIMIT}")
  fi
  "${fixed_cmd[@]}"

  "${PYTHON_BIN}" scripts/compare_model_variants.py \
    --stargan-alpha-root "${STARGAN_ALPHA_ROOT}" \
    --baseline-eval-dirs "${FIXED_EVAL_DIR}" \
    --output-dir "${FIXED_COMPARE_DIR}"
fi

if [[ "${RUN_INFERRED}" == "1" ]]; then
  echo "[2/3] Emo-StarGAN inferred-intensity alpha sweep"
  env \
    PYTHON_BIN="${PYTHON_BIN}" \
    GENERATED_MANIFEST="${GENERATED_MANIFEST}" \
    BENCHMARK_MANIFEST="${BENCHMARK_MANIFEST}" \
    ALPHAS="${ALPHAS}" \
    MODEL_VARIANT="${MODEL_VARIANT}" \
    EVAL_OUT_DIR="${INFERRED_EVAL_DIR}" \
    COMPARE_OUT_DIR="${INFERRED_COMPARE_DIR}" \
    STARGAN_ALPHA_ROOT="${STARGAN_ALPHA_ROOT}" \
    STARGAN_ALPHA_ROOTS="${STARGAN_ALPHA_ROOTS}" \
    STARGAN_LABELS="${STARGAN_LABELS}" \
    PRESERVE_STARGAN_VARIANT_NAMES="${PRESERVE_STARGAN_VARIANT_NAMES}" \
    LIMIT="${EVAL_LIMIT}" \
    scripts/run_baseline_inferred_intensity_pipeline.sh
fi

if [[ "${RUN_COMBINED_COMPARE}" == "1" ]]; then
  echo "[3/3] Combined fixed + inferred comparison"
  baseline_dirs=()
  [[ -f "${FIXED_EVAL_DIR}/overall_metrics.csv" ]] && baseline_dirs+=("${FIXED_EVAL_DIR}")
  [[ -f "${INFERRED_EVAL_DIR}/overall_metrics.csv" ]] && baseline_dirs+=("${INFERRED_EVAL_DIR}")

  if [[ ${#baseline_dirs[@]} -eq 0 ]]; then
    echo "No baseline outputs found for combined comparison." >&2
    exit 1
  fi

  compare_cmd=(
    "${PYTHON_BIN}" scripts/compare_model_variants.py
    --baseline-eval-dirs "${baseline_dirs[@]}"
    --output-dir "${COMBINED_COMPARE_DIR}"
  )

  if [[ -n "${STARGAN_ALPHA_ROOTS}" ]]; then
    read -r -a roots_arr <<< "${STARGAN_ALPHA_ROOTS}"
    compare_cmd+=(--stargan-alpha-roots "${roots_arr[@]}")
  else
    compare_cmd+=(--stargan-alpha-root "${STARGAN_ALPHA_ROOT}")
  fi

  if [[ -n "${STARGAN_LABELS}" ]]; then
    read -r -a labels_arr <<< "${STARGAN_LABELS}"
    compare_cmd+=(--stargan-labels "${labels_arr[@]}")
  fi

  if [[ "${PRESERVE_STARGAN_VARIANT_NAMES}" == "1" ]]; then
    compare_cmd+=(--preserve-stargan-variant-names)
  fi

  "${compare_cmd[@]}"
fi

echo "Done."
echo "Fixed eval: ${FIXED_EVAL_DIR}"
echo "Inferred eval: ${INFERRED_EVAL_DIR}"
echo "Combined compare: ${COMBINED_COMPARE_DIR}"
