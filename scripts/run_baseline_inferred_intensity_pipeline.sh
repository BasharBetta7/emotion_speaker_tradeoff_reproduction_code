#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
GENERATED_MANIFEST="${GENERATED_MANIFEST:-}"
BENCHMARK_MANIFEST="${BENCHMARK_MANIFEST:-Data/esd_balanced_benchmark_manifest.csv}"
ALPHAS="${ALPHAS:-0.0 0.2 0.4 0.6 0.8 1.0}"
MODEL_VARIANT="${MODEL_VARIANT:-}"
LIMIT="${LIMIT:-}"
STARGAN_ALPHA_ROOT="${STARGAN_ALPHA_ROOT:-experiments/full_set_alpha_variants}"
STARGAN_ALPHA_ROOTS="${STARGAN_ALPHA_ROOTS:-}"
STARGAN_LABELS="${STARGAN_LABELS:-}"
PRESERVE_STARGAN_VARIANT_NAMES="${PRESERVE_STARGAN_VARIANT_NAMES:-0}"

if [[ -z "${GENERATED_MANIFEST}" ]]; then
  echo "GENERATED_MANIFEST is required (CSV/TSV with sample_id + generated_wav)." >&2
  exit 1
fi

manifest_stem="$(basename "${GENERATED_MANIFEST}")"
manifest_stem="${manifest_stem%.*}"
EVAL_OUT_DIR="${EVAL_OUT_DIR:-experiments/baselines/${manifest_stem}_inferred_intensity}"
COMPARE_OUT_DIR="${COMPARE_OUT_DIR:-paper/artifacts/baseline_comparison_${manifest_stem}_inferred_intensity}"

read -r -a ALPHAS_ARRAY <<< "${ALPHAS}"

EVAL_CMD=(
  "${PYTHON_BIN}" scripts/eval_baseline_intensity_sweep.py
  --generated-manifest "${GENERATED_MANIFEST}"
  --benchmark-manifest "${BENCHMARK_MANIFEST}"
  --output-dir "${EVAL_OUT_DIR}"
  --intensity-mode inferred
  --alphas "${ALPHAS_ARRAY[@]}"
)

if [[ -n "${MODEL_VARIANT}" ]]; then
  EVAL_CMD+=(--model-variant "${MODEL_VARIANT}")
fi

if [[ -n "${LIMIT}" ]]; then
  EVAL_CMD+=(--limit "${LIMIT}")
fi

echo "[1/2] Running inferred-intensity sweep eval..."
"${EVAL_CMD[@]}"

echo "[2/2] Building comparison artifacts..."
COMPARE_CMD=(
  "${PYTHON_BIN}" scripts/compare_model_variants.py
  --baseline-eval-dirs "${EVAL_OUT_DIR}"
  --output-dir "${COMPARE_OUT_DIR}"
)

if [[ -n "${STARGAN_ALPHA_ROOTS}" ]]; then
  read -r -a STARGAN_ROOTS_ARRAY <<< "${STARGAN_ALPHA_ROOTS}"
  COMPARE_CMD+=(--stargan-alpha-roots "${STARGAN_ROOTS_ARRAY[@]}")
else
  COMPARE_CMD+=(--stargan-alpha-root "${STARGAN_ALPHA_ROOT}")
fi

if [[ -n "${STARGAN_LABELS}" ]]; then
  read -r -a STARGAN_LABELS_ARRAY <<< "${STARGAN_LABELS}"
  COMPARE_CMD+=(--stargan-labels "${STARGAN_LABELS_ARRAY[@]}")
fi

if [[ "${PRESERVE_STARGAN_VARIANT_NAMES}" == "1" ]]; then
  COMPARE_CMD+=(--preserve-stargan-variant-names)
fi

"${COMPARE_CMD[@]}"

echo "Done."
echo "Sweep outputs: ${EVAL_OUT_DIR}"
echo "Comparison outputs: ${COMPARE_OUT_DIR}"
