#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
TARGETS="${TARGETS:-Angry Happy Sad Surprise}"
PREP_OUTPUT_ROOT="${PREP_OUTPUT_ROOT:-Data/external_baselines}"
MATERIALIZE_AUDIO="${MATERIALIZE_AUDIO:-0}"
LINK_MODE="${LINK_MODE:-symlink}"
SKIP_PREP="${SKIP_PREP:-0}"
STARGAN_ALPHA_ROOT="${STARGAN_ALPHA_ROOT:-experiments/full_set_alpha_variants}"
STARGAN_ALPHA_ROOTS="${STARGAN_ALPHA_ROOTS:-}"
STARGAN_LABELS="${STARGAN_LABELS:-}"
PRESERVE_STARGAN_VARIANT_NAMES="${PRESERVE_STARGAN_VARIANT_NAMES:-0}"

# Evaluation mode:
# - INFER_INTENSITY=0: evaluate fixed generated outputs (single operating point).
# - INFER_INTENSITY=1: infer alpha sweep by source<->generated interpolation.
INFER_INTENSITY="${INFER_INTENSITY:-0}"
BENCHMARK_MANIFEST="${BENCHMARK_MANIFEST:-Data/esd_balanced_benchmark_manifest.csv}"
ALPHAS="${ALPHAS:-0.0 0.2 0.4 0.6 0.8 1.0}"
EVAL_LIMIT="${EVAL_LIMIT:-}"
INFER_ALIGN_MODE="${INFER_ALIGN_MODE:-trim}"
INFER_SAVE_AUDIO="${INFER_SAVE_AUDIO:-0}"
INFER_NO_PEAK_NORM="${INFER_NO_PEAK_NORM:-0}"

# Optional evaluation inputs (if external baselines already generated audio).
CYCLEGAN_GENERATED_MANIFEST="${CYCLEGAN_GENERATED_MANIFEST:-}"
VAWGAN_GENERATED_MANIFEST="${VAWGAN_GENERATED_MANIFEST:-}"
CYCLEGAN_MODEL_VARIANT="${CYCLEGAN_MODEL_VARIANT:-}"
VAWGAN_MODEL_VARIANT="${VAWGAN_MODEL_VARIANT:-}"

CYCLEGAN_EVAL_DIR="${CYCLEGAN_EVAL_DIR:-experiments/baselines/cyclegan_eval}"
VAWGAN_EVAL_DIR="${VAWGAN_EVAL_DIR:-experiments/baselines/vawgan_eval}"
COMPARE_OUT_DIR="${COMPARE_OUT_DIR:-paper/artifacts/baseline_comparison}"

mkdir -p experiments/baselines

echo "[1/3] Preparing baseline data packs..."
if [[ "${SKIP_PREP}" == "1" ]]; then
  echo "Skipping prep because SKIP_PREP=1"
else
  read -r -a TARGETS_ARRAY <<< "${TARGETS}"
  PREP_CMD=(
    "${PYTHON_BIN}" scripts/prepare_external_baseline_data.py
    --output-root "${PREP_OUTPUT_ROOT}"
    --targets "${TARGETS_ARRAY[@]}"
  )
  if [[ "${MATERIALIZE_AUDIO}" == "1" ]]; then
    PREP_CMD+=(--materialize-audio --link-mode "${LINK_MODE}")
  fi
  "${PREP_CMD[@]}"
fi

run_eval_for_manifest() {
  local label="$1"
  local generated_manifest="$2"
  local out_dir="$3"
  local model_variant="$4"

  if [[ -z "${generated_manifest}" ]]; then
    echo "Skipping ${label} eval: generated manifest is empty"
    return
  fi

  if [[ "${INFER_INTENSITY}" == "1" ]]; then
    read -r -a ALPHAS_ARRAY <<< "${ALPHAS}"
    local eval_cmd=(
      "${PYTHON_BIN}" scripts/eval_baseline_intensity_sweep.py
      --generated-manifest "${generated_manifest}"
      --benchmark-manifest "${BENCHMARK_MANIFEST}"
      --output-dir "${out_dir}"
      --intensity-mode inferred
      --align-mode "${INFER_ALIGN_MODE}"
      --alphas "${ALPHAS_ARRAY[@]}"
    )
    if [[ -n "${model_variant}" ]]; then
      eval_cmd+=(--model-variant "${model_variant}")
    fi
    if [[ -n "${EVAL_LIMIT}" ]]; then
      eval_cmd+=(--limit "${EVAL_LIMIT}")
    fi
    if [[ "${INFER_SAVE_AUDIO}" == "1" ]]; then
      eval_cmd+=(--save-audio)
    fi
    if [[ "${INFER_NO_PEAK_NORM}" == "1" ]]; then
      eval_cmd+=(--no-peak-norm)
    fi
    "${eval_cmd[@]}"
  else
    local eval_cmd=(
      "${PYTHON_BIN}" scripts/eval_generated_audio_manifest.py
      --generated-manifest "${generated_manifest}"
      --benchmark-manifest "${BENCHMARK_MANIFEST}"
      --output-dir "${out_dir}"
      --intensity-mode fixed
    )
    if [[ -n "${model_variant}" ]]; then
      eval_cmd+=(--model-variant "${model_variant}")
    fi
    if [[ -n "${EVAL_LIMIT}" ]]; then
      eval_cmd+=(--limit "${EVAL_LIMIT}")
    fi
    "${eval_cmd[@]}"
  fi
}

echo "[2/3] Evaluating generated baseline audio (if manifests are provided)..."
run_eval_for_manifest "CycleGAN" "${CYCLEGAN_GENERATED_MANIFEST}" "${CYCLEGAN_EVAL_DIR}" "${CYCLEGAN_MODEL_VARIANT}"
run_eval_for_manifest "VAWGAN" "${VAWGAN_GENERATED_MANIFEST}" "${VAWGAN_EVAL_DIR}" "${VAWGAN_MODEL_VARIANT}"

echo "[3/3] Building consolidated comparison artifacts..."
BASELINE_DIRS=()
[[ -f "${CYCLEGAN_EVAL_DIR}/overall_metrics.csv" ]] && BASELINE_DIRS+=("${CYCLEGAN_EVAL_DIR}")
[[ -f "${VAWGAN_EVAL_DIR}/overall_metrics.csv" ]] && BASELINE_DIRS+=("${VAWGAN_EVAL_DIR}")

COMPARE_CMD=(
  "${PYTHON_BIN}" scripts/compare_model_variants.py
  --baseline-eval-dirs "${BASELINE_DIRS[@]}"
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
echo "Baseline prep summary: ${PREP_OUTPUT_ROOT}/baseline_data_prep_summary.csv"
echo "Comparison summary: ${COMPARE_OUT_DIR}/summary.md"
