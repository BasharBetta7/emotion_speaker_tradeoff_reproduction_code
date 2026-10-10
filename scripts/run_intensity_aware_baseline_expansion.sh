#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_MODE="${RUN_MODE:-smoke}"    # smoke | full | both

# Shared benchmark settings
BENCHMARK_MANIFEST="${BENCHMARK_MANIFEST:-Data/esd_balanced_benchmark_manifest.csv}"
ALPHAS="${ALPHAS:-0.0 0.2 0.4 0.6 0.8 1.0}"
STARGAN_ALPHA_ROOT="${STARGAN_ALPHA_ROOT:-experiments/full_set_alpha_variants}"

# External baseline inputs
RUN_EXTERNAL="${RUN_EXTERNAL:-1}"
CYCLEGAN_GENERATED_MANIFEST="${CYCLEGAN_GENERATED_MANIFEST:-}"
VAWGAN_GENERATED_MANIFEST="${VAWGAN_GENERATED_MANIFEST:-}"
CYCLEGAN_MODEL_VARIANT="${CYCLEGAN_MODEL_VARIANT:-cyclegan_external}"
VAWGAN_MODEL_VARIANT="${VAWGAN_MODEL_VARIANT:-vawgan_external}"

# Native StarGAN speaker-preserving variant
RUN_NATIVE="${RUN_NATIVE:-1}"
NATIVE_CONFIG_PATH="${NATIVE_CONFIG_PATH:-Configs/config_emo_balanced_holdout_spkpreserve_moderate.yml}"
NATIVE_CHECKPOINT_PATH="${NATIVE_CHECKPOINT_PATH:-}"
NATIVE_VARIANT_NAME="${NATIVE_VARIANT_NAME:-stargan_spkpreserve_moderate}"

# Optional pretrained gate stage
RUN_PRETRAINED_GATE="${RUN_PRETRAINED_GATE:-0}"
PRETRAINED_MODE="${PRETRAINED_MODE:-fixed}"
ALLOW_PROXY_PRETRAINED="${ALLOW_PROXY_PRETRAINED:-0}"
PRETRAINED_MODEL_VARIANT="${PRETRAINED_MODEL_VARIANT:-pretrained_emo_stargan}"
PRETRAINED_GENERATED_MANIFEST="${PRETRAINED_GENERATED_MANIFEST:-}"
PRETRAINED_CHECKPOINT_PATH="${PRETRAINED_CHECKPOINT_PATH:-}"
PRETRAINED_CONFIG_PATH="${PRETRAINED_CONFIG_PATH:-Configs/config_emo_balanced_holdout.yml}"

# Smoke/full controls
SMOKE_LIMIT="${SMOKE_LIMIT:-80}"

run_external_stage() {
  local stage_name="$1"
  local eval_limit="$2"

  local fixed_cyc="experiments/baselines/cyclegan_eval_${stage_name}"
  local fixed_vaw="experiments/baselines/vawgan_eval_${stage_name}"
  local inferred_cyc="experiments/baselines/cyclegan_inferred_eval_${stage_name}"
  local inferred_vaw="experiments/baselines/vawgan_inferred_eval_${stage_name}"

  if [[ -z "${CYCLEGAN_GENERATED_MANIFEST}" && -z "${VAWGAN_GENERATED_MANIFEST}" ]]; then
    echo "[external:${stage_name}] skipped (no generated manifests provided)"
    return
  fi

  echo "[external:${stage_name}] fixed-point evaluation"
  env \
    PYTHON_BIN="${PYTHON_BIN}" \
    SKIP_PREP=1 \
    STARGAN_ALPHA_ROOT="${STARGAN_ALPHA_ROOT}" \
    BENCHMARK_MANIFEST="${BENCHMARK_MANIFEST}" \
    INFER_INTENSITY=0 \
    EVAL_LIMIT="${eval_limit}" \
    CYCLEGAN_GENERATED_MANIFEST="${CYCLEGAN_GENERATED_MANIFEST}" \
    VAWGAN_GENERATED_MANIFEST="${VAWGAN_GENERATED_MANIFEST}" \
    CYCLEGAN_MODEL_VARIANT="${CYCLEGAN_MODEL_VARIANT}" \
    VAWGAN_MODEL_VARIANT="${VAWGAN_MODEL_VARIANT}" \
    CYCLEGAN_EVAL_DIR="${fixed_cyc}" \
    VAWGAN_EVAL_DIR="${fixed_vaw}" \
    COMPARE_OUT_DIR="paper/artifacts/baseline_comparison_external_fixed_${stage_name}" \
    scripts/run_external_baseline_pipeline.sh

  echo "[external:${stage_name}] inferred-intensity evaluation"
  env \
    PYTHON_BIN="${PYTHON_BIN}" \
    SKIP_PREP=1 \
    STARGAN_ALPHA_ROOT="${STARGAN_ALPHA_ROOT}" \
    BENCHMARK_MANIFEST="${BENCHMARK_MANIFEST}" \
    INFER_INTENSITY=1 \
    ALPHAS="${ALPHAS}" \
    EVAL_LIMIT="${eval_limit}" \
    CYCLEGAN_GENERATED_MANIFEST="${CYCLEGAN_GENERATED_MANIFEST}" \
    VAWGAN_GENERATED_MANIFEST="${VAWGAN_GENERATED_MANIFEST}" \
    CYCLEGAN_MODEL_VARIANT="${CYCLEGAN_MODEL_VARIANT}" \
    VAWGAN_MODEL_VARIANT="${VAWGAN_MODEL_VARIANT}" \
    CYCLEGAN_EVAL_DIR="${inferred_cyc}" \
    VAWGAN_EVAL_DIR="${inferred_vaw}" \
    COMPARE_OUT_DIR="paper/artifacts/baseline_comparison_external_inferred_${stage_name}" \
    scripts/run_external_baseline_pipeline.sh

  local baseline_dirs=()
  [[ -f "${fixed_cyc}/overall_metrics.csv" ]] && baseline_dirs+=("${fixed_cyc}")
  [[ -f "${fixed_vaw}/overall_metrics.csv" ]] && baseline_dirs+=("${fixed_vaw}")
  [[ -f "${inferred_cyc}/overall_metrics.csv" ]] && baseline_dirs+=("${inferred_cyc}")
  [[ -f "${inferred_vaw}/overall_metrics.csv" ]] && baseline_dirs+=("${inferred_vaw}")

  if [[ ${#baseline_dirs[@]} -gt 0 ]]; then
    "${PYTHON_BIN}" scripts/compare_model_variants.py \
      --stargan-alpha-root "${STARGAN_ALPHA_ROOT}" \
      --baseline-eval-dirs "${baseline_dirs[@]}" \
      --output-dir "paper/artifacts/baseline_comparison_external_combined_${stage_name}"
  fi
}

run_native_stage() {
  local stage_name="$1"
  local eval_limit="$2"

  if [[ -z "${NATIVE_CHECKPOINT_PATH}" ]]; then
    echo "[native:${stage_name}] NATIVE_CHECKPOINT_PATH is required when RUN_NATIVE=1." >&2
    exit 1
  fi

  local native_root="experiments/native_variants/${NATIVE_VARIANT_NAME}_${stage_name}"

  env \
    PYTHON_BIN="${PYTHON_BIN}" \
    CONFIG_PATH="${NATIVE_CONFIG_PATH}" \
    CHECKPOINT_PATH="${NATIVE_CHECKPOINT_PATH}" \
    VARIANT_NAME="${NATIVE_VARIANT_NAME}" \
    MANIFEST_PATH="${BENCHMARK_MANIFEST}" \
    ALPHAS="${ALPHAS}" \
    OUTPUT_ROOT="${native_root}" \
    LIMIT="${eval_limit}" \
    scripts/run_native_intensity_variant_sweep.sh

  "${PYTHON_BIN}" scripts/compare_model_variants.py \
    --stargan-alpha-roots "${STARGAN_ALPHA_ROOT}" "${native_root}" \
    --stargan-labels "starganv2_vc" "${NATIVE_VARIANT_NAME}" \
    --output-dir "paper/artifacts/baseline_comparison_native_${stage_name}"
}

run_pretrained_gate_stage() {
  local stage_name="$1"
  local eval_limit="$2"

  local native_out="experiments/native_variants/${PRETRAINED_MODEL_VARIANT}_${stage_name}"
  local fixed_out="experiments/baselines/${PRETRAINED_MODEL_VARIANT}_fixed_${stage_name}"
  local inferred_out="experiments/baselines/${PRETRAINED_MODEL_VARIANT}_inferred_${stage_name}"

  env \
    PYTHON_BIN="${PYTHON_BIN}" \
    PRETRAINED_MODE="${PRETRAINED_MODE}" \
    ALLOW_PROXY_PRETRAINED="${ALLOW_PROXY_PRETRAINED}" \
    MODEL_VARIANT="${PRETRAINED_MODEL_VARIANT}" \
    BENCHMARK_MANIFEST="${BENCHMARK_MANIFEST}" \
    ALPHAS="${ALPHAS}" \
    LIMIT="${eval_limit}" \
    CONFIG_PATH="${PRETRAINED_CONFIG_PATH}" \
    CHECKPOINT_PATH="${PRETRAINED_CHECKPOINT_PATH}" \
    NATIVE_OUTPUT_ROOT="${native_out}" \
    GENERATED_MANIFEST="${PRETRAINED_GENERATED_MANIFEST}" \
    FIXED_EVAL_DIR="${fixed_out}" \
    INFERRED_EVAL_DIR="${inferred_out}" \
    STARGAN_ALPHA_ROOT="${STARGAN_ALPHA_ROOT}" \
    COMPARE_OUT_DIR="paper/artifacts/baseline_comparison_pretrained_${stage_name}" \
    scripts/run_pretrained_baseline_gate.sh
}

run_combined_comparison() {
  local stage_name="$1"

  local native_root="experiments/native_variants/${NATIVE_VARIANT_NAME}_${stage_name}"
  local baseline_dirs=()
  local stargan_roots=("${STARGAN_ALPHA_ROOT}")
  local stargan_labels=("starganv2_vc")

  [[ -f "experiments/baselines/cyclegan_eval_${stage_name}/overall_metrics.csv" ]] && baseline_dirs+=("experiments/baselines/cyclegan_eval_${stage_name}")
  [[ -f "experiments/baselines/vawgan_eval_${stage_name}/overall_metrics.csv" ]] && baseline_dirs+=("experiments/baselines/vawgan_eval_${stage_name}")
  [[ -f "experiments/baselines/cyclegan_inferred_eval_${stage_name}/overall_metrics.csv" ]] && baseline_dirs+=("experiments/baselines/cyclegan_inferred_eval_${stage_name}")
  [[ -f "experiments/baselines/vawgan_inferred_eval_${stage_name}/overall_metrics.csv" ]] && baseline_dirs+=("experiments/baselines/vawgan_inferred_eval_${stage_name}")

  if [[ -f "${native_root}/full_eval_${NATIVE_VARIANT_NAME}_alpha_1.0/overall_metrics.csv" ]]; then
    stargan_roots+=("${native_root}")
    stargan_labels+=("${NATIVE_VARIANT_NAME}")
  fi

  if [[ ${#baseline_dirs[@]} -eq 0 ]]; then
    echo "[combined:${stage_name}] skipped (no baseline eval dirs found)"
    return
  fi

  "${PYTHON_BIN}" scripts/compare_model_variants.py \
    --stargan-alpha-roots "${stargan_roots[@]}" \
    --stargan-labels "${stargan_labels[@]}" \
    --baseline-eval-dirs "${baseline_dirs[@]}" \
    --output-dir "paper/artifacts/baseline_comparison_intensity_aware_${stage_name}"
}

run_validation_if_available() {
  local stage_name="$1"

  if [[ "${stage_name}" != "full" ]]; then
    echo "[validation:${stage_name}] skipped (validation is enforced in full stage only)"
    return
  fi

  local fixed_manifest="${CYCLEGAN_GENERATED_MANIFEST:-experiments/baselines/cyclegan_generated_smoke.csv}"
  local fixed_per_sample="experiments/baselines/cyclegan_eval_${stage_name}/per_sample_metrics.csv"
  local inferred_per_sample="experiments/baselines/cyclegan_inferred_eval_${stage_name}/per_sample_metrics.csv"

  if [[ ! -f "${fixed_per_sample}" || ! -f "${inferred_per_sample}" || ! -f "${fixed_manifest}" ]]; then
    echo "[validation:${stage_name}] skipped (CycleGAN fixed/inferred artifacts not fully available)"
    return
  fi

  local expected_rows
  expected_rows=$(( $(wc -l < "${fixed_per_sample}") - 1 ))

  "${PYTHON_BIN}" scripts/cyclegan/validate_cyclegan_full_comparison.py \
    --benchmark-manifest "${BENCHMARK_MANIFEST}" \
    --fixed-manifest "${fixed_manifest}" \
    --fixed-per-sample "${fixed_per_sample}" \
    --inferred-per-sample "${inferred_per_sample}" \
    --alphas ${ALPHAS} \
    --expected-benchmark-rows "${expected_rows}" \
    --scope-from-fixed-manifest \
    --output-dir "paper/artifacts/baseline_comparison_validation_${stage_name}"
}

run_stage() {
  local stage_name="$1"
  local eval_limit="$2"

  echo "==== Stage: ${stage_name} ===="

  if [[ "${RUN_EXTERNAL}" == "1" ]]; then
    run_external_stage "${stage_name}" "${eval_limit}"
  fi

  if [[ "${RUN_NATIVE}" == "1" ]]; then
    run_native_stage "${stage_name}" "${eval_limit}"
  fi

  if [[ "${RUN_PRETRAINED_GATE}" == "1" ]]; then
    run_pretrained_gate_stage "${stage_name}" "${eval_limit}"
  fi

  if [[ "${RUN_EXTERNAL}" == "1" || "${RUN_NATIVE}" == "1" || "${RUN_PRETRAINED_GATE}" == "1" ]]; then
    run_combined_comparison "${stage_name}"
  else
    echo "[combined:${stage_name}] skipped (all source stages disabled)"
  fi
  run_validation_if_available "${stage_name}"
}

case "${RUN_MODE}" in
  smoke)
    run_stage "smoke" "${SMOKE_LIMIT}"
    ;;
  full)
    run_stage "full" ""
    ;;
  both)
    run_stage "smoke" "${SMOKE_LIMIT}"
    run_stage "full" ""
    ;;
  *)
    echo "Unsupported RUN_MODE='${RUN_MODE}'. Expected smoke, full, or both." >&2
    exit 1
    ;;
esac

echo "Intensity-aware baseline expansion completed (RUN_MODE=${RUN_MODE})."
