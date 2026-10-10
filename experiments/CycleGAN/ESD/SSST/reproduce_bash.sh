#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DRY_RUN="${DRY_RUN:-0}"

EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-experiments/CycleGAN/ESD/SSST}"
MODEL_VARIANT_PREFIX="${MODEL_VARIANT_PREFIX:-cyclegan_spectrum}"
PAIR_PREFIX="${PAIR_PREFIX:-n2}"
FRONTIER_NAME="${FRONTIER_NAME:-${MODEL_VARIANT_PREFIX}_inferred_frontier_cyclegan_features}"

TARGETS="${TARGETS:-Happy Angry Sad Surprise}"
TARGETS_LOWER="${TARGETS_LOWER:-happy angry sad surprise}"
ALPHAS="${ALPHAS:-0.0 0.2 0.4 0.6 0.8 1.0}"

TRAIN_METADATA="${TRAIN_METADATA:-Data/esd_balanced_train.csv}"
BENCHMARK_MANIFEST="${BENCHMARK_MANIFEST:-Data/esd_balanced_benchmark_manifest_different_speaker_same_text_3texts.csv}"
BENCHMARK_POOL="${BENCHMARK_POOL:-Data/esd_balanced_benchmark_pool.csv}"
DATA_ROOT="${DATA_ROOT:-Data/external_baselines}"

EXTERNAL_ROOT="${EXTERNAL_ROOT:-experiments/external_repos}"
CYCLEGAN_REPO_PATH="${CYCLEGAN_REPO_PATH:-experiments/external_repos/emotional-voice-conversion-with-CycleGAN-and-CWT-for-Spectrum-and-F0}"
CYCLEGAN_CODE_SUBDIR="${CYCLEGAN_CODE_SUBDIR:-Parallel-data-free emotional voice conversion with CycleGAN and CWT}"
CYCLEGAN_CODE_DIR="${CYCLEGAN_CODE_DIR:-${CYCLEGAN_REPO_PATH}/${CYCLEGAN_CODE_SUBDIR}}"
CYCLEGAN_ENV="${CYCLEGAN_ENV:-}"
CYCLEGAN_PYTHON_BIN="${CYCLEGAN_PYTHON_BIN:-python}"

MODEL_ROOT="${MODEL_ROOT:-Models/cyclegan_spectrum_models}"
LOG_ROOT="${LOG_ROOT:-Models/cyclegan_spectrum_logs}"
VALIDATION_OUTPUT_ROOT="${VALIDATION_OUTPUT_ROOT:-${EXPERIMENT_ROOT}/${MODEL_VARIANT_PREFIX}_validation_outputs}"
PACKAGE_ROOT="${PACKAGE_ROOT:-${EXPERIMENT_ROOT}/${MODEL_VARIANT_PREFIX}_benchmark_package}"
CONVERTED_ROOT="${CONVERTED_ROOT:-${EXPERIMENT_ROOT}/${MODEL_VARIANT_PREFIX}_converted_full}"
GENERATED_MANIFEST="${GENERATED_MANIFEST:-${EXPERIMENT_ROOT}/${MODEL_VARIANT_PREFIX}_generated_full.csv}"
FIXED_EVAL_DIR="${FIXED_EVAL_DIR:-${EXPERIMENT_ROOT}/${MODEL_VARIANT_PREFIX}_fixed_eval_full}"
INFERRED_EVAL_DIR="${INFERRED_EVAL_DIR:-${EXPERIMENT_ROOT}/${MODEL_VARIANT_PREFIX}_inferred_eval_full}"
FRONTIER_DIR="${FRONTIER_DIR:-${EXPERIMENT_ROOT}/${FRONTIER_NAME}}"
VALIDATION_DIR="${VALIDATION_DIR:-${FRONTIER_DIR}/validation}"

NUM_EPOCHS="${NUM_EPOCHS:-100}"
MAX_SAMPLES="${MAX_SAMPLES:-1000}"
MAX_FILES_PER_DOMAIN="${MAX_FILES_PER_DOMAIN:-0}"
RANDOM_SEED="${RANDOM_SEED:-0}"
LINK_MODE="${LINK_MODE:-symlink}"
SAMPLE_RATE="${SAMPLE_RATE:-24000}"
ALIGN_MODE="${ALIGN_MODE:-trim}"
EVAL_LIMIT="${EVAL_LIMIT:-}"

RUN_SETUP="${RUN_SETUP:-0}"
RUN_PREP="${RUN_PREP:-1}"
RUN_TRAIN="${RUN_TRAIN:-0}"
RUN_PACKAGE="${RUN_PACKAGE:-1}"
RUN_CONVERT="${RUN_CONVERT:-1}"
RUN_ASSEMBLE="${RUN_ASSEMBLE:-1}"
RUN_FIXED_EVAL="${RUN_FIXED_EVAL:-1}"
RUN_INFER_EVAL="${RUN_INFER_EVAL:-1}"
RUN_PLOT="${RUN_PLOT:-1}"
RUN_ALPHA_PLOTS="${RUN_ALPHA_PLOTS:-1}"
RUN_VALIDATE="${RUN_VALIDATE:-1}"
SKIP_IF_EXISTS="${SKIP_IF_EXISTS:-1}"

print_cmd() {
  local quoted=""
  printf -v quoted "%q " "$@"
  echo "${quoted}"
}

run_cmd() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run] $(print_cmd "$@")"
  else
    echo "[run] $(print_cmd "$@")"
    "$@"
  fi
}

abs_path() {
  if [[ "$1" = /* ]]; then
    echo "$1"
  else
    echo "${REPO_ROOT}/$1"
  fi
}

run_cyclegan_python() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    if [[ -n "${CYCLEGAN_ENV}" ]]; then
      echo "[dry-run] (cd ${REPO_ROOT} && conda run --no-capture-output -n ${CYCLEGAN_ENV} python -u $(print_cmd "$@"))"
    else
      echo "[dry-run] (cd ${REPO_ROOT} && ${CYCLEGAN_PYTHON_BIN} $(print_cmd "$@"))"
    fi
    return
  fi

  echo "[run] CycleGAN env command: $*"
  (
    cd "${REPO_ROOT}"
    if [[ -n "${CYCLEGAN_ENV}" ]]; then
      conda run --no-capture-output -n "${CYCLEGAN_ENV}" python -u "$@"
    else
      "${CYCLEGAN_PYTHON_BIN}" "$@"
    fi
  )
}

require_external_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "Missing required external CycleGAN file: ${path}" >&2
    exit 1
  fi
}

pair_name() {
  echo "${PAIR_PREFIX}$1"
}

read -r -a TARGETS_ARRAY <<< "${TARGETS}"
read -r -a TARGETS_LOWER_ARRAY <<< "${TARGETS_LOWER}"
read -r -a ALPHAS_ARRAY <<< "${ALPHAS}"

DATA_ROOT_ABS="$(abs_path "${DATA_ROOT}")"
MODEL_ROOT_ABS="$(abs_path "${MODEL_ROOT}")"
LOG_ROOT_ABS="$(abs_path "${LOG_ROOT}")"
VALIDATION_OUTPUT_ROOT_ABS="$(abs_path "${VALIDATION_OUTPUT_ROOT}")"
PACKAGE_ROOT_ABS="$(abs_path "${PACKAGE_ROOT}")"
CONVERTED_ROOT_ABS="$(abs_path "${CONVERTED_ROOT}")"

if [[ "${#TARGETS_ARRAY[@]}" -ne "${#TARGETS_LOWER_ARRAY[@]}" ]]; then
  echo "TARGETS and TARGETS_LOWER must have the same number of entries." >&2
  exit 1
fi

if [[ "${RUN_SETUP}" == "1" ]]; then
  echo "[1] Setup external CycleGAN spectrum repo"
  run_cmd env \
    DRY_RUN="${DRY_RUN}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    EXTERNAL_ROOT="${EXTERNAL_ROOT}" \
    CYCLEGAN_REPO_PATH="${CYCLEGAN_REPO_PATH}" \
    scripts/cyclegan/setup_cyclegan_spectrum_repo.sh
fi

if [[ "${RUN_PREP}" == "1" ]]; then
  echo "[2] Prepare ESD CycleGAN training data"
  run_cmd "${PYTHON_BIN}" scripts/prepare_external_baseline_data.py \
    --train-metadata "${TRAIN_METADATA}" \
    --benchmark-pool "${BENCHMARK_POOL}" \
    --benchmark-manifest "${BENCHMARK_MANIFEST}" \
    --output-root "${DATA_ROOT}" \
    --targets "${TARGETS_ARRAY[@]}" \
    --materialize-audio \
    --link-mode "${LINK_MODE}"
fi

if [[ "${DRY_RUN}" != "1" ]]; then
  require_external_file "${CYCLEGAN_CODE_DIR}/train.py"
  require_external_file "${CYCLEGAN_CODE_DIR}/model.py"
  require_external_file "${CYCLEGAN_CODE_DIR}/preprocess.py"
fi

if [[ "${RUN_TRAIN}" == "1" ]]; then
  echo "[3] Train spectrum CycleGAN models"
  for emo in "${TARGETS_LOWER_ARRAY[@]}"; do
    pair="$(pair_name "${emo}")"
    pair_root="${DATA_ROOT_ABS}/cyclegan/${pair}"
    model_dir="${MODEL_ROOT_ABS}/${pair}"
    log_dir="${LOG_ROOT_ABS}/${pair}"
    output_dir="${VALIDATION_OUTPUT_ROOT_ABS}/${pair}"
    if [[ "${DRY_RUN}" != "1" ]]; then
      mkdir -p "${model_dir}" "${log_dir}" "${output_dir}"
    fi

    if [[ "${SKIP_IF_EXISTS}" == "1" && -f "${model_dir}/${pair}.ckpt.index" ]]; then
      echo "[skip] checkpoint already exists: ${model_dir}/${pair}.ckpt.index"
      continue
    fi

    run_cyclegan_python scripts/cyclegan/train_cyclegan_spectrum.py \
      --code-dir "${CYCLEGAN_CODE_DIR}" \
      --train-A-dir "${pair_root}/train_A" \
      --train-B-dir "${pair_root}/train_B" \
      --validation-A-dir "${pair_root}/validation_A" \
      --validation-B-dir "${pair_root}/validation_B" \
      --model-dir "${model_dir}" \
      --model-name "${pair}.ckpt" \
      --random-seed "${RANDOM_SEED}" \
      --output-dir "${output_dir}" \
      --tensorboard-log-dir "${log_dir}" \
      --num-epochs "${NUM_EPOCHS}" \
      --max-samples "${MAX_SAMPLES}" \
      --max-files-per-domain "${MAX_FILES_PER_DOMAIN}"
  done
fi

if [[ "${RUN_PACKAGE}" == "1" ]]; then
  echo "[4] Prepare benchmark conversion package"
  run_cmd "${PYTHON_BIN}" scripts/cyclegan/prepare_cyclegan_benchmark_inputs.py \
    --benchmark-manifest "${BENCHMARK_MANIFEST}" \
    --output-root "${PACKAGE_ROOT}" \
    --converted-root "${CONVERTED_ROOT}" \
    --targets "${TARGETS_ARRAY[@]}" \
    --model-variant-prefix "${MODEL_VARIANT_PREFIX}" \
    --link-mode "${LINK_MODE}" \
    --force
fi

if [[ "${RUN_CONVERT}" == "1" ]]; then
  echo "[5] Convert benchmark sources"
  for emo in "${TARGETS_LOWER_ARRAY[@]}"; do
    pair="$(pair_name "${emo}")"
    run_cyclegan_python scripts/cyclegan/convert_cyclegan_spectrum.py \
      --code-dir "${CYCLEGAN_CODE_DIR}" \
      --model-dir "${MODEL_ROOT_ABS}/${pair}" \
      --model-name "${pair}.ckpt" \
      --data-dir "${PACKAGE_ROOT_ABS}/inputs/${pair}" \
      --conversion-direction A2B \
      --output-dir "${CONVERTED_ROOT_ABS}/${pair}"
  done
fi

if [[ "${RUN_ASSEMBLE}" == "1" ]]; then
  echo "[6] Assemble generated manifest"
  run_cmd "${PYTHON_BIN}" scripts/cyclegan/assemble_cyclegan_fixed_manifest.py \
    --benchmark-manifest "${BENCHMARK_MANIFEST}" \
    --converted-root "${CONVERTED_ROOT}" \
    --output-path "${GENERATED_MANIFEST}" \
    --targets "${TARGETS_ARRAY[@]}" \
    --model-variant-prefix "${MODEL_VARIANT_PREFIX}" \
    --strict
fi

if [[ "${RUN_FIXED_EVAL}" == "1" ]]; then
  echo "[7] Evaluate fixed CycleGAN outputs"
  fixed_cmd=(
    "${PYTHON_BIN}" scripts/eval_generated_audio_manifest.py
    --generated-manifest "${GENERATED_MANIFEST}"
    --benchmark-manifest "${BENCHMARK_MANIFEST}"
    --output-dir "${FIXED_EVAL_DIR}"
    --model-variant "${MODEL_VARIANT_PREFIX}"
    --intensity-mode fixed
    --sample-rate "${SAMPLE_RATE}"
  )
  if [[ -n "${EVAL_LIMIT}" ]]; then
    fixed_cmd+=(--limit "${EVAL_LIMIT}")
  fi
  run_cmd "${fixed_cmd[@]}"
fi

if [[ "${RUN_INFER_EVAL}" == "1" ]]; then
  echo "[8] Evaluate inferred-intensity sweep"
  infer_cmd=(
    "${PYTHON_BIN}" scripts/eval_baseline_intensity_sweep.py
    --generated-manifest "${GENERATED_MANIFEST}"
    --benchmark-manifest "${BENCHMARK_MANIFEST}"
    --output-dir "${INFERRED_EVAL_DIR}"
    --model-variant "${MODEL_VARIANT_PREFIX}"
    --intensity-mode inferred
    --sample-rate "${SAMPLE_RATE}"
    --align-mode "${ALIGN_MODE}"
    --alphas "${ALPHAS_ARRAY[@]}"
  )
  if [[ -n "${EVAL_LIMIT}" ]]; then
    infer_cmd+=(--limit "${EVAL_LIMIT}")
  fi
  run_cmd "${infer_cmd[@]}"
fi

if [[ "${RUN_PLOT}" == "1" ]]; then
  echo "[9] Plot CycleGAN-only inferred frontier"
  run_cmd "${PYTHON_BIN}" scripts/cyclegan/plot_cyclegan_inferred_frontier.py \
    --overall-metrics "${INFERRED_EVAL_DIR}/overall_metrics.csv" \
    --output-dir "${FRONTIER_DIR}"
fi

if [[ "${RUN_ALPHA_PLOTS}" == "1" ]]; then
  echo "[10] Plot inferred alpha metrics"
  run_cmd "${PYTHON_BIN}" scripts/cyclegan/plot_inferred_alpha_metrics.py \
    --eval-dir "${INFERRED_EVAL_DIR}" \
    --out-dir "${FRONTIER_DIR}/alpha_metric_comparisons" \
    --title-prefix "CycleGAN ESD DSST Inferred Intensity"
fi


if [[ "${RUN_VALIDATE}" == "1" ]]; then
  echo "[11] Validate full CycleGAN inferred frontier outputs"
  run_cmd "${PYTHON_BIN}" scripts/cyclegan/validate_cyclegan_full_comparison.py \
    --benchmark-manifest "${BENCHMARK_MANIFEST}" \
    --fixed-manifest "${GENERATED_MANIFEST}" \
    --fixed-per-sample "${FIXED_EVAL_DIR}/per_sample_metrics.csv" \
    --inferred-per-sample "${INFERRED_EVAL_DIR}/per_sample_metrics.csv" \
    --targets "${TARGETS_ARRAY[@]}" \
    --alphas "${ALPHAS_ARRAY[@]}" \
    --expected-benchmark-rows 800 \
    --output-dir "${VALIDATION_DIR}"
fi

echo "CycleGAN spectrum inferred frontier pipeline finished."
echo "Generated manifest: ${GENERATED_MANIFEST}"
echo "Inferred eval: ${INFERRED_EVAL_DIR}"
echo "Frontier plot: ${FRONTIER_DIR}/tradeoff_frontier.png"
