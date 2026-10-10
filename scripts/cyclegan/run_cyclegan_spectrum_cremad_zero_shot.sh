#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DRY_RUN="${DRY_RUN:-1}"

TARGETS="${TARGETS:-Angry Happy Sad}"
TARGETS_LOWER="${TARGETS_LOWER:-angry happy sad}"
ALPHAS="${ALPHAS:-0.0 0.2 0.4 0.6 0.8 1.0}"

CREMAD_ROOT="${CREMAD_ROOT:-Data/CREMA-D}"
TEST_SPEAKER_COUNT="${TEST_SPEAKER_COUNT:-24}"
RANDOM_SEED="${RANDOM_SEED:-0}"

BENCHMARK_MANIFEST="${BENCHMARK_MANIFEST:-Data/cremad_speaker_disjoint_benchmark_manifest.csv}"
BENCHMARK_POOL="${BENCHMARK_POOL:-Data/cremad_speaker_disjoint_benchmark_pool.csv}"
TRAIN_METADATA="${TRAIN_METADATA:-Data/cremad_speaker_disjoint_train.csv}"

LINK_MODE="${LINK_MODE:-symlink}"
SAMPLE_RATE="${SAMPLE_RATE:-24000}"
ALIGN_MODE="${ALIGN_MODE:-trim}"
INFERRED_MIX_DOMAIN="${INFERRED_MIX_DOMAIN:-cyclegan_features}"
MODEL_VARIANT_PREFIX="${MODEL_VARIANT_PREFIX:-cyclegan_spectrum_esd_zeroshot_cremad}"
EVAL_LIMIT="${EVAL_LIMIT:-}"
SAVE_AUDIO="${SAVE_AUDIO:-0}"

CYCLEGAN_ENV="${CYCLEGAN_ENV:-cyclegan_tf_gpu}"
EVAL_ENV="${EVAL_ENV:-torch_stable}"
CYCLEGAN_PYTHON_BIN="${CYCLEGAN_PYTHON_BIN:-python}"
EVAL_PYTHON_BIN="${EVAL_PYTHON_BIN:-python}"

CYCLEGAN_REPO_PATH="${CYCLEGAN_REPO_PATH:-experiments/external_repos/emotional-voice-conversion-with-CycleGAN-and-CWT-for-Spectrum-and-F0}"
CYCLEGAN_CODE_SUBDIR="${CYCLEGAN_CODE_SUBDIR:-Parallel-data-free emotional voice conversion with CycleGAN and CWT}"
CYCLEGAN_CODE_DIR="${CYCLEGAN_CODE_DIR:-${CYCLEGAN_REPO_PATH}/${CYCLEGAN_CODE_SUBDIR}}"

MODEL_ROOT="${MODEL_ROOT:-experiments/baselines/cyclegan_spectrum_models}"
PACKAGE_ROOT="${PACKAGE_ROOT:-experiments/baselines/cyclegan_spectrum_cremad_benchmark_package}"
CONVERTED_ROOT="${CONVERTED_ROOT:-experiments/baselines/cyclegan_spectrum_cremad_converted}"
FEATURE_ROOT="${FEATURE_ROOT:-experiments/baselines/cyclegan_spectrum_cremad_features}"
GENERATED_MANIFEST="${GENERATED_MANIFEST:-experiments/baselines/cyclegan_spectrum_cremad_generated.csv}"
INFERRED_EVAL_DIR="${INFERRED_EVAL_DIR:-experiments/baselines/cyclegan_spectrum_cremad_inferred_eval_${INFERRED_MIX_DOMAIN}}"
ARTIFACT_DIR="${ARTIFACT_DIR:-paper/artifacts/cyclegan_spectrum_cremad_zero_shot_${INFERRED_MIX_DOMAIN}}"

RUN_BUILD_MANIFEST="${RUN_BUILD_MANIFEST:-1}"
RUN_PACKAGE="${RUN_PACKAGE:-1}"
RUN_CONVERT="${RUN_CONVERT:-1}"
RUN_ASSEMBLE="${RUN_ASSEMBLE:-1}"
RUN_INFER_EVAL="${RUN_INFER_EVAL:-1}"
RUN_FRONTIER_PLOT="${RUN_FRONTIER_PLOT:-1}"
RUN_ALPHA_PLOTS="${RUN_ALPHA_PLOTS:-1}"

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

run_env_python() {
  local env_name="$1"
  local fallback_python="$2"
  shift 2

  if [[ "${DRY_RUN}" == "1" ]]; then
    if [[ -n "${env_name}" ]]; then
      echo "[dry-run] conda run --no-capture-output -n ${env_name} python -u $(print_cmd "$@")"
    else
      echo "[dry-run] ${fallback_python} -u $(print_cmd "$@")"
    fi
    return
  fi

  if [[ -n "${env_name}" ]]; then
    conda run --no-capture-output -n "${env_name}" python -u "$@"
  else
    "${fallback_python}" -u "$@"
  fi
}

abs_path() {
  if [[ "$1" = /* ]]; then
    echo "$1"
  else
    echo "${REPO_ROOT}/$1"
  fi
}

read -r -a TARGETS_ARRAY <<< "${TARGETS}"
read -r -a TARGETS_LOWER_ARRAY <<< "${TARGETS_LOWER}"
read -r -a ALPHAS_ARRAY <<< "${ALPHAS}"

if [[ "${#TARGETS_ARRAY[@]}" -ne "${#TARGETS_LOWER_ARRAY[@]}" ]]; then
  echo "TARGETS and TARGETS_LOWER must have the same number of entries." >&2
  exit 1
fi

PACKAGE_ROOT_ABS="$(abs_path "${PACKAGE_ROOT}")"
CONVERTED_ROOT_ABS="$(abs_path "${CONVERTED_ROOT}")"
FEATURE_ROOT_ABS="$(abs_path "${FEATURE_ROOT}")"
MODEL_ROOT_ABS="$(abs_path "${MODEL_ROOT}")"

if [[ "${DRY_RUN}" != "1" ]]; then
  if [[ ! -f "${CYCLEGAN_CODE_DIR}/model.py" || ! -f "${CYCLEGAN_CODE_DIR}/preprocess.py" ]]; then
    echo "Missing external CycleGAN code under: ${CYCLEGAN_CODE_DIR}" >&2
    exit 1
  fi
  for emo in "${TARGETS_LOWER_ARRAY[@]}"; do
    pair="n2${emo}"
    if [[ ! -f "${MODEL_ROOT_ABS}/${pair}/${pair}.ckpt.index" ]]; then
      echo "Missing ESD-trained CycleGAN checkpoint: ${MODEL_ROOT_ABS}/${pair}/${pair}.ckpt.index" >&2
      exit 1
    fi
  done
fi

echo "CycleGAN spectrum CREMA-D zero-shot evaluation"
echo "Targets: ${TARGETS}"
echo "CREMA-D root: ${CREMAD_ROOT}"
echo "Benchmark manifest: ${BENCHMARK_MANIFEST}"
echo "Mix domain: ${INFERRED_MIX_DOMAIN}"

if [[ "${RUN_BUILD_MANIFEST}" == "1" ]]; then
  echo "[1/7] Build CREMA-D speaker-disjoint benchmark"
  run_cmd "${PYTHON_BIN}" scripts/build_cremad_speaker_disjoint_benchmark.py \
    --dataset-root "${CREMAD_ROOT}" \
    --test-speaker-count "${TEST_SPEAKER_COUNT}" \
    --random-seed "${RANDOM_SEED}"
fi

if [[ "${RUN_PACKAGE}" == "1" ]]; then
  echo "[2/7] Prepare CREMA-D benchmark inputs"
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
  echo "[3/7] Convert CREMA-D benchmark audio with ESD-trained checkpoints"
  for emo in "${TARGETS_LOWER_ARRAY[@]}"; do
    pair="n2${emo}"
    run_env_python "${CYCLEGAN_ENV}" "${CYCLEGAN_PYTHON_BIN}" scripts/cyclegan/convert_cyclegan_spectrum.py \
      --code-dir "${CYCLEGAN_CODE_DIR}" \
      --model-dir "${MODEL_ROOT_ABS}/${pair}" \
      --model-name "${pair}.ckpt" \
      --data-dir "${PACKAGE_ROOT_ABS}/inputs/${pair}" \
      --conversion-direction A2B \
      --output-dir "${CONVERTED_ROOT_ABS}/${pair}" \
      --feature-dir "${FEATURE_ROOT_ABS}/${pair}" \
      --sample-rate "${SAMPLE_RATE}"
  done
fi

if [[ "${RUN_ASSEMBLE}" == "1" ]]; then
  echo "[4/7] Assemble generated manifest"
  run_cmd "${PYTHON_BIN}" scripts/cyclegan/assemble_cyclegan_fixed_manifest.py \
    --benchmark-manifest "${BENCHMARK_MANIFEST}" \
    --converted-root "${CONVERTED_ROOT}" \
    --output-path "${GENERATED_MANIFEST}" \
    --targets "${TARGETS_ARRAY[@]}" \
    --model-variant-prefix "${MODEL_VARIANT_PREFIX}" \
    --strict
fi

if [[ "${RUN_INFER_EVAL}" == "1" ]]; then
  echo "[5/7] Evaluate inferred-intensity alpha sweep"
  eval_cmd=(
    scripts/eval_baseline_intensity_sweep.py
    --generated-manifest "${GENERATED_MANIFEST}"
    --benchmark-manifest "${BENCHMARK_MANIFEST}"
    --output-dir "${INFERRED_EVAL_DIR}"
    --model-variant "${MODEL_VARIANT_PREFIX}"
    --intensity-mode inferred
    --sample-rate "${SAMPLE_RATE}"
    --align-mode "${ALIGN_MODE}"
    --inferred-mix-domain "${INFERRED_MIX_DOMAIN}"
    --cyclegan-feature-root "${FEATURE_ROOT_ABS}"
    --alphas "${ALPHAS_ARRAY[@]}"
  )
  if [[ -n "${EVAL_LIMIT}" ]]; then
    eval_cmd+=(--limit "${EVAL_LIMIT}")
  fi
  if [[ "${SAVE_AUDIO}" == "1" ]]; then
    eval_cmd+=(--save-audio)
  fi
  run_env_python "${EVAL_ENV}" "${EVAL_PYTHON_BIN}" "${eval_cmd[@]}"
fi

if [[ "${RUN_FRONTIER_PLOT}" == "1" ]]; then
  echo "[6/7] Plot tradeoff frontier"
  run_cmd "${PYTHON_BIN}" scripts/cyclegan/plot_cyclegan_inferred_frontier.py \
    --overall-metrics "${INFERRED_EVAL_DIR}/overall_metrics.csv" \
    --output-dir "${ARTIFACT_DIR}" \
    --title "CycleGAN ESD Zero-Shot on CREMA-D (${INFERRED_MIX_DOMAIN})"
fi

if [[ "${RUN_ALPHA_PLOTS}" == "1" ]]; then
  echo "[7/7] Plot alpha metric comparisons"
  run_cmd "${PYTHON_BIN}" scripts/cyclegan/plot_inferred_alpha_metrics.py \
    --eval-dir "${INFERRED_EVAL_DIR}" \
    --out-dir "${ARTIFACT_DIR}/alpha_metric_comparisons" \
    --title-prefix "CycleGAN ESD Zero-Shot on CREMA-D (${INFERRED_MIX_DOMAIN})"
fi

echo "Done."
echo "CREMA-D train metadata: ${TRAIN_METADATA}"
echo "CREMA-D benchmark pool: ${BENCHMARK_POOL}"
echo "CREMA-D benchmark manifest: ${BENCHMARK_MANIFEST}"
echo "Generated manifest: ${GENERATED_MANIFEST}"
echo "Inferred eval: ${INFERRED_EVAL_DIR}"
echo "Frontier plot: ${ARTIFACT_DIR}/tradeoff_frontier.png"
