#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

BASELINE="${BASELINE:-cyclegan}"
DEFAULT_CYCLEGAN_REPO="experiments/external_repos/EmoCycleGAN"
REPO_PATH="${REPO_PATH:-${CYCLEGAN_REPO_PATH:-${DEFAULT_CYCLEGAN_REPO}}}"
DATA_ROOT="${DATA_ROOT:-Data/external_baselines}"
TARGETS="${TARGETS:-angry happy sad surprise}"
DRY_RUN="${DRY_RUN:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_ROOT="${MODEL_ROOT:-${REPO_ROOT}/experiments/baselines/cyclegan_models}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/experiments/baselines/cyclegan_logs}"
SKIP_IF_EXISTS="${SKIP_IF_EXISTS:-1}"

if [[ "${DATA_ROOT}" != /* ]]; then
  DATA_ROOT="${REPO_ROOT}/${DATA_ROOT}"
fi
if [[ "${MODEL_ROOT}" != /* ]]; then
  MODEL_ROOT="${REPO_ROOT}/${MODEL_ROOT}"
fi
if [[ "${LOG_ROOT}" != /* ]]; then
  LOG_ROOT="${REPO_ROOT}/${LOG_ROOT}"
fi

if [[ -z "${REPO_PATH}" ]]; then
  echo "REPO_PATH is required (path to external baseline repository)." >&2
  exit 1
fi

if [[ ! -d "${REPO_PATH}" ]]; then
  echo "REPO_PATH does not exist: ${REPO_PATH}" >&2
  exit 1
fi

print_cmd() {
  local quoted=""
  printf -v quoted "%q " "$@"
  echo "${quoted}"
}

run_cmd_in_repo() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run] (cd ${REPO_PATH} && $(print_cmd "$@"))"
    return
  fi
  echo "[run] (cd ${REPO_PATH} && $(print_cmd "$@"))"
  (
    cd "${REPO_PATH}"
    "$@"
  )
}

if [[ "${BASELINE}" == "cyclegan" ]]; then
  for emo in ${TARGETS}; do
    pair="n2${emo,,}"
    pair_root="${DATA_ROOT}/cyclegan/${pair}"

    train_a="${pair_root}/train_A.list"
    train_b="${pair_root}/train_B.list"
    val_a="${pair_root}/validation_A.list"
    val_b="${pair_root}/validation_B.list"

    if [[ ! -f "${train_a}" || ! -f "${train_b}" ]]; then
      echo "Missing prepared lists for ${pair}. Run prepare_external_baseline_data.py first." >&2
      exit 1
    fi

    # Legacy repo expects directory inputs. If you used --materialize-audio, these directories exist.
    train_a_dir="${pair_root}/train_A"
    train_b_dir="${pair_root}/train_B"
    val_a_dir="${pair_root}/validation_A"
    val_b_dir="${pair_root}/validation_B"

    if [[ ! -d "${train_a_dir}" || ! -d "${train_b_dir}" ]]; then
      echo "Missing ${train_a_dir} or ${train_b_dir}. Re-run prepare script with --materialize-audio." >&2
      exit 1
    fi

    model_dir="${MODEL_ROOT}/${pair}"
    log_dir="${LOG_ROOT}/${pair}"
    mkdir -p "${model_dir}" "${log_dir}"

    if [[ "${SKIP_IF_EXISTS}" == "1" && -f "${model_dir}/${pair}.ckpt.index" ]]; then
      echo "[skip] checkpoint already exists: ${model_dir}/${pair}.ckpt.index"
      continue
    fi

    run_cmd_in_repo "${PYTHON_BIN}" train.py \
      --train_A_dir "${train_a_dir}" \
      --train_B_dir "${train_b_dir}" \
      --validation_A_dir "${val_a_dir}" \
      --validation_B_dir "${val_b_dir}" \
      --model_dir "${model_dir}" \
      --tensorboard_log_dir "${log_dir}" \
      --model_name "${pair}.ckpt"
  done
elif [[ "${BASELINE}" == "vawgan" ]]; then
  for emo in ${TARGETS}; do
    pair="n2${emo,,}"
    pair_root="${DATA_ROOT}/vawgan/${pair}"
    train_pairs="${pair_root}/train_pairs.tsv"
    eval_pairs="${pair_root}/eval_pairs.tsv"

    if [[ ! -f "${train_pairs}" || ! -f "${eval_pairs}" ]]; then
      echo "Missing prepared pair tables for ${pair}." >&2
      exit 1
    fi

    out_dir="${REPO_ROOT}/experiments/baselines/vawgan_models/${pair}"
    mkdir -p "${out_dir}"

    # Command is intentionally generic because different VAWGAN forks expose different entrypoints.
    run_cmd_in_repo "${PYTHON_BIN}" main-vawgan.py \
      --train_pairs "${train_pairs}" \
      --eval_pairs "${eval_pairs}" \
      --output_dir "${out_dir}"
  done
else
  echo "Unsupported BASELINE='${BASELINE}'. Expected 'cyclegan' or 'vawgan'." >&2
  exit 1
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "Dry-run completed. Set DRY_RUN=0 to execute commands."
fi
