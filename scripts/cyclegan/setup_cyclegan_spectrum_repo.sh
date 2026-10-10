#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

EXTERNAL_ROOT="${EXTERNAL_ROOT:-experiments/external_repos}"
CYCLEGAN_REPO_URL="${CYCLEGAN_REPO_URL:-https://github.com/KunZhou9646/emotional-voice-conversion-with-CycleGAN-and-CWT-for-Spectrum-and-F0.git}"
CYCLEGAN_REPO_PATH="${CYCLEGAN_REPO_PATH:-${EXTERNAL_ROOT}/emotional-voice-conversion-with-CycleGAN-and-CWT-for-Spectrum-and-F0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DRY_RUN="${DRY_RUN:-1}"

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

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[dry-run] mkdir -p ${EXTERNAL_ROOT}"
else
  mkdir -p "${EXTERNAL_ROOT}"
fi

if [[ -d "${CYCLEGAN_REPO_PATH}/.git" ]]; then
  echo "External CycleGAN repo already exists: ${CYCLEGAN_REPO_PATH}"
else
  run_cmd git clone "${CYCLEGAN_REPO_URL}" "${CYCLEGAN_REPO_PATH}"
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[dry-run] ${PYTHON_BIN} scripts/cyclegan/patch_cyclegan_spectrum_repo.py --repo-path ${CYCLEGAN_REPO_PATH}"
else
  "${PYTHON_BIN}" scripts/cyclegan/patch_cyclegan_spectrum_repo.py --repo-path "${CYCLEGAN_REPO_PATH}"
fi

echo "CycleGAN spectrum repo path: ${CYCLEGAN_REPO_PATH}"
echo "Set CYCLEGAN_REPO_PATH=${CYCLEGAN_REPO_PATH} when running the full pipeline."
