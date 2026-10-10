#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_PATH="${OUTPUT_PATH:-experiments/emovox_pilot/pretrained/checkpoint_stage1.pt}"
DRIVE_URL="${DRIVE_URL:-https://drive.google.com/file/d/1oqk-PSREwpFNTyeREwcUry13WZ1LYl6U/view?usp=sharing}"

mkdir -p "$(dirname "${OUTPUT_PATH}")"

echo "Downloading Stage-I EmoVox checkpoint..."
"${PYTHON_BIN}" -m gdown --fuzzy "${DRIVE_URL}" -O "${OUTPUT_PATH}"

if [[ ! -s "${OUTPUT_PATH}" ]]; then
  echo "Download failed or empty file: ${OUTPUT_PATH}" >&2
  exit 1
fi

echo "Checkpoint downloaded: ${OUTPUT_PATH}"
