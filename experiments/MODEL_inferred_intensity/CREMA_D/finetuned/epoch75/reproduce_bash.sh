#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-python}"
# for alpha in 0.0 0.2 0.4 0.6 0.8 1.0; do
#     EPOCH=75
#     EPOCH_PADDED="$(printf "%05d" "${EPOCH}")"
#     PYTHON_BIN="${PYTHON_BIN:-python}"
#     CONFIG_PATH="${CONFIG_PATH:-Configs/config_emo_inferred_intensity_cremad_finetuned.yml}"
#     CHECKPOINT_PATH="${CHECKPOINT_PATH:-Models/exp5_emo_balanced_holdout_inferred_intensity_cremad_finetuned/epoch_${EPOCH_PADDED}.pth}"
#     MANIFEST_PATH="${MANIFEST_PATH:-Data/cremad_speaker_disjoint_benchmark_manifest.csv}"
#     MODEL_VARIANT="${MODEL_VARIANT:-$(basename "${CHECKPOINT_PATH}" .pth)}"
#     OUTPUT_DIR="${OUTPUT_ROOT:-experiments/MODEL_inferred_intensity/CREMA_D/finetuned}/full_eval_epoch_${EPOCH_PADDED}_alpha_${alpha}_$(date +%Y%m%d_%H%M%S)"
#     DEVICE="${DEVICE:-}"
#     LIMIT="${LIMIT:-}"
#     SAMPLE_ID="${SAMPLE_ID:-}"
#     INTENSITY_MODE="${INTENSITY_MODE:-inferred}" 
#     PYTHON_BIN="${PYTHON_BIN}" \
#     CONFIG_PATH="${CONFIG_PATH}" \
#     CHECKPOINT_PATH="${CHECKPOINT_PATH}" \
#     MANIFEST_PATH="${MANIFEST_PATH}" \
#     MODEL_VARIANT="${MODEL_VARIANT}" \
#     OUTPUT_DIR="${OUTPUT_DIR}" \
#     DEVICE="${DEVICE}" \
#     LIMIT="${LIMIT}" \
#     SAMPLE_ID="${SAMPLE_ID}" \
#     INTENSITY_MODE="${INTENSITY_MODE}" \
#     ALPHA="${alpha}" \
#     EPOCH="${EPOCH}" \
#     bash scripts/run_full_evaluation_pipeline.sh
# done

"${PYTHON_BIN}" "scripts/compare_alpha_variants.py" \
  --exp-dir "experiments/MODEL_inferred_intensity/CREMA_D/finetuned" \
  --out-dir "experiments/MODEL_inferred_intensity/CREMA_D/finetuned"

  "${PYTHON_BIN}" scripts/compare_model_variants.py --stargan-alpha-root experiments/MODEL_inferred_intensity/CREMA_D/finetuned \
--output-dir experiments/MODEL_inferred_intensity/CREMA_D/finetuned