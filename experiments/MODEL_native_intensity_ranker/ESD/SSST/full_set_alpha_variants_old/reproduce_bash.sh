#!/usr/bin/env bash
set -euo pipefail

for alpha in 0.0 0.2 0.4 0.6 0.8 1.0; do
    EPOCH=25
    EPOCH_PADDED="$(printf "%05d" "${EPOCH}")"
    PYTHON_BIN="${PYTHON_BIN:-python}"
    CONFIG_PATH="${CONFIG_PATH:-Configs/config_emo_native_intensity.yml}"
    CHECKPOINT_PATH="${CHECKPOINT_PATH:-Models/exp3_emo_balanced_holdout_native_intensity_2/epoch_${EPOCH_PADDED}.pth}"
    MANIFEST_PATH="${MANIFEST_PATH:-Data/esd_balanced_benchmark_manifest.csv}"
    MODEL_VARIANT="${MODEL_VARIANT:-$(basename "${CHECKPOINT_PATH}" .pth)}"
    OUTPUT_DIR="${OUTPUT_ROOT:-experiments/MODEL_native_intensity_ranker/ESD/SSST/full_set_alpha_variants}/full_eval_epoch_${EPOCH_PADDED}_alpha_${alpha}_$(date +%Y%m%d_%H%M%S)"
    DEVICE="${DEVICE:-}"
    LIMIT="${LIMIT:-}"
    SAMPLE_ID="${SAMPLE_ID:-}"
    INTENSITY_MODE="${INTENSITY_MODE:-native}" 
    PYTHON_BIN="${PYTHON_BIN}" \
    CONFIG_PATH="${CONFIG_PATH}" \
    CHECKPOINT_PATH="${CHECKPOINT_PATH}" \
    MANIFEST_PATH="${MANIFEST_PATH}" \
    MODEL_VARIANT="${MODEL_VARIANT}" \
    OUTPUT_DIR="${OUTPUT_DIR}" \
    DEVICE="${DEVICE}" \
    LIMIT="${LIMIT}" \
    SAMPLE_ID="${SAMPLE_ID}" \
    INTENSITY_MODE="${INTENSITY_MODE}" \
    ALPHA="${alpha}" \
    EPOCH="${EPOCH}" \
    bash scripts/run_full_evaluation_pipeline.sh
done

"${PYTHON_BIN}" "scripts/compare_alpha_variants.py" \
  --exp-dir "experiments/MODEL_native_intensity_ranker/ESD/SSST/full_set_alpha_variants" \
  --out-dir "experiments/MODEL_native_intensity_ranker/ESD/SSST/full_set_alpha_variants"
  
  
"${PYTHON_BIN}" scripts/compare_model_variants.py --stargan-alpha-root experiments/MODEL_native_intensity_ranker/ESD/SSST/full_set_alpha_variants/ \
--output-dir experiments/MODEL_native_intensity_ranker/ESD/SSST/full_set_alpha_variants/