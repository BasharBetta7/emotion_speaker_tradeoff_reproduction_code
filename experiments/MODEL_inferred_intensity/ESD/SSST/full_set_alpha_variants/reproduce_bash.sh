#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-experiments/MODEL_inferred_intensity/ESD/SSST/full_set_alpha_variants}"
CONFIG_PATH="${CONFIG_PATH:-Configs/config_emo.yml}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-Models/exp3_emo_balanced_holdout_inferred_intensity_from_scratch}"
MANIFEST_PATH="${MANIFEST_PATH:-Data/esd_balanced_benchmark_manifest.csv}"
DEVICE="${DEVICE:-}"
LIMIT="${LIMIT:-}"
SAMPLE_ID="${SAMPLE_ID:-}"
INTENSITY_MODE="${INTENSITY_MODE:-inferred}"
EPOCH="${EPOCH:-25}"
ALPHAS="${ALPHAS:-0.0 0.2 0.4 0.6 0.8 1.0}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_COMPARE="${RUN_COMPARE:-1}"
RUN_CONTENT_EVAL="${RUN_CONTENT_EVAL:-1}"
SAVE_AUDIO="${SAVE_AUDIO:-0}"
ASR_MODEL_NAME="${ASR_MODEL_NAME:-openai/whisper-small}"
ASR_DEVICE="${ASR_DEVICE:-${DEVICE}}"
ASR_BATCH_SIZE="${ASR_BATCH_SIZE:-4}"
RUN_STYLE_VIZ="${RUN_STYLE_VIZ:-1}"
STYLE_VIZ_INPUT_CSV="${STYLE_VIZ_INPUT_CSV:-Data/esd_balanced_train.csv}"
STYLE_VIZ_OUTPUT_DIR="${STYLE_VIZ_OUTPUT_DIR:-${OUTPUT_ROOT}/style_embedding_viz_epoch_$(printf "%05d" "${EPOCH}")}"
STYLE_VIZ_METHODS="${STYLE_VIZ_METHODS:-pca}"
STYLE_VIZ_DIMS="${STYLE_VIZ_DIMS:-2}"
STYLE_VIZ_MAX_SAMPLES="${STYLE_VIZ_MAX_SAMPLES:-}"
STYLE_VIZ_SAMPLES_PER_EMOTION="${STYLE_VIZ_SAMPLES_PER_EMOTION:-}"
STYLE_VIZ_SAMPLES_PER_SPEAKER="${STYLE_VIZ_SAMPLES_PER_SPEAKER:-}"
STYLE_VIZ_BATCH_SIZE="${STYLE_VIZ_BATCH_SIZE:-1}"
STYLE_VIZ_RANDOM_SEED="${STYLE_VIZ_RANDOM_SEED:-42}"

mkdir -p "${OUTPUT_ROOT}"

EPOCH_PADDED="$(printf "%05d" "${EPOCH}")"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${CHECKPOINT_ROOT}/epoch_${EPOCH_PADDED}.pth}"
MODEL_VARIANT="${MODEL_VARIANT:-epoch_${EPOCH_PADDED}}"

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "ERROR: checkpoint not found: ${CHECKPOINT_PATH}" >&2
  exit 1
fi

run_evaluation() {
  local alpha="$1"
  local output_dir="${OUTPUT_ROOT}/full_eval_epoch_${EPOCH_PADDED}_alpha_${alpha}_$(date +%Y%m%d_%H%M%S)"
  local save_audio="${SAVE_AUDIO}"
  if [[ "${RUN_CONTENT_EVAL}" == "1" ]]; then
    save_audio="1"
  fi

  echo "Running evaluation for epoch ${EPOCH_PADDED}, alpha ${alpha}..."

  PYTHON_BIN="${PYTHON_BIN}" \
    CONFIG_PATH="${CONFIG_PATH}" \
    CHECKPOINT_PATH="${CHECKPOINT_PATH}" \
    MANIFEST_PATH="${MANIFEST_PATH}" \
    MODEL_VARIANT="${MODEL_VARIANT}" \
    OUTPUT_DIR="${output_dir}" \
    DEVICE="${DEVICE}" \
    LIMIT="${LIMIT}" \
    SAMPLE_ID="${SAMPLE_ID}" \
    INTENSITY_MODE="${INTENSITY_MODE}" \
    ALPHA="${alpha}" \
    EPOCH="${EPOCH}" \
    SAVE_AUDIO="${save_audio}" \
    bash scripts/run_full_evaluation_pipeline.sh
}

if [[ "${RUN_EVAL}" == "1" ]]; then
  read -r -a ALPHA_ARRAY <<< "${ALPHAS}"
  for alpha in "${ALPHA_ARRAY[@]}"; do
    run_evaluation "${alpha}"
  done
fi

run_comparisons() {
  "${PYTHON_BIN}" scripts/compare_alpha_variants.py \
    --exp-dir "${OUTPUT_ROOT}" \
    --out-dir "${OUTPUT_ROOT}"

  "${PYTHON_BIN}" scripts/compare_model_variants.py \
    --stargan-alpha-root "${OUTPUT_ROOT}" \
    --output-dir "${OUTPUT_ROOT}"
}

if [[ "${RUN_COMPARE}" == "1" ]]; then
  run_comparisons
fi

run_content_eval() {
  local content_dir="${OUTPUT_ROOT}/content_eval"
  local template_csv="${content_dir}/transcript_template.csv"
  local asr_csv="${content_dir}/asr_output.csv"
  local scored_csv="${content_dir}/score_metrics.csv"
  local alpha_emotion_summary_csv="${content_dir}/metrics_by_alpha_emotion.csv"
  local alpha_summary_csv="${content_dir}/metrics_by_alpha.csv"

  mkdir -p "${content_dir}"

  echo "Preparing ASR transcript template from generated audio..."
  OUTPUT_ROOT="${OUTPUT_ROOT}" \
  TEMPLATE_CSV="${template_csv}" \
  "${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path

import pandas as pd

root = Path(os.environ["OUTPUT_ROOT"])
template_path = Path(os.environ["TEMPLATE_CSV"])
rows = []

for metrics_file in sorted(root.glob("full_eval_*/per_sample_metrics.csv")):
    frame = pd.read_csv(metrics_file)
    if "generated_wav" not in frame.columns:
        continue
    frame = frame.dropna(subset=["generated_wav"]).copy()
    frame = frame[frame["generated_wav"].astype(str).str.strip().ne("")]
    if frame.empty:
        continue
    required = ["sample_id", "speaker_id", "target_emotion", "alpha", "text", "generated_wav"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"{metrics_file} is missing columns required for ASR scoring: {missing}")
    subset = frame[required].rename(columns={"generated_wav": "audio_path"})
    rows.append(subset)

if not rows:
    raise SystemExit(
        "No generated audio paths found. Re-run evaluation with RUN_CONTENT_EVAL=1 or SAVE_AUDIO=1."
    )

template = pd.concat(rows, ignore_index=True)
template["audio_path"] = template["audio_path"].astype(str)
template = template.drop_duplicates(subset=["sample_id", "alpha", "audio_path"]).sort_values(
    ["alpha", "speaker_id", "sample_id"]
)
template_path.parent.mkdir(parents=True, exist_ok=True)
template.to_csv(template_path, index=False)
print(f"Wrote {len(template)} ASR template rows to {template_path}")
PY

  echo "Running ASR transcription..."
  asr_cmd=(
    "${PYTHON_BIN}" scripts/run_asr_transcription.py
    --input-csv "${template_csv}"
    --output-csv "${asr_csv}"
    --audio-column audio_path
    --model-name "${ASR_MODEL_NAME}"
    --batch-size "${ASR_BATCH_SIZE}"
  )
  if [[ -n "${ASR_DEVICE}" ]]; then
    asr_cmd+=(--device "${ASR_DEVICE}")
  fi
  "${asr_cmd[@]}"

  echo "Computing CER/WER..."
  "${PYTHON_BIN}" scripts/compute_text_metrics.py \
    --input-csv "${asr_csv}" \
    --reference-column text \
    --hypothesis-column predicted_text \
    --group-by alpha target_emotion \
    --output-scored-csv "${scored_csv}" \
    --output-summary-csv "${alpha_emotion_summary_csv}" \
    --plots-dir "${content_dir}/plots"

  echo "Writing per-alpha CER/WER summary and merging into alpha_metrics_summary.csv..."
  CONTENT_DIR="${content_dir}" \
  OUTPUT_ROOT="${OUTPUT_ROOT}" \
  "${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path

import pandas as pd

content_dir = Path(os.environ["CONTENT_DIR"])
output_root = Path(os.environ["OUTPUT_ROOT"])
scored_path = content_dir / "score_metrics.csv"
alpha_summary_path = content_dir / "metrics_by_alpha.csv"
main_summary_path = output_root / "alpha_metrics_summary.csv"

scored = pd.read_csv(scored_path)
alpha_summary = (
    scored.groupby("alpha", dropna=False)
    .agg(
        n=("sample_id", "size"),
        wer=("word_errors", lambda values: values.sum()),
        cer=("char_errors", lambda values: values.sum()),
        word_count=("word_count", "sum"),
        char_count=("char_count", "sum"),
        exact_match=("exact_match", "mean"),
        mean_wer=("wer", "mean"),
        mean_cer=("cer", "mean"),
    )
    .reset_index()
)
alpha_summary["wer"] = alpha_summary["wer"] / alpha_summary["word_count"].clip(lower=1)
alpha_summary["cer"] = alpha_summary["cer"] / alpha_summary["char_count"].clip(lower=1)
alpha_summary = alpha_summary.drop(columns=["word_count", "char_count"]).sort_values("alpha")
alpha_summary.to_csv(alpha_summary_path, index=False)
print(f"Wrote per-alpha CER/WER summary to {alpha_summary_path}")

if main_summary_path.exists():
    main_summary = pd.read_csv(main_summary_path)
    for column in ["wer", "cer", "exact_match", "mean_wer", "mean_cer", "n"]:
        main_summary = main_summary.drop(columns=[column], errors="ignore")
    merged = main_summary.merge(alpha_summary, on="alpha", how="left")
    merged.to_csv(main_summary_path, index=False)
    print(f"Merged CER/WER columns into {main_summary_path}")
else:
    print(f"Skipped merge because {main_summary_path} does not exist")
PY
}

if [[ "${RUN_CONTENT_EVAL}" == "1" ]]; then
  run_content_eval
fi

run_style_viz() {
  local style_cmd=(
    "${PYTHON_BIN}" scripts/visualize_style_embeddings.py
    --config-path "${CONFIG_PATH}"
    --checkpoint-path "${CHECKPOINT_PATH}"
    --input-csv "${STYLE_VIZ_INPUT_CSV}"
    --output-dir "${STYLE_VIZ_OUTPUT_DIR}"
    --methods ${STYLE_VIZ_METHODS}
    --dims ${STYLE_VIZ_DIMS}
    --random-seed "${STYLE_VIZ_RANDOM_SEED}"
  )

  if [[ -n "${STYLE_VIZ_MAX_SAMPLES}" ]]; then
    style_cmd+=(--max-samples "${STYLE_VIZ_MAX_SAMPLES}")
  fi
  if [[ -n "${STYLE_VIZ_SAMPLES_PER_EMOTION}" ]]; then
    style_cmd+=(--samples-per-emotion "${STYLE_VIZ_SAMPLES_PER_EMOTION}")
  fi
  if [[ -n "${STYLE_VIZ_SAMPLES_PER_SPEAKER}" ]]; then
    style_cmd+=(--samples-per-speaker "${STYLE_VIZ_SAMPLES_PER_SPEAKER}")
  fi
  if [[ -n "${STYLE_VIZ_BATCH_SIZE}" ]]; then
    style_cmd+=(--batch-size "${STYLE_VIZ_BATCH_SIZE}")
  fi
  if [[ -n "${DEVICE}" ]]; then
    style_cmd+=(--device "${DEVICE}")
  fi

  "${style_cmd[@]}"
}

if [[ "${RUN_STYLE_VIZ}" == "1" ]]; then
  run_style_viz
fi
