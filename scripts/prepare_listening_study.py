#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


EMOTION_ORDER = ["Angry", "Happy", "Sad", "Surprise"]


def select_representative_subset(
    frame: pd.DataFrame,
    alpha_values: list[float],
    samples_per_emotion_alpha: int,
) -> pd.DataFrame:
    selected: list[pd.DataFrame] = []
    for alpha in alpha_values:
        alpha_frame = frame[frame["alpha"].round(6) == round(alpha, 6)].copy()
        for emotion in EMOTION_ORDER:
            emotion_frame = alpha_frame[alpha_frame["target_emotion"] == emotion].copy()
            if emotion_frame.empty:
                continue
            chosen = (
                emotion_frame.sort_values(["speaker_id", "text_rank", "sample_id"])
                .groupby("speaker_id", as_index=False, group_keys=False)
                .head(1)
                .head(samples_per_emotion_alpha)
                .copy()
            )
            selected.append(chosen)
    if not selected:
        raise ValueError("No listening-study rows selected. Check alpha values and input CSV.")
    return pd.concat(selected, ignore_index=True)


def build_rating_sheet(subset: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for row in subset.itertuples(index=False):
        rows.append(
            {
                "clip_id": row.sample_id,
                "alpha": row.alpha,
                "speaker_id": row.speaker_id,
                "target_emotion": row.target_emotion,
                "audio_path": row.generated_wav,
                "reference_text": row.text,
                "naturalness_1_5": "",
                "speaker_similarity_1_5": "",
                "emotion_match_1_5": "",
                "notes": "",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare balanced listening-study packets from benchmark outputs.")
    parser.add_argument("--input-csv", required=True, help="Per-sample metrics CSV with generated_wav paths.")
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.4, 0.8, 1.0])
    parser.add_argument("--samples-per-emotion-alpha", type=int, default=4)
    parser.add_argument("--output-dir", default="paper/artifacts/listening_study")
    args = parser.parse_args()

    frame = pd.read_csv(args.input_csv)
    if "generated_wav" not in frame.columns:
        raise KeyError("input CSV must contain generated_wav")

    subset = select_representative_subset(frame, args.alphas, args.samples_per_emotion_alpha)
    rating_sheet = build_rating_sheet(subset)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    subset.to_csv(out_dir / "selected_samples.csv", index=False)
    rating_sheet.to_csv(out_dir / "rating_sheet.csv", index=False)

    instructions = """# Listening Study Instructions

- Rate each generated clip on three axes:
  - naturalness: 1 very poor, 5 very natural
  - speaker similarity: 1 different speaker, 5 same speaker
  - emotion match: 1 wrong emotion, 5 strong target emotion
- Use the reference text only to confirm content, not pronunciation style.
- Randomize clip order before deployment to raters.
- Report results by alpha and by target emotion.
"""
    (out_dir / "README.md").write_text(instructions, encoding="utf-8")
    print(f"Wrote listening-study packet to {out_dir}")


if __name__ == "__main__":
    main()
