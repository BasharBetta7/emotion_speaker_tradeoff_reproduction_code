#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_records(frame: pd.DataFrame, audio_column: str) -> pd.DataFrame:
    if audio_column not in frame.columns:
        raise KeyError(f"Column not found: {audio_column}")

    keep_cols = [
        column
        for column in ["sample_id", "speaker_id", "target_emotion", "alpha", "text", audio_column]
        if column in frame.columns
    ]
    records = frame[keep_cols].copy()
    records = records.rename(columns={audio_column: "audio_path"})
    records = records.dropna(subset=["audio_path"]).drop_duplicates()
    records["predicted_text"] = ""
    records["backend"] = ""
    records["backend_model"] = ""
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a transcript worklist for ASR-based content evaluation."
    )
    parser.add_argument("--input-csv", required=True, help="Usually per_sample_metrics.csv from a benchmark run.")
    parser.add_argument("--audio-column", default="generated_wav")
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.input_csv)
    records = build_records(frame, args.audio_column)

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records.to_csv(output_path, index=False)

    print(f"Wrote transcript template to {output_path}")
    print("Fill predicted_text with ASR transcripts, then score it with scripts/compute_text_metrics.py.")


if __name__ == "__main__":
    main()
