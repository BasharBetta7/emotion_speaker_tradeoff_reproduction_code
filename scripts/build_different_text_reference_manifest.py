#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "sample_id",
    "speaker_id",
    "text",
    "text_rank",
    "source_file_id",
    "source_wav",
    "reference_file_id",
    "reference_wav",
    "target_emotion",
    "reference_emotion",
    "pair_type",
    "split",
}


def validate_input(manifest: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS.difference(manifest.columns)
    if missing:
        raise ValueError(f"Input manifest is missing required columns: {sorted(missing)}")

    mismatched_emotion = manifest["target_emotion"] != manifest["reference_emotion"]
    if mismatched_emotion.any():
        raise ValueError("Input manifest contains rows where target_emotion != reference_emotion.")


def build_different_text_manifest(manifest: pd.DataFrame, reference_shift: int) -> pd.DataFrame:
    validate_input(manifest)
    if reference_shift < 1:
        raise ValueError("--reference-shift must be >= 1 to avoid same-text references.")

    rows = []
    group_columns = ["speaker_id", "target_emotion"]
    for _, group in manifest.groupby(group_columns, sort=True):
        group = group.sort_values(["text_rank", "sample_id"]).reset_index(drop=True)
        if len(group) < 2:
            speaker_id = group.loc[0, "speaker_id"]
            target_emotion = group.loc[0, "target_emotion"]
            raise ValueError(
                f"Need at least two texts for speaker={speaker_id}, target_emotion={target_emotion}."
            )

        if reference_shift % len(group) == 0:
            raise ValueError(
                f"--reference-shift={reference_shift} maps a group of {len(group)} texts back to itself."
            )

        for source_idx, source in enumerate(group.itertuples(index=False)):
            reference = group.iloc[(source_idx + reference_shift) % len(group)]
            if source.text == reference.text:
                raise ValueError(
                    f"Same-text reference selected for speaker={source.speaker_id}, "
                    f"text_rank={source.text_rank}, target_emotion={source.target_emotion}."
                )

            row = source._asdict()
            row["sample_id"] = (
                f"{source.sample_id}_diffref_txt{int(reference.text_rank):03d}"
            )
            row["reference_file_id"] = reference.reference_file_id
            row["reference_wav"] = reference.reference_wav
            row["reference_text"] = reference.text
            row["reference_text_rank"] = int(reference.text_rank)
            row["reference_emotion"] = reference.reference_emotion
            row["pair_type"] = "same_speaker_different_text"
            rows.append(row)

    output = pd.DataFrame(rows)
    output = output.sort_values(["speaker_id", "text_rank", "target_emotion"]).reset_index(drop=True)

    if (output["text"] == output["reference_text"]).any():
        raise ValueError("Output manifest contains at least one same-text reference.")
    if (output["target_emotion"] != output["reference_emotion"]).any():
        raise ValueError("Output manifest contains at least one target/reference emotion mismatch.")

    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a same-speaker, different-text reference benchmark manifest from the "
            "balanced same-text benchmark manifest."
        )
    )
    parser.add_argument(
        "--input-manifest",
        type=Path,
        default=Path("Data/esd_balanced_benchmark_manifest.csv"),
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=Path("Data/esd_balanced_benchmark_manifest_different_text_ref.csv"),
    )
    parser.add_argument(
        "--reference-shift",
        type=int,
        default=1,
        help="Cyclic shift within each speaker/emotion group. Must not be a multiple of texts per group.",
    )
    args = parser.parse_args()

    manifest = pd.read_csv(args.input_manifest, sep="\t")
    output = build_different_text_manifest(manifest, args.reference_shift)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_manifest, sep="\t", index=False)

    print(f"Wrote {len(output)} rows to {args.output_manifest}")
    print(f"Speakers: {output['speaker_id'].nunique()}")
    print(f"Target emotions: {', '.join(sorted(output['target_emotion'].unique()))}")
    print(f"Pair type: {output['pair_type'].unique()[0]}")
    print(f"Same text pairs: {int((output['text'] == output['reference_text']).sum())}")


if __name__ == "__main__":
    main()
