#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd


TARGET_EMOTIONS = ["Angry", "Happy", "Sad", "Surprise"]
REQUIRED_COLUMNS = {"file_id", "speaker_id", "text", "emotion", "wav_path"}


def validate_pool(pool: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS.difference(pool.columns)
    if missing:
        raise ValueError(f"Input pool is missing required columns: {sorted(missing)}")

    duplicate_keys = pool.duplicated(subset=["speaker_id", "text", "emotion"], keep=False)
    if duplicate_keys.any():
        preview = pool.loc[duplicate_keys, ["speaker_id", "text", "emotion", "file_id"]].head(10)
        raise ValueError(f"Input pool has duplicate speaker/text/emotion rows:\n{preview}")

    missing_neutral = set(pool["speaker_id"].unique()) - set(
        pool.loc[pool["emotion"] == "Neutral", "speaker_id"].unique()
    )
    if missing_neutral:
        raise ValueError(f"Speakers with no neutral rows: {sorted(missing_neutral)}")


def add_text_rank(pool: pd.DataFrame) -> pd.DataFrame:
    pool = pool.copy()
    neutral = pool[pool["emotion"] == "Neutral"][["speaker_id", "text", "file_id"]].copy()
    neutral["text_rank"] = neutral["file_id"].str.split("_").str[1].astype(int)
    return pool.merge(neutral.drop(columns=["file_id"]), on=["speaker_id", "text"], how="left")


def select_text_subset(pool: pd.DataFrame, num_texts: int | None, text_ranks: Sequence[int] | None) -> pd.DataFrame:
    if num_texts is not None and num_texts < 1:
        raise ValueError("--num-texts must be >= 1.")
    if text_ranks is not None and len(text_ranks) == 0:
        raise ValueError("--text-ranks must not be empty when provided.")
    if num_texts is not None and text_ranks is not None:
        raise ValueError("Use either --num-texts or --text-ranks, not both.")

    if num_texts is None and text_ranks is None:
        return pool

    neutral_texts = (
        pool[pool["emotion"] == "Neutral"][["text", "text_rank"]]
        .drop_duplicates()
        .sort_values(["text_rank", "text"])
    )
    if text_ranks is not None:
        selected = neutral_texts[neutral_texts["text_rank"].isin(text_ranks)]
        missing = sorted(set(text_ranks).difference(set(selected["text_rank"])))
        if missing:
            raise ValueError(f"Requested text ranks are not in the pool: {missing}")
    else:
        if len(neutral_texts) < num_texts:
            raise ValueError(f"Requested {num_texts} texts, but only {len(neutral_texts)} are available.")
        selected = neutral_texts.head(num_texts)

    return pool[pool["text"].isin(selected["text"])].copy()


def build_manifest(
    pool: pd.DataFrame,
    num_texts: int | None = None,
    text_ranks: Sequence[int] | None = None,
) -> pd.DataFrame:
    validate_pool(pool)
    pool = add_text_rank(pool)
    pool = select_text_subset(pool, num_texts=num_texts, text_ranks=text_ranks)

    rows: List[Dict[str, object]] = []
    neutral_rows = pool[pool["emotion"] == "Neutral"].sort_values(["speaker_id", "text_rank", "file_id"])

    for source in neutral_rows.itertuples(index=False):
        candidates = pool[
            (pool["text"] == source.text)
            & (pool["speaker_id"] != source.speaker_id)
            & (pool["emotion"].isin(TARGET_EMOTIONS))
        ].sort_values(["speaker_id", "emotion", "file_id"])

        for reference in candidates.itertuples(index=False):
            rows.append(
                {
                    "sample_id": (
                        f"spk{int(source.speaker_id)}_txt{int(source.text_rank):03d}_"
                        f"{reference.emotion.lower()}_refspk{int(reference.speaker_id)}"
                    ),
                    "speaker_id": int(source.speaker_id),
                    "source_speaker_id": int(source.speaker_id),
                    "reference_speaker_id": int(reference.speaker_id),
                    "text": source.text,
                    "text_rank": int(source.text_rank),
                    "source_file_id": source.file_id,
                    "source_wav": source.wav_path,
                    "reference_file_id": reference.file_id,
                    "reference_wav": reference.wav_path,
                    "target_emotion": reference.emotion,
                    "reference_emotion": reference.emotion,
                    "pair_type": "different_speaker_same_text",
                    "split": "benchmark",
                }
            )

    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise ValueError("No different-speaker same-text pairs could be created.")

    same_speaker = manifest["source_speaker_id"] == manifest["reference_speaker_id"]
    if same_speaker.any():
        raise ValueError("Output manifest contains same-speaker pairs.")

    return manifest.sort_values(
        ["source_speaker_id", "text_rank", "target_emotion", "reference_speaker_id"]
    ).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build ESD neutral-to-emotion same-text pairs with different source/reference speakers."
    )
    parser.add_argument("--input-pool", type=Path, default=Path("Data/esd_balanced_benchmark_pool.csv"))
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=Path("Data/esd_balanced_benchmark_manifest_different_speaker_same_text.csv"),
    )
    parser.add_argument(
        "--num-texts",
        type=int,
        default=None,
        help="Use only the first N held-out texts by neutral text_rank. Default: use all texts.",
    )
    parser.add_argument(
        "--text-ranks",
        nargs="+",
        type=int,
        default=None,
        help="Use specific neutral text_rank values instead of the first N texts.",
    )
    args = parser.parse_args()

    pool = pd.read_csv(args.input_pool, sep="\t")
    manifest = build_manifest(pool, num_texts=args.num_texts, text_ranks=args.text_ranks)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output_manifest, sep="\t", index=False)

    print(f"Wrote {len(manifest)} rows to {args.output_manifest}")
    print(f"Source speakers: {manifest['source_speaker_id'].nunique()}")
    print(f"Reference speakers: {manifest['reference_speaker_id'].nunique()}")
    print(f"Texts: {manifest['text'].nunique()}")
    print(f"Text ranks: {', '.join(str(rank) for rank in sorted(manifest['text_rank'].unique()))}")
    print(f"Pair type: {manifest['pair_type'].unique()[0]}")
    print(f"Same-speaker pairs: {int((manifest['source_speaker_id'] == manifest['reference_speaker_id']).sum())}")
    print("Rows per target emotion:")
    print(manifest["target_emotion"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
