#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd


EMOTION_TO_LABEL = {
    "Angry": 0,
    "Happy": 1,
    "Neutral": 2,
    "Sad": 3,
    "Surprise": 4,
}
ALL_EMOTIONS = ["Neutral", "Angry", "Happy", "Sad", "Surprise"]
TARGET_EMOTIONS = ["Angry", "Happy", "Sad", "Surprise"]


def build_wav_path(dataset_root: Path, speaker_id: int, emotion: str, file_id: str) -> str:
    return str((dataset_root / f"{speaker_id:04d}" / emotion / f"{file_id}.wav").resolve())


def load_metadata(metadata_paths: List[Path], dataset_root: Path) -> pd.DataFrame:
    frames = []
    for path in metadata_paths:
        frame = pd.read_csv(path, sep="\t")
        frame["metadata_source"] = path.name
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["file_id"])
    df["speaker_id"] = df["speaker_id"].astype(int)
    df["wav_path"] = df.apply(
        lambda row: build_wav_path(dataset_root, row["speaker_id"], row["emotion"], row["file_id"]),
        axis=1,
    )
    return df


def select_shared_holdout_texts(df: pd.DataFrame, texts_per_speaker: int) -> List[str]:
    grouped = df.groupby(["speaker_id", "text"])["emotion"].nunique().reset_index(name="emotion_count")
    complete = grouped[grouped["emotion_count"] == len(ALL_EMOTIONS)]

    num_speakers = df["speaker_id"].nunique()
    shared = complete.groupby("text")["speaker_id"].nunique().reset_index(name="speaker_count")
    shared = shared[shared["speaker_count"] == num_speakers]

    neutral_rows = df[df["emotion"] == "Neutral"].copy()
    neutral_rows["neutral_rank"] = neutral_rows["file_id"].str.split("_").str[1].astype(int)
    canonical_ranks = (
        neutral_rows.groupby("text")["neutral_rank"].min().reset_index().sort_values(["neutral_rank", "text"])
    )
    ordered_shared = canonical_ranks[canonical_ranks["text"].isin(shared["text"])]

    if len(ordered_shared) < texts_per_speaker:
        raise ValueError(
            f"Requested {texts_per_speaker} holdout texts per speaker but only "
            f"{len(ordered_shared)} complete shared texts exist across all speakers."
        )

    return ordered_shared["text"].head(texts_per_speaker).tolist()


def create_holdout_table(df: pd.DataFrame, selected_texts: List[str]) -> pd.DataFrame:
    holdout = df[df["text"].isin(selected_texts)].copy()
    pivot = (
        holdout.pivot_table(
            index=["speaker_id", "text"],
            columns="emotion",
            values=["file_id", "wav_path"],
            aggfunc="first",
        )
        .sort_index()
    )
    pivot.columns = [f"{value_type}_{emotion.lower()}" for value_type, emotion in pivot.columns]
    holdout_texts = pivot.reset_index()

    neutral_order = (
        holdout[holdout["emotion"] == "Neutral"][["speaker_id", "text", "file_id"]]
        .assign(text_rank=lambda x: x["file_id"].str.split("_").str[1].astype(int))
        .drop(columns=["file_id"])
    )
    holdout_texts = holdout_texts.merge(neutral_order, on=["speaker_id", "text"], how="left")
    return holdout_texts.sort_values(["speaker_id", "text_rank", "text"]).reset_index(drop=True)


def create_benchmark_manifest(holdout_texts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for item in holdout_texts.itertuples(index=False):
        for target_emotion in TARGET_EMOTIONS:
            rows.append(
                {
                    "sample_id": f"spk{item.speaker_id}_txt{item.text_rank:03d}_{target_emotion.lower()}",
                    "speaker_id": item.speaker_id,
                    "text": item.text,
                    "text_rank": item.text_rank,
                    "source_file_id": item.file_id_neutral,
                    "source_wav": item.wav_path_neutral,
                    "reference_file_id": getattr(item, f"file_id_{target_emotion.lower()}"),
                    "reference_wav": getattr(item, f"wav_path_{target_emotion.lower()}"),
                    "target_emotion": target_emotion,
                    "reference_emotion": target_emotion,
                    "pair_type": "same_speaker_same_text",
                    "split": "benchmark",
                }
            )
    return pd.DataFrame(rows).sort_values(["speaker_id", "text_rank", "target_emotion"]).reset_index(drop=True)


def write_metadata(df: pd.DataFrame, path: Path) -> None:
    columns = ["file_id", "speaker_id", "text", "emotion", "wav_path", "metadata_source"]
    df.loc[:, columns].to_csv(path, sep="\t", index=False)


def write_list(df: pd.DataFrame, path: Path) -> None:
    ordered = df.copy()
    ordered["emotion_order"] = ordered["emotion"].map({emotion: idx for idx, emotion in enumerate(ALL_EMOTIONS)})
    ordered["text_rank"] = ordered["file_id"].str.split("_").str[1].astype(int)
    ordered = ordered.sort_values(["speaker_id", "text_rank", "emotion_order", "file_id"])
    lines = [f"{row.wav_path}|{EMOTION_TO_LABEL[row.emotion]}" for row in ordered.itertuples(index=False)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a balanced held-out-text benchmark for ESD emotion conversion.")
    parser.add_argument("--texts-per-speaker", type=int, default=20)
    parser.add_argument("--dataset-root", type=Path, default=Path("Data/ESD_DATA_24KHz"))
    parser.add_argument(
        "--metadata",
        nargs="+",
        type=Path,
        default=[Path("Data/esd_train.csv"), Path("Data/esd_val.csv")],
    )
    parser.add_argument("--output-dir", type=Path, default=Path("Data"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = load_metadata(args.metadata, args.dataset_root)

    selected_texts = select_shared_holdout_texts(metadata, args.texts_per_speaker)
    holdout_texts = create_holdout_table(metadata, selected_texts)

    benchmark_pool = metadata[metadata["text"].isin(selected_texts)].copy()
    train_balanced = metadata[~metadata["text"].isin(selected_texts)].copy()
    benchmark_manifest = create_benchmark_manifest(holdout_texts)

    write_metadata(train_balanced, args.output_dir / "esd_balanced_train.csv")
    write_metadata(benchmark_pool, args.output_dir / "esd_balanced_benchmark_pool.csv")
    write_list(train_balanced, args.output_dir / "train_list_emo_balanced.txt")
    write_list(benchmark_pool, args.output_dir / "val_list_emo_balanced.txt")
    holdout_texts.to_csv(args.output_dir / "esd_balanced_holdout_texts.csv", sep="\t", index=False)
    benchmark_manifest.to_csv(args.output_dir / "esd_balanced_benchmark_manifest.csv", sep="\t", index=False)

    print("Balanced hold-out benchmark created.")
    print(f"Speakers: {metadata['speaker_id'].nunique()}")
    print(f"Shared holdout texts per speaker: {args.texts_per_speaker}")
    print(f"Train utterances: {len(train_balanced)}")
    print(f"Benchmark utterance pool: {len(benchmark_pool)}")
    print(f"Benchmark manifest rows: {len(benchmark_manifest)}")


if __name__ == "__main__":
    main()
