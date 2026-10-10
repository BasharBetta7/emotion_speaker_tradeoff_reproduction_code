#!/usr/bin/env python3

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd


CREMAD_EMOTION_MAP = {
    "ANG": "Angry",
    "HAP": "Happy",
    "NEU": "Neutral",
    "SAD": "Sad",
}
EMOTION_TO_LABEL = {
    "Angry": 0,
    "Happy": 1,
    "Neutral": 2,
    "Sad": 3,
}
TARGET_EMOTIONS = ["Angry", "Happy", "Sad"]

# CREMA-D's IEO sentence is the special intensity-varied sentence. The ESD
# benchmark has one reference per target, so the default CREMA-D benchmark uses
# only the regular XX-intensity sentences.
SENTENCE_TEXT = {
    "DFA": "Don't forget a jacket.",
    "IOM": "I'm on my way to the meeting.",
    "ITH": "I think I have a doctor's appointment.",
    "ITS": "I think I've seen this before.",
    "IWL": "I would like a new alarm clock.",
    "IWW": "I wonder what this is about.",
    "MTI": "Maybe tomorrow it will be cold.",
    "TAI": "The airplane is almost full.",
    "TIE": "That is exactly what happened.",
    "TSI": "The surface is slick.",
    "WSI": "We'll stop in a couple of minutes.",
}
SENTENCE_CODES = list(SENTENCE_TEXT)
ALL_EMOTIONS = ["Neutral", *TARGET_EMOTIONS]


def parse_cremad_filename(path: Path) -> dict[str, object] | None:
    parts = path.stem.split("_")
    if len(parts) != 4:
        return None

    speaker_id, sentence_code, emotion_code, intensity_code = parts
    if sentence_code not in SENTENCE_TEXT:
        return None
    if emotion_code not in CREMAD_EMOTION_MAP:
        return None
    if intensity_code != "XX":
        return None

    return {
        "file_id": path.stem,
        "speaker_id": int(speaker_id),
        "text": SENTENCE_TEXT[sentence_code],
        "sentence_code": sentence_code,
        "text_rank": SENTENCE_CODES.index(sentence_code) + 1,
        "emotion": CREMAD_EMOTION_MAP[emotion_code],
        "intensity": intensity_code,
        "wav_path": str(path.resolve()),
        "metadata_source": "CREMA-D AudioWAV filename",
    }


def load_metadata(dataset_root: Path) -> pd.DataFrame:
    audio_root = dataset_root / "AudioWAV"
    if not audio_root.exists():
        raise FileNotFoundError(f"Missing CREMA-D AudioWAV directory: {audio_root}")

    rows = []
    for wav_path in sorted(audio_root.glob("*.wav")):
        row = parse_cremad_filename(wav_path)
        if row is not None:
            rows.append(row)

    metadata = pd.DataFrame(rows)
    if metadata.empty:
        raise ValueError(f"No matching CREMA-D WAVs found under: {audio_root}")
    return metadata


def complete_speakers(metadata: pd.DataFrame) -> list[int]:
    required = {(sentence_code, emotion) for sentence_code in SENTENCE_CODES for emotion in ALL_EMOTIONS}
    complete = []
    for speaker_id, group in metadata.groupby("speaker_id"):
        observed = set(zip(group["sentence_code"], group["emotion"]))
        if required.issubset(observed):
            complete.append(int(speaker_id))
    return sorted(complete)


def split_speakers(speakers: list[int], test_speaker_count: int, seed: int) -> tuple[list[int], list[int]]:
    if test_speaker_count <= 0:
        raise ValueError("--test-speaker-count must be positive")
    if test_speaker_count >= len(speakers):
        raise ValueError(
            f"--test-speaker-count={test_speaker_count} leaves no training speakers; "
            f"only {len(speakers)} complete speakers are available."
        )

    shuffled = speakers[:]
    random.Random(seed).shuffle(shuffled)
    test_speakers = sorted(shuffled[:test_speaker_count])
    train_speakers = sorted(shuffled[test_speaker_count:])
    return train_speakers, test_speakers


def create_benchmark_manifest(test_pool: pd.DataFrame) -> pd.DataFrame:
    neutral = test_pool[test_pool["emotion"] == "Neutral"].copy()
    rows = []

    for source in neutral.sort_values(["speaker_id", "text_rank"]).itertuples(index=False):
        same_text = test_pool[
            (test_pool["speaker_id"] == source.speaker_id)
            & (test_pool["sentence_code"] == source.sentence_code)
        ]
        for target_emotion in TARGET_EMOTIONS:
            reference = same_text[same_text["emotion"] == target_emotion]
            if len(reference) != 1:
                raise ValueError(
                    "Expected one reference for "
                    f"speaker={source.speaker_id}, sentence={source.sentence_code}, emotion={target_emotion}; "
                    f"found {len(reference)}."
                )
            ref = reference.iloc[0]
            rows.append(
                {
                    "sample_id": f"spk{int(source.speaker_id)}_txt{int(source.text_rank):03d}_{target_emotion.lower()}",
                    "speaker_id": int(source.speaker_id),
                    "text": source.text,
                    "text_rank": int(source.text_rank),
                    "sentence_code": source.sentence_code,
                    "source_file_id": source.file_id,
                    "source_wav": source.wav_path,
                    "reference_file_id": ref.file_id,
                    "reference_wav": ref.wav_path,
                    "target_emotion": target_emotion,
                    "reference_emotion": target_emotion,
                    "pair_type": "same_speaker_same_text",
                    "split": "benchmark",
                }
            )

    return pd.DataFrame(rows).sort_values(["speaker_id", "text_rank", "target_emotion"]).reset_index(drop=True)


def write_metadata(df: pd.DataFrame, path: Path) -> None:
    columns = [
        "file_id",
        "speaker_id",
        "text",
        "sentence_code",
        "text_rank",
        "emotion",
        "intensity",
        "wav_path",
        "metadata_source",
    ]
    df.loc[:, columns].sort_values(["speaker_id", "text_rank", "emotion"]).to_csv(path, sep="\t", index=False)


def write_list(df: pd.DataFrame, path: Path) -> None:
    ordered = df.copy()
    ordered["emotion_order"] = ordered["emotion"].map({emotion: idx for idx, emotion in enumerate(ALL_EMOTIONS)})
    ordered = ordered.sort_values(["speaker_id", "text_rank", "emotion_order", "file_id"])
    lines = [f"{row.wav_path}|{EMOTION_TO_LABEL[row.emotion]}" for row in ordered.itertuples(index=False)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a speaker-disjoint CREMA-D benchmark aligned with the ESD neutral-to-emotion manifest format."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("Data/CREMA-D"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("Data"))
    parser.add_argument(
        "--test-speaker-count",
        type=int,
        default=24,
        help="24 speakers gives 24 * 11 sentences * 3 targets = 792 benchmark rows.",
    )
    parser.add_argument("--random-seed", type=int, default=0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(args.dataset_root)
    speakers = complete_speakers(metadata)
    train_speakers, test_speakers = split_speakers(speakers, args.test_speaker_count, args.random_seed)

    usable = metadata[metadata["speaker_id"].isin(speakers)].copy()
    train = usable[usable["speaker_id"].isin(train_speakers)].copy()
    benchmark_pool = usable[usable["speaker_id"].isin(test_speakers)].copy()
    benchmark_manifest = create_benchmark_manifest(benchmark_pool)

    write_metadata(train, args.output_dir / "cremad_speaker_disjoint_train.csv")
    write_metadata(benchmark_pool, args.output_dir / "cremad_speaker_disjoint_benchmark_pool.csv")
    write_list(train, args.output_dir / "cremad_train_list_emo_speaker_disjoint.txt")
    write_list(benchmark_pool, args.output_dir / "cremad_val_list_emo_speaker_disjoint.txt")
    benchmark_manifest.to_csv(args.output_dir / "cremad_speaker_disjoint_benchmark_manifest.csv", sep="\t", index=False)

    split_summary = pd.DataFrame(
        [
            {"split": "train", "speaker_count": len(train_speakers), "speakers": " ".join(map(str, train_speakers))},
            {"split": "benchmark", "speaker_count": len(test_speakers), "speakers": " ".join(map(str, test_speakers))},
        ]
    )
    split_summary.to_csv(args.output_dir / "cremad_speaker_disjoint_split.csv", index=False)

    print("CREMA-D speaker-disjoint benchmark created.")
    print(f"Dataset root: {args.dataset_root}")
    print(f"Complete speakers used: {len(speakers)}")
    print(f"Excluded incomplete speakers: {metadata['speaker_id'].nunique() - len(speakers)}")
    print(f"Train speakers: {len(train_speakers)}")
    print(f"Benchmark speakers: {len(test_speakers)}")
    print(f"Train utterances: {len(train)}")
    print(f"Benchmark utterance pool: {len(benchmark_pool)}")
    print(f"Benchmark manifest rows: {len(benchmark_manifest)}")


if __name__ == "__main__":
    main()
