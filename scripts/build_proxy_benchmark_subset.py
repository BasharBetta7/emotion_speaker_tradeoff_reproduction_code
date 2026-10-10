#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


TARGET_EMOTIONS = ["Angry", "Happy", "Sad", "Surprise"]


def load_manifest(manifest_path: Path) -> pd.DataFrame:
    df = pd.read_csv(manifest_path, sep="\t")
    required = {
        "sample_id",
        "speaker_id",
        "text",
        "text_rank",
        "source_file_id",
        "source_wav",
        "reference_file_id",
        "reference_wav",
        "target_emotion",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")
    return df


def select_proxy_texts(
    manifest_df: pd.DataFrame,
    texts_per_speaker: int,
    selection: str,
) -> pd.DataFrame:
    key_cols = ["speaker_id", "text", "text_rank", "source_file_id", "source_wav"]
    text_rows = manifest_df[key_cols].drop_duplicates().copy()
    text_rows = text_rows.sort_values(["speaker_id", "text_rank", "text"]).reset_index(drop=True)

    if selection == "first":
        proxy_texts = (
            text_rows.groupby("speaker_id", group_keys=False)
            .head(texts_per_speaker)
            .reset_index(drop=True)
        )
    elif selection == "last":
        proxy_texts = (
            text_rows.groupby("speaker_id", group_keys=False)
            .tail(texts_per_speaker)
            .reset_index(drop=True)
        )
    elif selection == "middle":
        groups = []
        for _, group in text_rows.groupby("speaker_id", sort=True):
            if len(group) < texts_per_speaker:
                raise ValueError(
                    f"Speaker {group.iloc[0]['speaker_id']} has only {len(group)} texts, "
                    f"but {texts_per_speaker} were requested."
                )
            start = max((len(group) - texts_per_speaker) // 2, 0)
            groups.append(group.iloc[start : start + texts_per_speaker])
        proxy_texts = pd.concat(groups, ignore_index=True)
    else:
        raise ValueError(f"Unsupported selection strategy: {selection}")

    counts = proxy_texts.groupby("speaker_id").size()
    if not (counts == texts_per_speaker).all():
        raise ValueError("Could not select the requested number of texts for every speaker.")

    return proxy_texts.sort_values(["speaker_id", "text_rank", "text"]).reset_index(drop=True)


def build_proxy_manifest(manifest_df: pd.DataFrame, proxy_texts: pd.DataFrame) -> pd.DataFrame:
    proxy = manifest_df.merge(
        proxy_texts[["speaker_id", "text", "text_rank"]],
        on=["speaker_id", "text", "text_rank"],
        how="inner",
    )
    proxy = proxy[proxy["target_emotion"].isin(TARGET_EMOTIONS)].copy()
    proxy = proxy.sort_values(["speaker_id", "text_rank", "target_emotion"]).reset_index(drop=True)
    proxy["split"] = "proxy_benchmark"
    return proxy


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a balanced proxy subset from the full benchmark manifest.")
    parser.add_argument("--manifest-path", type=Path, default=Path("Data/esd_balanced_benchmark_manifest.csv"))
    parser.add_argument("--output-path", type=Path, default=Path("Data/esd_balanced_proxy_manifest.csv"))
    parser.add_argument("--texts-per-speaker", type=int, default=1)
    parser.add_argument(
        "--selection",
        choices=["first", "middle", "last"],
        default="middle",
        help="Which held-out texts to keep per speaker.",
    )
    parser.add_argument(
        "--proxy-texts-output",
        type=Path,
        default=Path("Data/esd_balanced_proxy_texts.csv"),
        help="Optional CSV with the selected speaker/text pairs.",
    )
    args = parser.parse_args()

    manifest_df = load_manifest(args.manifest_path)
    proxy_texts = select_proxy_texts(
        manifest_df=manifest_df,
        texts_per_speaker=args.texts_per_speaker,
        selection=args.selection,
    )
    proxy_manifest = build_proxy_manifest(manifest_df, proxy_texts)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    proxy_manifest.to_csv(args.output_path, sep="\t", index=False)
    args.proxy_texts_output.parent.mkdir(parents=True, exist_ok=True)
    proxy_texts.to_csv(args.proxy_texts_output, sep="\t", index=False)

    print("Proxy benchmark subset created.")
    print(f"Speakers: {proxy_texts['speaker_id'].nunique()}")
    print(f"Texts per speaker: {args.texts_per_speaker}")
    print(f"Proxy text pairs: {len(proxy_texts)}")
    print(f"Proxy manifest rows: {len(proxy_manifest)}")
    print(f"Saved manifest to {args.output_path}")
    print(f"Saved selected texts to {args.proxy_texts_output}")


if __name__ == "__main__":
    main()
