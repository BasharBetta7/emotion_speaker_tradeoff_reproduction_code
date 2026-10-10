#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SELECTED_SAMPLES = REPO_ROOT / "paper" / "artifacts" / "listening_study" / "selected_samples.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "paper" / "artifacts" / "listening_study" / "samples for listening"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "paper" / "artifacts" / "listening_study" / "rating_sheet_copied_audio.json"


def resolve_repo_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def relative_to_repo(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def make_item_id(sample_id: str, alpha: float) -> str:
    return f"{sample_id}__a{alpha:.2f}"


def copy_if_needed(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy listening-study source/generated audio and build a JSON manifest with copied paths."
    )
    parser.add_argument("--selected-samples", default=str(DEFAULT_SELECTED_SAMPLES))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--shuffle-seed", type=int, default=1337)
    args = parser.parse_args()

    selected_samples_path = Path(args.selected_samples)
    output_dir = Path(args.output_dir)
    output_json = Path(args.output_json)

    df = pd.read_csv(selected_samples_path)
    df = df.sample(frac=1.0, random_state=args.shuffle_seed).reset_index(drop=True)

    generated_dir = output_dir / "generated"
    source_dir = output_dir / "source"

    records = []
    for idx, row in enumerate(df.itertuples(index=False), start=1):
        sample_id = str(row.sample_id)
        alpha = float(row.alpha)
        source_src = resolve_repo_path(str(row.source_wav))
        generated_src = resolve_repo_path(str(row.generated_wav))

        source_dst = source_dir / source_src.name
        generated_dst = generated_dir / generated_src.name

        copy_if_needed(source_src, source_dst)
        copy_if_needed(generated_src, generated_dst)

        generated_rel = relative_to_repo(generated_dst)
        source_rel = relative_to_repo(source_dst)

        records.append(
            {
                "model": "audio_samples.audiosample",
                "pk": idx,
                "fields": {
                    "item_id": make_item_id(sample_id, alpha),
                    "clip_id": sample_id,
                    "target_emotion": str(row.target_emotion),
                    "speaker_id": int(row.speaker_id),
                    "alpha": alpha,
                    "audio_path": generated_rel,
                    "generated_audio_path": generated_rel,
                    "source_audio_path": source_rel,
                    "reference_text": str(row.text),
                },
            }
        )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=4, ensure_ascii=True)
        f.write("\n")

    print(f"Copied generated files to {generated_dir}")
    print(f"Copied source files to {source_dir}")
    print(f"Wrote manifest to {output_json}")


if __name__ == "__main__":
    main()
