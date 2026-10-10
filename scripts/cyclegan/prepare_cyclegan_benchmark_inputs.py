#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable

import pandas as pd


TARGET_EMOTIONS = ["Angry", "Happy", "Sad", "Surprise"]


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=None, engine="python")


def ensure_columns(frame: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise KeyError(f"{name} is missing required columns: {missing}")


def relink_or_copy(src: Path, dst: Path, mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    if mode == "symlink":
        dst.symlink_to(src)
    elif mode == "hardlink":
        dst.hardlink_to(src)
    else:
        shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare sample_id-stable CycleGAN benchmark conversion inputs. "
            "Creates per-emotion input folders with filenames '<sample_id>.wav'."
        )
    )
    parser.add_argument("--benchmark-manifest", default="Data/esd_balanced_benchmark_manifest.csv")
    parser.add_argument("--output-root", default="experiments/baselines/cyclegan_spectrum_benchmark_package")
    parser.add_argument("--converted-root", default="experiments/baselines/cyclegan_spectrum_converted_full")
    parser.add_argument("--targets", nargs="+", default=TARGET_EMOTIONS)
    parser.add_argument("--model-variant-prefix", default="cyclegan_spectrum")
    parser.add_argument("--link-mode", choices=["symlink", "hardlink", "copy"], default="symlink")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest = read_table(Path(args.benchmark_manifest))
    ensure_columns(
        manifest,
        ["sample_id", "target_emotion", "source_wav", "reference_wav", "speaker_id", "text", "text_rank"],
        "benchmark manifest",
    )
    manifest = manifest.copy()
    manifest["target_emotion_norm"] = manifest["target_emotion"].astype(str).str.strip().str.lower()
    target_map = {target.strip().lower(): target for target in args.targets}

    output_root = Path(args.output_root)
    if output_root.exists() and args.force:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    converted_root = Path(args.converted_root)
    rows = []

    for target_lower in target_map:
        pair_tag = f"n2{target_lower}"
        subset = manifest[manifest["target_emotion_norm"] == target_lower].copy().sort_values("sample_id")
        input_dir = output_root / "inputs" / pair_tag
        input_dir.mkdir(parents=True, exist_ok=True)

        for item in subset.itertuples(index=False):
            src_path = Path(item.source_wav)
            if not src_path.exists():
                raise FileNotFoundError(f"Missing source wav in benchmark manifest: {src_path}")

            input_wav = input_dir / f"{item.sample_id}.wav"
            relink_or_copy(src_path.resolve(), input_wav, args.link_mode)

            converted_wav = (converted_root / pair_tag / f"{item.sample_id}.wav").resolve()
            rows.append(
                {
                    "sample_id": item.sample_id,
                    "target_emotion": item.target_emotion,
                    "speaker_id": item.speaker_id,
                    "text": item.text,
                    "text_rank": item.text_rank,
                    "source_wav": item.source_wav,
                    "reference_wav": item.reference_wav,
                    "pair_tag": pair_tag,
                    "conversion_direction": "A2B",
                    "input_wav": str(input_wav.resolve()),
                    "expected_converted_wav": str(converted_wav),
                    "model_variant": f"{args.model_variant_prefix}_{pair_tag}",
                }
            )

        pd.DataFrame(rows).query("pair_tag == @pair_tag").to_csv(
            output_root / f"{pair_tag}_conversion_manifest.csv",
            index=False,
        )

    combined = pd.DataFrame(rows).sort_values(["pair_tag", "sample_id"]).reset_index(drop=True)
    combined.to_csv(output_root / "conversion_manifest_all.csv", index=False)

    print(f"Prepared benchmark conversion package: {output_root}")
    print(f"Rows: {len(combined)}")
    print(f"Pair counts:\n{combined['pair_tag'].value_counts().sort_index().to_string()}")


if __name__ == "__main__":
    main()
