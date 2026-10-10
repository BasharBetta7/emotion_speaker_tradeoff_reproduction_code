#!/usr/bin/env python3

from __future__ import annotations

import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Assemble merged fixed-output manifest (sample_id, generated_wav, model_variant) "
            "from per-emotion CycleGAN conversion directories."
        )
    )
    parser.add_argument("--benchmark-manifest", default="Data/esd_balanced_benchmark_manifest.csv")
    parser.add_argument("--converted-root", default="experiments/baselines/cyclegan_converted_full")
    parser.add_argument("--output-path", default="experiments/baselines/cyclegan_generated_full.csv")
    parser.add_argument("--targets", nargs="+", default=TARGET_EMOTIONS)
    parser.add_argument("--model-variant-prefix", default="cyclegan_local")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    benchmark = read_table(Path(args.benchmark_manifest))
    ensure_columns(benchmark, ["sample_id", "target_emotion"], "benchmark manifest")
    benchmark = benchmark.copy()
    benchmark["target_emotion_norm"] = benchmark["target_emotion"].astype(str).str.strip().str.lower()
    target_norm = {target.strip().lower() for target in args.targets}

    converted_root = Path(args.converted_root)
    rows = []
    missing_rows = []

    for item in benchmark.itertuples(index=False):
        target_emotion = str(item.target_emotion).strip().lower()
        if target_emotion not in target_norm:
            continue

        pair_tag = f"n2{target_emotion}"
        wav_path = (converted_root / pair_tag / f"{item.sample_id}.wav").resolve()
        model_variant = f"{args.model_variant_prefix}_{pair_tag}"

        if not wav_path.exists():
            missing_rows.append((item.sample_id, pair_tag, str(wav_path)))
            if args.strict:
                continue

        rows.append(
            {
                "sample_id": item.sample_id,
                "generated_wav": str(wav_path),
                "model_variant": model_variant,
                "alpha": pd.NA,
            }
        )

    out = pd.DataFrame(rows)
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"Wrote fixed generated manifest: {out_path}")
    print(f"Rows written: {len(out)}")
    print(f"Missing converted wavs: {len(missing_rows)}")

    if missing_rows:
        preview = pd.DataFrame(missing_rows[:10], columns=["sample_id", "pair_tag", "expected_wav"])
        print("Missing preview:")
        print(preview.to_string(index=False))
        if args.strict:
            raise FileNotFoundError(
                "Strict mode enabled and some converted wavs are missing. "
                "See missing preview above."
            )


if __name__ == "__main__":
    main()
