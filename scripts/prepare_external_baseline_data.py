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


def materialize_audio_dir(file_df: pd.DataFrame, out_dir: Path, link_mode: str) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    created = 0
    for row in file_df.itertuples(index=False):
        src = Path(row.wav_path)
        if not src.exists():
            raise FileNotFoundError(f"Missing audio file: {src}")
        # Keep deterministic and collision-safe file names.
        dst_name = f"spk{int(row.speaker_id):02d}_{row.file_id}.wav"
        dst = out_dir / dst_name
        relink_or_copy(src.resolve(), dst, link_mode)
        created += 1
    return created


def write_list(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(df["wav_path"].astype(str).tolist()) + "\n", encoding="utf-8")


def build_pair_tables(pool: pd.DataFrame, target_emotion: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    neutral = pool[pool["emotion"] == "Neutral"].copy()
    target = pool[pool["emotion"] == target_emotion].copy()

    neutral = neutral.rename(columns={"wav_path": "source_wav", "file_id": "source_file_id"})
    target = target.rename(columns={"wav_path": "target_wav", "file_id": "target_file_id"})

    merged = neutral.merge(
        target,
        on=["speaker_id", "text"],
        how="inner",
        suffixes=("_src", "_trg"),
    )
    paired = merged[[
        "speaker_id",
        "text",
        "source_file_id",
        "source_wav",
        "target_file_id",
        "target_wav",
    ]].copy()
    return neutral, paired


def make_eval_template(
    benchmark_manifest: pd.DataFrame,
    output_path: Path,
    target_emotion: str,
    suggested_variant: str,
) -> None:
    subset = benchmark_manifest[benchmark_manifest["target_emotion"] == target_emotion].copy()
    subset = subset[["sample_id", "target_emotion", "source_wav", "reference_wav"]]
    subset["model_variant"] = suggested_variant
    subset["intensity_mode"] = "fixed"
    subset["alpha"] = pd.NA
    subset["generated_wav"] = ""
    subset.to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare neutral->emotion baseline data packs for external CycleGAN-EVC and VAWGAN-EVC training."
    )
    parser.add_argument("--train-metadata", default="Data/esd_balanced_train.csv")
    parser.add_argument("--benchmark-pool", default="Data/esd_balanced_benchmark_pool.csv")
    parser.add_argument("--benchmark-manifest", default="Data/esd_balanced_benchmark_manifest.csv")
    parser.add_argument("--output-root", default="Data/external_baselines")
    parser.add_argument("--targets", nargs="+", default=TARGET_EMOTIONS)
    parser.add_argument(
        "--materialize-audio",
        action="store_true",
        help="Create link/copy directories expected by legacy baseline repos.",
    )
    parser.add_argument(
        "--link-mode",
        choices=["symlink", "hardlink", "copy"],
        default="symlink",
        help="How to materialize wav files when --materialize-audio is set.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing output root.")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    if output_root.exists() and args.force:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    train_df = read_table(Path(args.train_metadata))
    bench_pool_df = read_table(Path(args.benchmark_pool))
    benchmark_manifest = read_table(Path(args.benchmark_manifest))

    required_cols = ["file_id", "speaker_id", "text", "emotion", "wav_path"]
    ensure_columns(train_df, required_cols, "train metadata")
    ensure_columns(bench_pool_df, required_cols, "benchmark pool")
    ensure_columns(
        benchmark_manifest,
        ["sample_id", "target_emotion", "source_wav", "reference_wav"],
        "benchmark manifest",
    )

    summary_rows = []

    for target in args.targets:
        pair_tag = f"n2{target.lower()}"

        cyc_root = output_root / "cyclegan" / pair_tag
        vaw_root = output_root / "vawgan" / pair_tag

        for root in [cyc_root, vaw_root]:
            root.mkdir(parents=True, exist_ok=True)

        train_neutral = train_df[train_df["emotion"] == "Neutral"].copy()
        train_target = train_df[train_df["emotion"] == target].copy()
        val_neutral = bench_pool_df[bench_pool_df["emotion"] == "Neutral"].copy()
        val_target = bench_pool_df[bench_pool_df["emotion"] == target].copy()

        if train_neutral.empty or train_target.empty:
            raise ValueError(f"No train rows found for pair Neutral->{target}")

        write_list(train_neutral, cyc_root / "train_A.list")
        write_list(train_target, cyc_root / "train_B.list")
        write_list(val_neutral, cyc_root / "validation_A.list")
        write_list(val_target, cyc_root / "validation_B.list")

        train_pair_neutral, train_pairs = build_pair_tables(train_df, target)
        _, eval_pairs = build_pair_tables(bench_pool_df, target)
        train_pairs.to_csv(vaw_root / "train_pairs.tsv", sep="\t", index=False)
        eval_pairs.to_csv(vaw_root / "eval_pairs.tsv", sep="\t", index=False)
        train_pair_neutral[["speaker_id", "text", "source_file_id", "source_wav"]].to_csv(
            vaw_root / "train_neutral_index.tsv", sep="\t", index=False
        )

        template_variant = f"cyclegan_evc_{pair_tag}"
        make_eval_template(
            benchmark_manifest=benchmark_manifest,
            output_path=cyc_root / "generated_manifest_template.csv",
            target_emotion=target,
            suggested_variant=template_variant,
        )
        make_eval_template(
            benchmark_manifest=benchmark_manifest,
            output_path=vaw_root / "generated_manifest_template.csv",
            target_emotion=target,
            suggested_variant=f"vawgan_evc_{pair_tag}",
        )

        cyc_materialized = 0
        vaw_materialized = 0
        if args.materialize_audio:
            cyc_materialized += materialize_audio_dir(train_neutral, cyc_root / "train_A", args.link_mode)
            cyc_materialized += materialize_audio_dir(train_target, cyc_root / "train_B", args.link_mode)
            cyc_materialized += materialize_audio_dir(val_neutral, cyc_root / "validation_A", args.link_mode)
            cyc_materialized += materialize_audio_dir(val_target, cyc_root / "validation_B", args.link_mode)

            # VAWGAN repos often consume pairwise folders directly; these mirror neutral/target pools.
            vaw_materialized += materialize_audio_dir(train_neutral, vaw_root / "train_neutral", args.link_mode)
            vaw_materialized += materialize_audio_dir(train_target, vaw_root / "train_target", args.link_mode)
            vaw_materialized += materialize_audio_dir(val_neutral, vaw_root / "eval_neutral", args.link_mode)
            vaw_materialized += materialize_audio_dir(val_target, vaw_root / "eval_target", args.link_mode)

        summary_rows.append(
            {
                "pair": pair_tag,
                "train_neutral": len(train_neutral),
                "train_target": len(train_target),
                "val_neutral": len(val_neutral),
                "val_target": len(val_target),
                "vaw_train_pairs": len(train_pairs),
                "vaw_eval_pairs": len(eval_pairs),
                "cyc_materialized": cyc_materialized,
                "vaw_materialized": vaw_materialized,
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_root / "baseline_data_prep_summary.csv", index=False)

    # Multi-emotion conditional manifest for VAWGAN-style training.
    conditional = train_df[train_df["emotion"].isin(["Neutral", *args.targets])].copy()
    conditional.to_csv(output_root / "vawgan" / "conditional_train.tsv", sep="\t", index=False)

    print("Prepared external baseline data packs.")
    print(f"Output root: {output_root}")
    print(summary.to_string(index=False))
    print("Wrote summary CSV:", output_root / "baseline_data_prep_summary.csv")


if __name__ == "__main__":
    main()
