#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional

import matplotlib

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm


METRIC_COLUMNS = [
    "speaker_cosine",
    "speaker_gen_src_cosine",
    "speaker_gen_ref_cosine",
    "speaker_ref_cosine",
    "speaker_ref_output_cosine",
    "speaker_src_ref_cosine",
    "emotion_acc",
    "emotion_conf",
    "emotion_ref_cosine",
    "emotion_src_cosine",
    "mcd_gen_src",
    "mcd_gen_ref",
    "mcd_src_ref",
]


def load_project_eval_helpers() -> None:
    global EMOTION_TO_LABEL
    global aggregate_results
    global align_waveforms
    global build_metric_models
    global compute_emotion_metrics
    global compute_mcd
    global compute_speaker_cosine
    global load_eval_bundle
    global load_vocoder
    global mix_reference_waveform
    global resolve_path
    global run_reference_conversion
    global synthesize_waveform

    from eval_mel_viz import load_eval_bundle
    from scripts.run_balanced_benchmark_eval import (
        EMOTION_TO_LABEL,
        aggregate_results,
        align_waveforms,
        build_metric_models,
        compute_emotion_metrics,
        compute_mcd,
        compute_speaker_cosine,
        load_vocoder,
        mix_reference_waveform,
        resolve_path,
        run_reference_conversion,
        synthesize_waveform,
    )


def extract_epoch(path: Path) -> Optional[int]:
    match = re.search(r"epoch[_-]?(\d+)", path.stem)
    if match is None:
        match = re.search(r"epoch[_-]?(\d+)", path.parent.name)
    return int(match.group(1)) if match is not None else None


def discover_checkpoints(checkpoint_dir: Path, pattern: str, recursive: bool) -> list[Path]:
    iterator: Iterable[Path] = checkpoint_dir.rglob(pattern) if recursive else checkpoint_dir.glob(pattern)
    checkpoints = sorted(
        [path for path in iterator if path.is_file()],
        key=lambda path: (extract_epoch(path) is None, extract_epoch(path) or 0, str(path)),
    )
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint files matched {checkpoint_dir / pattern}")
    return checkpoints


def filter_checkpoints(
    checkpoints: list[Path],
    min_epoch: Optional[int],
    max_epoch: Optional[int],
) -> list[Path]:
    filtered = []
    for checkpoint in checkpoints:
        epoch = extract_epoch(checkpoint)
        if epoch is not None and min_epoch is not None and epoch < min_epoch:
            continue
        if epoch is not None and max_epoch is not None and epoch > max_epoch:
            continue
        filtered.append(checkpoint)
    if not filtered:
        raise FileNotFoundError("No checkpoints remained after applying epoch filters.")
    return filtered


def evaluate_checkpoint(
    checkpoint_path: Path,
    config_path: Path,
    manifest_df: pd.DataFrame,
    vocoder,
    speaker_model,
    emotion_model,
    args: argparse.Namespace,
) -> pd.DataFrame:
    bundle = load_eval_bundle(str(config_path), str(checkpoint_path), device=args.device)
    epoch = extract_epoch(checkpoint_path)
    model_variant = args.model_variant_template.format(
        epoch=epoch if epoch is not None else "unknown",
        epoch_padded=f"{epoch:05d}" if epoch is not None else "unknown",
        checkpoint_stem=checkpoint_path.stem,
    )
    print(f"Starting checkpoint {checkpoint_path.name} epoch={epoch}", flush=True)

    rows = []
    for sample_idx, item in enumerate(tqdm(
        manifest_df.itertuples(index=False),
        desc=f"Evaluating {checkpoint_path.name}",
        total=len(manifest_df),
        leave=True,
        dynamic_ncols=True,
        mininterval=10.0,
    ), 1):
        if args.progress_every > 0 and (sample_idx == 1 or sample_idx % args.progress_every == 0):
            print(
                f"{checkpoint_path.name}: sample {sample_idx}/{len(manifest_df)} "
                f"({getattr(item, 'sample_id', 'unknown')})",
                flush=True,
            )
        target_label = EMOTION_TO_LABEL[item.target_emotion]
        src_wav, ref_wav, mel_out, rank_info = run_reference_conversion(
            bundle=bundle,
            source_wav_path=item.source_wav,
            reference_wav_path=item.reference_wav,
            target_label=target_label,
            alpha=args.alpha,
            intensity_control=args.intensity_control,
            rank_search_steps=args.rank_search_steps,
        )
        out_wav = synthesize_waveform(vocoder, mel_out)

        speaker_gen_src_cosine = compute_speaker_cosine(speaker_model, out_wav, src_wav)
        src_ref_aligned, ref_aligned = align_waveforms(src_wav, ref_wav)
        ref_mix_wav = mix_reference_waveform(src_ref_aligned, ref_aligned, alpha=args.alpha)
        speaker_ref_cosine = compute_speaker_cosine(speaker_model, src_ref_aligned, ref_mix_wav)
        speaker_src_ref_cosine = compute_speaker_cosine(speaker_model, src_ref_aligned, ref_aligned)
        ref_out_aligned, out_ref_aligned = align_waveforms(ref_wav, out_wav)
        speaker_gen_ref_cosine = compute_speaker_cosine(speaker_model, out_ref_aligned, ref_out_aligned)
        emotion_metrics = compute_emotion_metrics(emotion_model, src_wav, ref_wav, out_wav, item.target_emotion)

        if args.disable_mcd:
            mcd_gen_src = None
            mcd_gen_ref = None
            mcd_src_ref = None
        else:
            mcd_gen_src = compute_mcd(out_wav, src_wav, sample_rate=bundle.sr)
            mcd_gen_ref = compute_mcd(out_wav, ref_wav, sample_rate=bundle.sr)
            mcd_src_ref = compute_mcd(src_wav, ref_wav, sample_rate=bundle.sr)

        row = {
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_name": checkpoint_path.name,
            "epoch": epoch,
            "sample_id": item.sample_id,
            "speaker_id": item.speaker_id,
            "text": item.text,
            "text_rank": item.text_rank,
            "source_file_id": item.source_file_id,
            "reference_file_id": item.reference_file_id,
            "source_wav": item.source_wav,
            "reference_wav": item.reference_wav,
            "target_emotion": item.target_emotion,
            "model_variant": model_variant,
            "intensity_mode": args.intensity_mode,
            "alpha": args.alpha,
            "intensity_control": args.intensity_control,
            "style_beta": rank_info["beta"],
            "rank_intensity": rank_info["rank_intensity"],
            "rank_src": rank_info["rank_src"],
            "rank_ref": rank_info["rank_ref"],
            "rank_gap": rank_info["rank_gap"],
            "speaker_gen_src_cosine": speaker_gen_src_cosine,
            "speaker_gen_ref_cosine": speaker_gen_ref_cosine,
            "speaker_src_ref_cosine": speaker_src_ref_cosine,
            "speaker_cosine": speaker_gen_src_cosine,
            "speaker_ref_cosine": speaker_ref_cosine,
            "speaker_ref_output_cosine": speaker_gen_ref_cosine,
            "mcd_gen_src": mcd_gen_src,
            "mcd_gen_ref": mcd_gen_ref,
            "mcd_src_ref": mcd_src_ref,
        }
        if hasattr(item, "reference_text"):
            row["reference_text"] = item.reference_text
        if hasattr(item, "reference_text_rank"):
            row["reference_text_rank"] = item.reference_text_rank
        if hasattr(item, "pair_type"):
            row["pair_type"] = item.pair_type
        row.update(emotion_metrics)
        rows.append(row)

    print(f"Finished checkpoint {checkpoint_path.name}", flush=True)
    return pd.DataFrame(rows)


def add_selection_scores(overall: pd.DataFrame, speaker_floor: Optional[float], lambda_speaker: float) -> pd.DataFrame:
    scored = overall.copy()
    scored["selection_score"] = scored["emotion_ref_cosine"] + lambda_speaker * scored["speaker_cosine"]
    if speaker_floor is not None:
        scored["passes_speaker_floor"] = scored["speaker_cosine"] >= speaker_floor
    else:
        scored["passes_speaker_floor"] = True
    return scored


def save_plots(overall: pd.DataFrame, by_emotion: pd.DataFrame, output_dir: Path) -> None:
    plot_df = overall.dropna(subset=["epoch"]).sort_values("epoch")
    if plot_df.empty:
        return

    metrics = [metric for metric in ["emotion_acc", "emotion_ref_cosine", "speaker_cosine", "emotion_conf"] if metric in plot_df]
    fig, ax = plt.subplots(figsize=(10, 6))
    for metric in metrics:
        ax.plot(plot_df["epoch"], plot_df[metric], marker="o", label=metric)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Metric value")
    ax.set_title("Validation metrics across checkpoints")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "metrics_across_epochs.png", dpi=160)
    plt.close(fig)

    if {"speaker_cosine", "emotion_ref_cosine"}.issubset(plot_df.columns):
        fig, ax = plt.subplots(figsize=(7, 6))
        scatter = ax.scatter(
            plot_df["speaker_cosine"],
            plot_df["emotion_ref_cosine"],
            c=plot_df["epoch"],
            cmap="viridis",
            s=60,
        )
        ax.plot(plot_df["speaker_cosine"], plot_df["emotion_ref_cosine"], alpha=0.5)
        ax.set_xlabel("Speaker cosine")
        ax.set_ylabel("Emotion-reference cosine")
        ax.set_title("Speaker-emotion checkpoint frontier")
        ax.grid(True, alpha=0.3)
        fig.colorbar(scatter, ax=ax, label="Epoch")
        fig.tight_layout()
        fig.savefig(output_dir / "speaker_emotion_frontier_by_epoch.png", dpi=160)
        plt.close(fig)

    if not by_emotion.empty and {"epoch", "target_emotion", "emotion_acc"}.issubset(by_emotion.columns):
        emo_df = by_emotion.dropna(subset=["epoch"]).sort_values(["target_emotion", "epoch"])
        fig, ax = plt.subplots(figsize=(10, 6))
        for emotion, group in emo_df.groupby("target_emotion"):
            ax.plot(group["epoch"], group["emotion_acc"], marker="o", label=str(emotion))
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Emotion accuracy")
        ax.set_title("Emotion accuracy by target emotion")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "emotion_acc_by_epoch_and_emotion.png", dpi=160)
        plt.close(fig)


def save_aggregates(per_sample: pd.DataFrame, output_dir: Path, speaker_floor: Optional[float], lambda_speaker: float) -> Dict[str, pd.DataFrame]:
    aggregates = aggregate_results(per_sample)
    for frame in aggregates.values():
        if "model_variant" in frame.columns:
            frame["epoch"] = frame["model_variant"].map(lambda value: extract_epoch(Path(str(value))))

    overall = aggregates["overall"]
    if "epoch" not in overall.columns or overall["epoch"].isna().all():
        variant_to_epoch = per_sample.groupby("model_variant")["epoch"].first().to_dict()
        for frame in aggregates.values():
            frame["epoch"] = frame["model_variant"].map(variant_to_epoch)

    aggregates["overall"] = add_selection_scores(overall, speaker_floor, lambda_speaker).sort_values("epoch")
    aggregates["by_speaker"] = aggregates["by_speaker"].sort_values(["epoch", "speaker_id"])
    aggregates["by_emotion"] = aggregates["by_emotion"].sort_values(["epoch", "target_emotion"])

    per_sample.to_csv(output_dir / "per_sample_metrics_all_epochs.csv", index=False)
    aggregates["overall"].to_csv(output_dir / "overall_metrics_by_epoch.csv", index=False)
    aggregates["by_speaker"].to_csv(output_dir / "by_speaker_metrics_by_epoch.csv", index=False)
    aggregates["by_emotion"].to_csv(output_dir / "by_emotion_metrics_by_epoch.csv", index=False)

    eligible = aggregates["overall"][aggregates["overall"]["passes_speaker_floor"]].copy()
    if not eligible.empty:
        best_rows = []
        for name, sort_cols in {
            "best_selection_score": ["selection_score", "emotion_ref_cosine"],
            "best_emotion_ref_cosine": ["emotion_ref_cosine", "speaker_cosine"],
            "best_emotion_acc": ["emotion_acc", "speaker_cosine"],
            "best_speaker_cosine": ["speaker_cosine", "emotion_ref_cosine"],
        }.items():
            row = eligible.sort_values(sort_cols, ascending=False).iloc[0].copy()
            row["selection_name"] = name
            best_rows.append(row)
        pd.DataFrame(best_rows).to_csv(output_dir / "best_checkpoints.csv", index=False)

    return aggregates


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a directory of StarGANv2-VC epoch checkpoints and summarize metrics across epochs."
    )
    parser.add_argument("--checkpoint-dir", required=True, help="Directory containing epoch checkpoint files.")
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint-pattern", default="epoch_*.pth")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--min-epoch", type=int, default=None)
    parser.add_argument("--max-epoch", type=int, default=None)
    parser.add_argument("--manifest-path", default="Data/esd_balanced_benchmark_manifest.csv")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--intensity-mode", choices=["native", "inferred", "fixed", "unknown"], default="native")
    parser.add_argument("--intensity-control", choices=["style", "rank"], default="style")
    parser.add_argument("--rank-search-steps", type=int, default=101)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--speaker-model-dir", default="pretrained_models/spkrec-ecapa-voxceleb")
    parser.add_argument("--emotion-model-dir", default="pretrained_models/wavlm_ser_esd")
    parser.add_argument("--vocoder-path", default="Vocoder/checkpoint-400000steps.pkl")
    parser.add_argument("--model-variant-template", default="epoch_{epoch_padded}")
    parser.add_argument("--speaker-floor", type=float, default=None)
    parser.add_argument("--lambda-speaker", type=float, default=1.0)
    parser.add_argument("--disable-mcd", action="store_true", help="Skip MCD computation to make checkpoint sweeps faster.")
    parser.add_argument("--progress-every", type=int, default=10, help="Print one progress line every N samples; use 0 to disable.")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    config_path = Path(args.config_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    load_project_eval_helpers()

    repo_root = config_path.parent.parent
    manifest_path = resolve_path(args.manifest_path, repo_root)
    vocoder_path = resolve_path(args.vocoder_path, repo_root)
    speaker_model_dir = resolve_path(args.speaker_model_dir, repo_root)
    emotion_model_dir = resolve_path(args.emotion_model_dir, repo_root) if args.emotion_model_dir else None

    checkpoints = discover_checkpoints(checkpoint_dir, args.checkpoint_pattern, args.recursive)
    checkpoints = filter_checkpoints(checkpoints, args.min_epoch, args.max_epoch)
    manifest_df = pd.read_csv(manifest_path, sep="\t")
    if args.limit is not None:
        manifest_df = manifest_df.head(args.limit).copy()

    bootstrap_bundle = load_eval_bundle(str(config_path), str(checkpoints[0]), device=args.device)
    vocoder = load_vocoder(vocoder_path, bootstrap_bundle.device)
    speaker_model, emotion_model = build_metric_models(bootstrap_bundle.device, speaker_model_dir, emotion_model_dir)

    all_results = []
    for checkpoint_path in tqdm(checkpoints, desc="Checkpoints"):
        checkpoint_df = evaluate_checkpoint(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            manifest_df=manifest_df,
            vocoder=vocoder,
            speaker_model=speaker_model,
            emotion_model=emotion_model,
            args=args,
        )
        checkpoint_df.to_csv(output_dir / f"{checkpoint_path.stem}_per_sample_metrics.csv", index=False)
        all_results.append(checkpoint_df)

    per_sample = pd.concat(all_results, ignore_index=True)
    aggregates = save_aggregates(
        per_sample=per_sample,
        output_dir=output_dir,
        speaker_floor=args.speaker_floor,
        lambda_speaker=args.lambda_speaker,
    )
    save_plots(aggregates["overall"], aggregates["by_emotion"], output_dir)

    print(f"Evaluated {len(checkpoints)} checkpoints on {len(manifest_df)} samples each.")
    print(f"Wrote combined CSVs and plots to: {output_dir}")


if __name__ == "__main__":
    main()
