#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_METRICS = [
    "speaker_cosine",
    "speaker_ref_cosine",
    "speaker_ref_output_cosine",
    "speaker_gen_ref_cosine",
    "speaker_gen_src_cosine",
    "speaker_src_ref_cosine",
    "mcd_gen_ref",
    "mcd_gen_src",
    "mcd_src_ref",
    "emotion_acc",
    "emotion_conf",
    "emotion_ref_cosine",
    "emotion_src_cosine",
]

METRIC_LABELS = {
    "speaker_cosine": "Speaker Cosine",
    "speaker_ref_cosine": "Speaker Source-Reference Cosine",
    "speaker_ref_output_cosine": "Speaker Reference-Output Cosine",
    "speaker_gen_ref_cosine": "Speaker Generated-Reference Cosine",
    "speaker_gen_src_cosine": "Speaker Generated-Source Cosine",
    "speaker_src_ref_cosine": "Speaker Source-Reference Cosine",
    "mcd_gen_ref": "MCD Generated-Reference",
    "mcd_gen_src": "MCD Generated-Source",
    "mcd_src_ref": "MCD Source-Reference",
    "emotion_acc": "Emotion Accuracy",
    "emotion_conf": "Target Emotion Confidence",
    "emotion_ref_cosine": "Emotion Reference Cosine",
    "emotion_src_cosine": "Emotion Source Cosine",
}

METRIC_GROUPS = {
    "emotion": ["emotion_acc", "emotion_conf", "emotion_ref_cosine", "emotion_src_cosine"],
    "speaker": [
        "speaker_gen_src_cosine",
        "speaker_gen_ref_cosine",
        "speaker_src_ref_cosine",
        "speaker_cosine",
        "speaker_ref_cosine",
        "speaker_ref_output_cosine",
    ],
    "mcd": ["mcd_gen_src", "mcd_gen_ref", "mcd_src_ref"],
}


def present_group_metrics(metrics: list[str], group_name: str) -> list[str]:
    return [metric for metric in METRIC_GROUPS[group_name] if metric in metrics]


def minmax_normalize(frame: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    normalized = frame.copy()
    for metric in metrics:
        values = pd.to_numeric(normalized[metric], errors="coerce")
        min_value = values.min(skipna=True)
        max_value = values.max(skipna=True)
        if pd.isna(min_value) or pd.isna(max_value):
            normalized[metric] = pd.NA
        elif max_value == min_value:
            normalized[metric] = 0.5
        else:
            normalized[metric] = (values - min_value) / (max_value - min_value)
    return normalized


def save_metric_family_plot(
    summary: pd.DataFrame,
    metrics: list[str],
    out_dir: Path,
    title: str,
    filename: str,
    normalized: bool = False,
) -> None:
    if not metrics:
        return
    plot_df = minmax_normalize(summary, metrics) if normalized else summary
    fig, ax = plt.subplots(figsize=(10, 6))
    for metric in metrics:
        ax.plot(plot_df["alpha"], plot_df[metric], marker="o", linewidth=2, label=METRIC_LABELS.get(metric, metric))
    ax.set_xlabel("Inferred intensity alpha")
    ax.set_ylabel("Normalized metric value" if normalized else "Mean score")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=240, bbox_inches="tight")
    plt.close(fig)


def safe_name(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in str(value)).strip("_")


def resolve_tradeoff_speaker_metric(frame: pd.DataFrame) -> tuple[str, str] | None:
    if "speaker_gen_src_cosine" in frame.columns:
        return "speaker_gen_src_cosine", "Speaker Generated-Source Cosine"
    if "speaker_cosine" in frame.columns:
        return "speaker_cosine", "Speaker Cosine"
    return None


def read_metrics(eval_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    per_sample_path = eval_dir / "per_sample_metrics.csv"
    by_emotion_path = eval_dir / "by_emotion_metrics.csv"
    by_speaker_path = eval_dir / "by_speaker_metrics.csv"

    missing = [path for path in [per_sample_path, by_emotion_path, by_speaker_path] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing metric files: {missing}")

    return (
        pd.read_csv(per_sample_path),
        pd.read_csv(by_emotion_path),
        pd.read_csv(by_speaker_path),
    )


def metric_columns(frame: pd.DataFrame) -> list[str]:
    return [metric for metric in DEFAULT_METRICS if metric in frame.columns]


def prepare_alpha(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "alpha" not in frame.columns:
        raise KeyError("metrics must include an alpha column")
    frame["alpha"] = pd.to_numeric(frame["alpha"], errors="coerce")
    return frame.dropna(subset=["alpha"]).sort_values("alpha")


def save_overall_plot(per_sample: pd.DataFrame, out_dir: Path, title_prefix: str) -> pd.DataFrame:
    metrics = metric_columns(per_sample)
    summary = (
        prepare_alpha(per_sample)
        .groupby("alpha", as_index=False)[metrics]
        .mean()
        .sort_values("alpha")
    )
    summary.to_csv(out_dir / "alpha_metrics_summary.csv", index=False)

    for group_name in METRIC_GROUPS:
        group_dir = out_dir / group_name
        group_dir.mkdir(parents=True, exist_ok=True)
        group_metrics = present_group_metrics(metrics, group_name)
        save_metric_family_plot(
            summary,
            group_metrics,
            group_dir,
            f"{title_prefix}: {group_name.capitalize()} Metrics Across Alpha Values",
            f"alpha_{group_name}_metrics_comparison.png",
        )

    normalized_dir = out_dir / "normalized"
    normalized_dir.mkdir(parents=True, exist_ok=True)
    save_metric_family_plot(
        summary,
        metrics,
        normalized_dir,
        f"{title_prefix}: All Metrics Across Alpha Values (min-max normalized)",
        "alpha_metrics_comparison_normalized.png",
        normalized=True,
    )

    return summary


def save_by_emotion_plots(by_emotion: pd.DataFrame, out_dir: Path, title_prefix: str) -> pd.DataFrame:
    by_emotion = prepare_alpha(by_emotion)
    metrics = metric_columns(by_emotion)
    by_emotion.to_csv(out_dir / "alpha_metrics_by_emotion_summary.csv", index=False)
    emotions = by_emotion["target_emotion"].dropna().drop_duplicates().tolist()

    if not emotions:
        return by_emotion

    for group_name in METRIC_GROUPS:
        group_metrics = present_group_metrics(metrics, group_name)
        if not group_metrics:
            continue
        group_dir = out_dir / group_name
        group_dir.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(len(emotions), 1, figsize=(10, 4 * len(emotions)), squeeze=False)
        for ax, emotion in zip(axes.ravel(), emotions):
            emotion_df = by_emotion[by_emotion["target_emotion"] == emotion].sort_values("alpha")
            for metric in group_metrics:
                ax.plot(
                    emotion_df["alpha"],
                    emotion_df[metric],
                    marker="o",
                    linewidth=2,
                    label=METRIC_LABELS.get(metric, metric),
                )
            ax.set_title(f"{emotion}: {group_name} metrics across alpha values")
            ax.set_xlabel("Inferred intensity alpha")
            ax.set_ylabel("Mean score")
            ax.grid(True, alpha=0.3)
            ax.legend()
        fig.suptitle(f"{title_prefix}: Per-Emotion {group_name.capitalize()} Metrics", y=1.002)
        fig.tight_layout()
        fig.savefig(group_dir / f"alpha_{group_name}_metrics_by_emotion.png", dpi=240, bbox_inches="tight")
        plt.close(fig)

    normalized_dir = out_dir / "normalized"
    normalized_dir.mkdir(parents=True, exist_ok=True)
    normalized_by_emotion = minmax_normalize(by_emotion, metrics)
    fig, axes = plt.subplots(len(emotions), 1, figsize=(10, 4 * len(emotions)), squeeze=False)
    for ax, emotion in zip(axes.ravel(), emotions):
        emotion_df = normalized_by_emotion[normalized_by_emotion["target_emotion"] == emotion].sort_values("alpha")
        for metric in metrics:
            ax.plot(emotion_df["alpha"], emotion_df[metric], marker="o", linewidth=2, label=METRIC_LABELS.get(metric, metric))
        ax.set_title(f"{emotion}: all metrics across alpha values")
        ax.set_xlabel("Inferred intensity alpha")
        ax.set_ylabel("Normalized metric value")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle(f"{title_prefix}: Per-Emotion Alpha Metrics (min-max normalized)", y=1.002)
    fig.tight_layout()
    fig.savefig(normalized_dir / "alpha_metrics_by_emotion_normalized.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    metric_plot_specs = {
        "emotion_acc": "alpha_emotion_acc_by_emotion.png",
        "emotion_ref_cosine": "alpha_emotion_ref_cosine_by_emotion.png",
        "emotion_src_cosine": "alpha_emotion_src_cosine_by_emotion.png",
        "speaker_cosine": "alpha_speaker_cosine_by_emotion.png",
        "speaker_ref_cosine": "alpha_speaker_ref_cosine_by_emotion.png",
        "speaker_ref_output_cosine": "alpha_speaker_ref_output_cosine_by_emotion.png",
        "speaker_gen_ref_cosine": "alpha_speaker_gen_ref_cosine_by_emotion.png",
        "speaker_gen_src_cosine": "alpha_speaker_gen_src_cosine_by_emotion.png",
        "speaker_src_ref_cosine": "alpha_speaker_src_ref_cosine_by_emotion.png",
        "mcd_gen_ref": "alpha_mcd_gen_ref_by_emotion.png",
        "mcd_gen_src": "alpha_mcd_gen_src_by_emotion.png",
        "mcd_src_ref": "alpha_mcd_src_ref_by_emotion.png",
        "emotion_conf": "alpha_emotion_conf_by_emotion.png",
    }
    for metric, filename in metric_plot_specs.items():
        if metric not in metrics:
            continue
        group_name = next(group for group, group_metrics in METRIC_GROUPS.items() if metric in group_metrics)
        group_dir = out_dir / group_name
        group_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        for emotion in emotions:
            emotion_df = by_emotion[by_emotion["target_emotion"] == emotion].sort_values("alpha")
            ax.plot(emotion_df["alpha"], emotion_df[metric], marker="o", linewidth=2, label=emotion)
        ax.set_xlabel("Inferred intensity alpha")
        ax.set_ylabel(METRIC_LABELS.get(metric, metric))
        ax.set_title(f"{title_prefix}: {METRIC_LABELS.get(metric, metric)} by Emotion")
        ax.grid(True, alpha=0.3)
        ax.legend(title="Emotion")
        fig.tight_layout()
        fig.savefig(group_dir / filename, dpi=240, bbox_inches="tight")
        plt.close(fig)

    return by_emotion


def save_by_speaker_plot(by_speaker: pd.DataFrame, out_dir: Path, title_prefix: str) -> pd.DataFrame:
    by_speaker = prepare_alpha(by_speaker)
    metrics = metric_columns(by_speaker)
    by_speaker.to_csv(out_dir / "alpha_metrics_by_speaker_summary.csv", index=False)

    for group_name in METRIC_GROUPS:
        key_metrics = present_group_metrics(metrics, group_name)
        if not key_metrics:
            continue
        group_dir = out_dir / group_name
        group_dir.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(len(key_metrics), 1, figsize=(11, 4 * len(key_metrics)), squeeze=False)
        for ax, metric in zip(axes.ravel(), key_metrics):
            pivot = by_speaker.pivot_table(index="alpha", columns="speaker_id", values=metric, aggfunc="mean").sort_index()
            for speaker_id in pivot.columns:
                ax.plot(pivot.index, pivot[speaker_id], marker="o", linewidth=1.5, alpha=0.8, label=str(speaker_id))
            ax.set_xlabel("Inferred intensity alpha")
            ax.set_ylabel(METRIC_LABELS.get(metric, metric))
            ax.set_title(f"{METRIC_LABELS.get(metric, metric)} by Speaker")
            ax.grid(True, alpha=0.3)
            ax.legend(title="Speaker", ncol=5, fontsize=8)
        fig.suptitle(f"{title_prefix}: Per-Speaker {group_name.capitalize()} Metrics", y=1.002)
        fig.tight_layout()
        fig.savefig(group_dir / f"alpha_{group_name}_metrics_by_speaker.png", dpi=240, bbox_inches="tight")
        plt.close(fig)

    return by_speaker


def save_tradeoff_plot(summary: pd.DataFrame, out_dir: Path, title_prefix: str) -> None:
    resolved = resolve_tradeoff_speaker_metric(summary)
    if resolved is None or "emotion_ref_cosine" not in summary.columns:
        return
    speaker_metric, speaker_label = resolved

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        summary[speaker_metric],
        summary["emotion_ref_cosine"],
        c=summary["alpha"],
        cmap="viridis",
        s=120,
        zorder=3,
    )
    ax.plot(summary[speaker_metric], summary["emotion_ref_cosine"], linewidth=2, alpha=0.8, zorder=2)
    for row in summary.itertuples(index=False):
        ax.text(getattr(row, speaker_metric) + 0.003, row.emotion_ref_cosine + 0.006, f"a={row.alpha:.1f}", fontsize=9)
    ax.set_xlabel(speaker_label)
    ax.set_ylabel("Emotion Reference Cosine")
    ax.set_title(f"{title_prefix}: Speaker-Emotion Frontier")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "tradeoff_frontier.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def save_tradeoff_plots_by_emotion(by_emotion: pd.DataFrame, out_dir: Path, title_prefix: str) -> None:
    by_emotion = prepare_alpha(by_emotion)
    resolved = resolve_tradeoff_speaker_metric(by_emotion)
    if resolved is None or "emotion_ref_cosine" not in by_emotion.columns:
        return
    speaker_metric, speaker_label = resolved
    emotions = by_emotion["target_emotion"].dropna().drop_duplicates().tolist()
    if not emotions:
        return

    tradeoff_dir = out_dir / "tradeoff"
    per_emotion_dir = tradeoff_dir / "by_emotion"
    per_emotion_dir.mkdir(parents=True, exist_ok=True)

    for emotion in emotions:
        emotion_df = (
            by_emotion[by_emotion["target_emotion"] == emotion]
            .dropna(subset=["alpha", speaker_metric, "emotion_ref_cosine"])
            .sort_values("alpha")
        )
        if emotion_df.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(
            emotion_df[speaker_metric],
            emotion_df["emotion_ref_cosine"],
            c=emotion_df["alpha"],
            cmap="viridis",
            s=110,
            zorder=3,
        )
        ax.plot(emotion_df[speaker_metric], emotion_df["emotion_ref_cosine"], linewidth=2, alpha=0.8, zorder=2)
        for row in emotion_df.itertuples(index=False):
            ax.text(getattr(row, speaker_metric) + 0.003, row.emotion_ref_cosine + 0.006, f"a={row.alpha:.1f}", fontsize=9)
        ax.set_xlabel(speaker_label)
        ax.set_ylabel("Emotion Reference Cosine")
        ax.set_title(f"{title_prefix}: {emotion} Frontier")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(per_emotion_dir / f"tradeoff_frontier_{safe_name(emotion)}.png", dpi=240, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(len(emotions), 1, figsize=(8, 5 * max(len(emotions), 1)), squeeze=False)
    axes = axes.ravel()
    for ax, emotion in zip(axes, emotions):
        emotion_df = (
            by_emotion[by_emotion["target_emotion"] == emotion]
            .dropna(subset=["alpha", speaker_metric, "emotion_ref_cosine"])
            .sort_values("alpha")
        )
        if emotion_df.empty:
            ax.set_visible(False)
            continue
        ax.scatter(
            emotion_df[speaker_metric],
            emotion_df["emotion_ref_cosine"],
            c=emotion_df["alpha"],
            cmap="viridis",
            s=90,
            zorder=3,
        )
        ax.plot(emotion_df[speaker_metric], emotion_df["emotion_ref_cosine"], linewidth=2, alpha=0.8, zorder=2)
        for row in emotion_df.itertuples(index=False):
            ax.text(getattr(row, speaker_metric) + 0.003, row.emotion_ref_cosine + 0.006, f"{row.alpha:.1f}", fontsize=8)
        ax.set_xlabel(speaker_label)
        ax.set_ylabel("Emotion Reference Cosine")
        ax.set_title(str(emotion))
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{title_prefix}: Speaker-Emotion Frontiers by Emotion", y=1.002)
    fig.subplots_adjust(top=0.94, hspace=0.45)
    fig.savefig(tradeoff_dir / "tradeoff_frontiers_by_emotion.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot inferred-intensity alpha metrics from one evaluation directory.")
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--title-prefix", default="CycleGAN Inferred Intensity")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    out_dir = Path(args.out_dir) if args.out_dir else eval_dir / "alpha_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    per_sample, by_emotion, by_speaker = read_metrics(eval_dir)
    summary = save_overall_plot(per_sample, out_dir, args.title_prefix)
    save_by_emotion_plots(by_emotion, out_dir, args.title_prefix)
    save_by_speaker_plot(by_speaker, out_dir, args.title_prefix)
    save_tradeoff_plot(summary, out_dir, args.title_prefix)
    save_tradeoff_plots_by_emotion(by_emotion, out_dir, args.title_prefix)

    print(f"Saved alpha metric plots to: {out_dir}")


if __name__ == "__main__":
    main()
