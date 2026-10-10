#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

EMOTION_ORDER = ["Angry", "Happy", "Sad", "Surprise"]
EMOTION_PALETTE = {
    "Angry": "#d1495b",
    "Happy": "#edae49",
    "Sad": "#00798c",
    "Surprise": "#6a4c93",
}
PRIMARY_COLOR_SETTINGS = {"color":"#457b9d", "alpha":0.95}

METRIC_PALETTE = {
    "Speaker Cosine Similarity": "#2a9d8f",
    "Generated-Source Speaker Cosine": "#2a9d8f",
    "Generated-Reference Speaker Cosine": "#264653",
    "Source-Reference Speaker Cosine": "#6c757d",
    "Emotion Recognition Accuracy": "#e76f51",
    "Target Emotion Confidence": "#f4a261",
    "Emotion Reference Cosine Similarity": "#457b9d",
    "Emotion Source Cosine Similarity": "#8ecae6",
    "MCD Generated-Source": "#7b2cbf",
    "MCD Generated-Reference": "#c77dff",
    "MCD Source-Reference": "#6c757d",
}

EMOTION_ONLY_METRICS = [
    ("emotion_acc", "Emotion Recognition Accuracy"),
    ("emotion_conf", "Target Emotion Confidence"),
    ("emotion_ref_cosine", "Emotion Reference Cosine Similarity"),
    ("emotion_src_cosine", "Emotion Source Cosine Similarity"),
]

SPEAKER_METRICS = [
    ("speaker_gen_src_cosine", "Generated-Source Speaker Cosine"),
    ("speaker_gen_ref_cosine", "Generated-Reference Speaker Cosine"),
    ("speaker_src_ref_cosine", "Source-Reference Speaker Cosine"),
    ("speaker_cosine", "Speaker Cosine Similarity"),
]

MCD_METRICS = [
    ("mcd_gen_src", "MCD Generated-Source"),
    ("mcd_gen_ref", "MCD Generated-Reference"),
    ("mcd_src_ref", "MCD Source-Reference"),
]


def load_frames(experiment_dir: Path):
    return {
        "overall": pd.read_csv(experiment_dir / "overall_metrics.csv"),
        "by_emotion": pd.read_csv(experiment_dir / "by_emotion_metrics.csv"),
        "by_speaker": pd.read_csv(experiment_dir / "by_speaker_metrics.csv"),
        "per_sample": pd.read_csv(experiment_dir / "per_sample_metrics.csv"),
    }


def _available_metrics(frame: pd.DataFrame, metric_specs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    return [(column, label) for column, label in metric_specs if column in frame.columns]


def _minmax_normalize(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    normalized = frame.copy()
    for column in columns:
        values = pd.to_numeric(normalized[column], errors="coerce")
        min_value = values.min(skipna=True)
        max_value = values.max(skipna=True)
        if pd.isna(min_value) or pd.isna(max_value):
            normalized[column] = pd.NA
        elif max_value == min_value:
            normalized[column] = 0.5
        else:
            normalized[column] = (values - min_value) / (max_value - min_value)
    return normalized


def save_tradeoff_by_emotion_plot(per_sample: pd.DataFrame, by_emotion: pd.DataFrame, output_dir: Path) -> None:
    speaker_col = "speaker_gen_src_cosine" if "speaker_gen_src_cosine" in per_sample.columns else "speaker_cosine"
    fig, ax = plt.subplots(figsize=(9, 7))

    sns.scatterplot(
        data=per_sample,
        x=speaker_col,
        y="emotion_ref_cosine",
        hue="target_emotion",
        hue_order=EMOTION_ORDER,
        palette=EMOTION_PALETTE,
        alpha=0.35,
        s=55,
        ax=ax,
    )

    sns.scatterplot(
        data=by_emotion,
        x=speaker_col,
        y="emotion_ref_cosine",
        hue="target_emotion",
        hue_order=EMOTION_ORDER,
        palette=EMOTION_PALETTE,
        s=220,
        marker="X",
        legend=False,
        ax=ax,
    )

    for row in by_emotion.itertuples(index=False):
        ax.text(getattr(row, speaker_col) + 0.005, row.emotion_ref_cosine + 0.005, row.target_emotion, fontsize=10)

    first = by_emotion.iloc[0]
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Generated-Source Speaker Cosine Similarity" if speaker_col == "speaker_gen_src_cosine" else "Speaker Cosine Similarity")
    ax.set_ylabel("Emotion Reference Cosine Similarity")
    ax.set_title(f"Speaker-Emotion Tradeoff per Emotion Label\n{first['model_variant']} | alpha={first['alpha']}")
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output_dir / "speaker_emotion_tradeoff_per_emotion_label.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_normalized_overview(overall: pd.DataFrame, output_dir: Path) -> None:
    row = overall.iloc[0]
    metric_specs = (
        _available_metrics(overall, EMOTION_ONLY_METRICS)
        + _available_metrics(overall, SPEAKER_METRICS)
        + _available_metrics(overall, MCD_METRICS)
    )
    if not metric_specs:
        return

    columns = [column for column, _ in metric_specs]
    normalized = _minmax_normalize(overall, columns)
    labels = [label for _, label in metric_specs]
    values = [normalized.iloc[0][column] for column in columns]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(labels, values, **PRIMARY_COLOR_SETTINGS)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Normalized mean score")
    ax.set_title(f"All Metrics: Mean Values (min-max normalized)\n{row['model_variant']} | alpha={row['alpha']}")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", rotation=25)

    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.3f}", ha="center", va="bottom")

    fig.tight_layout()
    fig.savefig(output_dir / "normalized_metrics_means.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_metric_family_bars(
    overall: pd.DataFrame,
    metric_specs: list[tuple[str, str]],
    output_dir: Path,
    filename: str,
    title: str,
    ylabel: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    available = _available_metrics(overall, metric_specs)
    if not available:
        return

    row = overall.iloc[0]
    labels = [label for _, label in available]
    values = [row[column] for column, _ in available]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values, **PRIMARY_COLOR_SETTINGS)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\n{row['model_variant']} | alpha={row['alpha']}")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", rotation=15)

    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.3f}", ha="center", va="bottom")

    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_emotion_metrics_plot(by_emotion: pd.DataFrame, output_dir: Path) -> None:
    available = _available_metrics(by_emotion, EMOTION_ONLY_METRICS)
    if not available:
        return
    melted = by_emotion.melt(
        id_vars=["target_emotion", "model_variant", "alpha"],
        value_vars=[column for column, _ in available],
        var_name="metric",
        value_name="score",
    )
    metric_labels = {column: label for column, label in available}
    melted["metric"] = melted["metric"].map(metric_labels)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=melted,
        x="target_emotion",
        y="score",
        hue="metric",
        order=EMOTION_ORDER,
        palette=METRIC_PALETTE,
        ax=ax,
    )
    first = by_emotion.iloc[0]
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Emotion label")
    ax.set_ylabel("Mean score")
    ax.set_title(f"Emotion Metrics per Emotion Label\n{first['model_variant']} | alpha={first['alpha']}")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(title="")
    fig.tight_layout()
    fig.savefig(output_dir / "emotion_metrics_per_label.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_speaker_metrics_by_emotion_plot(by_emotion: pd.DataFrame, output_dir: Path) -> None:
    available = _available_metrics(by_emotion, SPEAKER_METRICS)
    if not available:
        return
    melted = by_emotion.melt(
        id_vars=["target_emotion", "model_variant", "alpha"],
        value_vars=[column for column, _ in available],
        var_name="metric",
        value_name="score",
    )
    metric_labels = {column: label for column, label in available}
    melted["metric"] = melted["metric"].map(metric_labels)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=melted,
        x="target_emotion",
        y="score",
        hue="metric",
        order=EMOTION_ORDER,
        palette=METRIC_PALETTE,
        ax=ax,
    )
    first = by_emotion.iloc[0]
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Emotion label")
    ax.set_ylabel("Mean speaker cosine")
    ax.set_title(f"Speaker Metrics per Emotion Label\n{first['model_variant']} | alpha={first['alpha']}")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(title="")
    fig.tight_layout()
    fig.savefig(output_dir / "speaker_metrics_per_emotion_label.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_mcd_metrics_by_emotion_plot(by_emotion: pd.DataFrame, output_dir: Path) -> None:
    available = _available_metrics(by_emotion, MCD_METRICS)
    if not available:
        return
    melted = by_emotion.melt(
        id_vars=["target_emotion", "model_variant", "alpha"],
        value_vars=[column for column, _ in available],
        var_name="metric",
        value_name="score",
    )
    metric_labels = {column: label for column, label in available}
    melted["metric"] = melted["metric"].map(metric_labels)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=melted,
        x="target_emotion",
        y="score",
        hue="metric",
        order=EMOTION_ORDER,
        palette=METRIC_PALETTE,
        ax=ax,
    )
    first = by_emotion.iloc[0]
    ax.set_xlabel("Emotion label")
    ax.set_ylabel("Mean MCD")
    ax.set_title(f"MCD Metrics per Emotion Label\n{first['model_variant']} | alpha={first['alpha']}")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(title="")
    fig.tight_layout()
    fig.savefig(output_dir / "mcd_metrics_per_emotion_label.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_general_metrics_plot(overall: pd.DataFrame, per_sample: pd.DataFrame, output_dir: Path) -> None:
    row = overall.iloc[0]
    speaker_col = "speaker_gen_src_cosine" if "speaker_gen_src_cosine" in per_sample.columns else "speaker_cosine"
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.scatterplot(
        data=per_sample,
        x=speaker_col,
        y="emotion_ref_cosine",
        hue="target_emotion",
        hue_order=EMOTION_ORDER,
        palette=EMOTION_PALETTE,
        alpha=0.7,
        s=60,
        ax=ax,
    )
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Generated-Source Speaker Cosine Similarity" if speaker_col == "speaker_gen_src_cosine" else "Speaker Cosine Similarity")
    ax.set_ylabel("Emotion Reference Cosine Similarity")
    ax.set_title(f"General Metrics: Sample Tradeoff\n{row['model_variant']} | alpha={row['alpha']}")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "general_metrics_scatter.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize balanced benchmark results.")
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    output_dir = Path(args.output_dir) if args.output_dir else experiment_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    emotion_dir = output_dir / "emotion"
    speaker_dir = output_dir / "speaker"
    mcd_dir = output_dir / "mcd"
    normalized_dir = output_dir / "normalized"
    tradeoff_dir = output_dir / "tradeoff"
    for directory in [emotion_dir, speaker_dir, mcd_dir, normalized_dir, tradeoff_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", context="talk")
    frames = load_frames(experiment_dir)

    save_general_metrics_plot(frames["overall"], frames["per_sample"], tradeoff_dir)
    save_tradeoff_by_emotion_plot(frames["per_sample"], frames["by_emotion"], tradeoff_dir)
    save_metric_family_bars(
        frames["overall"],
        EMOTION_ONLY_METRICS,
        emotion_dir,
        "emotion_metrics_means.png",
        "Emotion Metrics: Mean Values",
        "Mean score",
        ylim=(0, 1.05),
    )
    save_metric_family_bars(
        frames["overall"],
        SPEAKER_METRICS,
        speaker_dir,
        "speaker_metrics_means.png",
        "Speaker Metrics: Mean Values",
        "Mean speaker cosine",
        ylim=(0, 1.05),
    )
    save_metric_family_bars(
        frames["overall"],
        MCD_METRICS,
        mcd_dir,
        "mcd_metrics_means.png",
        "MCD Metrics: Mean Values",
        "Mean MCD",
    )
    save_normalized_overview(frames["overall"], normalized_dir)
    save_emotion_metrics_plot(frames["by_emotion"], emotion_dir)
    save_speaker_metrics_by_emotion_plot(frames["by_emotion"], speaker_dir)
    save_mcd_metrics_by_emotion_plot(frames["by_emotion"], mcd_dir)

    print(f"Saved plots to {output_dir}")


if __name__ == "__main__":
    main()
