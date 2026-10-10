#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_EMOTION_ORDER = ["Angry", "Happy", "Sad", "Surprise"]
DEFAULT_SPEAKER_METRIC = "speaker_cosine"
DEFAULT_EMOTION_METRIC = "emotion_ref_cosine"
FIG_DPI = 240


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot per-emotion speaker-emotion tradeoff curves from an alpha summary CSV."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="CSV with alpha, target_emotion, speaker metric, and emotion metric columns.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path. Defaults to <input parent>/tradeoff/per_emotion_tradeoff_overlay.png.",
    )
    parser.add_argument(
        "--speaker-metric",
        default=DEFAULT_SPEAKER_METRIC,
        help=f"X-axis metric. Default: {DEFAULT_SPEAKER_METRIC}.",
    )
    parser.add_argument(
        "--emotion-metric",
        default=DEFAULT_EMOTION_METRIC,
        help=f"Y-axis metric. Default: {DEFAULT_EMOTION_METRIC}.",
    )
    parser.add_argument(
        "--annotate-alpha",
        action="store_true",
        help="Annotate each point with its alpha value.",
    )
    return parser.parse_args()


def title_for_metric(metric: str) -> str:
    return metric.replace("_", " ").title()


def ordered_emotions(frame: pd.DataFrame) -> list[str]:
    present = list(frame["target_emotion"].dropna().unique())
    ordered = [emotion for emotion in DEFAULT_EMOTION_ORDER if emotion in present]
    ordered.extend(sorted(emotion for emotion in present if emotion not in ordered))
    return ordered


def plot_tradeoff_curves(
    frame: pd.DataFrame,
    output_path: Path,
    speaker_metric: str,
    emotion_metric: str,
    annotate_alpha: bool,
) -> None:
    required = {"alpha", "target_emotion", speaker_metric, emotion_metric}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    plot_df = frame.dropna(subset=["alpha", "target_emotion", speaker_metric, emotion_metric]).copy()
    if plot_df.empty:
        raise ValueError("No complete rows available for plotting.")

    emotions = ordered_emotions(plot_df)
    palette = dict(zip(emotions, plt.get_cmap("tab10").colors))

    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    for emotion in emotions:
        emotion_df = plot_df[plot_df["target_emotion"] == emotion].sort_values("alpha")
        color = palette[emotion]
        ax.plot(
            emotion_df[speaker_metric],
            emotion_df[emotion_metric],
            marker="o",
            markersize=7,
            linewidth=2.2,
            color=color,
            label=emotion,
            zorder=3,
        )
        if annotate_alpha:
            for row in emotion_df.itertuples(index=False):
                ax.text(
                    getattr(row, speaker_metric) + 0.002,
                    getattr(row, emotion_metric) + 0.006,
                    f"{row.alpha:.1f}",
                    fontsize=8,
                    color=color,
                )

    ax.set_xlabel(title_for_metric(speaker_metric))
    ax.set_ylabel(title_for_metric(emotion_metric))
    ax.set_title("Speaker-Emotion Tradeoff Curves by Target Emotion")
    ax.grid(True, alpha=0.28)
    ax.legend(title="Target emotion", frameon=True)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output = args.output
    if output is None:
        output = args.input_csv.parent / "tradeoff" / "per_emotion_tradeoff_overlay.png"

    frame = pd.read_csv(args.input_csv)
    plot_tradeoff_curves(
        frame=frame,
        output_path=output,
        speaker_metric=args.speaker_metric,
        emotion_metric=args.emotion_metric,
        annotate_alpha=args.annotate_alpha,
    )
    print(f"Saved plot to: {output}")


if __name__ == "__main__":
    main()
