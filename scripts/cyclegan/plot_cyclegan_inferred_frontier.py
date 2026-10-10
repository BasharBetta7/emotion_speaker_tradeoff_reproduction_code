#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import pandas as pd


REQUIRED_COLUMNS = ["alpha", "speaker_cosine", "emotion_ref_cosine"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a CycleGAN-only inferred-intensity frontier.")
    parser.add_argument(
        "--overall-metrics",
        default="experiments/baselines/cyclegan_spectrum_inferred_eval_full/overall_metrics.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="paper/artifacts/cyclegan_spectrum_inferred_frontier",
    )
    parser.add_argument("--title", default="CycleGAN Spectrum Inferred-Intensity Frontier")
    args = parser.parse_args()

    metrics_path = Path(args.overall_metrics)
    frame = pd.read_csv(metrics_path)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise KeyError(f"{metrics_path} is missing required columns: {missing}")

    frame = frame.copy()
    frame["alpha"] = pd.to_numeric(frame["alpha"], errors="coerce")
    frame = frame.dropna(subset=["alpha", "speaker_cosine", "emotion_ref_cosine"])
    if frame.empty:
        raise ValueError(f"No plottable alpha rows found in {metrics_path}")

    frontier = (
        frame.groupby("alpha", as_index=False)[["speaker_cosine", "emotion_ref_cosine"]]
        .mean()
        .sort_values("alpha")
        .reset_index(drop=True)
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frontier.to_csv(out_dir / "frontier_points.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    scatter = ax.scatter(
        frontier["speaker_cosine"],
        frontier["emotion_ref_cosine"],
        c=frontier["alpha"],
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        s=95,
        edgecolors="none",
        zorder=3,
    )
    ax.plot(
        frontier["speaker_cosine"],
        frontier["emotion_ref_cosine"],
        linewidth=2.2,
        alpha=0.85,
        zorder=2,
    )

    for row in frontier.itertuples(index=False):
        ax.text(
            row.speaker_cosine + 0.002,
            row.emotion_ref_cosine + 0.006,
            f"{row.alpha:.1f}",
            fontsize=9,
        )

    ax.set_xlabel("Speaker Cosine Similarity")
    ax.set_ylabel("Emotion Reference Cosine Similarity")
    ax.set_title(args.title)
    ax.grid(alpha=0.25)
    fig.colorbar(scatter, ax=ax, label="Inferred intensity alpha")
    fig.tight_layout()
    fig.savefig(out_dir / "tradeoff_frontier.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote frontier points: {out_dir / 'frontier_points.csv'}")
    print(f"Wrote frontier plot: {out_dir / 'tradeoff_frontier.png'}")


if __name__ == "__main__":
    main()
