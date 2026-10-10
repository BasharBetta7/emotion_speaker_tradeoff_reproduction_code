#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


EMOTION_ORDER = ["Angry", "Happy", "Sad", "Surprise"]
METRICS = ["naturalness", "emotion_similarity_score", "speaker_similarity_score"]
BOOTSTRAP_SEED = 20260409
N_BOOTSTRAP = 10000
CI_LEVEL = 0.95


def bootstrap_mean_ci(series: pd.Series) -> tuple[float, float, float, int]:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    n = int(values.size)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    mean = float(values.mean())
    if n == 1:
        return mean, mean, mean, n

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, n, size=(N_BOOTSTRAP, n))
    boot_means = values[draws].mean(axis=1)
    alpha = (1.0 - CI_LEVEL) / 2.0
    ci_low, ci_high = np.quantile(boot_means, [alpha, 1.0 - alpha])
    return mean, float(ci_low), float(ci_high), n


def metric_summary(frame: pd.DataFrame, group_cols: list[str] | None = None) -> pd.DataFrame:
    rows = []
    grouped = [((), frame)] if not group_cols else frame.groupby(group_cols, dropna=False)

    for keys, group in grouped:
        if group_cols:
            if not isinstance(keys, tuple):
                keys = (keys,)
            base = dict(zip(group_cols, keys))
        else:
            base = {}

        for metric in METRICS:
            mean, ci_low, ci_high, n = bootstrap_mean_ci(group[metric])
            rows.append(
                {
                    **base,
                    "metric": metric,
                    "mean": mean,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "std": group[metric].std(ddof=1),
                    "n": n,
                    "ci_level": CI_LEVEL,
                    "n_bootstrap": N_BOOTSTRAP,
                }
            )

    return pd.DataFrame(rows)


def load_rating_sheet_with_ids(rating_sheet_path: Path) -> pd.DataFrame:
    rating_sheet = pd.read_csv(rating_sheet_path).copy()
    rating_sheet.insert(0, "sample_id", range(1, len(rating_sheet) + 1))
    return rating_sheet


def merge_votes(user_votes_path: Path, rating_sheet_path: Path) -> pd.DataFrame:
    votes = pd.read_csv(user_votes_path).copy()
    rating_sheet = load_rating_sheet_with_ids(rating_sheet_path)

    merged = votes.merge(
        rating_sheet[["sample_id", "clip_id", "alpha", "speaker_id", "target_emotion", "audio_path", "reference_text"]],
        on=["sample_id", "clip_id"],
        how="left",
        suffixes=("", "_sheet"),
    )

    merged["naturalness"] = pd.to_numeric(merged["naturalness"], errors="coerce")
    merged["emotion_match_rank"] = pd.to_numeric(merged["emotion_match_rank"], errors="coerce")
    merged["speaker_similarity"] = pd.to_numeric(merged["speaker_similarity"], errors="coerce")

    merged["emotion_similarity_score"] = merged["emotion_match_rank"]
    merged["speaker_similarity_score"] = merged["speaker_similarity"]

    return merged


def overall_summary(merged: pd.DataFrame) -> pd.DataFrame:
    return metric_summary(merged)


def grouped_summary(merged: pd.DataFrame, group_col: str) -> pd.DataFrame:
    return metric_summary(merged, [group_col])


def plot_alpha_metrics(merged: pd.DataFrame, output_dir: Path) -> None:
    alpha_summary = grouped_summary(merged, "alpha").sort_values("alpha")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharex=True)
    specs = [
        ("naturalness", "Naturalness vs Alpha", "#2a9d8f", "Mean naturalness"),
        ("emotion_similarity_score", "Emotion Similarity Score vs Alpha", "#e76f51", "Mean emotion similarity score"),
        ("speaker_similarity_score", "Speaker Similarity Score vs Alpha", "#577590", "Mean speaker similarity score"),
    ]
    for ax, (metric, title, color, ylabel) in zip(axes, specs):
        metric_summary = alpha_summary[alpha_summary["metric"] == metric]
        yerr = [
            metric_summary["mean"] - metric_summary["ci_low"],
            metric_summary["ci_high"] - metric_summary["mean"],
        ]
        ax.errorbar(
            metric_summary["alpha"],
            metric_summary["mean"],
            yerr=yerr,
            marker="o",
            linewidth=2,
            capsize=4,
            color=color,
        )
        ax.set_title(title)
        ax.set_xlabel("Alpha")
        ax.set_ylabel(ylabel)
        # make x-axis ticks 0.2 apart and grid lines at the same positions
        ax.set_xticks(sorted(alpha_summary["alpha"].dropna().unique()))
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "user_votes_metrics_vs_alpha.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_emotion_speaker_tradeoff_by_alpha(merged: pd.DataFrame, output_dir: Path) -> None:
    alpha_summary = grouped_summary(merged, "alpha")
    tradeoff = (
        alpha_summary[
            alpha_summary["metric"].isin(["emotion_similarity_score", "speaker_similarity_score"])
        ]
        .pivot(index="alpha", columns="metric", values=["mean", "ci_low", "ci_high"])
        .reset_index()
    )
    tradeoff.columns = [
        column[0] if column[1] == "" else f"{column[1]}_{column[0]}"
        for column in tradeoff.columns.to_flat_index()
    ]
    tradeoff = tradeoff.sort_values("alpha")
    tradeoff.to_csv(output_dir / "user_votes_emotion_speaker_tradeoff_by_alpha.csv", index=False)

    x = tradeoff["speaker_similarity_score_mean"]
    y = tradeoff["emotion_similarity_score_mean"]
    xerr = [
        x - tradeoff["speaker_similarity_score_ci_low"],
        tradeoff["speaker_similarity_score_ci_high"] - x,
    ]
    yerr = [
        y - tradeoff["emotion_similarity_score_ci_low"],
        tradeoff["emotion_similarity_score_ci_high"] - y,
    ]
    model_color = "#1f77b4"
    uncertainty_color = (31 / 255, 119 / 255, 180 / 255, 0.28)

    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    ax.errorbar(
        x,
        y,
        xerr=xerr,
        yerr=yerr,
        fmt="o-",
        linewidth=2,
        markersize=7,
        capsize=4,
        color=model_color,
        ecolor=uncertainty_color,
        elinewidth=1.6,
        label="StarGANv2-NEVC",
    )
    for _, row in tradeoff.iterrows():
        ax.annotate(
            f"alpha={row['alpha']:.1f}",
            (row["speaker_similarity_score_mean"], row["emotion_similarity_score_mean"]),
            xytext=(7, 6),
            textcoords="offset points",
            fontsize=10,
        )

    ax.set_title("Emotion vs Speaker Similarity by Alpha")
    ax.set_xlabel("Mean speaker similarity score")
    ax.set_ylabel("Mean emotion similarity score")
    ax.legend(title="Model")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "user_votes_emotion_speaker_tradeoff_by_alpha.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_emotion_metrics(merged: pd.DataFrame, output_dir: Path) -> None:
    emotion_summary = grouped_summary(merged, "target_emotion")
    emotion_summary["target_emotion"] = pd.Categorical(
        emotion_summary["target_emotion"], categories=EMOTION_ORDER, ordered=True
    )
    emotion_summary = emotion_summary.sort_values("target_emotion")

    fig, ax = plt.subplots(figsize=(11, 5.2))
    sns.barplot(data=emotion_summary, x="target_emotion", y="mean", hue="metric", ax=ax, errorbar=None)
    add_grouped_bar_ci(ax, emotion_summary, "target_emotion")
    ax.set_title("Listening Metrics vs Target Emotion")
    ax.set_xlabel("Target Emotion")
    ax.set_ylabel("Mean score")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "user_votes_metrics_vs_emotion.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_speaker_metrics(merged: pd.DataFrame, output_dir: Path) -> None:
    speaker_summary = grouped_summary(merged, "speaker_id").sort_values("speaker_id")

    fig, ax = plt.subplots(figsize=(12, 5.2))
    sns.barplot(data=speaker_summary, x="speaker_id", y="mean", hue="metric", ax=ax, errorbar=None)
    add_grouped_bar_ci(ax, speaker_summary, "speaker_id")
    ax.set_title("Listening Metrics vs Speaker")
    ax.set_xlabel("Speaker ID")
    ax.set_ylabel("Mean score")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "user_votes_metrics_vs_speaker.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_overall_metrics(merged: pd.DataFrame, output_dir: Path) -> None:
    summary = overall_summary(merged)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.barplot(
        data=summary,
        x="metric",
        y="mean",
        hue="metric",
        dodge=False,
        ax=ax,
        palette=["#2a9d8f", "#e76f51", "#577590"],
        errorbar=None,
    )
    for patch, (_, row) in zip(ax.patches, summary.iterrows()):
        x = patch.get_x() + patch.get_width() / 2
        yerr = [[row["mean"] - row["ci_low"]], [row["ci_high"] - row["mean"]]]
        ax.errorbar(x, row["mean"], yerr=yerr, fmt="none", color="black", capsize=4, linewidth=1.5)
    if ax.legend_ is not None:
        ax.legend_.remove()
    ax.set_title("Overall Listening Metrics")
    ax.set_xlabel("")
    ax.set_ylabel("Mean score")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "user_votes_overall_metrics.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def add_grouped_bar_ci(ax, summary: pd.DataFrame, group_col: str) -> None:
    ordered_groups = [tick.get_text() for tick in ax.get_xticklabels()]
    ordered_metrics = [text.get_text() for text in ax.legend_.texts] if ax.legend_ is not None else METRICS
    lookup = {
        (str(row[group_col]), row["metric"]): row
        for _, row in summary.iterrows()
    }
    group_patch_counts: dict[int, int] = {}

    for patch in ax.patches:
        x = patch.get_x() + patch.get_width() / 2
        group_index = int(round(x))
        if group_index < 0 or group_index >= len(ordered_groups):
            continue
        metric_index = group_patch_counts.get(group_index, 0)
        group_patch_counts[group_index] = metric_index + 1
        if metric_index >= len(ordered_metrics):
            continue
        row = lookup.get((ordered_groups[group_index], ordered_metrics[metric_index]))
        if row is None:
            continue
        yerr = [[row["mean"] - row["ci_low"]], [row["ci_high"] - row["mean"]]]
        ax.errorbar(x, row["mean"], yerr=yerr, fmt="none", color="black", capsize=3, linewidth=1.2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze listening-study user votes and generate summary plots.")
    parser.add_argument("--user-votes", default="paper/artifacts/listening_study/user_votes.csv")
    parser.add_argument("--rating-sheet", default="paper/artifacts/listening_study/rating_sheet.csv")
    parser.add_argument("--output-dir", default="paper/artifacts/listening_study/analysis")
    args = parser.parse_args()

    user_votes_path = Path(args.user_votes)
    rating_sheet_path = Path(args.rating_sheet)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", context="talk")

    merged = merge_votes(user_votes_path, rating_sheet_path)
    merged.to_csv(output_dir / "user_votes_merged.csv", index=False)
    overall_summary(merged).to_csv(output_dir / "user_votes_overall_summary.csv", index=False)
    grouped_summary(merged, "alpha").to_csv(output_dir / "user_votes_by_alpha_summary.csv", index=False)
    grouped_summary(merged, "target_emotion").to_csv(output_dir / "user_votes_by_emotion_summary.csv", index=False)
    grouped_summary(merged, "speaker_id").to_csv(output_dir / "user_votes_by_speaker_summary.csv", index=False)

    plot_overall_metrics(merged, output_dir)
    plot_alpha_metrics(merged, output_dir)
    plot_emotion_speaker_tradeoff_by_alpha(merged, output_dir)
    plot_emotion_metrics(merged, output_dir)
    plot_speaker_metrics(merged, output_dir)

    print(f"Wrote merged votes to {output_dir / 'user_votes_merged.csv'}")
    print(f"Wrote summaries to {output_dir}")
    print(f"Wrote plots to {output_dir}")


if __name__ == "__main__":
    main()
