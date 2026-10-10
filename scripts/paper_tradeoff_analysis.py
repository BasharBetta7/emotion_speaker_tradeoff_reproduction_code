#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


EMOTION_ORDER = ["Angry", "Happy", "Sad", "Surprise"]
METRICS = [
    "speaker_cosine",
    "speaker_ref_cosine",
    "emotion_acc",
    "emotion_conf",
    "emotion_ref_cosine",
    "emotion_src_cosine",
]
FIG_DPI = 240
SEED = 1337


def load_eval_dirs(root: Path, prefix: str) -> list[Path]:
    return sorted(path for path in root.glob(prefix) if path.is_dir() and (path / "per_sample_metrics.csv").exists())


def bootstrap_mean_ci(series: pd.Series, n_boot: int = 2000, seed: int = SEED) -> tuple[float, float, float]:
    values = series.dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return (math.nan, math.nan, math.nan)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    for idx in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        boot[idx] = sample.mean()
    return float(values.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def add_ci_columns(frame: pd.DataFrame, group_cols: list[str], metrics: Iterable[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for key, group in frame.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = {column: value for column, value in zip(group_cols, key)}
        for metric in metrics:
            if metric not in group.columns:
                continue
            mean, ci_low, ci_high = bootstrap_mean_ci(group[metric])
            row[metric] = mean
            row[f"{metric}_ci_low"] = ci_low
            row[f"{metric}_ci_high"] = ci_high
        row["n"] = len(group)
        rows.append(row)
    return pd.DataFrame(rows)


def compute_pareto_frontier(frame: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    rows = []
    for row in frame.sort_values([x, y], ascending=[False, False]).itertuples(index=False):
        dominated = False
        for candidate in rows:
            if candidate[x] >= getattr(row, x) and candidate[y] >= getattr(row, y):
                dominated = True
                break
        if not dominated:
            rows.append({"alpha": row.alpha, x: getattr(row, x), y: getattr(row, y)})
    return pd.DataFrame(rows).sort_values("alpha")


def load_alpha_frames(alpha_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    per_sample_frames: list[pd.DataFrame] = []
    by_emotion_frames: list[pd.DataFrame] = []
    by_speaker_frames: list[pd.DataFrame] = []
    overall_frames: list[pd.DataFrame] = []
    for eval_dir in load_eval_dirs(alpha_root, "full_eval_epoch_00025_alpha_*"):
        per_sample_frames.append(pd.read_csv(eval_dir / "per_sample_metrics.csv"))
        by_emotion_frames.append(pd.read_csv(eval_dir / "by_emotion_metrics.csv"))
        by_speaker_frames.append(pd.read_csv(eval_dir / "by_speaker_metrics.csv"))
        overall_frames.append(pd.read_csv(eval_dir / "overall_metrics.csv"))
    if not per_sample_frames:
        raise FileNotFoundError(f"No alpha evaluation directories found under {alpha_root}")
    return (
        pd.concat(per_sample_frames, ignore_index=True),
        pd.concat(by_emotion_frames, ignore_index=True),
        pd.concat(by_speaker_frames, ignore_index=True),
        pd.concat(overall_frames, ignore_index=True),
    )


def load_checkpoint_summary(path: Path) -> pd.DataFrame:
    candidate = path / "overall_metrics_comparison.csv" if path.is_dir() else path
    if not candidate.exists():
        raise FileNotFoundError(f"Checkpoint summary not found: {candidate}")
    frame = pd.read_csv(candidate)
    if "epoch" in frame.columns:
        frame["epoch"] = pd.to_numeric(frame["epoch"], errors="coerce")
    return frame.sort_values("epoch")


def load_ser_summary(confusion_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    confusion = pd.read_csv(confusion_csv).set_index("true_emotion")
    total = float(confusion.to_numpy().sum())
    overall_acc = float(np.trace(confusion.to_numpy()) / total)
    per_class = []
    recalls = []
    for emotion in confusion.index:
        support = float(confusion.loc[emotion].sum())
        recall = float(confusion.loc[emotion, emotion] / support) if support else math.nan
        recalls.append(recall)
        per_class.append({"emotion": emotion, "support": int(support), "recall": recall})
    summary = pd.DataFrame(
        [{"overall_accuracy": overall_acc, "uar": float(np.nanmean(recalls)), "n_validation_samples": int(total)}]
    )
    return summary, pd.DataFrame(per_class)


def save_plot_alpha_sweep(alpha_summary: pd.DataFrame, out_dir: Path) -> None:
    plot_specs = [
        ("emotion_acc", "Emotion Accuracy"),
        ("emotion_ref_cosine", "Emotion Reference Cosine"),
        ("speaker_cosine", "Speaker Cosine"),
        ("speaker_ref_cosine", "Speaker Source-Reference Cosine"),
    ]
    plot_specs = [(metric, title) for metric, title in plot_specs if metric in alpha_summary.columns]
    fig, axes = plt.subplots(1, len(plot_specs), figsize=(5 * len(plot_specs), 4.8), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (metric, title) in zip(axes, plot_specs):
        ax.plot(alpha_summary["alpha"], alpha_summary[metric], marker="o", linewidth=2.5)
        ax.fill_between(
            alpha_summary["alpha"],
            alpha_summary[f"{metric}_ci_low"],
            alpha_summary[f"{metric}_ci_high"],
            alpha=0.18,
        )
        ax.set_title(title)
        ax.set_xlabel("Alpha")
        ax.set_ylabel("Mean score")
        ax.set_ylim(-0.1 if "cosine" in metric else 0.0, 1.02)
        ax.grid(alpha=0.25)
    fig.suptitle("Global Intensity Sweep")
    fig.tight_layout()
    fig.savefig(out_dir / "alpha_sweep_metrics.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def save_plot_tradeoff(alpha_summary: pd.DataFrame, frontier: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    scatter = ax.scatter(
        alpha_summary["speaker_cosine"],
        alpha_summary["emotion_ref_cosine"],
        c=alpha_summary["alpha"],
        cmap="viridis",
        s=100,
        zorder=3,
    )
    ax.plot(frontier["speaker_cosine"], frontier["emotion_ref_cosine"], linewidth=2.2, alpha=0.8, zorder=2)
    for row in alpha_summary.itertuples(index=False):
        ax.text(row.speaker_cosine + 0.003, row.emotion_ref_cosine + 0.008, f"a={row.alpha:.1f}", fontsize=9)
    ax.set_xlabel("Speaker Cosine")
    ax.set_ylabel("Emotion Reference Cosine")
    ax.set_title("Emotion-Speaker Tradeoff Frontier")
    ax.grid(alpha=0.25)
    fig.colorbar(scatter, ax=ax, label="Alpha")
    fig.tight_layout()
    fig.savefig(out_dir / "tradeoff_frontier.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def save_plot_per_emotion(by_emotion: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    palette = sns.color_palette("Set2", len(EMOTION_ORDER))
    for emotion, color in zip(EMOTION_ORDER, palette):
        subset = by_emotion[by_emotion["target_emotion"] == emotion].sort_values("alpha")
        if subset.empty:
            continue
        axes[0].plot(subset["alpha"], subset["emotion_ref_cosine"], marker="o", linewidth=2, label=emotion, color=color)
        axes[1].plot(subset["alpha"], subset["speaker_cosine"], marker="o", linewidth=2, label=emotion, color=color)
    axes[0].set_title("Emotion Transfer by Emotion")
    axes[0].set_xlabel("Alpha")
    axes[0].set_ylabel("Emotion Reference Cosine")
    axes[0].grid(alpha=0.25)
    axes[1].set_title("Speaker Preservation by Emotion")
    axes[1].set_xlabel("Alpha")
    axes[1].set_ylabel("Speaker Cosine")
    axes[1].grid(alpha=0.25)
    axes[1].legend(title="Target emotion", loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "per_emotion_tradeoff.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def save_plot_per_speaker(per_sample: pd.DataFrame, out_dir: Path) -> None:
    focus = per_sample[per_sample["alpha"].isin([0.0, 0.4, 0.8, 1.0])].copy()
    focus["alpha_label"] = focus["alpha"].map(lambda value: f"{value:.1f}")
    plot_specs = [
        ("emotion_ref_cosine", "Emotion Transfer Spread by Speaker", "Emotion Reference Cosine"),
        ("speaker_cosine", "Speaker Preservation Spread by Speaker", "Speaker Source-Output Cosine"),
        ("speaker_ref_cosine", "Source-Reference Speaker Spread", "Speaker Source-Reference Cosine"),
    ]
    plot_specs = [(metric, title, ylabel) for metric, title, ylabel in plot_specs if metric in focus.columns]
    fig, axes = plt.subplots(1, len(plot_specs), figsize=(7.5 * len(plot_specs), 5.2), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (metric, title, ylabel) in zip(axes, plot_specs):
        sns.boxplot(data=focus, x="speaker_id", y=metric, hue="alpha_label", ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Speaker ID")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(title="Alpha", loc="lower left")
    fig.tight_layout()
    fig.savefig(out_dir / "per_speaker_spread.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def save_plot_checkpoint_sweep(checkpoints: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for metric, label in [
        ("emotion_acc", "Emotion Accuracy"),
        ("emotion_ref_cosine", "Emotion Reference Cosine"),
        ("speaker_cosine", "Speaker Cosine"),
    ]:
        ax.plot(checkpoints["epoch"], checkpoints[metric], marker="o", linewidth=2, label=label)
    if "epoch" in checkpoints.columns and not checkpoints.empty:
        best_row = checkpoints.sort_values(["emotion_acc", "emotion_ref_cosine"], ascending=[False, False]).iloc[0]
        ax.axvline(best_row["epoch"], linestyle="--", alpha=0.35, color="#555555")
    ax.set_title("Checkpoint Comparison on Proxy Benchmark")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean score")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "checkpoint_comparison.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def write_summary_markdown(
    alpha_summary: pd.DataFrame,
    checkpoint_summary: pd.DataFrame,
    ser_summary: pd.DataFrame,
    ser_per_class: pd.DataFrame,
    by_emotion_summary: pd.DataFrame,
    out_path: Path,
) -> None:
    alpha_0 = alpha_summary.loc[np.isclose(alpha_summary["alpha"], 0.0)].iloc[0]
    alpha_1 = alpha_summary.loc[np.isclose(alpha_summary["alpha"], 1.0)].iloc[0]
    best_ckpt = checkpoint_summary.sort_values(["emotion_acc", "emotion_ref_cosine"], ascending=[False, False]).iloc[0]
    sad_row = by_emotion_summary.query("target_emotion == 'Sad'").sort_values("alpha").iloc[-1]
    angry_row = by_emotion_summary.query("target_emotion == 'Angry'").sort_values("alpha").iloc[-1]

    text = f"""# Tradeoff Analysis Summary

## Core finding
- On the full held-out set, increasing alpha from `0.0` to `1.0` raises `emotion_acc` from `{alpha_0['emotion_acc']:.3f}` to `{alpha_1['emotion_acc']:.3f}`.
- Over the same sweep, `speaker_cosine` drops from `{alpha_0['speaker_cosine']:.3f}` to `{alpha_1['speaker_cosine']:.3f}`.
- `emotion_ref_cosine` rises from `{alpha_0['emotion_ref_cosine']:.3f}` to `{alpha_1['emotion_ref_cosine']:.3f}`.

## Checkpoint anchor
- Best proxy checkpoint by emotion accuracy is `epoch_{int(best_ckpt['epoch']):05d}` with `emotion_acc={best_ckpt['emotion_acc']:.3f}`, `emotion_ref_cosine={best_ckpt['emotion_ref_cosine']:.3f}`, and `speaker_cosine={best_ckpt['speaker_cosine']:.3f}`.

## Emotion asymmetry
- At the highest tested intensity, `Sad` reaches `emotion_acc={sad_row['emotion_acc']:.3f}`.
- At the same intensity, `Angry` reaches `emotion_acc={angry_row['emotion_acc']:.3f}`.
- This supports the paper claim that tradeoff slope and saturation vary by emotion.

## SER reliability
- Validation `overall_accuracy={ser_summary.iloc[0]['overall_accuracy']:.3f}` and `UAR={ser_summary.iloc[0]['uar']:.3f}`.
- Hardest classes from recall:
"""
    for row in ser_per_class.sort_values("recall").itertuples(index=False):
        text += f"- `{row.emotion}` recall = {row.recall:.3f} on {row.support} utterances.\n"
    out_path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build publication-ready tradeoff artifacts from benchmark CSVs.")
    parser.add_argument("--alpha-root", default="experiments/full_set_alpha_variants")
    parser.add_argument("--checkpoint-summary", default="experiments/proxy_epoch_variants/overall_metrics_comparison.csv")
    parser.add_argument("--ser-confusion", default="pretrained_models/wavlm_ser_esd/validation_confusion_matrix.csv")
    parser.add_argument("--output-dir", default="paper/artifacts/tradeoff_analysis")
    args = parser.parse_args()

    sns.set_theme(style="whitegrid", context="talk")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_sample, _, _, _ = load_alpha_frames(Path(args.alpha_root))
    alpha_summary = add_ci_columns(per_sample, ["alpha"], METRICS).sort_values("alpha")
    by_emotion_summary = add_ci_columns(per_sample, ["alpha", "target_emotion"], METRICS).sort_values(["target_emotion", "alpha"])
    by_speaker_summary = add_ci_columns(per_sample, ["alpha", "speaker_id"], METRICS).sort_values(["speaker_id", "alpha"])
    frontier = compute_pareto_frontier(alpha_summary, "speaker_cosine", "emotion_ref_cosine")

    checkpoint_summary = load_checkpoint_summary(Path(args.checkpoint_summary))
    ser_summary, ser_per_class = load_ser_summary(Path(args.ser_confusion))

    alpha_summary.to_csv(output_dir / "alpha_overall_summary.csv", index=False)
    by_emotion_summary.to_csv(output_dir / "alpha_by_emotion_summary.csv", index=False)
    by_speaker_summary.to_csv(output_dir / "alpha_by_speaker_summary.csv", index=False)
    frontier.to_csv(output_dir / "tradeoff_frontier.csv", index=False)
    checkpoint_summary.to_csv(output_dir / "checkpoint_summary.csv", index=False)
    ser_summary.to_csv(output_dir / "ser_reliability_summary.csv", index=False)
    ser_per_class.to_csv(output_dir / "ser_reliability_by_class.csv", index=False)

    save_plot_alpha_sweep(alpha_summary, output_dir)
    save_plot_tradeoff(alpha_summary, frontier, output_dir)
    save_plot_per_emotion(by_emotion_summary, output_dir)
    save_plot_per_speaker(per_sample, output_dir)
    save_plot_checkpoint_sweep(checkpoint_summary, output_dir)
    write_summary_markdown(
        alpha_summary=alpha_summary,
        checkpoint_summary=checkpoint_summary,
        ser_summary=ser_summary,
        ser_per_class=ser_per_class,
        by_emotion_summary=by_emotion_summary,
        out_path=output_dir / "summary.md",
    )
    print(f"Saved publication artifacts to {output_dir}")


if __name__ == "__main__":
    main()
