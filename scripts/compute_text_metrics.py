#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


WHITESPACE_RE = re.compile(r"\s+")
NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")
EMOTION_ORDER = ["Angry", "Happy", "Sad", "Surprise"]


def normalize_text(text: str) -> str:
    value = str(text or "").lower()
    value = NON_ALNUM_RE.sub(" ", value)
    value = WHITESPACE_RE.sub(" ", value).strip()
    return value


def levenshtein(seq_a: list[str], seq_b: list[str]) -> int:
    if len(seq_a) < len(seq_b):
        seq_a, seq_b = seq_b, seq_a

    previous = list(range(len(seq_b) + 1))
    for i, token_a in enumerate(seq_a, start=1):
        current = [i]
        for j, token_b in enumerate(seq_b, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (0 if token_a == token_b else 1)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def score_pair(reference: str, hypothesis: str) -> dict[str, float | int | str]:
    ref_norm = normalize_text(reference)
    hyp_norm = normalize_text(hypothesis)

    ref_words = ref_norm.split() if ref_norm else []
    hyp_words = hyp_norm.split() if hyp_norm else []
    ref_chars = list(ref_norm.replace(" ", ""))
    hyp_chars = list(hyp_norm.replace(" ", ""))

    word_errors = levenshtein(ref_words, hyp_words)
    char_errors = levenshtein(ref_chars, hyp_chars)
    word_denom = max(len(ref_words), 1)
    char_denom = max(len(ref_chars), 1)

    return {
        "reference_text_normalized": ref_norm,
        "hypothesis_text_normalized": hyp_norm,
        "word_errors": word_errors,
        "char_errors": char_errors,
        "word_count": len(ref_words),
        "char_count": len(ref_chars),
        "wer": word_errors / word_denom,
        "cer": char_errors / char_denom,
        "exact_match": float(ref_norm == hyp_norm),
    }


def build_summary(scored: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if group_cols:
        groups = scored.groupby(group_cols, dropna=False)
    else:
        groups = [(("all",), scored)]

    rows: list[dict] = []
    for key, frame in groups:
        if group_cols:
            if not isinstance(key, tuple):
                key = (key,)
            row = {column: value for column, value in zip(group_cols, key)}
        else:
            row = {"scope": "all"}

        row.update(
            {
                "n": len(frame),
                "wer": frame["word_errors"].sum() / max(frame["word_count"].sum(), 1),
                "cer": frame["char_errors"].sum() / max(frame["char_count"].sum(), 1),
                "exact_match": frame["exact_match"].mean(),
                "mean_wer": frame["wer"].mean(),
                "mean_cer": frame["cer"].mean(),
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


def save_alpha_plots(scored: pd.DataFrame, output_dir: Path) -> None:
    if "alpha" not in scored.columns:
        return

    alpha_summary = (
        scored.groupby("alpha", dropna=False)[["wer", "cer"]]
        .mean()
        .reset_index()
        .sort_values("alpha")
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    for ax, metric, color in zip(axes, ["wer", "cer"], ["#d1495b", "#00798c"]):
        ax.plot(alpha_summary["alpha"], alpha_summary[metric], marker="o", linewidth=2, color=color)
        ax.set_title(f"{metric.upper()} vs Alpha")
        ax.set_xlabel("Alpha")
        ax.set_ylabel(metric.upper())
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "wer_cer_vs_alpha.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_emotion_plots(scored: pd.DataFrame, output_dir: Path) -> None:
    if "target_emotion" not in scored.columns:
        return

    emotion_summary = (
        scored.groupby("target_emotion", dropna=False)[["wer", "cer"]]
        .mean()
        .reset_index()
    )
    emotion_summary["target_emotion"] = pd.Categorical(
        emotion_summary["target_emotion"], categories=EMOTION_ORDER, ordered=True
    )
    emotion_summary = emotion_summary.sort_values("target_emotion")
    melted = emotion_summary.melt(id_vars="target_emotion", value_vars=["wer", "cer"], var_name="metric", value_name="score")

    fig, ax = plt.subplots(figsize=(10, 5.2))
    sns.barplot(data=melted, x="target_emotion", y="score", hue="metric", ax=ax)
    ax.set_title("WER/CER vs Target Emotion")
    ax.set_xlabel("Target Emotion")
    ax.set_ylabel("Mean score")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "wer_cer_vs_emotion.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_speaker_plots(scored: pd.DataFrame, output_dir: Path) -> None:
    if "speaker_id" not in scored.columns:
        return

    speaker_summary = (
        scored.groupby("speaker_id", dropna=False)[["wer", "cer"]]
        .mean()
        .reset_index()
        .sort_values("speaker_id")
    )
    melted = speaker_summary.melt(id_vars="speaker_id", value_vars=["wer", "cer"], var_name="metric", value_name="score")

    fig, ax = plt.subplots(figsize=(12, 5.2))
    sns.barplot(data=melted, x="speaker_id", y="score", hue="metric", ax=ax)
    ax.set_title("WER/CER vs Speaker")
    ax.set_xlabel("Speaker ID")
    ax.set_ylabel("Mean score")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "wer_cer_vs_speaker.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_plots(scored: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")
    save_alpha_plots(scored, output_dir)
    save_emotion_plots(scored, output_dir)
    save_speaker_plots(scored, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute CER/WER from reference and transcript text columns.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--reference-column", default="text")
    parser.add_argument("--hypothesis-column", default="predicted_text")
    parser.add_argument("--group-by", nargs="*", default=["alpha", "target_emotion"])
    parser.add_argument("--output-scored-csv", required=True)
    parser.add_argument("--output-summary-csv", required=True)
    parser.add_argument("--plots-dir", default=None, help="Optional directory for WER/CER plots.")
    args = parser.parse_args()

    frame = pd.read_csv(args.input_csv)
    required = [args.reference_column, args.hypothesis_column]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    metrics = frame.apply(
        lambda row: score_pair(row[args.reference_column], row[args.hypothesis_column]),
        axis=1,
        result_type="expand",
    )
    scored = pd.concat([frame, metrics], axis=1)
    summary = build_summary(scored, [column for column in args.group_by if column in scored.columns])

    scored_path = Path(args.output_scored_csv)
    summary_path = Path(args.output_summary_csv)
    scored_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(scored_path, index=False)
    summary.to_csv(summary_path, index=False)

    plots_dir = Path(args.plots_dir) if args.plots_dir else summary_path.parent / "plots"
    save_plots(scored, plots_dir)

    print(f"Wrote scored rows to {scored_path}")
    print(f"Wrote summary rows to {summary_path}")
    print(f"Wrote plots to {plots_dir}")


if __name__ == "__main__":
    main()
