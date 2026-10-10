from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import pandas as pd

METRIC_GROUPS = {
    "emotion": [
        "emotion_acc",
        "emotion_ref_cosine",
        "emotion_src_cosine",
        "emotion_conf",
    ],
    "speaker": [
        "speaker_gen_src_cosine",
        "speaker_gen_ref_cosine",
        "speaker_src_ref_cosine",
        "speaker_cosine",
        "speaker_ref_cosine",
        "speaker_ref_output_cosine",
    ],
    "mcd": [
        "mcd_gen_src",
        "mcd_gen_ref",
        "mcd_src_ref",
    ],
}

METRIC_PLOT_SPECS = {
    "emotion_acc": ("Emotion Accuracy", "Emotion Accuracy Across Alpha Values", "alpha_emotion_acc_by_emotion.png"),
    "emotion_ref_cosine": ("Emotion Reference Cosine", "Emotion Reference Cosine Across Alpha Values", "alpha_emotion_ref_cosine_by_emotion.png"),
    "emotion_src_cosine": ("Emotion Source Cosine", "Emotion Source Cosine Across Alpha Values", "alpha_emotion_src_cosine_by_emotion.png"),
    "emotion_conf": ("Emotion Confidence", "Emotion Confidence Across Alpha Values", "alpha_emotion_conf_by_emotion.png"),
    "speaker_cosine": ("Speaker Cosine", "Speaker Cosine Across Alpha Values", "alpha_speaker_cosine_by_emotion.png"),
    "speaker_ref_cosine": ("Speaker Source-Reference Cosine", "Speaker Source-Reference Cosine Across Alpha Values", "alpha_speaker_ref_cosine_by_emotion.png"),
    "speaker_ref_output_cosine": ("Speaker Reference-Output Cosine", "Speaker Reference-Output Cosine Across Alpha Values", "alpha_speaker_ref_output_cosine_by_emotion.png"),
    "speaker_gen_ref_cosine": ("Speaker Generated-Reference Cosine", "Speaker Generated-Reference Cosine Across Alpha Values", "alpha_speaker_gen_ref_cosine_by_emotion.png"),
    "speaker_gen_src_cosine": ("Speaker Generated-Source Cosine", "Speaker Generated-Source Cosine Across Alpha Values", "alpha_speaker_gen_src_cosine_by_emotion.png"),
    "speaker_src_ref_cosine": ("Speaker Source-Reference Cosine", "Speaker Source-Reference Cosine Across Alpha Values", "alpha_speaker_src_ref_cosine_by_emotion.png"),
    "mcd_gen_ref": ("MCD Generated-Reference", "MCD Generated-Reference Across Alpha Values", "alpha_mcd_gen_ref_by_emotion.png"),
    "mcd_gen_src": ("MCD Generated-Source", "MCD Generated-Source Across Alpha Values", "alpha_mcd_gen_src_by_emotion.png"),
    "mcd_src_ref": ("MCD Source-Reference", "MCD Source-Reference Across Alpha Values", "alpha_mcd_src_ref_by_emotion.png"),
}


def _present_group_metrics(metrics: list[str], group_name: str) -> list[str]:
    return [metric for metric in METRIC_GROUPS[group_name] if metric in metrics]


def _minmax_normalize(frame: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
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


def _save_family_overview(
    summary_df: pd.DataFrame,
    metrics: list[str],
    family_name: str,
    out_path: Path,
    normalized: bool = False,
) -> None:
    if not metrics:
        return

    plot_df = _minmax_normalize(summary_df, metrics) if normalized else summary_df
    fig, ax = plt.subplots(figsize=(10, 6))
    for metric in metrics:
        ax.plot(plot_df["alpha"], plot_df[metric], marker="o", label=metric)
    ax.set_xlabel("Alpha")
    ax.set_ylabel("Normalized metric value" if normalized else "Metric value")
    suffix = " (min-max normalized)" if normalized else ""
    ax.set_title(f"{family_name.capitalize()} Metrics Across Alpha Values{suffix}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    filename = f"alpha_{family_name}_metrics_comparison{'_normalized' if normalized else ''}.png"
    plt.savefig(out_path / filename)
    print(f"Saved {family_name} metrics plot to: {out_path / filename}")
    plt.close(fig)


def _save_family_by_emotion_plot(
    summary_by_emotion: pd.DataFrame,
    emotions,
    metrics: list[str],
    family_name: str,
    out_path: Path,
    normalized: bool = False,
) -> None:
    if not metrics:
        return

    plot_df = _minmax_normalize(summary_by_emotion, metrics) if normalized else summary_by_emotion
    fig, axes = plt.subplots(len(emotions), 1, figsize=(10, 4 * max(len(emotions), 1)), squeeze=False)
    axes = axes.ravel()

    for ax, emotion in zip(axes, emotions):
        emotion_df = plot_df[plot_df["target_emotion"] == emotion]
        for metric in metrics:
            ax.plot(emotion_df["alpha"], emotion_df[metric], marker="o", label=metric)
        ax.set_title(f"{emotion}: {family_name} metrics across alpha values")
        ax.set_xlabel("Alpha")
        ax.set_ylabel("Normalized metric value" if normalized else "Metric value")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.tight_layout()
    filename = f"alpha_{family_name}_metrics_by_emotion{'_normalized' if normalized else ''}.png"
    plt.savefig(out_path / filename)
    print(f"Saved per-emotion {family_name} metrics plot to: {out_path / filename}")
    plt.close(fig)


def _safe_name(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in str(value)).strip("_")


def _resolve_tradeoff_speaker_metric(frame: pd.DataFrame) -> tuple[str, str] | None:
    if "speaker_gen_src_cosine" in frame.columns:
        return "speaker_gen_src_cosine", "Generated-Source Speaker Cosine"
    if "speaker_cosine" in frame.columns:
        return "speaker_cosine", "Speaker Cosine"
    return None


def _save_tradeoff_frontiers_by_emotion(
    summary_by_emotion: pd.DataFrame,
    emotions,
    out_path: Path,
) -> None:
    resolved = _resolve_tradeoff_speaker_metric(summary_by_emotion)
    if resolved is None or "emotion_ref_cosine" not in summary_by_emotion.columns:
        return

    speaker_metric, speaker_label = resolved
    emotion_list = list(emotions)
    if not emotion_list:
        return

    tradeoff_dir = out_path / "tradeoff"
    per_emotion_dir = tradeoff_dir / "by_emotion"
    per_emotion_dir.mkdir(parents=True, exist_ok=True)

    for emotion in emotion_list:
        emotion_df = (
            summary_by_emotion[summary_by_emotion["target_emotion"] == emotion]
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
        ax.plot(
            emotion_df[speaker_metric],
            emotion_df["emotion_ref_cosine"],
            linewidth=2,
            alpha=0.8,
            zorder=2,
        )
        for row in emotion_df.itertuples(index=False):
            ax.text(
                getattr(row, speaker_metric) + 0.002,
                row.emotion_ref_cosine + 0.006,
                f"a={row.alpha:.1f}",
                fontsize=9,
            )
        ax.set_xlabel(speaker_label)
        ax.set_ylabel("Emotion Reference Cosine")
        ax.set_title(f"{emotion}: Speaker-Emotion Tradeoff Frontier")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        filename = f"tradeoff_frontier_{_safe_name(emotion)}.png"
        fig.savefig(per_emotion_dir / filename, dpi=240, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(len(emotion_list), 1, figsize=(8, 5 * max(len(emotion_list), 1)), squeeze=False)
    axes = axes.ravel()
    for ax, emotion in zip(axes, emotion_list):
        emotion_df = (
            summary_by_emotion[summary_by_emotion["target_emotion"] == emotion]
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
            ax.text(getattr(row, speaker_metric) + 0.002, row.emotion_ref_cosine + 0.006, f"{row.alpha:.1f}", fontsize=8)
        ax.set_xlabel(speaker_label)
        ax.set_ylabel("Emotion Reference Cosine")
        ax.set_title(str(emotion))
        ax.grid(True, alpha=0.3)

    fig.suptitle("Speaker-Emotion Tradeoff Frontiers by Emotion", y=1.002)
    fig.subplots_adjust(top=0.94, hspace=0.45)
    fig.savefig(tradeoff_dir / "tradeoff_frontiers_by_emotion.png", dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved per-emotion tradeoff frontiers to: {per_emotion_dir}")


def _save_mean_tradeoff_frontier(summary_df: pd.DataFrame, out_path: Path) -> None:
    resolved = _resolve_tradeoff_speaker_metric(summary_df)
    if resolved is None or "emotion_ref_cosine" not in summary_df.columns:
        return

    speaker_metric, speaker_label = resolved
    frontier = summary_df.dropna(subset=["alpha", speaker_metric, "emotion_ref_cosine"]).sort_values("alpha")
    if frontier.empty:
        return

    tradeoff_dir = out_path / "tradeoff"
    tradeoff_dir.mkdir(parents=True, exist_ok=True)
    frontier[["alpha", speaker_metric, "emotion_ref_cosine"]].to_csv(
        tradeoff_dir / "tradeoff_frontier_points.csv",
        index=False,
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        frontier[speaker_metric],
        frontier["emotion_ref_cosine"],
        c=frontier["alpha"],
        cmap="viridis",
        s=110,
        zorder=3,
    )
    ax.plot(frontier[speaker_metric], frontier["emotion_ref_cosine"], linewidth=2, alpha=0.8, zorder=2)
    for row in frontier.itertuples(index=False):
        ax.text(getattr(row, speaker_metric) + 0.002, row.emotion_ref_cosine + 0.006, f"a={row.alpha:.1f}", fontsize=9)
    ax.set_xlabel(speaker_label)
    ax.set_ylabel("Emotion Reference Cosine")
    ax.set_title("Mean Speaker-Emotion Tradeoff Frontier")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(tradeoff_dir / "tradeoff_frontier.png", dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved mean tradeoff frontier to: {tradeoff_dir / 'tradeoff_frontier.png'}")


def _warn_incomplete_emotion_grid(all_metrics_df: pd.DataFrame) -> pd.DataFrame:
    sample_counts = (
        all_metrics_df.groupby(["alpha", "target_emotion"])
        .size()
        .rename("sample_count")
        .reset_index()
    )
    expected_count = sample_counts["sample_count"].max()
    count_grid = (
        sample_counts.pivot(index="alpha", columns="target_emotion", values="sample_count")
        .sort_index()
        .sort_index(axis=1)
    )

    incomplete = count_grid.isna() | count_grid.lt(expected_count)
    if incomplete.any().any():
        print("\nWARNING: incomplete alpha/emotion coverage detected.")
        print(f"Expected up to {expected_count} samples per alpha/emotion from the loaded files.")
        print(count_grid.fillna(0).astype(int).to_string())

    return sample_counts


def compare_variants(exp_dir:str, out_dir:str = None, epoch:str | None = None):
    exp_path = Path(exp_dir)
    if out_dir is None:
        out_path = exp_path
    else:
        out_path = Path(out_dir)
    if not out_path.exists():
        out_path.mkdir(parents=True, exist_ok=True)
    if epoch is not None:
        matching_dirs = list(exp_path.glob(f"full_eval_{epoch:05d}_alpha_*"))
    else:
        matching_dirs = list(exp_path.glob("full_eval_*"))
    # check inside exp_dir for any folders that contains the name full_eval_{epoch_num}
    if not matching_dirs:
        raise Exception("Not evaluation folder found that matches the specified pattern")
    all_metrics = []
    for eval_dir in matching_dirs:
        if eval_dir.is_dir():
            print(f"Found evaluation directory: {eval_dir}")
            metrics_file = eval_dir / "per_sample_metrics.csv"
            if metrics_file.exists():
               
                df = pd.read_csv(metrics_file)
               
                #concat df with the epoch number extracted from the eval_dir name
                epoch_num = eval_dir.name.split("_")[3]
                df["epoch"] = epoch_num
                all_metrics.append(df)
            else:
                print(f"WARNING: missing metrics file: {metrics_file}")

    if not all_metrics:
        raise Exception("No per_sample_metrics.csv files found in matching evaluation folders")

    all_metrics_df = pd.concat(all_metrics, ignore_index=True)
    sample_counts = _warn_incomplete_emotion_grid(all_metrics_df)

    all_known_metrics = [metric for group in METRIC_GROUPS.values() for metric in group]
    metrics = [metric for metric in all_known_metrics if metric in all_metrics_df.columns]

    summary_df = (
        all_metrics_df.groupby("alpha")[metrics]
        .mean()
        .reset_index()
    )
    summary_df["alpha"] = pd.to_numeric(summary_df["alpha"], errors="coerce")
    summary_df = summary_df.sort_values("alpha")

    summary_by_emotion = (
        all_metrics_df.groupby(["alpha", "target_emotion"])[metrics]
        .mean()
        .reset_index()
    )
    summary_by_emotion["alpha"] = pd.to_numeric(summary_by_emotion["alpha"], errors="coerce")
    sample_counts["alpha"] = pd.to_numeric(sample_counts["alpha"], errors="coerce")
    summary_by_emotion = summary_by_emotion.merge(
        sample_counts,
        on=["alpha", "target_emotion"],
        how="left",
    )

    all_alphas = summary_by_emotion["alpha"].dropna().sort_values().unique()
    all_emotions = summary_by_emotion["target_emotion"].dropna().sort_values().unique()
    complete_index = pd.MultiIndex.from_product(
        [all_alphas, all_emotions],
        names=["alpha", "target_emotion"],
    )
    summary_by_emotion = (
        summary_by_emotion.set_index(["alpha", "target_emotion"])
        .reindex(complete_index)
        .reset_index()
        .sort_values(["target_emotion", "alpha"])
    )
    summary_by_emotion.to_csv(out_path / "alpha_metrics_by_emotion_summary.csv", index=False)
    summary_df.to_csv(out_path / "alpha_metrics_summary.csv", index=False)
    print(f"Saved per-emotion summary to: {out_path / 'alpha_metrics_by_emotion_summary.csv'}")
    print(f"Saved alpha summary to: {out_path / 'alpha_metrics_summary.csv'}")

    emotions = summary_by_emotion["target_emotion"].dropna().unique()
    family_dirs = {}
    for family_name in METRIC_GROUPS:
        family_dir = out_path / family_name
        family_dir.mkdir(parents=True, exist_ok=True)
        family_dirs[family_name] = family_dir
        family_metrics = _present_group_metrics(metrics, family_name)
        _save_family_overview(summary_df, family_metrics, family_name, family_dir)
        _save_family_by_emotion_plot(summary_by_emotion, emotions, family_metrics, family_name, family_dir)

    normalized_dir = out_path / "normalized"
    normalized_dir.mkdir(parents=True, exist_ok=True)
    _save_family_overview(summary_df, metrics, "all", normalized_dir, normalized=True)
    _save_family_by_emotion_plot(summary_by_emotion, emotions, metrics, "all", normalized_dir, normalized=True)
    _save_mean_tradeoff_frontier(summary_df, out_path)
    _save_tradeoff_frontiers_by_emotion(summary_by_emotion, emotions, out_path)

    for metric, (ylabel, title, filename) in METRIC_PLOT_SPECS.items():
        if metric not in metrics:
            continue
        family_name = next(
            group_name for group_name, group_metrics in METRIC_GROUPS.items() if metric in group_metrics
        )
        metric_out_path = family_dirs[family_name]
        fig, ax = plt.subplots(figsize=(10, 6))
        for emotion in emotions:
            emotion_df = summary_by_emotion[summary_by_emotion["target_emotion"] == emotion]
            ax.plot(emotion_df["alpha"], emotion_df[metric], marker="o", label=emotion)
        ax.set_xlabel("Alpha")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(title="Emotion")
        fig.tight_layout()
        plt.savefig(metric_out_path / filename)
        print(f"Saved {metric} plot to: {metric_out_path / filename}")
        plt.close(fig)

    if {"speaker_gen_src_cosine", "speaker_gen_ref_cosine", "speaker_src_ref_cosine"}.issubset(summary_by_emotion.columns):
        fig, ax = plt.subplots(figsize=(10, 6))
        for emotion in emotions:
            emotion_df = summary_by_emotion[summary_by_emotion["target_emotion"] == emotion].sort_values("alpha")
            line = ax.plot(
                emotion_df["alpha"],
                emotion_df["speaker_gen_src_cosine"],
                marker="o",
                label=f"{emotion} generated-source",
            )[0]
            ax.plot(
                emotion_df["alpha"],
                emotion_df["speaker_gen_ref_cosine"],
                marker="o",
                linestyle="--",
                color=line.get_color(),
                label=f"{emotion} generated-reference",
            )
            ax.plot(
                emotion_df["alpha"],
                emotion_df["speaker_src_ref_cosine"],
                marker=".",
                linestyle=":",
                color=line.get_color(),
                label=f"{emotion} source-reference",
            )
        ax.set_xlabel("Alpha")
        ax.set_ylabel("Speaker cosine similarity")
        ax.set_title("Speaker Cosine Similarity Drift by Emotion")
        ax.grid(True, alpha=0.3)
        ax.legend(title="Emotion / pair", ncol=2)
        fig.tight_layout()
        plt.savefig(family_dirs["speaker"] / "alpha_speaker_cosine_drift.png")
        print(f"Saved speaker cosine drift plot to: {family_dirs['speaker'] / 'alpha_speaker_cosine_drift.png'}")
        plt.close(fig)

    all_metrics_df.to_csv(out_path / "intensity_per_sample_metrics.csv", index=False)
    print(f"saved combined metrics to: {out_path / 'intensity_per_sample_metrics.csv'}")
    normalized_summary = _minmax_normalize(summary_df, metrics)
    fig, ax = plt.subplots(figsize=(10, 6))
    for metric in metrics:
        ax.plot(normalized_summary["alpha"], normalized_summary[metric], marker="o", label=metric)
    ax.set_xlabel("Alpha")
    ax.set_ylabel("Normalized metric value")
    ax.set_title("All Metrics Across Alpha Values (min-max normalized)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plt.savefig(out_path / "alpha_metrics_comparison_normalized.png")
    print(f"Saved normalized metrics comparison plot to: {out_path / 'alpha_metrics_comparison_normalized.png'}")
    plt.close(fig)

    return all_metrics_df


def main():
    parser = argparse.ArgumentParser(description="Compare evaluation metrics across different epochs.")
    parser.add_argument("--exp-dir", type=str, help="Path to the experiments directory containing evaluation results.", default='experiments/full_set_alpha_variants')
    parser.add_argument("--out-dir", type=str, help="Epoch number to filter the evaluation directories. If not provided, all epochs will be included.", default="experiments/full_set_alpha_variants")
    args = parser.parse_args()
    compare_variants(args.exp_dir, args.out_dir)

if __name__ == "__main__":
    main()
