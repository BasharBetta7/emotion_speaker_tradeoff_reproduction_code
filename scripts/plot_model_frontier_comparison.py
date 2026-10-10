#!/usr/bin/env python3
"""Plot model-level emotion/speaker tradeoff frontiers."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
PLOT_STYLE = {
    "font.size": 13,
    "axes.titlesize": 17,
    "axes.labelsize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
}

DATASETS = {
    "esd_ssst": {
        "title": "ESD SSST",
        "output": "esd_ssst_frontier_comparison",
        "models": {
            "CycleGAN-IEVC": "experiments/CycleGAN/ESD/SSST/cyclegan_spectrum_inferred_frontier/frontier_points.csv",
            "StarGANv2-IEVC": "experiments/MODEL_inferred_intensity/ESD/SSST/full_set_alpha_variants/tradeoff/tradeoff_frontier_points.csv",
            "StarGANv2-NEVC": "experiments/MODEL_native_intensity_ranker/ESD/SSST/full_set_alpha_variants/tradeoff/tradeoff_frontier_points.csv",
            "StarGANv2-NEVC + Speaker Contrastive": "experiments/MODEL_native_intensity_ranker/ablation/speaker_contrastive_loss/ESD/SSST/full_set_alpha_variants/tradeoff/tradeoff_frontier_points.csv",
            "StarGANv2-NEVC + Speaker Classifier": "experiments/MODEL_native_intensity_speaker_cls/ESD/SSST/full_set_alpha_variants/tradeoff/tradeoff_frontier_points.csv",
        },
    },
    "esd_dsst": {
        "title": "ESD DSST",
        "output": "esd_dsst_frontier_comparison",
        "models": {
            "CycleGAN-IEVC": "experiments/CycleGAN/ESD/DSST/cyclegan_spectrum_inferred_frontier/frontier_points.csv",
            "StarGANv2-IEVC": "experiments/MODEL_inferred_intensity/ESD/DSST/tradeoff/tradeoff_frontier_points.csv",
            "StarGANv2-NEVC": "experiments/MODEL_native_intensity_ranker/ESD/DSST/full_set_alpha_variants/tradeoff/tradeoff_frontier_points.csv",
            "StarGANv2-NEVC + Speaker Contrastive": "experiments/MODEL_native_intensity_ranker/ablation/speaker_contrastive_loss/ESD/DSST/full_set_alpha_variants/tradeoff/tradeoff_frontier_points.csv",
            "StarGANv2-NEVC + Speaker Classifier": "experiments/MODEL_native_intensity_speaker_cls/ESD/DSST/full_set_alpha_variants/tradeoff/tradeoff_frontier_points.csv",
        },
    },
    "crema_d": {
        "title": "CREMA-D",
        "output": "crema_d_frontier_comparison",
        "models": {
            "CycleGAN-IEVC": "experiments/CycleGAN/CREMA_D/from_scratch/cyclegan_spectrum_inferred_frontier/frontier_points.csv",
            "StarGANv2-IEVC": "experiments/MODEL_inferred_intensity/CREMA_D/from_scratch/full_set_alpha_variants/tradeoff/tradeoff_frontier_points.csv",
            "StarGANv2-NEVC": "experiments/MODEL_native_intensity_ranker/CREMA_D/from_scratch/full_set_alpha_variants/tradeoff/tradeoff_frontier_points.csv",
            "StarGANv2-NEVC + Speaker Classifier": "experiments/MODEL_native_intensity_speaker_cls/CREMA_D/from_scratch/full_set_alpha_variants/tradeoff/tradeoff_frontier_points.csv",
        },
    },
}

COLORS = {
    "CycleGAN-IEVC": "#4c78a8",
    "StarGANv2-IEVC": "#f58518",
    "StarGANv2-NEVC": "#54a24b",
    "StarGANv2-NEVC + Speaker Contrastive": "#e45756",
    "StarGANv2-NEVC + Speaker Classifier": "#b279a2",
}

DATASET_DISPLAY = {
    "esd_ssst": "ESD SSST",
    "esd_dsst": "ESD DSST",
    "crema_d": "CREMA-D",
}

DATASET_COLORS = {
    "esd_ssst": "#4c78a8",
    "esd_dsst": "#f58518",
    "crema_d": "#54a24b",
}

CI_Z = 1.96


def speaker_column(df: pd.DataFrame) -> str:
    for column in ("speaker_gen_src_cosine", "speaker_cosine"):
        if column in df.columns:
            return column
    raise ValueError("No speaker cosine column found; expected speaker_gen_src_cosine or speaker_cosine.")


def read_frontier(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"alpha", "emotion_ref_cosine"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    spk_col = speaker_column(df)
    out = df[["alpha", spk_col, "emotion_ref_cosine"]].copy()
    out = out.rename(columns={spk_col: "speaker_cosine"})
    return out.sort_values("alpha")


def read_metric_samples(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"alpha", "emotion_ref_cosine"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    spk_col = speaker_column(df)
    out = df[["alpha", spk_col, "emotion_ref_cosine"]].copy()
    out = out.rename(columns={spk_col: "speaker_cosine"})
    return out.dropna(subset=["alpha", "speaker_cosine", "emotion_ref_cosine"])


def infer_sample_metrics_path(frontier_path: Path) -> Path | None:
    if frontier_path.name != "frontier_points.csv" and frontier_path.name != "tradeoff_frontier_points.csv":
        return None

    parts = frontier_path.parts
    if "cyclegan_spectrum_inferred_frontier" in parts:
        idx = parts.index("cyclegan_spectrum_inferred_frontier")
        eval_dir = Path(*parts[:idx], "cyclegan_spectrum_inferred_eval_full", "per_sample_metrics.csv")
        return eval_dir if eval_dir.exists() else None

    if frontier_path.parent.name == "tradeoff":
        metrics_path = frontier_path.parent.parent / "intensity_per_sample_metrics.csv"
        return metrics_path if metrics_path.exists() else None

    return None


def summarize_ci(samples: pd.DataFrame) -> pd.DataFrame:
    grouped = samples.groupby("alpha", as_index=False).agg(
        speaker_cosine_sample_mean=("speaker_cosine", "mean"),
        speaker_cosine_sample_std=("speaker_cosine", "std"),
        emotion_ref_cosine_sample_mean=("emotion_ref_cosine", "mean"),
        emotion_ref_cosine_sample_std=("emotion_ref_cosine", "std"),
        sample_count=("emotion_ref_cosine", "count"),
    )
    grouped = grouped.fillna({
        "speaker_cosine_sample_std": 0.0,
        "emotion_ref_cosine_sample_std": 0.0,
    })
    denom = grouped["sample_count"].clip(lower=1).pow(0.5)
    grouped["speaker_cosine_ci95"] = CI_Z * grouped["speaker_cosine_sample_std"] / denom
    grouped["emotion_ref_cosine_ci95"] = CI_Z * grouped["emotion_ref_cosine_sample_std"] / denom
    return grouped


def read_frontier_with_ci(path: Path) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    frontier = read_frontier(path)
    sample_path = infer_sample_metrics_path(path)
    if sample_path is None:
        return frontier, None

    ci = summarize_ci(read_metric_samples(sample_path))
    frontier = frontier.merge(
        ci[["alpha", "speaker_cosine_ci95", "emotion_ref_cosine_ci95", "sample_count"]],
        on="alpha",
        how="left",
    )
    return frontier, ci


def plot_dataset(dataset_key: str, spec: dict, output_dir: Path) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(8.2, 5.6), constrained_layout=True)
    manifest_rows = []

    for model_name, rel_path in spec["models"].items():
        csv_path = REPO_ROOT / rel_path
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)

        if dataset_key.startswith("esd"):
            df, ci = read_frontier_with_ci(csv_path)
            if ci is not None:
                ci = ci.copy()
                ci.insert(0, "model", model_name)
                ci.insert(0, "dataset", dataset_key)
                manifest_rows.extend(ci.to_dict("records"))
        else:
            df = read_frontier(csv_path)
            ci = None
        manifest_rows.append({"dataset": dataset_key, "model": model_name, "csv_path": rel_path})
        if dataset_key.startswith("esd") and {"speaker_cosine_ci95", "emotion_ref_cosine_ci95"}.issubset(df.columns):
            container = ax.errorbar(
                df["speaker_cosine"],
                df["emotion_ref_cosine"],
                xerr=df["speaker_cosine_ci95"],
                yerr=df["emotion_ref_cosine_ci95"],
                marker="o",
                linewidth=2.2,
                markersize=5.2,
                elinewidth=0.9,
                capsize=2.0,
                color=COLORS.get(model_name),
                label=model_name,
            )
            for bar_collection in container[2]:
                bar_collection.set_alpha(0.26)
            for cap in container[1]:
                cap.set_alpha(0.32)
        else:
            ax.plot(
                df["speaker_cosine"],
                df["emotion_ref_cosine"],
                marker="o",
                linewidth=2.2,
                markersize=5.2,
                color=COLORS.get(model_name),
                label=model_name,
            )

        for _, row in df.iterrows():
            ax.annotate(
                f"{row['alpha']:.1f}",
                (row["speaker_cosine"], row["emotion_ref_cosine"]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=9,
                color=COLORS.get(model_name),
            )

    ax.set_title(f"{spec['title']} Emotion/Speaker Frontier")
    ax.set_xlabel(r"Source-Speaker Similarity: $\mathrm{SIM}_{\mathrm{spk-src}}$")
    ax.set_ylabel(r"Target-Emotion Similarity: $\mathrm{SIM}_{\mathrm{emo}}$")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.legend(frameon=False, loc="best")

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{spec['output']}.png"
    pdf_path = output_dir / f"{spec['output']}.pdf"
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)

    manifest_df = pd.DataFrame([row for row in manifest_rows if "csv_path" in row])
    manifest_df.to_csv(output_dir / f"{spec['output']}_sources.csv", index=False)
    ci_rows = [row for row in manifest_rows if "sample_count" in row]
    if ci_rows:
        pd.DataFrame(ci_rows).to_csv(output_dir / f"{spec['output']}_ci95.csv", index=False)
    return [png_path, pdf_path]


def read_manifest_frontiers(manifest_path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    rows = []
    for item in manifest.itertuples(index=False):
        csv_path = REPO_ROOT / item.csv_path
        df = read_frontier(csv_path)
        df["dataset"] = item.dataset
        df["model"] = item.model
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def aggregate_by_alpha(frontiers: pd.DataFrame) -> pd.DataFrame:
    grouped = frontiers.groupby("alpha", as_index=False).agg(
        speaker_cosine_mean=("speaker_cosine", "mean"),
        speaker_cosine_std=("speaker_cosine", "std"),
        emotion_ref_cosine_mean=("emotion_ref_cosine", "mean"),
        emotion_ref_cosine_std=("emotion_ref_cosine", "std"),
        model_count=("model", "nunique"),
    )
    grouped = grouped.fillna({"speaker_cosine_std": 0.0, "emotion_ref_cosine_std": 0.0})
    denom = grouped["model_count"].clip(lower=1).pow(0.5)
    grouped["speaker_cosine_ci95"] = CI_Z * grouped["speaker_cosine_std"] / denom
    grouped["emotion_ref_cosine_ci95"] = CI_Z * grouped["emotion_ref_cosine_std"] / denom
    return grouped


def plot_average_dataset(manifest_path: Path, output_dir: Path) -> list[Path]:
    frontiers = read_manifest_frontiers(manifest_path)
    dataset_key = frontiers["dataset"].iloc[0]
    dataset_title = DATASET_DISPLAY.get(dataset_key, dataset_key)
    average = aggregate_by_alpha(frontiers)

    output_stem = manifest_path.name.replace("_frontier_comparison_sources.csv", "_average_frontier")
    average.to_csv(output_dir / f"{output_stem}.csv", index=False)

    plt.rcParams.update(PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(7.0, 5.4), constrained_layout=True)
    ax.errorbar(
        average["speaker_cosine_mean"],
        average["emotion_ref_cosine_mean"],
        xerr=average["speaker_cosine_ci95"],
        yerr=average["emotion_ref_cosine_ci95"],
        marker="o",
        linewidth=2.4,
        elinewidth=1.2,
        capsize=3,
        color="#2f4b7c",
        label="Model mean",
    )

    for _, row in average.iterrows():
        ax.annotate(
            f"{row['alpha']:.1f}",
            (row["speaker_cosine_mean"], row["emotion_ref_cosine_mean"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=9,
            color="#2f4b7c",
        )

    ax.set_title(f"{dataset_title} Average Frontier")
    ax.set_xlabel(r"Source-Speaker Similarity: $\mathrm{SIM}_{\mathrm{spk-src}}$")
    ax.set_ylabel(r"Target-Emotion Similarity: $\mathrm{SIM}_{\mathrm{emo}}$")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.legend(frameon=False, loc="best")

    png_path = output_dir / f"{output_stem}.png"
    pdf_path = output_dir / f"{output_stem}.pdf"
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)
    return [png_path, pdf_path, output_dir / f"{output_stem}.csv"]


def plot_average_datasets(output_dir: Path) -> list[Path]:
    return plot_average_dataset_overlay(output_dir)


def plot_average_dataset_overlay(output_dir: Path) -> list[Path]:
    plt.rcParams.update(PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(8.2, 5.6), constrained_layout=True)
    overlay_rows = []

    for manifest_path in sorted(output_dir.glob("*_frontier_comparison_sources.csv")):
        frontiers = read_manifest_frontiers(manifest_path)
        dataset_key = frontiers["dataset"].iloc[0]
        average = aggregate_by_alpha(frontiers)
        average["dataset"] = dataset_key
        overlay_rows.append(average)

        color = DATASET_COLORS.get(dataset_key)
        container = ax.errorbar(
            average["speaker_cosine_mean"],
            average["emotion_ref_cosine_mean"],
            xerr=average["speaker_cosine_ci95"],
            yerr=average["emotion_ref_cosine_ci95"],
            marker="o",
            linewidth=2.4,
            elinewidth=1.1,
            capsize=3,
            markersize=5.4,
            color=color,
            label=DATASET_DISPLAY.get(dataset_key, dataset_key),
        )
        for bar_collection in container[2]:
            bar_collection.set_alpha(0.32)
            bar_collection.set_linestyle("--")
        for cap in container[1]:
            cap.set_alpha(0.32)

        for _, row in average.iterrows():
            ax.annotate(
                f"{row['alpha']:.1f}",
                (row["speaker_cosine_mean"], row["emotion_ref_cosine_mean"]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=9,
                color=color,
            )

    overlay = pd.concat(overlay_rows, ignore_index=True)
    csv_path = output_dir / "dataset_average_frontier_comparison.csv"
    overlay.to_csv(csv_path, index=False)

    ax.set_title("Dataset-Averaged Emotion/Speaker Frontier")
    ax.set_xlabel(r"Source-Speaker Similarity: $\mathrm{SIM}_{\mathrm{spk-src}}$")
    ax.set_ylabel(r"Target-Emotion Similarity: $\mathrm{SIM}_{\mathrm{emo}}$")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.legend(frameon=False, loc="best")

    png_path = output_dir / "dataset_average_frontier_comparison.png"
    pdf_path = output_dir / "dataset_average_frontier_comparison.pdf"
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)
    return [png_path, pdf_path, csv_path]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="experiments/model_frontier_comparison",
        help="Directory where comparison plots and source manifests are written.",
    )
    args = parser.parse_args()

    output_dir = REPO_ROOT / args.output_dir
    written = []
    for dataset_key, spec in DATASETS.items():
        written.extend(plot_dataset(dataset_key, spec, output_dir))
    for manifest_path in sorted(output_dir.glob("*_frontier_comparison_sources.csv")):
        written.extend(plot_average_dataset(manifest_path, output_dir))
    written.extend(plot_average_datasets(output_dir))

    print("Wrote:")
    for path in written:
        print(path.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
