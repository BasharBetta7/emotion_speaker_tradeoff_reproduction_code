#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd

METRIC_COLUMNS = [
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
    "emotion_ref_cosine",
    "emotion_conf",
]
SPEAKER_FRONTIER_METRICS = [
    (
        "speaker_gen_src_cosine",
        "Generated-Source Speaker Cosine Similarity",
        "StarGAN Source-Speaker Frontier with External Baseline Points",
        "tradeoff_frontier_source_speaker_with_baselines.png",
    ),
    (
        "speaker_gen_ref_cosine",
        "Generated-Reference Speaker Cosine Similarity",
        "StarGAN Reference-Speaker Frontier with External Baseline Points",
        "tradeoff_frontier_reference_speaker_with_baselines.png",
    ),
]
INTENSITY_MODES = {"native", "inferred", "fixed", "unknown"}


def has_metric_data(frame: pd.DataFrame, metric: str) -> bool:
    return metric in frame.columns and frame[metric].notna().any()


def resolve_reference_frontier_metric(stargan: pd.DataFrame, baseline: pd.DataFrame) -> tuple[str, str, str, str, bool]:
    if has_metric_data(stargan, "speaker_gen_ref_cosine") or has_metric_data(baseline, "speaker_gen_ref_cosine"):
        return (
            "speaker_gen_ref_cosine",
            "Generated-Reference Speaker Cosine Similarity",
            "StarGAN Reference-Speaker Frontier with External Baseline Points",
            "tradeoff_frontier_reference_speaker_with_baselines.png",
            False,
        )

    if has_metric_data(stargan, "speaker_ref_output_cosine") or has_metric_data(baseline, "speaker_ref_output_cosine"):
        return (
            "speaker_ref_output_cosine",
            "Reference-Output Speaker Cosine Similarity",
            "StarGAN Reference-Speaker Frontier with External Baseline Points",
            "tradeoff_frontier_reference_speaker_with_baselines.png",
            False,
        )

    return (
        "speaker_ref_cosine",
        "Source-Mixed-Reference Speaker Cosine Similarity (legacy control)",
        "StarGAN Legacy Reference-Control Frontier with External Baseline Points",
        "tradeoff_frontier_reference_speaker_with_baselines.png",
        True,
    )


def resolve_source_frontier_metric(stargan: pd.DataFrame, baseline: pd.DataFrame) -> tuple[str, str, str, str]:
    if has_metric_data(stargan, "speaker_gen_src_cosine") or has_metric_data(baseline, "speaker_gen_src_cosine"):
        return SPEAKER_FRONTIER_METRICS[0]

    return (
        "speaker_cosine",
        "Source-Output Speaker Cosine Similarity (legacy alias)",
        "StarGAN Source-Speaker Frontier with External Baseline Points",
        "tradeoff_frontier_source_speaker_with_baselines.png",
    )


def load_overall_metrics(eval_dir: Path) -> pd.DataFrame:
    csv_path = eval_dir / "overall_metrics.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing overall metrics file: {csv_path}")
    frame = pd.read_csv(csv_path)
    if frame.empty:
        raise ValueError(f"No rows in overall metrics: {csv_path}")
    return frame


def normalize_intensity_mode(frame: pd.DataFrame, default_mode: str) -> pd.DataFrame:
    out = frame.copy()
    if "intensity_mode" not in out.columns:
        out["intensity_mode"] = default_mode

    mode = out["intensity_mode"].astype(str).str.strip().str.lower()
    mode = mode.where(~mode.isin(["", "nan", "none"]), default_mode)
    mode = mode.where(mode.isin(INTENSITY_MODES), "unknown")
    out["intensity_mode"] = mode
    return out


def ensure_metric_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in METRIC_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    return out


def collect_stargan_points(
    alpha_roots: list[Path],
    labels: list[str],
    preserve_variant_names: bool,
) -> pd.DataFrame:
    rows = []

    if labels and len(labels) != len(alpha_roots):
        raise ValueError("--stargan-labels must be empty or match --stargan-alpha-roots length")

    for idx, alpha_root in enumerate(alpha_roots):
        if not alpha_root.exists():
            raise FileNotFoundError(f"Missing StarGAN alpha root: {alpha_root}")

        eval_dirs = sorted(alpha_root.glob("full_eval_*"))
        for eval_dir in eval_dirs:
            csv_path = eval_dir / "overall_metrics.csv"
            if not csv_path.exists():
                continue

            frame = pd.read_csv(csv_path)
            frame["alpha"] = pd.to_numeric(frame.get("alpha"), errors="coerce")
            frame = normalize_intensity_mode(frame, default_mode="native")
            frame = ensure_metric_columns(frame)

            if not preserve_variant_names:
                if labels:
                    assigned_name = labels[idx]
                elif len(alpha_roots) == 1:
                    assigned_name = "starganv2_vc"
                else:
                    assigned_name = alpha_root.name
                frame["model_variant"] = assigned_name
            elif "model_variant" not in frame.columns:
                frame["model_variant"] = labels[idx] if labels else alpha_root.name
            else:
                frame["model_variant"] = frame["model_variant"].fillna(labels[idx] if labels else alpha_root.name)

            rows.append(frame)

    if not rows:
        roots_text = ", ".join(str(root) for root in alpha_roots)
        raise FileNotFoundError(f"No overall_metrics.csv files found under: {roots_text}")

    all_rows = pd.concat(rows, ignore_index=True)
    grouped = (
        all_rows.groupby(["model_variant", "intensity_mode", "alpha"], dropna=False)[METRIC_COLUMNS]
        .mean()
        .reset_index()
        .sort_values(["model_variant", "alpha"])
    )
    return grouped


def infer_baseline_intensity_mode(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()

    if "intensity_mode" in out.columns:
        return normalize_intensity_mode(out, default_mode="unknown")

    model_text = out.get("model_variant", pd.Series([""] * len(out))).astype(str).str.lower()
    inferred = pd.Series(["fixed"] * len(out), index=out.index)
    inferred = inferred.where(~model_text.str.contains("inferred", na=False), "inferred")
    out["intensity_mode"] = inferred
    return normalize_intensity_mode(out, default_mode="fixed")


def collect_baseline_rows(eval_dirs: Iterable[Path]) -> pd.DataFrame:
    rows = []
    for eval_dir in eval_dirs:
        frame = load_overall_metrics(eval_dir)
        if "model_variant" not in frame.columns:
            frame["model_variant"] = eval_dir.name

        frame["alpha"] = pd.to_numeric(frame.get("alpha"), errors="coerce")
        frame = infer_baseline_intensity_mode(frame)
        frame = ensure_metric_columns(frame)
        rows.append(frame)

    if not rows:
        return pd.DataFrame(columns=["model_variant", "intensity_mode", "alpha", *METRIC_COLUMNS])

    baseline = pd.concat(rows, ignore_index=True)
    keep = ["model_variant", "intensity_mode", "alpha", *METRIC_COLUMNS]
    return baseline[keep]


def dominated_by_stargan(
    stargan: pd.DataFrame,
    speaker_metric: str,
    speaker_cosine: float,
    emotion_ref_cosine: float,
) -> tuple[bool, str | None, float | None]:
    if pd.isna(speaker_cosine) or pd.isna(emotion_ref_cosine):
        return False, None, None

    feasible = stargan[
        (stargan[speaker_metric] >= speaker_cosine)
        & (stargan["emotion_ref_cosine"] >= emotion_ref_cosine)
    ]
    if feasible.empty:
        return False, None, None

    strict = feasible[
        (feasible[speaker_metric] > speaker_cosine)
        | (feasible["emotion_ref_cosine"] > emotion_ref_cosine)
    ]
    if strict.empty:
        return False, None, None

    strict = strict.sort_values(["model_variant", "alpha"], na_position="last")
    first = strict.iloc[0]
    return True, str(first["model_variant"]), float(first["alpha"]) if pd.notna(first["alpha"]) else None


def build_dominance_table(stargan: pd.DataFrame, baseline: pd.DataFrame, speaker_metric: str) -> pd.DataFrame:
    if baseline.empty:
        return pd.DataFrame()

    stargan_native = stargan[stargan["intensity_mode"] == "native"]
    stargan_for_dom = stargan_native if not stargan_native.empty else stargan
    stargan_for_dom = stargan_for_dom.dropna(subset=[speaker_metric, "emotion_ref_cosine"])

    rows = []
    for item in baseline.itertuples(index=False):
        speaker_value = getattr(item, speaker_metric)
        is_dom, dom_model, dom_alpha = dominated_by_stargan(
            stargan_for_dom,
            speaker_metric,
            speaker_value,
            item.emotion_ref_cosine,
        )
        rows.append(
            {
                "model_variant": item.model_variant,
                "intensity_mode": item.intensity_mode,
                "alpha": item.alpha,
                "speaker_cosine": item.speaker_cosine,
                "speaker_ref_cosine": item.speaker_ref_cosine,
                "speaker_ref_output_cosine": item.speaker_ref_output_cosine,
                "speaker_gen_ref_cosine": item.speaker_gen_ref_cosine,
                "speaker_gen_src_cosine": item.speaker_gen_src_cosine,
                "speaker_src_ref_cosine": item.speaker_src_ref_cosine,
                "mcd_gen_ref": item.mcd_gen_ref,
                "mcd_gen_src": item.mcd_gen_src,
                "mcd_src_ref": item.mcd_src_ref,
                "emotion_acc": item.emotion_acc,
                "emotion_ref_cosine": item.emotion_ref_cosine,
                "emotion_conf": item.emotion_conf,
                "dominated_by_stargan_frontier": is_dom,
                "first_dominating_model_variant": dom_model,
                "first_dominating_alpha": dom_alpha,
            }
        )
    return pd.DataFrame(rows)


def save_frontier_plot(
    stargan: pd.DataFrame,
    baseline: pd.DataFrame,
    out_path: Path,
    speaker_metric: str,
    speaker_label: str,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 6.5))

    native = stargan[stargan["intensity_mode"] == "native"]
    if native.empty:
        native = stargan
    native = native.dropna(subset=[speaker_metric, "emotion_ref_cosine"])
    if native.empty:
        raise ValueError(f"No StarGAN rows have both {speaker_metric} and emotion_ref_cosine")

    scatter = None
    for model_variant, frame in native.groupby("model_variant"):
        frame = frame.sort_values("alpha")
        points = ax.scatter(
            frame[speaker_metric],
            frame["emotion_ref_cosine"],
            c=frame["alpha"],
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            s=95,
            edgecolors="none",
            zorder=3,
            label=f"{model_variant} (native)",
        )
        scatter = points
        ax.plot(
            frame[speaker_metric],
            frame["emotion_ref_cosine"],
            linewidth=2.0,
            alpha=0.85,
            zorder=2,
        )

    marker_by_mode = {
        "fixed": "X",
        "inferred": "P",
        "native": "*",
        "unknown": "D",
    }
    color_by_mode = {
        "fixed": "#d1495b",
        "inferred": "#f79256",
        "native": "#6c757d",
        "unknown": "#8d99ae",
    }

    if not baseline.empty:
        for mode, frame in baseline.groupby("intensity_mode"):
            frame = frame.dropna(subset=[speaker_metric, "emotion_ref_cosine"])
            if frame.empty:
                continue
            ax.scatter(
                frame[speaker_metric],
                frame["emotion_ref_cosine"],
                marker=marker_by_mode.get(mode, "D"),
                s=150,
                color=color_by_mode.get(mode, "#8d99ae"),
                zorder=4,
                label=f"External baselines ({mode})",
            )
            for row in frame.itertuples(index=False):
                alpha_text = "" if pd.isna(row.alpha) else f" a={row.alpha:.1f}"
                ax.text(
                    getattr(row, speaker_metric) + 0.002,
                    row.emotion_ref_cosine + 0.006,
                    f"{row.model_variant}{alpha_text}",
                    fontsize=8,
                )

    ax.set_xlabel(speaker_label)
    ax.set_ylabel("Emotion Reference Cosine Similarity")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    if scatter is not None:
        fig.colorbar(scatter, ax=ax, label="StarGAN alpha")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_markdown_summary(path: Path, dominance: pd.DataFrame, consolidated: pd.DataFrame) -> None:
    def table_text(frame: pd.DataFrame) -> str:
        try:
            return frame.to_markdown(index=False, floatfmt=".3f")
        except ImportError:
            return frame.to_string(index=False, float_format=lambda x: f"{x:.3f}")

    lines = ["# Baseline Comparison Summary", "", "## Consolidated operating points", ""]
    lines.append(table_text(consolidated))
    lines.append("")

    if dominance.empty:
        lines.append("No external baselines were provided.")
    else:
        lines.extend(["## Baseline frontier dominance against StarGAN alpha sweep", ""])
        lines.append(table_text(dominance))

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def resolve_stargan_roots(args: argparse.Namespace) -> list[Path]:
    if args.stargan_alpha_roots:
        return [Path(path) for path in args.stargan_alpha_roots]
    return [Path(args.stargan_alpha_root)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build consolidated StarGAN vs external-baseline table and overlay plot."
    )
    parser.add_argument("--stargan-alpha-root", default="experiments/full_set_alpha_variants")
    parser.add_argument(
        "--stargan-alpha-roots",
        nargs="*",
        default=[],
        help="Optional multi-root input for multiple native StarGAN variants.",
    )
    parser.add_argument(
        "--stargan-labels",
        nargs="*",
        default=[],
        help="Optional names aligned with --stargan-alpha-roots.",
    )
    parser.add_argument(
        "--preserve-stargan-variant-names",
        action="store_true",
        help="Preserve model_variant values from StarGAN overall_metrics.csv instead of assigning one label per root.",
    )
    parser.add_argument(
        "--baseline-eval-dirs",
        nargs="*",
        default=[],
        help="Directories containing overall_metrics.csv for external baseline evaluations.",
    )
    parser.add_argument("--output-dir", default="paper/artifacts/baseline_comparison")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stargan_roots = resolve_stargan_roots(args)
    stargan = collect_stargan_points(
        alpha_roots=stargan_roots,
        labels=list(args.stargan_labels),
        preserve_variant_names=args.preserve_stargan_variant_names,
    )

    baseline_dirs = [Path(path) for path in args.baseline_eval_dirs]
    baseline = collect_baseline_rows(baseline_dirs)

    if baseline.empty:
        consolidated = stargan.copy()
    else:
        consolidated = pd.concat([stargan, baseline], ignore_index=True, sort=False)
    consolidated = consolidated[[
        "model_variant",
        "intensity_mode",
        "alpha",
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
        "emotion_ref_cosine",
        "emotion_conf",
    ]]

    source_metric, source_label, source_title, source_filename = resolve_source_frontier_metric(stargan, baseline)
    source_dominance = build_dominance_table(stargan, baseline, source_metric)
    ref_metric, ref_label, ref_title, ref_filename, used_ref_fallback = resolve_reference_frontier_metric(stargan, baseline)
    ref_dominance = build_dominance_table(stargan, baseline, ref_metric)

    consolidated.to_csv(out_dir / "consolidated_overall_metrics.csv", index=False)
    source_dominance.to_csv(out_dir / "baseline_frontier_dominance_source_speaker.csv", index=False)
    ref_dominance.to_csv(out_dir / "baseline_frontier_dominance_reference_speaker.csv", index=False)
    source_plot = None
    ref_plot = None
    plot_configs = [
        (source_metric, source_label, source_title, source_filename),
        (ref_metric, ref_label, ref_title, ref_filename),
    ]
    for metric, label, title, filename in plot_configs:
        plot_path = out_dir / filename
        save_frontier_plot(stargan, baseline, plot_path, metric, label, title)
        if metric == source_metric:
            source_plot = plot_path
            save_frontier_plot(
                stargan,
                baseline,
                out_dir / "tradeoff_frontier_with_baselines.png",
                metric,
                label,
                title,
            )
        else:
            ref_plot = plot_path
    save_markdown_summary(out_dir / "summary.md", source_dominance, consolidated)

    print(f"Wrote consolidated metrics: {out_dir / 'consolidated_overall_metrics.csv'}")
    print(f"Wrote source-speaker dominance table: {out_dir / 'baseline_frontier_dominance_source_speaker.csv'}")
    print(f"Wrote reference-speaker dominance table: {out_dir / 'baseline_frontier_dominance_reference_speaker.csv'}")
    print(f"Wrote source-speaker frontier plot: {source_plot}")
    print(f"Wrote reference-speaker frontier plot: {ref_plot}")
    if used_ref_fallback:
        print(
            "WARNING: speaker_gen_ref_cosine/speaker_ref_output_cosine were not found; "
            "reference-speaker frontier used legacy speaker_ref_cosine instead."
        )
    print(f"Wrote legacy frontier plot alias: {out_dir / 'tradeoff_frontier_with_baselines.png'}")
    print(f"Wrote summary: {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
