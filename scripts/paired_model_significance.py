#!/usr/bin/env python3

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon


DEFAULT_METRICS = [
    "speaker_cosine",
    "speaker_gen_src_cosine",
    "speaker_gen_ref_cosine",
    "mcd_gen_src",
    "mcd_gen_ref",
    "emotion_acc",
    "emotion_ref_cosine",
    "emotion_conf",
]

LOWER_IS_BETTER = {
    "mcd_gen_src",
    "mcd_gen_ref",
    "mcd_src_ref",
}

BINARY_METRICS = {
    "emotion_acc",
}


@dataclass(frozen=True)
class ModelInput:
    label: str
    path: Path


def parse_model_arg(value: str) -> ModelInput:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--model must be formatted as LABEL=PATH")
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("Model label cannot be empty")
    return ModelInput(label=label, path=Path(path).expanduser())


def resolve_per_sample_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"Missing model path: {path}")
    paths = sorted(path.glob("full_eval_*/per_sample_metrics.csv"))
    if not paths:
        raise FileNotFoundError(f"No per_sample_metrics.csv files found under: {path}")
    return paths


def load_model_frame(model: ModelInput, alpha_values: set[float] | None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for csv_path in resolve_per_sample_paths(model.path):
        frame = pd.read_csv(csv_path)
        if "alpha" not in frame.columns:
            alpha = parse_alpha_from_path(csv_path)
            if alpha is not None:
                frame["alpha"] = alpha
        if alpha_values is not None:
            if "alpha" not in frame.columns:
                continue
            frame = frame[pd.to_numeric(frame["alpha"], errors="coerce").isin(alpha_values)]
        if not frame.empty:
            frames.append(frame)

    if not frames:
        alpha_text = "all alphas" if alpha_values is None else sorted(alpha_values)
        raise ValueError(f"No usable rows for {model.label} after alpha filter: {alpha_text}")

    out = pd.concat(frames, ignore_index=True)
    out["__model_label"] = model.label
    if "alpha" in out.columns:
        out["alpha"] = pd.to_numeric(out["alpha"], errors="coerce")
    return out


def parse_alpha_from_path(path: Path) -> float | None:
    for part in path.parts:
        if "_alpha_" not in part:
            continue
        suffix = part.split("_alpha_", 1)[1]
        alpha_text = suffix.split("_", 1)[0]
        try:
            return float(alpha_text)
        except ValueError:
            return None
    return None


def choose_key_columns(frames: dict[str, pd.DataFrame], requested: list[str] | None) -> list[str]:
    if requested:
        missing = {
            label: sorted(set(requested).difference(frame.columns))
            for label, frame in frames.items()
            if set(requested).difference(frame.columns)
        }
        if missing:
            raise KeyError(f"Requested key columns are missing: {missing}")
        return requested

    candidate_sets = [
        ["sample_id", "alpha"],
        ["sample_id"],
        ["source_wav", "reference_wav", "target_emotion", "alpha"],
        ["source_wav", "reference_wav", "target_emotion"],
    ]
    for cols in candidate_sets:
        if all(set(cols).issubset(frame.columns) for frame in frames.values()):
            return cols
    raise KeyError("Could not infer key columns. Pass --key-cols explicitly.")


def available_metrics(frames: dict[str, pd.DataFrame], requested: list[str]) -> list[str]:
    metrics = []
    for metric in requested:
        if all(metric in frame.columns for frame in frames.values()):
            metrics.append(metric)
    return metrics


def deduplicate_for_keys(frame: pd.DataFrame, key_cols: list[str], metrics: list[str], group_cols: list[str] | None = None) -> pd.DataFrame:
    group_cols = group_cols or []
    cols = list(dict.fromkeys(key_cols + group_cols + metrics))
    out = frame[cols].copy()
    for metric in metrics:
        out[metric] = pd.to_numeric(out[metric], errors="coerce")
    out = out.dropna(subset=key_cols)
    aggregations = {metric: "mean" for metric in metrics}
    aggregations.update({group_col: "first" for group_col in group_cols if group_col not in key_cols})
    return out.groupby(key_cols, as_index=False).agg(aggregations)


def paired_metric_frame(
    left: pd.DataFrame,
    right: pd.DataFrame,
    key_cols: list[str],
    metric: str,
) -> pd.DataFrame:
    joined = left[key_cols + [metric]].merge(
        right[key_cols + [metric]],
        on=key_cols,
        how="inner",
        suffixes=("_left", "_right"),
    )
    return joined.dropna(subset=[f"{metric}_left", f"{metric}_right"])


def bootstrap_mean_ci(
    values: np.ndarray,
    confidence: float,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    if n_bootstrap <= 0:
        return float("nan"), float("nan")
    draws = rng.integers(0, values.size, size=(n_bootstrap, values.size))
    means = values[draws].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(means, [alpha, 1.0 - alpha])
    return float(lower), float(upper)


def wilcoxon_pvalue(diff: np.ndarray) -> float:
    nonzero = diff[diff != 0]
    if nonzero.size == 0:
        return 1.0
    return float(wilcoxon(nonzero, alternative="two-sided", zero_method="wilcox", method="auto").pvalue)


def exact_mcnemar_pvalue(left_values: np.ndarray, right_values: np.ndarray) -> tuple[float, int, int]:
    left_binary = left_values.astype(float) > 0.5
    right_binary = right_values.astype(float) > 0.5
    left_only = int(np.sum(left_binary & ~right_binary))
    right_only = int(np.sum(~left_binary & right_binary))
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0, left_only, right_only
    p_value = binomtest(min(left_only, right_only), discordant, p=0.5, alternative="two-sided").pvalue
    return float(p_value), left_only, right_only


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    values = p_values.to_numpy(dtype=float)
    out = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values)
    if not valid.any():
        return pd.Series(out, index=p_values.index)

    valid_values = values[valid]
    order = np.argsort(valid_values)
    ranked = valid_values[order]
    n = ranked.size
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    valid_out = np.empty_like(adjusted)
    valid_out[order] = adjusted
    out[np.where(valid)[0]] = valid_out
    return pd.Series(out, index=p_values.index)


def compare_pair(
    left_label: str,
    right_label: str,
    left: pd.DataFrame,
    right: pd.DataFrame,
    key_cols: list[str],
    metrics: list[str],
    confidence: float,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> list[dict]:
    rows: list[dict] = []
    for metric in metrics:
        joined = paired_metric_frame(left, right, key_cols, metric)
        left_values = joined[f"{metric}_left"].to_numpy(dtype=float)
        right_values = joined[f"{metric}_right"].to_numpy(dtype=float)
        diff = left_values - right_values
        ci_low, ci_high = bootstrap_mean_ci(diff, confidence, n_bootstrap, rng)
        direction = "lower" if metric in LOWER_IS_BETTER else "higher"

        if metric in BINARY_METRICS:
            p_value, left_only, right_only = exact_mcnemar_pvalue(left_values, right_values)
            test = "exact_mcnemar"
        else:
            p_value = wilcoxon_pvalue(diff)
            left_only = right_only = pd.NA
            test = "wilcoxon_signed_rank"

        if direction == "higher":
            significant_favorable = ci_low > 0.0
            significant_unfavorable = ci_high < 0.0
        else:
            significant_favorable = ci_high < 0.0
            significant_unfavorable = ci_low > 0.0

        rows.append(
            {
                "left_model": left_label,
                "right_model": right_label,
                "metric": metric,
                "direction": direction,
                "n_paired": int(joined.shape[0]),
                "left_mean": float(np.nanmean(left_values)) if left_values.size else float("nan"),
                "right_mean": float(np.nanmean(right_values)) if right_values.size else float("nan"),
                "mean_diff_left_minus_right": float(np.nanmean(diff)) if diff.size else float("nan"),
                "median_diff_left_minus_right": float(np.nanmedian(diff)) if diff.size else float("nan"),
                "ci_level": confidence,
                "mean_diff_ci_low": ci_low,
                "mean_diff_ci_high": ci_high,
                "test": test,
                "p_value": p_value,
                "left_correct_right_wrong": left_only,
                "left_wrong_right_correct": right_only,
                "significant_favorable_by_ci": bool(significant_favorable),
                "significant_unfavorable_by_ci": bool(significant_unfavorable),
            }
        )
    return rows


def run_comparisons(
    prepared: dict[str, pd.DataFrame],
    pairs: list[tuple[str, str]],
    key_cols: list[str],
    metrics: list[str],
    confidence: float,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows: list[dict] = []
    for left_label, right_label in pairs:
        rows.extend(
            compare_pair(
                left_label,
                right_label,
                prepared[left_label],
                prepared[right_label],
                key_cols,
                metrics,
                confidence,
                n_bootstrap,
                rng,
            )
        )

    results = pd.DataFrame(rows)
    results["p_value_fdr_bh"] = benjamini_hochberg(results["p_value"])
    return results


def run_grouped_comparisons(
    prepared: dict[str, pd.DataFrame],
    pairs: list[tuple[str, str]],
    key_cols: list[str],
    group_cols: list[str],
    metrics: list[str],
    confidence: float,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    group_key_cols = list(dict.fromkeys(key_cols + group_cols))
    rows: list[dict] = []

    for group_values in common_group_values(prepared, group_cols):
        filtered = {
            label: filter_group(frame, group_cols, group_values)
            for label, frame in prepared.items()
        }
        for left_label, right_label in pairs:
            pair_rows = compare_pair(
                left_label,
                right_label,
                filtered[left_label],
                filtered[right_label],
                group_key_cols,
                metrics,
                confidence,
                n_bootstrap,
                rng,
            )
            for row in pair_rows:
                row.update({col: value for col, value in zip(group_cols, group_values)})
            rows.extend(pair_rows)

    results = pd.DataFrame(rows)
    if not results.empty:
        results["p_value_fdr_bh"] = benjamini_hochberg(results["p_value"])
    return results


def common_group_values(prepared: dict[str, pd.DataFrame], group_cols: list[str]) -> list[tuple]:
    value_sets = []
    for frame in prepared.values():
        values = set(map(tuple, frame[group_cols].dropna().drop_duplicates().to_numpy()))
        value_sets.append(values)
    if not value_sets:
        return []
    common = set.intersection(*value_sets)
    return sorted(common)


def filter_group(frame: pd.DataFrame, group_cols: list[str], group_values: tuple) -> pd.DataFrame:
    mask = pd.Series(True, index=frame.index)
    for col, value in zip(group_cols, group_values):
        mask &= frame[col] == value
    return frame.loc[mask].copy()


def write_markdown(
    results: pd.DataFrame,
    output_path: Path,
    key_cols: list[str],
    model_paths: list[ModelInput],
    metrics: list[str],
    confidence: float,
    n_bootstrap: int,
    group_cols: list[str] | None = None,
) -> None:
    group_cols = group_cols or []
    lines = [
        "# Paired Model Significance Tests",
        "",
        "## Method",
        "",
        "Rows are aligned by paired evaluation keys, so each comparison asks whether two models differ on the same source/reference/example pairs rather than on two independent samples.",
        "",
        f"- Pairing keys: `{', '.join(key_cols)}`.",
        f"- Grouping columns: `{', '.join(group_cols)}`." if group_cols else "- Grouping columns: none; this is an aggregate test.",
        f"- Confidence interval: non-parametric paired bootstrap over aligned rows, `{confidence:.0%}` interval, `{n_bootstrap}` resamples.",
        "- Continuous metrics: two-sided Wilcoxon signed-rank test on paired differences.",
        "- Binary `emotion_acc`: exact McNemar/binomial sign test on discordant pairs.",
        "- Multiple comparisons: Benjamini-Hochberg FDR correction across all reported pair/metric tests.",
        "",
        "Interpretation rule used here: an improvement is marked significant only when the confidence interval excludes zero in the favorable direction. The p-value and FDR-adjusted p-value are reported as supporting evidence, not as a replacement for effect size.",
        "",
        "References: Wilcoxon signed-rank test (Wilcoxon, 1945); McNemar's paired nominal test (McNemar, 1947); bootstrap confidence intervals (Efron and Tibshirani, 1993); Benjamini-Hochberg false-discovery-rate control (Benjamini and Hochberg, 1995).",
        "",
        "## Inputs",
        "",
    ]
    for model in model_paths:
        lines.append(f"- `{model.label}`: `{model.path}`")

    display = results.copy()
    for col in [
        "left_mean",
        "right_mean",
        "mean_diff_left_minus_right",
        "mean_diff_ci_low",
        "mean_diff_ci_high",
        "p_value",
        "p_value_fdr_bh",
    ]:
        display[col] = display[col].map(lambda value: "" if pd.isna(value) else f"{value:.6g}")

    compact_cols = [
        *group_cols,
        "left_model",
        "right_model",
        "metric",
        "direction",
        "n_paired",
        "left_mean",
        "right_mean",
        "mean_diff_left_minus_right",
        "mean_diff_ci_low",
        "mean_diff_ci_high",
        "test",
        "p_value",
        "p_value_fdr_bh",
        "significant_favorable_by_ci",
        "significant_unfavorable_by_ci",
    ]

    lines.extend(
        [
            "",
            "## Results",
            "",
            display[compact_cols].to_markdown(index=False),
            "",
            "## Practical Notes",
            "",
            "- For `higher` metrics, positive `left - right` is favorable to the left model.",
            "- For `lower` metrics, negative `left - right` is favorable to the left model.",
            "- A statistically clear speaker gain with no clear emotion loss is stronger evidence for a useful tradeoff than a p-value alone.",
            "- Treat repeated checkpoints from the same training run as checkpoint-selection evidence, not independent training-seed evidence.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run paired significance tests with bootstrap confidence intervals between model per-sample metrics."
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        type=parse_model_arg,
        help="Model input as LABEL=PATH. PATH may be a per_sample_metrics.csv file or an alpha-sweep root containing full_eval_*/per_sample_metrics.csv.",
    )
    parser.add_argument(
        "--reference",
        default=None,
        help="If set, compare only REFERENCE vs every other model. Default: all pairwise comparisons.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        help="Metric columns to test when present in every model.",
    )
    parser.add_argument(
        "--alpha",
        nargs="+",
        type=float,
        default=None,
        help="Optional alpha values to include, e.g. --alpha 1.0 or --alpha 0.0 0.2 0.4 0.6 0.8 1.0.",
    )
    parser.add_argument(
        "--key-cols",
        nargs="+",
        default=None,
        help="Columns used to align paired rows. Default tries sample_id alpha, then sample_id.",
    )
    parser.add_argument(
        "--group-cols",
        nargs="+",
        default=None,
        help="Optional grouping columns for stratified tests, e.g. --group-cols target_emotion.",
    )
    parser.add_argument("--confidence", type=float, default=0.95, help="Bootstrap confidence level.")
    parser.add_argument("--n-bootstrap", type=int, default=5000, help="Number of paired bootstrap resamples.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for bootstrap resampling.")
    parser.add_argument("--output-dir", default="research_mindhub/exp_wiki/significance_tests", help="Output directory.")
    parser.add_argument("--output-prefix", default="paired_model_significance", help="Output file prefix.")
    args = parser.parse_args()

    if len(args.model) < 2:
        raise ValueError("At least two --model inputs are required")
    labels = [model.label for model in args.model]
    if len(labels) != len(set(labels)):
        raise ValueError(f"Model labels must be unique: {labels}")
    if args.reference is not None and args.reference not in labels:
        raise ValueError(f"--reference must match one of the model labels: {labels}")

    alpha_values = set(args.alpha) if args.alpha is not None else None
    frames = {model.label: load_model_frame(model, alpha_values) for model in args.model}
    key_cols = choose_key_columns(frames, args.key_cols)
    metrics = available_metrics(frames, args.metrics)
    if not metrics:
        raise ValueError("None of the requested metrics are present in every model")
    group_cols = args.group_cols or []
    if group_cols:
        missing_groups = {
            label: sorted(set(group_cols).difference(frame.columns))
            for label, frame in frames.items()
            if set(group_cols).difference(frame.columns)
        }
        if missing_groups:
            raise KeyError(f"Requested group columns are missing: {missing_groups}")

    prepared = {
        label: deduplicate_for_keys(frame, key_cols, metrics, group_cols=group_cols)
        for label, frame in frames.items()
    }

    if args.reference:
        pairs = [(args.reference, label) for label in labels if label != args.reference]
    else:
        pairs = list(combinations(labels, 2))

    rng = np.random.default_rng(args.seed)
    results = run_comparisons(
        prepared,
        pairs,
        key_cols,
        metrics,
        args.confidence,
        args.n_bootstrap,
        rng,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{args.output_prefix}.csv"
    md_path = output_dir / f"{args.output_prefix}.md"
    results.to_csv(csv_path, index=False)
    write_markdown(results, md_path, key_cols, args.model, metrics, args.confidence, args.n_bootstrap)
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {md_path}")

    if group_cols:
        grouped_results = run_grouped_comparisons(
            prepared,
            pairs,
            key_cols,
            group_cols,
            metrics,
            args.confidence,
            args.n_bootstrap,
            rng,
        )
        group_suffix = "_by_" + "_".join(group_cols)
        grouped_csv_path = output_dir / f"{args.output_prefix}{group_suffix}.csv"
        grouped_md_path = output_dir / f"{args.output_prefix}{group_suffix}.md"
        grouped_results.to_csv(grouped_csv_path, index=False)
        write_markdown(
            grouped_results,
            grouped_md_path,
            list(dict.fromkeys(key_cols + group_cols)),
            args.model,
            metrics,
            args.confidence,
            args.n_bootstrap,
            group_cols=group_cols,
        )
        print(f"Wrote: {grouped_csv_path}")
        print(f"Wrote: {grouped_md_path}")


if __name__ == "__main__":
    main()
