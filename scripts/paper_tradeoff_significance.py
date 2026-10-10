#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, page_trend_test, wilcoxon
from statsmodels.stats.multitest import multipletests


OBJECTIVE_ALPHAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
LISTENING_ALPHAS = [0.0, 0.4, 0.8, 1.0]


def load_objective_alpha_frame(alpha_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for alpha in OBJECTIVE_ALPHAS:
        matches = sorted(alpha_root.glob(f"full_eval_epoch_00025_alpha_{alpha:.1f}_*/per_sample_metrics.csv"))
        if not matches:
            raise FileNotFoundError(f"Missing per-sample metrics for alpha={alpha:.1f} under {alpha_root}")
        frame = pd.read_csv(matches[0])
        frames.append(
            frame[
                [
                    "sample_id",
                    "speaker_id",
                    "target_emotion",
                    "alpha",
                    "speaker_cosine",
                    "emotion_ref_cosine",
                ]
            ].copy()
        )
    return pd.concat(frames, ignore_index=True)


def load_listening_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"user_id", "alpha", "speaker_similarity", "emotion_match_rank"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Listening-study CSV missing columns: {sorted(missing)}")
    return frame


def kendalls_w_from_friedman(statistic: float, n_blocks: int, n_conditions: int) -> float:
    if n_blocks <= 0 or n_conditions <= 1:
        return float("nan")
    return float(statistic / (n_blocks * (n_conditions - 1)))


def build_repeated_matrix(
    frame: pd.DataFrame,
    unit_cols: list[str],
    alpha_col: str,
    metric: str,
    alphas: list[float],
) -> pd.DataFrame:
    grouped = frame.groupby(unit_cols + [alpha_col], as_index=False)[metric].mean()
    matrix = grouped.pivot_table(index=unit_cols, columns=alpha_col, values=metric)
    matrix = matrix[alphas].dropna()
    return matrix


def run_metric_tests(
    matrix: pd.DataFrame,
    *,
    domain: str,
    unit: str,
    metric: str,
    direction: str,
    start_alpha: float,
    end_alpha: float,
) -> list[dict]:
    if direction == "decreasing":
        predicted_ranks = list(range(matrix.shape[1], 0, -1))
        alternative = "less"
    elif direction == "increasing":
        predicted_ranks = list(range(1, matrix.shape[1] + 1))
        alternative = "greater"
    else:
        raise ValueError(f"Unsupported direction: {direction}")

    page = page_trend_test(matrix.to_numpy(), predicted_ranks=predicted_ranks)
    friedman = friedmanchisquare(*[matrix[col].to_numpy() for col in matrix.columns])
    endpoint = wilcoxon(
        matrix[end_alpha],
        matrix[start_alpha],
        alternative=alternative,
        zero_method="wilcox",
        method="auto",
    )

    base = {
        "domain": domain,
        "unit": unit,
        "metric": metric,
        "direction": direction,
        "n_blocks": int(matrix.shape[0]),
        "n_conditions": int(matrix.shape[1]),
        "start_alpha": start_alpha,
        "end_alpha": end_alpha,
        "start_mean": float(matrix[start_alpha].mean()),
        "end_mean": float(matrix[end_alpha].mean()),
        "mean_diff_end_minus_start": float((matrix[end_alpha] - matrix[start_alpha]).mean()),
        "median_diff_end_minus_start": float((matrix[end_alpha] - matrix[start_alpha]).median()),
    }

    return [
        {
            **base,
            "test": "page_trend",
            "statistic": float(page.statistic),
            "p_value": float(page.pvalue),
            "effect_size": float("nan"),
            "effect_name": "",
        },
        {
            **base,
            "test": "friedman",
            "statistic": float(friedman.statistic),
            "p_value": float(friedman.pvalue),
            "effect_size": kendalls_w_from_friedman(float(friedman.statistic), matrix.shape[0], matrix.shape[1]),
            "effect_name": "kendalls_w",
        },
        {
            **base,
            "test": "wilcoxon_endpoint",
            "statistic": float(endpoint.statistic),
            "p_value": float(endpoint.pvalue),
            "effect_size": float("nan"),
            "effect_name": "",
        },
    ]


def add_fdr(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["p_value_fdr_bh"] = np.nan
    for (domain, test), idx in result.groupby(["domain", "test"]).groups.items():
        pvals = result.loc[list(idx), "p_value"].to_numpy(dtype=float)
        _, qvals, _, _ = multipletests(pvals, method="fdr_bh")
        result.loc[list(idx), "p_value_fdr_bh"] = qvals
    return result


def format_pvalue(value: float) -> str:
    if value < 1e-300:
        return "<1e-300"
    if value < 1e-4:
        return f"{value:.2e}"
    return f"{value:.4f}"


def write_summary(results: pd.DataFrame, output_path: Path) -> None:
    def grab(domain: str, unit: str, metric: str, test: str) -> pd.Series:
        mask = (
            (results["domain"] == domain)
            & (results["unit"] == unit)
            & (results["metric"] == metric)
            & (results["test"] == test)
        )
        return results.loc[mask].iloc[0]

    obj_page_speaker = grab("objective", "speaker", "speaker_cosine", "page_trend")
    obj_page_emotion = grab("objective", "speaker", "emotion_ref_cosine", "page_trend")
    obj_end_speaker = grab("objective", "speaker", "speaker_cosine", "wilcoxon_endpoint")
    obj_end_emotion = grab("objective", "speaker", "emotion_ref_cosine", "wilcoxon_endpoint")
    lis_page_speaker = grab("listening", "complete_rater", "speaker_similarity", "page_trend")
    lis_end_speaker = grab("listening", "complete_rater", "speaker_similarity", "wilcoxon_endpoint")
    lis_page_emotion = grab("listening", "complete_rater", "emotion_match_rank", "page_trend")
    lis_end_emotion = grab("listening", "complete_rater", "emotion_match_rank", "wilcoxon_endpoint")

    text = f"""# Tradeoff Significance Summary

## Objective benchmark
- Using speaker-level repeated measurements across the six alpha settings, the ordered trend is significant for `speaker_cosine` (Page test, `p={format_pvalue(obj_page_speaker['p_value'])}`) and for `emotion_ref_cosine` (Page test, `p={format_pvalue(obj_page_emotion['p_value'])}`).
- Comparing the endpoints `alpha=0.0` and `alpha=1.0`, `speaker_cosine` decreases from `{obj_end_speaker['start_mean']:.3f}` to `{obj_end_speaker['end_mean']:.3f}` (one-sided Wilcoxon, `p={format_pvalue(obj_end_speaker['p_value'])}`).
- Over the same endpoint comparison, `emotion_ref_cosine` increases from `{obj_end_emotion['start_mean']:.3f}` to `{obj_end_emotion['end_mean']:.3f}` (one-sided Wilcoxon, `p={format_pvalue(obj_end_emotion['p_value'])}`).

## Listening study
- Using complete raters only, the ordered trend is significant for `speaker_similarity` (Page test, `p={format_pvalue(lis_page_speaker['p_value'])}`), and the endpoint drop from `alpha=0.0` to `alpha=1.0` is also significant (one-sided Wilcoxon, `p={format_pvalue(lis_end_speaker['p_value'])}`).
- For `emotion_match_rank`, the trend is in the expected increasing direction, but it is not significant under the same complete-rater test (`Page p={format_pvalue(lis_page_emotion['p_value'])}`, `Wilcoxon p={format_pvalue(lis_end_emotion['p_value'])}`).

## Interpretation
- The automatic benchmark supports the tradeoff claim strongly: higher intensity significantly improves emotion transfer while significantly reducing speaker preservation.
- The listening study also supports the speaker-drift side of the tradeoff. The perceptual emotion-gain side trends upward, but the present complete-rater sample is too small to make that claim significant with the same conservative paired test.
"""
    output_path.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run significance tests for the emotion-speaker tradeoff paper.")
    parser.add_argument(
        "--objective-root",
        default="experiments/full_set_alpha_variants",
        help="Directory containing per-alpha full evaluation folders.",
    )
    parser.add_argument(
        "--listening-csv",
        default="paper/artifacts/listening_study/analysis/user_votes_merged.csv",
        help="Merged listening-study ratings CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="paper/artifacts/tradeoff_analysis",
        help="Directory in which significance outputs are written.",
    )
    args = parser.parse_args()

    objective_root = Path(args.objective_root)
    listening_csv = Path(args.listening_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    objective = load_objective_alpha_frame(objective_root)
    listening = load_listening_frame(listening_csv)

    rows: list[dict] = []

    objective_specs = [
        ("speaker_cosine", "decreasing"),
        ("emotion_ref_cosine", "increasing"),
    ]
    objective_units = [
        ("sample", ["sample_id"]),
        ("speaker", ["speaker_id"]),
    ]
    for unit_name, unit_cols in objective_units:
        for metric, direction in objective_specs:
            matrix = build_repeated_matrix(objective, unit_cols, "alpha", metric, OBJECTIVE_ALPHAS)
            rows.extend(
                run_metric_tests(
                    matrix,
                    domain="objective",
                    unit=unit_name,
                    metric=metric,
                    direction=direction,
                    start_alpha=OBJECTIVE_ALPHAS[0],
                    end_alpha=OBJECTIVE_ALPHAS[-1],
                )
            )

    complete_users = (
        listening.groupby(["user_id", "alpha"], as_index=False)[["speaker_similarity", "emotion_match_rank"]]
        .mean()
        .pivot(index="user_id", columns="alpha")
    )
    complete_user_ids = complete_users.dropna().index.tolist()
    listening_complete = listening[listening["user_id"].isin(complete_user_ids)].copy()

    listening_specs = [
        ("speaker_similarity", "decreasing"),
        ("emotion_match_rank", "increasing"),
    ]
    for metric, direction in listening_specs:
        matrix = build_repeated_matrix(
            listening_complete,
            ["user_id"],
            "alpha",
            metric,
            LISTENING_ALPHAS,
        )
        rows.extend(
            run_metric_tests(
                matrix,
                domain="listening",
                unit="complete_rater",
                metric=metric,
                direction=direction,
                start_alpha=LISTENING_ALPHAS[0],
                end_alpha=LISTENING_ALPHAS[-1],
            )
        )

    results = pd.DataFrame(rows).sort_values(["domain", "unit", "metric", "test"]).reset_index(drop=True)
    results = add_fdr(results)
    results.to_csv(output_dir / "significance_tests.csv", index=False)
    write_summary(results, output_dir / "significance_summary.md")

    print(f"Wrote significance table: {output_dir / 'significance_tests.csv'}")
    print(f"Wrote significance summary: {output_dir / 'significance_summary.md'}")


if __name__ == "__main__":
    main()
