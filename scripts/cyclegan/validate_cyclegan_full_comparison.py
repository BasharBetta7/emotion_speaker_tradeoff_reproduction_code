#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=None, engine="python")


def ensure_columns(frame: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise KeyError(f"{name} is missing required columns: {missing}")


def normalize_targets(values: Iterable[str]) -> list[str]:
    return [str(value).strip().lower() for value in values]


def normalize_alphas(values: Iterable[float]) -> list[float]:
    return sorted({round(float(v), 6) for v in values})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate full CycleGAN fixed/inferred comparison outputs for benchmark coverage and schema consistency."
    )
    parser.add_argument("--benchmark-manifest", default="Data/esd_balanced_benchmark_manifest.csv")
    parser.add_argument("--fixed-manifest", required=True)
    parser.add_argument("--fixed-per-sample", required=True)
    parser.add_argument("--inferred-per-sample", required=True)
    parser.add_argument("--targets", nargs="+", default=["Angry", "Happy", "Sad", "Surprise"])
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    parser.add_argument("--expected-benchmark-rows", type=int, default=800)
    parser.add_argument(
        "--scope-from-fixed-manifest",
        action="store_true",
        help="Validate against the benchmark subset referenced by fixed manifest sample_id values (useful for smoke runs).",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark = read_table(Path(args.benchmark_manifest))
    fixed_manifest = read_table(Path(args.fixed_manifest))
    fixed_per_sample = read_table(Path(args.fixed_per_sample))
    inferred_per_sample = read_table(Path(args.inferred_per_sample))

    ensure_columns(benchmark, ["sample_id", "target_emotion"], "benchmark manifest")
    ensure_columns(fixed_manifest, ["sample_id", "generated_wav", "model_variant"], "fixed generated manifest")
    ensure_columns(fixed_per_sample, ["sample_id", "target_emotion"], "fixed per-sample metrics")
    ensure_columns(inferred_per_sample, ["sample_id", "target_emotion", "alpha"], "inferred per-sample metrics")

    target_norm = set(normalize_targets(args.targets))
    alpha_expected = normalize_alphas(args.alphas)

    fixed_manifest = fixed_manifest.copy()
    fixed_manifest["sample_id"] = fixed_manifest["sample_id"].astype(str)
    fixed_ids = set(fixed_manifest["sample_id"].tolist())

    benchmark = benchmark.copy()
    benchmark["sample_id"] = benchmark["sample_id"].astype(str)
    benchmark["target_emotion_norm"] = benchmark["target_emotion"].astype(str).str.strip().str.lower()
    bench_target_all = benchmark[benchmark["target_emotion_norm"].isin(target_norm)].copy()
    if args.scope_from_fixed_manifest:
        bench_target = bench_target_all[bench_target_all["sample_id"].isin(fixed_ids)].copy()
    else:
        bench_target = bench_target_all
    bench_ids = set(bench_target["sample_id"].tolist())

    fixed_per_sample = fixed_per_sample.copy()
    fixed_per_sample["sample_id"] = fixed_per_sample["sample_id"].astype(str)
    fixed_metric_ids = set(fixed_per_sample["sample_id"].tolist())
    fixed_per_sample["target_emotion_norm"] = fixed_per_sample["target_emotion"].astype(str).str.strip().str.lower()

    inferred_per_sample = inferred_per_sample.copy()
    inferred_per_sample["sample_id"] = inferred_per_sample["sample_id"].astype(str)
    inferred_per_sample["target_emotion_norm"] = (
        inferred_per_sample["target_emotion"].astype(str).str.strip().str.lower()
    )
    inferred_alphas = normalize_alphas(inferred_per_sample["alpha"].dropna().tolist())

    checks: list[tuple[str, bool, str]] = []

    checks.append(
        (
            "benchmark_row_count",
            len(bench_target) == args.expected_benchmark_rows,
            f"expected={args.expected_benchmark_rows}, observed={len(bench_target)}",
        )
    )
    extra_ids_in_fixed = sorted(fixed_ids - set(bench_target_all["sample_id"].tolist()))
    checks.append(
        (
            "fixed_manifest_sample_id_validity",
            len(extra_ids_in_fixed) == 0,
            f"extra_ids={len(extra_ids_in_fixed)}",
        )
    )
    checks.append(
        (
            "fixed_manifest_row_count",
            len(fixed_manifest) == len(bench_target),
            f"expected={len(bench_target)}, observed={len(fixed_manifest)}",
        )
    )
    checks.append(
        (
            "fixed_metrics_row_count",
            len(fixed_per_sample) == len(bench_target),
            f"expected={len(bench_target)}, observed={len(fixed_per_sample)}",
        )
    )
    checks.append(
        (
            "inferred_metrics_row_count",
            len(inferred_per_sample) == len(bench_target) * len(alpha_expected),
            f"expected={len(bench_target) * len(alpha_expected)}, observed={len(inferred_per_sample)}",
        )
    )

    missing_in_fixed_manifest = sorted(bench_ids - fixed_ids)
    missing_in_fixed_metrics = sorted(bench_ids - fixed_metric_ids)
    checks.append(
        (
            "fixed_manifest_sample_id_coverage",
            len(missing_in_fixed_manifest) == 0,
            f"missing={len(missing_in_fixed_manifest)}",
        )
    )
    checks.append(
        (
            "fixed_metrics_sample_id_coverage",
            len(missing_in_fixed_metrics) == 0,
            f"missing={len(missing_in_fixed_metrics)}",
        )
    )

    fixed_target_set = set(fixed_per_sample["target_emotion_norm"].tolist())
    inferred_target_set = set(inferred_per_sample["target_emotion_norm"].tolist())
    checks.append(
        (
            "fixed_target_emotion_coverage",
            fixed_target_set == target_norm,
            f"expected={sorted(target_norm)}, observed={sorted(fixed_target_set)}",
        )
    )
    checks.append(
        (
            "inferred_target_emotion_coverage",
            inferred_target_set == target_norm,
            f"expected={sorted(target_norm)}, observed={sorted(inferred_target_set)}",
        )
    )
    checks.append(
        (
            "inferred_alpha_cardinality",
            inferred_alphas == alpha_expected,
            f"expected={alpha_expected}, observed={inferred_alphas}",
        )
    )

    rows_per_sample = inferred_per_sample.groupby("sample_id").size()
    bad_alpha_rows = rows_per_sample[rows_per_sample != len(alpha_expected)]
    checks.append(
        (
            "inferred_rows_per_sample",
            bad_alpha_rows.empty,
            f"samples_with_unexpected_alpha_count={len(bad_alpha_rows)}",
        )
    )

    if "intensity_mode" in fixed_per_sample.columns:
        fixed_modes = set(fixed_per_sample["intensity_mode"].astype(str).str.strip().str.lower().tolist())
        checks.append(
            (
                "fixed_intensity_mode_label",
                fixed_modes == {"fixed"},
                f"expected=['fixed'], observed={sorted(fixed_modes)}",
            )
        )

    if "intensity_mode" in inferred_per_sample.columns:
        inferred_modes = set(inferred_per_sample["intensity_mode"].astype(str).str.strip().str.lower().tolist())
        checks.append(
            (
                "inferred_intensity_mode_label",
                inferred_modes == {"inferred"},
                f"expected=['inferred'], observed={sorted(inferred_modes)}",
            )
        )

    all_ok = all(item[1] for item in checks)

    report = {
        "all_checks_passed": all_ok,
        "benchmark_target_rows": int(len(bench_target)),
        "fixed_manifest_rows": int(len(fixed_manifest)),
        "fixed_metrics_rows": int(len(fixed_per_sample)),
        "inferred_metrics_rows": int(len(inferred_per_sample)),
        "expected_alphas": alpha_expected,
        "observed_alphas": inferred_alphas,
        "checks": [
            {
                "name": name,
                "passed": passed,
                "detail": detail,
            }
            for name, passed, detail in checks
        ],
        "missing_sample_id_preview": {
            "fixed_manifest": missing_in_fixed_manifest[:10],
            "fixed_metrics": missing_in_fixed_metrics[:10],
            "fixed_manifest_extra_ids": extra_ids_in_fixed[:10],
        },
    }

    (output_dir / "validation_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        [{"check": name, "passed": passed, "detail": detail} for name, passed, detail in checks]
    ).to_csv(output_dir / "validation_checks.csv", index=False)

    print(f"Validation summary written: {output_dir / 'validation_summary.json'}")
    print(f"Validation checks written: {output_dir / 'validation_checks.csv'}")
    if not all_ok:
        raise SystemExit("Validation failed: one or more checks did not pass.")


if __name__ == "__main__":
    main()
