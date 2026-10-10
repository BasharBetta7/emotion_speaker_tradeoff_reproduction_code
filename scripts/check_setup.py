#!/usr/bin/env python3
"""Check the portable experiment package before launching a long evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path


REQUIRED = (
    "models.py",
    "eval_mel_viz.py",
    "emotion_recongizer.py",
    "Vocoder/config.yml",
    "Vocoder/checkpoint-400000steps.pkl",
    "Utils/JDC/model.py",
    "Utils/JDC/bst.t7",
    "scripts/run_full_evaluation_pipeline.sh",
    "scripts/run_balanced_benchmark_eval.py",
    "pretrained_models/wavlm_ser_esd/config.json",
    "pretrained_models/wavlm_ser_esd/pytorch_model.bin",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--allow-missing-experiment-checkpoints", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()

    missing = [relative for relative in REQUIRED if not (root / relative).is_file()]
    absolute_metadata: list[str] = []
    for path in sorted((root / "Data").glob("*")):
        if path.suffix.lower() not in {".csv", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if "/home/" in text or ":\\" in text:
            absolute_metadata.append(str(path.relative_to(root)))

    checkpoints = list((root / "Models").glob("**/*.pth"))
    scripts = list((root / "experiments").glob("**/reproduce_bash.sh"))
    print(f"Reproduction scripts: {len(scripts)}")
    print(f"StarGAN experiment checkpoints found: {len(checkpoints)}")
    if missing:
        print("Missing required runtime assets:")
        for relative in missing:
            print(f"  - {relative}")
    if absolute_metadata:
        print("Metadata still containing workstation-specific absolute paths:")
        for relative in absolute_metadata:
            print(f"  - {relative}")
    if not checkpoints and not args.allow_missing_experiment_checkpoints:
        print("No Models/**/*.pth experiment checkpoints were found.")

    failed = bool(missing or absolute_metadata)
    failed |= not checkpoints and not args.allow_missing_experiment_checkpoints
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
