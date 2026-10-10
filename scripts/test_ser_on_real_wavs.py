#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from emotion_recongizer import SpeechEmotionRecognizer


SER_LABEL_TO_CANONICAL = {
    "ang": "Angry",
    "hap": "Happy",
    "neu": "Neutral",
    "sad": "Sad",
}


def normalize_prediction(label: str) -> str:
    return SER_LABEL_TO_CANONICAL.get(str(label).strip().lower(), str(label).strip())


def build_records_from_manifest(manifest_path: Path, mode: str) -> pd.DataFrame:
    df = pd.read_csv(manifest_path, sep="\t")
    if mode == "source":
        out = df[["sample_id", "source_wav"]].drop_duplicates().rename(columns={"source_wav": "wav_path"})
        out["true_emotion"] = "Neutral"
        out["role"] = "source"
    elif mode == "reference":
        out = df[["sample_id", "reference_wav", "target_emotion"]].rename(
            columns={"reference_wav": "wav_path", "target_emotion": "true_emotion"}
        )
        out["role"] = "reference"
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return out.reset_index(drop=True)


def build_records_from_pool(pool_path: Path, source_emotion: str | None) -> pd.DataFrame:
    df = pd.read_csv(pool_path, sep="\t")
    if source_emotion is not None:
        df = df[df["emotion"].str.lower() == source_emotion.strip().lower()].copy()
    out = df[["file_id", "wav_path", "emotion"]].rename(
        columns={"file_id": "sample_id", "emotion": "true_emotion"}
    )
    out["role"] = "pool"
    return out.reset_index(drop=True)


def build_records_from_wav_list(wav_list_path: Path, true_emotion: str | None) -> pd.DataFrame:
    rows = []
    with open(wav_list_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            wav_path = line.strip()
            if not wav_path:
                continue
            rows.append(
                {
                    "sample_id": f"wav_{idx:04d}",
                    "wav_path": wav_path,
                    "true_emotion": true_emotion,
                    "role": "custom",
                }
            )
    return pd.DataFrame(rows)


def build_records_from_single_wav(wav_path: str, true_emotion: str | None) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sample_id": Path(wav_path).stem,
                "wav_path": wav_path,
                "true_emotion": true_emotion,
                "role": "custom",
            }
        ]
    )


def run_predictions(records: pd.DataFrame, ser: SpeechEmotionRecognizer) -> pd.DataFrame:
    output_rows = []
    for row in records.itertuples(index=False):
        result = ser.predict_file(row.wav_path)
        predicted = normalize_prediction(result["label"])
        true_emotion = getattr(row, "true_emotion", None)
        is_correct = None
        if isinstance(true_emotion, str) and true_emotion.strip():
            is_correct = float(predicted.lower() == true_emotion.strip().lower())

        output_rows.append(
            {
                "sample_id": row.sample_id,
                "wav_path": row.wav_path,
                "role": row.role,
                "true_emotion": true_emotion,
                "predicted_emotion": predicted,
                "class_idx": result["class_idx"],
                "score": result["score"],
                "is_correct": is_correct,
            }
        )
    return pd.DataFrame(output_rows)


def print_summary(results: pd.DataFrame) -> None:
    print("\nPrediction counts:")
    print(results["predicted_emotion"].value_counts(dropna=False).to_string())

    if "true_emotion" in results.columns and results["true_emotion"].notna().any():
        labeled = results[results["true_emotion"].notna()].copy()
        if not labeled.empty:
            print("\nPer-label counts:")
            print(labeled.groupby("true_emotion")["predicted_emotion"].value_counts().to_string())

            confusion = pd.crosstab(
                labeled["true_emotion"],
                labeled["predicted_emotion"],
                rownames=["true_emotion"],
                colnames=["predicted_emotion"],
                dropna=False,
            )
            print("\nConfusion matrix:")
            print(confusion.to_string())

            valid = labeled["is_correct"].notna()
            if valid.any():
                print("\nAccuracy:")
                print(f"overall = {labeled.loc[valid, 'is_correct'].mean():.4f}")
                print(labeled.groupby("true_emotion")["is_correct"].mean().to_string())


def build_confusion_matrix(results: pd.DataFrame) -> pd.DataFrame | None:
    if "true_emotion" not in results.columns or not results["true_emotion"].notna().any():
        return None
    labeled = results[results["true_emotion"].notna()].copy()
    if labeled.empty:
        return None
    confusion = pd.crosstab(
        labeled["true_emotion"],
        labeled["predicted_emotion"],
        rownames=["true_emotion"],
        colnames=["predicted_emotion"],
        dropna=False,
    )
    return confusion


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the SER model on real WAV files.")
    parser.add_argument("--wav-path", default=None, help="Single wav file to test.")
    parser.add_argument("--wav-list", default=None, help="Text file containing one wav path per line.")
    parser.add_argument("--manifest-path", default=None, help="Benchmark manifest TSV.")
    parser.add_argument("--pool-path", default=None, help="Benchmark pool TSV with wav_path and emotion columns.")
    parser.add_argument(
        "--manifest-mode",
        choices=["source", "reference"],
        default="source",
        help="When using --manifest-path, test either source wavs or reference wavs.",
    )
    parser.add_argument(
        "--source-emotion",
        default=None,
        help="Optional emotion filter when using --pool-path, e.g. Angry, Happy, Neutral, Sad, Surprise.",
    )
    parser.add_argument("--true-emotion", default=None, help="Optional label for --wav-path or --wav-list.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--emotion-model-dir", default="Models/wavlm_ser_esd")
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    provided = [
        args.wav_path is not None,
        args.wav_list is not None,
        args.manifest_path is not None,
        args.pool_path is not None,
    ]
    if sum(provided) != 1:
        raise ValueError("Provide exactly one of --wav-path, --wav-list, --manifest-path, or --pool-path.")

    if args.manifest_path is not None:
        records = build_records_from_manifest(Path(args.manifest_path), args.manifest_mode)
    elif args.pool_path is not None:
        records = build_records_from_pool(Path(args.pool_path), args.source_emotion)
    elif args.wav_list is not None:
        records = build_records_from_wav_list(Path(args.wav_list), args.true_emotion)
    else:
        records = build_records_from_single_wav(args.wav_path, args.true_emotion)

    ser = SpeechEmotionRecognizer(device=args.device, savedir=args.emotion_model_dir)
    results = run_predictions(records, ser)

    print(results.head(20).to_string(index=False))
    print_summary(results)

    if args.output_csv:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_path, index=False)
        print(f"\nSaved predictions to {output_path}")

        confusion = build_confusion_matrix(results)
        if confusion is not None:
            confusion_path = output_path.with_name(f"{output_path.stem}_confusion_matrix.csv")
            confusion.to_csv(confusion_path)
            print(f"Saved confusion matrix to {confusion_path}")


if __name__ == "__main__":
    main()
