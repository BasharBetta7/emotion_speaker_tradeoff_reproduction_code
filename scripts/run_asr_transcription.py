#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


def load_transformers_pipeline(model_name: str, device: str | None):
    try:
        import torch
        from transformers import pipeline
    except ImportError as exc:
        raise ImportError(
            "transformers and torch are required for ASR inference. "
            "Install the project requirements in the environment where you run this script."
        ) from exc

    if device is None:
        pipe_device: int | str = 0 if torch.cuda.is_available() else "cpu"
    else:
        normalized = str(device).strip().lower()
        if normalized == "cpu":
            pipe_device = "cpu"
        elif normalized.startswith("cuda"):
            if ":" in normalized:
                pipe_device = int(normalized.split(":", 1)[1])
            else:
                pipe_device = 0
        else:
            pipe_device = device

    return pipeline(
        task="automatic-speech-recognition",
        model=model_name,
        device=pipe_device,
    )


def transcribe_with_transformers(
    frame: pd.DataFrame,
    audio_column: str,
    model_name: str,
    device: str | None,
    batch_size: int,
) -> pd.DataFrame:
    pipe = load_transformers_pipeline(model_name=model_name, device=device)
    audio_paths = frame[audio_column].astype(str).tolist()
    results = pipe(audio_paths, batch_size=batch_size)
    if isinstance(results, dict):
        results = [results]

    predicted_texts: list[str] = []
    for result in results:
        if isinstance(result, dict):
            predicted_texts.append(str(result.get("text", "")).strip())
        else:
            predicted_texts.append(str(result).strip())

    output = frame.copy()
    output["predicted_text"] = predicted_texts
    output["backend"] = "transformers"
    output["backend_model"] = model_name
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ASR over a transcript template and fill predicted_text."
    )
    parser.add_argument("--input-csv", required=True, help="Transcript template or any CSV with an audio path column.")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--audio-column", default="audio_path")
    parser.add_argument("--backend", choices=["transformers"], default="transformers")
    parser.add_argument(
        "--model-name",
        default="openai/whisper-small",
        help="Model identifier understood by the selected backend.",
    )
    parser.add_argument("--device", default=None, help="cpu, cuda, or cuda:N")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    frame = pd.read_csv(args.input_csv)
    if args.audio_column not in frame.columns:
        raise KeyError(f"Missing audio column: {args.audio_column}")

    frame = frame.dropna(subset=[args.audio_column]).copy()
    if frame.empty:
        raise ValueError("No audio rows found after dropping missing paths.")

    if args.backend == "transformers":
        output = transcribe_with_transformers(
            frame=frame,
            audio_column=args.audio_column,
            model_name=args.model_name,
            device=args.device,
            batch_size=args.batch_size,
        )
    else:
        raise ValueError(f"Unsupported backend: {args.backend}")

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(f"Wrote ASR transcripts to {output_path}")
    print("Next step: score CER/WER with scripts/compute_text_metrics.py.")


if __name__ == "__main__":
    main()
