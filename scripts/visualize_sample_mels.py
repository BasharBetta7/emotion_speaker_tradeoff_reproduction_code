#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchaudio
import yaml

from eval_mel_viz import MEL_PARAMS


def load_config(config_path: Path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_wave(wav_path: str, sample_rate: int) -> torch.Tensor:
    wav, sr = torchaudio.load(wav_path)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    return wav


def to_log_mel(wav: torch.Tensor, to_mel) -> torch.Tensor:
    mel = to_mel(wav).squeeze(0)
    return torch.log(1e-5 + mel)


def find_sample_row(metrics_csv: Path, sample_id: str):
    df = pd.read_csv(metrics_csv)
    matches = df[df["sample_id"] == sample_id]
    if matches.empty:
        raise ValueError(f"sample_id={sample_id!r} not found in {metrics_csv}")
    return matches.iloc[0]


def resolve_generated_wav(row, generated_wav: str | None) -> str:
    if generated_wav:
        return generated_wav
    value = row.get("generated_wav")
    if isinstance(value, str) and value.strip():
        return value
    raise ValueError(
        "No generated waveform path is available for this sample. "
        "Use a metrics CSV produced with --save-audio or pass --generated-wav explicitly."
    )


def plot_triplet(source_mel, generated_mel, reference_mel, sample_id: str, target_emotion: str, output_path: Path):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False, sharey=True)
    panels = [
        ("Source", source_mel),
        ("Generated", generated_mel),
        ("Reference", reference_mel),
    ]

    for ax, (title, mel) in zip(axes, panels):
        image = ax.imshow(mel.numpy(), origin="lower", aspect="auto")
        ax.set_title(f"{title} Mel")
        ax.set_ylabel("Mel bins")
        fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)

    axes[-1].set_xlabel("Frames")
    fig.suptitle(f"Mel Spectrogram Comparison\n{sample_id} | target={target_emotion}", y=0.98)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize source/generated/reference mel spectrograms for one sample.")
    parser.add_argument("--metrics-csv", required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--config-path", default="Configs/config_emo_balanced_holdout.yml")
    parser.add_argument("--generated-wav", default=None)
    parser.add_argument("--output-path", default=None)
    args = parser.parse_args()

    metrics_csv = Path(args.metrics_csv)
    config_path = Path(args.config_path)
    row = find_sample_row(metrics_csv, args.sample_id)
    generated_wav = resolve_generated_wav(row, args.generated_wav)

    config = load_config(config_path)
    sample_rate = int(config.get("preprocess_params", {}).get("sr", 24000))
    to_mel = torchaudio.transforms.MelSpectrogram(**MEL_PARAMS)

    source_wav = load_wave(row["source_wav"], sample_rate)
    reference_wav = load_wave(row["reference_wav"], sample_rate)
    generated_wav_tensor = load_wave(generated_wav, sample_rate)

    source_mel = to_log_mel(source_wav, to_mel)
    generated_mel = to_log_mel(generated_wav_tensor, to_mel)
    reference_mel = to_log_mel(reference_wav, to_mel)

    output_path = Path(args.output_path) if args.output_path else metrics_csv.parent / f"{args.sample_id}_mel_triplet.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_triplet(source_mel, generated_mel, reference_mel, args.sample_id, row["target_emotion"], output_path)
    print(f"Saved mel comparison to {output_path}")


if __name__ == "__main__":
    main()
