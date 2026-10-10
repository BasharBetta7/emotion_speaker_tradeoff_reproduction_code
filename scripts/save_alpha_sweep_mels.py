#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchaudio

from eval_mel_viz import load_eval_bundle


EMOTION_TO_LABEL = {
    "Angry": 0,
    "Happy": 1,
    "Neutral": 2,
    "Sad": 3,
    "Surprise": 4,
}


def load_audio_mono(wav_path: str, sample_rate: int, device: str) -> torch.Tensor:
    wav, sr = torchaudio.load(wav_path)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    return wav.to(device)


def waveform_to_mel(wav: torch.Tensor, to_mel, mean: float, std: float) -> torch.Tensor:
    mel = to_mel(wav).squeeze(0)
    return (torch.log(1e-5 + mel) - mean) / std


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save generated mel spectrograms for one alpha sweep sample.")
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    parser.add_argument("--device", default=None)
    parser.add_argument("--random-seed", type=int, default=20260525)
    return parser.parse_args()


def load_sample(manifest_path: Path, sample_id: str | None, random_seed: int) -> pd.Series:
    manifest = pd.read_csv(manifest_path, sep="\t")
    if sample_id:
        matches = manifest[manifest["sample_id"] == sample_id]
        if matches.empty:
            raise ValueError(f"sample_id={sample_id!r} not found in {manifest_path}")
        return matches.iloc[0]

    ssst = manifest[manifest.get("pair_type", "") == "same_speaker_same_text"]
    candidates = ssst if not ssst.empty else manifest
    return candidates.sample(n=1, random_state=random_seed).iloc[0]


@torch.no_grad()
def run_alpha_conversion(bundle, row: pd.Series, alpha: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_src_wav = load_audio_mono(row["source_wav"], bundle.sr, bundle.device)
    x_ref_wav = load_audio_mono(row["reference_wav"], bundle.sr, bundle.device)

    src_mel = waveform_to_mel(x_src_wav, bundle.to_mel, bundle.mean, bundle.std)
    ref_mel = waveform_to_mel(x_ref_wav, bundle.to_mel, bundle.mean, bundle.std)

    x_src = src_mel.unsqueeze(0).unsqueeze(0).to(bundle.device)
    x_ref = ref_mel.unsqueeze(0).unsqueeze(0).to(bundle.device)

    y_src = torch.LongTensor([EMOTION_TO_LABEL["Neutral"]]).to(bundle.device)
    y_trg = torch.LongTensor([EMOTION_TO_LABEL[row["target_emotion"]]]).to(bundle.device)

    s_src = bundle.model.style_encoder(x_src, y_src)
    s_trg = bundle.model.style_encoder(x_ref, y_trg)
    s_mix = (1.0 - alpha) * s_src + alpha * s_trg

    f0_feat = bundle.f0_model.get_feature_GAN(x_src)
    mel_out = bundle.model.generator(x_src, s_mix, masks=None, F0=f0_feat)
    return src_mel.cpu(), ref_mel.cpu(), mel_out.squeeze(0).cpu()


def denormalize_mel(mel: torch.Tensor, mean: float, std: float) -> np.ndarray:
    return (mel.detach().float().squeeze() * std + mean).numpy()


def save_single_mel(mel: np.ndarray, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    image = ax.imshow(mel, origin="lower", aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Frames")
    ax.set_ylabel("Mel bins")
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_grid(mels: list[tuple[str, np.ndarray]], output_path: Path) -> None:
    fig, axes = plt.subplots(len(mels), 1, figsize=(14, 2.8 * len(mels)), sharex=False, sharey=True)
    if len(mels) == 1:
        axes = [axes]
    for ax, (title, mel) in zip(axes, mels):
        image = ax.imshow(mel, origin="lower", aspect="auto")
        ax.set_title(title)
        ax.set_ylabel("Mel bins")
        fig.colorbar(image, ax=ax, fraction=0.018, pad=0.01)
    axes[-1].set_xlabel("Frames")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    row = load_sample(Path(args.manifest_path), args.sample_id, args.random_seed)
    bundle = load_eval_bundle(args.config_path, args.checkpoint_path, device=args.device)

    metadata = {
        "sample_id": row["sample_id"],
        "speaker_id": row["speaker_id"],
        "text": row["text"],
        "source_wav": row["source_wav"],
        "reference_wav": row["reference_wav"],
        "target_emotion": row["target_emotion"],
        "reference_emotion": row.get("reference_emotion", row["target_emotion"]),
        "pair_type": row.get("pair_type", ""),
        "alphas": " ".join(f"{alpha:.1f}" for alpha in args.alphas),
        "config_path": args.config_path,
        "checkpoint_path": args.checkpoint_path,
    }
    pd.DataFrame([metadata]).to_csv(output_dir / "selected_sample_metadata.csv", index=False)

    grid_mels: list[tuple[str, np.ndarray]] = []
    saved_rows: list[dict[str, str | float]] = []
    source_saved = False

    for alpha in args.alphas:
        source_mel, reference_mel, generated_mel = run_alpha_conversion(bundle, row, alpha)
        source_arr = denormalize_mel(source_mel, bundle.mean, bundle.std)
        reference_arr = denormalize_mel(reference_mel, bundle.mean, bundle.std)
        generated_arr = denormalize_mel(generated_mel, bundle.mean, bundle.std)

        if not source_saved:
            np.save(output_dir / "source_neutral_mel.npy", source_arr)
            np.save(output_dir / "reference_emotion_mel.npy", reference_arr)
            save_single_mel(source_arr, "Source Neutral Mel", output_dir / "source_neutral_mel.png")
            save_single_mel(reference_arr, f"Reference {row['target_emotion']} Mel", output_dir / "reference_emotion_mel.png")
            source_saved = True

        alpha_label = f"{alpha:.1f}"
        stem = f"generated_alpha_{alpha_label.replace('.', '_')}"
        npy_path = output_dir / f"{stem}.npy"
        png_path = output_dir / f"{stem}.png"
        np.save(npy_path, generated_arr)
        save_single_mel(generated_arr, f"Generated Mel alpha={alpha_label}", png_path)
        grid_mels.append((f"alpha={alpha_label}", generated_arr))
        saved_rows.append({"alpha": alpha, "mel_npy": str(npy_path), "mel_png": str(png_path)})

    pd.DataFrame(saved_rows).to_csv(output_dir / "generated_mel_paths.csv", index=False)
    save_grid(grid_mels, output_dir / "generated_alpha_sweep_mels.png")
    print(f"Selected sample: {row['sample_id']} -> {row['target_emotion']}")
    print(f"Saved mel spectrograms to: {output_dir}")


if __name__ == "__main__":
    main()
