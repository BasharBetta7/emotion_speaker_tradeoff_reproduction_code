#!/usr/bin/env python3
# coding: utf-8

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf


MEL_PARAMS = {
    "n_mels": 80,
    "n_fft": 2048,
    "win_length": 1200,
    "hop_length": 300,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute dataset log-mel mean/std from a StarGANv2-VC train list."
    )
    parser.add_argument(
        "--list-path",
        type=str,
        required=True,
        help="Path to list file in format: /abs/or/rel/path.wav|label",
    )
    parser.add_argument("--sr", type=int, default=24000, help="Expected sample rate.")
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Optional limit for quick estimation. 0 means use all files.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=500,
        help="Print progress every N processed files.",
    )
    return parser.parse_args()


def read_list(list_path):
    entries = []
    with open(list_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            wav_path = line.split("|", 1)[0]
            entries.append(wav_path)
    return entries


def load_wave(path):
    import torch

    wave, _ = sf.read(path)
    if wave.ndim > 1:
        wave = np.mean(wave, axis=1)
    return torch.from_numpy(wave).float()


def main():
    args = parse_args()
    import torch
    import torchaudio
    wav_paths = read_list(args.list_path)
    if args.max_files > 0:
        wav_paths = wav_paths[:args.max_files]

    if not wav_paths:
        raise ValueError("No wav paths found in list file.")

    to_melspec = torchaudio.transforms.MelSpectrogram(
        sample_rate=args.sr, **MEL_PARAMS
    )

    total_sum = 0.0
    total_sq_sum = 0.0
    total_count = 0

    for idx, wav_path in enumerate(wav_paths, start=1):
        path_obj = Path(wav_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Missing file: {wav_path}")

        wave_tensor = load_wave(path_obj)
        mel_tensor = to_melspec(wave_tensor)
        log_mel = torch.log(1e-5 + mel_tensor)

        total_sum += log_mel.sum().item()
        total_sq_sum += (log_mel ** 2).sum().item()
        total_count += log_mel.numel()

        if args.print_every > 0 and (idx % args.print_every) == 0:
            print(f"Processed {idx}/{len(wav_paths)} files...")

    mean = total_sum / total_count
    variance = (total_sq_sum / total_count) - (mean ** 2)
    std = max(variance, 0.0) ** 0.5

    print("\nComputed dataset log-mel stats:")
    print(f"mean: {mean:.6f}")
    print(f"std:  {std:.6f}")
    print("\nUse in Configs/config_*.yml:")
    print("dataset_params:")
    print(f"  mean: {mean:.6f}")
    print(f"  std: {std:.6f}")


if __name__ == "__main__":
    main()
