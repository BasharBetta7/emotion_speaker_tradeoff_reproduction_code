from __future__ import annotations

import math

import librosa
import numpy as np
import torch


def _to_mono_numpy(wav: torch.Tensor) -> np.ndarray:
    array = wav.detach().cpu().float().numpy()
    if array.ndim == 2:
        array = array.mean(axis=0)
    elif array.ndim > 2:
        array = array.reshape(-1)
    return np.ascontiguousarray(array, dtype=np.float32)


def compute_mcd(
    wav_a: torch.Tensor,
    wav_b: torch.Tensor,
    sample_rate: int,
    n_mfcc: int = 25,
    hop_length: int = 256,
    n_fft: int = 1024,
) -> float:
    """Compute DTW-aligned Mel-Cepstral Distortion in dB.

    The 0th cepstral coefficient is excluded, matching the common MCD convention.
    Lower values indicate closer spectral envelopes.
    """
    audio_a = _to_mono_numpy(wav_a)
    audio_b = _to_mono_numpy(wav_b)
    if audio_a.size == 0 or audio_b.size == 0:
        return float("nan")

    mfcc_a = librosa.feature.mfcc(
        y=audio_a,
        sr=sample_rate,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
    )[1:]
    mfcc_b = librosa.feature.mfcc(
        y=audio_b,
        sr=sample_rate,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
    )[1:]
    if mfcc_a.shape[1] == 0 or mfcc_b.shape[1] == 0:
        return float("nan")

    _, path = librosa.sequence.dtw(X=mfcc_a, Y=mfcc_b, metric="euclidean")
    if path.size == 0:
        return float("nan")

    diffs = mfcc_a[:, path[:, 0]] - mfcc_b[:, path[:, 1]]
    frame_distances = np.sqrt(np.sum(diffs * diffs, axis=0))
    scale = 10.0 * math.sqrt(2.0) / math.log(10.0)
    return float(scale * np.mean(frame_distances))
