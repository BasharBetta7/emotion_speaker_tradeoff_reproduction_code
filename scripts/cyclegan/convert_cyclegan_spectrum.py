#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import types

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import librosa
import numpy as np


def import_external_code(code_dir: Path):
    if not code_dir.exists():
        raise FileNotFoundError(f"Missing CycleGAN code directory: {code_dir}")
    sys.path.insert(0, str(code_dir.resolve()))

    try:
        import tensorflow.compat.v1 as tf  # type: ignore

        tf.disable_v2_behavior()

        if not hasattr(tf, "contrib"):
            def instance_norm(inputs, epsilon=1e-6, activation_fn=None):
                axes = list(range(1, len(inputs.shape) - 1))
                mean, variance = tf.nn.moments(inputs, axes=axes, keepdims=True)
                outputs = (inputs - mean) / tf.sqrt(variance + epsilon)
                return activation_fn(outputs) if activation_fn is not None else outputs

            tf.contrib = types.SimpleNamespace(  # type: ignore[attr-defined]
                layers=types.SimpleNamespace(instance_norm=instance_norm)
            )

        sys.modules["tensorflow"] = tf
    except Exception:
        pass

    if "pycwt" not in sys.modules:
        sys.modules["pycwt"] = types.SimpleNamespace()
    if "pywt" not in sys.modules:
        sys.modules["pywt"] = types.SimpleNamespace()

    from model import CycleGAN  # type: ignore
    from preprocess import (  # type: ignore
        pitch_conversion,
        wav_padding,
        world_decode_spectral_envelop,
        world_decompose,
        world_encode_spectral_envelop,
        world_speech_synthesis,
    )

    return {
        "CycleGAN": CycleGAN,
        "pitch_conversion": pitch_conversion,
        "wav_padding": wav_padding,
        "world_decode_spectral_envelop": world_decode_spectral_envelop,
        "world_decompose": world_decompose,
        "world_encode_spectral_envelop": world_encode_spectral_envelop,
        "world_speech_synthesis": world_speech_synthesis,
    }


def save_wav(path: Path, wav: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import soundfile as sf

        sf.write(str(path), wav, sample_rate)
    except ImportError:
        librosa.output.write_wav(str(path), wav, sample_rate)


def save_feature_npz(
    path: Path,
    sample_rate: int,
    frame_period: float,
    source_f0: np.ndarray,
    converted_f0: np.ndarray,
    source_coded_sp: np.ndarray,
    converted_coded_sp: np.ndarray,
    source_ap: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        sample_rate=np.array(sample_rate),
        frame_period=np.array(frame_period),
        source_f0=source_f0,
        converted_f0=converted_f0,
        source_coded_sp=source_coded_sp,
        converted_coded_sp=converted_coded_sp,
        source_ap=source_ap,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert wavs with a spectrum-only CycleGAN checkpoint.")
    parser.add_argument("--code-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--conversion-direction", choices=["A2B", "B2A"], default="A2B")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--feature-dir", default=None, help="Optional directory for per-sample CycleGAN feature npz files.")
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--num-mcep", type=int, default=24)
    parser.add_argument("--frame-period", type=float, default=5.0)
    args = parser.parse_args()

    funcs = import_external_code(Path(args.code_dir))
    model_dir = Path(args.model_dir)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    feature_dir = Path(args.feature_dir) if args.feature_dir else None
    output_dir.mkdir(parents=True, exist_ok=True)
    if feature_dir is not None:
        feature_dir.mkdir(parents=True, exist_ok=True)

    model = funcs["CycleGAN"](num_features=args.num_mcep, mode="test")
    model.load(filepath=str(model_dir / args.model_name))

    mcep_params = np.load(model_dir / "mcep_normalization.npz")
    mcep_mean_a = mcep_params["mean_A"]
    mcep_std_a = mcep_params["std_A"]
    mcep_mean_b = mcep_params["mean_B"]
    mcep_std_b = mcep_params["std_B"]

    logf0_params = np.load(model_dir / "logf0s_normalization.npz")
    logf0_mean_a = logf0_params["mean_A"]
    logf0_std_a = logf0_params["std_A"]
    logf0_mean_b = logf0_params["mean_B"]
    logf0_std_b = logf0_params["std_B"]

    for file_name in sorted(os.listdir(str(data_dir))):
        wav, _ = librosa.load(str(data_dir / file_name), sr=args.sample_rate, mono=True)
        wav = funcs["wav_padding"](wav=wav, sr=args.sample_rate, frame_period=args.frame_period, multiple=4)
        f0, _, sp, ap = funcs["world_decompose"](wav=wav, fs=args.sample_rate, frame_period=args.frame_period)
        coded_sp = funcs["world_encode_spectral_envelop"](sp=sp, fs=args.sample_rate, dim=args.num_mcep)
        coded_sp_t = coded_sp.T

        if args.conversion_direction == "A2B":
            f0_converted = funcs["pitch_conversion"](
                f0=f0,
                mean_log_src=logf0_mean_a,
                std_log_src=logf0_std_a,
                mean_log_target=logf0_mean_b,
                std_log_target=logf0_std_b,
            )
            coded_sp_norm = (coded_sp_t - mcep_mean_a) / mcep_std_a
            coded_sp_converted_norm = model.test(inputs=np.array([coded_sp_norm]), direction="A2B")[0]
            coded_sp_converted = coded_sp_converted_norm * mcep_std_b + mcep_mean_b
        else:
            f0_converted = funcs["pitch_conversion"](
                f0=f0,
                mean_log_src=logf0_mean_b,
                std_log_src=logf0_std_b,
                mean_log_target=logf0_mean_a,
                std_log_target=logf0_std_a,
            )
            coded_sp_norm = (coded_sp_t - mcep_mean_b) / mcep_std_b
            coded_sp_converted_norm = model.test(inputs=np.array([coded_sp_norm]), direction="B2A")[0]
            coded_sp_converted = coded_sp_converted_norm * mcep_std_a + mcep_mean_a

        decoded_sp = funcs["world_decode_spectral_envelop"](
            coded_sp=np.ascontiguousarray(coded_sp_converted.T),
            fs=args.sample_rate,
        )
        wav_out = funcs["world_speech_synthesis"](
            f0=f0_converted,
            decoded_sp=decoded_sp,
            ap=ap,
            fs=args.sample_rate,
            frame_period=args.frame_period,
        )
        save_wav(output_dir / file_name, wav_out, args.sample_rate)
        if feature_dir is not None:
            save_feature_npz(
                feature_dir / f"{Path(file_name).stem}.npz",
                sample_rate=args.sample_rate,
                frame_period=args.frame_period,
                source_f0=f0,
                converted_f0=f0_converted,
                source_coded_sp=coded_sp_t,
                converted_coded_sp=coded_sp_converted,
                source_ap=ap,
            )


if __name__ == "__main__":
    main()
