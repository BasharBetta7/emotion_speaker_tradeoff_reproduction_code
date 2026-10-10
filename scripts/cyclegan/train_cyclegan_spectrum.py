#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
import types
from tqdm import tqdm

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
        coded_sps_normalization_fit_transoform,
        load_wavs,
        logf0_statistics,
        pitch_conversion,
        sample_train_data,
        transpose_in_list,
        wav_padding,
        world_decode_spectral_envelop,
        world_decompose,
        world_encode_data,
        world_encode_spectral_envelop,
        world_speech_synthesis,
    )

    return {
        "CycleGAN": CycleGAN,
        "coded_sps_normalization_fit_transoform": coded_sps_normalization_fit_transoform,
        "load_wavs": load_wavs,
        "logf0_statistics": logf0_statistics,
        "pitch_conversion": pitch_conversion,
        "sample_train_data": sample_train_data,
        "transpose_in_list": transpose_in_list,
        "wav_padding": wav_padding,
        "world_decode_spectral_envelop": world_decode_spectral_envelop,
        "world_decompose": world_decompose,
        "world_encode_data": world_encode_data,
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


def load_wavs_limited(wav_dir: str, sample_rate: int, max_files: int) -> list[np.ndarray]:
    wav_paths = sorted(Path(wav_dir).glob("*.wav"))
    if max_files > 0:
        wav_paths = wav_paths[:max_files]
    wavs = []
    for wav_path in wav_paths:
        wav, _ = librosa.load(str(wav_path), sr=sample_rate, mono=True)
        wavs.append(wav)
    return wavs


def run_validation(
    funcs: dict,
    model,
    validation_dir: str | None,
    output_dir: Path,
    direction: str,
    sample_rate: int,
    frame_period: float,
    num_mcep: int,
    log_f0s_mean_a: float,
    log_f0s_std_a: float,
    log_f0s_mean_b: float,
    log_f0s_std_b: float,
    coded_sps_a_mean: np.ndarray,
    coded_sps_a_std: np.ndarray,
    coded_sps_b_mean: np.ndarray,
    coded_sps_b_std: np.ndarray,
) -> None:
    if validation_dir is None:
        return

    validation_path = Path(validation_dir)
    if not validation_path.exists():
        raise FileNotFoundError(f"Missing validation directory: {validation_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    for file_name in sorted(os.listdir(str(validation_path))):
        wav, _ = librosa.load(str(validation_path / file_name), sr=sample_rate, mono=True)
        wav = funcs["wav_padding"](wav=wav, sr=sample_rate, frame_period=frame_period, multiple=4)
        f0, _, sp, ap = funcs["world_decompose"](wav=wav, fs=sample_rate, frame_period=frame_period)
        coded_sp = funcs["world_encode_spectral_envelop"](sp=sp, fs=sample_rate, dim=num_mcep)
        coded_sp_t = coded_sp.T

        if direction == "A2B":
            f0_converted = funcs["pitch_conversion"](
                f0=f0,
                mean_log_src=log_f0s_mean_a,
                std_log_src=log_f0s_std_a,
                mean_log_target=log_f0s_mean_b,
                std_log_target=log_f0s_std_b,
            )
            coded_sp_norm = (coded_sp_t - coded_sps_a_mean) / coded_sps_a_std
            coded_sp_converted_norm = model.test(inputs=np.array([coded_sp_norm]), direction="A2B")[0]
            coded_sp_converted = coded_sp_converted_norm * coded_sps_b_std + coded_sps_b_mean
        else:
            f0_converted = funcs["pitch_conversion"](
                f0=f0,
                mean_log_src=log_f0s_mean_b,
                std_log_src=log_f0s_std_b,
                mean_log_target=log_f0s_mean_a,
                std_log_target=log_f0s_std_a,
            )
            coded_sp_norm = (coded_sp_t - coded_sps_b_mean) / coded_sps_b_std
            coded_sp_converted_norm = model.test(inputs=np.array([coded_sp_norm]), direction="B2A")[0]
            coded_sp_converted = coded_sp_converted_norm * coded_sps_a_std + coded_sps_a_mean

        decoded_sp = funcs["world_decode_spectral_envelop"](
            coded_sp=np.ascontiguousarray(coded_sp_converted.T),
            fs=sample_rate,
        )
        wav_out = funcs["world_speech_synthesis"](
            f0=f0_converted,
            decoded_sp=decoded_sp,
            ap=ap,
            fs=sample_rate,
            frame_period=frame_period,
        )
        save_wav(output_dir / file_name, wav_out, sample_rate)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train spectrum-only CycleGAN using an external CycleGAN code clone.")
    parser.add_argument("--code-dir", required=True)
    parser.add_argument("--train-A-dir", required=True)
    parser.add_argument("--train-B-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--validation-A-dir", default=None)
    parser.add_argument("--validation-B-dir", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tensorboard-log-dir", required=True)
    parser.add_argument("--num-epochs", type=int, default=1000)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--max-files-per-domain", type=int, default=0)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--num-mcep", type=int, default=24)
    parser.add_argument("--frame-period", type=float, default=5.0)
    parser.add_argument("--n-frames", type=int, default=128)
    parser.add_argument("--validation-every", type=int, default=10)
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()

    funcs = import_external_code(Path(args.code_dir))
    np.random.seed(args.random_seed)

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    log_dir = Path(args.tensorboard_log_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    generator_lr = 0.0002
    generator_lr_decay = generator_lr / 200000
    discriminator_lr = 0.0001
    discriminator_lr_decay = discriminator_lr / 200000
    lambda_cycle = 10
    lambda_identity = 5
    mini_batch_size = 1

    print("Preprocessing CycleGAN spectrum data...")
    start = time.time()
    if args.max_files_per_domain > 0:
        wavs_a = load_wavs_limited(args.train_A_dir, args.sample_rate, args.max_files_per_domain)
        wavs_b = load_wavs_limited(args.train_B_dir, args.sample_rate, args.max_files_per_domain)
    else:
        wavs_a = funcs["load_wavs"](wav_dir=args.train_A_dir, sr=args.sample_rate)
        wavs_b = funcs["load_wavs"](wav_dir=args.train_B_dir, sr=args.sample_rate)
    f0s_a, _, _, _, coded_sps_a = funcs["world_encode_data"](
        wavs=wavs_a,
        fs=args.sample_rate,
        frame_period=args.frame_period,
        coded_dim=args.num_mcep,
    )
    f0s_b, _, _, _, coded_sps_b = funcs["world_encode_data"](
        wavs=wavs_b,
        fs=args.sample_rate,
        frame_period=args.frame_period,
        coded_dim=args.num_mcep,
    )

    log_f0s_mean_a, log_f0s_std_a = funcs["logf0_statistics"](f0s_a)
    log_f0s_mean_b, log_f0s_std_b = funcs["logf0_statistics"](f0s_b)
    coded_sps_a_t = funcs["transpose_in_list"](lst=coded_sps_a)
    coded_sps_b_t = funcs["transpose_in_list"](lst=coded_sps_b)
    coded_sps_a_norm, coded_sps_a_mean, coded_sps_a_std = funcs["coded_sps_normalization_fit_transoform"](
        coded_sps=coded_sps_a_t
    )
    coded_sps_b_norm, coded_sps_b_mean, coded_sps_b_std = funcs["coded_sps_normalization_fit_transoform"](
        coded_sps=coded_sps_b_t
    )

    np.savez(
        model_dir / "logf0s_normalization.npz",
        mean_A=log_f0s_mean_a,
        std_A=log_f0s_std_a,
        mean_B=log_f0s_mean_b,
        std_B=log_f0s_std_b,
    )
    np.savez(
        model_dir / "mcep_normalization.npz",
        mean_A=coded_sps_a_mean,
        std_A=coded_sps_a_std,
        mean_B=coded_sps_b_mean,
        std_B=coded_sps_b_std,
    )
    print(f"Preprocessing done in {time.time() - start:.1f}s")

    model = funcs["CycleGAN"](num_features=args.num_mcep, log_dir=str(log_dir))
    num_iterations = 0

    for epoch in range(args.num_epochs):
        print(f"Epoch: {epoch}")
        epoch_start = time.time()
        dataset_a, dataset_b = funcs["sample_train_data"](
            dataset_A=coded_sps_a_norm,
            dataset_B=coded_sps_b_norm,
            n_frames=args.n_frames,
        )
        if args.max_samples > 0:
            dataset_a = dataset_a[: args.max_samples]
            dataset_b = dataset_b[: args.max_samples]

        n_samples = dataset_a.shape[0]
        for i in range(n_samples // mini_batch_size):
            num_iterations += 1
            if num_iterations > 10000:
                lambda_identity = 0
            if num_iterations > 200000:
                generator_lr = max(0, generator_lr - generator_lr_decay)
                discriminator_lr = max(0, discriminator_lr - discriminator_lr_decay)

            start_i = i * mini_batch_size
            end_i = (i + 1) * mini_batch_size
            generator_loss, discriminator_loss = model.train(
                input_A=dataset_a[start_i:end_i],
                input_B=dataset_b[start_i:end_i],
                lambda_cycle=lambda_cycle,
                lambda_identity=lambda_identity,
                generator_learning_rate=generator_lr,
                discriminator_learning_rate=discriminator_lr,
            )
            if i % 50 == 0:
                print(
                    "Iteration: {:07d}, Generator LR: {:.7f}, Discriminator LR: {:.7f}, "
                    "Generator Loss: {:.3f}, Discriminator Loss: {:.3f}".format(
                        num_iterations,
                        generator_lr,
                        discriminator_lr,
                        generator_loss,
                        discriminator_loss,
                    )
                )

        model.save(directory=str(model_dir), filename=args.model_name)
        print(f"Epoch elapsed: {time.time() - epoch_start:.1f}s")

        if generator_lr <= 0:
            print("Learning rate reached zero; stopping.")
            break

        if not args.skip_validation and args.validation_every > 0 and epoch % args.validation_every == 0:
            run_validation(
                funcs,
                model,
                args.validation_A_dir,
                output_dir / "converted_A",
                "A2B",
                args.sample_rate,
                args.frame_period,
                args.num_mcep,
                log_f0s_mean_a,
                log_f0s_std_a,
                log_f0s_mean_b,
                log_f0s_std_b,
                coded_sps_a_mean,
                coded_sps_a_std,
                coded_sps_b_mean,
                coded_sps_b_std,
            )
            run_validation(
                funcs,
                model,
                args.validation_B_dir,
                output_dir / "converted_B",
                "B2A",
                args.sample_rate,
                args.frame_period,
                args.num_mcep,
                log_f0s_mean_a,
                log_f0s_std_a,
                log_f0s_mean_b,
                log_f0s_std_b,
                coded_sps_a_mean,
                coded_sps_a_std,
                coded_sps_b_mean,
                coded_sps_b_std,
            )


if __name__ == "__main__":
    main()
