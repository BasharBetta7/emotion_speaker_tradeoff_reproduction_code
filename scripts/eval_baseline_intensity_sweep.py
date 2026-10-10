#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from emotion_recongizer import SpeechEmotionRecognizer
from scripts.audio_distance_metrics import compute_mcd

SER_LABEL_TO_CANONICAL = {
    "ang": "angry",
    "hap": "happy",
    "neu": "neutral",
    "sad": "sad",
}


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=None, engine="python")


def ensure_wave_batch(wav: torch.Tensor) -> torch.Tensor:
    if wav.dim() == 1:
        return wav.unsqueeze(0)
    if wav.dim() == 2 and wav.size(0) == 1:
        return wav
    if wav.dim() == 2:
        return wav.mean(dim=0, keepdim=True)
    if wav.dim() == 3 and wav.size(1) == 1:
        return wav[:, 0, :]
    return wav


def scalar_cosine(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.reshape(1, -1)
    y = y.reshape(1, -1)
    return F.cosine_similarity(x, y, dim=-1).item()


def normalize_label(label_value) -> str:
    if isinstance(label_value, (list, tuple)):
        label_value = label_value[0]
    if isinstance(label_value, bytes):
        label_value = label_value.decode("utf-8")
    label = str(label_value).strip().lower()
    return SER_LABEL_TO_CANONICAL.get(label, label)


def resolve_audio_path(raw_path: str, manifest_path: Path) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate

    options = [
        (manifest_path.parent / candidate).resolve(),
        (REPO_ROOT / candidate).resolve(),
        candidate.resolve(),
    ]
    for path in options:
        if path.exists():
            return path
    return options[0]


def load_audio_mono(wav_path: str, sample_rate: int, device: str) -> torch.Tensor:
    wav, sr = torchaudio.load(wav_path)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    return wav.to(device)


def align_waveforms(src: torch.Tensor, trg: torch.Tensor, mode: str = "trim") -> tuple[torch.Tensor, torch.Tensor]:
    src_len = src.shape[-1]
    trg_len = trg.shape[-1]

    if src_len == trg_len:
        return src, trg

    if mode == "trim":
        n = min(src_len, trg_len)
        return src[..., :n], trg[..., :n]

    # pad mode
    n = max(src_len, trg_len)
    if src_len < n:
        src = F.pad(src, (0, n - src_len))
    if trg_len < n:
        trg = F.pad(trg, (0, n - trg_len))
    return src, trg


def infer_intensity_waveform(src_wav: torch.Tensor, baseline_wav: torch.Tensor, alpha: float, peak_norm: bool) -> torch.Tensor:
    out = (1.0 - alpha) * src_wav + alpha * baseline_wav
    return peak_normalize(out, peak_norm=peak_norm)


def peak_normalize(wav: torch.Tensor, peak_norm: bool) -> torch.Tensor:
    if peak_norm:
        peak = float(wav.abs().max().item())
        if peak > 0.99:
            wav = wav * (0.99 / peak)
    return wav


def mix_reference_waveform(src_wav: torch.Tensor, ref_wav: torch.Tensor, alpha: float) -> torch.Tensor:
    return (1.0 - alpha) * src_wav + alpha * ref_wav


def align_feature_frames(arrays: list[np.ndarray], mode: str) -> list[np.ndarray]:
    lengths = [array.shape[0] for array in arrays]
    if len(set(lengths)) == 1:
        return arrays

    if mode == "trim":
        n_frames = min(lengths)
        return [array[:n_frames] for array in arrays]

    n_frames = max(lengths)
    aligned = []
    for array in arrays:
        pad_frames = n_frames - array.shape[0]
        if pad_frames <= 0:
            aligned.append(array)
        elif array.ndim == 1:
            aligned.append(np.pad(array, (0, pad_frames), mode="constant", constant_values=0.0))
        else:
            aligned.append(np.pad(array, ((0, pad_frames), (0, 0)), mode="edge"))
    return aligned


def interpolate_f0(src_f0: np.ndarray, baseline_f0: np.ndarray, alpha: float) -> np.ndarray:
    src_voiced = src_f0 > 0
    baseline_voiced = baseline_f0 > 0
    voiced_score = (1.0 - alpha) * src_voiced.astype(np.float64) + alpha * baseline_voiced.astype(np.float64)
    out_voiced = voiced_score >= 0.5

    src_log = np.zeros_like(src_f0, dtype=np.float64)
    baseline_log = np.zeros_like(baseline_f0, dtype=np.float64)

    both_voiced = src_voiced & baseline_voiced
    src_only = src_voiced & ~baseline_voiced
    baseline_only = ~src_voiced & baseline_voiced

    src_log[both_voiced | src_only] = np.log(src_f0[both_voiced | src_only])
    baseline_log[both_voiced | baseline_only] = np.log(baseline_f0[both_voiced | baseline_only])

    src_log[baseline_only] = baseline_log[baseline_only]
    baseline_log[src_only] = src_log[src_only]

    mixed_log = (1.0 - alpha) * src_log + alpha * baseline_log
    out_f0 = np.zeros_like(src_f0, dtype=np.float64)
    out_f0[out_voiced] = np.exp(mixed_log[out_voiced])
    return out_f0


def infer_intensity_world(
    src_wav: torch.Tensor,
    baseline_wav: torch.Tensor,
    alpha: float,
    sample_rate: int,
    frame_period: float,
    align_mode: str,
    peak_norm: bool,
) -> torch.Tensor:
    try:
        import pyworld
    except ImportError as exc:
        raise ImportError(
            "WORLD-domain inferred intensity requires pyworld. Install it in the evaluation environment "
            "or run with --inferred-mix-domain waveform."
        ) from exc

    src_np = src_wav.detach().cpu().numpy().astype(np.float64)
    baseline_np = baseline_wav.detach().cpu().numpy().astype(np.float64)

    src_f0, src_timeaxis = pyworld.harvest(src_np, sample_rate, frame_period=frame_period, f0_floor=71.0, f0_ceil=800.0)
    baseline_f0, baseline_timeaxis = pyworld.harvest(
        baseline_np,
        sample_rate,
        frame_period=frame_period,
        f0_floor=71.0,
        f0_ceil=800.0,
    )
    src_sp = pyworld.cheaptrick(src_np, src_f0, src_timeaxis, sample_rate)
    baseline_sp = pyworld.cheaptrick(baseline_np, baseline_f0, baseline_timeaxis, sample_rate)
    src_ap = pyworld.d4c(src_np, src_f0, src_timeaxis, sample_rate)
    baseline_ap = pyworld.d4c(baseline_np, baseline_f0, baseline_timeaxis, sample_rate)

    src_f0, baseline_f0, src_sp, baseline_sp, src_ap, baseline_ap = align_feature_frames(
        [src_f0, baseline_f0, src_sp, baseline_sp, src_ap, baseline_ap],
        mode=align_mode,
    )

    out_f0 = interpolate_f0(src_f0, baseline_f0, alpha=alpha)
    out_sp = (1.0 - alpha) * src_sp + alpha * baseline_sp
    out_ap = np.clip((1.0 - alpha) * src_ap + alpha * baseline_ap, 0.0, 1.0)
    out_np = pyworld.synthesize(out_f0, out_sp, out_ap, sample_rate, frame_period).astype(np.float32)

    out = torch.from_numpy(out_np)
    return peak_normalize(out, peak_norm=peak_norm)


def resolve_cyclegan_feature_path(item, generated_path: Path, generated_manifest_path: Path, feature_root: Optional[Path]) -> Path:
    for column in ["generated_feature_npz", "feature_npz", "cyclegan_feature_npz", "generated_features"]:
        if hasattr(item, column):
            value = getattr(item, column)
            if pd.notna(value):
                return resolve_audio_path(str(value), generated_manifest_path)

    if feature_root is not None:
        sample_path = feature_root / f"{item.sample_id}.npz"
        if sample_path.exists():
            return sample_path

        if hasattr(item, "model_variant"):
            variant = str(getattr(item, "model_variant"))
            pair_tag = variant.split("_")[-1]
            pair_path = feature_root / pair_tag / f"{item.sample_id}.npz"
            if pair_path.exists():
                return pair_path

    return generated_path.with_suffix(".npz")


def infer_intensity_cyclegan_features(
    feature_path: Path,
    alpha: float,
    sample_rate: int,
    align_mode: str,
    peak_norm: bool,
) -> torch.Tensor:
    try:
        import pyworld
    except ImportError as exc:
        raise ImportError(
            "CycleGAN-feature inferred intensity requires pyworld. Install it in the evaluation environment."
        ) from exc

    if not feature_path.exists():
        raise FileNotFoundError(
            f"Missing CycleGAN feature file: {feature_path}. "
            "Re-run scripts/cyclegan/convert_cyclegan_spectrum.py with --feature-dir."
        )

    with np.load(feature_path) as data:
        feature_sample_rate = int(data["sample_rate"])
        frame_period = float(data["frame_period"])
        source_f0 = data["source_f0"]
        converted_f0 = data["converted_f0"]
        source_coded_sp = data["source_coded_sp"]
        converted_coded_sp = data["converted_coded_sp"]
        source_ap = data["source_ap"]

    if feature_sample_rate != sample_rate:
        raise ValueError(
            f"Feature sample rate mismatch for {feature_path}: "
            f"features={feature_sample_rate}, evaluator={sample_rate}"
        )

    source_coded_sp_t = source_coded_sp.T
    converted_coded_sp_t = converted_coded_sp.T
    source_f0, converted_f0, source_coded_sp_t, converted_coded_sp_t, source_ap = align_feature_frames(
        [source_f0, converted_f0, source_coded_sp_t, converted_coded_sp_t, source_ap],
        mode=align_mode,
    )

    out_f0 = interpolate_f0(source_f0, converted_f0, alpha=alpha)
    out_coded_sp = (1.0 - alpha) * source_coded_sp_t + alpha * converted_coded_sp_t
    decoded_sp = pyworld.decode_spectral_envelope(
        np.ascontiguousarray(out_coded_sp),
        sample_rate,
        pyworld.get_cheaptrick_fft_size(sample_rate),
    )
    out_np = pyworld.synthesize(out_f0, decoded_sp, source_ap, sample_rate, frame_period).astype(np.float32)
    out = torch.from_numpy(out_np)
    return peak_normalize(out, peak_norm=peak_norm)


def build_metric_models(device: str, speaker_model_dir: Path, emotion_model_dir: Optional[Path]):
    speaker_model = EncoderClassifier.from_hparams(
        source=str(speaker_model_dir),
        savedir=str(speaker_model_dir),
        run_opts={"device": device},
    )

    emotion_model = None
    if emotion_model_dir is not None:
        emotion_model = SpeechEmotionRecognizer(device=device, savedir=str(emotion_model_dir))
    return speaker_model, emotion_model


def compute_speaker_cosine(speaker_model, wav_a: torch.Tensor, wav_b: torch.Tensor) -> float:
    emb_a = speaker_model.encode_batch(ensure_wave_batch(wav_a))
    emb_b = speaker_model.encode_batch(ensure_wave_batch(wav_b))
    return scalar_cosine(emb_a.detach().cpu(), emb_b.detach().cpu())


def compute_emotion_metrics(
    emotion_model,
    src_wav: torch.Tensor,
    ref_wav: torch.Tensor,
    out_wav: torch.Tensor,
    target_emotion: str,
    sample_rate: int,
) -> Dict[str, Optional[float]]:
    if emotion_model is None:
        return {
            "emotion_acc": None,
            "emotion_conf": None,
            "emotion_ref_cosine": None,
            "emotion_src_cosine": None,
            "predicted_emotion": None,
        }

    ref_batch = ensure_wave_batch(ref_wav)
    out_batch = ensure_wave_batch(out_wav)
    src_batch = ensure_wave_batch(src_wav)

    ref_lengths = torch.LongTensor([ref_batch.shape[-1]])
    out_lengths = torch.LongTensor([out_batch.shape[-1]])
    src_lengths = torch.LongTensor([src_batch.shape[-1]])

    out_prob, _, _, out_label = emotion_model.predict_batch(out_batch, out_lengths, sample_rate=sample_rate)
    ref_emb = emotion_model.encode_batch(ref_batch, ref_lengths, sample_rate=sample_rate)
    out_emb = emotion_model.encode_batch(out_batch, out_lengths, sample_rate=sample_rate)
    src_emb = emotion_model.encode_batch(src_batch, src_lengths, sample_rate=sample_rate)

    predicted_label = normalize_label(out_label[0])
    target_label = target_emotion.strip().lower()
    target_index = emotion_model.get_label_index(target_label)

    target_conf = None
    if target_index is not None:
        target_conf = float(out_prob[0][target_index].item())

    return {
        "emotion_acc": float(predicted_label == target_label) if target_index is not None else None,
        "emotion_conf": target_conf,
        "emotion_ref_cosine": scalar_cosine(ref_emb.detach().cpu(), out_emb.detach().cpu()),
        "emotion_src_cosine": scalar_cosine(src_emb.detach().cpu(), out_emb.detach().cpu()),
        "predicted_emotion": predicted_label,
    }


def aggregate_results(results_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    metric_columns = [
        col
        for col in [
            "speaker_cosine",
            "speaker_ref_cosine",
            "speaker_ref_output_cosine",
            "speaker_gen_ref_cosine",
            "speaker_gen_src_cosine",
            "speaker_src_ref_cosine",
            "mcd_gen_ref",
            "mcd_gen_src",
            "mcd_src_ref",
            "emotion_acc",
            "emotion_conf",
            "emotion_ref_cosine",
            "emotion_src_cosine",
        ]
        if col in results_df.columns
    ]
    aggregations = {metric: "mean" for metric in metric_columns}

    group_prefix = ["model_variant", "alpha"]
    if "intensity_mode" in results_df.columns:
        group_prefix = ["model_variant", "intensity_mode", "alpha"]

    overall = results_df.groupby(group_prefix, dropna=False).agg(aggregations).reset_index()
    by_speaker = (
        results_df.groupby([*group_prefix, "speaker_id"], dropna=False).agg(aggregations).reset_index()
    )
    by_emotion = (
        results_df.groupby([*group_prefix, "target_emotion"], dropna=False)
        .agg(aggregations)
        .reset_index()
    )
    return {
        "overall": overall,
        "by_speaker": by_speaker,
        "by_emotion": by_emotion,
    }


def pick_generated_column(frame: pd.DataFrame) -> str:
    candidates = ["generated_wav", "generated_wav_path", "audio_path", "wav_path"]
    for col in candidates:
        if col in frame.columns:
            return col
    raise KeyError(f"generated manifest must include one of: {candidates}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Infer intensity sweep for fixed-output baselines by interpolating source and baseline-generated audio, "
            "then evaluate each alpha with the same benchmark metric stack."
        )
    )
    parser.add_argument("--generated-manifest", required=True)
    parser.add_argument("--benchmark-manifest", default="Data/esd_balanced_benchmark_manifest.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-variant", default=None)
    parser.add_argument(
        "--intensity-mode",
        choices=["native", "inferred", "fixed", "unknown"],
        default="inferred",
        help="How the evaluated outputs expose intensity control.",
    )
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    parser.add_argument("--device", default=None)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--align-mode", choices=["trim", "pad"], default="trim")
    parser.add_argument(
        "--inferred-mix-domain",
        choices=["waveform", "world", "cyclegan_features"],
        default="cyclegan_features",
        help=(
            "waveform mixes source and baseline wavs directly; world re-extracts WORLD features from both wavs; "
            "cyclegan_features interpolates source and CycleGAN-converted model features saved during conversion."
        ),
    )
    parser.add_argument(
        "--cyclegan-feature-root",
        default=None,
        help=(
            "Root directory for CycleGAN feature npz files when --inferred-mix-domain cyclegan_features. "
            "Can point directly at a pair directory containing <sample_id>.npz."
        ),
    )
    parser.add_argument("--world-frame-period", type=float, default=5.0)
    parser.add_argument("--no-peak-norm", action="store_true")
    parser.add_argument("--save-audio", action="store_true")
    parser.add_argument("--speaker-model-dir", default="pretrained_models/spkrec-ecapa-voxceleb")
    parser.add_argument("--emotion-model-dir", default="pretrained_models/wavlm_ser_esd")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_root = output_dir / "audio"
    if args.save_audio:
        audio_root.mkdir(parents=True, exist_ok=True)

    generated_manifest_path = Path(args.generated_manifest)
    benchmark_manifest_path = Path(args.benchmark_manifest)

    generated_df = read_table(generated_manifest_path)
    benchmark_df = read_table(benchmark_manifest_path)

    if "sample_id" not in generated_df.columns:
        raise KeyError("generated manifest must include sample_id")
    if "sample_id" not in benchmark_df.columns:
        raise KeyError("benchmark manifest must include sample_id")

    gen_col = pick_generated_column(generated_df)
    generated_df = generated_df.rename(columns={gen_col: "generated_wav"})

    merged = benchmark_df.merge(generated_df, on="sample_id", how="inner", suffixes=("", "_gen"))
    if args.limit is not None:
        merged = merged.head(args.limit).copy()

    if merged.empty:
        raise ValueError("No rows after joining benchmark and generated manifests")

    base_variant = args.model_variant
    if base_variant is None:
        if "model_variant" in merged.columns and merged["model_variant"].notna().any():
            base_variant = str(merged.loc[merged["model_variant"].notna(), "model_variant"].iloc[0])
        else:
            base_variant = generated_manifest_path.stem
    base_variant = f"{base_variant}_intensity_inferred"

    if "intensity_mode" not in merged.columns:
        merged["intensity_mode"] = args.intensity_mode
    else:
        merged["intensity_mode"] = merged["intensity_mode"].fillna(args.intensity_mode)

    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    speaker_model_dir = Path(args.speaker_model_dir).resolve()
    emotion_model_dir = Path(args.emotion_model_dir).resolve() if args.emotion_model_dir else None
    speaker_model, emotion_model = build_metric_models(device, speaker_model_dir, emotion_model_dir)

    rows = []
    peak_norm = not args.no_peak_norm
    cyclegan_feature_root = Path(args.cyclegan_feature_root) if args.cyclegan_feature_root else None

    for item in tqdm(merged.itertuples(index=False), desc="Baseline intensity sweep", total=len(merged)):
        src_wav = load_audio_mono(item.source_wav, args.sample_rate, device).squeeze(0).cpu()
        ref_wav = load_audio_mono(item.reference_wav, args.sample_rate, device).squeeze(0).cpu()

        generated_path = resolve_audio_path(str(item.generated_wav), generated_manifest_path)
        if not generated_path.exists():
            raise FileNotFoundError(f"Missing generated wav for sample_id={item.sample_id}: {generated_path}")

        base_wav = load_audio_mono(str(generated_path), args.sample_rate, device).squeeze(0).cpu()
        src_aligned, base_aligned = align_waveforms(src_wav, base_wav, mode=args.align_mode)
        src_ref_aligned, ref_aligned = align_waveforms(src_wav, ref_wav, mode=args.align_mode)

        for alpha in args.alphas:
            alpha = float(alpha)
            feature_path_for_row = None
            if args.inferred_mix_domain == "world":
                out_wav = infer_intensity_world(
                    src_wav,
                    base_wav,
                    alpha=alpha,
                    sample_rate=args.sample_rate,
                    frame_period=args.world_frame_period,
                    align_mode=args.align_mode,
                    peak_norm=peak_norm,
                )
            elif args.inferred_mix_domain == "cyclegan_features":
                feature_path = resolve_cyclegan_feature_path(
                    item,
                    generated_path=generated_path,
                    generated_manifest_path=generated_manifest_path,
                    feature_root=cyclegan_feature_root,
                )
                feature_path_for_row = feature_path
                out_wav = infer_intensity_cyclegan_features(
                    feature_path=feature_path,
                    alpha=alpha,
                    sample_rate=args.sample_rate,
                    align_mode=args.align_mode,
                    peak_norm=peak_norm,
                )
            else:
                out_wav = infer_intensity_waveform(src_aligned, base_aligned, alpha=alpha, peak_norm=peak_norm)
            ref_mix_wav = mix_reference_waveform(src_ref_aligned, ref_aligned, alpha=alpha)

            generated_alpha_path = None
            if args.save_audio:
                alpha_dir = audio_root / f"alpha_{alpha:.2f}"
                alpha_dir.mkdir(parents=True, exist_ok=True)
                generated_alpha_path = alpha_dir / f"{item.sample_id}_a{alpha:.2f}.wav"
                torchaudio.save(str(generated_alpha_path), out_wav.unsqueeze(0), args.sample_rate)

            speaker_gen_src_cosine = compute_speaker_cosine(speaker_model, out_wav, src_aligned)
            speaker_ref_cosine = compute_speaker_cosine(speaker_model, src_ref_aligned, ref_mix_wav)
            speaker_src_ref_cosine = compute_speaker_cosine(speaker_model, src_ref_aligned, ref_aligned)
            ref_out_aligned, out_ref_aligned = align_waveforms(ref_wav, out_wav, mode=args.align_mode)
            speaker_gen_ref_cosine = compute_speaker_cosine(speaker_model, out_ref_aligned, ref_out_aligned)
            mcd_gen_src = compute_mcd(out_wav, src_wav, sample_rate=args.sample_rate)
            mcd_gen_ref = compute_mcd(out_wav, ref_wav, sample_rate=args.sample_rate)
            mcd_src_ref = compute_mcd(src_wav, ref_wav, sample_rate=args.sample_rate)
            emotion_metrics = compute_emotion_metrics(
                emotion_model,
                src_wav=src_aligned,
                ref_wav=ref_wav,
                out_wav=out_wav,
                target_emotion=item.target_emotion,
                sample_rate=args.sample_rate,
            )

            row = {
                "sample_id": item.sample_id,
                "speaker_id": item.speaker_id,
                "text": item.text,
                "text_rank": item.text_rank,
                "source_file_id": item.source_file_id,
                "reference_file_id": item.reference_file_id,
                "source_wav": item.source_wav,
                "reference_wav": item.reference_wav,
                "target_emotion": item.target_emotion,
                "model_variant": base_variant,
                "intensity_mode": item.intensity_mode,
                "inferred_mix_domain": args.inferred_mix_domain,
                "alpha": alpha,
                "speaker_gen_src_cosine": speaker_gen_src_cosine,
                "speaker_gen_ref_cosine": speaker_gen_ref_cosine,
                "speaker_src_ref_cosine": speaker_src_ref_cosine,
                "mcd_gen_src": mcd_gen_src,
                "mcd_gen_ref": mcd_gen_ref,
                "mcd_src_ref": mcd_src_ref,
                "speaker_cosine": speaker_gen_src_cosine,
                "speaker_ref_cosine": speaker_ref_cosine,
                "speaker_ref_output_cosine": speaker_gen_ref_cosine,
                "generated_wav": str(generated_alpha_path) if generated_alpha_path is not None else str(generated_path),
                "baseline_generated_wav": str(generated_path),
                "generated_feature_npz": str(feature_path_for_row) if feature_path_for_row is not None else None,
            }
            row.update(emotion_metrics)
            rows.append(row)

    results_df = pd.DataFrame(rows)
    results_df.to_csv(output_dir / "per_sample_metrics.csv", index=False)

    aggregates = aggregate_results(results_df)
    for name, frame in aggregates.items():
        frame.to_csv(output_dir / f"{name}_metrics.csv", index=False)

    print(f"Wrote {len(results_df)} per-sample rows to {output_dir / 'per_sample_metrics.csv'}")
    for name, frame in aggregates.items():
        print(f"Wrote {len(frame)} rows to {output_dir / f'{name}_metrics.csv'}")


if __name__ == "__main__":
    main()
