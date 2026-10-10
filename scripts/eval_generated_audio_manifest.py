#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, Optional

import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier
from tqdm import tqdm

# Allow importing project-local modules when invoked as scripts/eval_generated_audio_manifest.py.
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

DEFAULT_SPEAKER_MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"


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


def load_audio_mono(wav_path: str, sample_rate: int, device: str) -> torch.Tensor:
    wav, sr = torchaudio.load(wav_path)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    return wav.to(device)


def build_metric_models(device: str, speaker_model_dir: Path, emotion_model_dir: Optional[Path]):
    speaker_source = str(speaker_model_dir)
    speaker_savedir = speaker_model_dir
    speaker_hparams = speaker_model_dir / "hyperparams.yaml"
    local_speaker_files = [
        speaker_hparams,
        speaker_model_dir / "embedding_model.ckpt",
        speaker_model_dir / "mean_var_norm_emb.ckpt",
        speaker_model_dir / "classifier.ckpt",
        speaker_model_dir / "label_encoder.txt",
    ]
    if speaker_model_dir.exists():
        if all(path.is_file() for path in local_speaker_files):
            speaker_savedir = speaker_model_dir
        else:
            speaker_savedir = speaker_model_dir.parent / f"{speaker_model_dir.name}_speechbrain_cache"
            speaker_source = DEFAULT_SPEAKER_MODEL_SOURCE
    speaker_model = EncoderClassifier.from_hparams(
        source=speaker_source,
        savedir=str(speaker_savedir),
        run_opts={"device": device},
    )

    emotion_model = None
    if emotion_model_dir is not None:
        emotion_model = SpeechEmotionRecognizer(device=device, savedir=str(emotion_model_dir))
    return speaker_model, emotion_model


def compute_speaker_cosine(speaker_model, src_wav: torch.Tensor, out_wav: torch.Tensor) -> float:
    src_emb = speaker_model.encode_batch(ensure_wave_batch(src_wav))
    out_emb = speaker_model.encode_batch(ensure_wave_batch(out_wav))
    return scalar_cosine(src_emb.detach().cpu(), out_emb.detach().cpu())


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


def resolve_generated_path(raw_path: str, manifest_path: Path) -> Path:
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
    # Return the first deterministic option for error reporting.
    return options[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate pre-generated external baseline audio against the balanced benchmark using the current metric stack."
    )
    parser.add_argument("--generated-manifest", required=True, help="CSV/TSV with sample_id and generated wav path.")
    parser.add_argument("--benchmark-manifest", default="Data/esd_balanced_benchmark_manifest.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-variant", default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument(
        "--intensity-mode",
        choices=["native", "inferred", "fixed", "unknown"],
        default="fixed",
        help="How the evaluated outputs expose intensity control.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--strict", action="store_true", help="Fail on missing generated wavs instead of skipping.")
    parser.add_argument("--speaker-model-dir", default="pretrained_models/spkrec-ecapa-voxceleb")
    parser.add_argument("--emotion-model-dir", default="pretrained_models/wavlm_ser_esd")
    args = parser.parse_args()

    generated_manifest_path = Path(args.generated_manifest)
    benchmark_manifest_path = Path(args.benchmark_manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_df = read_table(generated_manifest_path)
    benchmark_df = read_table(benchmark_manifest_path)

    if "sample_id" not in generated_df.columns:
        raise KeyError("generated manifest must include sample_id")
    if "sample_id" not in benchmark_df.columns:
        raise KeyError("benchmark manifest must include sample_id")

    gen_col = pick_generated_column(generated_df)
    generated_df = generated_df.rename(columns={gen_col: "generated_wav"})

    # Allow absolute and repo-relative generated wav paths.
    generated_df["generated_wav"] = generated_df["generated_wav"].astype(str)

    merged = benchmark_df.merge(generated_df, on="sample_id", how="inner", suffixes=("", "_gen"))
    if args.limit is not None:
        merged = merged.head(args.limit).copy()

    if merged.empty:
        raise ValueError("No rows after joining benchmark manifest and generated manifest on sample_id")

    model_variant_default = args.model_variant or generated_manifest_path.stem
    if "model_variant" not in merged.columns:
        merged["model_variant"] = model_variant_default
    else:
        merged["model_variant"] = merged["model_variant"].fillna(model_variant_default)

    if "alpha" not in merged.columns:
        merged["alpha"] = args.alpha
    elif args.alpha is not None:
        merged["alpha"] = merged["alpha"].fillna(args.alpha)

    if "intensity_mode" not in merged.columns:
        merged["intensity_mode"] = args.intensity_mode
    else:
        merged["intensity_mode"] = merged["intensity_mode"].fillna(args.intensity_mode)

    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    speaker_model_dir = Path(args.speaker_model_dir).resolve()
    emotion_model_dir = Path(args.emotion_model_dir).resolve() if args.emotion_model_dir else None

    speaker_model, emotion_model = build_metric_models(device, speaker_model_dir, emotion_model_dir)

    rows = []
    skipped = 0
    for item in tqdm(merged.itertuples(index=False), desc="Evaluating generated audio", total=len(merged)):
        generated_path = resolve_generated_path(item.generated_wav, generated_manifest_path)

        if not generated_path.exists():
            msg = f"Missing generated wav for sample_id={item.sample_id}: {generated_path}"
            if args.strict:
                raise FileNotFoundError(msg)
            print(f"[skip] {msg}")
            skipped += 1
            continue

        src_wav = load_audio_mono(item.source_wav, args.sample_rate, device).squeeze(0).cpu()
        ref_wav = load_audio_mono(item.reference_wav, args.sample_rate, device).squeeze(0).cpu()
        out_wav = load_audio_mono(str(generated_path), args.sample_rate, device).squeeze(0).cpu()

        speaker_gen_src_cosine = compute_speaker_cosine(speaker_model, out_wav, src_wav)
        speaker_gen_ref_cosine = compute_speaker_cosine(speaker_model, out_wav, ref_wav)
        speaker_src_ref_cosine = compute_speaker_cosine(speaker_model, src_wav, ref_wav)
        mcd_gen_src = compute_mcd(out_wav, src_wav, sample_rate=args.sample_rate)
        mcd_gen_ref = compute_mcd(out_wav, ref_wav, sample_rate=args.sample_rate)
        mcd_src_ref = compute_mcd(src_wav, ref_wav, sample_rate=args.sample_rate)
        emotion_metrics = compute_emotion_metrics(
            emotion_model,
            src_wav=src_wav,
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
            "model_variant": item.model_variant,
            "intensity_mode": item.intensity_mode,
            "alpha": item.alpha,
            "speaker_gen_src_cosine": speaker_gen_src_cosine,
            "speaker_gen_ref_cosine": speaker_gen_ref_cosine,
            "speaker_src_ref_cosine": speaker_src_ref_cosine,
            "mcd_gen_src": mcd_gen_src,
            "mcd_gen_ref": mcd_gen_ref,
            "mcd_src_ref": mcd_src_ref,
            "speaker_cosine": speaker_gen_src_cosine,
            "speaker_ref_output_cosine": speaker_gen_ref_cosine,
            "generated_wav": str(generated_path),
        }
        row.update(emotion_metrics)
        rows.append(row)

    results_df = pd.DataFrame(rows)
    if results_df.empty:
        raise ValueError("All rows were skipped; no metrics were computed")

    results_df.to_csv(output_dir / "per_sample_metrics.csv", index=False)
    aggregates = aggregate_results(results_df)
    for name, frame in aggregates.items():
        frame.to_csv(output_dir / f"{name}_metrics.csv", index=False)

    print(f"Wrote {len(results_df)} per-sample rows to {output_dir / 'per_sample_metrics.csv'}")
    for name, frame in aggregates.items():
        print(f"Wrote {len(frame)} rows to {output_dir / f'{name}_metrics.csv'}")
    if skipped:
        print(f"Skipped missing generated files: {skipped}")


if __name__ == "__main__":
    main()
