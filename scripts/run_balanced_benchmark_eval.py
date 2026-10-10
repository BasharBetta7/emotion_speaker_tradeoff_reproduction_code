#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Dict, Optional
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from matplotlib.pylab import kaiser
import pandas as pd
import scipy
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier

from emotion_recongizer import SpeechEmotionRecognizer
from eval_mel_viz import MEL_PARAMS, load_eval_bundle
from scripts.audio_distance_metrics import compute_mcd


EMOTION_TO_LABEL = {
    "Angry": 0,
    "Happy": 1,
    "Neutral": 2,
    "Sad": 3,
    "Surprise": 4,
}


SER_LABEL_TO_CANONICAL = {
    "ang": "angry",
    "hap": "happy",
    "neu": "neutral",
    "sad": "sad",
}

DEFAULT_SPEAKER_MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"


def resolve_path(path: str, base_dir: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (base_dir / candidate).resolve()


def load_audio_mono(wav_path: str, sample_rate: int, device: str) -> torch.Tensor:
    wav, sr = torchaudio.load(wav_path)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    return wav.to(device)


def waveform_to_mel(wav: torch.Tensor, to_mel, mean: float, std: float) -> torch.Tensor:
    mel = to_mel(wav).squeeze(0)
    mel = (torch.log(1e-5 + mel) - mean) / std
    return mel


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


def align_waveforms(src: torch.Tensor, trg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    src_len = src.shape[-1]
    trg_len = trg.shape[-1]
    if src_len == trg_len:
        return src, trg
    n = min(src_len, trg_len)
    return src[..., :n], trg[..., :n]


def mix_reference_waveform(src_wav: torch.Tensor, ref_wav: torch.Tensor, alpha: float) -> torch.Tensor:
    return (1.0 - alpha) * src_wav + alpha * ref_wav


def normalize_label(label_value) -> str:
    if isinstance(label_value, (list, tuple)):
        label_value = label_value[0]
    if isinstance(label_value, bytes):
        label_value = label_value.decode("utf-8")
    label = str(label_value).strip().lower()
    return SER_LABEL_TO_CANONICAL.get(label, label)


def load_vocoder(vocoder_path: Path, device: str):
    try:
        from parallel_wavegan.utils import load_model as load_vocoder_model
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "parallel_wavegan is required to synthesize waveforms for evaluation. "
            "Install the project dependencies in your evaluation environment, e.g. "
            "`pip install -r requirements.txt` or `pip install parallel_wavegan==0.6.1`."
        ) from exc

    import scipy.signal
    from scipy.signal.windows import kaiser
    scipy.signal.kaiser = kaiser
    vocoder = load_vocoder_model(str(vocoder_path)).to(device).eval()
    if hasattr(vocoder, "remove_weight_norm"):
        vocoder.remove_weight_norm()
    return vocoder


def synthesize_waveform(vocoder, mel_out: torch.Tensor) -> torch.Tensor:
    c = mel_out.transpose(-1, -2).squeeze(0).to(next(vocoder.parameters()).device)
    wav = vocoder.inference(c).view(-1)
    return wav.detach().cpu()


def select_rank_calibrated_beta(bundle, s_src: torch.Tensor, s_trg: torch.Tensor, target_label: int, target_intensity: float, search_steps: int):
    if not hasattr(bundle.model, "rank_head"):
        raise ValueError("Rank-calibrated intensity requires a checkpoint/config with rank_head.")

    y_idx = torch.LongTensor([target_label]).to(s_src.device)
    r_src = bundle.model.rank_head(s_src)[0, y_idx.item()]
    r_trg = bundle.model.rank_head(s_trg)[0, y_idx.item()]
    rank_gap = r_trg - r_src
    if torch.abs(rank_gap).item() < 1e-6:
        return {
            "beta": float(target_intensity),
            "rank_intensity": None,
            "rank_src": float(r_src.item()),
            "rank_ref": float(r_trg.item()),
            "rank_gap": float(rank_gap.item()),
        }

    betas = torch.linspace(0.0, 1.0, steps=max(search_steps, 2), device=s_src.device, dtype=s_src.dtype).view(-1, 1)
    s_candidates = torch.lerp(s_src.expand(betas.size(0), -1), s_trg.expand(betas.size(0), -1), betas)
    scores = bundle.model.rank_head(s_candidates)[:, target_label]
    intensities = (scores - r_src) / rank_gap
    target = torch.tensor(float(target_intensity), device=s_src.device, dtype=intensities.dtype)
    best_idx = torch.argmin(torch.abs(intensities - target)).item()
    return {
        "beta": float(betas[best_idx].item()),
        "rank_intensity": float(intensities[best_idx].item()),
        "rank_src": float(r_src.item()),
        "rank_ref": float(r_trg.item()),
        "rank_gap": float(rank_gap.item()),
    }


@torch.no_grad()
def run_reference_conversion(
    bundle,
    source_wav_path: str,
    reference_wav_path: str,
    target_label: int,
    alpha: float,
    intensity_control: str = "style",
    rank_search_steps: int = 101,
):
    x_src_wav = load_audio_mono(source_wav_path, bundle.sr, bundle.device)
    x_ref_wav = load_audio_mono(reference_wav_path, bundle.sr, bundle.device)

    src_mel = waveform_to_mel(x_src_wav, bundle.to_mel, bundle.mean, bundle.std)
    ref_mel = waveform_to_mel(x_ref_wav, bundle.to_mel, bundle.mean, bundle.std)

    x_src = src_mel.unsqueeze(0).unsqueeze(0).to(bundle.device)
    x_ref = ref_mel.unsqueeze(0).unsqueeze(0).to(bundle.device)

    y_src = torch.LongTensor([EMOTION_TO_LABEL["Neutral"]]).to(bundle.device)
    y_trg = torch.LongTensor([target_label]).to(bundle.device)

    s_src = bundle.model.style_encoder(x_src, y_src)
    s_trg = bundle.model.style_encoder(x_ref, y_trg)
    rank_info = {
        "beta": float(alpha),
        "rank_intensity": None,
        "rank_src": None,
        "rank_ref": None,
        "rank_gap": None,
    }
    if intensity_control == "rank":
        rank_info = select_rank_calibrated_beta(
            bundle=bundle,
            s_src=s_src,
            s_trg=s_trg,
            target_label=target_label,
            target_intensity=alpha,
            search_steps=rank_search_steps,
        )
    beta = rank_info["beta"]
    s_mix = (1.0 - beta) * s_src + beta * s_trg

    f0_feat = bundle.f0_model.get_feature_GAN(x_src)
    mel_out = bundle.model.generator(x_src, s_mix, masks=None, F0=f0_feat)
    return x_src_wav.squeeze(0).cpu(), x_ref_wav.squeeze(0).cpu(), mel_out.squeeze(0).cpu(), rank_info


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


def compute_speaker_cosine(speaker_model, wav_a: torch.Tensor, wav_b: torch.Tensor) -> float:
    emb_a = speaker_model.encode_batch(ensure_wave_batch(wav_a))
    emb_b = speaker_model.encode_batch(ensure_wave_batch(wav_b))
    return scalar_cosine(emb_a.detach().cpu(), emb_b.detach().cpu())


def compute_emotion_metrics(emotion_model, src_wav:torch.Tensor, ref_wav: torch.Tensor, out_wav: torch.Tensor, target_emotion: str) -> Dict[str, Optional[float]]:
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

    out_prob, out_score, out_idx, out_label = emotion_model.predict_batch(out_batch, out_lengths, sample_rate=24000)
    ref_emb = emotion_model.encode_batch(ref_batch, ref_lengths, sample_rate=24000)
    out_emb = emotion_model.encode_batch(out_batch, out_lengths, sample_rate=24000)
    src_emb = emotion_model.encode_batch(src_batch, src_lengths, sample_rate=24000)

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
    if "intensity_control" in results_df.columns:
        group_prefix = [*group_prefix[:-1], "intensity_control", group_prefix[-1]]

    overall = results_df.groupby(group_prefix, dropna=False).agg(aggregations).reset_index()
    by_speaker = results_df.groupby([*group_prefix, "speaker_id"], dropna=False).agg(aggregations).reset_index()
    by_emotion = results_df.groupby([*group_prefix, "target_emotion"], dropna=False).agg(aggregations).reset_index()
    return {
        "overall": overall,
        "by_speaker": by_speaker,
        "by_emotion": by_emotion,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the balanced held-out benchmark for emotion conversion.")
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--manifest-path", default="Data/esd_balanced_benchmark_manifest.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-variant", default="baseline")
    parser.add_argument(
        "--intensity-mode",
        choices=["native", "inferred", "fixed", "unknown"],
        default="native",
        help="How the evaluated outputs expose intensity control.",
    )
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument(
        "--intensity-control",
        choices=["style", "rank"],
        default="style",
        help="style uses alpha directly; rank searches for a style beta whose rank-head intensity matches alpha.",
    )
    parser.add_argument("--rank-search-steps", type=int, default=101)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save-audio", action="store_true")
    parser.add_argument("--speaker-model-dir", default="pretrained_models/spkrec-ecapa-voxceleb")
    parser.add_argument("--emotion-model-dir", default="pretrained_models/wavlm_ser_esd")
    parser.add_argument("--vocoder-path", default="Vocoder/checkpoint-400000steps.pkl")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    audio_dir = output_dir / "audio"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_audio:
        audio_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_eval_bundle(args.config_path, args.checkpoint_path, device=args.device)
    repo_root = Path(os.path.abspath(args.config_path)).resolve().parent.parent
    manifest_path = resolve_path(args.manifest_path, repo_root)
    vocoder_path = resolve_path(args.vocoder_path, repo_root)
    speaker_model_dir = resolve_path(args.speaker_model_dir, repo_root)
    emotion_model_dir = resolve_path(args.emotion_model_dir, repo_root)

    manifest_df = pd.read_csv(manifest_path, sep="\t")
    if args.limit is not None:
        manifest_df = manifest_df.head(args.limit).copy()

    vocoder = load_vocoder(vocoder_path, bundle.device)
    speaker_model, emotion_model = build_metric_models(bundle.device, speaker_model_dir, emotion_model_dir)

    rows = []
    for item in tqdm(manifest_df.itertuples(index=False), desc='Evaluating on benchmark', total=len(manifest_df)):
        target_label = EMOTION_TO_LABEL[item.target_emotion]
        src_wav, ref_wav, mel_out, rank_info = run_reference_conversion(
            bundle=bundle,
            source_wav_path=item.source_wav,
            reference_wav_path=item.reference_wav,
            target_label=target_label,
            alpha=args.alpha,
            intensity_control=args.intensity_control,
            rank_search_steps=args.rank_search_steps,
        )
        out_wav = synthesize_waveform(vocoder, mel_out)

        speaker_gen_src_cosine = compute_speaker_cosine(speaker_model, out_wav, src_wav)
        src_ref_aligned, ref_aligned = align_waveforms(src_wav, ref_wav)
        ref_mix_wav = mix_reference_waveform(src_ref_aligned, ref_aligned, alpha=args.alpha)
        speaker_ref_cosine = compute_speaker_cosine(speaker_model, src_ref_aligned, ref_mix_wav)
        speaker_src_ref_cosine = compute_speaker_cosine(speaker_model, src_ref_aligned, ref_aligned)
        ref_out_aligned, out_ref_aligned = align_waveforms(ref_wav, out_wav)
        speaker_gen_ref_cosine = compute_speaker_cosine(speaker_model, out_ref_aligned, ref_out_aligned)
        mcd_gen_src = compute_mcd(out_wav, src_wav, sample_rate=bundle.sr)
        mcd_gen_ref = compute_mcd(out_wav, ref_wav, sample_rate=bundle.sr)
        mcd_src_ref = compute_mcd(src_wav, ref_wav, sample_rate=bundle.sr)
        emotion_metrics = compute_emotion_metrics(emotion_model, src_wav,  ref_wav, out_wav, item.target_emotion)

        generated_wav_path = None
        if args.save_audio:
            generated_wav_path = audio_dir / f"{item.sample_id}_a{args.alpha:.2f}_b{rank_info['beta']:.2f}.wav"
            sf.write(generated_wav_path, out_wav.numpy(), bundle.sr)

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
            "model_variant": args.model_variant,
            "intensity_mode": args.intensity_mode,
            "alpha": args.alpha,
            "intensity_control": args.intensity_control,
            "style_beta": rank_info["beta"],
            "rank_intensity": rank_info["rank_intensity"],
            "rank_src": rank_info["rank_src"],
            "rank_ref": rank_info["rank_ref"],
            "rank_gap": rank_info["rank_gap"],
            "speaker_gen_src_cosine": speaker_gen_src_cosine,
            "speaker_gen_ref_cosine": speaker_gen_ref_cosine,
            "speaker_src_ref_cosine": speaker_src_ref_cosine,
            "mcd_gen_src": mcd_gen_src,
            "mcd_gen_ref": mcd_gen_ref,
            "mcd_src_ref": mcd_src_ref,
            "speaker_cosine": speaker_gen_src_cosine,
            "speaker_ref_cosine": speaker_ref_cosine,
            "speaker_ref_output_cosine": speaker_gen_ref_cosine,
            "generated_wav": str(generated_wav_path) if generated_wav_path is not None else None,
        }
        if hasattr(item, "reference_text"):
            row["reference_text"] = item.reference_text
        if hasattr(item, "reference_text_rank"):
            row["reference_text_rank"] = item.reference_text_rank
        if hasattr(item, "pair_type"):
            row["pair_type"] = item.pair_type
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
