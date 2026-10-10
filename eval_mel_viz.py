import os
from typing import Optional, Tuple

import matplotlib.pyplot as plt
from munch import Munch
import torch
import torchaudio
import yaml

from models import EmotionRankHead, Generator, MappingNetwork, StyleEncoder
from Utils.JDC.model import JDCNet


MEL_PARAMS = {
    "n_mels": 80,
    "n_fft": 2048,
    "win_length": 1200,
    "hop_length": 300,
}


def _resolve_path(path: str, base_dir: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base_dir, path))


def _build_stargan_ema(model_params: dict, include_rank_head: bool = False):
    args = Munch(model_params)
    model = Munch(
        generator=Generator(
            args.dim_in,
            args.style_dim,
            args.max_conv_dim,
            w_hpf=args.w_hpf,
            F0_channel=args.F0_channel,
        ),
        mapping_network=MappingNetwork(
            args.latent_dim,
            args.style_dim,
            args.num_domains,
            hidden_dim=args.max_conv_dim,
        ),
        style_encoder=StyleEncoder(
            args.dim_in,
            args.style_dim,
            args.num_domains,
            args.max_conv_dim,
        ),
    )
    if include_rank_head:
        rank_hidden_dim = int(model_params.get("rank_hidden_dim", 128))
        model.rank_head = EmotionRankHead(args.style_dim, args.num_domains, rank_hidden_dim)
    return model


def _preprocess_wave(
    wav_path: str,
    sample_rate: int,
    mean: float,
    std: float,
    to_mel: torchaudio.transforms.MelSpectrogram,
) -> torch.Tensor:
    wav, sr = torchaudio.load(wav_path)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    mel = to_mel(wav).squeeze(0)
    mel = (torch.log(1e-5 + mel) - mean) / std
    return mel


def load_eval_bundle(
    config_path: str,
    checkpoint_path: str,
    device: Optional[str] = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    base_dir = os.path.dirname(os.path.abspath(config_path))
    ckpt_path = checkpoint_path# _resolve_path(checkpoint_path, base_dir)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_state = state["model_ema"] if "model_ema" in state else state["model"]
    include_rank_head = bool(cfg["model_params"].get("rank_enable", cfg["model_params"].get("use_rank_intensity", False)))
    include_rank_head = include_rank_head or ("rank_head" in model_state)
    model = _build_stargan_ema(cfg["model_params"], include_rank_head=include_rank_head)
    for key in model:
        model[key].load_state_dict(model_state[key], strict=True)
        model[key] = model[key].to(device).eval()

    f0_model = JDCNet(num_class=1, seq_len=192)
    f0_path = cfg["F0_path"] # _resolve_path(cfg["F0_path"], base_dir)
    f0_state = torch.load(f0_path, map_location="cpu", weights_only=False)["net"]
    f0_model.load_state_dict(f0_state, strict=True)
    f0_model = f0_model.to(device).eval()

    mean = float(cfg.get("dataset_params", {}).get("mean", -4.0))
    std = float(cfg.get("dataset_params", {}).get("std", 4.0))
    sr = int(cfg.get("preprocess_params", {}).get("sr", 24000))
    to_mel = torchaudio.transforms.MelSpectrogram(**MEL_PARAMS).to(device)

    return Munch(
        model=model,
        f0_model=f0_model,
        to_mel=to_mel,
        mean=mean,
        std=std,
        sr=sr,
        num_domains=int(cfg["model_params"]["num_domains"]),
        device=device,
    )


@torch.no_grad()
def run_generator_eval(
    bundle,
    source_wav_path: str,
    target_label: int,
    ref_wav_path: Optional[str] = None,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if target_label < 0 or target_label >= bundle.num_domains:
        raise ValueError(
            f"target_label={target_label} is out of range [0, {bundle.num_domains - 1}]"
        )

    source_mel = _preprocess_wave(
        source_wav_path, bundle.sr, bundle.mean, bundle.std, bundle.to_mel
    )
    x_src = source_mel.unsqueeze(0).unsqueeze(0).to(bundle.device)
    y_trg = torch.LongTensor([target_label]).to(bundle.device)

    if ref_wav_path:
        ref_mel = _preprocess_wave(
            ref_wav_path, bundle.sr, bundle.mean, bundle.std, bundle.to_mel
        )
        x_ref = ref_mel.unsqueeze(0).unsqueeze(0).to(bundle.device)
        s_trg = bundle.model.style_encoder(x_ref, y_trg)
    else:
        torch.manual_seed(seed)
        latent_dim = bundle.model.mapping_network.shared[0].in_features
        z = torch.randn((1, latent_dim), device=bundle.device)
        s_trg = bundle.model.mapping_network(z, y_trg)

    f0_feat = bundle.f0_model.get_feature_GAN(x_src)
    x_out = bundle.model.generator(x_src, s_trg, masks=None, F0=f0_feat)
    return x_src.squeeze(0).squeeze(0).cpu(), x_out.squeeze(0).squeeze(0).cpu()


def plot_input_output_mels(
    source_mel: torch.Tensor,
    output_mel: torch.Tensor,
    mean: float = -4.0,
    std: float = 4.0,
    denormalize: bool = True,
    figsize: Tuple[int, int] = (14, 4),
):
    src = source_mel.detach().float()
    out = output_mel.detach().float()
    if denormalize:
        src = src * std + mean
        out = out * std + mean

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    axes[0].imshow(src.numpy(), origin="lower", aspect="auto")
    axes[0].set_title("Input Mel")
    axes[0].set_xlabel("Frames")
    axes[0].set_ylabel("Mel bins")

    axes[1].imshow(out.numpy(), origin="lower", aspect="auto")
    axes[1].set_title("Generator Output Mel")
    axes[1].set_xlabel("Frames")

    plt.tight_layout()
    plt.show()
    return fig
