from pathlib import Path

import torch
import torchaudio
from speechbrain.inference.classifiers import EncoderClassifier
from speechbrain.inference.interfaces import foreign_class
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification


def _canonicalize_label(label) -> str:
    if isinstance(label, bytes):
        label = label.decode("utf-8")
    return str(label).strip().lower()


class SpeechEmotionRecognizer:
    """
    SER wrapper supporting either:
    - local Hugging Face audio-classification checkpoints such as WavLM fine-tuned on ESD
    - SpeechBrain IEMOCAP SER checkpoints
    """
    def __init__(self, device=None, source=None, savedir=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        default_savedir = Path(savedir or "pretrained_models/ser-iemocap")
        self.savedir = default_savedir
        self.backend = None
        self.feature_extractor = None
        self.sample_rate = 16000
        self.id2label = {}
        self.label2id = {}

        if self._looks_like_hf_audio_classifier(default_savedir):
            self._load_hf_audio_classifier(default_savedir)
            return

        source = str(default_savedir) if default_savedir.exists() else (source or "speechbrain/emotion-recognition-wav2vec2-IEMOCAP")
        savedir = str(default_savedir)
        run_opts = {"device": self.device}

        # This model ships with a custom inference class and does not expose
        # the standard `compute_features` module expected by EncoderClassifier.
        # Try custom interfaces first, then fall back for compatibility.
        self.model = None
        custom_loaders = [
            ("custom_interface.py", "CustomEncoderWav2vec2Classifier"),
            ("custom.py", "CustomInterface"),
        ]
        for pymodule_file, classname in custom_loaders:
            try:
                self.model = foreign_class(
                    source=source,
                    savedir=savedir,
                    pymodule_file=pymodule_file,
                    classname=classname,
                    run_opts=run_opts,
                )
                break
            except Exception:
                continue

        if self.model is None:
            self.model = EncoderClassifier.from_hparams(
                source=source,
                savedir=savedir,
                run_opts=run_opts,
            )
        self.backend = "speechbrain"
        self.id2label = {
            0: "neutral",
            1: "angry",
            2: "happy",
            3: "sad",
        }
        self.label2id = {_canonicalize_label(label): idx for idx, label in self.id2label.items()}

    def _looks_like_hf_audio_classifier(self, model_dir: Path) -> bool:
        return model_dir.exists() and (model_dir / "config.json").exists() and (model_dir / "preprocessor_config.json").exists()

    def _load_hf_audio_classifier(self, model_dir: Path) -> None:
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(str(model_dir))
        self.model = AutoModelForAudioClassification.from_pretrained(str(model_dir)).to(self.device).eval()
        self.backend = "hf_audio_classification"
        self.sample_rate = int(getattr(self.feature_extractor, "sampling_rate", 16000))
        self.id2label = {int(key): value for key, value in self.model.config.id2label.items()}
        self.label2id = {_canonicalize_label(label): idx for idx, label in self.id2label.items()}

    def get_label_index(self, label: str):
        return self.label2id.get(_canonicalize_label(label))

    def _ensure_16k_mono(self, wav, sr):
        # wav: [T] or [1, T] or [C, T]
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)  # mono
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        return wav.squeeze(0)  # [T]

    def _prepare_batch(self, waveforms, lengths, sample_rate):
        x = waveforms
        if x.dim() == 3 and x.size(1) == 1:
            x = x[:, 0, :]
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.to(self.device)

        if sample_rate != self.sample_rate:
            x = torchaudio.functional.resample(x, sample_rate, self.sample_rate)

        clipped = []
        for idx in range(x.shape[0]):
            length = int(lengths[idx].item())
            clipped.append(x[idx, :length].detach().cpu().numpy())
        batch = self.feature_extractor(
            clipped,
            sampling_rate=self.sample_rate,
            padding=True,
            return_tensors="pt",
        )
        return {key: value.to(self.device) for key, value in batch.items()}

    @torch.no_grad()
    def _hf_predict_batch(self, waveforms, lengths, sample_rate=24000):
        batch = self._prepare_batch(waveforms, lengths, sample_rate)
        outputs = self.model(**batch)
        probs = torch.softmax(outputs.logits, dim=-1)
        score, idx = torch.max(probs, dim=-1)
        labels = [self.id2label[int(item)] for item in idx.detach().cpu().tolist()]
        return probs.detach().cpu(), score.detach().cpu(), idx.detach().cpu(), labels

    @torch.no_grad()
    def _hf_encode_batch(self, waveforms, lengths, sample_rate=24000):
        batch = self._prepare_batch(waveforms, lengths, sample_rate)
        outputs = self.model.wavlm(
            batch["input_values"],
            attention_mask=batch.get("attention_mask"),
            output_hidden_states=self.model.config.use_weighted_layer_sum,
            return_dict=True,
        )

        if self.model.config.use_weighted_layer_sum:
            hidden = torch.stack(outputs.hidden_states, dim=1)
            norm_weights = torch.nn.functional.softmax(self.model.layer_weights, dim=-1)
            hidden = (hidden * norm_weights.view(-1, 1, 1)).sum(dim=1)
        else:
            hidden = outputs[0]

        hidden = self.model.projector(hidden)
        attention_mask = batch.get("attention_mask")
        if attention_mask is None:
            pooled = hidden.mean(dim=1)
        else:
            padding_mask = self.model._get_feature_vector_attention_mask(hidden.shape[1], attention_mask)
            hidden = hidden.masked_fill(~padding_mask.unsqueeze(-1), 0.0)
            pooled = hidden.sum(dim=1) / padding_mask.sum(dim=1, keepdim=True).clamp_min(1)
        return pooled.detach()

    @torch.no_grad()
    def predict_file(self, wav_path):
        wav, sr = torchaudio.load(wav_path)
        wav = self._ensure_16k_mono(wav, sr).to(self.device)
        if self.backend == "hf_audio_classification":
            lengths = torch.LongTensor([wav.shape[-1]])
            out_prob, score, idx, label = self._hf_predict_batch(wav.unsqueeze(0), lengths, sample_rate=self.sample_rate)
        else:
            out_prob, score, idx, label = self.model.classify_batch(wav.unsqueeze(0))
        return {
            "label": label[0],
            "class_idx": int(idx[0]),
            "score": float(score[0]),
            "probs": out_prob[0].detach().cpu(),
        }

    @torch.no_grad()
    def predict_batch(self, waveforms, lengths, sample_rate=24000):
        """
        waveforms: [B, T_max] padded
        lengths: [B] absolute lengths (original before pad)
        """
        if self.backend == "hf_audio_classification":
            return self._hf_predict_batch(waveforms, lengths, sample_rate=sample_rate)

        x = waveforms
        if x.dim() == 3 and x.size(1) == 1:
            x = x[:, 0, :]  # [B, T]
        x = x.to(self.device)

        if sample_rate != self.sample_rate:
            x = torchaudio.functional.resample(x, sample_rate, self.sample_rate)

        # SpeechBrain expects relative lengths in [0,1]
        rel_lens = lengths.float().to(self.device) / lengths.max().float().to(self.device)

        out_prob, score, idx, label = self.model.classify_batch(x, rel_lens)
        return out_prob.detach().cpu(), score.detach().cpu(), idx.detach().cpu(), label

    @torch.no_grad()
    def encode_batch(self, waveforms, lengths, sample_rate=24000):
        if self.backend == "hf_audio_classification":
            return self._hf_encode_batch(waveforms, lengths, sample_rate=sample_rate)

        x = waveforms
        if x.dim() == 3 and x.size(1) == 1:
            x = x[:, 0, :]
        x = x.to(self.device)

        if sample_rate != self.sample_rate:
            x = torchaudio.functional.resample(x, sample_rate, self.sample_rate)

        rel_lens = lengths.float().to(self.device) / lengths.max().float().to(self.device)
        return self.model.encode_batch(x, rel_lens).detach()
