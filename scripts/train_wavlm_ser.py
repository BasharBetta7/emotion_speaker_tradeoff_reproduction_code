#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torchaudio
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)


EMOTION_TO_ID = {
    "Angry": 0,
    "Happy": 1,
    "Neutral": 2,
    "Sad": 3,
    "Surprise": 4,
}
ID_TO_EMOTION = {idx: label for label, idx in EMOTION_TO_ID.items()}


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_metadata(metadata_path: Path) -> pd.DataFrame:
    df = pd.read_csv(metadata_path, sep="\t")
    required = {"wav_path", "emotion", "speaker_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required metadata columns: {sorted(missing)}")
    df = df[df["emotion"].isin(EMOTION_TO_ID)].copy()
    df["label"] = df["emotion"].map(EMOTION_TO_ID)
    return df.reset_index(drop=True)


def split_train_val(df: pd.DataFrame, seed: int, val_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_idx, val_idx = next(splitter.split(df, y=df["label"], groups=df["speaker_id"]))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    return train_df, val_df


class ESDSERDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        sample_rate: int,
        max_seconds: float,
        training: bool,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * max_seconds) if max_seconds > 0 else None
        self.training = training

    def __len__(self) -> int:
        return len(self.frame)

    def _crop(self, wav: torch.Tensor) -> torch.Tensor:
        if self.max_samples is None or wav.shape[-1] <= self.max_samples:
            return wav
        if self.training:
            start = torch.randint(0, wav.shape[-1] - self.max_samples + 1, (1,)).item()
        else:
            start = (wav.shape[-1] - self.max_samples) // 2
        return wav[..., start : start + self.max_samples]

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self.frame.iloc[index]
        wav, sr = torchaudio.load(row["wav_path"])
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        wav = self._crop(wav).squeeze(0)
        return {
            "input_values": wav,
            "label": int(row["label"]),
            "emotion": row["emotion"],
            "speaker_id": int(row["speaker_id"]),
            "wav_path": row["wav_path"],
        }


@dataclass
class AudioClassificationCollator:
    feature_extractor: AutoFeatureExtractor
    sample_rate: int

    def __call__(self, features: List[Dict[str, object]]) -> Dict[str, torch.Tensor]:
        waves = [item["input_values"].numpy() for item in features]
        labels = torch.tensor([int(item["label"]) for item in features], dtype=torch.long)
        batch = self.feature_extractor(
            waves,
            sampling_rate=self.sample_rate,
            padding=True,
            return_tensors="pt",
        )
        batch["labels"] = labels
        return batch


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "weighted_f1": f1_score(labels, preds, average="weighted"),
    }
    return metrics


def save_confusion_artifacts(trainer: Trainer, val_dataset: ESDSERDataset, output_dir: Path) -> None:
    prediction_output = trainer.predict(val_dataset)
    preds = np.argmax(prediction_output.predictions, axis=-1)
    labels = prediction_output.label_ids

    confusion = confusion_matrix(labels, preds, labels=list(range(len(EMOTION_TO_ID))))
    confusion_df = pd.DataFrame(
        confusion,
        index=[ID_TO_EMOTION[idx] for idx in range(len(EMOTION_TO_ID))],
        columns=[ID_TO_EMOTION[idx] for idx in range(len(EMOTION_TO_ID))],
    )
    confusion_df.index.name = "true_emotion"
    confusion_df.columns.name = "predicted_emotion"
    confusion_df.to_csv(output_dir / "validation_confusion_matrix.csv")

    pred_df = val_dataset.frame.copy()
    pred_df["pred_label_id"] = preds
    pred_df["pred_emotion"] = [ID_TO_EMOTION[idx] for idx in preds]
    pred_df["is_correct"] = (pred_df["label"].to_numpy() == preds).astype(float)
    pred_df.to_csv(output_dir / "validation_predictions.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune WavLM-base for 5-class SER on ESD.")
    parser.add_argument("--config-path", default="Configs/config_wavlm_ser.yml")
    args = parser.parse_args()

    config = load_config(Path(args.config_path))
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(int(config.get("seed", 42)))

    metadata = load_metadata(Path(config["train_metadata"]))
    train_df, val_df = split_train_val(
        metadata,
        seed=int(config.get("seed", 42)),
        val_ratio=float(config.get("val_ratio", 0.2)),
    )

    feature_extractor = AutoFeatureExtractor.from_pretrained(config["model_name"])
    model = AutoModelForAudioClassification.from_pretrained(
        config["model_name"],
        num_labels=len(EMOTION_TO_ID),
        label2id=EMOTION_TO_ID,
        id2label=ID_TO_EMOTION,
        ignore_mismatched_sizes=True,
    )
    if bool(config.get("freeze_feature_encoder", True)):
        model.freeze_feature_encoder()

    sample_rate = int(config.get("sample_rate", 16000))
    max_seconds = float(config.get("max_seconds", 6.0))

    train_dataset = ESDSERDataset(train_df, sample_rate=sample_rate, max_seconds=max_seconds, training=True)
    val_dataset = ESDSERDataset(val_df, sample_rate=sample_rate, max_seconds=max_seconds, training=False)
    collator = AudioClassificationCollator(feature_extractor=feature_extractor, sample_rate=sample_rate)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(config.get("train_batch_size", 8)),
        per_device_eval_batch_size=int(config.get("eval_batch_size", 8)),
        learning_rate=float(config.get("learning_rate", 1e-5)),
        num_train_epochs=float(config.get("num_train_epochs", 10)),
        warmup_ratio=float(config.get("warmup_ratio", 0.1)),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=str(config.get("metric_for_best_model", "macro_f1")),
        greater_is_better=True,
        logging_strategy="steps",
        logging_steps=int(config.get("logging_steps", 50)),
        save_total_limit=int(config.get("save_total_limit", 3)),
        fp16=bool(config.get("fp16", False)),
        report_to=[],
        dataloader_num_workers=int(config.get("num_workers", 2)),
        remove_unused_columns=False,
    )

    early_stopping_patience = int(config.get("early_stopping_patience", 2))
    early_stopping_threshold = float(config.get("early_stopping_threshold", 0.0))
    callbacks = []
    if early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience,
                early_stopping_threshold=early_stopping_threshold,
            )
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=feature_extractor,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    trainer.train()
    trainer.save_model()
    feature_extractor.save_pretrained(output_dir)

    split_info = {
        "train_size": len(train_df),
        "val_size": len(val_df),
        "train_speakers": sorted(train_df["speaker_id"].unique().tolist()),
        "val_speakers": sorted(val_df["speaker_id"].unique().tolist()),
        "label2id": EMOTION_TO_ID,
    }
    with open(output_dir / "split_info.json", "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)

    save_confusion_artifacts(trainer, val_dataset, output_dir)
    print(f"Saved WavLM SER model to {output_dir}")


if __name__ == "__main__":
    main()
