#coding: utf-8

import os
import time
import random
import random
from pathlib import Path
from speaker_labels import speaker_key, training_speaker_map, validate_speaker_map
import torch
import torchaudio
from munch import Munch

import numpy as np
import soundfile as sf
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

np.random.seed(1)
random.seed(1)

SPECT_PARAMS = {
    "n_fft": 2048,
    "win_length": 1200,
    "hop_length": 300
}
MEL_PARAMS = {
    "n_mels": 80,
    "n_fft": 2048,
    "win_length": 1200,
    "hop_length": 300
}

class MelDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_list,
                 sr=24000,
                 validation=False,
                 mean=-4.0,
                 std=4.0,
                 return_rank_pair=False,
                 return_speaker_info=False,
                 neutral_label=2,
                 speaker_id_mode='legacy',
                 speaker_to_index=None,
                 ):
        _data_list = [l.rstrip('\r\n').split('|') for l in data_list]
        self.data_list = [(path, int(label)) for path, label in _data_list]
        self.data_list_per_class = {
            target: [(path, label) for path, label in self.data_list if label == target] \
            for target in list(set([label for _, label in self.data_list]))}
        self.emotion_labels = sorted([label for label in self.data_list_per_class if label != neutral_label])

        self.sr = sr
        self.to_melspec = torchaudio.transforms.MelSpectrogram(**MEL_PARAMS)

        self.mean, self.std = float(mean), float(std)
        self.validation = validation
        self.max_mel_length = 192
        self.return_rank_pair = return_rank_pair
        self.return_speaker_info = return_speaker_info
        self.neutral_label = neutral_label
        self.speaker_id_mode = speaker_id_mode
        self.speaker_to_index = {}
        if self.return_speaker_info:
            if speaker_to_index is None:
                if validation and speaker_id_mode != 'legacy':
                    raise ValueError('Validation requires the frozen training speaker mapping')
                self.speaker_to_index = training_speaker_map(data_list, speaker_id_mode)
            else:
                self.speaker_to_index = validate_speaker_map(speaker_to_index)
            if not validation:
                missing = {speaker_key(path, speaker_id_mode) for path, _ in self.data_list} - set(self.speaker_to_index)
                if missing:
                    raise ValueError('Training speakers missing from mapping: ' + str(sorted(missing)))
        if self.return_rank_pair:
            if neutral_label not in self.data_list_per_class:
                raise ValueError(f"neutral_label={neutral_label} is not present in the dataset")
            if not self.emotion_labels:
                raise ValueError("return_rank_pair=True requires at least one non-neutral label")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        mel_tensor, label, wave_tensor = self._load_data(data)
        ref_data = random.choice(self.data_list)
        ref_mel_tensor, ref_label, ref_wave_tensor = self._load_data(ref_data)
        ref2_data = random.choice(self.data_list_per_class[ref_label])
        ref2_mel_tensor, _, _ = self._load_data(ref2_data)
        item = [
            mel_tensor,
            label,
            wave_tensor,
            ref_mel_tensor,
            ref_wave_tensor,
            ref2_mel_tensor,
            ref_label,
        ]

        if self.return_speaker_info:
            item.extend([
                self._parse_speaker_id(data[0]),
                self._parse_speaker_id(ref_data[0]),
                self._parse_speaker_id(ref2_data[0]),
            ])

        if self.return_rank_pair:
            rank_neu_data = random.choice(self.data_list_per_class[self.neutral_label])
            rank_emo_label = random.choice(self.emotion_labels)
            rank_emo_data = random.choice(self.data_list_per_class[rank_emo_label])
            rank_neu_mel_tensor, rank_neu_label, _ = self._load_data(rank_neu_data)
            rank_emo_mel_tensor, rank_emo_label, _ = self._load_data(rank_emo_data)
            item.extend([
                rank_neu_mel_tensor,
                rank_neu_label,
                rank_emo_mel_tensor,
                rank_emo_label,
            ])

        return tuple(item)
    
    def _load_data(self, path):
        wave_tensor, label = self._load_tensor(path)
        
        if not self.validation: # random scale for robustness
            random_scale = 0.5 + 0.5 * np.random.random()
            wave_tensor = random_scale * wave_tensor

        mel_tensor = self.to_melspec(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor) - self.mean) / self.std
        mel_length = mel_tensor.size(1)
        if mel_length > self.max_mel_length:
            random_start = np.random.randint(0, mel_length - self.max_mel_length)
            mel_tensor = mel_tensor[:, random_start:random_start + self.max_mel_length]

        return mel_tensor, label, wave_tensor

    def _preprocess(self, wave_tensor, ):
        mel_tensor = self.to_melspec(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor) - self.mean) / self.std
        return mel_tensor

    def _load_tensor(self, data):
        wave_path, label = data
        label = int(label)
        wave, sr = sf.read(wave_path)
        wave_tensor = torch.from_numpy(wave).float()
        return wave_tensor, label

    def _parse_speaker_id(self, wave_path):
        speaker_dir = speaker_key(wave_path, self.speaker_id_mode)
        if speaker_dir in self.speaker_to_index:
            return self.speaker_to_index[speaker_dir]
        if self.speaker_id_mode != 'legacy':
            if self.validation:
                return -1  # Unseen speaker: never alias a training class.
            raise ValueError('Unknown training speaker: ' + speaker_dir)
        try:
            return int(speaker_dir)
        except ValueError as exc:
            raise ValueError(
                f"Could not parse speaker_id from '{wave_path}'. "
                "Expected an ESD-style path like .../<speaker_id>/<emotion>/<file>.wav."
            ) from exc

class Collater(object):
    """
    Args:
      adaptive_batch_size (bool): if true, decrease batch size when long data comes.
    """

    def __init__(self, return_wave=False, return_rank_pair=False, return_speaker_info=False):
        self.text_pad_index = 0
        self.return_wave = return_wave
        self.return_rank_pair = return_rank_pair
        self.return_speaker_info = return_speaker_info
        self.max_mel_length = 192
        self.mel_length_step = 16
        self.latent_dim = 16

    def __call__(self, batch):
        batch_size = len(batch)
        nmels = batch[0][0].size(0)
        mels = torch.zeros((batch_size, nmels, self.max_mel_length)).float()
        labels = torch.zeros((batch_size)).long()
        ref_mels = torch.zeros((batch_size, nmels, self.max_mel_length)).float()
        ref2_mels = torch.zeros((batch_size, nmels, self.max_mel_length)).float()
        ref_labels = torch.zeros((batch_size)).long()
        if self.return_rank_pair:
            rank_neu_mels = torch.zeros((batch_size, nmels, self.max_mel_length)).float()
            rank_neu_labels = torch.zeros((batch_size)).long()
            rank_emo_mels = torch.zeros((batch_size, nmels, self.max_mel_length)).float()
            rank_emo_labels = torch.zeros((batch_size)).long()
        if self.return_speaker_info:
            speaker_ids = torch.zeros((batch_size)).long()
            ref_speaker_ids = torch.zeros((batch_size)).long()
            ref2_speaker_ids = torch.zeros((batch_size)).long()

        wave_lengths = torch.zeros((batch_size), dtype=torch.long)
        waves = []
        ref_waves = []
        ref_waves_lengths = torch.zeros((batch_size), dtype=torch.long)

        for bid, item in enumerate(batch):
            mel, label, wave, ref_mel, ref_wave, ref2_mel, ref_label = item[:7]
            mel_size = mel.size(1)
            mels[bid, :, :mel_size] = mel
            
            ref_mel_size = ref_mel.size(1)
            ref_mels[bid, :, :ref_mel_size] = ref_mel
            
            ref2_mel_size = ref2_mel.size(1)
            ref2_mels[bid, :, :ref2_mel_size] = ref2_mel
            
            labels[bid] = label
            ref_labels[bid] = ref_label
            wave_lengths[bid] = wave.size(0)
            ref_waves_lengths[bid] = ref_wave.size(0)
            waves.append(wave)
            ref_waves.append(ref_wave)

            offset = 7
            if self.return_speaker_info:
                speaker_id, ref_speaker_id, ref2_speaker_id = item[offset:offset + 3]
                speaker_ids[bid] = speaker_id
                ref_speaker_ids[bid] = ref_speaker_id
                ref2_speaker_ids[bid] = ref2_speaker_id
                offset += 3

            if self.return_rank_pair:
                rank_neu_mel, rank_neu_label, rank_emo_mel, rank_emo_label = item[offset:]
                rank_neu_mel_size = rank_neu_mel.size(1)
                rank_emo_mel_size = rank_emo_mel.size(1)
                rank_neu_mels[bid, :, :rank_neu_mel_size] = rank_neu_mel
                rank_emo_mels[bid, :, :rank_emo_mel_size] = rank_emo_mel
                rank_neu_labels[bid] = rank_neu_label
                rank_emo_labels[bid] = rank_emo_label
            

        waves = nn.utils.rnn.pad_sequence(waves, batch_first=True)
        ref_waves = nn.utils.rnn.pad_sequence(ref_waves,batch_first=True)

        
        z_trg = torch.randn(batch_size, self.latent_dim)
        z_trg2 = torch.randn(batch_size, self.latent_dim)
        
        mels, ref_mels, ref2_mels = mels.unsqueeze(1), ref_mels.unsqueeze(1), ref2_mels.unsqueeze(1)
        if self.return_wave:
            # Output for speaker-encoder style evaluation.
            return Munch(waves=waves, wave_lengths=wave_lengths, mels=mels, labels=labels, ref_waves = ref_waves, ref_waves_lengths=ref_waves_lengths, ref_mels=ref_mels, ref_labels=ref_labels)

        speaker_batch = None
        if self.return_speaker_info:
            speaker_batch = Munch(
                speaker_ids=speaker_ids,
                ref_speaker_ids=ref_speaker_ids,
                ref2_speaker_ids=ref2_speaker_ids,
            )

        if self.return_rank_pair:
            rank_neu_mels = rank_neu_mels.unsqueeze(1)
            rank_emo_mels = rank_emo_mels.unsqueeze(1)
            rank_batch = Munch(
                x_neu=rank_neu_mels,
                y_neu=rank_neu_labels,
                x_emo=rank_emo_mels,
                y_emo=rank_emo_labels,
            )
            if speaker_batch is not None:
                return mels, labels, ref_mels, ref2_mels, ref_labels, z_trg, z_trg2, rank_batch, speaker_batch
            return mels, labels, ref_mels, ref2_mels, ref_labels, z_trg, z_trg2, rank_batch

        if speaker_batch is not None:
            return mels, labels, ref_mels, ref2_mels, ref_labels, z_trg, z_trg2, speaker_batch

        return mels, labels, ref_mels, ref2_mels, ref_labels, z_trg, z_trg2

def build_dataloader(path_list,
                     validation=False,
                     batch_size=4,
                     num_workers=1,
                     device='cpu',
                     collate_config={},
                     dataset_config={},
                     seed=None,
                     drop_last=True):

    dataset = MelDataset(path_list, validation=validation, **dataset_config)
    collate_fn = Collater(**collate_config)
    generator = None
    worker_init_fn = None
    if seed is not None:
        from seed_support import seed_worker
        generator = torch.Generator().manual_seed(seed)
        worker_init_fn = seed_worker
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=(not validation),
                             num_workers=num_workers,
                             drop_last=drop_last,
                             generator=generator,
                             worker_init_fn=worker_init_fn,
                             collate_fn=collate_fn,
                             pin_memory=(device != 'cpu'))

    return data_loader
