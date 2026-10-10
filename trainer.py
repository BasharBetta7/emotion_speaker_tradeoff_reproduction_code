# -*- coding: utf-8 -*-

import os
import os.path as osp
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from PIL import Image
from tqdm import tqdm

from losses import compute_d_loss, compute_g_loss, compute_rank_intensity_loss, compute_style_speaker_invariance_loss, compute_speaker_classifier_loss

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class Trainer(object):
    def __init__(self,
                 args,
                 model=None,
                 model_ema=None,
                 optimizer=None,
                 scheduler=None,
                 config={},
                 device=torch.device("cpu"),
                 logger=logger,
                 train_dataloader=None,
                 val_dataloader=None,
                 initial_steps=0,
                 initial_epochs=0,
                 fp16_run=False,
                 finetune=False
    ):
        self.args = args
        self.steps = initial_steps
        self.epochs = initial_epochs
        self.model = model
        self.model_ema = model_ema
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.device = device
        self.finish_train = False
        self.logger = logger
        self.fp16_run = fp16_run
        self.finetune = finetune

    def _train_epoch(self):
        """Train model one epoch."""
        raise NotImplementedError

    @torch.no_grad()
    def _eval_epoch(self):
        """Evaluate model one epoch."""
        pass

    def save_checkpoint(self, checkpoint_path):
        """Save checkpoint.
        Args:
            checkpoint_path (str): Checkpoint path to be saved.
        """
        state_dict = {
            "optimizer": self.optimizer.state_dict(),
            "steps": self.steps,
            "epochs": self.epochs,
            "model": {key: self.model[key].state_dict() for key in self.model}
        }
        if self.model_ema is not None:
            state_dict['model_ema'] = {key: self.model_ema[key].state_dict() for key in self.model_ema}

        if not os.path.exists(os.path.dirname(checkpoint_path)):
            os.makedirs(os.path.dirname(checkpoint_path))
        torch.save(state_dict, checkpoint_path)

    def load_checkpoint(self, checkpoint_path, load_only_params=False):
        """Load checkpoint.

        Args:
            checkpoint_path (str): Checkpoint path to be loaded.
            load_only_params (bool): Whether to load only model parameters.

        """
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        for key in self.model:
            self._load(state_dict["model"][key], self.model[key])

        if self.model_ema is not None:
            for key in self.model_ema:
                self._load(state_dict["model_ema"][key], self.model_ema[key])
        
        if not load_only_params:
            self.steps = state_dict["steps"]
            self.epochs = state_dict["epochs"]
            self.optimizer.load_state_dict(state_dict["optimizer"])


    def _load(self, states, model, force_load=True):
        model_states = model.state_dict()
        for key, val in states.items():
            try:
                if key not in model_states:
                    continue
                if isinstance(val, nn.Parameter):
                    val = val.data

                if val.shape != model_states[key].shape:
                    self.logger.info("%s does not have same shape" % key)
                    print(val.shape, model_states[key].shape)
                    if not force_load:
                        continue

                    min_shape = np.minimum(np.array(val.shape), np.array(model_states[key].shape))
                    slices = [slice(0, min_index) for min_index in min_shape]
                    model_states[key][slices].copy_(val[slices])
                else:
                    model_states[key].copy_(val)
            except:
                self.logger.info("not exist :%s" % key)
                print("not exist ", key)

    @staticmethod
    def get_gradient_norm(model):
        total_norm = 0
        for p in model.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2

        total_norm = np.sqrt(total_norm)
        return total_norm

    @staticmethod
    def length_to_mask(lengths):
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask+1, lengths.unsqueeze(1))
        return mask

    def _get_lr(self):
        for param_group in self.optimizer.param_groups:
            lr = param_group['lr']
            break
        return lr

    @staticmethod
    def moving_average(model, model_test, beta=0.999):
        for param, param_test in zip(model.parameters(), model_test.parameters()):
            param_test.data = torch.lerp(param.data, param_test.data, beta)

    def _unpack_batch(self, batch):
        rank_batch = None
        speaker_batch = None
        batch = list(batch)
        while len(batch) > 7:
            extra = batch.pop()
            if hasattr(extra, "speaker_ids"):
                speaker_batch = extra
            else:
                rank_batch = extra
        return batch, rank_batch, speaker_batch

    def _move_munch_to_device(self, value):
        if value is None:
            return None
        for key, item in value.items():
            value[key] = item.to(self.device)
        return value

    def _speaker_loss_enabled(self):
        speaker_loss_args = self.args.get("speaker_loss", {})
        return bool(speaker_loss_args.get("enabled", False)) and float(speaker_loss_args.get("lambda_spk_inv", 0.0)) > 0

    def _speaker_classifier_enabled(self):
        return hasattr(self.model, "speaker_classifier")

    def _set_speaker_classifier_trainable(self, trainable):
        if not self._speaker_classifier_enabled():
            return
        for param in self.model.speaker_classifier.parameters():
            param.requires_grad = trainable
        self.model.speaker_classifier.train(trainable)

    def _compute_speaker_invariance_loss(self, x_real, y_org, x_ref, x_ref2, y_trg, speaker_batch):
        speaker_loss_args = self.args.get("speaker_loss", {})
        style_embeddings = torch.cat([
            self.model.style_encoder(x_real, y_org),
            self.model.style_encoder(x_ref, y_trg),
            self.model.style_encoder(x_ref2, y_trg),
        ], dim=0)
        emotion_labels = torch.cat([y_org, y_trg, y_trg], dim=0)
        speaker_ids = torch.cat([
            speaker_batch.speaker_ids,
            speaker_batch.ref_speaker_ids,
            speaker_batch.ref2_speaker_ids,
        ], dim=0)
        speaker_loss, speaker_losses = compute_style_speaker_invariance_loss(
            style_embeddings,
            emotion_labels,
            speaker_ids,
            temperature=speaker_loss_args.get("temperature", 0.1),
        )
        weighted_loss = float(speaker_loss_args.get("lambda_spk_inv", 0.0)) * speaker_loss
        return weighted_loss, speaker_losses

    def _train_epoch(self):
        self.epochs += 1
        
        train_losses = defaultdict(list)
        _ = [self.model[k].train() for k in self.model]
        if self.finetune:
            self.model.generator.stem.eval()
            self.model.generator.encode.eval()
            self.model.asr_model.eval()
            self.model.f0_model.eval()
        scaler = torch.cuda.amp.GradScaler() if (('cuda' in str(self.device)) and self.fp16_run) else None

        use_con_reg = (self.epochs >= self.args.con_reg_epoch)
        use_adv_cls = (self.epochs >= self.args.adv_cls_epoch)
        use_adv_spk_cls = (self.epochs >= getattr(self.args, "adv_spk_cls_epoch", float("inf")))
        train_spk_cls = self._speaker_classifier_enabled() and not use_adv_spk_cls
        self._set_speaker_classifier_trainable(train_spk_cls)
        
        use_speaker_loss = self._speaker_loss_enabled()
        
        for train_steps_per_epoch, batch in enumerate(tqdm(self.train_dataloader, desc="[train]"), 1):

            ### load data
            batch, rank_batch, speaker_batch = self._unpack_batch(batch)
            batch = [b.to(self.device) for b in batch]
            rank_batch = self._move_munch_to_device(rank_batch)
            speaker_batch = self._move_munch_to_device(speaker_batch)
            x_real, y_org, x_ref, x_ref2, y_trg, z_trg, z_trg2 = batch

            # train speaker classifier on real mels until adv_spk_cls_epoch
            if train_spk_cls and speaker_batch is not None:
                self.optimizer.zero_grad()
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        spk_cls_loss, spk_cls_losses = compute_speaker_classifier_loss(
                            self.model,
                            self.args.d_loss,
                            x_real,
                            speaker_batch.speaker_ids,
                            x_ref=x_ref,
                            ref_speaker_ids=speaker_batch.ref_speaker_ids,
                            x_ref2=x_ref2,
                            ref2_speaker_ids=speaker_batch.ref2_speaker_ids)
                    scaler.scale(spk_cls_loss).backward()
                else:
                    spk_cls_loss, spk_cls_losses = compute_speaker_classifier_loss(
                        self.model,
                        self.args.d_loss,
                        x_real,
                        speaker_batch.speaker_ids,
                        x_ref=x_ref,
                        ref_speaker_ids=speaker_batch.ref_speaker_ids,
                        x_ref2=x_ref2,
                        ref2_speaker_ids=speaker_batch.ref2_speaker_ids)
                    spk_cls_loss.backward()
                self.optimizer.step('speaker_classifier', scaler=scaler)
                for key in spk_cls_losses:
                    train_losses["train/%s" % key].append(spk_cls_losses[key])
            
            # train the discriminator (by random reference)
            self.optimizer.zero_grad()
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    d_loss, d_losses_latent = compute_d_loss(self.model, self.args.d_loss, x_real, y_org, y_trg, z_trg=z_trg, use_adv_cls=use_adv_cls, use_con_reg=use_con_reg)
                scaler.scale(d_loss).backward()
            else:
                d_loss, d_losses_latent = compute_d_loss(self.model, self.args.d_loss, x_real, y_org, y_trg, z_trg=z_trg, use_adv_cls=use_adv_cls, use_con_reg=use_con_reg)
                d_loss.backward()
            self.optimizer.step('discriminator', scaler=scaler)

            # train the discriminator (by target reference)
            self.optimizer.zero_grad()
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    d_loss, d_losses_ref = compute_d_loss(self.model, self.args.d_loss, x_real, y_org, y_trg, x_ref=x_ref, use_adv_cls=use_adv_cls, use_con_reg=use_con_reg)
                scaler.scale(d_loss).backward()
            else:
                d_loss, d_losses_ref = compute_d_loss(self.model, self.args.d_loss, x_real, y_org, y_trg, x_ref=x_ref, use_adv_cls=use_adv_cls, use_con_reg=use_con_reg)
                d_loss.backward()

            self.optimizer.step('discriminator', scaler=scaler)

            # train the generator (by random reference)
            self.optimizer.zero_grad()
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    g_loss, g_losses_latent = compute_g_loss(
                        self.model, self.args.g_loss, x_real, y_org, y_trg,
                        z_trgs=[z_trg, z_trg2],
                        use_adv_cls=use_adv_cls,
                        detach_auxiliary_features=self.finetune,
                        use_spk_adv_cls=use_adv_spk_cls and speaker_batch is not None,
                        speaker_ids=speaker_batch.speaker_ids if speaker_batch is not None else None)
                    if use_speaker_loss and speaker_batch is not None:
                        speaker_loss, speaker_losses = self._compute_speaker_invariance_loss(
                            x_real, y_org, x_ref, x_ref2, y_trg, speaker_batch)
                        g_loss = g_loss + speaker_loss
                scaler.scale(g_loss).backward()
            else:
                g_loss, g_losses_latent = compute_g_loss(
                    self.model, self.args.g_loss, x_real, y_org, y_trg,
                    z_trgs=[z_trg, z_trg2],
                    use_adv_cls=use_adv_cls,
                    detach_auxiliary_features=self.finetune,
                    use_spk_adv_cls=use_adv_spk_cls and speaker_batch is not None,
                    speaker_ids=speaker_batch.speaker_ids if speaker_batch is not None else None)
                if use_speaker_loss and speaker_batch is not None:
                    speaker_loss, speaker_losses = self._compute_speaker_invariance_loss(
                        x_real, y_org, x_ref, x_ref2, y_trg, speaker_batch)
                    g_loss = g_loss + speaker_loss
                g_loss.backward()

            self.optimizer.step('generator', scaler=scaler)
            self.optimizer.step('mapping_network', scaler=scaler)
            self.optimizer.step('style_encoder', scaler=scaler)
            if use_speaker_loss and speaker_batch is not None:
                for key in speaker_losses:
                    train_losses["train/%s" % key].append(speaker_losses[key])

            # train the generator (by target reference)
            self.optimizer.zero_grad()
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    g_loss, g_losses_ref = compute_g_loss(
                        self.model, self.args.g_loss, x_real, y_org, y_trg,
                        x_refs=[x_ref, x_ref2],
                        use_adv_cls=use_adv_cls,
                        detach_auxiliary_features=self.finetune,
                        use_spk_adv_cls=use_adv_spk_cls and speaker_batch is not None,
                        speaker_ids=speaker_batch.speaker_ids if speaker_batch is not None else None)
                scaler.scale(g_loss).backward()
            else:
                g_loss, g_losses_ref = compute_g_loss(
                    self.model, self.args.g_loss, x_real, y_org, y_trg,
                    x_refs=[x_ref, x_ref2],
                    use_adv_cls=use_adv_cls,
                    detach_auxiliary_features=self.finetune,
                    use_spk_adv_cls=use_adv_spk_cls and speaker_batch is not None,
                    speaker_ids=speaker_batch.speaker_ids if speaker_batch is not None else None)
                g_loss.backward()
            self.optimizer.step('generator', scaler=scaler)

            if rank_batch is not None and hasattr(self.model, "rank_head"):
                self.optimizer.zero_grad()
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        rank_loss, rank_losses = compute_rank_intensity_loss(
                            self.model,
                            self.args.rank_loss,
                            rank_batch.x_neu,
                            rank_batch.y_neu,
                            rank_batch.x_emo,
                            rank_batch.y_emo,
                        )
                    scaler.scale(rank_loss).backward()
                else:
                    rank_loss, rank_losses = compute_rank_intensity_loss(
                        self.model,
                        self.args.rank_loss,
                        rank_batch.x_neu,
                        rank_batch.y_neu,
                        rank_batch.x_emo,
                        rank_batch.y_emo,
                    )
                    rank_loss.backward()
                self.optimizer.step('style_encoder', scaler=scaler)
                self.optimizer.step('rank_head', scaler=scaler)
                for key in rank_losses:
                    train_losses["train/rank_%s" % key].append(rank_losses[key])

            # compute moving average of network parameters
            self.moving_average(self.model.generator, self.model_ema.generator, beta=0.999)
            self.moving_average(self.model.mapping_network, self.model_ema.mapping_network, beta=0.999)
            self.moving_average(self.model.style_encoder, self.model_ema.style_encoder, beta=0.999)
            if hasattr(self.model, "rank_head") and hasattr(self.model_ema, "rank_head"):
                self.moving_average(self.model.rank_head, self.model_ema.rank_head, beta=0.999)
            if hasattr(self.model, "speaker_classifier") and hasattr(self.model_ema, "speaker_classifier"):
                self.moving_average(self.model.speaker_classifier, self.model_ema.speaker_classifier, beta=0.999)
            self.optimizer.scheduler()

            for key in d_losses_latent:
                train_losses["train/%s" % key].append(d_losses_latent[key])
            for key in g_losses_latent:
                train_losses["train/%s" % key].append(g_losses_latent[key])


        train_losses = {key: np.mean(value) for key, value in train_losses.items()}
        return train_losses

    @torch.no_grad()
    def _eval_epoch(self):
        use_adv_cls = (self.epochs >= self.args.adv_cls_epoch)
        
        eval_losses = defaultdict(list)
        eval_images = defaultdict(list)
        _ = [self.model[k].eval() for k in self.model]
        for eval_steps_per_epoch, batch in enumerate(tqdm(self.val_dataloader, desc="[eval]"), 1):

            ### load data
            batch, _, _ = self._unpack_batch(batch)
            batch = [b.to(self.device) for b in batch]
            x_real, y_org, x_ref, x_ref2, y_trg, z_trg, z_trg2 = batch

            # train the discriminator
            d_loss, d_losses_latent = compute_d_loss(
                self.model, self.args.d_loss, x_real, y_org, y_trg, z_trg=z_trg, use_r1_reg=False, use_adv_cls=use_adv_cls)
            d_loss, d_losses_ref = compute_d_loss(
                self.model, self.args.d_loss, x_real, y_org, y_trg, x_ref=x_ref, use_r1_reg=False, use_adv_cls=use_adv_cls)

            # train the generator
            g_loss, g_losses_latent = compute_g_loss(
                self.model, self.args.g_loss, x_real, y_org, y_trg, z_trgs=[z_trg, z_trg2], use_adv_cls=use_adv_cls, detach_auxiliary_features=self.finetune)
            g_loss, g_losses_ref = compute_g_loss(
                self.model, self.args.g_loss, x_real, y_org, y_trg, x_refs=[x_ref, x_ref2], use_adv_cls=use_adv_cls, detach_auxiliary_features=self.finetune)

            for key in d_losses_latent:
                eval_losses["eval/%s" % key].append(d_losses_latent[key])
            for key in g_losses_latent:
                eval_losses["eval/%s" % key].append(g_losses_latent[key])

#             if eval_steps_per_epoch % 10 == 0:
#                 # generate x_fake
#                 s_trg = self.model_ema.style_encoder(x_ref, y_trg)
#                 F0 = self.model.f0_model.get_feature_GAN(x_real)
#                 x_fake = self.model_ema.generator(x_real, s_trg, masks=None, F0=F0)
#                 # generate x_recon
#                 s_real = self.model_ema.style_encoder(x_real, y_org)
#                 F0_fake = self.model.f0_model.get_feature_GAN(x_fake)
#                 x_recon = self.model_ema.generator(x_fake, s_real, masks=None, F0=F0_fake)
                
#                 eval_images['eval/image'].append(
#                     ([x_real[0, 0].cpu().numpy(),
#                     x_fake[0, 0].cpu().numpy(),
#                     x_recon[0, 0].cpu().numpy()]))

        eval_losses = {key: np.mean(value) for key, value in eval_losses.items()}
        eval_losses.update(eval_images)
        return eval_losses
 
