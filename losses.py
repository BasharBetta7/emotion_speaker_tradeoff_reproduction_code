#coding:utf-8

import os
import torch

from torch import nn
from munch import Munch
from transforms import build_transforms

import torch.nn.functional as F
import numpy as np

def compute_d_loss(nets, args, x_real, y_org, y_trg, z_trg=None, x_ref=None, use_r1_reg=True, use_adv_cls=False, use_con_reg=False):
    args = Munch(args)

    assert (z_trg is None) != (x_ref is None)
    # with real audios
    x_real.requires_grad_()
    out = nets.discriminator(x_real, y_org)
    loss_real = adv_loss(out, 1)
    
    # R1 regularizaition (https://arxiv.org/abs/1801.04406v4)
    if use_r1_reg:
        loss_reg = r1_reg(out, x_real)
    else:
        loss_reg = torch.FloatTensor([0]).to(x_real.device)
    
    # consistency regularization (bCR-GAN: https://arxiv.org/abs/2002.04724)
    loss_con_reg = torch.FloatTensor([0]).to(x_real.device)
    if use_con_reg:
        t = build_transforms()
        out_aug = nets.discriminator(t(x_real).detach(), y_org)
        loss_con_reg += F.smooth_l1_loss(out, out_aug)
    
    # with fake audios
    with torch.no_grad():
        if z_trg is not None:
            s_trg = nets.mapping_network(z_trg, y_trg)
        else:  # x_ref is not None
            s_trg = nets.style_encoder(x_ref, y_trg)
            
        F0 = nets.f0_model.get_feature_GAN(x_real)
        x_fake = nets.generator(x_real, s_trg, masks=None, F0=F0)
    out = nets.discriminator(x_fake, y_trg)
    loss_fake = adv_loss(out, 0)
    if use_con_reg:
        out_aug = nets.discriminator(t(x_fake).detach(), y_trg)
        loss_con_reg += F.smooth_l1_loss(out, out_aug)
    
    # adversarial classifier loss
    if use_adv_cls:
        out_de = nets.discriminator.classifier(x_fake)
        loss_real_adv_cls = F.cross_entropy(out_de[y_org != y_trg], y_org[y_org != y_trg])
        
        if use_con_reg:
            out_de_aug = nets.discriminator.classifier(t(x_fake).detach())
            loss_con_reg += F.smooth_l1_loss(out_de, out_de_aug)
    else:
        loss_real_adv_cls = torch.zeros(1).mean()
        
    loss = loss_real + loss_fake + args.lambda_reg * loss_reg + \
            args.lambda_adv_cls * loss_real_adv_cls + \
            args.lambda_con_reg * loss_con_reg 

    return loss, Munch(real=loss_real.item(),
                       fake=loss_fake.item(),
                       reg=loss_reg.item(),
                       real_adv_cls=loss_real_adv_cls.item(),
                       con_reg=loss_con_reg.item())

def compute_g_loss(nets, args, x_real, y_org, y_trg, z_trgs=None, x_refs=None,
                   use_adv_cls=False, detach_auxiliary_features=False,
                   use_spk_adv_cls=False, speaker_ids=None):
    args = Munch(args)
    
    assert (z_trgs is None) != (x_refs is None)
    if z_trgs is not None:
        z_trg, z_trg2 = z_trgs
    if x_refs is not None:
        x_ref, x_ref2 = x_refs
        
    # compute style vectors
    if z_trgs is not None:
        s_trg = nets.mapping_network(z_trg, y_trg)
    else:
        s_trg = nets.style_encoder(x_ref, y_trg)
    
    # compute ASR/F0 features (real)
    with torch.no_grad():
        F0_real, GAN_F0_real, cyc_F0_real = nets.f0_model(x_real)
        ASR_real = nets.asr_model.get_feature(x_real)
    
    # adversarial loss
    x_fake = nets.generator(x_real, s_trg, masks=None, F0=GAN_F0_real)
    out = nets.discriminator(x_fake, y_trg) 
    loss_adv = adv_loss(out, 1)
    
    # compute ASR/F0 features (fake)
    if detach_auxiliary_features:
        with torch.no_grad():
            F0_fake, GAN_F0_fake, _ = nets.f0_model(x_fake)
            ASR_fake = nets.asr_model.get_feature(x_fake)
    else:
        F0_fake, GAN_F0_fake, _ = nets.f0_model(x_fake)
        ASR_fake = nets.asr_model.get_feature(x_fake)
    
    # norm consistency loss
    x_fake_norm = log_norm(x_fake)
    x_real_norm = log_norm(x_real)
    loss_norm = ((torch.nn.ReLU()(torch.abs(x_fake_norm - x_real_norm) - args.norm_bias))**2).mean()
    
    # F0 loss
    loss_f0 = f0_loss(F0_fake, F0_real)
    
    # style F0 loss (style initialization)
    if x_refs is not None and args.lambda_f0_sty > 0 and not use_adv_cls:
        if detach_auxiliary_features:
            with torch.no_grad():
                F0_sty, _, _ = nets.f0_model(x_ref)
        else:
            F0_sty, _, _ = nets.f0_model(x_ref)
        loss_f0_sty = F.l1_loss(compute_mean_f0(F0_fake), compute_mean_f0(F0_sty))
    else:
        loss_f0_sty = torch.zeros(1).mean()
    
    # ASR loss
    loss_asr = F.smooth_l1_loss(ASR_fake, ASR_real)
    
    # style reconstruction loss
    s_pred = nets.style_encoder(x_fake, y_trg)
    loss_sty = torch.mean(torch.abs(s_pred - s_trg))
    
    # diversity sensitive loss
    if z_trgs is not None:
        s_trg2 = nets.mapping_network(z_trg2, y_trg)
    else:
        s_trg2 = nets.style_encoder(x_ref2, y_trg)
    x_fake2 = nets.generator(x_real, s_trg2, masks=None, F0=GAN_F0_real)
    x_fake2 = x_fake2.detach()
    if detach_auxiliary_features:
        with torch.no_grad():
            _, GAN_F0_fake2, _ = nets.f0_model(x_fake2)
    else:
        _, GAN_F0_fake2, _ = nets.f0_model(x_fake2)
    loss_ds = torch.mean(torch.abs(x_fake - x_fake2))
    loss_ds += F.smooth_l1_loss(GAN_F0_fake, GAN_F0_fake2.detach())
    
    # cycle-consistency loss
    s_org = nets.style_encoder(x_real, y_org)
    x_rec = nets.generator(x_fake, s_org, masks=None, F0=GAN_F0_fake)
    loss_cyc = torch.mean(torch.abs(x_rec - x_real))
    # F0 loss in cycle-consistency loss
    if args.lambda_f0 > 0:
        if detach_auxiliary_features:
            with torch.no_grad():
                _, _, cyc_F0_rec = nets.f0_model(x_rec)
        else:
            _, _, cyc_F0_rec = nets.f0_model(x_rec)
        loss_cyc += F.smooth_l1_loss(cyc_F0_rec, cyc_F0_real)
    if args.lambda_asr > 0:
        if detach_auxiliary_features:
            with torch.no_grad():
                ASR_recon = nets.asr_model.get_feature(x_rec)
        else:
            ASR_recon = nets.asr_model.get_feature(x_rec)
        loss_cyc += F.smooth_l1_loss(ASR_recon, ASR_real)
    
    # adversarial classifier loss
    if use_adv_cls:
        out_de = nets.discriminator.classifier(x_fake)
        loss_adv_cls = F.cross_entropy(out_de[y_org != y_trg], y_trg[y_org != y_trg])
    else:
        loss_adv_cls = torch.zeros(1).mean()

    # source-speaker preservation loss
    if use_spk_adv_cls and speaker_ids is not None and hasattr(nets, "speaker_classifier"):
        out_spk = nets.speaker_classifier(x_fake)
        loss_spk_adv = speaker_cross_entropy(out_spk, speaker_ids)
    else:
        loss_spk_adv = torch.zeros(1, device=x_real.device).mean()
    
    loss = args.lambda_adv * loss_adv + args.lambda_sty * loss_sty \
           - args.lambda_ds * loss_ds + args.lambda_cyc * loss_cyc\
           + args.lambda_norm * loss_norm \
           + args.lambda_asr * loss_asr \
           + args.lambda_f0 * loss_f0 \
           + args.lambda_f0_sty * loss_f0_sty \
           + args.lambda_adv_cls * loss_adv_cls \
           + float(args.get("lambda_adv_spk_cls", 0.0)) * loss_spk_adv

    return loss, Munch(adv=loss_adv.item(),
                       sty=loss_sty.item(),
                       ds=loss_ds.item(),
                       cyc=loss_cyc.item(),
                       norm=loss_norm.item(),
                       asr=loss_asr.item(),
                       f0=loss_f0.item(),
                       adv_cls=loss_adv_cls.item(),
                       spk_adv_loss=loss_spk_adv.item())


def speaker_cross_entropy(logits, speaker_ids):
    """Ignore unseen (-1) speakers; all-unknown batches have finite zero loss."""
    valid = speaker_ids >= 0
    if not bool(valid.any()):
        return logits.sum() * 0.0
    return F.cross_entropy(logits[valid], speaker_ids[valid])


def compute_speaker_classifier_loss(nets, args, x_real, speaker_ids,
                                    x_ref=None, ref_speaker_ids=None,
                                    x_ref2=None, ref2_speaker_ids=None):
    args = Munch(args)
    if not hasattr(nets, "speaker_classifier"):
        zero = torch.zeros(1, device=x_real.device).mean()
        return zero, Munch(spk_cls=0.0, spk_cls_acc=0.0)

    xs = [x_real]
    ys = [speaker_ids]
    if x_ref is not None and ref_speaker_ids is not None:
        xs.append(x_ref)
        ys.append(ref_speaker_ids)
    if x_ref2 is not None and ref2_speaker_ids is not None:
        xs.append(x_ref2)
        ys.append(ref2_speaker_ids)

    x = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0)
    logits = nets.speaker_classifier(x)
    loss_cls = speaker_cross_entropy(logits, y)
    loss = float(args.get("lambda_spk_cls", 1.0)) * loss_cls
    valid = y >= 0
    acc = (logits[valid].argmax(dim=1) == y[valid]).float().mean() if bool(valid.any()) else logits.new_tensor(float('nan'))

    return loss, Munch(spk_cls=loss_cls.item(),
                       spk_cls_acc=acc.item())


def compute_rank_intensity_loss(nets, args, x_neu, y_neu, x_emo, y_emo):
    args = Munch(args)
    if not hasattr(nets, "rank_head"):
        zero = torch.zeros(1, device=x_neu.device).mean()
        return zero, Munch(rank=0.0, interp=0.0, mono=0.0)

    margin = float(args.get("rank_margin", 0.2))
    margin_alpha = float(args.get("rank_margin_alpha", 0.05))
    eps = float(args.get("rank_eps", 1e-6))
    lambda_rank = float(args.get("lambda_rank_margin", 0.2))
    lambda_interp = float(args.get("lambda_rank_interp", 0.2))
    lambda_mono = float(args.get("lambda_rank_mono", 0.1))

    s_neu = nets.style_encoder(x_neu, y_neu)
    s_emo = nets.style_encoder(x_emo, y_emo)
    batch_idx = torch.arange(y_emo.size(0), device=y_emo.device)

    r_neu = nets.rank_head(s_neu)[batch_idx, y_emo]
    r_emo = nets.rank_head(s_emo)[batch_idx, y_emo]
    rank_gap = r_emo - r_neu
    loss_rank = F.relu(margin - rank_gap).mean()

    alpha = torch.rand(y_emo.size(0), 1, device=x_neu.device, dtype=s_neu.dtype)
    s_alpha = torch.lerp(s_neu, s_emo, alpha)
    r_alpha = nets.rank_head(s_alpha)[batch_idx, y_emo]
    rank_gap_safe = rank_gap.detach().abs().clamp_min(eps)
    intensity_alpha = (r_alpha - r_neu) / rank_gap_safe
    loss_interp = F.l1_loss(intensity_alpha, alpha.squeeze(1))

    alpha_pair = torch.rand(y_emo.size(0), 2, device=x_neu.device, dtype=s_neu.dtype)
    alpha_low, alpha_high = alpha_pair.sort(dim=1).values.unbind(dim=1)
    s_low = torch.lerp(s_neu, s_emo, alpha_low.unsqueeze(1))
    s_high = torch.lerp(s_neu, s_emo, alpha_high.unsqueeze(1))
    r_low = nets.rank_head(s_low)[batch_idx, y_emo]
    r_high = nets.rank_head(s_high)[batch_idx, y_emo]
    loss_mono = F.relu(margin_alpha - r_high + r_low).mean()

    loss = lambda_rank * loss_rank + lambda_interp * loss_interp + lambda_mono * loss_mono
    return loss, Munch(rank=loss_rank.item(),
                       interp=loss_interp.item(),
                       mono=loss_mono.item())

def compute_style_speaker_invariance_loss(style_embeddings, emotion_labels, speaker_ids, temperature=0.1):
    known = speaker_ids >= 0
    if not bool(known.all()):
        style_embeddings = style_embeddings[known]
        emotion_labels = emotion_labels[known]
        speaker_ids = speaker_ids[known]
    if style_embeddings.size(0) <= 1:
        zero = style_embeddings.sum() * 0
        return zero, Munch(spk_inv=0.0, spk_inv_valid_anchors=0.0)

    temperature = max(float(temperature), 1e-6)
    embeddings = F.normalize(style_embeddings, dim=1)
    logits = torch.matmul(embeddings, embeddings.t()) / temperature

    batch_size = style_embeddings.size(0)
    device = style_embeddings.device
    eye = torch.eye(batch_size, dtype=torch.bool, device=device)
    same_emotion = emotion_labels.unsqueeze(0) == emotion_labels.unsqueeze(1)
    same_speaker = speaker_ids.unsqueeze(0) == speaker_ids.unsqueeze(1)

    positive_mask = same_emotion & (~same_speaker) & (~eye)
    negative_mask = (~same_emotion) & (~eye)
    candidate_mask = positive_mask | negative_mask

    valid_anchor_mask = positive_mask.any(dim=1)
    if not valid_anchor_mask.any():
        zero = style_embeddings.sum() * 0
        return zero, Munch(spk_inv=0.0, spk_inv_valid_anchors=0.0)

    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits) * candidate_mask.float()
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    positive_count = positive_mask.sum(dim=1).clamp_min(1)
    mean_log_prob_pos = (positive_mask.float() * log_prob).sum(dim=1) / positive_count
    loss = -mean_log_prob_pos[valid_anchor_mask].mean()

    return loss, Munch(spk_inv=loss.item(),
                       spk_inv_valid_anchors=float(valid_anchor_mask.float().sum().item()))
    


# for norm consistency loss
def log_norm(x, mean=-4, std=4, dim=2):
    """
    normalized log mel -> mel -> norm -> log(norm)
    """
    x = torch.log(torch.exp(x * std + mean).norm(dim=dim))
    return x

# for adversarial loss
def adv_loss(logits, target):
    assert target in [1, 0]
    if len(logits.shape) > 1:
        logits = logits.reshape(-1)
    targets = torch.full_like(logits, fill_value=target)
    logits = logits.clamp(min=-10, max=10) # prevent nan
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss

# for R1 regularization loss
def r1_reg(d_out, x_in):
    # zero-centered gradient penalty for real images
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg

# for F0 consistency loss
def compute_mean_f0(f0):
    f0_mean = f0.mean(-1)
    f0_mean = f0_mean.expand(f0.shape[-1], f0_mean.shape[0]).transpose(0, 1) # (B, M)
    return f0_mean

def f0_loss(x_f0, y_f0):
    """
    x.shape = (B, 1, M, L): predict
    y.shape = (B, 1, M, L): target
    """
    # compute the mean
    x_mean = compute_mean_f0(x_f0)
    y_mean = compute_mean_f0(y_f0)
    loss = F.l1_loss(x_f0 / x_mean, y_f0 / y_mean)
    return loss
