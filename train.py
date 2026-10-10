#!/usr/bin/env python3
#coding:utf-8

import os
import os.path as osp
import re
import sys
import yaml
import shutil
import numpy as np
import torch
import click
import warnings
import json
import numbers
from contextlib import nullcontext
from pathlib import Path
from seed_support import seed_everything, validation_rng, check_splits, SeedTracker, evaluate_checkpoint
warnings.simplefilter('ignore')

from functools import reduce
from munch import Munch

from meldataset import build_dataloader
from speaker_labels import configure_speakers
from optimizers import build_optimizer
from models import build_model
from trainer import Trainer
from torch.utils.tensorboard import SummaryWriter

from Utils.ASR.models import ASRCNN
from Utils.JDC.model import JDCNet

import logging
from logging import StreamHandler
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = StreamHandler()
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)

torch.backends.cudnn.benchmark = True #

def set_requires_grad(module, requires_grad):
    for param in module.parameters():
        param.requires_grad = requires_grad

def apply_finetune_freeze(model):
    # Freeze both content paths: the generator-side encoder and the pretrained ASR content model.
    set_requires_grad(model.generator.stem, False)
    set_requires_grad(model.generator.encode, False)
    model.generator.freeze_content_encoder = True
    set_requires_grad(model.asr_model, False)
    set_requires_grad(model.f0_model, False)

def count_trainable_params(module):
    return sum(param.numel() for param in module.parameters() if param.requires_grad)

@click.command()
@click.option('-p', '--config_path', default='Configs/config.yml', type=str)

def main(config_path):
    config = yaml.safe_load(open(config_path))

    # Validate/freeze explicit speaker labels before creating a run or loading models.
    speaker_mapping = None
    if config.get('dataset_params', {}).get('speaker_id_mode', 'legacy') != 'legacy':
        speaker_mapping = configure_speakers(config, Path(config['train_data']).read_text().splitlines())

    seeded = 'seed_protocol' in config
    if seeded:
        if config.get('pretrained_model') and not config.get('load_only_params', False):
            raise ValueError('Seed runs support fresh training or parameter-only initialization, not resume')
        check_splits(config)
        seed_everything(config['seed'], config['seed_protocol']['deterministic'])

    log_dir = config['log_dir']
    if seeded and Path(log_dir).exists() and any(Path(log_dir).iterdir()):
        raise FileExistsError(f'Refusing to overwrite seed run: {log_dir}')
    if seeded:
        os.makedirs(log_dir, exist_ok=False)
    elif not osp.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    shutil.copy(config_path, osp.join(log_dir, osp.basename(config_path)))
    if speaker_mapping is not None:
        Path(log_dir, 'speaker-map.json').write_text(json.dumps({
            'mode': config['dataset_params']['speaker_id_mode'],
            'speaker_to_index': speaker_mapping, 'unknown_validation_id': -1,
        }, indent=2) + '\n')
        # Persist the effective mapping without changing the caller's config.
        Path(log_dir, osp.basename(config_path)).write_text(yaml.safe_dump(config, sort_keys=False))
    writer = SummaryWriter(log_dir + "/tensorboard")

    if seeded:
        config['_resolved_config'] = str(Path(config_path).resolve())
    tracker = SeedTracker(config) if seeded else None
    if seeded:
        # Tracking initialization must not perturb the training RNG stream.
        seed_everything(config['seed'], config['seed_protocol']['deterministic'])
        metadata = {'seed': config['seed'], 'torch': torch.__version__,
                    'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
                    'gpu': torch.cuda.get_device_name() if torch.cuda.is_available() else None}
        Path(log_dir, 'environment.json').write_text(json.dumps(metadata, indent=2))

    # write logs
    file_handler = logging.FileHandler(osp.join(log_dir, 'train.log'))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(levelname)s:%(asctime)s: %(message)s'))
    logger.addHandler(file_handler)
    
    batch_size = config.get('batch_size', 10)
    device = config.get('device', 'cpu')
    epochs = config.get('epochs', 1000)
    save_freq = config.get('save_freq', 20)
    train_path = config.get('train_data', None)
    val_path = config.get('val_data', None)
    stage = config.get('stage', 'star')
    fp16_run = config.get('fp16_run', False)
    dataset_config = config.get('dataset_params', {})
    model_params = config.get('model_params', {})
    use_rank_intensity = model_params.get('rank_enable', model_params.get('use_rank_intensity', False))
    use_spk_cls = bool(model_params.get('spk_cls_enable', False))
    speaker_loss_config = config.get('loss_params', {}).get('speaker_loss', {})
    use_speaker_loss = bool(speaker_loss_config.get('enabled', False))
    neutral_label = config.get('loss_params', {}).get('rank_loss', {}).get('neutral_label', 2)
    if use_rank_intensity or use_speaker_loss or use_spk_cls:
        dataset_config = dataset_config.copy()
    if use_rank_intensity:
        dataset_config['return_rank_pair'] = True
        dataset_config['neutral_label'] = neutral_label
    if use_speaker_loss or use_spk_cls:
        dataset_config['return_speaker_info'] = True
    collate_config = {
        "return_rank_pair": use_rank_intensity,
        "return_speaker_info": use_speaker_loss or use_spk_cls,
    }
    
    # load data
    train_list, val_list = get_data_path_list(train_path, val_path)
    if use_spk_cls and 'num_speakers' not in model_params and 'n_speakers' not in model_params:
        speaker_dirs = sorted({osp.basename(osp.dirname(osp.dirname(line.split('|')[0]))) for line in train_list})
        model_params['num_speakers'] = len(speaker_dirs)
    train_dataloader = build_dataloader(train_list,
                                        batch_size=batch_size,
                                        num_workers=1,
                                        device=device,
                                        collate_config=collate_config,
                                        dataset_config=dataset_config,
                                        seed=config['seed'] if seeded else None)
    val_dataloader = build_dataloader(val_list,
                                      batch_size=batch_size,
                                      validation=True,
                                      num_workers=1,
                                      device=device,
                                      collate_config=collate_config,
                                      dataset_config=dataset_config,
                                      seed=config['seed_protocol']['evaluation_seed'] if seeded else None,
                                      drop_last=False if seeded else True)
    # load pretrained ASR model
    ASR_config = config.get('ASR_config', False)
    ASR_path = config.get('ASR_path', False)
    with open(ASR_config) as f:
            ASR_config = yaml.safe_load(f)
    ASR_model_config = ASR_config['model_params']
    ASR_model = ASRCNN(**ASR_model_config)
    params = torch.load(ASR_path, map_location='cpu', weights_only=False)['model']
    ASR_model.load_state_dict(params)
    _ = ASR_model.eval()
    
    # load pretrained F0 model
    F0_path = config.get('F0_path', False)
    F0_model = JDCNet(num_class=1, seq_len=192)
    params = torch.load(F0_path, map_location='cpu')['net']
    F0_model.load_state_dict(params)
    
    # build model
    model, model_ema = build_model(Munch(config['model_params']), F0_model, ASR_model)
    finetune = config.get('finetune', False)
    if finetune:
        apply_finetune_freeze(model)

    scheduler_params = {
        "max_lr": float(config['optimizer_params'].get('lr', 2e-4)),
        "pct_start": float(config['optimizer_params'].get('pct_start', 0.0)),
        "epochs": epochs,
        "steps_per_epoch": len(train_dataloader),
    }
    
    _ = [model[key].to(device) for key in model]
    _ = [model_ema[key].to(device) for key in model_ema]
    if finetune:
        logger.info("Finetune mode enabled: frozen generator.stem, generator.encode, asr_model, and f0_model")
        for key in model:
            logger.info("%-15s trainable params: %d" % (key, count_trainable_params(model[key])))
    scheduler_params_dict = {key: scheduler_params.copy() for key in model}
    scheduler_params_dict['mapping_network']['max_lr'] = 2e-6
    if 'rank_head' in scheduler_params_dict:
        scheduler_params_dict['rank_head']['max_lr'] = float(config['optimizer_params'].get('rank_lr', config['optimizer_params'].get('lr', 2e-4)))
    optimizer = build_optimizer({key: model[key].parameters() for key in model},
                                      scheduler_params_dict=scheduler_params_dict)

    trainer = Trainer(args=Munch(config['loss_params']), model=model,
                            model_ema=model_ema,
                            optimizer=optimizer,
                            device=device,
                            train_dataloader=train_dataloader,
                            val_dataloader=val_dataloader,
                            logger=logger,
                            fp16_run=fp16_run,
                            finetune=finetune)

    if config.get('pretrained_model', '') != '':
        trainer.load_checkpoint(config['pretrained_model'],
                                load_only_params=config.get('load_only_params', True))

    try:
        for _ in range(1, epochs+1):
            epoch = trainer.epochs
            train_results = trainer._train_epoch()
            # The legacy counter increments inside _train_epoch; label completed epochs.
            if seeded:
                epoch = trainer.epochs
            context = validation_rng(config['seed_protocol']['evaluation_seed'], val_dataloader) if seeded else nullcontext()
            with context:
                eval_results = trainer._eval_epoch()
            results = train_results.copy()
            results.update(eval_results)
            logger.info('--- epoch %d ---' % epoch)
            for key, value in results.items():
                if isinstance(value, numbers.Real):
                    logger.info('%-15s: %.4f' % (key, value))
                    writer.add_scalar(key, value, epoch)
                else:
                    for v in value:
                        writer.add_figure('eval_spec', v, epoch)
            validation_due = seeded and epoch % config['seed_protocol']['validate_every'] == 0
            if (epoch % save_freq) == 0 or validation_due or (seeded and epoch == epochs):
                checkpoint = osp.join(log_dir, 'epoch_%05d.pth' % epoch)
                trainer.save_checkpoint(checkpoint)
            if seeded:
                scalars = {key: value for key, value in results.items() if isinstance(value, numbers.Real)}
                if validation_due or epoch == epochs:
                    scalars.update(evaluate_checkpoint(config, checkpoint, epoch))
                tracker.log(epoch, scalars)

    finally:
        writer.close()
        if tracker:
            tracker.finish()

    return 0

def get_data_path_list(train_path=None, val_path=None):
    if train_path is None:
        train_path = "Data/train_list.txt"
    if val_path is None:
        val_path = "Data/val_list.txt"

    with open(train_path, 'r') as f:
        train_list = f.readlines()
    with open(val_path, 'r') as f:
        val_list = f.readlines()

    return train_list, val_list

if __name__=="__main__":
    main()
