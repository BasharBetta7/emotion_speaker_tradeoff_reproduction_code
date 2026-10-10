"""Prepare, validate, and optionally execute independent inferred-intensity runs."""
import argparse
import copy
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from seed_support import check_splits, evaluate_checkpoint, sha256

BASES = {
    'esd': 'Configs/config_emo.yml',
    'cremad-from-scratch': 'Configs/config_emo_inferred_intensity_cremad_pretrained.yml',
    'cremad-finetuned': 'Configs/config_emo_inferred_intensity_cremad_finetuned copy.yml',
}


NATIVE_BASES = {
    'esd': 'Configs/config_emo_native_intensity.yml',
    'cremad-from-scratch': 'Configs/config_emo_intensity_rank_cremad.yml',
    'cremad-finetuned': 'Configs/config_emo_intensity_rank_cremad.yml',
}


def build_configs(protocol, root=ROOT, variant='inferred'):
    if variant not in ('inferred', 'native'):
        raise ValueError('Unknown model variant')
    seeds = protocol['seeds']
    if not seeds or len(seeds) != len(set(seeds)) or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds):
        raise ValueError('Seeds must be distinct integers in [0, 2**32)')
    setting = protocol['setting']
    name = protocol['experiment_name']
    if not isinstance(name, str) or not re.fullmatch(r'[a-zA-Z0-9_-]+', name):
        raise ValueError('Set a unique experiment_name using letters, numbers, _ or -')
    for key in ('epochs', 'validate_every', 'save_freq'):
        if type(protocol.get(key)) is not int or protocol[key] < 1:
            raise ValueError(f'{key} requires an explicitly chosen positive integer')
    if type(protocol.get('evaluation_seed')) is not int or not 0 <= protocol['evaluation_seed'] < 2**32:
        raise ValueError('Choose evaluation_seed in [0, 2**32)')
    if type(protocol.get('deterministic')) is not bool:
        raise ValueError('Choose deterministic: true or false')
    if not isinstance(protocol.get('selection_rule'), str) or not protocol['selection_rule'].strip():
        raise ValueError('Document the approved checkpoint selection rule before training')
    alphas = protocol['alphas']
    if len(alphas) < 2 or sorted(set(alphas)) != alphas or alphas[0] != 0 or alphas[-1] != 1:
        raise ValueError('alphas must be strictly increasing from 0 to 1')
    for field in ('validation_manifests', 'test_manifests'):
        if not protocol.get(field) or any(not v for v in protocol[field].values()):
            raise ValueError(f'Provide approved {field}')
        if any(not re.fullmatch(r'[a-zA-Z0-9_-]+', k) for k in protocol[field]):
            raise ValueError('Condition names must be simple path-safe labels')
    if set(protocol['validation_manifests']) != set(protocol['test_manifests']):
        raise ValueError('Validation and test condition names must match')
    if not protocol.get('train_data') or not protocol.get('val_data'):
        raise ValueError('Provide approved train_data and val_data; base split defaults are unsafe')
    wb = protocol.get('wandb', {})
    if wb.get('mode', 'disabled') not in ('disabled', 'offline', 'online'):
        raise ValueError('Invalid W&B mode')
    defaults = NATIVE_BASES if variant == 'native' else BASES
    if setting not in defaults:
        raise ValueError(f'Unknown experimental setting: {setting}')
    base_config = protocol.get('base_config', defaults[setting])
    if not isinstance(base_config, str) or not base_config.strip():
        raise ValueError('base_config must be a nonempty YAML file path')
    # Relative paths resolve against the repository, as do the default bases.
    # Preserve this path in input_sha256 below for experiment provenance.
    base_path = root / base_config
    if not base_path.is_file():
        raise FileNotFoundError(f'Base configuration not found: {base_path}')
    base = yaml.safe_load(base_path.read_text())
    rank_enabled = base['model_params'].get('rank_enable', base['model_params'].get('use_rank_intensity', False))
    if bool(rank_enabled) != (variant == 'native'):
        raise ValueError(f'Rank supervision does not match requested {variant} variant')
    configs = []
    for seed in seeds:
        cfg = copy.deepcopy(base)
        for field in ('epochs', 'save_freq', 'train_data', 'val_data'):
            cfg[field] = protocol[field]
        cfg.update(seed=seed, log_dir=str(root / 'Models/seed_retraining' / name / f'seed_{seed}'),
                   pretrained_model='', load_only_params=False, finetune=False)
        if setting == 'cremad-finetuned':
            initial = protocol.get('initial_checkpoints', {}).get(seed)
            if not initial or not Path(initial).is_file():
                raise ValueError(f'Provide the approved ESD initialization checkpoint for seed {seed}')
            cfg.update(pretrained_model=str(Path(initial).resolve()), load_only_params=True, finetune=True)
        if protocol.get('speaker_id_mode') is not None:
            cfg.setdefault('dataset_params', {})['speaker_id_mode'] = protocol['speaker_id_mode']
        if cfg.get('dataset_params', {}).get('speaker_id_mode', 'legacy') != 'legacy':
            from speaker_labels import configure_speakers
            configure_speakers(cfg, (root / cfg['train_data']).read_text().splitlines(), prepare=True)
        cfg['seed_protocol'] = {k: protocol[k] for k in ('validate_every', 'evaluation_seed',
            'deterministic', 'alphas', 'validation_manifests', 'test_manifests')}
        for field in ('speaker_model_dir', 'emotion_model_dir', 'vocoder_path'):
            if protocol.get(field):
                cfg['seed_protocol'][field] = str(Path(protocol[field]).resolve())
        if variant == 'native':
            cfg['seed_protocol']['intensity_mode'] = 'native'
            cfg['seed_protocol']['intensity_control'] = 'style'
        cfg['wandb'] = dict(wb, group=name)
        cfg['selection_rule'] = protocol['selection_rule']
        if cfg['wandb'].get('mode', 'disabled') != 'disabled' and not wb.get('project'):
            raise ValueError('W&B project is required when enabled')
        paths = [base_path, Path(cfg['train_data']), Path(cfg['val_data'])]
        paths += [Path(p) for field in ('validation_manifests', 'test_manifests') for p in protocol[field].values()]
        if cfg['pretrained_model']:
            paths.append(Path(cfg['pretrained_model']))
        cfg['input_sha256'] = {str(p.resolve()): sha256(p) for p in paths}
        cfg['code_sha256'] = {name: sha256(root/name) for name in
            ('train.py', 'trainer.py', 'meldataset.py', 'models.py', 'losses.py',
             'optimizers.py', 'seed_support.py', 'scripts/run_balanced_benchmark_eval.py',
             'scripts/seeded_evaluate.py', 'scripts/run_inferred_seeds.py') if (root/name).exists()}
        if variant == 'native' and (root/'scripts/run_native_seeds.py').exists():
            cfg['code_sha256']['scripts/run_native_seeds.py'] = sha256(root/'scripts/run_native_seeds.py')
        if cfg.get('dataset_params', {}).get('speaker_id_mode', 'legacy') != 'legacy':
            cfg['code_sha256']['speaker_labels.py'] = sha256(root/'speaker_labels.py')
        cfg['split_audit'] = check_splits(cfg)
        configs.append(cfg)
    return configs


def main(config_builder=build_configs, description=__doc__):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--execute', action='store_true', help='Launch training after preflight; otherwise prepare only')
    parser.add_argument('--test-selections', help='JSON with rule description and one selected epoch per seed')
    args = parser.parse_args()
    os.chdir(ROOT)
    protocol = yaml.safe_load(Path(args.protocol).read_text())
    configs = config_builder(protocol)
    # Configs must be directly inside Configs/: legacy evaluator infers repo root from this location.
    config_paths = [ROOT / 'Configs' / f'seeds_{protocol["experiment_name"]}_{c["seed"]}.yml' for c in configs]
    for cfg, path in zip(configs, config_paths):
        text = yaml.safe_dump(cfg, sort_keys=False)
        if path.exists() and path.read_text() != text:
            raise FileExistsError(f'Configuration changed; use a new experiment_name: {path}')
    if args.test_selections:
        if args.execute:
            raise ValueError('Training and final test must be separate invocations')
        selections = json.loads(Path(args.test_selections).read_text())
        if selections.get('rule') != protocol['selection_rule']:
            raise ValueError('Selection rule differs from the rule registered before training')
        if not selections.get('rule') or set(selections['epochs']) != {str(c['seed']) for c in configs}:
            raise ValueError('Supply a documented selection rule and exactly one epoch for every seed')
        for cfg, path in zip(configs, config_paths):
            if not path.exists():
                raise FileNotFoundError('Prepare and train these configs before testing')
            epoch = selections['epochs'][str(cfg['seed'])]
            if type(epoch) is not int or not 1 <= epoch <= cfg['epochs']:
                raise ValueError('Invalid selected epoch')
            ckpt = Path(cfg['log_dir']) / f'epoch_{epoch:05d}.pth'
            if not ckpt.exists() or not (Path(cfg['log_dir'])/'validation'/f'epoch_{epoch:05d}'/'summary.json').exists():
                raise ValueError(f'Selected checkpoint lacks completed validation: {ckpt}')
            if (Path(cfg['log_dir'])/'test-selection.json').exists():
                raise FileExistsError('Final test already selected for this run')
        for cfg, path in zip(configs, config_paths):
            epoch = selections['epochs'][str(cfg['seed'])]
            cfg['_resolved_config'] = str(path)
            (Path(cfg['log_dir'])/'test-selection.json').write_text(json.dumps(selections, indent=2))
            evaluate_checkpoint(cfg, Path(cfg['log_dir'])/f'epoch_{epoch:05d}.pth', epoch, 'test')
        return
    for cfg, path in zip(configs, config_paths):
        log = Path(cfg['log_dir'])
        if args.execute and log.exists() and any(log.iterdir()):
            raise FileExistsError(f'Existing seed run: {log}')
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        print(f'Prepared seed {cfg["seed"]}: {path}')
    if args.execute:
        for cfg, path in zip(configs, config_paths):
            env = dict(os.environ, PYTHONHASHSEED=str(cfg['seed']))
            if cfg['seed_protocol']['deterministic']:
                env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            subprocess.run([sys.executable, 'train.py', '--config_path', str(path)], env=env, check=True)


if __name__ == '__main__':
    main()
