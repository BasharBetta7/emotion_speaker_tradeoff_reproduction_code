"""Opt-in reproducibility and validation support for independent training runs."""
import contextlib
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def seed_everything(seed, deterministic=False):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


def seed_worker(worker_id):
    import numpy as np
    import torch
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


@contextlib.contextmanager
def validation_rng(seed, loader):
    """Validation must not advance training's Python/NumPy/Torch RNG streams."""
    import numpy as np
    import torch
    py_state, np_state = random.getstate(), np.random.get_state()
    loader_state = loader.generator.get_state() if loader.generator is not None else None
    with torch.random.fork_rng():
        try:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if loader.generator is not None:
                loader.generator.manual_seed(seed)
            yield
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)
            if loader_state is not None:
                loader.generator.set_state(loader_state)


def file_set(path, manifest=False):
    """Run from repository root, matching training/evaluator path semantics."""
    with open(path) as stream:
        if manifest:
            rows = list(csv.DictReader(stream, delimiter='\t'))
            if not rows or not {'sample_id', 'source_wav', 'reference_wav'} <= rows[0].keys():
                raise ValueError(f'Empty/invalid TSV manifest: {path}')
            ids = [row['sample_id'] for row in rows]
            if len(ids) != len(set(ids)):
                raise ValueError(f'Duplicate sample IDs: {path}')
            names = [row[k] for row in rows for k in ('source_wav', 'reference_wav')]
        else:
            names = [line.split('|')[0].strip() for line in stream if line.strip()]
    if not names:
        raise ValueError(f'Empty data list: {path}')
    return {str(Path(name).resolve()) for name in names}


def check_splits(config):
    protocol = config['seed_protocol']
    train = file_set(config['train_data'])
    val = file_set(config['val_data'])
    tests = set()
    for manifest in protocol['test_manifests'].values():
        tests |= file_set(manifest, True)
    validations = set()
    for manifest in protocol['validation_manifests'].values():
        validations |= file_set(manifest, True)
    comparisons = [('train / validation loss', train, val),
                   ('train / validation sweep', train, validations),
                   ('train / test', train, tests),
                   ('validation loss / test', val, tests),
                   ('validation sweep / test', validations, tests)]
    for label, left, right in comparisons:
        overlap = left & right
        if overlap:
            raise ValueError(f'{label}: {len(overlap)} overlapping audio paths; example: {sorted(overlap)[0]}')
    return {'train_wavs': len(train), 'validation_loss_wavs': len(val),
            'validation_sweep_wavs': len(validations), 'test_wavs': len(tests)}


def summarize_sweep(frames):
    """Sample-weighted metric means, integrated over alpha; no seed pooling."""
    import pandas as pd
    metrics = ['emotion_acc', 'emotion_conf', 'emotion_ref_cosine', 'speaker_cosine']
    data = pd.concat(frames, ignore_index=True)
    if data.duplicated(['alpha', 'sample_id']).any():
        raise ValueError('Duplicate alpha/sample rows')
    identities = [set(zip(g.sample_id, g.target_emotion)) for _, g in data.groupby('alpha')]
    if any(ids != identities[0] for ids in identities):
        raise ValueError('Alpha levels evaluated on different samples/emotions')
    if not identities or not identities[0]:
        raise ValueError('Empty sweep')
    curves, summary = [], {}
    groups = [('overall', data)] + [(str(k), v) for k, v in data.groupby('target_emotion')]
    for emotion, frame in groups:
        if frame[metrics].isna().any().any():
            raise ValueError('Missing metric values: refusing partial AUC')
        curve = frame.groupby('alpha')[metrics].mean().sort_index()
        xs = list(curve.index)
        if len(xs) < 2 or xs[0] != 0 or xs[-1] != 1:
            raise ValueError('AUC requires the complete alpha grid including 0 and 1')
        for metric in metrics:
            ys = list(curve[metric])
            if not all(math.isfinite(y) for y in ys):
                raise ValueError('Nonfinite metric values')
            auc = sum((xs[i+1]-xs[i])*(ys[i+1]+ys[i])/2 for i in range(len(xs)-1))
            prefix = f'{emotion}/{metric}'
            summary.update({f'{prefix}/auc_alpha': auc, f'{prefix}/alpha0': ys[0],
                            f'{prefix}/alpha1': ys[-1], f'{prefix}/delta': ys[-1]-ys[0]})
        out = curve.reset_index()
        out['target_emotion'] = emotion
        curves.append(out)
    return pd.concat(curves, ignore_index=True), summary


def evaluate_checkpoint(config, checkpoint, epoch, split='validation', manifests=None):
    import pandas as pd
    protocol = config['seed_protocol']
    mode = protocol.get('intensity_mode', 'inferred')
    control = protocol.get('intensity_control', 'style')
    if mode not in ('inferred', 'native') or control != 'style':
        raise ValueError('Seed study supports inferred/native style interpolation only')
    manifests = manifests or protocol[f'{split}_manifests']
    root = Path(config['log_dir']) / split / f'epoch_{epoch:05d}'
    root.mkdir(parents=True, exist_ok=False)
    result = {}
    evaluator = Path(__file__).resolve().parent / 'scripts/run_balanced_benchmark_eval.py'
    wrapper = Path(__file__).resolve().parent / 'scripts/seeded_evaluate.py'
    for condition, manifest in manifests.items():
        frames = []
        for alpha in protocol['alphas']:
            output = root / condition / f'alpha_{alpha:g}'
            cmd = [sys.executable, str(wrapper), '--seed', str(protocol['evaluation_seed']),
                   '--script', str(evaluator), '--', '--config-path', config['_resolved_config'],
                   '--checkpoint-path', str(checkpoint), '--manifest-path', manifest,
                   '--output-dir', str(output), '--model-variant', f'{mode}_seed_{config["seed"]}',
                   '--intensity-mode', mode, '--intensity-control', control, '--alpha', str(alpha),
                   '--device', config.get('device', 'cuda')]
            for key in ('speaker_model_dir', 'emotion_model_dir', 'vocoder_path'):
                if key in protocol:
                    cmd += ['--' + key.replace('_', '-'), protocol[key]]
            env = dict(os.environ, PYTHONHASHSEED=str(protocol['evaluation_seed']))
            env['PYTHONPATH'] = str(evaluator.parent.parent) + os.pathsep + env.get('PYTHONPATH', '')
            subprocess.run(cmd, check=True, env=env)
            frame = pd.read_csv(output / 'per_sample_metrics.csv')
            if set(frame.alpha) != {alpha}:
                raise ValueError('Evaluator returned unexpected alpha')
            frames.append(frame)
        curve, summary = summarize_sweep(frames)
        curve.to_csv(root / condition / 'frontier.csv', index=False)
        result.update({f'{split}/{condition}/{k}': v for k, v in summary.items()})
    record = {'seed': config['seed'], 'epoch': epoch, 'checkpoint': str(checkpoint),
              'checkpoint_sha256': sha256(checkpoint), 'metrics': result}
    (root / 'summary.json').write_text(json.dumps(record, indent=2))
    return result


class SeedTracker:
    def __init__(self, config):
        self.config = config
        self.root = Path(config['log_dir'])
        self.run = None
        cfg = config.get('wandb', {})
        if cfg.get('mode', 'disabled') != 'disabled':
            import wandb
            self.run = wandb.init(project=cfg['project'], entity=cfg.get('entity'),
                                  group=cfg['group'], name=f'{cfg["group"]}-seed-{config["seed"]}',
                                  config=config, mode=cfg['mode'], dir=str(self.root),
                                  job_type='train')
            self.run.define_metric('epoch')
            self.run.define_metric('*', step_metric='epoch')

    def log(self, epoch, values):
        record = {'epoch': epoch, **{k: float(v) for k, v in values.items()}}
        with (self.root / 'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(record, allow_nan=True) + '\n')
        if self.run:
            self.run.log(record)

    def finish(self):
        if self.run:
            self.run.finish()
