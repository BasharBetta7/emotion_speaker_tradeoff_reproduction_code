"""Explicit, versioned speaker identities; omitted mode preserves legacy ESD parsing."""
from pathlib import Path
import re


def speaker_key(path, mode='legacy'):
    p = Path(path)
    if mode in ('legacy', 'esd'):
        return p.parent.parent.name
    if mode == 'cremad':
        if not re.fullmatch(r'\d{4}_[A-Z]{3}_(ANG|DIS|FEA|HAP|NEU|SAD)_[A-Z]{2}', p.stem):
            raise ValueError('Expected CREMA-D filename <speaker>_<sentence>_<emotion>_<level>.wav: ' + str(path))
        return p.stem.split('_', 1)[0]
    raise ValueError('Unknown speaker_id_mode: ' + str(mode))


def training_speaker_map(lines, mode):
    keys = sorted({speaker_key(line.strip().split('|')[0], mode) for line in lines if line.strip()})
    if not keys:
        raise ValueError('Cannot build speaker map from empty training split')
    if mode == 'cremad' and len(keys) < 2:
        raise ValueError('CREMA-D speaker training needs at least two distinct speakers')
    return {key: i for i, key in enumerate(keys)}


def validate_speaker_map(mapping):
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError('speaker_to_index must be a nonempty mapping')
    if any(not isinstance(k, str) or type(v) is not int for k, v in mapping.items()):
        raise ValueError('Speaker map requires string keys and integer indices')
    if sorted(mapping.values()) != list(range(len(mapping))):
        raise ValueError('Speaker indices must be unique and contiguous from zero')
    return dict(mapping)


def configure_speakers(config, lines, prepare=False):
    """Freeze new-run mapping; runtime validates instead of resizing old checkpoints."""
    params = config.setdefault('dataset_params', {})
    mode = params.get('speaker_id_mode', 'legacy')
    if mode == 'legacy':
        return None
    if mode not in ('cremad', 'esd'):
        raise ValueError('Unknown speaker_id_mode: ' + str(mode))
    mapping = training_speaker_map(lines, mode)
    old = params.get('speaker_to_index')
    if old is not None and validate_speaker_map(old) != mapping:
        raise ValueError('Training speakers differ from the frozen speaker mapping')
    model = config['model_params']
    if model.get('spk_cls_enable', False):
        n = len(mapping)
        for key in ('n_speakers', 'num_speakers'):
            if key in model and model[key] != n and (not prepare or config.get('pretrained_model')):
                raise ValueError('Classifier size differs from training speakers; prepare a fresh corrected protocol, do not resize an existing checkpoint')
        model['num_speakers'] = n
        if 'n_speakers' in model:
            model['n_speakers'] = n
    params['speaker_to_index'] = mapping
    return mapping
