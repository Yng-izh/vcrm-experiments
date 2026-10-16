"""Immutable, label-free ICAE event features shared by all compression methods."""
from pathlib import Path

from .data import ingestion_view
from .io import digest, file_digest, read_json, row_path, write_json


def identity(config):
    return {'checkpoint_sha256': file_digest(config['checkpoint']),
            'adapted_sha256': file_digest(config['adapted_checkpoint']) if config.get('adapted_checkpoint') else None,
            'model_revision': config['model_revision'], 'reader_sha256': file_digest(Path(__file__).with_name('reader.py')),
            'source_sha256': config['icae_source_sha256'], 'event_format': 'timestamped_event_v1'}


def load(config, split, episode_id, event):
    import torch
    from safetensors.torch import load_file
    base = Path(config['feature_root']) / split
    path = row_path(base, 'events', episode_id + ':' + event['id']).with_suffix('.safetensors')
    metadata = read_json(path.with_suffix('.json'))
    if metadata['event_sha256'] != digest(event) or metadata['tensor_sha256'] != file_digest(path):
        raise ValueError('Feature content or event identity changed')
    value = load_file(str(path))['memory']
    if (tuple(value.shape) != (128, 4096) or value.dtype != torch.float16 or
            not torch.isfinite(value).all()):
        raise ValueError('This experiment requires exactly 128 native ICAE slots per fact')
    return value


def check(config, split, complete=True):
    manifest = read_json(Path(config['feature_root']) / split / 'manifest.json')
    if manifest['reader_identity'] != identity(config):
        raise ValueError('Features were encoded by a different reader')
    episodes = read_json(Path(config['data_root']) / f'prepared/templama_{split}.json')
    if manifest['data_sha256'] != digest(episodes):
        raise ValueError('Feature data split changed')
    if complete and not manifest.get('complete'):
        raise RuntimeError('Feature cache is incomplete')
    return manifest


def prepare(config, split):
    episodes = read_json(Path(config['data_root']) / f'prepared/templama_{split}.json')
    base = Path(config['feature_root']) / split
    result = {'reader_identity': identity(config), 'data_sha256': digest(episodes), 'split': split,
              'episodes': len(episodes), 'expected_events': sum(len(e['events']) for e in episodes),
              'complete': False}
    path = base / 'manifest.json'
    if path.exists():
        previous = read_json(path)
        if any(previous[k] != result[k] for k in ('reader_identity', 'data_sha256', 'split')):
            raise ValueError('Do not overwrite feature cache from another protocol')
    else:
        write_json(path, result)
    return result


def encode(config, split, shard=0, shards=1):
    from .reader import FrozenReader
    reader = FrozenReader(config)
    episodes = read_json(Path(config['data_root']) / f'prepared/templama_{split}.json')[shard::shards]
    base = Path(config['feature_root']) / split
    for i, episode in enumerate(episodes):
        for event in ingestion_view(episode):
            path = row_path(base, 'events', episode['id'] + ':' + event['id']).with_suffix('.safetensors')
            if path.exists() and path.with_suffix('.json').exists():
                load(config, split, episode['id'], event)
                continue
            memory = reader.encode_event(event)
            save(config, split, episode['id'], event, memory)
        print(f'features {split} shard={shard} {i+1}/{len(episodes)}', flush=True)


def save(config, split, episode_id, event, memory):
    import torch
    from .training import save_weights
    if tuple(memory.shape) != (128, 4096) or not torch.isfinite(memory.half()).all():
        raise ValueError('Fact needs more than 128 native slots; do not silently truncate')
    base = Path(config['feature_root']) / split
    path = row_path(base, 'events', episode_id + ':' + event['id']).with_suffix('.safetensors')
    if path.exists() and path.with_suffix('.json').exists():
        if not torch.equal(load(config, split, episode_id, event), memory.cpu().half()):
            raise ValueError('Attempt to change an immutable cached feature')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    save_weights(path, {'memory': memory.cpu().half()})
    write_json(path.with_suffix('.json'), {'episode_id': episode_id, 'event_id': event['id'],
        'event_sha256': digest(event), 'tensor_sha256': file_digest(path), 'shape': list(memory.shape),
        'payload_bytes': memory.numel() * 2, 'raw_history_saved': False})


def finalize(config, split):
    manifest = check(config, split, False)
    episodes = read_json(Path(config['data_root']) / f'prepared/templama_{split}.json')
    for episode in episodes:
        for event in ingestion_view(episode):
            load(config, split, episode['id'], event)
    manifest['complete'] = True
    write_json(Path(config['feature_root']) / split / 'manifest.json', manifest)
    return manifest
