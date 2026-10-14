from pathlib import Path
import random
import time

import numpy as np
import torch

from .codec import VersionCodec
from .io import atomic_bytes, digest, read_json, write_json
from .reader import FrozenReader


def extract_features(config, split, limit=None):
    episodes = read_json(Path(config['data_root']) / 'prepared' / f'templama_{split}.json')
    if limit:
        episodes = episodes[:limit]
    reader = FrozenReader(config)
    root = Path(config['data_root']) / 'features' / split
    root.mkdir(parents=True, exist_ok=True)
    for i, episode in enumerate(episodes):
        path = root / (digest(episode['id']) + '.pt')
        if path.exists():
            row = torch.load(path, map_location='cpu', weights_only=False)
            if row['config_signature'] != feature_signature(config) or row.get('episode_sha256') != digest(episode):
                raise ValueError('Existing feature cache uses a different model/encoding configuration')
            continue
        started = time.perf_counter()
        features = torch.stack([reader.encode_event(event) for event in episode['events']])
        temp = path.with_suffix('.tmp')
        torch.save({'episode': episode, 'features': features, 'config_signature': feature_signature(config),
                    'episode_sha256': digest(episode)}, temp)
        temp.replace(path)
        print(f'features {split} {i + 1}/{len(episodes)} {episode["id"]}: {time.perf_counter() - started:.1f}s', flush=True)


def feature_signature(config):
    return digest({k: config[k] for k in ('model_path', 'checkpoint', 'dtype', 'attention', 'encode_chunk_tokens', 'event_format')})


def train(config, method, destination, limit=None):
    if method not in ('independent', 'vcrm'):
        raise ValueError('Only C and D have trainable codecs')
    torch.manual_seed(config['seed'])
    random.seed(config['seed'])
    paths = sorted((Path(config['data_root']) / 'features/train').glob('*.pt'))
    if not paths:
        raise RuntimeError('Extract official TempLAMA train features first')
    official = read_json(Path(config['data_root']) / 'prepared/templama_train.json')
    official_hashes = {e['id']: digest(e) for e in official}
    if limit is None and {p.stem for p in paths} != {digest(e['id']) for e in official}:
        raise RuntimeError('Formal training requires features for the complete official train split')
    if limit:
        paths = paths[:limit]
    reader = FrozenReader(config)
    codec = VersionCodec(reader.shape[-1], config['rank'], method == 'vcrm').to(reader.device).float()
    optimizer = torch.optim.AdamW(codec.parameters(), lr=config['learning_rate'])
    scaler = torch.amp.GradScaler('cuda', init_scale=1.)
    steps, history = 0, []
    for epoch in range(config['epochs']):
        order = list(paths)
        random.Random(config['seed'] + epoch).shuffle(order)
        totals = {'historical': [], 'latest': []}
        for path in order:
            row = torch.load(path, map_location='cpu', weights_only=False)
            episode = row['episode']
            if episode['split'] != 'train' or row['config_signature'] != feature_signature(config):
                raise ValueError('Feature mismatch or non-train supervision')
            if row.get('episode_sha256') != official_hashes[episode['id']]:
                raise ValueError('Training feature episode differs from current prepared official data')
            features = row['features'].to(reader.device).float()
            query_by_time = {q['time']: q for q in episode['queries']}
            # Every prefix is an update state; pair old facts with the then-current fact.
            for current_index in range(1, len(features)):
                old_index = random.Random(f'{config["seed"]}:{epoch}:{episode["id"]}:{current_index}').randrange(current_index)
                for kind, target_index in (('historical', old_index), ('latest', current_index)):
                    old, current = features[target_index], features[current_index]
                    optimizer.zero_grad(set_to_none=True)
                    # Simulate FP16 persistent codes with a straight-through rounding path.
                    code = codec.encode(old, current)
                    rounded = code + (code.half().float() - code).detach()
                    reconstructed = codec.decode(rounded, current)
                    reconstruction = torch.nn.functional.mse_loss(reconstructed, old)
                    query = query_by_time[episode['events'][target_index]['time']]
                    qa = reader.qa_loss(query, query['answers'][0], reconstructed)
                    loss = config['reconstruction_weight'] * reconstruction + config['qa_weight'] * qa
                    if not torch.isfinite(loss):
                        raise ValueError('Non-finite training loss')
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(codec.parameters(), 1., error_if_nonfinite=True)
                    scale_before = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    if scaler.get_scale() < scale_before:
                        raise RuntimeError('Skipped FP16 optimizer update; cannot claim matched C/D training steps')
                    totals[kind].append(float(loss.detach()))
                    steps += 1
                print(f'{method}: epoch {epoch + 1}, step {steps}, {episode["id"]}, prefix {current_index + 1}', flush=True)
        history.append({k: float(np.mean(v)) if v else None for k, v in totals.items()})
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp = destination.with_suffix('.tmp')
        torch.save({'method': method, 'dimensions': reader.shape[-1], 'rank': config['rank'],
                    'training_split': 'train', 'training_ids': [p.stem for p in paths],
                    'smoke_only': limit is not None, 'state_dict': codec.cpu().state_dict(),
                    'steps': steps, 'epochs': epoch + 1, 'loss_history': history,
                    'trainable_parameters': sum(p.numel() for p in codec.parameters()),
                    'training_data_sha256': digest([(e['id'], official_hashes[e['id']]) for e in official]),
                    'feature_signature': feature_signature(config), 'config': config}, temp)
        temp.replace(destination)
        codec.to(reader.device)
    write_json(destination.with_suffix('.metrics.json'), {'method': method, 'steps': steps, 'loss_history': history,
               'latest_auxiliary': 'Latest self-reconstruction QA auxiliary; deployment stores the current anchor without compression.'})
