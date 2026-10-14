"""Stage-one ICAE encoder adaptation. No history codec is trained here."""
import os
from pathlib import Path
import random
import time

import torch

from .data import ingestion_view, query_view
from .io import digest, file_digest, read_json, write_json
from .reader import FrozenReader


def adaptation_samples(episodes):
    result = []
    for episode in episodes:
        if episode['split'] != 'train' or episode['dataset'] != 'templama':
            raise ValueError('Only official-source filtered TempLAMA training facts may adapt this encoder')
        events = ingestion_view(episode)
        for query in episode['queries']:
            eligible = [e for e in events if e['time'] <= query['as_of']]
            if not eligible:
                raise ValueError('Missing observed version for training query')
            result.append((episode['id'], eligible[-1], query_view(query), query['answers'][0]))
    return result


def adapt_encoder(config, root, limit=None):
    from safetensors.torch import save_file
    episodes = read_json(Path(config['data_root']) / 'prepared/templama_train.json')
    if limit:
        episodes = episodes[:limit]
    samples = adaptation_samples(episodes)
    reader = FrozenReader(config)
    parameters = reader.enable_encoder_adaptation()
    optimizer = torch.optim.AdamW(parameters, lr=config['learning_rate'], weight_decay=0.)
    scaler = torch.amp.GradScaler('cuda')
    rng = random.Random(config['seed'])
    root = Path(root)
    checkpoint = root / 'encoder.safetensors'
    if checkpoint.exists():
        raise RuntimeError('Adaptation is not optimizer-resumable; use a new run name, not a silent restart')
    started, steps, completed = time.perf_counter(), 0, 0
    accumulation = config['gradient_accumulation']
    task_embeddings = reader.net.memory_token_embed.weight[-3:].detach().clone()
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for epoch in range(config['epochs']):
        order = list(range(len(samples)))
        rng.shuffle(order)
        for index, sample_index in enumerate(order):
            _identity, event, query, answer = samples[sample_index]
            memory = reader.encode_event(event, training=True)
            loss = reader.qa_loss(query, answer, memory)
            if not torch.isfinite(loss):
                raise ValueError('Non-finite historical/current answer loss')
            # Normalize the final partial accumulation group by its actual sample count.
            group_start = index // accumulation * accumulation
            group_size = min(accumulation, len(order) - group_start)
            scaler.scale(loss / group_size).backward()
            losses.append(float(loss.detach()))
            completed += 1
            if (index + 1) % accumulation == 0 or index + 1 == len(order):
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(parameters, config['gradient_clip'])
                if not torch.isfinite(norm):
                    raise ValueError('Non-finite encoder gradients; do not skip bad steps silently')
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                steps += 1
                write_json(root / 'progress.json', {'epoch': epoch, 'samples': completed, 'steps': steps,
                           'loss': sum(losses[-accumulation:]) / min(len(losses), accumulation)})
                print(f'epoch={epoch+1} sample={index+1}/{len(order)} step={steps} loss={float(loss):.4f}', flush=True)
        if not torch.equal(task_embeddings, reader.net.memory_token_embed.weight[-3:].detach()):
            raise AssertionError('Frozen decoder task embeddings changed')
        temporary = root / '.encoder.partial.safetensors'
        save_file(reader.encoder_state(), str(temporary))
        os.replace(temporary, checkpoint)
    write_json(root / 'adaptation.json', {
        'completed': True, 'scope': 'smoke' if limit else 'full_training_split',
        'checkpoint': str(checkpoint), 'checkpoint_sha256': file_digest(checkpoint),
        'base_checkpoint_sha256': file_digest(config['checkpoint']), 'model_revision': config['model_revision'],
        'training_data_sha256': digest(episodes), 'training_ids': [e['id'] for e in episodes],
        'training_samples': len(samples), 'processed_samples': completed, 'epochs': config['epochs'],
        'optimizer_steps': steps, 'trainable_parameters': sum(p.numel() for p in parameters),
        'loss': 'QA answer cross-entropy only; first published valid answer, no hidden-state MSE',
        'decoder_frozen': True, 'codec_used': False, 'seconds': time.perf_counter() - started,
        'split_protocol': 'Published TempLAMA train, entity-overlap filtered; not the unchanged official split'})
    return {'checkpoint': str(checkpoint), 'steps': steps, 'scope': 'smoke' if limit else 'full_training_split'}
