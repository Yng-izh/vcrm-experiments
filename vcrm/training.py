"""Memory-only adaptation and paired temporal slot-codec training (opt-in)."""
import os
import json
import math
from pathlib import Path
import random
import time

import torch
from torch.nn import functional as F

from .data import ingestion_view, query_view
from .io import digest, file_digest, read_json, write_json
from .metrics import normalize


def adaptation_samples(episodes):
    result = []
    for episode in episodes:
        if episode['split'] != 'train' or episode['dataset'] != 'templama':
            raise ValueError('Only filtered training facts may train adaptation or codecs')
        events = ingestion_view(episode)
        for query in episode['queries']:
            eligible = [e for e in events if e['time'] <= query['as_of']]
            if not eligible:
                raise ValueError('Missing observed version for training query')
            result.append((episode['id'], eligible[-1], query_view(query), query['answers'][0]))
    return result


def paragraph_samples(episodes, seed=42):
    samples = adaptation_samples(episodes)
    facts = [(e['id'], event) for e in episodes for event in ingestion_view(e)]
    rng = random.Random(seed)
    output = []
    for eid, event, query, answer in samples:
        candidates = [(group, fact) for group, fact in facts if group != eid]
        background = rng.sample(candidates, min(3, len(candidates)))
        events = [event] + [fact for _, fact in background]
        rng.shuffle(events)
        text = 'The following observations describe facts recorded at different times.\n\n' + '\n\n'.join(
            f"An observation dated {e['time'][:10]} records the following: {e['text']}" for e in events)
        output.append((eid, text, query, answer))
    return output


def temporal_pairs(episodes, interval=5, protocol='immutable_keyframe_v1'):
    adaptation_samples(episodes)  # Reject evaluation data before using any labels.
    if protocol == 'final_current_snapshot_v1':
        return snapshot_pairs(episodes)
    if protocol != 'immutable_keyframe_v1':
        raise ValueError('Unknown temporal-pair protocol')
    if interval < 2:
        raise ValueError('Invalid keyframe interval')
    pairs = []
    for episode in episodes:
        events = ingestion_view(episode)
        for current_index in range(2, len(events)):
            old_index = current_index - 1
            if old_index % interval == 0:
                continue  # Keyframes remain full, not codec targets.
            old, current = events[old_index], events[current_index]
            anchor = events[old_index // interval * interval]
            historical = next(q for q in episode['queries'] if q['time'] == old['time'])
            latest = next(q for q in episode['queries'] if q['time'] == current['time'])
            # Shared valid answers are not negative versions.
            negatives = [a for a in latest['answers'] if normalize(a) not in {normalize(v) for v in historical['answers']}]
            pairs.append({'id': episode['id'] + ':' + current['id'], 'state_key': old['state_key'],
                          'old': old, 'current': current, 'anchor': anchor,
                          'historical': historical, 'latest': latest, 'negative_answers': negatives})
    return pairs


def snapshot_pairs(episodes):
    """One existing QA per historical version, including initial/keyframe versions."""
    pairs = []
    for episode in episodes:
        events = ingestion_view(episode)
        if len(events) < 2:
            continue
        if any(a['time'] >= b['time'] for a, b in zip(events, events[1:])):
            raise ValueError('Snapshot training needs strictly chronological observed versions')
        latest_queries = [q for q in episode['queries'] if q['latest']]
        if len(latest_queries) != 1:
            raise ValueError('Snapshot requires exactly one final-current query')
        latest, current = latest_queries[0], events[-1]
        observed = max((e for e in events if e['time'] <= latest['as_of']), key=lambda e: e['time'])
        if observed['id'] != current['id']:
            raise ValueError('Final-current QA must select the final observed version')
        historical_by_event = {}
        for q in episode['queries']:
            if q['latest']:
                continue
            eligible = [e for e in events if e['time'] <= q['as_of']]
            if not eligible:
                raise ValueError('Historical QA has no observed version')
            selected = eligible[-1]
            if selected['id'] != current['id']:
                historical_by_event.setdefault(selected['id'], []).append(q)
        for index, old in enumerate(events[:-1]):
            queries = historical_by_event.get(old['id'], [])
            if not queries:
                continue  # Only versions with public training QA can supervise the codec.
            historical = min(queries, key=lambda q: (q['time'] != old['time'], q['time'], q['id']))
            valid = {normalize(a) for a in historical['answers']}
            negatives = list(dict.fromkeys(a for a in latest['answers'] if normalize(a) not in valid))
            # Retain old pair IDs so the fixed 200-query diagnostic stays identical.
            pairs.append({'id': episode['id'] + ':' + events[index + 1]['id'],
                          'state_key': old['state_key'], 'old': old, 'current': current, 'anchor': current,
                          'historical': historical, 'latest': latest, 'negative_answers': negatives})
    return pairs


@torch.no_grad()
def matched_teacher_bundle(markers, current, old, pair):
    # Same train-time marker weights and ordering as student; no teacher gradients.
    return torch.cat((markers(current, 'current', pair['current']['time']),
                      markers(old, 'historical', pair['old']['time']))).detach()


def distillation_loss(student, teacher, target, temperature=2.):
    if student.shape != teacher.shape or student.shape[:-1] != target.shape or temperature <= 0:
        raise ValueError('Teacher/student must share answer-prefix positions and vocabulary')
    ce = F.cross_entropy(student.reshape(-1, student.shape[-1]), target.reshape(-1))
    kl = F.kl_div(F.log_softmax(student.float() / temperature, dim=-1),
                  F.softmax(teacher.detach().float() / temperature, dim=-1), reduction='none').sum(-1).mean()
    return ce, kl * temperature ** 2


def sequence_unlikelihood(logits, target):
    # Penalize the complete wrong-answer sequence, not each word shared by versions.
    probabilities = F.log_softmax(logits.float(), dim=-1).gather(-1, target.unsqueeze(-1)).squeeze(-1)
    log_probability = probabilities.sum(-1)
    return -torch.log1p(-log_probability.exp().clamp(max=1. - 1e-6)).mean()


def mean_answer_logprob(logits, target):
    # Reader targets end in EOS; compare answer text tokens, excluding EOS and length bias.
    if logits.shape[:-1] != target.shape or target.shape[-1] < 2:
        raise ValueError('Contrastive answers need at least one text token plus EOS')
    values = F.log_softmax(logits.float()[..., :-1, :], dim=-1)
    return values.gather(-1, target[..., :-1].unsqueeze(-1)).squeeze(-1).mean()


def contrastive_margin(correct, wrong, margin=1.5):
    if margin <= 0:
        raise ValueError('Contrastive margin must be positive')
    return F.relu(margin + wrong - correct)


def cosine_warmup_factor(step, total, warmup):
    if not 0 <= warmup < total or total < 1:
        raise ValueError('Invalid cosine schedule')
    if step < warmup:
        return (step + 1) / max(1, warmup)
    return .5 * (1. + math.cos(math.pi * min(1., (step - warmup) / (total - warmup))))


def train_only(config, limit):
    episodes = read_json(Path(config['data_root']) / 'prepared/templama_train.json')
    return episodes[:limit] if limit else episodes


def save_weights(path, tensors):
    from safetensors.torch import save_file
    path = Path(path)
    temp = path.with_name('.' + path.name + '.partial')
    save_file({k: v.detach().cpu().contiguous() for k, v in tensors.items()}, str(temp))
    os.replace(temp, path)


def adapt_encoder(config, root, limit=None):
    from .reader import FrozenReader
    if config.get('adaptation_protocol') != 'paragraph_memory_only_v1':
        raise RuntimeError('Short-fact adaptation is retired; select paragraph_memory_only_v1 explicitly')
    if not config.get('allow_training'):
        raise RuntimeError('Training disabled; explicitly authorize a new config')
    episodes = train_only(config, limit)
    samples = paragraph_samples(episodes, config['seed'])
    root = Path(root)
    if (root / 'encoder.safetensors').exists():
        raise RuntimeError('Use a new run; optimizer resume is not implemented')
    write_json(root / 'training_input_audit.json', {'protocol': config['adaptation_protocol'],
        'samples': len(samples), 'train_ids': [e['id'] for e in episodes],
        'input_sha256': digest(samples), 'note': 'Synthetic multi-fact paragraphs, not natural-dialogue validation'})
    reader = FrozenReader(config)
    parameters = reader.enable_encoder_adaptation()
    optimizer = torch.optim.AdamW(parameters, lr=config['learning_rate'], weight_decay=0.)
    original_lora = {k: digest(v.detach().cpu().numpy().tobytes()) for k, v in reader.net.named_parameters() if 'lora_' in k}
    task_embeddings = reader.net.memory_token_embed.weight[-3:].detach().clone()
    rng = random.Random(config['seed'])
    steps, started = 0, time.perf_counter()
    accumulation = config['gradient_accumulation']
    for epoch in range(config['epochs']):
        rng.shuffle(samples)
        optimizer.zero_grad(set_to_none=True)
        for i, (_, text, query, answer) in enumerate(samples):
            memory = reader.encode(text, training=True)
            loss = reader.qa_loss(query, answer, memory)
            if not torch.isfinite(loss):
                raise ValueError('Non-finite adaptation loss')
            size = min(accumulation, len(samples) - i // accumulation * accumulation)
            (loss / size).backward()
            if (i + 1) % accumulation == 0 or i + 1 == len(samples):
                norm = torch.nn.utils.clip_grad_norm_(parameters, config['gradient_clip'])
                if not torch.isfinite(norm):
                    raise ValueError('Non-finite adaptation gradient')
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                steps += 1
                print(f'adapt epoch={epoch} step={steps} loss={float(loss.detach()):.4f}', flush=True)
        if not torch.equal(task_embeddings, reader.net.memory_token_embed.weight[-3:].detach()):
            raise AssertionError('Frozen task tokens changed')
        if original_lora != {k: digest(v.detach().cpu().numpy().tobytes()) for k, v in reader.net.named_parameters() if 'lora_' in k}:
            raise AssertionError('Released LoRA changed during memory-only adaptation')
    checkpoint = root / 'encoder.safetensors'
    save_weights(checkpoint, reader.encoder_state())
    result = {'completed': True, 'scope': 'smoke' if limit else 'full_training_split',
              'checkpoint': str(checkpoint), 'checkpoint_sha256': file_digest(checkpoint),
              'base_checkpoint_sha256': file_digest(config['checkpoint']), 'model_revision': config['model_revision'],
              'training_data_sha256': digest(episodes), 'optimizer_steps': steps,
              'training_samples': len(samples), 'trainable_parameters': sum(p.numel() for p in parameters),
              'protocol': config['adaptation_protocol'], 'rank512_lora_unchanged': True,
              'task_tokens_frozen': True, 'decoder_frozen': True, 'seconds': time.perf_counter() - started,
              'note': 'Parameter freezing does not guarantee preservation of long-context performance.'}
    write_json(root / 'adaptation.json', result)
    return result


def train_codec(config, root, conditioned=False, limit=None):
    from .codec import SlotCodec, VersionMarkers
    from .index import SemanticIndex
    from .reader import FrozenReader
    if not config.get('allow_training'):
        raise RuntimeError('Training disabled; explicitly authorize a new config')
    validate_codec_gate(config)
    episodes = train_only(config, limit)
    protocol = config.get('memory_protocol', 'immutable_keyframe_v1')
    if protocol not in ('immutable_keyframe_v1', 'final_current_snapshot_v1'):
        raise ValueError('Unknown codec memory protocol')
    pairs = temporal_pairs(episodes, config['keyframe_interval'], protocol)
    teacher_context = config.get('distillation_context', 'single_target')
    if teacher_context not in ('single_target', 'matched_current_history'):
        raise ValueError('Unknown distillation context')
    if teacher_context == 'matched_current_history' and protocol != 'final_current_snapshot_v1':
        raise ValueError('Matched current/history teacher is defined for the snapshot protocol')
    teacher_policy = ('Uncompressed current + historical, identical detached role/time markers and gold prefixes'
                      if teacher_context == 'matched_current_history' else
                      'Single uncompressed target entry, detached logits and gold prefixes')
    if protocol == 'final_current_snapshot_v1':
        from .refinement import align_final_current
        pairs = align_final_current(pairs, episodes, condition_on_current=True)
        if config.get('decoder_anchor_dropout', 0.):
            raise ValueError('Main final-current C/D comparison requires equal regularization: no anchor dropout')
    feature_episode = {event['id']: episode['id'] for episode in episodes for event in ingestion_view(episode)}
    if config.get('use_feature_cache'):
        from .features import check
        check(config, 'train')
    if not pairs:
        raise ValueError('No eligible historical QA pairs under the selected memory protocol')
    root = Path(root)
    if (root / 'codec.safetensors').exists() or (root / 'loss_history.jsonl').exists():
        raise RuntimeError('Use a new training run; partial optimizer resume is not implemented')
    started = time.perf_counter()
    initial_ids = {ingestion_view(e)[0]['id'] for e in episodes if e['events']}
    write_json(root / 'training_input_audit.json', {
        'memory_protocol': protocol, 'training_data_sha256': digest(episodes),
        'pair_ids_sha256': digest(sorted(p['id'] for p in pairs)), 'pair_content_sha256': digest(pairs),
        'pairs': len(pairs), 'negative_supervised_pairs': sum(bool(p['negative_answers']) for p in pairs),
        'current_policy': 'final observed current' if protocol == 'final_current_snapshot_v1' else 'next observed update',
        'condition_policy': 'same final current for encoding and decoding' if protocol == 'final_current_snapshot_v1'
                            else 'immutable historical keyframe',
        'teacher_policy': teacher_policy, 'distillation_context': teacher_context,
        'codec_variant': config.get('codec_variant', 'residual'),
        'codec_value_path': config.get('codec_value_path', 'projected'),
        'codec_activation_rms_cap': config.get('codec_activation_rms_cap'),
        'decode_conditioned': config.get('decode_conditioned', conditioned),
        'historical_target_policy': ('all non-final versions with QA, one canonical QA per version'
                                    if protocol == 'final_current_snapshot_v1' else 'skip initial/keyframe versions'),
        'initial_version_targets': sum(p['old']['id'] in initial_ids for p in pairs),
        'training_episode_count': len(episodes), 'supervised_episode_count': len({p['state_key'] for p in pairs}),
        'historical_targets': [{'pair_id': p['id'], 'old_id': p['old']['id'], 'query_id': p['historical']['id']}
                               for p in pairs]})
    torch.manual_seed(config['seed'])
    reader = FrozenReader(config)
    dropout = config.get('decoder_anchor_dropout', 0.) if conditioned else 0.
    codec = SlotCodec(reader.shape[1], config['history_slots'], config['codec_width'],
                      config['codec_heads'], conditioned, dropout, config['seed'],
                      config.get('codec_variant', 'residual'),
                      config.get('decode_conditioned'), config.get('codec_value_path'),
                      config.get('codec_activation_rms_cap')).to(reader.device)
    markers = VersionMarkers(reader.shape[1]).to(reader.device)
    index = SemanticIndex(reader.shape[1], config['key_dimensions']).to(reader.device)
    parameters = list(codec.parameters()) + list(markers.parameters())
    index_parameters = list(index.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=config['codec_learning_rate'])
    # Separate clipping/optimizer avoids C/D gradient magnitudes changing the
    # otherwise shared index updates and confounding the compression ablation.
    index_optimizer = torch.optim.AdamW(index_parameters, lr=config['codec_learning_rate'])
    rng = random.Random(config['seed'])
    accumulation = config['gradient_accumulation']
    steps = 0
    total_steps = config['codec_epochs'] * math.ceil(len(pairs) / accumulation)
    if config.get('expected_temporal_pairs') is not None and len(pairs) != config['expected_temporal_pairs']:
        raise ValueError('Architecture study training target coverage changed')
    scheduler = index_scheduler = None
    if config.get('codec_lr_schedule') == 'warmup_cosine':
        warmup = min(total_steps - 1, max(1, int(total_steps * config.get('codec_warmup_fraction', .05))))
        schedule = lambda step: cosine_warmup_factor(step, total_steps, warmup)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
        index_scheduler = torch.optim.lr_scheduler.LambdaLR(index_optimizer, schedule)
    diagnostic_pairs = []
    if config.get('train_diagnostic_questions', 0):
        from .refinement import fixed_train_pairs
        previous_run = config.get('previous_train_diagnostic_run')
        reference_ids = None
        if previous_run:
            if Path(previous_run).name != previous_run:
                raise ValueError('Diagnostic reference must be a run name')
            previous = read_json(Path(config['output_root']) / previous_run / 'train_diagnostic_manifest.json')
            if previous['training_data_sha256'] != digest(episodes):
                raise ValueError('Do not change the fixed 200 train diagnostic questions or underlying data')
            reference_ids = previous['pair_ids']
        diagnostic_pairs = fixed_train_pairs(pairs, config['train_diagnostic_questions'], config['seed'], episodes,
                                            reference_ids=reference_ids)
        if previous_run and previous['diagnostic_pairs_sha256'] != digest(diagnostic_pairs):
            raise ValueError('Fixed train diagnostic question/context content changed')
        write_json(root / 'train_diagnostic_manifest.json', {
            'pair_ids': [p['id'] for p in diagnostic_pairs], 'training_data_sha256': digest(episodes),
            'seed': config['seed'], 'scope': 'fixed in-sample disjoint-answer conflict pairs',
            'diagnostic_pairs_sha256': digest(diagnostic_pairs),
            'current_policy': 'final observed current version', 'memory_protocol': protocol,
            'optimizer_current_policy': 'final observed current' if protocol == 'final_current_snapshot_v1'
                                        else 'next observed update'})
    epoch_history = []
    def memory_of(event):
        if config.get('use_feature_cache'):
            from .features import load
            return load(config, 'train', feature_episode[event['id']], event).to(reader.device).float()
        return reader.encode_event(event).to(reader.device).float()

    def tensor_stats(value):
        value = value.detach().float()
        finite = torch.isfinite(value)
        valid = value[finite]
        return {'shape': list(value.shape), 'nonfinite_count': int((~finite).sum()),
                'abs_max': float(valid.abs().max()) if valid.numel() else None,
                'rms': float(valid.double().square().mean().sqrt()) if valid.numel() else None}

    def require_finite(value, label, pair):
        if not torch.isfinite(value).all():
            write_json(root / 'numerical_failure.json', {
                'stage': label, 'epoch': epoch + 1, 'pair': i + 1, 'pair_id': pair['id'],
                'optimizer_steps': steps, 'current_entry_id': pair['current']['id'],
                'old_entry_id': pair['old']['id'], 'tensor': tensor_stats(value),
                'old': tensor_stats(old), 'recovered': tensor_stats(recovered),
                'activation_rms_cap': codec.activation_rms_cap})
            raise ValueError(f'Non-finite {label} at pair {pair["id"]} '
                             f'(old={pair["old"]["id"]}, current={pair["current"]["id"]})')

    for epoch in range(config['codec_epochs']):
        rng.shuffle(pairs)
        optimizer.zero_grad(set_to_none=True)
        index_optimizer.zero_grad(set_to_none=True)
        epoch_rows = []
        for i, pair in enumerate(pairs):
            current, old, anchor = [memory_of(pair[name])
                                    for name in ('current', 'old', 'anchor')]
            recovered = codec(old, anchor)
            require_finite(recovered, 'recovered memory', pair)
            student_bundle = torch.cat((markers(current, 'current', pair['current']['time']),
                                        markers(recovered, 'historical', pair['old']['time'])))
            teacher_bundle = (matched_teacher_bundle(markers, current, old, pair)
                              if teacher_context == 'matched_current_history' else None)
            losses, ce_values, kl_values = [], [], []
            for kind, teacher_memory in (('latest', current), ('historical', old)):
                q = pair[kind]
                visible, answer = query_view(q), q['answers'][0]
                with torch.no_grad():
                    teacher, teacher_target = reader.answer_logits(visible, answer,
                                            teacher_bundle if teacher_bundle is not None else teacher_memory)
                student, target = reader.answer_logits(visible, answer, student_bundle)
                require_finite(student, f'{kind} student logits', pair)
                require_finite(teacher, f'{kind} teacher logits', pair)
                if not torch.equal(teacher_target, target):
                    raise ValueError('Teacher/student answer prefixes differ')
                ce, kl = distillation_loss(student, teacher, target, config['distillation_temperature'])
                losses.append(ce + config['distillation_weight'] * kl)
                ce_values.append(float(ce.detach()))
                kl_values.append(float(kl.detach()))
                if kind == 'historical' and config.get('contrastive_weight', 0):
                    correct_logprob = mean_answer_logprob(student, target)
            unlikelihood = recovered.new_zeros(())
            contrast = recovered.new_zeros(())
            for wrong in pair['negative_answers']:
                logits, target = reader.answer_logits(query_view(pair['historical']), wrong, student_bundle)
                require_finite(logits, 'negative-answer student logits', pair)
                unlikelihood = unlikelihood + sequence_unlikelihood(logits, target)
                if config.get('contrastive_weight', 0):
                    contrast = contrast + contrastive_margin(correct_logprob, mean_answer_logprob(logits, target),
                                                             config.get('contrastive_margin', 1.5))
            if pair['negative_answers']:
                contrast = contrast / len(pair['negative_answers'])
            pooled = 1. - F.cosine_similarity(old.mean(0), recovered.mean(0), dim=0)
            # Same-state train labels supervise the shared pooler; different entity/
            # relation groups are negatives. No benchmark labels are used.
            positive = 1. - (index(old) * index(current)).sum()
            other = next((p for p in pairs if p['state_key'] != pair['state_key']), None)
            negative = recovered.new_zeros(())
            if other is not None:
                distractor = memory_of(other['current'])
                negative = F.relu((index(old) * index(distractor)).sum() - config['index_negative_margin'])
            loss = (sum(losses) / 2 + config['unlikelihood_weight'] * unlikelihood +
                    config['pooled_alignment_weight'] * pooled + config['index_loss_weight'] * (positive + negative) +
                    config.get('contrastive_weight', 0.) * contrast)
            if not torch.isfinite(loss):
                components = {'loss': loss, 'unlikelihood': unlikelihood, 'contrastive': contrast,
                              'pooled': pooled, 'positive': positive, 'negative': negative,
                              'qa': sum(losses) / 2}
                bad = [name for name, value in components.items() if not torch.isfinite(value)]
                require_finite(loss, f'codec loss; components={bad}', pair)
            size = min(accumulation, len(pairs) - i // accumulation * accumulation)
            (loss / size).backward()
            learning_rate = optimizer.param_groups[0]['lr']
            if (i + 1) % accumulation == 0 or i + 1 == len(pairs):
                norm = torch.nn.utils.clip_grad_norm_(parameters, config['gradient_clip'])
                index_norm = torch.nn.utils.clip_grad_norm_(index_parameters, config['gradient_clip'])
                if not torch.isfinite(norm) or not torch.isfinite(index_norm):
                    require_finite(torch.stack((norm, index_norm)), 'codec/index gradient norm', pair)
                optimizer.step()
                index_optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                    index_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                index_optimizer.zero_grad(set_to_none=True)
                steps += 1
            progress = {'epoch': epoch + 1, 'epochs': config['codec_epochs'], 'pair': i + 1, 'steps': steps,
                       'loss': float(loss.detach()), 'qa_ce': ce_values, 'teacher_kl': kl_values,
                       'negative_version_loss': float(unlikelihood.detach()), 'contrastive_loss': float(contrast.detach()),
                       'pooled_alignment_loss': float(pooled.detach()), 'index_loss': float((positive + negative).detach()),
                       'learning_rate': learning_rate, 'total_steps': total_steps,
                       'negative_supervised': bool(pair['negative_answers']),
                       'current_entry_id': pair['current']['id'], 'condition_entry_id': pair['anchor']['id'],
                       'student_latent_slots': int(student_bundle.shape[0]),
                       'teacher_latent_slots': int(teacher_bundle.shape[0]) if teacher_bundle is not None else int(old.shape[0]),
                       'recovered_stats': tensor_stats(recovered),
                       'elapsed_seconds': time.perf_counter() - started}
            epoch_rows.append(progress)
            write_json(root / 'progress.json', progress)
            with (root / 'loss_history.jsonl').open('a') as handle:
                handle.write(json.dumps(progress) + '\n')
            print(f'codec epoch={epoch} pair={i+1}/{len(pairs)} loss={float(loss.detach()):.4f}', flush=True)
        epoch_result = {'epoch': epoch + 1, 'steps': steps, 'pairs': len(pairs),
                        'mean': {k: sum(r[k] for r in epoch_rows) / len(epoch_rows)
                                 for k in ('loss', 'negative_version_loss', 'contrastive_loss',
                                           'pooled_alignment_loss', 'index_loss')},
                        'qa_ce_mean': [sum(r['qa_ce'][j] for r in epoch_rows) / len(epoch_rows) for j in range(2)],
                        'teacher_kl_mean': [sum(r['teacher_kl'][j] for r in epoch_rows) / len(epoch_rows) for j in range(2)]}
        if diagnostic_pairs:
            from .refinement import train_diagnostic
            write_json(root / 'progress.json', {'phase': 'train_200_diagnostic', 'epoch': epoch + 1, 'steps': steps})
            diagnostic = train_diagnostic(reader, codec, markers, memory_of, diagnostic_pairs)
            write_json(root / 'train_evaluation' / f'epoch_{epoch+1:02d}.json', diagnostic)
            epoch_result['train_conflict_diagnostic'] = {k: v for k, v in diagnostic.items() if k != 'rows'}
            print(f'train-diagnostic epoch={epoch+1} F1={diagnostic["f1"]:.4f} stale={diagnostic["stale_answer"]:.4f}', flush=True)
        epoch_history.append(epoch_result)
        write_json(root / 'epoch_history.json', epoch_history)
        if config.get('save_epoch_checkpoints'):
            tensors = {prefix + '.' + k: v for prefix, module in [('codec', codec), ('markers', markers), ('index', index)]
                       for k, v in module.state_dict().items()}
            save_weights(root / f'epoch_{epoch+1:02d}.safetensors', tensors)
    state = {prefix + '.' + k: v for prefix, module in [('codec', codec), ('markers', markers), ('index', index)]
             for k, v in module.state_dict().items()}
    checkpoint = root / 'codec.safetensors'
    save_weights(checkpoint, state)
    result = {'completed': True, 'scope': 'smoke' if limit else 'full_training_split',
              'method': 'D' if conditioned else 'C', 'checkpoint': str(checkpoint),
              'checkpoint_sha256': file_digest(checkpoint), 'training_data_sha256': digest(episodes),
              'base_checkpoint_sha256': file_digest(config['checkpoint']),
              'adapted_checkpoint_sha256': file_digest(config['adapted_checkpoint']) if config.get('adapted_checkpoint') else None,
              'model_revision': config['model_revision'], 'codec_width': config['codec_width'],
              'codec_heads': config['codec_heads'], 'keyframe_interval': config['keyframe_interval'],
              'reader_sha256': file_digest(Path(__file__).with_name('reader.py')),
              'semantic_index_sha256': digest({k: digest(v.detach().cpu().numpy().tobytes()) for k, v in index.state_dict().items()}),
              'pair_ids_sha256': digest(sorted(p['id'] for p in pairs)), 'pairs': len(pairs),
              'pair_content_sha256': read_json(root / 'training_input_audit.json')['pair_content_sha256'],
              'memory_protocol': protocol, 'seed': config['seed'],
              'codec_parameters': codec.parameter_count(), 'marker_parameters': sum(p.numel() for p in markers.parameters()),
              'index_parameters': sum(p.numel() for p in index.parameters()), 'steps': steps,
              'seconds': time.perf_counter() - started,
              'history_slots': codec.slots, 'role_time_slots_per_entry': 2,
              'codec_variant': codec.variant, 'decode_conditioned': codec.decode_conditioned,
              'codec_value_path': codec.value_path,
              'codec_activation_rms_cap': codec.activation_rms_cap,
              'anchor_dropout': dropout, 'codec_epochs': config['codec_epochs'], 'epoch_history': epoch_history,
              'learning_rate': config['codec_learning_rate'], 'lr_schedule': config.get('codec_lr_schedule', 'constant'),
              'warmup_fraction': config.get('codec_warmup_fraction'), 'shuffle_each_epoch': True,
              'contrastive_margin': config.get('contrastive_margin'),
              'contrastive_weight': config.get('contrastive_weight', 0.),
              'contrastive_protocol': 'Mean answer text token log-probability excluding EOS; mean over valid negative aliases',
              'anchor_policy': ('final current shared with reader; offline snapshot, not online rebase'
                                if protocol == 'final_current_snapshot_v1' else
                                'immutable keyframe; auxiliary anchor cost is reported separately'),
              'decoder_frozen': True, 'encoder_frozen': True, 'index_validated_for_routing': False,
              'teacher_policy': teacher_policy, 'distillation_context': teacher_context,
              'loss_weights': {k: config[k] for k in ('distillation_weight', 'unlikelihood_weight',
                                                      'pooled_alignment_weight', 'index_loss_weight')},
              'readability_gate_override': read_json(config['codec_gate_override']) if config.get('codec_gate_override') else None,
              'warning': 'Training completion does not validate readability, retrieval, or research claims.'}
    write_json(root / 'codec_training.json', result)
    return result


def validate_codec_gate(config):
    name = config.get('codec_gate_run')
    override_path = config.get('codec_gate_override')
    if ((not config.get('codec_manual_review_approved') and not override_path) or
            not name or Path(name).name != name):
        raise RuntimeError('Codec training requires a complete readability gate and explicit manual review')
    root = Path(config['output_root']) / name
    gate, summary, manifest = [read_json(root / file) for file in ('gate.json', 'summary.json', 'manifest.json')]
    if not summary.get('complete') or summary.get('scope') != 'validation':
        raise RuntimeError('Partial/pilot or failed readability cannot authorize codec training')
    if override_path:
        override = read_json(override_path)
        if (not override.get('explicit_user_authorization') or not override.get('authorization_note') or
                override.get('gate_run') != name or
                override.get('summary_sha256') != file_digest(root / 'summary.json') or
                override.get('gate_sha256') != file_digest(root / 'gate.json') or
                override.get('original_decision') != gate.get('decision')):
            raise RuntimeError('Invalid or stale explicit user gate override')
    elif gate.get('decision') != 'eligible_for_manual_review':
        raise RuntimeError('Partial/pilot or failed readability cannot authorize codec training')
    previous = manifest['config']
    for key in ('model_revision', 'icae_source_sha256', 'checkpoint', 'adapted_checkpoint'):
        if previous.get(key) != config.get(key):
            raise RuntimeError('Readability gate uses a different reader configuration')
    if manifest['code'].get('vcrm/reader.py') != file_digest(Path(__file__).with_name('reader.py')):
        raise RuntimeError('Reader changed since readability gate')
    for path, sha in manifest['encoder_weights'].items():
        if file_digest(path) != sha:
            raise RuntimeError('Readability checkpoint weights changed')
