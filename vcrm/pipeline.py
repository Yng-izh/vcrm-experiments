from collections import defaultdict
from pathlib import Path
import statistics
import time

import torch

from .codec import load_codec
from .data import ingestion_view, query_view
from .io import digest, read_json, row_path, write_json
from .metrics import exact, f1, paired_bootstrap
from .reader import FrozenReader
from .store import MemoryStore

TRAINING_MATCH_FIELDS = ('training_ids', 'training_data_sha256', 'steps', 'epochs', 'rank',
                         'trainable_parameters', 'feature_signature')


def checkpoint_for(config, method, debug=False):
    path = Path(config['project']) / ('debug_checkpoints' if debug else 'checkpoints') / f'{method}.pt'
    if not path.exists():
        raise RuntimeError(f'Train {method} on official TempLAMA train before evaluation')
    row = torch.load(path, map_location='cpu', weights_only=False)
    from .training import feature_signature
    if row['feature_signature'] != feature_signature(config):
        raise RuntimeError('Codec trained under a different backbone/reader/feature configuration')
    if row.get('smoke_only') and not debug:
        raise RuntimeError('Smoke-trained codecs are not accepted for formal evaluation')
    if not debug and method == 'vcrm':
        other_method = 'independent' if method == 'vcrm' else 'vcrm'
        other_path = Path(config['project']) / 'checkpoints' / f'{other_method}.pt'
        if not other_path.exists():
            raise RuntimeError('Formal C/D comparison requires both matched trained codecs')
        other = torch.load(other_path, map_location='cpu', weights_only=False)
        if other.get('smoke_only') or any(row.get(k) != other.get(k) for k in TRAINING_MATCH_FIELDS):
            raise RuntimeError('C/D codec training provenance is not matched')
    return path


def infer(config, root, dataset, split, method, budget, limit=None, shard=0, shards=1):
    if split == 'train':
        raise ValueError('Evaluation uses val/test, not training instances')
    episodes = read_json(Path(config['data_root']) / 'prepared' / f'{dataset}_{split}.json')
    if limit:
        episodes = episodes[:limit]
    targets = episodes[shard::shards]
    reader = FrozenReader(config)
    codec_path = checkpoint_for(config, method, bool(limit)) if method in ('independent', 'vcrm') else None
    codec = load_codec(codec_path, method) if codec_path else None
    codec_hash = digest(codec_path.read_bytes()) if codec_path else None
    codec_training = torch.load(codec_path, map_location='cpu', weights_only=False) if codec_path else None
    identity = f'{dataset}_{split}/{method}/budget_{budget}'
    base = Path(root) / identity
    signature = digest([config, codec_hash])
    frozen_weights = {'backbone_parameters': sum(p.numel() for p in reader.model.parameters()),
                      'reader_parameters': sum(p.numel() for p in reader.mem.parameters()),
                      'codec_parameters': sum(p.numel() for p in codec.parameters()) if codec else 0,
                      'codec_checkpoint_bytes': codec_path.stat().st_size if codec_path else 0,
                      'codec_sha256': codec_hash,
                      'codec_training_signature': digest({k: codec_training.get(k) for k in TRAINING_MATCH_FIELDS})
                      if codec_training else None,
                      'note': 'Shared model weights are separate from per-episode persistent memory bytes.'}
    for i, episode in enumerate(targets):
        result_path = row_path(base, 'episodes', episode['id'])
        if result_path.exists():
            if read_json(result_path).get('method_signature') != signature:
                raise RuntimeError('Existing outputs use different codec weights or inference settings')
            continue
        started = time.perf_counter()
        store = MemoryStore(method, budget, reader.shape, config['key_dimensions'], codec, config['state_similarity'])
        trace = []
        for event in ingestion_view(episode):
            before = time.perf_counter()
            bundle = reader.encode_event(event) if method != 'text_log' else None
            state = store.ingest(event, bundle)
            trace.append({'id': event['id'], 'time': event['time'], 'write_seconds': time.perf_counter() - before, **state})
            del bundle
        store_path = base / 'stores' / (digest(episode['id']) + '.memory')
        nbytes = store.save(store_path)
        write_seconds = time.perf_counter() - started
        # Drop the live store and load ONLY the durable representation before any query.
        del store
        store = MemoryStore.load(store_path, budget, codec, config['state_similarity'])
        outputs = []
        for query in episode['queries']:
            selected = store.select(query_view(query),
                                    2 ** 40 if method == 'text_log' else config['read_latent_slots'])
            text, bundle = '', None
            if method == 'text_log':
                # Whole entries only, never truncate part of a selected fact/session.
                accepted = []
                for entry in selected:
                    candidate = accepted + [entry]
                    if len(reader.tok.encode(store.text(candidate), add_special_tokens=False)) <= config['read_text_tokens']:
                        accepted = candidate
                selected = accepted
                text = store.text(selected)
            else:
                bundle = store.bundle(selected)
            answer = reader.answer(query_view(query), bundle, text)
            outputs.append({'query_id': query['id'], 'type': query['type'], 'latest': query.get('latest'),
                            'selected_entry_ids': [e['id'] for e in selected],
                            'retrieval_audit': store.retrieval_audit(query_view(query), selected),
                            'selected_times': [e['time'] for e in selected], **answer})
        write_json(result_path, {'id': episode['id'], 'dataset': dataset, 'split': split, 'group': episode['group'],
                                'method': method, 'budget': budget, 'store_bytes': nbytes,
                                'accounting': store.accounting(), 'write_seconds': write_seconds,
                                'write_trace': trace, 'queries': outputs, 'query_history_access': False,
                                'state_index_protocol': 'Official entity-relation for TempLAMA; shared lexical association for conversations',
                                'scope': 'smoke' if limit else 'formal',
                                'retained_entry_ids': [e['id'] for e in store.entries],
                                'capacity_protocol': 'no-eviction baseline; not fixed-budget comparison'
                                if config.get('no_eviction_baseline') else 'fixed persistent bytes',
                                'method_signature': signature, 'model_weights': frozen_weights})
        print(f'{identity}: {i + 1}/{len(targets)} {episode["id"]}, {nbytes}/{budget} bytes', flush=True)


def score(config, root):
    from .judge import score as judge_score
    datasets = {}
    evidence = {r['question_id']: r for r in read_json(Path(config['data_root']) / 'longmemeval_s_cleaned.json')}
    for path in (Path(config['data_root']) / 'prepared').glob('*.json'):
        if path.name in ('manifest.json', 'entity_split_audit.json'):
            continue
        for episode in read_json(path):
            datasets[episode['id']] = episode
    count = 0
    for path in Path(root).glob('*/*/budget_*/episodes/*.json'):
        row = read_json(path)
        episode = datasets[row['id']]
        if episode['dataset'] == 'stale':
            continue
        queries = {q['id']: q for q in episode['queries']}
        dest = path.parent.parent / 'scores' / path.name
        if dest.exists():
            continue
        scores = []
        for prediction in row['queries']:
            query = queries[prediction['query_id']]
            answer = prediction['answer']
            if episode['dataset'] == 'longmemeval':
                values = judge_score(config, query, answer, root)
                annotations = evidence[query['id']]
                wanted = set(map(str, annotations['answer_session_ids']))
                gold_ids = {f'event_{i:06d}' for i, sid in enumerate(annotations['haystack_session_ids']) if str(sid) in wanted}
                retained = set(row['retained_entry_ids'])
                selected = set(prediction['selected_entry_ids'])
                values['evidence_diagnostics'] = {
                    'gold_count': len(gold_ids), 'retained_count': len(gold_ids & retained),
                    'selected_count': len(gold_ids & selected),
                    'failure_bucket': 'correct' if values['accuracy'] else 'abstention_or_no_evidence'
                    if not gold_ids else 'all_evidence_evicted' if not gold_ids & retained
                    else 'retained_but_unselected' if not gold_ids & selected else 'selected_but_wrong',
                    'note': 'Offline evidence join only; selected-but-wrong does not isolate compression from reasoning.'}
            elif episode['dataset'] == 'locomo':
                from deltamem.eval.locomo_protocol import LoCoMoQuestionSpec, canonicalize_locomo_prediction, score_locomo_prediction
                spec = LoCoMoQuestionSpec(query['question'], query['category'], None)
                canonical = canonicalize_locomo_prediction(answer, spec)
                values = {'f1': score_locomo_prediction(query['qa'], canonical), 'protocol': 'LoCoMo official token F1'}
            else:
                # Recent-version false returns exclude facts also valid in the target year.
                newer = [a for q in episode['queries'] if q['time'] > query['time'] for a in q['answers']]
                stale = int(not exact(answer, query['answers']) and bool(newer) and exact(answer, newer))
                values = {'accuracy': exact(answer, query['answers']), 'f1': f1(answer, query['answers']),
                          'stale_answer': stale, 'latest': query['latest'],
                          'protocol': 'TempLAMA-Streaming original yearly labels; max over valid answers'}
            scores.append({'query_id': query['id'], 'type': query['type'], **values})
        write_json(dest, {'id': row['id'], 'group': row['group'], 'scores': scores})
        count += 1
    return {'new_scored_episodes': count}


def report(config, root):
    summary = {}
    paired = defaultdict(dict)
    for folder in Path(root).glob('*/*/budget_*'):
        episode_rows = [read_json(p) for p in (folder / 'episodes').glob('*.json')]
        score_rows = [read_json(p) for p in (folder / 'scores').glob('*.json')]
        if not episode_rows:
            continue
        flat = [s for row in score_rows for s in row['scores']]
        dataset, method, budget_name = folder.relative_to(root).parts
        result = {'generated_episodes': len(episode_rows), 'scored_episodes': len(score_rows),
                  'store_bytes_max': max(r['store_bytes'] for r in episode_rows),
                  'write_seconds_mean': statistics.mean(r['write_seconds'] for r in episode_rows),
                  'read_seconds_mean': statistics.mean(q['read_seconds'] for r in episode_rows for q in r['queries']),
                  'generation_limit_hits': sum(q['generation_limit_hit'] for r in episode_rows for q in r['queries'])}
        result['capacity_protocol'] = episode_rows[0].get('capacity_protocol', 'fixed persistent bytes')
        result['retained_entries_mean'] = statistics.mean(len(r.get('retained_entry_ids', [])) for r in episode_rows)
        expected = len(read_json(Path(config['data_root']) / 'prepared' / f'{dataset}.json'))
        limit = read_json(Path(root) / 'manifest.json')['limit']
        if limit:
            expected = min(expected, limit)
        result.update(expected_episodes=expected, complete=len(episode_rows) == len(score_rows) == expected,
                      scope='smoke' if limit else 'formal')
        result['peak_gpu_bytes'] = max(q['peak_gpu_bytes'] for r in episode_rows for q in r['queries'])
        if method == 'vcrm':
            independent_folder = folder.parent.parent / 'independent' / folder.name
            if independent_folder.exists():
                other = [read_json(p) for p in (independent_folder / 'episodes').glob('*.json')]
                if other:
                    c = other[0]['model_weights']
                    d = episode_rows[0]['model_weights']
                    if c['codec_parameters'] != d['codec_parameters']:
                        raise ValueError('C/D trainable parameter counts differ')
                    if not limit and (not c.get('codec_training_signature') or
                                      c['codec_training_signature'] != d.get('codec_training_signature')):
                        raise ValueError('Formal C/D prediction artifacts have different training provenance')
        if dataset.startswith('templama'):
            latest = [s for s in flat if s['latest']]
            historical = [s for s in flat if not s['latest']]
            for name, subset, metric in (('latest_accuracy', latest, 'accuracy'), ('historical_f1', historical, 'f1'),
                                         ('historical_retention', historical, 'accuracy'), ('stale_answer_rate', historical, 'stale_answer')):
                result[name] = statistics.mean(s[metric] for s in subset) if subset else None
            group_metrics = defaultdict(lambda: defaultdict(list))
            for row in score_rows:
                for s in row['scores']:
                    if s['latest']:
                        group_metrics[row['group']]['latest_accuracy'].append(s['accuracy'])
                    else:
                        group_metrics[row['group']]['historical_f1'].append(s['f1'])
                        group_metrics[row['group']]['historical_retention'].append(s['accuracy'])
                        group_metrics[row['group']]['stale_answer_rate'].append(s['stale_answer'])
            result['group_macro'] = {k: statistics.mean(statistics.mean(v[k]) for v in group_metrics.values() if v[k])
                                     for k in ('latest_accuracy', 'historical_f1', 'historical_retention', 'stale_answer_rate')
                                     if any(v[k] for v in group_metrics.values())}
            metric = 'f1'
        else:
            metric = 'accuracy' if dataset.startswith('longmemeval') else 'f1'
            result[metric] = statistics.mean(s[metric] for s in flat) if flat else None
            result['by_type'] = {t: statistics.mean(s[metric] for s in flat if s['type'] == t) for t in sorted({s['type'] for s in flat})}
            if dataset.startswith('longmemeval'):
                for label, subset in [('all', flat), ('knowledge-update', [s for s in flat if s['type'] == 'knowledge-update'])]:
                    buckets = defaultdict(int)
                    annotated = [s['evidence_diagnostics'] for s in subset if s.get('evidence_diagnostics')]
                    for item in annotated:
                        buckets[item['failure_bucket']] += 1
                    result.setdefault('evidence_diagnostics', {})[label] = {
                        'failure_buckets': dict(buckets),
                        'any_evidence_selected_rate': statistics.mean(e['selected_count'] > 0 for e in annotated if e['gold_count'])
                        if any(e['gold_count'] for e in annotated) else None}
        summary[str(folder.relative_to(root))] = result
        paired[(dataset, budget_name)][method] = {s['query_id']: (s[metric], r['group'], s.get('accuracy')) for r in score_rows for s in r['scores']}
    comparisons = {}
    for (dataset, budget), methods in paired.items():
        if 'vcrm' not in methods or 'independent' not in methods:
            continue
        ids = sorted(methods['vcrm'].keys() & methods['independent'].keys())
        for subset in ('all', 'historical', 'latest') if dataset.startswith('templama') else ('all',):
            # Yearly query grouping is taken from prepared official metadata, never from the prediction.
            official = read_json(Path(config['data_root']) / 'prepared' / f'{dataset}.json')
            latest_flags = {q['id']: q.get('latest', False) for e in official for q in e['queries']}
            chosen = [i for i in ids if subset == 'all' or latest_flags[i] == (subset == 'latest')]
            comparisons[f'{dataset}/{budget}/vcrm_minus_independent/{subset}'] = paired_bootstrap(
                [methods['vcrm'][i][2 if subset == 'latest' else 0] for i in chosen],
                [methods['independent'][i][2 if subset == 'latest' else 0] for i in chosen],
                [methods['vcrm'][i][1] for i in chosen], config['bootstrap_samples'], config['seed'])
    output = {'results': summary, 'paired_comparisons': comparisons,
              'warning': 'Smoke runs and unfinished runs are not publication results. C/D comparisons require equal training provenance and a frozen budget plan.'}
    write_json(Path(root) / 'reports/summary.json', output)
    lines = ['# VCRM Results', '', '| Run | Episodes | Scored | Peak storage bytes | Latest acc | Historical F1 | QA metric |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name, value in summary.items():
        lines.append(f"| {name} | {value['generated_episodes']} | {value['scored_episodes']} | {value['store_bytes_max']} | {value.get('latest_accuracy', '')} | {value.get('historical_f1', '')} | {value.get('accuracy', value.get('f1', ''))} |")
    from .io import atomic_bytes
    atomic_bytes(Path(root) / 'reports/summary.md', ('\n'.join(lines) + '\n').encode())
    plot_results(summary, Path(root) / 'reports')
    return output


def plot_results(summary, directory):
    import csv
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with (directory / 'storage_tradeoff.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['dataset_split', 'method', 'budget_bytes', 'actual_store_bytes_max',
                         'latest_accuracy', 'historical_f1', 'qa_metric', 'complete', 'scope'])
        for name, result in summary.items():
            dataset, method, budget = name.split('/')
            writer.writerow([dataset, method, int(budget.replace('budget_', '')), result['store_bytes_max'],
                             result.get('latest_accuracy'), result.get('historical_f1'),
                             result.get('accuracy', result.get('f1')), result['complete'], result['scope']])
    datasets = sorted({name.split('/')[0] for name in summary})
    for dataset in datasets:
        metrics = ('historical_f1', 'latest_accuracy') if dataset.startswith('templama') else (
            'accuracy' if dataset.startswith('longmemeval') else 'f1',)
        figure, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4), squeeze=False)
        for axis, metric in zip(axes[0], metrics):
            grouped = defaultdict(list)
            for name, result in summary.items():
                ds, method, budget = name.split('/')
                if ds == dataset and result.get(metric) is not None:
                    grouped[method].append((int(budget.replace('budget_', '')), result[metric]))
            for method, points in sorted(grouped.items()):
                points.sort()
                axis.plot([p[0] for p in points], [p[1] for p in points], marker='o', label=method)
            axis.set_xlabel('Persistent memory budget (bytes)')
            axis.set_ylabel(metric.replace('_', ' '))
            axis.set_ylim(0, 1)
            if grouped:
                axis.legend(fontsize=8)
        scope = 'Smoke/partial results' if any(r['scope'] == 'smoke' or not r['complete'] for n, r in summary.items() if n.startswith(dataset + '/')) else 'Formal results'
        figure.suptitle(f'{dataset}: {scope}')
        figure.tight_layout()
        figure.savefig(directory / f'{dataset}_tradeoff.png', dpi=160)
        plt.close(figure)
