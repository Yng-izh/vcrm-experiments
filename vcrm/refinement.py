"""Fixed training diagnostics and immutable post-hoc validation selection."""
from pathlib import Path
import random
import statistics

import torch

from .data import ingestion_view, query_view
from .io import file_digest, read_json, write_json
from .metrics import exact, f1, normalize, paired_bootstrap


def conflicting(historical, current):
    return not ({normalize(a) for a in historical} & {normalize(a) for a in current})


def align_final_current(pairs, episodes, condition_on_current=False):
    contexts = {e['id'] + ':' + event['id']: e for e in episodes for event in e['events']}
    aligned = []
    for pair in pairs:
        episode = contexts[pair['id']]
        queries = [q for q in episode['queries'] if q['latest']]
        if len(queries) != 1:
            raise ValueError('Snapshot requires exactly one final-current query per episode')
        latest = queries[0]
        current = max((e for e in ingestion_view(episode) if e['time'] <= latest['as_of']),
                      key=lambda e: e['time'])
        if current['id'] != ingestion_view(episode)[-1]['id'] or current['time'] <= pair['old']['time']:
            raise ValueError('Final current must be the last observation and later than the historical target')
        valid = {normalize(a) for a in pair['historical']['answers']}
        negatives = list(dict.fromkeys(a for a in latest['answers'] if normalize(a) not in valid))
        aligned.append(pair | {'latest': latest, 'current': current, 'negative_answers': negatives,
                               'anchor': current if condition_on_current else pair['anchor']})
    return aligned


def fixed_train_pairs(pairs, count, seed, episodes=None, reference_ids=None):
    if episodes is not None:
        pairs = align_final_current(pairs, episodes)
    eligible = sorted((p for p in pairs if conflicting(p['historical']['answers'], p['latest']['answers'])),
                      key=lambda p: p['id'])
    if reference_ids is not None:
        by_id = {p['id']: p for p in eligible}
        if (len(reference_ids) != count or len(set(reference_ids)) != count or
                any(pid not in by_id for pid in reference_ids) or len(by_id) != len(eligible)):
            raise ValueError('Fixed train diagnostic IDs are missing, duplicate, or no longer conflict')
        return [by_id[pid] for pid in reference_ids]
    if count > len(eligible):
        raise ValueError(f'Requested {count} train conflict pairs, only {len(eligible)} available')
    return random.Random(seed).sample(eligible, count)


@torch.no_grad()
def train_diagnostic(reader, codec, markers, memory_of, pairs):
    modes = codec.training, markers.training
    codec.eval()
    markers.eval()
    rows = []
    try:
        for pair in pairs:
            current, old, anchor = [memory_of(pair[name]) for name in ('current', 'old', 'anchor')]
            # Match persistent FP16 code roundtrip; no decoder-side dropout at evaluation.
            code = codec.encode(old, anchor).half().float()
            recovered = codec.decode(code, anchor)
            bundle = torch.cat((markers(current, 'current', pair['current']['time']),
                                markers(recovered, 'historical', pair['old']['time'])))
            answer = reader.answer(query_view(pair['historical']), bundle)['answer']
            gold = pair['historical']['answers']
            correct = exact(answer, gold)
            rows.append({'pair_id': pair['id'], 'answer': answer, 'accuracy': correct,
                         'f1': f1(answer, gold),
                         'stale_answer': int(not correct and exact(answer, pair['latest']['answers']))})
    finally:
        codec.train(modes[0])
        markers.train(modes[1])
    return {'scope': 'in_sample_training_diagnostic_not_validation', 'queries': len(rows),
            **{k: statistics.mean(r[k] for r in rows) if rows else None
               for k in ('accuracy', 'f1', 'stale_answer')}, 'rows': rows,
            'current_policy': 'final observed current version, matching validation co-injection'}


def selected_episodes(config, episodes, split):
    path = config.get('evaluation_query_manifest')
    if not path:
        return episodes
    manifest = read_json(path)
    source = Path(config['data_root']) / f'prepared/templama_{split}.json'
    if manifest['split'] != split or manifest['data_sha256'] != file_digest(source):
        raise ValueError('Fixed diagnostic selection data/split mismatch')
    requested = manifest['query_ids']
    if len(requested) != len(set(requested)):
        raise ValueError('Duplicate diagnostic query IDs')
    wanted = set(requested)
    rows = [e | {'queries': [q for q in e['queries'] if q['id'] in wanted]} for e in episodes]
    rows = [e for e in rows if e['queries']]
    observed = [q['id'] for e in rows for q in e['queries']]
    if len(observed) != len(wanted) or set(observed) != wanted:
        raise ValueError('Fixed diagnostic query coverage mismatch')
    # Preserve all original events. Query filtering must never alter memory ingestion.
    return rows


def result_rows(root):
    rows = {}
    for folder in ('partial_episodes', 'episodes'):
        for path in sorted((Path(root) / folder).glob('*.json')):
            episode = read_json(path)
            rows.update({q['query_id']: q | {'episode_id': episode['id']} for q in episode['queries']})
    return rows


def freeze_validation(config, root):
    root = Path(root)
    path = root / 'validation_481.json'
    source = Path(config['data_root']) / 'prepared/templama_val.json'
    if path.exists():
        manifest = read_json(path)
        if manifest['data_sha256'] != file_digest(source) or len(manifest['query_ids']) != 481:
            raise ValueError('Existing frozen validation selection changed')
        return path
    episodes = read_json(source)
    previous = {m: result_rows(Path(config['output_root']) / f'paper_v1_{m}16_val') for m in ('C', 'D')}
    prefix = [(e, q) for e in episodes for q in e['queries']][:2185]
    selected = []
    for episode, query in prefix:
        if query['latest'] or len(episode['events']) <= 1:
            continue
        if all(query['id'] in previous[m] and previous[m][query['id']]['selected_kind'] == 'code'
               for m in previous):
            selected.append(query['id'])
    if len(selected) != 481:
        raise ValueError(f'Cannot reconstruct original 481-query snapshot: found {len(selected)}; do not substitute samples')
    current = {e['id']: next(q['answers'] for q in e['queries'] if q['latest']) for e in episodes}
    metadata = {q['id']: {'episode_id': e['id'], 'historical_answers': q['answers'],
                          'current_answers': current[e['id']],
                          'conflicting': conflicting(q['answers'], current[e['id']])}
                for e in episodes for q in e['queries'] if q['id'] in set(selected)}
    manifest = {'split': 'val', 'data_sha256': file_digest(source), 'query_ids': selected,
                'metadata': metadata, 'source': 'Original shared first 2185 scored validation questions; both methods read code',
                'scope': 'post_hoc_fixed_validation_diagnostic_not_untouched_test',
                'conflict_queries': sum(v['conflicting'] for v in metadata.values())}
    write_json(path, manifest)
    write_json(root / 'original_1epoch_rows.json', {m: {qid: previous[m][qid] for qid in selected} for m in previous})
    return path


def diagnostic_metrics(rows, manifest):
    ids = manifest['query_ids']
    if set(rows) != set(ids):
        raise ValueError('Incomplete fixed validation results')
    def subset(selected):
        scored = []
        for qid in selected:
            row, meta = rows[qid], manifest['metadata'][qid]
            answer = row['answer']['answer']
            correct = exact(answer, meta['historical_answers'])
            scored.append({'query_id': qid, 'group': meta['episode_id'],
                           'f1': f1(answer, meta['historical_answers']), 'accuracy': correct,
                           'stale_answer': int(not correct and exact(answer, meta['current_answers']))})
        return {'queries': len(scored), **{k: statistics.mean(r[k] for r in scored) if scored else None
                                        for k in ('f1', 'accuracy', 'stale_answer')}, 'rows': scored}
    return {'all_compressed': subset(ids),
            'disjoint_current_conflicts': subset([q for q in ids if manifest['metadata'][q]['conflicting']])}


def refinement_report(config, root, prefix, protocol=None):
    root = Path(root)
    manifest = read_json(root / 'validation_481.json')
    old = read_json(root / 'original_1epoch_rows.json')
    results = {'original_1epoch': {m: diagnostic_metrics(old[m], manifest) for m in old},
               'refined_5epoch': {m: diagnostic_metrics(result_rows(Path(config['output_root']) / f'{prefix}_{m}16_val'), manifest)
                                  for m in ('C', 'D')}}
    paired = {}
    for scope in ('all_compressed', 'disjoint_current_conflicts'):
        c, d = [results['refined_5epoch'][m][scope]['rows'] for m in ('C', 'D')]
        paired[scope] = {k: paired_bootstrap([r[k] for r in d], [r[k] for r in c],
                                            [r['group'] for r in c]) for k in ('f1', 'stale_answer')}
    note = ('C/D share final-current pairing, loss, reader, optimizer and zero anchor dropout. '
            'D alone conditions code encoding/decoding on final current. Fixed offline snapshots only; '
            'all codes are encoded from original full features, not rebased lossy codes. '
            'Old results use a different training/storage protocol, not an isolated causal control.'
            if protocol == 'final_current_snapshot_v1' else
            'C/D share margin supervision. D alone has decoder keyframe dropout; '
            'this is not an isolated architecture-only ablation.')
    write_json(root / 'comparison.json', {'results': results, 'paired_D_minus_C': paired,
               'protocol': protocol, 'scope': manifest['scope'],
               'note': note + ' No claim of held-out test improvement.'})
