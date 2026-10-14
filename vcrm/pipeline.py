"""One-entry and multi-version readability checks using the same frozen reader."""
from collections import defaultdict
from pathlib import Path
import statistics
import time

from .data import ingestion_view, query_view
from .io import digest, read_json, row_path, write_json
from .metrics import exact, f1
from .store import MemoryStore


def evaluate(config, root, split='val', limit=None, shard=0, shards=1):
    from .reader import FrozenReader
    if split != 'val':
        raise ValueError('Readability gate uses independent validation only, never benchmark test')
    episodes = read_json(Path(config['data_root']) / 'prepared/templama_val.json')
    episodes = (episodes[:limit] if limit else episodes)[shard::shards]
    reader = FrozenReader(config)
    ceiling = 2 ** 50
    for i, episode in enumerate(episodes):
        destination = row_path(root, 'episodes', episode['id'])
        if destination.exists():
            continue
        store = MemoryStore('icae_independent', ceiling, reader.shape, config['key_dimensions'])
        text_store = MemoryStore('icae_text', ceiling, reader.shape, config['key_dimensions'])
        started = time.perf_counter()
        trace = []
        for event in ingestion_view(episode):
            memory = reader.encode_event(event)
            receipt = store.ingest(event, memory)
            if receipt['dropped']:
                raise AssertionError('No eviction is allowed at the readability gate')
            text_store.ingest(event)
            trace.append({'id': event['id'], 'latent_tokens': memory.shape[0],
                          'latent_payload_bytes': memory.numel() * 2, **receipt})
        write_seconds = time.perf_counter() - started
        store_path = row_path(root, 'stores', episode['id']).with_suffix('.memory')
        store.save(store_path)
        store = MemoryStore.load(store_path, ceiling)
        outputs = []
        for query in episode['queries']:
            visible = query_view(query)
            chosen = store.select(visible, config['read_latent_slots'])
            if len(chosen) != 1:
                raise ValueError('Explicit fact/time index must choose exactly one observed version')
            text_chosen = text_store.select(visible, config['read_latent_slots'])
            if [e['id'] for e in text_chosen] != [e['id'] for e in chosen]:
                raise AssertionError('Text and latent controls must use the same fact version')
            target_event = next(e for e in ingestion_view(episode) if e['id'] == chosen[0]['id'])
            single = MemoryStore('icae_independent', ceiling, reader.shape, config['key_dimensions'])
            single.ingest(target_event, store.decode(chosen[0]))
            if not (single.bundle(single.select(visible, config['read_latent_slots'])) == store.bundle(chosen)).all():
                raise AssertionError('Versioned storage changed the independently encoded memory')
            # Single-entry and indexed multi-version read are numerically identical by
            # construction. Generate once; record the invariant, not two independent trials.
            answers = {
                'latent': reader.answer(visible, store.bundle(chosen)),
                'text': reader.answer(visible, text=reader.event_text(target_event)),
                'no_memory': reader.answer(visible)}
            if len(store.entries) > 1:
                wrong = next(e for e in store.entries if e['id'] != chosen[0]['id'])
                answers['wrong_version'] = reader.answer(visible, store.bundle([wrong]))
            scores = {method: {'accuracy': exact(result['answer'], query['answers']),
                               'f1': f1(result['answer'], query['answers'])} for method, result in answers.items()}
            newer = [a for q in episode['queries'] if q['time'] > query['time'] for a in q['answers']]
            scores['latent']['stale_answer'] = int(not scores['latent']['accuracy'] and bool(newer) and
                                                   exact(answers['latent']['answer'], newer))
            outputs.append({'query_id': query['id'], 'latest': query['latest'], 'type': query['type'],
                            'as_of': query['as_of'], 'selected_entry_id': chosen[0]['id'],
                            'selected_time': chosen[0]['time'], 'single_entry_equals_indexed': True,
                            'answers': answers, 'scores': scores})
        write_json(destination, {'id': episode['id'], 'group': episode['group'], 'dataset': 'templama',
            'split': 'val', 'versions': len(store.entries), 'queries': outputs, 'write_seconds': write_seconds,
            'write_trace': trace, 'latent_accounting': store.accounting(), 'text_accounting': text_store.accounting(),
            'capacity_protocol': 'unbounded readability diagnostic, no codec/eviction',
            'state_index_protocol': 'Explicit published entity/relation and requested observation year',
            'scope': 'smoke' if limit else 'validation'})
        print(f'shard={shard} {i+1}/{len(episodes)} {episode["id"]} versions={len(store.entries)}', flush=True)


def summarize(config, root, limit=None):
    episodes = read_json(Path(config['data_root']) / 'prepared/templama_val.json')
    expected_ids = {e['id'] for e in (episodes[:limit] if limit else episodes)}
    rows = [read_json(p) for p in (Path(root) / 'episodes').glob('*.json')]
    ids = [r['id'] for r in rows]
    if len(ids) != len(set(ids)) or not set(ids) <= expected_ids:
        raise ValueError('Unexpected or duplicate validation episode ids')
    expected_queries = {e['id']: {q['id'] for q in e['queries']} for e in episodes if e['id'] in expected_ids}
    for row in rows:
        queries = [q['query_id'] for q in row['queries']]
        if len(queries) != len(set(queries)) or set(queries) != expected_queries[row['id']]:
            raise ValueError('Validation output has incomplete or duplicate yearly question coverage')
    flat = [q for row in rows for q in row['queries']]
    current = [q for q in flat if q['latest']]
    historical = [q for row in rows if row['versions'] > 1 for q in row['queries'] if not q['latest']]
    mean = lambda values: statistics.mean(values) if values else None
    methods = {}
    for method in ('latent', 'text', 'no_memory', 'wrong_version'):
        chosen = [q for q in flat if method in q['scores']]
        methods[method] = {
            'accuracy': mean([q['scores'][method]['accuracy'] for q in chosen]),
            'f1': mean([q['scores'][method]['f1'] for q in chosen]),
            'current_accuracy': mean([q['scores'][method]['accuracy'] for q in current if method in q['scores']]),
            'historical_f1': mean([q['scores'][method]['f1'] for q in historical if method in q['scores']]),
            'read_seconds_mean': mean([q['answers'][method]['read_seconds'] for q in chosen]),
            'prompt_tokens_mean': mean([q['answers'][method]['prompt_tokens'] for q in chosen]),
            'generation_limit_hits': sum(q['answers'][method]['generation_limit_hit'] for q in chosen)}
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['group']].extend(row['queries'])
    result = {'complete': set(ids) == expected_ids and bool(rows), 'scope': 'smoke' if limit else 'validation',
              'expected_episodes': len(expected_ids), 'completed_episodes': len(rows), 'queries': len(flat),
              'multi_version_episodes': sum(r['versions'] > 1 for r in rows),
              'historical_queries_on_changed_facts': len(historical), 'methods': methods,
              'single_entry_accuracy': methods['latent']['accuracy'],
              'single_entry_equals_indexed': bool(flat) and all(q['single_entry_equals_indexed'] for q in flat),
              'latent_payload_bytes_mean': mean([r['latent_accounting']['memory_bytes'] for r in rows]),
              'latent_total_bytes_mean': mean([r['latent_accounting']['total_bytes'] for r in rows]),
              'text_total_bytes_mean': mean([r['text_accounting']['total_bytes'] for r in rows]),
              'write_seconds_mean': mean([r['write_seconds'] for r in rows]),
              'group_macro_accuracy': mean([mean([q['scores']['latent']['accuracy'] for q in qs])
                                            for qs in grouped.values()]),
              'notes': ['Current/Historical subset definitions use original yearly labels.',
                        'Historical subset requires at least one observed fact change.',
                        'Single-entry and indexed memory identity is checked, not a duplicate trial.',
                        'Text/no-memory/wrong-version controls use the same frozen decoder.',
                        'A different version may share a valid answer; wrong-version is a routing control, not always a contradictory fact.',
                        'FP16 latent entries may use far more bytes than text; no storage superiority assumed.']}
    write_json(Path(root) / 'summary.json', result)
    return result


def gate_decision(config, summary, adaptation):
    if summary.get('scope') != 'validation' or not summary.get('complete'):
        raise RuntimeError('A smoke/partial run cannot open the ICAE research gate')
    if not adaptation.get('completed') or adaptation.get('scope') != 'full_training_split':
        raise RuntimeError('Complete train-only encoder adaptation before accepting validation')
    latent, empty = summary['methods']['latent'], summary['methods']['no_memory']
    gap = latent['accuracy'] - empty['accuracy']
    checks = {
        'single_entry_readability': summary['single_entry_accuracy'] >= config['gate']['single_entry_accuracy_min'],
        'current_readability': latent['current_accuracy'] is not None and
                               latent['current_accuracy'] >= config['gate']['current_accuracy_min'],
        'historical_readability': latent['historical_f1'] is not None and
                                  latent['historical_f1'] >= config['gate']['historical_f1_min'],
        'uses_memory': gap >= config['gate']['latent_minus_no_memory_min'],
        'version_roundtrip': summary['single_entry_equals_indexed'],
        'changed_fact_coverage': summary['multi_version_episodes'] > 0 and
                                summary['historical_queries_on_changed_facts'] > 0}
    return {'decision': 'eligible_for_manual_review' if all(checks.values()) else 'hold_replace_or_fix_icae_checkpoint',
            'checks': checks, 'criteria': config['gate'], 'latent_minus_no_memory': gap,
            'codec_training_allowed': False, 'manual_review_required': True,
            'note': 'This is the only research-entry gate. TransMem/LoCoMo reproduction is archival, not required.'}
