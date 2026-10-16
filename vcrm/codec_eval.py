"""Held-out, persistent temporal C/D evaluation with explicit state/time routing."""
from pathlib import Path
import statistics
import time

import torch

from .data import ingestion_view, query_view
from .io import read_json, row_path, write_json
from .metrics import exact, f1
from .pipeline import validation_subset
from .store import CurrentSnapshotStore, KeyframeStore
from .refinement import selected_episodes, diagnostic_metrics


def evaluate(config, root, training_root, split='val', limit=None, shard=0, shards=1):
    from .features import check, load
    from .runtime import SlotMemoryRuntime
    if split not in ('val', 'test'):
        raise ValueError('Codec evaluation cannot use training data')
    check(config, split)
    runtime = SlotMemoryRuntime.from_checkpoint(config, training_root)
    reader = runtime.reader
    episodes = validation_subset(read_json(Path(config['data_root']) / f'prepared/templama_{split}.json'), limit)
    all_queries = {e['id']: e['queries'] for e in episodes}
    episodes = selected_episodes(config, episodes, split)
    for i, episode in enumerate(episodes[shard::shards]):
        destination = row_path(root, 'episodes', episode['id'])
        if destination.exists():
            continue
        store_type = CurrentSnapshotStore if config.get('memory_protocol') == 'final_current_snapshot_v1' else KeyframeStore
        runtime.store = store_type(runtime.codec, config['resident_slot_budget'],
                                   config['keyframe_interval'], config['key_dimensions'])
        trace, started = [], time.perf_counter()
        source_memories = {}
        events = ingestion_view(episode)
        source_keys = {}
        for event in events:
            memory = load(config, split, episode['id'], event)
            source_memories[event['id']] = memory
            key = runtime.index.key(memory)
            if store_type is CurrentSnapshotStore:
                source_keys[event['id']] = key
            else:
                receipt = runtime.store.ingest(event, memory, key)
                if receipt['dropped']:
                    raise ValueError('Main readability/codec table forbids eviction; increase a preregistered shared budget')
                trace.append({'id': event['id'], **receipt})
        if store_type is CurrentSnapshotStore:
            trace.append(runtime.store.build(events, source_memories, source_keys))
        write_seconds = time.perf_counter() - started
        path = row_path(root, 'stores', episode['id']).with_suffix('.memory')
        runtime.save(path)
        before = runtime.store.serialize()
        runtime.store = store_type.load(path, runtime.codec, config['resident_slot_budget'])
        if before != runtime.store.serialize():
            raise AssertionError('Persistent codec store roundtrip changed bytes')
        outputs = []
        partial = row_path(root, 'partial_episodes', episode['id'])
        if partial.exists():
            outputs = read_json(partial)['queries']
        finished = {q['query_id'] for q in outputs}
        for query in episode['queries']:
            if query['id'] in finished:
                continue
            selected = runtime.store.select_version(query['state_key'], query['as_of'])
            result = runtime.answer(query_view(query))
            # Counterfactual teacher read uses identical learned markers and
            # selected entry IDs, replacing only the compressed representation.
            with torch.no_grad():
                entries = [runtime.store._entry(eid) for eid in result['selected_entry_ids']]
                pieces = [runtime.markers(source_memories[e['id']],
                          'current' if e is runtime.store.current(e['state']) else 'historical', e['time']) for e in entries]
                upper = reader.answer(query_view(query), torch.cat(pieces))
            upper_score = {'accuracy': exact(upper['answer'], query['answers']),
                           'f1': f1(upper['answer'], query['answers'])}
            score = {'accuracy': exact(result['answer'], query['answers']), 'f1': f1(result['answer'], query['answers'])}
            newer = [a for q in all_queries[episode['id']] if q['time'] > query['time'] for a in q['answers']]
            score['wrong_version'] = int(not score['accuracy'] and bool(newer) and exact(result['answer'], newer))
            outputs.append({'query_id': query['id'], 'latest': query['latest'], 'as_of': query['as_of'],
                            'selected_entry_id': selected['id'], 'selected_kind': selected['kind'],
                            'answer': result, 'score': score, 'uncompressed_pair': upper,
                            'uncompressed_pair_score': upper_score})
            write_json(partial, {'id': episode['id'], 'queries': outputs})
            print(f'codec-eval shard={shard} episode={i+1} query={len(outputs)}/{len(episode["queries"])} '
                  f'EM={score["accuracy"]}', flush=True)
        write_json(destination, {'id': episode['id'], 'group': episode['group'], 'split': split,
                                'versions': len(runtime.store.entries), 'queries': outputs,
                                'write_seconds': write_seconds, 'write_trace': trace,
                                'accounting': runtime.store.accounting(), 'roundtrip_equal': True})
        partial.unlink(missing_ok=True)


def summarize(config, root, split='val', limit=None):
    episodes = validation_subset(read_json(Path(config['data_root']) / f'prepared/templama_{split}.json'), limit)
    episodes = selected_episodes(config, episodes, split)
    expected = {e['id']: {q['id'] for q in e['queries']} for e in episodes}
    rows = [read_json(p) for p in (Path(root) / 'episodes').glob('*.json')]
    ids = [r['id'] for r in rows]
    if len(ids) != len(set(ids)) or not set(ids) <= expected.keys():
        raise ValueError('Unexpected or duplicate codec episodes')
    for row in rows:
        query_ids = [q['query_id'] for q in row['queries']]
        if len(query_ids) != len(set(query_ids)) or set(query_ids) != expected[row['id']]:
            raise ValueError('Incomplete yearly codec query coverage')
    flat = [q for r in rows for q in r['queries']]
    current = [q for q in flat if q['latest']]
    historical = [q for r in rows if r['versions'] > 1 for q in r['queries'] if not q['latest']]
    compressed = [q for q in historical if q['selected_kind'] == 'code']
    mean = lambda values: statistics.mean(values) if values else None
    result = {'complete': bool(rows) and set(ids) == set(expected),
              'scope': 'smoke' if limit else 'validation' if split == 'val' else 'held_out_test',
              'expected_episodes': len(expected), 'completed_episodes': len(rows), 'queries': len(flat),
              'current_accuracy': mean([q['score']['accuracy'] for q in current]),
              'historical_f1': mean([q['score']['f1'] for q in historical]),
              'uncompressed_pair_current_accuracy': mean([q['uncompressed_pair_score']['accuracy'] for q in current]),
              'uncompressed_pair_historical_f1': mean([q['uncompressed_pair_score']['f1'] for q in historical]),
              'compressed_history_f1': mean([q['score']['f1'] for q in compressed]),
              'compressed_history_queries': len(compressed),
              'accuracy': mean([q['score']['accuracy'] for q in flat]),
              'wrong_version_rate': mean([q['score']['wrong_version'] for q in historical]),
              'decoder_latent_slots_mean': mean([q['answer']['read_latent_slots'] for q in flat]),
              'auxiliary_anchor_slots_mean': mean([q['answer']['auxiliary_anchor_slots'] for q in flat]),
              'read_seconds_mean': mean([q['answer']['read_seconds'] for q in flat]),
              'recovery_seconds_mean': mean([q['answer']['codec_recovery_seconds'] for q in flat]),
              'kv_cache_bytes_mean': mean([q['answer']['kv_cache_bytes'] for q in flat]),
              'resident_latent_slots_mean': mean([r['accounting']['resident_latent_slots'] for r in rows]),
              'disk_bytes_mean': mean([r['accounting']['total_bytes'] for r in rows]),
              'write_seconds_mean': mean([r['write_seconds'] for r in rows]),
              'resident_slot_budget': config['resident_slot_budget'], 'history_slot_budget': config['history_slots'],
              'memory_protocol': config.get('memory_protocol', 'immutable_keyframe_v1'),
              'note': 'Current + selected historical co-injection, 2 markers per entry; immutable keyframes count in resident budget. '
                      'A/B single-entry baseline is a reference, not a co-injection-matched theoretical upper bound.'}
    if config.get('memory_protocol') == 'final_current_snapshot_v1':
        result['note'] = ('Offline final-current snapshots built from all original event features, then persisted/reloaded. '
                          'Only final current is full; every historical entry is m slots. Current is reused as D condition, '
                          'not counted twice. Source history features are builder inputs, not persisted reader inputs. '
                          'No online rebase/update claim. Old runs have different pairing and anchor protocols.')
    if config.get('evaluation_query_manifest'):
        result['scope'] = 'post_hoc_fixed_validation_diagnostic_not_untouched_test'
        if result['complete']:
            result['fixed_diagnostic'] = diagnostic_metrics({q['query_id']: q for q in flat},
                                                           read_json(config['evaluation_query_manifest']))
    write_json(Path(root) / 'summary.json', result)
    return result
