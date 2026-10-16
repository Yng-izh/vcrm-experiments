"""Unified text/full/C/D snapshot controls on immutable historical + latest QA."""
from itertools import combinations
from pathlib import Path
import statistics
import time

import torch

from .data import ingestion_view, query_view
from .io import atomic_bytes, digest, file_digest, read_json, row_path, write_json
from .metrics import exact, f1, paired_bootstrap
from .refinement import conflicting
from .store import CurrentSnapshotStore, MemoryStore


METHODS = ('text', 'full', 'C', 'D')


def select_entries(entries, query):
    state, as_of = query['state_key'], query['as_of']
    versions = sorted((e for e in entries if e.get('state_key', e.get('state')) == state),
                      key=lambda e: (e['time'], e['id']))
    eligible = [e for e in versions if e['time'] <= as_of]
    if not eligible:
        raise ValueError('Requested observed version is missing')
    current, selected = versions[-1], eligible[-1]
    historical = selected if selected is not current else versions[-2] if len(versions) > 1 else None
    return [current] + ([historical] if historical is not None else [])


def prepare_selection(config, source_manifest, destination, expected=(481, 262, 172)):
    source = Path(config['data_root']) / 'prepared/templama_val.json'
    manifest = read_json(source_manifest)
    if manifest['split'] != 'val' or manifest['data_sha256'] != file_digest(source):
        raise ValueError('Historical manifest data/split mismatch')
    episodes = read_json(source)
    ids = manifest['query_ids']
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate historical query IDs')
    wanted, metadata, latest_ids = set(ids), {}, []
    for episode in episodes:
        historical = [q for q in episode['queries'] if q['id'] in wanted]
        if not historical:
            continue
        latest = [q for q in episode['queries'] if q['latest']]
        if len(latest) != 1 or any(q['latest'] for q in historical):
            raise ValueError('Expected historical selection and one latest QA per Episode')
        latest = latest[0]
        latest_ids.append(latest['id'])
        for q in historical + [latest]:
            if q['id'] in metadata:
                raise ValueError('Duplicate source query ID')
            is_conflict = not q['latest'] and conflicting(q['answers'], latest['answers'])
            if not q['latest']:
                original = manifest['metadata'][q['id']]
                if (original['episode_id'] != episode['id'] or original['historical_answers'] != q['answers'] or
                        original['current_answers'] != latest['answers'] or original['conflicting'] != is_conflict):
                    raise ValueError('Historical selection labels or Episode identity changed')
            selected = select_entries(ingestion_view(episode), query_view(q))
            if not q['latest'] and len(selected) != 2:
                raise ValueError('Historical diagnostic requires two observed versions')
            metadata[q['id']] = {'episode_id': episode['id'], 'latest': q['latest'],
                                  'answers': q['answers'], 'current_answers': latest['answers'],
                                  'conflicting': is_conflict, 'as_of': q['as_of'],
                                  'selected_entry_ids': [e['id'] for e in selected]}
    if wanted != {q for q, v in metadata.items() if not v['latest']}:
        raise ValueError('Missing historical query coverage')
    counts = (len(ids), sum(v['conflicting'] for v in metadata.values()), len(latest_ids))
    if counts != expected:
        raise ValueError(f'Expected historical/conflict/latest counts {expected}, got {counts}')
    selection = {'split': 'val', 'data_sha256': file_digest(source),
                 'source_manifest_sha256': file_digest(source_manifest),
                 'historical_query_ids': ids, 'latest_query_ids': latest_ids,
                 'query_ids': ids + latest_ids, 'metadata': metadata,
                 'counts': dict(zip(('historical', 'conflict', 'latest'), counts)),
                 'routing': 'Explicit state_key + as_of; current + target history; latest + previous history',
                 'scope': 'post_hoc_fixed_validation_diagnostic_not_untouched_test'}
    if Path(destination).exists() and read_json(destination) != selection:
        raise ValueError('Frozen comparison selection changed; choose a new run prefix')
    write_json(destination, selection)
    return selection


def selected_data(config):
    selection = read_json(config['comparison_selection'])
    source = Path(config['data_root']) / 'prepared/templama_val.json'
    if selection['split'] != 'val' or selection['data_sha256'] != file_digest(source):
        raise ValueError('Comparison selection data changed')
    episodes = read_json(source)
    wanted = set(selection['query_ids'])
    rows = [e | {'queries': [q for q in e['queries'] if q['id'] in wanted]} for e in episodes]
    rows = [e for e in rows if e['queries']]
    actual = [q['id'] for e in rows for q in e['queries']]
    if len(actual) != len(wanted) or set(actual) != wanted:
        raise ValueError('Comparison query coverage mismatch')
    return rows, selection


def freeze_protocol(config, root):
    from .features import check
    method = config['comparison_method']
    if method not in METHODS or config['max_answer_tokens'] != 64 or config.get('adapted_checkpoint'):
        raise ValueError('Comparison requires supported method, official ICAE weights, greedy 64 tokens')
    episodes, selection = selected_data(config)
    provenance = {'config': config, 'selection_sha256': file_digest(config['comparison_selection']),
                  'method': method, 'episodes': len(episodes), 'counts': selection['counts'],
                  'model_revision': config['model_revision'], 'max_answer_tokens': 64,
                  'decoding': 'greedy', 'storage_MB_divisor': 1000000,
                  'code': {str(p.relative_to(Path(__file__).parents[1])): file_digest(p)
                           for folder in (Path(__file__).parent, Path(__file__).parents[1])
                           for p in folder.glob('*.py')},
                  'scope': selection['scope'], 'routing': selection['routing']}
    if method != 'text':
        parity = read_json(Path(config['project']) / 'icae_parity.json')
        if (not parity.get('chunked_prefill_greedy_match') or
                not parity.get('official_encoder_exact_fp16_match') or
                parity.get('reader_sha256') != file_digest(Path(__file__).with_name('reader.py')) or
                parity.get('checkpoint_sha256') != file_digest(config['checkpoint']) or
                parity.get('source_sha256') != config['icae_source_sha256'] or
                parity.get('model_revision') != config['model_revision']):
            raise ValueError('ICAE implementation parity is stale or incomplete')
        cache = check(config, 'val')
        provenance['feature_reader_identity'] = cache['reader_identity']
        name = config['comparison_training_run'] if method in ('C', 'D') else config['full_marker_run']
        if Path(name).name != name:
            raise ValueError('Training reference must be a single run name')
        training = Path(config['output_root']) / name
        metadata = read_json(training / 'codec_training.json')
        if (not metadata['completed'] or metadata['memory_protocol'] != 'final_current_snapshot_v1' or
                metadata['checkpoint_sha256'] != file_digest(training / 'codec.safetensors') or
                metadata['history_slots'] != 16 or
                metadata['reader_sha256'] != file_digest(Path(__file__).with_name('reader.py')) or
                metadata['base_checkpoint_sha256'] != file_digest(config['checkpoint']) or
                metadata['model_revision'] != config['model_revision'] or
                metadata['adapted_checkpoint_sha256'] is not None):
            raise ValueError('Incomplete, altered or incompatible snapshot training reference')
        if method in ('C', 'D') and metadata['method'] != method:
            raise ValueError('C/D training method mismatch')
        provenance['training_metadata_sha256'] = file_digest(training / 'codec_training.json')
        provenance['training_checkpoint_sha256'] = metadata['checkpoint_sha256']
        provenance['marker_source'] = name
        provenance['full_is_official_paper_result'] = False
    path = Path(root) / 'protocol.json'
    if path.exists() and read_json(path) != provenance:
        raise ValueError('Comparison protocol changed; choose a new run prefix')
    write_json(path, provenance)
    return provenance


def evaluate(config, root, shard=0, shards=1):
    from .features import check, load
    from .runtime import SlotMemoryRuntime
    from .text_reader import NativeTextReader
    method = config['comparison_method']
    episodes, selection = selected_data(config)
    protocol = read_json(Path(root) / 'protocol.json')
    if protocol['config'] != config or protocol['selection_sha256'] != file_digest(config['comparison_selection']):
        raise ValueError('Worker comparison protocol mismatch')
    runtime = None
    if method == 'text':
        reader = NativeTextReader(config)
    else:
        check(config, 'val')
        name = config['comparison_training_run'] if method in ('C', 'D') else config['full_marker_run']
        runtime = SlotMemoryRuntime.from_checkpoint(config, Path(config['output_root']) / name)
        reader = runtime.reader
    for episode in episodes[shard::shards]:
        destination = row_path(root, 'episodes', episode['id'])
        if destination.exists():
            continue
        events, memories, started = ingestion_view(episode), {}, time.perf_counter()
        if method != 'text':
            memories = {e['id']: load(config, 'val', episode['id'], e) for e in events}
        if method in ('C', 'D'):
            store = CurrentSnapshotStore(runtime.codec, config['resident_slot_budget'],
                                         config['keyframe_interval'], config['key_dimensions'])
            store.build(events, memories, {eid: runtime.index.key(m) for eid, m in memories.items()})
        else:
            # No eviction: disk size is a measurement, not a latent-vs-text equal-byte claim.
            store = MemoryStore('text_log' if method == 'text' else 'icae_independent',
                                2 ** 40, (128, 4096) if method == 'text' else reader.shape,
                                config['key_dimensions'])
            for event in events:
                receipt = store.ingest(event, memories.get(event['id']))
                if receipt['dropped']:
                    raise ValueError('Main table forbids eviction')
        path = row_path(root, 'stores', episode['id']).with_suffix('.memory')
        store.save(path)
        before = store.serialize()
        store = (CurrentSnapshotStore.load(path, runtime.codec, config['resident_slot_budget'])
                 if method in ('C', 'D') else MemoryStore.load(path, 2 ** 40))
        if before != store.serialize():
            raise AssertionError('Store serialization roundtrip changed bytes')
        if method in ('C', 'D'):
            runtime.store = store
        write_seconds = time.perf_counter() - started
        partial = row_path(root, 'partial_episodes', episode['id'])
        outputs = read_json(partial)['queries'] if partial.exists() else []
        finished = {q['query_id'] for q in outputs}
        for query in episode['queries']:
            if query['id'] in finished:
                continue
            selected = select_entries(store.entries, query_view(query))
            ids = [e['id'] for e in selected]
            if ids != selection['metadata'][query['id']]['selected_entry_ids']:
                raise ValueError('Methods must share exactly the same selected version IDs')
            if method == 'text':
                text = '\n\n'.join(f"ROLE: {'current' if i == 0 else 'historical'}\n"
                                     f"DATE: {e['time']}\n{e['payload'].decode()}" for i, e in enumerate(selected))
                result = reader.answer(query_view(query), text=text)
            elif method == 'full':
                with torch.no_grad():
                    bundle = torch.cat([runtime.markers(store.decode(e),
                                        'current' if i == 0 else 'historical', e['time'])
                                        for i, e in enumerate(selected)])
                    result = reader.answer(query_view(query), bundle)
            else:
                result = runtime.answer(query_view(query))
                if result['selected_entry_ids'] != ids:
                    raise ValueError('Codec runtime route differs from unified route')
            meta = selection['metadata'][query['id']]
            correct = exact(result['answer'], query['answers'])
            outputs.append({'query_id': query['id'], 'latest': query['latest'],
                            'conflicting': meta['conflicting'], 'selected_entry_ids': ids,
                            'answer': result, 'score': {'accuracy': correct,
                                'f1': f1(result['answer'], query['answers']),
                                'stale_answer': int(not query['latest'] and not correct and
                                                    exact(result['answer'], meta['current_answers']))}})
            write_json(partial, {'id': episode['id'], 'queries': outputs})
            print(f'comparison {method} shard={shard} episode={episode["id"]} query={len(outputs)}', flush=True)
        write_json(destination, {'id': episode['id'], 'queries': outputs, 'versions': len(events),
                                'accounting': store.accounting(), 'store_file_bytes': path.stat().st_size,
                                'store_sha256': file_digest(path),
                                'roundtrip_equal': True, 'write_seconds': write_seconds,
                                'write_cost_note': 'From cached full features; excludes ICAE encoding cost'})
        partial.unlink(missing_ok=True)


def collect(config, root):
    episodes, selection = selected_data(config)
    expected = {e['id']: {q['id'] for q in e['queries']} for e in episodes}
    files = sorted((Path(root) / 'episodes').glob('*.json'))
    rows, results = {}, {}
    for path in files:
        episode = read_json(path)
        eid = episode['id']
        if eid in rows or eid not in expected:
            raise ValueError('Duplicate or unexpected comparison Episode')
        ids = [q['query_id'] for q in episode['queries']]
        if len(ids) != len(set(ids)) or set(ids) != expected[eid]:
            raise ValueError('Incomplete comparison Episode query coverage')
        store_path = row_path(root, 'stores', eid).with_suffix('.memory')
        if (not episode['roundtrip_equal'] or store_path.stat().st_size != episode['store_file_bytes'] or
                file_digest(store_path) != episode['store_sha256'] or
                episode['store_file_bytes'] != episode['accounting']['total_bytes']):
            raise ValueError('Comparison storage accounting mismatch')
        rows[eid] = episode
        for q in episode['queries']:
            meta = selection['metadata'][q['query_id']]
            if (meta['episode_id'] != eid or q['selected_entry_ids'] != meta['selected_entry_ids'] or
                    q['latest'] != meta['latest'] or q['conflicting'] != meta['conflicting']):
                raise ValueError('Comparison query identity/route mismatch')
            answer = q['answer']['answer']
            results[q['query_id']] = q | {'episode_id': eid,
                'score': {'accuracy': exact(answer, meta['answers']), 'f1': f1(answer, meta['answers'])}}
    return rows, results, selection, set(rows) == set(expected)


def summarize(config, root):
    rows, results, selection, complete = collect(config, root)
    def metric(ids, key):
        return statistics.mean(results[q]['score'][key] for q in ids) if complete else None
    historical, latest = selection['historical_query_ids'], selection['latest_query_ids']
    conflicts = [q for q in historical if selection['metadata'][q]['conflicting']]
    mean_cost = lambda key: statistics.mean(q['answer'][key] for q in results.values()) if results else None
    storage = statistics.mean(r['store_file_bytes'] for r in rows.values()) if complete else None
    result = {'method': config['comparison_method'], 'complete': complete,
              'counts': selection['counts'], 'completed_episodes': len(rows), 'scored_queries': len(results),
              'historical_f1': metric(historical, 'f1'), 'conflict_f1': metric(conflicts, 'f1'),
              'latest_accuracy': metric(latest, 'accuracy'), 'disk_bytes_mean': storage,
              'storage_MB': storage / 1e6 if storage is not None else None,
              'storage_MiB': storage / 2 ** 20 if storage is not None else None,
              'prompt_tokens_mean': mean_cost('prompt_tokens'), 'kv_cache_bytes_mean': mean_cost('kv_cache_bytes'),
              'read_seconds_mean': mean_cost('read_seconds'), 'scope': selection['scope'],
              'storage_note': 'All versions + keys + metadata per Episode; excludes shared model/codec weights '
                              'and source feature cache. MB=1e6 bytes; not equal-byte text/latent comparison.'}
    write_json(Path(root) / 'summary.json', result)
    return result


def report(config, root, run_roots):
    summaries, results = {}, {}
    selection_hashes = set()
    for method, path in run_roots.items():
        cfg = read_json(Path(path) / 'protocol.json')['config']
        if cfg['comparison_method'] != method:
            raise ValueError('Report method mismatch')
        selection_hashes.add(file_digest(cfg['comparison_selection']))
        summaries[method] = summarize(cfg, path)
        if not summaries[method]['complete']:
            raise ValueError('Do not publish a partial comparison table')
        _, results[method], selection, _ = collect(cfg, path)
    if len(selection_hashes) != 1 or set(run_roots) != set(METHODS):
        raise ValueError('Four methods must share one selection')
    paired = {}
    scopes = {'historical_f1': (selection['historical_query_ids'], 'f1'),
              'conflict_f1': ([q for q in selection['historical_query_ids']
                               if selection['metadata'][q]['conflicting']], 'f1'),
              'latest_accuracy': (selection['latest_query_ids'], 'accuracy')}
    for a, b in combinations(METHODS, 2):
        paired[f'{b}_minus_{a}'] = {name: paired_bootstrap(
            [results[b][q]['score'][key] for q in ids], [results[a][q]['score'][key] for q in ids],
            [selection['metadata'][q]['episode_id'] for q in ids], seed=config['seed'])
            for name, (ids, key) in scopes.items()}
    result = {'summaries': summaries, 'paired_episode_bootstrap': paired,
              'units': 'Accuracy/F1 in JSON: fraction; Markdown: percent. Storage: decimal MB.',
              'scope': selection['scope'],
              'full_marker_source': read_json(Path(run_roots['full']) / 'protocol.json')['marker_source'],
              'note': 'ICAE-Full is a project dual-version protocol control, not an official paper result or '
                      'guaranteed upper bound. Text uses native Mistral chat; latent uses ICAE task template. '
                      'Storage is measured, not forced equal to text. Historical scores from older runs '
                      'are not substituted. Codec/marker training provenance is saved per method.'}
    write_json(Path(root) / 'table.json', result)
    lines = ['| Method | History Slots | Historical F1 (%) | Conflict F1 (%) | Latest Acc (%) | Storage (MB) |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    labels = {'text': ('Text Version Log', '-'), 'full': ('ICAE-Full', '128'),
              'C': ('Independent Latent (C16)', '16'), 'D': ('VCRM (D16)', '16')}
    for method in METHODS:
        s, (label, slots) = summaries[method], labels[method]
        lines.append(f'| {label} | {slots} | {100*s["historical_f1"]:.2f} | '
                     f'{100*s["conflict_f1"]:.2f} | {100*s["latest_accuracy"]:.2f} | {s["storage_MB"]:.3f} |')
    lines += ['', result['note'], '', f'ICAE-Full marker source: `{result["full_marker_source"]}`.',
              'Storage includes all persisted versions and metadata, not only the two read versions.']
    atomic_bytes(Path(root) / 'TABLE.md', ('\n'.join(lines) + '\n').encode())
    return result
