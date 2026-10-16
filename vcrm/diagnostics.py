"""Fixed, post-hoc LongMemEval error controls; never a training or benchmark split."""
import math
from pathlib import Path
import statistics
import time

from .data import ingestion_view, query_view
from .io import digest, read_json, row_path, write_json

METHODS = ('selected_latent', 'selected_text', 'no_memory', 'oracle_latent', 'oracle_text')


def token_chunks(ids, maximum):
    if maximum < 1 or not ids:
        raise ValueError('Invalid input or chunk width')
    return [ids[i:i + maximum] for i in range(0, len(ids), maximum)]


def load_cases(config):
    folder = Path(config['diagnostic_root'])
    protocol = read_json(folder / 'protocol.json')
    ids = protocol['case_ids']
    if len(ids) != 24 or len(set(ids)) != 24:
        raise ValueError('Expected the immutable 24-case pilot protocol')
    episodes = {e['id']: e for e in read_json(Path(config['data_root']) / 'prepared/longmemeval_test.json')}
    result = []
    for eid in ids:
        case = read_json(row_path(folder, 'cases', eid))
        episode = episodes[eid]
        events = {e['id']: e for e in ingestion_view(episode)}
        selected = case['selected_ids']
        if case['id'] != eid or len(selected) != len(set(selected)):
            raise ValueError('Diagnostic case identity/selection mismatch')
        if not set(case['gold_ids']) <= set(selected) or case['baseline_accuracy'] != 0:
            raise ValueError('Case is not an all-evidence-selected baseline error')
        if case['query'] != episode['queries'][0]:
            # Archived query omits bookkeeping id, but all task-bearing fields must match.
            fields = ('question', 'time', 'answers', 'type', 'abstention')
            if any(case['query'][k] != episode['queries'][0][k] for k in fields):
                raise ValueError('Prepared data changed the archived diagnostic query')
        result.append({'id': eid, 'query': case['query'],
                       'selected': [events[k] for k in selected],
                       'oracle': [events[k] for k in case['gold_ids']]})
    return protocol, result


def audit(config, root):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(config['model_path'], local_files_only=True)
    protocol, cases = load_cases(config)
    rows = []
    for case in cases:
        contexts = {}
        for mode in ('selected', 'oracle'):
            events = case[mode]
            texts = [f"DATE: {e['time']}\n{e['text']}" for e in events]
            lengths = [len(tok.encode(t, add_special_tokens=True)) for t in texts]
            chunks = [token_chunks(list(range(n)), config['encoder_max_tokens']) for n in lengths]
            slots = sum(config['memory_tokens'] * math.ceil(len(c) /
                        (config['memory_tokens'] * config['compression_rate'])) for cs in chunks for c in cs)
            text_tokens = len(tok.encode('\n\n'.join(texts), add_special_tokens=False))
            contexts[mode] = {'event_ids': [e['id'] for e in events], 'event_tokens': lengths,
                              'chunks_per_event': [len(c) for c in chunks], 'latent_slots': slots,
                              'text_tokens': text_tokens, 'input_sha256': digest(texts)}
            question_tokens = len(tok.encode(case['query']['question'], add_special_tokens=False)) + 100
            if slots > config['diagnostic_read_latent_slots'] or max(slots, text_tokens) + question_tokens + config['diagnostic_max_answer_tokens'] > 32768:
                raise ValueError(f'{case["id"]} exceeds the declared complete-coverage pilot budget')
        rows.append({'id': case['id'], 'contexts': contexts})
    result = {'scope': 'fixed post-hoc 24-error diagnosis; not benchmark accuracy',
              'source_protocol_sha256': digest(protocol), 'cases': rows,
              'order': 'archived retrieval rank, identical selected content for latent/text',
              'long_entry_policy': 'Contiguous token chunks <=5120, one original BOS only; no crop. '
                                   'Concatenation beyond the official 5120-input protocol is an explicit wrapper extension.',
              'oracle_policy': 'Gold evidence is used only in labelled offline oracle controls',
              'judge_policy': 'Official LongMemEval prompt, custom Qwen judge; not official scores'}
    write_json(Path(root) / 'diagnostic_audit.json', result)
    return result


def evaluate(config, root, shard=0, shards=1):
    import torch
    from .reader import FrozenReader
    cfg = config | {'read_latent_slots': config['diagnostic_read_latent_slots'],
                    'max_answer_tokens': config['diagnostic_max_answer_tokens']}
    reader = FrozenReader(cfg)
    _, cases = load_cases(config)
    for case in cases[shard::shards]:
        destination = row_path(root, 'diagnostic_cases', case['id'])
        if destination.exists() and read_json(destination).get('complete'):
            continue
        row = read_json(destination) if destination.exists() else {'id': case['id'], 'answers': {}, 'encoding': {}}
        visible = query_view(case['query'])
        for mode in ('selected', 'oracle'):
            latent_method, text_method = mode + '_latent', mode + '_text'
            events = case[mode]
            if latent_method not in row['answers']:
                started = time.perf_counter()
                memories, traces = [], []
                for event in events:
                    ids = reader.tok.encode(reader.event_text(event), add_special_tokens=True)
                    chunks = token_chunks(ids, config['encoder_max_tokens'])
                    memory = torch.cat([reader.encode_ids(c) for c in chunks])
                    # Persist/reload every entry before reading, without source text.
                    from .store import MemoryStore
                    store = MemoryStore('icae_independent', 2 ** 50, reader.shape, config['key_dimensions'])
                    store.ingest(event, memory)
                    path = row_path(root, 'diagnostic_stores', case['id'] + ':' + event['id']).with_suffix('.memory')
                    store.save(path)
                    restored = MemoryStore.load(path, 2 ** 50)
                    tensor = restored.decode(restored.entries[0]).to(torch.float16)
                    if not torch.equal(memory, tensor):
                        raise AssertionError('ICAE persistence changed the memory')
                    memories.append(tensor)
                    traces.append({'event_id': event['id'], 'input_tokens': len(ids),
                                   'chunks': len(chunks), **restored.accounting()})
                bundle = torch.cat(memories)
                row['encoding'][mode] = {'events': traces, 'write_seconds': time.perf_counter() - started,
                                         'resident_latent_slots': int(bundle.shape[0])}
                row['answers'][latent_method] = reader.answer(visible, bundle)
                write_json(destination, row)
            if text_method not in row['answers']:
                row['answers'][text_method] = reader.answer(visible, text='\n\n'.join(reader.event_text(e) for e in events))
                write_json(destination, row)
        if 'no_memory' not in row['answers']:
            row['answers']['no_memory'] = reader.answer(visible)
        row['complete'] = set(row['answers']) == set(METHODS)
        write_json(destination, row)
        print(f'shard={shard} completed {case["id"]}', flush=True)


def summarize(config, root):
    from .judge import score
    _, cases = load_cases(config)
    queries = {c['id']: c['query'] for c in cases}
    rows = [read_json(p) for p in (Path(root) / 'diagnostic_cases').glob('*.json')]
    if len({r['id'] for r in rows}) != len(rows) or not {r['id'] for r in rows} <= set(queries):
        raise ValueError('Unexpected diagnostic output IDs')
    completed = [r for r in rows if r.get('complete')]
    for row in completed:
        row['scores'] = {m: score(config, queries[row['id']], row['answers'][m]['answer'], root) for m in METHODS}
        write_json(row_path(root, 'diagnostic_cases', row['id']), row)
    mean = lambda vs: statistics.mean(vs) if vs else None
    methods = {m: {'correct': sum(r['scores'][m]['accuracy'] for r in completed),
                   'n': len(completed),
                   'accuracy': mean([r['scores'][m]['accuracy'] for r in completed]),
                   'read_seconds_mean': mean([r['answers'][m]['read_seconds'] for r in completed]),
                   'prompt_tokens_mean': mean([r['answers'][m]['prompt_tokens'] for r in completed]),
                   'decoder_latent_slots_mean': mean([r['answers'][m]['read_latent_slots'] for r in completed]),
                   'generation_limit_hits': sum(r['answers'][m]['generation_limit_hit'] for r in completed)} for m in METHODS}
    complete = len(completed) == 24
    correct = methods['selected_latent']['correct']
    signal = ('compression_candidate_pending_full_validation' if correct >= 10 else
              'diagnose_reading_or_routing_before_codec' if correct <= 4 else 'inconclusive_extend_validation') if complete else 'incomplete'
    result = {'complete': complete, 'scope': 'nonrepresentative fixed 24-error pilot', 'methods': methods,
              'engineering_signal': signal, 'codec_training_allowed': False,
              'warning': 'Thresholds do not prove cause or unlock research. Compare same-Mistral text, '
                         'inspect errors and pass independent validation. No training/tuning on these cases.'}
    write_json(Path(root) / 'diagnostic_summary.json', result)
    return result
