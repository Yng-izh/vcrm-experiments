from collections import defaultdict
from datetime import datetime
import re
from pathlib import Path

from .io import digest, read_json, read_records, write_json


def timestamp(value):
    value = str(value)
    if re.fullmatch(r'\d{4}', value):
        return value + '-01-01T00:00:00'
    for pattern in ('%Y/%m/%d (%a) %H:%M', '%Y-%m-%dT%H:%M:%S', '%I:%M %p on %d %B, %Y'):
        try:
            return datetime.strptime(value, pattern).isoformat()
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(value.replace('Z', '+00:00')).isoformat()
    except ValueError:
        pass
    raise ValueError(f'Unrecognized timestamp: {value}')


def templama(path, split):
    groups = defaultdict(list)
    for row in read_records(path):
        entity = row['id'].split('_')[0]
        groups[(entity, row['relation'])].append(row)
    result = []
    for (entity, relation), records in sorted(groups.items()):
        records.sort(key=lambda r: int(r['date']))
        if len({r['date'] for r in records}) != len(records):
            raise ValueError('Duplicate entity-relation-year query')
        events, queries = [], []
        previous_answers = None
        for row in records:
            answers = [a['name'] for a in row['answer']]
            date = timestamp(row['date'])
            # Published yearly facts are the stream input, not extra inferred facts.
            text = row['query'].replace('_X_', ' / '.join(answers))
            answer_set = tuple(sorted(a['wikidata_id'] if 'wikidata_id' in a else a['name'] for a in row['answer']))
            if answer_set != previous_answers:
                events.append({'id': row['id'], 'time': date, 'text': f"In {row['date']}: {text}",
                               'state_key': f'{entity}:{relation}', 'key_text': row['query']})
            previous_answers = answer_set
            queries.append({'id': row['id'], 'question': row['query'], 'time': date,
                            'as_of': date, 'answers': answers, 'latest': row is records[-1],
                            'type': 'latest' if row is records[-1] else 'historical'})
        result.append({'id': f'templama:{split}:{entity}:{relation}', 'dataset': 'templama',
                       'split': split, 'entity': entity, 'group': f'{entity}:{relation}',
                       'events': events, 'queries': queries})
    return result


def longmemeval(path):
    records = read_json(path)
    if len(records) != 500:
        raise ValueError('Formal LongMemEval preparation requires all 500 cleaned instances')
    episodes = []
    for row in records:
        values = zip(row['haystack_session_ids'], row['haystack_dates'], row['haystack_sessions'])
        if not len(row['haystack_session_ids']) == len(row['haystack_dates']) == len(row['haystack_sessions']):
            raise ValueError('Misaligned session ids/dates')
        events = [{'id': f'event_{i:06d}', 'time': timestamp(date),
                   'text': '\n'.join(f"{turn['role']}: {turn['content']}" for turn in turns)}
                  for i, (sid, date, turns) in enumerate(values)]
        if len(events) != len(row['haystack_sessions']):
            raise ValueError('Misaligned session ids/dates')
        events.sort(key=lambda e: e['time'])
        q = {'id': row['question_id'], 'question': row['question'], 'time': timestamp(row['question_date']),
             'answers': [str(row['answer'])], 'type': row['question_type'],
             'abstention': row['question_id'].endswith('_abs')}
        episodes.append({'id': 'longmemeval:' + row['question_id'], 'dataset': 'longmemeval',
                         'split': 'test', 'group': row['question_id'], 'events': events, 'queries': [q]})
    return episodes


def locomo(path):
    from deltamem.eval.locomo_protocol import prepare_locomo_question
    episodes = []
    for sample in read_json(path):
        conv = sample['conversation']
        indices = sorted(int(k.split('_')[1]) for k in conv
                         if re.fullmatch(r'session_\d+', k))
        events = []
        for number in indices:
            text = '\n'.join(f"{t['speaker']}: {t['text']}" +
                             (f" [image: {t['blip_caption']}]" if t.get('blip_caption') else '')
                             for t in conv[f'session_{number}'])
            events.append({'id': f'session_{number}',
                           'time': timestamp(conv[f'session_{number}_date_time']), 'text': text})
        queries = []
        for i, qa in enumerate(sample['qa']):
            if int(qa['category']) not in (1, 2, 3, 4):
                continue
            spec = prepare_locomo_question(qa, sample_id=str(sample['sample_id']), question_index=i, seed=42)
            queries.append({'id': f"{sample['sample_id']}:{i}", 'question': spec.prompt_text,
                            'time': max(e['time'] for e in events), 'answers': [str(qa['answer'])],
                            'category': qa['category'], 'qa': qa, 'type': str(qa['category'])})
        episodes.append({'id': 'locomo:' + str(sample['sample_id']), 'dataset': 'locomo',
                         'split': 'test', 'group': str(sample['sample_id']),
                         'events': sorted(events, key=lambda e: e['time']), 'queries': queries})
    return episodes


def prepare(config):
    data = Path(config['data_root'])
    output = data / 'prepared'
    all_sets = {}
    for split in ('train', 'val', 'test'):
        all_sets['templama_' + split] = templama(data / 'templama' / f'{split}.json', split)
    entities = {k: {e['entity'] for e in values} for k, values in all_sets.items()}
    excluded_train = entities['templama_train'] & (entities['templama_val'] | entities['templama_test'])
    excluded_val = entities['templama_val'] & entities['templama_test']
    all_sets['templama_train'] = [e for e in all_sets['templama_train'] if e['entity'] not in excluded_train]
    all_sets['templama_val'] = [e for e in all_sets['templama_val'] if e['entity'] not in excluded_val]
    write_json(output / 'entity_split_audit.json', {
        'policy': 'Keep official test entirely unchanged; exclude overlapping entities from train and val. No reassignment.',
        'original_entity_counts': {k: len(v) for k, v in entities.items()},
        'excluded_train_entities': sorted(excluded_train), 'excluded_val_entities': sorted(excluded_val),
        'note': 'Official splits are entity-relation-disjoint, not globally entity-disjoint. This filtered training protocol is an explicit adaptation.',
    })
    all_sets['longmemeval_test'] = longmemeval(data / 'longmemeval_s_cleaned.json')
    all_sets['locomo_test'] = locomo(data / 'locomo10.json')
    manifest = {}
    for name, episodes in all_sets.items():
        write_json(output / f'{name}.json', episodes)
        manifest[name] = {'episodes': len(episodes), 'queries': sum(len(e['queries']) for e in episodes),
                          'events': sum(len(e['events']) for e in episodes), 'sha256': digest(episodes)}
    write_json(output / 'manifest.json', manifest)
    return manifest


def ingestion_view(episode):
    """No query answers, QA annotations or oracle evidence reach ingestion."""
    return [{k: event[k] for k in ('id', 'time', 'text', 'state_key', 'key_text') if k in event}
            for event in episode['events']]


def query_view(query):
    return {k: query[k] for k in ('id', 'question', 'time', 'as_of') if k in query}


def import_stale(path, config, provenance):
    raw = read_json(path)
    raw = raw['data'] if isinstance(raw, dict) else raw
    episodes = []
    for row in raw:
        sessions = row['haystack_session']
        times = row['timestamps']
        if len(sessions) != len(times):
            raise ValueError('STALE timestamps do not align with sessions')
        events = [{'id': str(i), 'time': timestamp(time),
                   'text': '\n'.join(f"{t['role']}: {t['content']}" for t in session)}
                  for i, (time, session) in enumerate(zip(times, sessions))]
        query_time = timestamp(row.get('query_time', times[-1]))
        queries = [{'id': f"{row['uid']}:{dimension}", 'question': row['probing_queries'][f'dim{dimension}_query'],
                    'time': query_time, 'type': f'dim{dimension}'} for dimension in (1, 2, 3)]
        episodes.append({'id': 'stale:' + row['uid'], 'dataset': 'stale', 'split': 'test', 'group': row['uid'],
                         'events': sorted(events, key=lambda e: e['time']), 'queries': queries})
    write_json(Path(config['data_root']) / 'prepared/stale_test.json', episodes)
    write_json(Path(config['data_root']) / 'stale_provenance.json', {
        'source_path': str(Path(path).resolve()), 'sha256': digest(Path(path).read_bytes()),
        'provenance': provenance, 'episodes': len(episodes),
        'note': 'No M_old/M_new/explanation/relevant-session indices enter memory ingestion. Generated extensions are not official benchmark results.'})
    return {'episodes': len(episodes), 'queries': len(episodes) * 3}
