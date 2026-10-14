import json

import numpy as np
import pytest
import torch

from vcrm.codec import VersionCodec
from vcrm.data import ingestion_view, prepare_icae, query_view, templama
from vcrm.metrics import exact, f1
from vcrm.pipeline import gate_decision
from vcrm.store import MemoryStore
from vcrm.training import adaptation_samples


SHAPE = (4, 8)


def event(index):
    return {'id': f'event_{index}', 'state_key': 'Q1:P54', 'time': f'{2010+index}-01-01T00:00:00',
            'key_text': 'Alice works at', 'text': f'Alice works at company {index}'}


def test_variable_segments_roundtrip_and_accounting(tmp_path):
    store = MemoryStore('icae_independent', 10000, SHAPE, 8)
    store.ingest(event(0), torch.arange(64).reshape(8, 8))
    path = tmp_path / 'memory'
    store.save(path)
    reloaded = MemoryStore.load(path, 10000)
    assert reloaded.serialize() == path.read_bytes()
    assert reloaded.decode(reloaded.entries[0]).shape == (8, 8)
    accounting = reloaded.accounting()
    assert accounting['memory_bytes'] == 128
    assert accounting['index_key_bytes'] == 16
    assert accounting['total_bytes'] == path.stat().st_size
    assert sum(accounting[k] for k in ('memory_bytes', 'index_key_bytes', 'metadata_and_format_bytes')) == path.stat().st_size
    assert b'Alice' not in path.read_bytes()


def test_version_index_no_future_and_no_reencoding():
    store = MemoryStore('icae_independent', 10000, SHAPE, 8)
    for index in range(3):
        store.ingest(event(index), torch.ones(SHAPE) * index)
    for index in range(3):
        query = {'question': 'Alice works at', 'state_key': 'Q1:P54', 'as_of': event(index)['time']}
        selected = store.select(query, 4)
        assert selected[0]['id'] == event(index)['id']
        assert torch.equal(store.bundle(selected), torch.ones(SHAPE) * index)
    assert store.select({'question': 'Alice works at', 'time': '2000', 'state_key': 'Q1:P54'}, 4) == []


def test_slot_limit_uses_entry_shape_not_hidden_dimension():
    store = MemoryStore('icae_independent', 10000, SHAPE, 8)
    store.ingest(event(0), torch.zeros(8, 8))
    query = {'question': 'Alice works at', 'time': event(0)['time']}
    assert store.select(query, 7) == []
    assert len(store.select(query, 8)) == 1


def test_duplicate_and_invalid_entries():
    store = MemoryStore('icae_independent', 10000, SHAPE, 8)
    with pytest.raises(ValueError):
        store.ingest(event(0), torch.zeros(5, 8))
    with pytest.raises(ValueError):
        store.ingest(event(0), torch.full(SHAPE, float('nan')))
    store.ingest(event(0), torch.zeros(SHAPE))
    with pytest.raises(ValueError, match='Duplicate'):
        store.ingest(event(0), torch.zeros(SHAPE))


def test_fifo_and_text_bytes():
    store = MemoryStore('icae_text', 1100, SHAPE, 8)
    for index in range(8):
        store.ingest(event(index))
    assert len(store.serialize()) <= 1100
    assert store.entries[-1]['id'] == 'event_7'
    assert store.accounting()['memory_bytes'] == sum(len(e['payload']) for e in store.entries)


def test_legacy_codec_disabled():
    with pytest.raises(RuntimeError, match='retired'):
        VersionCodec(8, 4, True)


def test_training_rejects_eval_and_views_strip_labels():
    with pytest.raises(ValueError, match='training'):
        adaptation_samples([{'split': 'val', 'dataset': 'templama'}])
    query = {'id': 'q', 'question': 'q', 'time': '2010', 'state_key': 's', 'answers': ['oracle']}
    assert 'answers' not in query_view(query)
    assert query_view(query)['state_key'] == 's'
    assert 'answer_session_ids' not in ingestion_view({'events': [event(0) | {'answer_session_ids': ['x']}]})[0]


def test_preparation_preserves_test_and_filters_entities(tmp_path):
    folder = tmp_path / 'templama'
    folder.mkdir()
    def row(entity, year):
        return {'id': f'{entity}_P54_{year}', 'relation': 'P54', 'date': str(year),
                'query': f'{entity} works at _X_.', 'answer': [{'name': str(year)}]}
    values = {'train': [row('Q1', 2010), row('Q3', 2010)],
              'val': [row('Q2', 2010), row('Q1', 2010)], 'test': [row('Q4', 2010), row('Q1', 2011)]}
    for split, records in values.items():
        (folder / f'{split}.json').write_text(json.dumps(records))
    manifest = prepare_icae({'data_root': str(tmp_path)})
    assert manifest['splits']['train']['episodes'] == 1
    assert manifest['splits']['val']['episodes'] == 1
    assert json.loads((tmp_path / 'prepared/templama_test.json').read_text()) == templama(folder / 'test.json', 'test')


def good_summary():
    return {'scope': 'validation', 'complete': True, 'methods': {
        'latent': {'accuracy': .9, 'current_accuracy': .9, 'historical_f1': .9},
        'no_memory': {'accuracy': .2}}, 'single_entry_accuracy': .9,
        'single_entry_equals_indexed': True, 'multi_version_episodes': 3,
        'historical_queries_on_changed_facts': 20}


def gate_config():
    return {'gate': {'single_entry_accuracy_min': .8, 'current_accuracy_min': .8,
                     'historical_f1_min': .8, 'latent_minus_no_memory_min': .1}}


def test_gate_cannot_unlock_on_smoke_or_no_adaptation():
    with pytest.raises(RuntimeError, match='smoke'):
        gate_decision(gate_config(), good_summary() | {'scope': 'smoke'}, {})
    with pytest.raises(RuntimeError, match='adaptation'):
        gate_decision(gate_config(), good_summary(), {})


def test_gate_needs_historical_coverage_and_manual_review():
    adaptation = {'completed': True, 'scope': 'full_training_split'}
    result = gate_decision(gate_config(), good_summary(), adaptation)
    assert result['decision'] == 'eligible_for_manual_review'
    assert result['codec_training_allowed'] is False
    result = gate_decision(gate_config(), good_summary() | {'multi_version_episodes': 0}, adaptation)
    assert result['decision'] == 'hold_replace_or_fix_icae_checkpoint'


def test_gate_cannot_pass_without_memory_gain():
    summary = good_summary()
    summary['methods']['no_memory']['accuracy'] = .89
    result = gate_decision(gate_config(), summary, {'completed': True, 'scope': 'full_training_split'})
    assert result['checks']['uses_memory'] is False


def test_retained_metrics():
    assert exact('The team', ['Team']) == 1
    assert f1('one two', ['one two three']) == .8


def test_index_accounting_matches_fp16_wire_format_before_reload():
    store = MemoryStore('icae_text', 10000, SHAPE, 8)
    store.ingest(event(0))
    store.entries[0]['key'] = store.entries[0]['key'].astype(np.float32)
    assert store.entries[0]['key'].dtype == np.float32
    assert store.accounting()['index_key_bytes'] == 16


def test_old_transmem_three_dimensional_shape_still_roundtrips(tmp_path):
    shape = (2, 4, 8)
    store = MemoryStore('version_index', 10000, shape, 8)
    store.ingest(event(0), torch.zeros(shape))
    path = tmp_path / 'old.memory'
    store.save(path)
    loaded = MemoryStore.load(path, 10000)
    chosen = loaded.select({'question': 'Alice works at'}, 4)
    assert loaded.bundle(chosen).shape == shape
