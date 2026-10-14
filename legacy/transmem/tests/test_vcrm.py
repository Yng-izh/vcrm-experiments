import json
from pathlib import Path

import numpy as np
import pytest
import torch

from vcrm.codec import VersionCodec
from vcrm.data import import_stale, ingestion_view, query_view, templama, timestamp
from vcrm.index import inferred_as_of
from vcrm.metrics import exact, f1, paired_bootstrap
from vcrm.store import MemoryStore
from vcrm.pipeline import checkpoint_for
from vcrm.training import feature_signature


SHAPE = (2, 2, 4)


def event(i, state='s'):
    return {'id': str(i), 'state_key': state, 'key_text': 'Alice works for',
            'time': f'{2000 + i}-01-01T00:00:00', 'text': f'Alice works for company {i}'}


class ExactResidual:
    def encode(self, old, current):
        return old - current

    def decode(self, code, current):
        return code + current


def test_budget_includes_all_file_bytes(tmp_path):
    store = MemoryStore('version_index', 1800, SHAPE, 8)
    for i in range(20):
        store.ingest(event(i), torch.ones(SHAPE) * i)
        file = tmp_path / 'memory'
        size = store.save(file)
        assert size == file.stat().st_size <= 1800
        totals = store.accounting()
        assert totals['total_bytes'] == totals['memory_bytes'] + totals['index_key_bytes'] + totals['metadata_and_format_bytes']
    loaded = MemoryStore.load(file, 1800)
    assert loaded.serialize() == file.read_bytes()


def test_residual_archive_survives_anchor_changes(tmp_path):
    store = MemoryStore('vcrm', 100000, SHAPE, 8, ExactResidual())
    for i in range(5):
        store.ingest(event(i), torch.ones(SHAPE) * i)
    file = tmp_path / 'memory'
    store.save(file)
    loaded = MemoryStore.load(file, 100000, ExactResidual())
    for i in range(5):
        selected = loaded.select({'question': 'Alice works for', 'as_of': event(i)['time']}, 8)
        assert len(selected) == 1
        assert selected[0]['id'] == str(i)
        assert torch.equal(loaded.bundle(selected), torch.ones(SHAPE) * i)


def test_nonoracle_shared_views():
    episode = {'events': [event(0) | {'has_answer': True, 'answer_session_ids': ['oracle']}], 'answers': ['gold']}
    assert 'has_answer' not in ingestion_view(episode)[0]
    assert 'answer_session_ids' not in ingestion_view(episode)[0]
    assert query_view({'id': 'q', 'time': '2000', 'question': 'q', 'answers': ['gold']}) == {
        'id': 'q', 'time': '2000', 'question': 'q'}


def test_codec_parameter_matching_and_gradients():
    c, d = VersionCodec(4, 2, False), VersionCodec(4, 2, True)
    assert sum(p.numel() for p in c.parameters()) == sum(p.numel() for p in d.parameters())
    for codec in (c, d):
        old, current = torch.randn(SHAPE), torch.randn(SHAPE)
        loss = (codec(old, current) - old).square().mean()
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in codec.parameters())


def test_empty_store_and_invalid_budget(tmp_path):
    store = MemoryStore('transmem_streaming', 500, SHAPE, 8)
    assert store.bundle(store.select({'question': 'unknown'}, 4)) is None
    with pytest.raises(ValueError):
        MemoryStore('version_index', 1, SHAPE)
    with pytest.raises(ValueError):
        MemoryStore('vcrm', 10000, SHAPE)


def test_temporal_selection_is_not_query_date_or_future_access():
    assert inferred_as_of({'question': 'What is current?', 'time': '2023-01-01'}) is None
    assert inferred_as_of({'question': 'Where did I work in 2012?'}) == '2012-12-31T23:59:59'
    store = MemoryStore('version_index', 100000, SHAPE, 8)
    store.ingest(event(0), torch.zeros(SHAPE))
    store.ingest(event(2), torch.ones(SHAPE))
    assert store.select({'question': 'Alice works for', 'as_of': event(1)['time']}, 4)[0]['id'] == '0'


def test_templama_preserves_queries_and_multiple_answers(tmp_path):
    records = [{'id': 'Q1_P54_2010', 'relation': 'P54', 'date': '2010', 'query': 'A plays for _X_.',
                'answer': [{'name': 'One'}, {'name': 'Two'}]},
               {'id': 'Q1_P54_2011', 'relation': 'P54', 'date': '2011', 'query': 'A plays for _X_.',
                'answer': [{'name': 'Three'}]}]
    path = tmp_path / 'data'
    path.write_text('\n'.join(json.dumps(r) for r in records))
    ep = templama(path, 'train')[0]
    assert ep['queries'][0]['answers'] == ['One', 'Two']
    assert ep['queries'][0]['question'] == records[0]['query']
    assert ep['queries'][-1]['latest']
    assert 'most_recent_answer' not in str(ep)


def test_repeated_years_do_not_invent_extra_memory_versions(tmp_path):
    records = [{'id': f'Q1_P54_{year}', 'relation': 'P54', 'date': str(year),
                'query': 'A plays for _X_.', 'answer': [{'name': answer}]}
               for year, answer in ((2010, 'One'), (2011, 'One'), (2012, 'Two'), (2013, 'One'))]
    path = tmp_path / 'data'
    path.write_text('\n'.join(json.dumps(r) for r in records))
    ep = templama(path, 'train')[0]
    assert len(ep['queries']) == 4
    assert len(ep['events']) == 3
    assert ep['events'][-1]['time'].startswith('2013')


def test_metrics_and_cluster_bootstrap():
    assert exact('The Team', ['Team']) == 1
    assert f1('one two', ['one two three']) == .8
    result = paired_bootstrap([1, 1, 0], [0, 0, 0], ['g', 'g', 'h'], 100)
    assert result['groups'] == 2
    assert result['delta'] == pytest.approx(2 / 3)


def test_no_raw_text_survives_latent_store(tmp_path):
    store = MemoryStore('version_index', 100000, SHAPE, 8)
    store.ingest(event(0), torch.ones(SHAPE))
    assert b'Alice' not in store.serialize()
    assert b'company' not in store.serialize()


def test_archive_dependency_eviction_and_no_future_updates():
    store = MemoryStore('vcrm', 1500, SHAPE, 8, ExactResidual())
    for i in range(12):
        store.ingest(event(i), torch.ones(SHAPE) * i)
        for entry in store.entries:
            if entry['kind'] == 'code':
                assert store._current(entry['state']) is not None
                assert torch.isfinite(store.decode(entry)).all()
    with pytest.raises(ValueError):
        store.ingest(event(0), torch.zeros(SHAPE))


def test_independent_archive_is_not_reencoded_on_updates():
    class CountingIndependent:
        def __init__(self):
            self.calls = 0

        def encode(self, old, current):
            self.calls += 1
            return old

        def decode(self, code, current):
            return code
    codec = CountingIndependent()
    store = MemoryStore('independent', 100000, SHAPE, 8, codec)
    for i in range(5):
        store.ingest(event(i), torch.ones(SHAPE) * i)
    assert codec.calls == 4
    assert torch.equal(store.decode(store.entries[0]), torch.zeros(SHAPE))


def test_stale_import_strips_conflict_oracle_fields(tmp_path):
    raw = [{'uid': 'u', 'M_old': 'oracle-old', 'M_new': 'oracle-new', 'relevant_session_index': [0, 1],
            'timestamps': ['2024-01-01T00:00:00', '2025-01-01T00:00:00'],
            'haystack_session': [[{'role': 'user', 'content': 'first'}], [{'role': 'user', 'content': 'second'}]],
            'probing_queries': {f'dim{i}_query': f'query{i}' for i in (1, 2, 3)}}]
    path = tmp_path / 'source.json'
    path.write_text(json.dumps(raw))
    config = {'data_root': str(tmp_path)}
    assert import_stale(path, config, 'generated-extension')['queries'] == 3
    prepared = json.loads((tmp_path / 'prepared/stale_test.json').read_text())
    assert 'oracle' not in str(prepared)
    assert 'relevant_session_index' not in str(prepared)


def test_text_budget_and_roundtrip(tmp_path):
    store = MemoryStore('text_log', 1200, SHAPE, 8)
    for i in range(8):
        store.ingest(event(i))
    file = tmp_path / 'text.memory'
    store.save(file)
    loaded = MemoryStore.load(file, 1200)
    selected = loaded.select({'question': 'Alice works for'}, 8)
    assert str(2007) in loaded.text(selected)
    assert file.stat().st_size <= 1200


def test_independent_can_precede_vcrm_but_formal_d_requires_matched_c(tmp_path):
    config = {'project': str(tmp_path), 'model_path': 'model', 'checkpoint': 'reader',
              'dtype': 'float16', 'attention': 'sdpa', 'encode_chunk_tokens': 2048,
              'event_format': 'timestamped_session_v1'}
    folder = tmp_path / 'checkpoints'
    folder.mkdir()
    row = {'feature_signature': feature_signature(config), 'smoke_only': False,
           'training_ids': ['train-only'], 'training_data_sha256': 'data', 'steps': 4,
           'epochs': 2, 'rank': 128, 'trainable_parameters': 100}
    torch.save(row, folder / 'independent.pt')
    assert checkpoint_for(config, 'independent') == folder / 'independent.pt'
    torch.save(row | {'steps': 5}, folder / 'vcrm.pt')
    with pytest.raises(RuntimeError, match='not matched'):
        checkpoint_for(config, 'vcrm')
    torch.save(row, folder / 'vcrm.pt')
    assert checkpoint_for(config, 'vcrm') == folder / 'vcrm.pt'
