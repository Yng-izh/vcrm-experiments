import pytest

from vcrm.runtime import adapt_cache_keyword
from vcrm.data import longmemeval
from vcrm.io import write_json


def test_cache_keyword_compatibility_is_idempotent_and_rejects_ambiguity():
    class OldLayer:
        def forward(self, value, past_key_value=None, **kwargs):
            assert 'past_key_values' not in kwargs
            return value, past_key_value
    layer = OldLayer()
    marker = object()
    assert adapt_cache_keyword(OldLayer)
    assert adapt_cache_keyword(OldLayer)
    assert layer.forward(1, past_key_values=marker) == (1, marker)
    with pytest.raises(ValueError, match='Ambiguous'):
        layer.forward(1, past_key_values=marker, past_key_value=marker)
    class NewLayer:
        def forward(self, value, past_key_values=None):
            return value, past_key_values
    assert not adapt_cache_keyword(NewLayer)


def test_longmemeval_inference_ids_do_not_encode_gold_evidence(tmp_path):
    rows = [{'question_id': f'q{i}', 'question': 'where?', 'answer': 'gold',
             'question_type': 'knowledge-update', 'question_date': '2024/01/02 (Tue) 00:00',
             'haystack_session_ids': ['answer_leak', 'distractor'],
             'haystack_dates': ['2024/01/01 (Mon) 00:00'] * 2,
             'haystack_sessions': [[{'role': 'user', 'content': 'one', 'has_answer': True}],
                                  [{'role': 'user', 'content': 'two'}]]} for i in range(500)]
    path = tmp_path / 'data.json'
    write_json(path, rows)
    first = longmemeval(path)[0]
    assert [e['id'] for e in first['events']] == ['event_000000', 'event_000001']
    assert 'answer_leak' not in str(first)
    assert 'has_answer' not in str(first)
