from __future__ import annotations
# P0 边界回归测试：证明 Predicted 视图在结构上无法携带 Oracle 字段，
# 同时证明既有 Oracle 对照组入口（vcrm.data.ingestion_view / query_view）未被破坏。
# 用 stdlib unittest 编写，pytest 也可直接收集运行。

import pathlib
import unittest

from vcrm.data import ingestion_view, query_view
from vcrm.data_views import (
    ORACLE_FIELDS,
    PUBLIC_QUERY_KEYS,
    PUBLIC_WRITE_KEYS,
    TIME_MODES,
    TIME_SOURCES,
    PublicEventCounter,
    assert_no_oracle,
    assert_public_id_independent,
    public_query,
    public_write_event,
)
from vcrm.fact_schema import STATE_ACTIONS, AtomicFact, StateDecision

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

# 每个 Oracle 字段都带一个独一无二的哨兵值：只要它出现在 Predicted 视图里就能被抓住。
ORACLE_SENTINELS = {
    'state_key': 'SENTINEL_STATE_KEY_Q42_P54',
    'key_text': 'SENTINEL_KEY_TEXT_EMPLOYER',
    'entity': 'SENTINEL_ENTITY_Q42',
    'group': 'SENTINEL_GROUP_Q42_P54',
    'answers': ['SENTINEL_ANSWER_WIKIDATA'],
    'latest': True,
    'episode': 'SENTINEL_EPISODE_HANDLE',
    'as_of': 'SENTINEL_AS_OF_2020',
    'is_update': True,
    'target_event': 'SENTINEL_TARGET_EVENT',
}

SENTINEL_VALUES = [v for k, v in ORACLE_SENTINELS.items() if k not in ('latest', 'is_update')]


def oracle_event():
    event = {
        'id': 'Q42_2018-01-01',
        'time': '2018-01-01T00:00:00',
        'text': 'In 2018: Alice works at Acme',
    }
    event.update(ORACLE_SENTINELS)
    return event


def oracle_query():
    query = {'id': 'Q42_2018-01-01', 'question': 'Who does Alice work for?',
             'time': '2018-01-01T00:00:00'}
    query.update(ORACLE_SENTINELS)
    return query


class PredictedViewCannotCarryOracleTest(unittest.TestCase):
    def test_write_view_is_exactly_the_public_whitelist(self):
        view = public_write_event(oracle_event(), 'input_000001')
        self.assertEqual(set(view), set(PUBLIC_WRITE_KEYS))
        self.assertEqual(view['id'], 'input_000001')
        self.assertEqual(view['text'], 'In 2018: Alice works at Acme')
        self.assertEqual(view['time'], '2018-01-01T00:00:00')

    def test_write_view_contains_no_sentinel_value(self):
        view = public_write_event(oracle_event(), 'input_000001')
        flat = repr(view)
        for sentinel in SENTINEL_VALUES:
            self.assertNotIn(str(sentinel), flat)
        for forbidden in ORACLE_FIELDS:
            self.assertNotIn(forbidden, view)

    def test_query_view_is_exactly_the_public_whitelist(self):
        # 受控对照口径（as_of）才允许把 query['time'] 带出来
        view = public_query(oracle_query(), time_mode='as_of')
        self.assertEqual(set(view), set(PUBLIC_QUERY_KEYS))
        self.assertEqual(view['question'], 'Who does Alice work for?')
        self.assertEqual(view['time'], '2018-01-01T00:00:00')
        self.assertEqual(view['time_source'], 'as_of')
        flat = repr(view)
        for sentinel in SENTINEL_VALUES:
            self.assertNotIn(str(sentinel), flat)

    def test_assert_no_oracle_rejects_the_original_oracle_objects(self):
        with self.assertRaises(ValueError):
            assert_no_oracle(oracle_event(), 'raw_event')
        with self.assertRaises(ValueError):
            assert_no_oracle(oracle_query(), 'raw_query')
        # 嵌套结构同样被递归拦截
        with self.assertRaises(ValueError):
            assert_no_oracle({'batch': [{'ok': 1}, {'answers': ['x']}]}, 'nested')

    def test_assert_no_oracle_accepts_public_views_and_returns_them(self):
        view = public_write_event(oracle_event(), 'input_000007')
        self.assertIs(assert_no_oracle(view, 'ingest', allowed=PUBLIC_WRITE_KEYS), view)
        self.assertIs(assert_no_oracle(public_query(oracle_query()), 'query',
                                       allowed=PUBLIC_QUERY_KEYS)['question'],
                      'Who does Alice work for?')

    def test_strict_whitelist_rejects_unknown_keys(self):
        with self.assertRaises(ValueError):
            assert_no_oracle({'id': 'input_000001', 'text': 'x', 'time': 't', 'note': 'extra'},
                             'strict', allowed=PUBLIC_WRITE_KEYS)

    def test_public_write_event_requires_text_and_time(self):
        with self.assertRaises(ValueError):
            public_write_event({'id': 'x', 'time': 't'}, 'input_000001')
        with self.assertRaises(ValueError):
            public_write_event({'id': 'x', 'text': 'y'}, 'input_000001')
        with self.assertRaises(ValueError):
            public_write_event(oracle_event(), '')


class PublicIdIndependenceTest(unittest.TestCase):
    def test_counter_ids_are_sequential_and_entity_free(self):
        counter = PublicEventCounter()
        ids = [counter.next_id() for _ in range(3)]
        self.assertEqual(ids, ['input_000001', 'input_000002', 'input_000003'])
        self.assertEqual(counter.next_index, 4)
        self.assertEqual(counter(), 'input_000004')

    def test_public_id_has_no_substring_relation_with_source_id(self):
        counter = PublicEventCounter()
        for event in (oracle_event(), {'id': 'templama:train:Q42:P54', 'text': 'x', 'time': 't'},
                      {'id': 'longmemeval_0001_session_3', 'text': 'y', 'time': 't'}):
            public_id = counter.next_id()
            view = public_write_event(event, public_id)
            self.assertNotIn(event['id'], view['id'])
            self.assertNotIn(view['id'], event['id'])
            self.assertEqual(assert_public_id_independent(view['id'], event['id']), public_id)

    def test_id_independence_guard_catches_leaks(self):
        with self.assertRaises(ValueError):
            assert_public_id_independent('event_Q42_2018', 'Q42_2018')
        with self.assertRaises(ValueError):
            assert_public_id_independent('Q42_2018', 'event_Q42_2018')

    def test_counter_reset_and_validation(self):
        counter = PublicEventCounter(start=10)
        self.assertEqual(counter.next_id(), 'input_000010')
        counter.reset()
        self.assertEqual(counter.next_id(), 'input_000001')
        with self.assertRaises(ValueError):
            PublicEventCounter(start=0)


class OracleControlPathPreservedTest(unittest.TestCase):
    """既有 Oracle 对照组入口必须一行未改、继续可用。"""

    def episode(self):
        return {'id': 'ep1', 'events': [oracle_event()],
                'queries': [{'id': 'q1', 'question': 'Who?', 'time': '2018-01-01T00:00:00',
                             'as_of': '2018-01-01T00:00:00', 'state_key': 'Q42:P54',
                             'answers': ['Acme'], 'latest': True}]}

    def test_ingestion_view_still_exposes_state_key_and_key_text(self):
        events = ingestion_view(self.episode())
        self.assertEqual(len(events), 1)
        self.assertEqual(set(events[0]), {'id', 'time', 'text', 'state_key', 'key_text'})
        self.assertEqual(events[0]['state_key'], ORACLE_SENTINELS['state_key'])
        self.assertEqual(events[0]['key_text'], ORACLE_SENTINELS['key_text'])
        # Oracle 视图不受 Predicted 白名单约束：assert_no_oracle 必须拒绝它
        with self.assertRaises(ValueError):
            assert_no_oracle(events[0], 'oracle_ingestion_view')

    def test_query_view_still_exposes_state_key_and_as_of(self):
        visible = query_view(self.episode()['queries'][0])
        self.assertEqual(set(visible), {'id', 'question', 'time', 'as_of', 'state_key'})
        self.assertEqual(visible['state_key'], 'Q42:P54')

    def test_oracle_and_predicted_views_are_disjoint(self):
        oracle_keys = set(ingestion_view(self.episode())[0])
        predicted_keys = set(public_write_event(oracle_event(), 'input_000001'))
        self.assertTrue(oracle_keys & ORACLE_FIELDS)
        self.assertFalse(predicted_keys & ORACLE_FIELDS)
        self.assertEqual(predicted_keys, set(PUBLIC_WRITE_KEYS))


class FactSchemaTest(unittest.TestCase):
    def fact(self, **overrides):
        payload = {'id': 'fact_000001', 'text': 'Alice works at Acme',
                   'observed_at': '2018-01-01T00:00:00', 'subject': 'Alice',
                   'attribute': 'employer', 'value': 'Acme', 'source_id': 'input_000001'}
        payload.update(overrides)
        return AtomicFact(**payload)

    def test_state_identity_is_subject_plus_attribute_only(self):
        first, second = self.fact(), self.fact(value='Globex', observed_at='2020-01-01T00:00:00',
                                               id='fact_000002')
        self.assertEqual(first.state_identity, ('Alice', 'employer'))
        self.assertEqual(first.state_identity, second.state_identity)
        self.assertEqual(first.identity_key(), second.identity_key())
        self.assertNotIn(first.value, first.identity_key())
        self.assertNotIn(first.observed_at, first.identity_key())

    def test_fact_is_frozen_and_validated(self):
        fact = self.fact()
        with self.assertRaises(Exception):
            fact.value = 'Globex'
        with self.assertRaises(ValueError):
            self.fact(subject='')
        with self.assertRaises(ValueError):
            self.fact(attribute='')
        with self.assertRaises(ValueError):
            self.fact(id='')

    def test_state_decision_accepts_valid_actions(self):
        self.assertEqual(set(STATE_ACTIONS), {'CREATE', 'UPDATE', 'DUPLICATE', 'DEFER'})
        created = StateDecision(action='CREATE', state_id='state_000001', confidence=0.9,
                                reason='first observation')
        self.assertTrue(created.creates_state)
        updated = StateDecision(action='UPDATE', state_id='state_000001', confidence=1.0)
        self.assertTrue(updated.touches_existing_state)
        self.assertEqual(StateDecision(action='DEFER').action, 'DEFER')

    def test_state_decision_rejects_illegal_action(self):
        for bad in ('MERGE', 'create', '', None, 1):
            with self.assertRaises(ValueError):
                StateDecision(action=bad)  # type: ignore[arg-type]

    def test_state_decision_rejects_out_of_range_confidence(self):
        for bad in (-0.01, 1.01, 2, 'high', True):
            with self.assertRaises(ValueError):
                StateDecision(action='CREATE', state_id='s1', confidence=bad)  # type: ignore[arg-type]

    def test_state_decision_requires_state_id_for_update_and_duplicate(self):
        with self.assertRaises(ValueError):
            StateDecision(action='UPDATE', confidence=0.5)
        with self.assertRaises(ValueError):
            StateDecision(action='DUPLICATE', confidence=0.5)

    def test_state_decision_is_frozen(self):
        decision = StateDecision(action='CREATE', state_id='s1', confidence=0.5)
        with self.assertRaises(Exception):
            decision.action = 'UPDATE'


GOLD_AS_OF = '2019-03-04T05:06:07'          # 金标版本 event 时间戳（TempLAMA: query['time'] == as_of）
EPISODE_WINDOW_END = '2024-12-31T23:59:59'  # episode 观测窗口末刻（公开可得的“当前时刻”）


def templama_like_query():
    """模拟 TempLAMA 查询：time == as_of == 金标版本时间戳，且自带全部 Oracle 字段。"""
    return {'question': 'Who does Alice work for?', 'time': GOLD_AS_OF, 'as_of': GOLD_AS_OF,
            'answers': ['Acme'], 'latest': True, 'state_key': 'Q42:P54',
            'key_text': 'Alice works at', 'entity': 'Q42', 'group': 'Q42:P54',
            'episode': 'ep1', 'is_update': False, 'target_event': 'e1'}


class QueryTimeSemanticsTest(unittest.TestCase):
    """t13：Predicted 查询时间通道去 Oracle 化（time_mode 开关）。"""

    def test_window_end_mode_never_exposes_as_of_or_gold_timestamp(self):
        view = public_query(templama_like_query(), window_end=EPISODE_WINDOW_END)
        self.assertEqual(view['time'], EPISODE_WINDOW_END)
        self.assertEqual(view['time_source'], 'window_end')
        self.assertNotEqual(view['time'], GOLD_AS_OF)
        self.assertNotEqual(view['time'], templama_like_query()['as_of'])
        self.assertNotIn(GOLD_AS_OF, repr(view))
        self.assertEqual(set(view), set(PUBLIC_QUERY_KEYS))

    def test_window_end_mode_without_window_end_yields_none_and_never_falls_back(self):
        for call in (lambda: public_query(templama_like_query()),
                     lambda: public_query(templama_like_query(), time_mode='window_end'),
                     lambda: public_query(templama_like_query(), 'window_end', None)):
            view = call()
            self.assertIsNone(view['time'])
            self.assertEqual(view['time_source'], 'unknown')
            self.assertNotEqual(view['time'], GOLD_AS_OF)

    def test_window_end_mode_does_not_read_query_time_at_all(self):
        # query 完全没有 time/as_of 时 window_end 模式照样工作（证明未读 Oracle 时间通道）
        view = public_query({'question': 'Who does Alice work for?'},
                            window_end=EPISODE_WINDOW_END)
        self.assertEqual(view['time'], EPISODE_WINDOW_END)
        self.assertEqual(view['time_source'], 'window_end')
        # 反之，缺 window_end 也只降级为 unknown，不因 query 有 time 就回退
        self.assertIsNone(public_query({'question': 'Who?', 'time': GOLD_AS_OF})['time'])

    def test_as_of_mode_still_available_and_labelled(self):
        view = public_query(templama_like_query(), time_mode='as_of')
        self.assertEqual(view['time'], GOLD_AS_OF)
        self.assertEqual(view['time_source'], 'as_of')
        self.assertEqual(set(view), set(PUBLIC_QUERY_KEYS))

    def test_invalid_time_mode_and_arguments_are_rejected(self):
        for bad in ('latest', '', None, 'AS_OF'):
            with self.assertRaises(ValueError):
                public_query(templama_like_query(), time_mode=bad)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            public_query({'question': 'Who?'}, time_mode='as_of')  # 缺 query['time']
        with self.assertRaises(ValueError):
            public_query({'time': GOLD_AS_OF}, time_mode='as_of')  # 缺 question
        for bad_window in ('', '   ', 2024):
            with self.assertRaises(ValueError):
                public_query(templama_like_query(), window_end=bad_window)  # type: ignore[arg-type]

    def test_time_source_stays_in_declared_vocabulary(self):
        self.assertEqual(set(TIME_MODES), {'as_of', 'window_end'})
        self.assertEqual(set(TIME_SOURCES), {'as_of', 'window_end', 'unknown'})
        views = [public_query(templama_like_query(), time_mode='as_of'),
                 public_query(templama_like_query(), window_end=EPISODE_WINDOW_END),
                 public_query(templama_like_query())]
        for view in views:
            self.assertIn(view['time_source'], TIME_SOURCES)
            self.assertIs(assert_no_oracle(view, 'query', allowed=PUBLIC_QUERY_KEYS), view)


class ModuleBoundaryHygieneTest(unittest.TestCase):
    """新增模块必须能在 python 3.9 下解析，且首行为 future import。"""

    def test_new_modules_start_with_future_annotations(self):
        for relative in ('vcrm/fact_schema.py', 'vcrm/data_views.py',
                         'tests/test_oracle_isolation.py'):
            first_line = (REPO_ROOT / relative).read_text().splitlines()[0]
            self.assertEqual(first_line, 'from __future__ import annotations', relative)

    def test_fact_schema_mentions_no_oracle_routing_fields(self):
        source = (REPO_ROOT / 'vcrm/fact_schema.py').read_text()
        code_lines = [line for line in source.splitlines()
                      if line.strip() and not line.strip().startswith('#')]
        for forbidden in ('state_key', 'key_text', 'answers', 'episode'):
            self.assertNotIn(forbidden, '\n'.join(code_lines))


if __name__ == '__main__':
    unittest.main()
