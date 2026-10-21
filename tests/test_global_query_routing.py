from __future__ import annotations
# P3 端到端回归测试：stub reader + 真实 codec（CPU，未训练同构配置）+ 小规模合成数据。
#
# 覆盖：
#   1) 无候选 → answer='' 且 no_state_match=True
#   2) 混合事实池（同主体不同属性 / 不同主体同属性干扰项）定位到正确状态
#   3) target_time 选到正确版本；window_end=None → 最新可见版本（不补时间字段）
#   4) 决策日志字段齐全且不含 state_key
#   5) recency_update off（保守基线，全部 DEFER）vs on（rule5b 建版本链）
#   6) window_end 口径绝不回退 query['time'] / query['as_of']

import json
import unittest

import torch

from vcrm.codec import SlotCodec, VersionMarkers
from vcrm.data_views import PublicEventCounter, public_write_event
from vcrm.fact_extractor import FactExtractor
from vcrm.index import SemanticIndex
from vcrm.runtime import DECISION_LOG_KEYS, PredictedRuntime, QueryParser, validate_decision_log
from vcrm.state_index import SEMANTIC_DIMENSIONS, IndexCommitError, StateIndex
from vcrm.state_router import StateRouter
from vcrm.store import IncrementalCurrentStore
from vcrm.update_resolver import UpdateResolver

HIDDEN = SEMANTIC_DIMENSIONS
MEMORY_SLOTS = 8
CODEC_SLOTS = 4

GOV_FR_2016 = ('2016-01-01T00:00:00', 'In 2016: The head of government of France is Bernard Cazeneuve')
GOV_FR_2018 = ('2018-01-01T00:00:00', 'In 2018: The head of government of France is Edouard Philippe')
GOV_FR_2020 = ('2020-01-01T00:00:00', 'In 2020: The head of government of France is Jean Castex')
CAPITAL_FR = ('2016-01-01T00:00:00', 'In 2016: The capital city of France is Paris')
GOV_DE = ('2016-01-01T00:00:00', 'In 2016: The head of government of Germany is Angela Merkel')

QUESTION = 'The head of government of France is _X_'


class StubReader:
    """词袋 latent + 固定答案的假 reader（不加载 ICAE，不产生语言结论）。"""

    def __init__(self):
        self.cfg = {'read_latent_slots': None, 'state_similarity': 0.8}
        self.device = torch.device('cpu')

    def event_text(self, event):
        return f"DATE: {event['time']}\n{event['text']}"

    def encode(self, text, training=False):
        import hashlib
        import re
        vector = torch.zeros(HIDDEN)
        for token in re.findall(r'[a-z]+', str(text).lower()):
            if token in {'a', 'an', 'the', 'is', 'of', 'in', 'on', 'at', 'to', 'for'}:
                continue
            value = int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), 'little')
            vector[value % HIDDEN] += 1.0 if (value // HIDDEN) % 2 else -1.0
        norm = float(vector.norm())
        return (vector / norm if norm else vector).unsqueeze(0).repeat(MEMORY_SLOTS, 1)

    def encode_event(self, event, training=False):
        return self.encode(self.event_text(event), training)

    def question(self, query):
        date = query.get('time')
        return (f'Current date: {date}\n' if date else '') + str(query['question'])

    def answer(self, query, bundle=None, text=''):
        if bundle is None:
            return {'answer': '', 'read_latent_slots': 0}
        return {'answer': 'stub', 'read_latent_slots': int(bundle.shape[0])}


class DeterministicIndex(SemanticIndex):
    """key(memory) = L2 归一化的 slot 均值（无随机投影，排序可解释；仅测试用）。"""

    def __init__(self):
        super().__init__(HIDDEN, SEMANTIC_DIMENSIONS)

    def key(self, memory):
        vector = memory.float().mean(0)
        norm = vector.norm().clamp_min(1e-6)
        return (vector / norm).numpy().astype('<f2')


def make_codec():
    torch.manual_seed(42)
    return SlotCodec(HIDDEN, CODEC_SLOTS, 8, 2, False, 0., 42, 'residual', None,
                     'projected', None).eval().requires_grad_(False)


def event(public_id, pair):
    return public_write_event({'id': f'src_{public_id}', 'time': pair[0], 'text': pair[1]},
                              public_id)


def make_runtime(*, recency_update=False, recency_min_confidence=0.0,
                 time_mode='window_end', defer_store_memory=False, hybrid_fallback=False):
    return PredictedRuntime(StubReader(), make_codec(), VersionMarkers(HIDDEN),
                            DeterministicIndex(),
                            IncrementalCurrentStore(make_codec(), key_dimensions=SEMANTIC_DIMENSIONS),
                            StateIndex(),
                            router=StateRouter(top_k=5),
                            resolver=UpdateResolver(recency_update=recency_update,
                                                    recency_min_confidence=recency_min_confidence),
                            extractor=FactExtractor(), query_parser=QueryParser(),
                            time_mode=time_mode, allow_as_of=(time_mode == 'as_of'),
                            defer_store_memory=defer_store_memory,
                            hybrid_fallback=hybrid_fallback)


def query(question=QUESTION, time='2020-01-01T00:00:00', answers=('Jean Castex',),
          state_key='France:head of government', kind='latest'):
    return {'id': 'q1', 'question': question, 'time': time, 'as_of': time,
            'answers': list(answers), 'latest': kind == 'latest', 'type': kind,
            'state_key': state_key, 'key_text': question}


class GlobalQueryRoutingTest(unittest.TestCase):
    def setUp(self):
        self.counter = PublicEventCounter()

    def write(self, runtime, pairs):
        receipts = []
        for pair in pairs:
            receipts.extend(runtime.write_predicted(event(self.counter.next_id(), pair))['receipts'])
        return receipts

    def test_no_candidate_returns_no_state_match(self):
        runtime = make_runtime()
        result = runtime.answer_predicted(query(question='The spouse of Nobody is _X_'),
                                          window_end='2020-01-01T00:00:00')
        self.assertTrue(result['no_state_match'])
        self.assertEqual(result['answer'], '')
        self.assertEqual(result['candidates'], [])
        self.assertFalse(result['hybrid_fallback'])

    def test_mixed_pool_locates_the_correct_state(self):
        runtime = make_runtime()
        self.write(runtime, [GOV_FR_2016, CAPITAL_FR, GOV_DE])
        france_gov = {record.state_id for record in runtime.state_index.records
                      if (record.subject_key, record.attribute_key)
                      == ('france', 'head_of_government')}
        result = runtime.answer_predicted(query(), window_end='2020-01-01T00:00:00')
        self.assertFalse(result['no_state_match'])
        identities = {(candidate['subject_key'], candidate['attribute_key'])
                      for candidate in result['candidates']}
        # 干扰项必须同时在场：同主体不同属性 + 不同主体同属性
        self.assertIn(('france', 'capital'), identities)
        self.assertIn(('germany', 'head_of_government'), identities)
        self.assertEqual(result['candidates'][0]['subject_key'], 'france')
        self.assertEqual(result['candidates'][0]['attribute_key'], 'head_of_government')
        self.assertIn(result['state_id'], france_gov)
        self.assertEqual(result['selected_version_time'], '2016-01-01T00:00:00')

    def test_target_time_selects_the_correct_version(self):
        runtime = make_runtime(recency_update=True)
        self.write(runtime, [GOV_FR_2016, GOV_FR_2018, GOV_FR_2020])
        historical = runtime.answer_predicted(query(kind='historical'), window_end='2018-12-31T00:00:00')
        self.assertEqual(historical['selected_version_time'], '2018-01-01T00:00:00')
        self.assertEqual(historical['version_role'], 'historical')
        latest = runtime.answer_predicted(query(), window_end='2021-01-01T00:00:00')
        self.assertEqual(latest['selected_version_time'], '2020-01-01T00:00:00')
        self.assertEqual(latest['version_role'], 'current')

    def test_unknown_time_means_latest_visible_version(self):
        runtime = make_runtime(recency_update=True)
        self.write(runtime, [GOV_FR_2016, GOV_FR_2018, GOV_FR_2020])
        result = runtime.answer_predicted(query(), window_end=None)
        self.assertIsNone(result['time'])
        self.assertEqual(result['time_source'], 'unknown')
        self.assertEqual(result['selected_version_time'], '2020-01-01T00:00:00')

    def test_window_end_never_falls_back_to_query_time(self):
        runtime = make_runtime(recency_update=True)
        self.write(runtime, [GOV_FR_2016, GOV_FR_2018, GOV_FR_2020])
        # query['time'] == query['as_of'] == 金标版本时间（2016），window_end 是公开窗口末刻
        result = runtime.answer_predicted(query(time='2016-01-01T00:00:00'),
                                          window_end='2020-06-01T00:00:00')
        self.assertEqual(result['time_source'], 'window_end')
        self.assertEqual(result['target_time'], '2020-06-01T00:00:00')
        self.assertEqual(result['selected_version_time'], '2020-01-01T00:00:00')

    def test_decision_log_contract_and_no_oracle_leak(self):
        runtime = make_runtime(recency_update=True)
        self.write(runtime, [GOV_FR_2016, GOV_FR_2018])
        self.assertTrue(runtime.decisions)
        for record in runtime.decisions:
            self.assertEqual(set(record), set(DECISION_LOG_KEYS))
            validate_decision_log(record)
            self.assertNotIn('state_key', json.dumps(record))
        self.assertEqual([record['decision'] for record in runtime.decisions],
                         ['CREATE', 'UPDATE'])
        self.assertTrue(runtime.decisions[1]['compression_triggered'])
        self.assertNotEqual(runtime.decisions[1]['old_version_id'],
                            runtime.decisions[1]['new_version_id'])

    def test_recency_update_off_defers_and_on_builds_versions(self):
        conservative = make_runtime(recency_update=False)
        self.write(conservative, [GOV_FR_2016, GOV_FR_2018, GOV_FR_2020])
        self.assertEqual([record['decision'] for record in conservative.decisions],
                         ['CREATE', 'DEFER', 'DEFER'])
        self.assertEqual(conservative.state_index.pending_count, 2)

        recency = make_runtime(recency_update=True)
        self.write(recency, [GOV_FR_2016, GOV_FR_2018, GOV_FR_2020])
        self.assertEqual([record['decision'] for record in recency.decisions],
                         ['CREATE', 'UPDATE', 'UPDATE'])
        reasons = [record['reason'] for record in recency.decisions if record['decision'] == 'UPDATE']
        for reason in reasons:
            self.assertTrue(reason.startswith('exclusive_superseded_by_recency:'), reason)

    def test_recency_min_confidence_blocks_uncalibrated_updates(self):
        runtime = make_runtime(recency_update=True, recency_min_confidence=0.99)
        self.write(runtime, [GOV_FR_2016, GOV_FR_2018])
        self.assertEqual([record['decision'] for record in runtime.decisions],
                         ['CREATE', 'DEFER'])

    # -- t20 / P3-F1：CREATE 时索引失败必须回滚 store（不留悬空状态 + 1MB payload） ----
    def test_create_index_failure_rolls_back_store(self):
        runtime = make_runtime()
        self.write(runtime, [GOV_FR_2016])
        before = _fingerprint(runtime)
        original_add = runtime.state_index.add

        def boom(*args, **kwargs):
            raise RuntimeError('index insert exploded')

        runtime.state_index.add = boom
        try:
            with self.assertRaises(IndexCommitError) as ctx:
                runtime.write_predicted(event(self.counter.next_id(), CAPITAL_FR))
        finally:
            runtime.state_index.add = original_add

        message = str(ctx.exception)
        self.assertIn('index insert failed after store create', message)
        self.assertIn('index insert exploded', message)
        # store 逐字节不变、索引大小不变：没有悬空状态，也没有残留 payload
        self.assertEqual(_fingerprint(runtime), before)
        self.assertEqual(len(runtime.state_index), 1)
        self.assertEqual(runtime.state_index.pending_count, 0)
        self.assertEqual(runtime.rollback_errors, [])

    # -- t20 / P3-F5：DEFER + defer_store_memory 时 store 失败要撤销索引侧 pending ----
    def test_defer_store_failure_undoes_index_pending(self):
        runtime = make_runtime(recency_update=False, defer_store_memory=True)
        self.write(runtime, [GOV_FR_2016])
        before = _fingerprint(runtime)
        original = runtime.store.ingest_decision

        def boom(*args, **kwargs):
            raise RuntimeError('store pending exploded')

        runtime.store.ingest_decision = boom
        try:
            with self.assertRaises(IndexCommitError) as ctx:
                runtime.write_predicted(event(self.counter.next_id(), GOV_FR_2018))
        finally:
            runtime.store.ingest_decision = original
        self.assertIn('store pending write failed', str(ctx.exception))
        self.assertEqual(_fingerprint(runtime), before)     # 无"索引有 / store 无"单边状态
        self.assertEqual(runtime.state_index.pending_count, 0)


def _fingerprint(runtime):
    """store + index 的可比较指纹（serialize 逐字节 + entries/blocks 摘要）。"""
    store = runtime.store
    return {
        'store_bytes': json.dumps(store.serialize(), sort_keys=True, default=str),
        'entries': [(entry.get('id'), entry.get('state'), entry.get('kind'))
                    for entry in getattr(store, 'entries', [])],
        'blocks': json.dumps(getattr(store, 'blocks', []), sort_keys=True, default=str),
        'index': json.dumps(runtime.state_index.serialize(), sort_keys=True, default=str),
        'index_size': len(runtime.state_index),
    }


if __name__ == '__main__':
    unittest.main()
