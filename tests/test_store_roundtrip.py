from __future__ import annotations
# P1 持久化/回滚/预算回归：IncrementalCurrentStore 的 save/load 往返一致、
# UPDATE 中途失败必须回滚、超预算显式拒绝（不静默淘汰）、乱序拒绝并记录、
# DUPLICATE 不改 payload、DEFER 不降级旧版、已压缩 code 不二次压缩。
# codec 用同构 E3 结构的小尺寸 SlotCodec（随机初始化、独立/conditioned=False），
# 因为本文件只验证存储协议，不验证权重质量。
import pathlib
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from vcrm.codec import SlotCodec
from vcrm.fact_schema import AtomicFact, StateDecision
from vcrm.index import key_vector
from vcrm.store import IncrementalCurrentStore

try:  # t5 的 StateIndex 可能尚未同步到本 checkout；缺失时只跳过集成用例
    from vcrm.state_index import StateIndex, StoreCommitError
except ImportError:  # pragma: no cover - 取决于 checkout 内容
    StateIndex = StoreCommitError = None

STATE = 'Alice:employer'
DIMENSION, MEMORY_SLOTS, KEY_DIMENSIONS = 32, 8, 8
CODEC_SLOTS, CODEC_WIDTH, CODEC_HEADS = 4, 16, 4


def codec_pair(conditioned=False):
    torch.manual_seed(7)
    return SlotCodec(DIMENSION, CODEC_SLOTS, CODEC_WIDTH, CODEC_HEADS, conditioned, 0., 7,
                     'residual', None, 'full', 8.0).eval().requires_grad_(False)


def memory(seed, slots=MEMORY_SLOTS):
    generator = torch.Generator().manual_seed(seed)
    return (torch.randn(slots, DIMENSION, generator=generator) * .5).half().float()


def key(text='Alice works at Google'):
    return key_vector(text, KEY_DIMENSIONS)


def decision(action, state=STATE, confidence=.9, reason='test'):
    return StateDecision(action, state, confidence, reason)


def event(version_id, time):
    return {'id': version_id, 'time': time}


class RoundtripTest(unittest.TestCase):
    def setUp(self):
        self.codec = codec_pair()
        self.store = IncrementalCurrentStore(self.codec, key_dimensions=KEY_DIMENSIONS)
        self.tmp = pathlib.Path(tempfile.mkdtemp())

    def populated(self):
        self.store.ingest_decision(event('v1', '2018-01-01T00:00:00'), memory(1), key(),
                                   decision('CREATE'))
        self.store.ingest_decision(event('v2', '2019-01-01T00:00:00'), memory(2), key(),
                                   decision('UPDATE'))
        self.store.ingest_decision(event('d1', '2019-06-01T00:00:00'), memory(3), key(),
                                   decision('DUPLICATE'))
        self.store.ingest_decision(event('p1', '2019-09-01T00:00:00'), memory(4), key(),
                                   decision('DEFER', confidence=.4, reason='ambiguous'))
        return self.store

    def test_save_load_roundtrip_preserves_identity_order_and_payload(self):
        store = self.populated()
        path = self.tmp / 'incremental.mem'
        written = store.save(path)
        reloaded = IncrementalCurrentStore.load(path, self.codec, key_dimensions=KEY_DIMENSIONS)
        self.assertEqual(written, path.stat().st_size)
        self.assertEqual(reloaded.serialize(), store.serialize())
        self.assertEqual(reloaded.serialize(), path.read_bytes())
        # 状态身份 / 版本顺序 / 角色
        self.assertEqual({e['state'] for e in reloaded.entries}, {STATE})
        self.assertEqual([e['id'] for e in reloaded.entries], ['v1', 'v2'])
        self.assertEqual([e['kind'] for e in reloaded.entries], ['code', 'full'])
        self.assertEqual(reloaded.current(STATE)['id'], 'v2')
        self.assertEqual(reloaded.blocks, store.blocks)
        self.assertEqual(reloaded.counts, store.counts)
        # latent payload 逐比特一致
        for left, right in zip(store.entries, reloaded.entries):
            for field in ('id', 'state', 'time', 'kind', 'keyframe', 'anchor_id', 'compression_count'):
                self.assertEqual(left[field], right[field])
            self.assertTrue(np.array_equal(left['payload'], right['payload']))
            self.assertTrue(np.array_equal(left['key'], right['key']))
            self.assertTrue(torch.equal(store.decode(left), reloaded.decode(right)))
        # 待确认 / 重复 / 乱序记录也往返一致
        self.assertEqual([r['id'] for r in reloaded.pending], ['p1'])
        self.assertTrue(np.array_equal(reloaded.pending[0]['payload'], store.pending[0]['payload']))
        self.assertEqual(reloaded.pending[0]['confidence'], .4)
        self.assertEqual(reloaded.duplicates, [{'id': 'd1', 'state': STATE,
                                                'time': '2019-06-01T00:00:00', 'matched_id': 'v2'}])
        self.assertEqual(reloaded.rejections, store.rejections)
        self.assertEqual(reloaded.accounting(), store.accounting())
        # bundle 一致
        self.assertTrue(torch.equal(reloaded.bundle([reloaded.select_version(STATE)]),
                                    store.bundle([store.select_version(STATE)])))
        reloaded.validate_protocol()

    def test_load_rejects_wrong_codec_identity(self):
        store = self.populated()
        path = self.tmp / 'incremental.mem'
        store.save(path)
        with self.assertRaises(ValueError):
            IncrementalCurrentStore.load(path, codec_pair(conditioned=True),
                                         key_dimensions=KEY_DIMENSIONS)
        other = codec_pair()
        with torch.no_grad():
            other.queries.add_(1.)
        with self.assertRaises(ValueError):
            IncrementalCurrentStore.load(path, other, key_dimensions=KEY_DIMENSIONS)

    def test_keyframe_ingest_is_disabled(self):
        with self.assertRaises(RuntimeError):
            self.store.ingest(event('v1', '2018-01-01T00:00:00'), memory(1), key())


class RollbackTest(unittest.TestCase):
    def setUp(self):
        self.codec = codec_pair()
        self.store = IncrementalCurrentStore(self.codec, key_dimensions=KEY_DIMENSIONS)
        self.store.ingest_decision(event('v1', '2018-01-01T00:00:00'), memory(1), key(),
                                   decision('CREATE'))

    def snapshot(self):
        return (self.store.serialize(),
                [(e['id'], e['kind'], e['compression_count']) for e in self.store.entries])

    def test_update_codec_failure_rolls_back(self):
        before = self.snapshot()
        with mock.patch.object(self.codec, 'encode', side_effect=RuntimeError('codec exploded')):
            with self.assertRaises(RuntimeError):
                self.store.ingest_decision(event('v2', '2019-01-01T00:00:00'), memory(2), key(),
                                           decision('UPDATE'))
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(self.store.current(STATE)['kind'], 'full')
        self.assertEqual(len(self.store.entries), 1)
        self.store.validate_protocol()

    def test_update_wrong_code_shape_rolls_back(self):
        before = self.snapshot()
        with mock.patch.object(self.codec, 'encode', return_value=torch.zeros(2, DIMENSION)):
            with self.assertRaises(ValueError):
                self.store.ingest_decision(event('v2', '2019-01-01T00:00:00'), memory(2), key(),
                                           decision('UPDATE'))
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(self.store.current(STATE)['id'], 'v1')
        self.assertEqual(self.store.current(STATE)['compression_count'], 0)

    def test_nonfinite_code_rolls_back(self):
        before = self.snapshot()
        bad = torch.full((CODEC_SLOTS, DIMENSION), float('inf'))
        with mock.patch.object(self.codec, 'encode', return_value=bad):
            with self.assertRaises(ValueError):
                self.store.ingest_decision(event('v2', '2019-01-01T00:00:00'), memory(2), key(),
                                           decision('UPDATE'))
        self.assertEqual(self.snapshot(), before)


class BudgetTest(unittest.TestCase):
    def test_byte_budget_rejects_instead_of_evicting(self):
        codec = codec_pair()
        unlimited = IncrementalCurrentStore(codec, key_dimensions=KEY_DIMENSIONS)
        unlimited.ingest_decision(event('v1', '2018-01-01T00:00:00'), memory(1), key(), decision('CREATE'))
        one_version = len(unlimited.serialize())
        unlimited.ingest_decision(event('v2', '2019-01-01T00:00:00'), memory(2), key(), decision('UPDATE'))
        two_versions = len(unlimited.serialize())
        self.assertLess(one_version, two_versions)

        store = IncrementalCurrentStore(codec, key_dimensions=KEY_DIMENSIONS,
                                        byte_budget=two_versions - 1)
        store.ingest_decision(event('v1', '2018-01-01T00:00:00'), memory(1), key(), decision('CREATE'))
        before = store.serialize()
        with self.assertRaises(ValueError) as caught:
            store.ingest_decision(event('v2', '2019-01-01T00:00:00'), memory(2), key(), decision('UPDATE'))
        self.assertIn('byte budget exceeded', str(caught.exception))
        self.assertEqual(store.serialize(), before)
        self.assertEqual(len(store.entries), 1)
        self.assertEqual(store.current(STATE)['kind'], 'full')
        self.assertEqual(store.accounting()['rejection_records'], 0)

    def test_retain_all_mode_reports_bytes_and_slots(self):
        codec = codec_pair()
        store = IncrementalCurrentStore(codec, key_dimensions=KEY_DIMENSIONS)
        store.ingest_decision(event('v1', '2018-01-01T00:00:00'), memory(1), key(), decision('CREATE'))
        store.ingest_decision(event('v2', '2019-01-01T00:00:00'), memory(2), key(), decision('UPDATE'))
        accounting = store.accounting()
        self.assertEqual(accounting['budget_mode'], 'retain_all')
        self.assertFalse(accounting['resident_slot_budget_applied'])
        self.assertEqual(accounting['current_versions'], 1)
        self.assertEqual(accounting['compressed_versions'], 1)
        self.assertEqual(accounting['resident_latent_slots'], MEMORY_SLOTS + CODEC_SLOTS)
        self.assertEqual(accounting['memory_bytes'], MEMORY_SLOTS * DIMENSION * 2 + CODEC_SLOTS * DIMENSION * 2)
        self.assertEqual(accounting['total_bytes'], len(store.serialize()))
        with self.assertRaises(ValueError):
            IncrementalCurrentStore(codec, key_dimensions=KEY_DIMENSIONS, budget_mode='fifo')


class DecisionSemanticsTest(unittest.TestCase):
    def setUp(self):
        self.codec = codec_pair()
        self.store = IncrementalCurrentStore(self.codec, key_dimensions=KEY_DIMENSIONS)
        self.store.ingest_decision(event('v1', '2018-01-01T00:00:00'), memory(1), key(),
                                   decision('CREATE'))

    def test_conditioned_codec_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            IncrementalCurrentStore(codec_pair(conditioned=True), key_dimensions=KEY_DIMENSIONS)
        self.assertEqual(str(caught.exception), 'Incremental E3 store requires independent codec')

    def test_out_of_order_is_rejected_and_recorded(self):
        before = [(e['id'], e['kind'], e['payload'].tobytes()) for e in self.store.entries]
        with self.assertRaises(ValueError):
            self.store.ingest_decision(event('v0', '2017-01-01T00:00:00'), memory(9), key(),
                                       decision('UPDATE'))
        # 版本条目逐比特未变；只有一条显式的乱序拒绝记录被追加。
        self.assertEqual([(e['id'], e['kind'], e['payload'].tobytes()) for e in self.store.entries], before)
        self.assertEqual(self.store.rejections,
                         [{'id': 'v0', 'state': STATE, 'time': '2017-01-01T00:00:00',
                           'reason': 'out_of_order'}])

    def test_duplicate_keeps_payload_and_defer_keeps_current(self):
        payload_before = self.store.current(STATE)['payload'].copy()
        self.store.ingest_decision(event('d1', '2018-06-01T00:00:00'), memory(3), key(),
                                   decision('DUPLICATE'))
        self.assertTrue(np.array_equal(self.store.current(STATE)['payload'], payload_before))
        self.assertEqual(len(self.store.entries), 1)
        self.assertEqual([r['id'] for r in self.store.duplicates], ['d1'])

        self.store.ingest_decision(event('p1', '2018-07-01T00:00:00'), memory(4), key(),
                                   decision('DEFER', confidence=.3, reason='conflict'))
        current = self.store.current(STATE)
        self.assertEqual((current['id'], current['kind'], current['compression_count']), ('v1', 'full', 0))
        self.assertEqual([r['id'] for r in self.store.pending], ['p1'])
        self.assertEqual(self.store.pending[0]['payload'].shape, (MEMORY_SLOTS, DIMENSION))
        self.assertTrue(np.array_equal(self.store.current(STATE)['payload'], payload_before))

    def test_update_compresses_once_and_never_recompresses(self):
        self.store.ingest_decision(event('v2', '2019-01-01T00:00:00'), memory(2), key(),
                                   decision('UPDATE'))
        v1_code = self.store.entries[0]['payload'].copy()
        self.store.ingest_decision(event('v3', '2020-01-01T00:00:00'), memory(5), key(),
                                   decision('UPDATE'))
        codes = [e for e in self.store.entries if e['kind'] == 'code']
        self.assertEqual([e['id'] for e in codes], ['v1', 'v2'])
        for entry in codes:
            self.assertEqual(entry['compression_count'], 1)
            self.assertFalse(entry['keyframe'])
            self.assertIsNone(entry['anchor_id'])
            self.assertEqual(entry['payload'].shape, (CODEC_SLOTS, DIMENSION))
        # 旧 code 在后续 UPDATE 中保持逐比特不变（不二次压缩、不重基）
        self.assertTrue(np.array_equal(self.store.entries[0]['payload'], v1_code))
        self.assertEqual(self.store.current(STATE)['id'], 'v3')
        self.assertEqual(len([e for e in self.store.entries if e['kind'] == 'full']), 1)

    def test_create_requires_absent_state_and_update_requires_current(self):
        with self.assertRaises(ValueError):
            self.store.ingest_decision(event('v2', '2019-01-01T00:00:00'), memory(2), key(),
                                       decision('CREATE'))
        with self.assertRaises(ValueError):
            self.store.ingest_decision(event('x1', '2019-01-01T00:00:00'), memory(2), key(),
                                       decision('UPDATE', state='Alice:city'))
        with self.assertRaises(ValueError):
            self.store.ingest_decision(event('v1', '2019-01-01T00:00:00'), memory(2), key(),
                                       decision('UPDATE'))

    def test_code_slot_payload_cannot_be_written_as_full_memory(self):
        self.store.ingest_decision(event('v2', '2019-01-01T00:00:00'), memory(2), key(),
                                   decision('UPDATE'))
        code = next(e['payload'] for e in self.store.entries if e['kind'] == 'code')
        self.assertEqual(code.shape, (CODEC_SLOTS, DIMENSION))
        with self.assertRaises(ValueError):
            self.store.ingest_decision(event('v3', '2020-01-01T00:00:00'),
                                       torch.from_numpy(code.copy()), key(), decision('UPDATE'))


if __name__ == '__main__':
    unittest.main(verbosity=2)


class _Record:
    """最小 StateRecord 替身：只暴露 commit() 适配层读取的字段。"""

    def __init__(self, state_id=STATE, current_version_id=None, semantic_key=None):
        self.state_id = state_id
        self.current_version_id = current_version_id
        self.semantic_key = semantic_key


class CommitAdapterTest(unittest.TestCase):
    """t5 契约：store.commit(record=..., fact=...) 必须返回非空版本号，并支持 rollback。"""

    def setUp(self):
        self.codec = codec_pair()
        self.store = IncrementalCurrentStore(self.codec, key_dimensions=KEY_DIMENSIONS)

    def fact(self, fact_id='fact_000001', value='Google', observed_at='2018-01-01T00:00:00'):
        return AtomicFact(id=fact_id, text=f'Alice works at {value}', observed_at=observed_at,
                          subject='Alice', attribute='employer', value=value, source_id='input_000001')

    def test_create_then_update_through_commit(self):
        record = _Record(current_version_id=None, semantic_key=key().tolist())
        created = self.store.commit(record=record, fact=self.fact(), memory=memory(1))
        self.assertEqual(created['version_id'], 'fact_000001')
        self.assertEqual(created['action'], 'CREATE')
        record.current_version_id = created['version_id']
        updated = self.store.commit(record=record, fact=self.fact('fact_000002', 'Microsoft',
                                                                  '2019-01-01T00:00:00'),
                                    memory=memory(2))
        self.assertEqual(updated['version_id'], 'fact_000002')
        self.assertEqual(updated['action'], 'UPDATE')
        self.assertEqual(updated['downgraded'], 'fact_000001')
        self.assertEqual([(e['id'], e['kind']) for e in self.store.entries],
                         [('fact_000001', 'code'), ('fact_000002', 'full')])

    def test_commit_maps_record_and_fact_fields(self):
        record = _Record(current_version_id=None)
        receipt = self.store.commit(record=record, fact=self.fact(), memory=memory(1),
                                    semantic_key=key())
        entry = self.store.entries[0]
        self.assertEqual(entry['state'], record.state_id)       # record.state_id → decision.state_id
        self.assertEqual(entry['id'], 'fact_000001')            # fact.id → 版本号
        self.assertEqual(entry['time'], '2018-01-01T00:00:00')  # fact.observed_at → event['time']
        self.assertNotIn('state_key', entry)                    # event 里没有任何 oracle 字段
        self.assertEqual(receipt['state'], record.state_id)

    def test_commit_uses_explicit_decision(self):
        record = _Record(current_version_id='fact_000001')
        receipt = self.store.commit(record=record, fact=self.fact('fact_000009', 'OpenAI',
                                                                 '2021-01-01T00:00:00'),
                                    memory=memory(3), semantic_key=key(),
                                    decision=StateDecision('DEFER', STATE, .2, 'ambiguous'))
        self.assertEqual(receipt['action'], 'DEFER')
        self.assertEqual([r['id'] for r in self.store.pending], ['fact_000009'])
        self.assertEqual(self.store.entries, [])

    def test_rollback_undoes_commit(self):
        record = _Record(current_version_id=None)
        before = self.store.serialize()
        result = self.store.commit(record=record, fact=self.fact(), memory=memory(1),
                                   semantic_key=key())
        self.assertNotEqual(self.store.serialize(), before)
        self.assertTrue(self.store.rollback(result))
        self.assertEqual(self.store.serialize(), before)
        self.assertEqual(self.store.entries, [])
        self.assertFalse(self.store.rollback(result))

    def test_commit_rejects_incomplete_inputs(self):
        record = _Record(current_version_id=None)
        with self.assertRaises(ValueError):
            self.store.commit(fact=self.fact(), memory=memory(1), semantic_key=key())
        with self.assertRaises(ValueError):
            self.store.commit(record=record, memory=memory(1), semantic_key=key())
        with self.assertRaises(ValueError):
            self.store.commit(record=record, fact=self.fact(), semantic_key=key())
        with self.assertRaises(ValueError):
            self.store.commit(record=record, fact=self.fact(), memory=memory(1))
        with self.assertRaises(ValueError):
            self.store.commit(record=record, fact=self.fact(), memory=memory(1), semantic_key=key(),
                              decision=StateDecision('UPDATE', 'other_state', .9, 'mismatch'))

    def test_state_index_commit_with_end_to_end(self):
        if StateIndex is None:
            self.skipTest('vcrm.state_index (t5) not available in this checkout')
        index = StateIndex()
        record = index.add_keys('Alice', 'employer', 'Google', version_id='fact_000001')
        first = self.store.commit(record=record, fact=self.fact(), memory=memory(1), semantic_key=key(),
                                  decision=StateDecision('CREATE', record.state_id, 1., 'first observation'))
        self.assertEqual(first['version_id'], record.current_version_id)
        self.assertEqual(first['action'], 'CREATE')
        updated = self.fact('fact_000002', 'Microsoft', '2019-01-01T00:00:00')
        version = index.commit_with(
            self.store, record.state_id, updated,
            apply=lambda store, rec, f: store.commit(record=rec, fact=f, memory=memory(2),
                                                     semantic_key=key()),
            value_fingerprint='fp_microsoft', observed_at=updated.observed_at)
        self.assertEqual(version, 'fact_000002')
        self.assertEqual(record.current_version_id, 'fact_000002')
        self.assertEqual(record.current_observed_at, '2019-01-01T00:00:00')
        self.assertEqual(self.store.current(record.state_id)['id'], 'fact_000002')
        # 没传 memory 时是干净失败：索引指针保持原样（commit_with → StoreCommitError）
        before = record.current_version_id
        with self.assertRaises(StoreCommitError):
            index.commit_with(self.store, record.state_id,
                              self.fact('fact_000003', 'OpenAI', '2020-01-01T00:00:00'))
        self.assertEqual(record.current_version_id, before)


class BudgetGuardCoversEveryActionTest(unittest.TestCase):
    """F1/F3：DEFER/DUPLICATE 与 CREATE/UPDATE 共用同一条事务（validate + 预算守卫 + 回滚）。

    修复前 DEFER/DUPLICATE 在守卫之前就 append 并 return，store 会进入
    "已接受但 save() 必抛 AssertionError" 的坏状态。
    """

    def setUp(self):
        self.codec = codec_pair()
        self.tmp = pathlib.Path(tempfile.mkdtemp())

    def measured(self, action, fact_id):
        unlimited = IncrementalCurrentStore(self.codec, key_dimensions=KEY_DIMENSIONS)
        unlimited.ingest_decision(event('v1', '2018-01-01T00:00:00'), memory(1), key(),
                                  decision('CREATE'))
        base = len(unlimited.serialize())
        unlimited.ingest_decision(event(fact_id, '2019-01-01T00:00:00'), memory(2), key(),
                                  decision(action, confidence=.4, reason='probe'))
        return base, len(unlimited.serialize())

    def budgeted(self, budget):
        store = IncrementalCurrentStore(self.codec, key_dimensions=KEY_DIMENSIONS, byte_budget=budget)
        store.ingest_decision(event('v1', '2018-01-01T00:00:00'), memory(1), key(), decision('CREATE'))
        return store

    def assert_rejected_without_change(self, action, fact_id, slack):
        base, grown = self.measured(action, fact_id)
        self.assertGreater(grown, base + slack)
        store = self.budgeted(base + slack)
        before = store.serialize()
        with self.assertRaises(ValueError) as caught:
            store.ingest_decision(event(fact_id, '2019-01-01T00:00:00'), memory(2), key(),
                                  decision(action, confidence=.4, reason='probe'))
        self.assertIn('byte budget exceeded', str(caught.exception))
        # 状态逐比特不变（serialize 前后哈希相同），且随后仍可持久化
        self.assertEqual(store.serialize(), before)
        self.assertEqual(store.pending, [])
        self.assertEqual(store.duplicates, [])
        path = self.tmp / f'{action.lower()}.mem'
        written = store.save(path)
        self.assertEqual(written, len(before))
        reloaded = IncrementalCurrentStore.load(path, self.codec, key_dimensions=KEY_DIMENSIONS,
                                                byte_budget=base + slack)
        self.assertEqual(reloaded.serialize(), before)
        return base, grown, base + slack

    def test_defer_over_budget_is_rejected_and_rolled_back(self):
        base, grown, budget = self.assert_rejected_without_change('DEFER', 'p1', 50)
        print(f'\nF1 DEFER base={base} grown={grown} budget={budget} -> rejected, hash unchanged, save ok')

    def test_duplicate_over_budget_is_rejected_and_rolled_back(self):
        base, grown, budget = self.assert_rejected_without_change('DUPLICATE', 'd1', 20)
        print(f'\nF3 DUPLICATE base={base} grown={grown} budget={budget} -> rejected, hash unchanged, save ok')

    def test_defer_within_budget_still_works(self):
        base, grown = self.measured('DEFER', 'p1')
        store = self.budgeted(grown)  # 恰好等于写入后的字节数：允许，不淘汰
        receipt = store.ingest_decision(event('p1', '2019-01-01T00:00:00'), memory(2), key(),
                                        decision('DEFER', confidence=.4, reason='probe'))
        self.assertEqual(receipt['action'], 'DEFER')
        self.assertEqual([r['id'] for r in store.pending], ['p1'])
        self.assertEqual(store.current(STATE)['id'], 'v1')
        self.assertEqual(store.save(self.tmp / 'defer.mem'), len(store.serialize()))


class RejectionDoesNotReserveIdTest(unittest.TestCase):
    """F2：乱序拒绝只是诊断痕迹，不得永久占用 event id（否则同一条事件按正确顺序重投会丢数据）。"""

    def setUp(self):
        self.codec = codec_pair()
        self.store = IncrementalCurrentStore(self.codec, key_dimensions=KEY_DIMENSIONS)
        self.store.ingest_decision(event('v1', '2018-01-01T00:00:00'), memory(1), key(),
                                   decision('CREATE'))

    def test_rejected_event_can_be_retried_in_order(self):
        with self.assertRaises(ValueError) as caught:
            self.store.ingest_decision(event('v0', '2017-01-01T00:00:00'), memory(3), key(),
                                       decision('UPDATE'))
        self.assertIn('chronological', str(caught.exception))
        self.assertEqual([r['id'] for r in self.store.rejections], ['v0'])
        self.assertEqual([e['id'] for e in self.store.entries], ['v1'])
        # 同一条事件按正确顺序重投：必须真正入库，而不是 'Duplicate version id'
        receipt = self.store.ingest_decision(event('v0', '2019-01-01T00:00:00'), memory(3), key(),
                                             decision('UPDATE'))
        self.assertEqual((receipt['action'], receipt['id']), ('UPDATE', 'v0'))
        self.assertEqual([e['id'] for e in self.store.entries], ['v1', 'v0'])
        self.assertEqual([e['kind'] for e in self.store.entries], ['code', 'full'])
        self.assertEqual(self.store.current(STATE)['id'], 'v0')
        # 原拒绝记录仍可追溯，且不参与 id 唯一性校验
        self.assertEqual([(r['id'], r['reason']) for r in self.store.rejections], [('v0', 'out_of_order')])
        self.store.validate_protocol()
        print(f'\nF2 retry after rejection accepted: entries={[e["id"] for e in self.store.entries]} '
              f'rejections={[r["id"] for r in self.store.rejections]}')

    def test_rejection_record_survives_roundtrip(self):
        with self.assertRaises(ValueError):
            self.store.ingest_decision(event('v0', '2017-01-01T00:00:00'), memory(3), key(),
                                       decision('UPDATE'))
        path = pathlib.Path(tempfile.mkdtemp()) / 'rejected.mem'
        self.store.save(path)
        reloaded = IncrementalCurrentStore.load(path, self.codec, key_dimensions=KEY_DIMENSIONS)
        self.assertEqual(reloaded.rejections, self.store.rejections)
        reloaded.ingest_decision(event('v0', '2019-01-01T00:00:00'), memory(3), key(),
                                 decision('UPDATE'))
        self.assertEqual(reloaded.current(STATE)['id'], 'v0')


class CommitUndoIsBoundedTest(unittest.TestCase):
    """F6：commit() 的 undo 快照只保留最近一次，不随写入次数增长；失败写入不登记快照。"""

    def setUp(self):
        self.codec = codec_pair()
        self.store = IncrementalCurrentStore(self.codec, key_dimensions=KEY_DIMENSIONS)

    def fact(self, index, value='Google', observed_at=None):
        return AtomicFact(id=f'fact_{index:06d}', text=f'Alice works at {value}',
                          observed_at=observed_at or f'{2018 + index}-01-01T00:00:00',
                          subject='Alice', attribute='employer', value=value, source_id='input_000001')

    def test_undo_snapshot_does_not_grow_with_commits(self):
        record = _Record(current_version_id=None)
        results = []
        for index in range(6):
            results.append(self.store.commit(record=record, fact=self.fact(index),
                                             memory=memory(10 + index), semantic_key=key()))
            record.current_version_id = results[-1]['version_id']
            self.assertLessEqual(len(self.store._commit_undo), 1)
        self.assertEqual(len(self.store._commit_undo), 1)
        self.assertEqual(list(self.store._commit_undo), [results[-1]['version_id']])
        # 最近一次仍可回滚，更早的已经不能（跨调用不再持有完整状态副本）
        self.assertFalse(self.store.rollback(results[0]))
        self.assertTrue(self.store.rollback(results[-1]))
        self.assertEqual(len(self.store._commit_undo), 0)
        self.assertEqual(self.store.current(STATE)['id'], results[-2]['version_id'])

    def test_failed_commit_does_not_register_undo(self):
        record = _Record(current_version_id=None)
        with self.assertRaises(ValueError):
            self.store.commit(record=record, fact=self.fact(1), memory=memory(20), semantic_key=key(),
                              decision=StateDecision('UPDATE', 'missing_state', .9, 'mismatch'))
        self.assertEqual(self.store._commit_undo, {})
        self.assertEqual(self.store.entries, [])
