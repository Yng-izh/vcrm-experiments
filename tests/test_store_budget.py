from __future__ import annotations
# P1 存储字节预算：budget_mode='drop_oldest' 的确定性淘汰策略 + retain_all 零回归。
#
# 口径（与 vcrm/store.py 类 docstring 一致）：
#   1. 非版本账（pending → duplicates → rejections）先清；
#   2. 再按 (entry['time'], entry['id']) 升序淘汰历史 code；
#   3. 仍超预算 → 显式 ValueError（整笔写入回滚）。
#   current(full) 版本永不淘汰；每次淘汰都记 audit（entry id + 释放字节）。
# codec 用同构 E3 结构的小尺寸 SlotCodec（随机初始化、conditioned=False），
# 本文件只验证预算协议，不验证权重质量。
import pathlib
import tempfile
import unittest

import torch

from vcrm.codec import SlotCodec
from vcrm.fact_schema import StateDecision
from vcrm.index import key_vector
from vcrm.store import IncrementalCurrentStore

DIMENSION, MEMORY_SLOTS, KEY_DIMENSIONS = 32, 8, 8
CODEC_SLOTS, CODEC_WIDTH, CODEC_HEADS = 4, 16, 4
STATE_A, STATE_B = 'Alice:employer', 'Bob:employer'

# (state, version_id, time, action, memory_seed)
WRITES = [
    (STATE_A, 'a1', '2016-01-01T00:00:00', 'CREATE', 1),
    (STATE_A, 'a2', '2018-01-01T00:00:00', 'UPDATE', 2),
    (STATE_A, 'a3', '2020-01-01T00:00:00', 'UPDATE', 3),
    (STATE_B, 'b1', '2017-01-01T00:00:00', 'CREATE', 4),
    (STATE_B, 'b2', '2019-01-01T00:00:00', 'UPDATE', 5),
]


def codec_pair():
    torch.manual_seed(7)
    return SlotCodec(DIMENSION, CODEC_SLOTS, CODEC_WIDTH, CODEC_HEADS, False, 0., 7,
                     'residual', None, 'full', 8.0).eval().requires_grad_(False)


def memory(seed):
    generator = torch.Generator().manual_seed(seed)
    return (torch.randn(MEMORY_SLOTS, DIMENSION, generator=generator) * .5).half().float()


def key(text='Alice works at Google'):
    return key_vector(text, KEY_DIMENSIONS)


def decision(action, state):
    return StateDecision(action, state, .9, 'budget test')


def event(version_id, time):
    return {'id': version_id, 'time': time}


class BudgetTestBase(unittest.TestCase):
    def setUp(self):
        self.codec = codec_pair()
        self.tmp = pathlib.Path(tempfile.mkdtemp())

    def fresh(self, byte_budget=None, budget_mode='retain_all'):
        return IncrementalCurrentStore(self.codec, key_dimensions=KEY_DIMENSIONS,
                                       byte_budget=byte_budget, budget_mode=budget_mode)

    def replay(self, store, writes=WRITES):
        receipts = []
        for state, version_id, time, action, seed in writes:
            receipts.append(store.ingest_decision(event(version_id, time), memory(seed), key(),
                                                  decision(action, state)))
        return receipts

    def reference_sizes(self):
        """用无预算参照 store 量真实字节：满配体积与"只剩 current"的地板体积。"""
        reference = self.fresh()
        self.replay(reference)
        full = len(reference.serialize())
        # drop_oldest 的 aux header 多带 1 字节模式名（'drop_oldest' vs 'retain_all'），
        # 所以它自己的"满配体积"要单独量，否则边界会差 1 字节。
        drop_reference = self.fresh(byte_budget=10 ** 9, budget_mode='drop_oldest')
        self.replay(drop_reference)
        full_drop = len(drop_reference.serialize())
        self.assertEqual(drop_reference.evictions, [])
        floor_store = self.fresh()
        floor_store.ingest_decision(event('a3', '2020-01-01T00:00:00'), memory(3), key(),
                                    decision('CREATE', STATE_A))
        floor_store.ingest_decision(event('b2', '2019-01-01T00:00:00'), memory(5), key(),
                                    decision('CREATE', STATE_B))
        floor = len(floor_store.serialize())
        codes = sorted((e for e in reference.entries if e['kind'] == 'code'),
                       key=lambda e: (e['time'], e['id']))
        return full, floor, [e['id'] for e in codes], full_drop

    def populated_drop_oldest(self, byte_budget):
        """按 WRITES 写入；预算在最后一次写入时才真正卡住（体积单调增长）。"""
        store = self.fresh(byte_budget=byte_budget, budget_mode='drop_oldest')
        return store, self.replay(store)


class DropOldestPolicyTest(BudgetTestBase):
    def test_oldest_code_is_evicted_first_and_audited(self):
        _full, _floor, order, full_drop = self.reference_sizes()
        store, receipts = self.populated_drop_oldest(full_drop - 1)
        self.assertEqual(order, ['a1', 'b1', 'a2'])          # (time, id) 升序
        audit = receipts[-1]['evicted']
        self.assertEqual(len(audit), 1)
        record = audit[0]
        self.assertEqual(record['seq'], 0)
        self.assertEqual(record['byte_budget'], full_drop - 1)
        self.assertGreater(record['freed_bytes'], 0)
        self.assertEqual([group['group'] for group in record['groups']], ['historical_code'])
        self.assertEqual(record['groups'][0]['ids'], ['a1'])
        self.assertEqual(record['groups'][0]['state'], STATE_A)
        self.assertEqual(record['groups'][0]['time'], '2016-01-01T00:00:00')
        self.assertLessEqual(len(store.serialize()), full_drop - 1)
        self.assertEqual(store.accounting()['evicted_ids'], ['a1'])
        self.assertEqual(store.accounting()['evictions'], 1)
        self.assertEqual(store.accounting()['budget_mode'], 'drop_oldest')
        self.assertGreater(store.accounting()['evicted_bytes'], 0)
        # current 仍在位：每状态唯一 full 版本 + blocks 指向它
        self.assertEqual([e['id'] for e in store.entries if e['kind'] == 'full'], ['a3', 'b2'])
        self.assertEqual(store.current(STATE_A)['id'], 'a3')
        self.assertEqual(store.current(STATE_B)['id'], 'b2')
        self.assertEqual(store.blocks, {STATE_A: 'a3', STATE_B: 'b2'})

    def test_eviction_order_is_deterministic_across_independent_stores(self):
        _full, floor, _order, _full_drop = self.reference_sizes()
        budget = floor + 60
        first, _ = self.populated_drop_oldest(budget)
        second, _ = self.populated_drop_oldest(budget)
        self.assertEqual(first.accounting()['evicted_ids'], second.accounting()['evicted_ids'])
        self.assertEqual(first.serialize(), second.serialize())
        evicted = first.accounting()['evicted_ids']
        self.assertGreaterEqual(len(evicted), 2)
        self.assertNotIn('a3', evicted)                 # current 永不淘汰
        self.assertNotIn('b2', evicted)
        self.assertLessEqual(len(first.serialize()), budget)
        self.assertTrue(all(e['kind'] == 'full' for e in first.entries))

    def test_single_guard_evicts_multiple_victims_in_time_order(self):
        """同一次守卫里淘汰多条时，严格按 (time, id) 升序（预先写明的顺序）。"""
        _full, floor, order, _full_drop = self.reference_sizes()
        store = self.fresh(byte_budget=10 ** 9, budget_mode='drop_oldest')
        self.replay(store)
        self.assertEqual(store.evictions, [])
        store.byte_budget = floor + 60                     # 下一次写入要淘汰多条
        receipt = store.ingest_decision(event('a4', '2022-01-01T00:00:00'), memory(8), key(),
                                        decision('UPDATE', STATE_A))
        groups = receipt['evicted'][0]['groups']
        ids = [version_id for group in groups for version_id in group['ids']]
        self.assertGreaterEqual(len(ids), 2)
        # a3 被这次 UPDATE 降级成 code 后也进入候选池，排在最后（2020 最晚）
        self.assertEqual(ids, (order + ['a3'])[:len(ids)])
        self.assertEqual([group['group'] for group in groups], ['historical_code'] * len(groups))
        self.assertLessEqual(len(store.serialize()), floor + 60)
        self.assertEqual([e['id'] for e in store.entries if e['kind'] == 'full'], ['b2', 'a4'])
        store.validate_protocol()

    def test_current_versions_are_never_evicted(self):
        _full, floor, _order, _full_drop = self.reference_sizes()
        store, _ = self.populated_drop_oldest(10 ** 9)      # 先按大预算写满
        before = store.serialize()
        store.byte_budget = floor - 1                        # 地板都装不下的预算
        with self.assertRaises(ValueError) as caught:
            store.ingest_decision(event('b3', '2021-01-01T00:00:00'), memory(6), key(),
                                  decision('UPDATE', STATE_B))
        self.assertIn('never be evicted', str(caught.exception))
        # 整笔写入回滚：字节逐比特不变，淘汰审计也撤销（不留半截淘汰记录）
        self.assertEqual(store.serialize(), before)
        self.assertEqual(store.evictions, [])
        self.assertEqual([e['id'] for e in store.entries if e['kind'] == 'full'], ['a3', 'b2'])
        self.assertEqual(store.current(STATE_A)['id'], 'a3')
        self.assertEqual(store.current(STATE_B)['id'], 'b2')
        store.validate_protocol()

    def test_pending_is_cleared_before_any_historical_code(self):
        store = self.fresh(byte_budget=10 ** 9, budget_mode='drop_oldest')
        self.replay(store)
        store.ingest_decision(event('p1', '2020-06-01T00:00:00'), memory(7), key(),
                              decision('DEFER', STATE_A))
        self.assertEqual([r['id'] for r in store.pending], ['p1'])
        store.byte_budget = len(store.serialize()) - 1       # 下一次写入必然超预算
        receipt = store.ingest_decision(event('b3', '2021-01-01T00:00:00'), memory(6), key(),
                                        decision('UPDATE', STATE_B))
        record = receipt['evicted'][0]
        self.assertEqual([group['group'] for group in record['groups']], ['pending'])
        self.assertEqual(record['groups'][0]['ids'], ['p1'])
        self.assertEqual(record['groups'][0]['reason'], 'budget_drop_non_version_record')
        self.assertEqual(store.pending, [])                  # 清 pending 就够，未动历史
        self.assertEqual(sorted(e['id'] for e in store.entries if e['kind'] == 'code'),
                         ['a1', 'a2', 'b1', 'b2'])   # b2 被本次 UPDATE 降级为 code
        self.assertEqual(store.current(STATE_A)['id'], 'a3')
        self.assertEqual(store.current(STATE_B)['id'], 'b3')
        self.assertLessEqual(len(store.serialize()), store.byte_budget)

    def test_evicted_store_roundtrips_and_keeps_protocol(self):
        _full, _floor, _order, full_drop = self.reference_sizes()
        store, _ = self.populated_drop_oldest(full_drop - 1)
        store.validate_protocol()
        path = self.tmp / 'drop_oldest.mem'
        written = store.save(path)
        reloaded = IncrementalCurrentStore.load(path, self.codec, key_dimensions=KEY_DIMENSIONS,
                                                byte_budget=full_drop - 1, budget_mode='drop_oldest')
        self.assertEqual(written, path.stat().st_size)
        self.assertEqual(reloaded.serialize(), store.serialize())
        self.assertEqual(reloaded.serialize(), path.read_bytes())
        self.assertEqual([e['id'] for e in reloaded.entries], ['a2', 'a3', 'b1', 'b2'])
        self.assertEqual(reloaded.current(STATE_A)['id'], 'a3')
        self.assertEqual(reloaded.current(STATE_B)['id'], 'b2')
        reloaded.validate_protocol()
        # 预算模式必须显式匹配，不静默换模式
        with self.assertRaises(ValueError) as caught:
            IncrementalCurrentStore.load(path, self.codec, key_dimensions=KEY_DIMENSIONS,
                                         byte_budget=full_drop - 1)
        self.assertIn('budget mode mismatch', str(caught.exception))

    def test_exact_budget_boundary_evicts_nothing(self):
        _full, _floor, _order, full_drop = self.reference_sizes()
        store, receipts = self.populated_drop_oldest(full_drop)
        self.assertEqual(len(store.serialize()), full_drop)
        self.assertEqual(store.evictions, [])
        self.assertTrue(all(receipt['evicted'] == [] for receipt in receipts))
        self.assertEqual([e['id'] for e in store.entries], ['a1', 'a2', 'a3', 'b1', 'b2'])
        store.validate_protocol()

    def test_drop_oldest_requires_an_explicit_budget(self):
        with self.assertRaises(ValueError) as caught:
            self.fresh(byte_budget=None, budget_mode='drop_oldest')
        self.assertIn('requires an explicit byte_budget', str(caught.exception))
        with self.assertRaises(ValueError):
            IncrementalCurrentStore(self.codec, key_dimensions=KEY_DIMENSIONS,
                                    byte_budget=1024, budget_mode='evict_fifo')


class RetainAllRegressionTest(BudgetTestBase):
    """retain_all 语义零回归：超预算显式拒绝 + 整体回滚，永不淘汰。"""

    def test_over_budget_write_is_rejected_and_rolled_back(self):
        full, _floor, _order, _full_drop = self.reference_sizes()
        store = self.fresh(byte_budget=full - 1)
        self.replay(store, WRITES[:-1])
        before = store.serialize()
        with self.assertRaises(ValueError) as caught:
            store.ingest_decision(event('b2', '2019-01-01T00:00:00'), memory(5), key(),
                                  decision('UPDATE', STATE_B))
        self.assertIn('explicit rejection instead of silent eviction', str(caught.exception))
        self.assertEqual(store.serialize(), before)          # 逐比特回滚
        self.assertEqual(store.evictions, [])                # retain_all 从不淘汰
        self.assertEqual(store.accounting()['evictions'], 0)
        self.assertEqual(store.accounting()['evicted_ids'], [])
        self.assertEqual(store.accounting()['budget_mode'], 'retain_all')
        self.assertEqual(store.accounting()['byte_guardrail'], full - 1)

    def test_retain_all_within_budget_keeps_every_version(self):
        full, _floor, _order, _full_drop = self.reference_sizes()
        store = self.fresh(byte_budget=full)
        receipts = self.replay(store)
        self.assertEqual(len(store.serialize()), full)
        self.assertEqual([e['id'] for e in store.entries], ['a1', 'a2', 'a3', 'b1', 'b2'])
        self.assertEqual([e['kind'] for e in store.entries], ['code', 'code', 'full', 'code', 'full'])
        self.assertTrue(all(receipt['evicted'] == [] for receipt in receipts))
        self.assertEqual(store.accounting()['evictions'], 0)
        store.validate_protocol()

    def test_retain_all_and_drop_oldest_agree_below_budget(self):
        full, _floor, _order, full_drop = self.reference_sizes()
        retained = self.fresh(byte_budget=full)
        dropped = self.fresh(byte_budget=full_drop, budget_mode='drop_oldest')
        self.replay(retained)
        self.replay(dropped)
        # 预算够用时两种模式的存储内容完全一致（唯一变量是超预算后的行为）。
        # 注意 aux header 里显式记录 budget_mode 字符串，因此两者字节数相差
        # len('drop_oldest') - len('retain_all') = 1 字节，其余逐项相同。
        self.assertEqual(dropped.evictions, [])
        self.assertEqual([(e['id'], e['state'], e['time'], e['kind'], e['keyframe'],
                           e['anchor_id'], e['compression_count']) for e in dropped.entries],
                         [(e['id'], e['state'], e['time'], e['kind'], e['keyframe'],
                           e['anchor_id'], e['compression_count']) for e in retained.entries])
        self.assertEqual([e['payload'].tobytes() for e in dropped.entries],
                         [e['payload'].tobytes() for e in retained.entries])
        self.assertEqual([e['key'].tobytes() for e in dropped.entries],
                         [e['key'].tobytes() for e in retained.entries])
        self.assertEqual((dropped.blocks, dropped.counts), (retained.blocks, retained.counts))
        self.assertEqual(len(dropped.serialize()), len(retained.serialize()) + 1)


if __name__ == '__main__':
    unittest.main()
