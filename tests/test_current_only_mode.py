from __future__ import annotations
# C 组（Current-only Latent）vs D 组（Versioned Latent）读取口径对照测试。
#
# 同一份写入结果、同一个 store、同一个 codec/reader，唯一变量是 read_mode：
#   current_only  → bundle 只含 current（full）版本，历史 code 一个都不解码；
#   trajectory    → canonical 布局的 current + 全部可见历史；
#   pair          → 既有口径（current + 一个历史位），零回归。
# 断言重点是"读取是否允许历史"这一件事，不做任何语言推理（stub reader）。

import unittest

import torch

from vcrm.codec import SlotCodec, VersionMarkers
from vcrm.data_views import PublicEventCounter, public_write_event
from vcrm.fact_extractor import FactExtractor
from vcrm.index import SemanticIndex
from vcrm.runtime import PredictedRuntime, QueryParser
from vcrm.state_index import SEMANTIC_DIMENSIONS, StateIndex
from vcrm.state_router import StateRouter
from vcrm.store import IncrementalCurrentStore
from vcrm.update_resolver import UpdateResolver

HIDDEN = SEMANTIC_DIMENSIONS
MEMORY_SLOTS = 8
CODEC_SLOTS = 4
QUESTION = 'The head of government of France is _X_'

FR_2016 = ('2016-01-01T00:00:00', 'In 2016: The head of government of France is Bernard Cazeneuve')
FR_2018 = ('2018-01-01T00:00:00', 'In 2018: The head of government of France is Edouard Philippe')
FR_2020 = ('2020-01-01T00:00:00', 'In 2020: The head of government of France is Jean Castex')


class VersionAwareReader:
    """stub reader：按 bundle 里**实际出现**的版本内存给答案（不做语言推理）。

    ``labels`` 把版本 id 映射到答案文本，空串表示该版本没有金标。
    这样"金标只在历史版本"的构造用例就能直接区分两种读取模式：
    读到了历史版本才可能答出答案，只读 current 则答不出。
    """

    def __init__(self):
        self.cfg = {'read_latent_slots': None, 'state_similarity': 0.8}
        self.device = torch.device('cpu')
        self.versions = {}      # version_id -> decoded memory
        self.order = {}         # version_id -> 时间序（越小越旧）
        self.labels = {}        # version_id -> 答案文本
        self.bundles = []       # 每次读取实际收到的 bundle

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

    def present(self, bundle):
        """bundle 里逐行出现的版本 id 集合（要求整段连续命中，避免巧合命中）。"""
        found = set()
        if bundle is None:
            return found
        for version_id, memory in self.versions.items():
            width = int(memory.shape[0])
            for start in range(0, int(bundle.shape[0]) - width + 1):
                if torch.equal(bundle[start:start + width], memory):
                    found.add(version_id)
                    break
        return found

    def answer(self, query, bundle=None, text=''):
        self.bundles.append(bundle)
        if bundle is None:
            return {'answer': '', 'read_latent_slots': 0}
        present = self.present(bundle)
        # 从新到旧：答出"bundle 里最新的、且带金标的版本"
        for version_id in sorted(present, key=lambda vid: self.order[vid], reverse=True):
            if self.labels.get(version_id):
                return {'answer': self.labels[version_id], 'read_latent_slots': int(bundle.shape[0])}
        return {'answer': '', 'read_latent_slots': int(bundle.shape[0])}


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


def public_event(public_id, pair):
    return public_write_event({'id': f'src_{public_id}', 'time': pair[0], 'text': pair[1]},
                              public_id)


class CurrentOnlyModeTest(unittest.TestCase):
    """一个三版本状态：current=v3(2020, full)，v1/v2 是被压成 16-slot code 的历史。"""

    def setUp(self):
        self.reader = VersionAwareReader()
        self.store = IncrementalCurrentStore(make_codec(), key_dimensions=SEMANTIC_DIMENSIONS)
        self.runtime = PredictedRuntime(
            self.reader, make_codec(), VersionMarkers(HIDDEN), DeterministicIndex(),
            self.store, StateIndex(), router=StateRouter(top_k=5),
            resolver=UpdateResolver(recency_update=True, recency_min_confidence=0.0),
            extractor=FactExtractor(), query_parser=QueryParser(),
            time_mode='window_end', allow_as_of=False)
        self.counter = PublicEventCounter()
        for pair in (FR_2016, FR_2018, FR_2020):
            self.runtime.write_predicted(public_event(self.counter.next_id(), pair))
        self.entries = list(self.store.entries)
        self.assertEqual([entry['time'] for entry in self.entries],
                         ['2016-01-01T00:00:00', '2018-01-01T00:00:00', '2020-01-01T00:00:00'])
        self.assertEqual([entry['kind'] for entry in self.entries], ['code', 'code', 'full'])
        self.state_id = self.entries[0]['state']
        self.assertEqual({entry['state'] for entry in self.entries}, {self.state_id})
        self.current = self.store.current(self.state_id)
        self.assertEqual(self.current['id'], self.entries[-1]['id'])
        self.assertEqual(self.store.blocks[self.state_id], self.current['id'])
        # 让 stub reader 认得每个版本的内存（id 只有写完之后才知道）
        for index, entry in enumerate(self.entries):
            self.reader.versions[entry['id']] = self.store.decode(entry)
            self.reader.order[entry['id']] = index
        self.snapshot = self.store.serialize()

    def ask(self, read_mode, window_end='2021-01-01T00:00:00', observation_cutoff=None):
        self.reader.bundles.clear()
        result = self.runtime.answer_predicted(
            {'id': 'q1', 'question': QUESTION, 'time': window_end, 'as_of': window_end,
             'answers': [], 'latest': True, 'type': 'latest'},
            window_end=window_end, read_mode=read_mode,
            observation_cutoff=observation_cutoff)
        return result, (self.reader.bundles[-1] if self.reader.bundles else None)

    # -- (a) current_only：bundle 恰为 current ---------------------------------
    def test_current_only_bundle_is_exactly_the_current_version(self):
        result, bundle = self.ask('current_only')
        self.assertFalse(result['no_state_match'])
        self.assertEqual(result['read_mode'], 'current_only')
        current_slots = self.store.entry_slots(self.current)
        self.assertEqual(current_slots, MEMORY_SLOTS)
        # bundle = [role marker] + current 的 payload + [time marker]，没有任何历史 code
        self.assertEqual(int(bundle.shape[0]), current_slots + 2)
        self.assertTrue(torch.equal(bundle[1:1 + current_slots],
                                    self.reader.versions[self.current['id']]))
        self.assertEqual(self.reader.present(bundle), {self.current['id']})
        # 审计字段：槽位账 = 每状态 current 槽位数之和；历史全部被排除
        self.assertEqual(result['read_latent_slots'], current_slots)
        self.assertEqual(result['bundle_latent_slots'], current_slots)
        self.assertEqual(result['excluded_historical_versions'], 2)
        self.assertEqual(result['bundle_layout'], ['current'])
        self.assertEqual(result['bundle_version_ids'], [self.current['id']])
        self.assertEqual(result['trajectory_length'], 1)
        self.assertEqual(result['selected_version_id'], self.current['id'])
        self.assertEqual(result['version_role'], 'current')
        # 读取不改变写入结果（C/D 共用同一份 store）
        self.assertEqual(self.store.serialize(), self.snapshot)

    # -- (b) trajectory：canonical 布局含 current + 全部历史 -------------------
    def test_trajectory_bundle_contains_current_and_history_in_canonical_order(self):
        result, bundle = self.ask('trajectory')
        expected_ids = [self.current['id']] + [entry['id'] for entry in self.entries[:-1]]
        self.assertEqual(result['bundle_version_ids'], expected_ids)
        self.assertEqual(result['bundle_layout'], ['current', 'historical', 'historical'])
        self.assertEqual(result['trajectory_length'], 3)
        self.assertEqual(result['excluded_historical_versions'], 0)
        payload_slots = sum(self.store.entry_slots(entry) for entry in self.entries)
        self.assertEqual(payload_slots, MEMORY_SLOTS + 2 * CODEC_SLOTS)
        self.assertEqual(result['read_latent_slots'], payload_slots)
        self.assertEqual(int(bundle.shape[0]), payload_slots + 2 * 3)   # 每条一个 marker
        self.assertEqual(self.reader.present(bundle), set(expected_ids))
        self.assertEqual(self.store.serialize(), self.snapshot)

    # -- (c) 金标只在历史版本：两种模式必须表现不同 ----------------------------
    def test_gold_answer_only_in_history_is_unreachable_for_current_only(self):
        historical = self.entries[1]                 # 2018 = Edouard Philippe
        self.reader.labels = {historical['id']: 'Edouard Philippe'}
        trajectory, trajectory_bundle = self.ask('trajectory', window_end='2018-12-31T00:00:00')
        self.assertIn(historical['id'], self.reader.present(trajectory_bundle))
        self.assertEqual(trajectory['answer'], 'Edouard Philippe')
        # trajectory 的 selected 是当时可见的最新版本（2020 current），金标来自历史条目
        self.assertEqual(trajectory['version_role'], 'current')
        self.assertIn(historical['id'], trajectory['bundle_version_ids'])
        current_only, current_bundle = self.ask('current_only', window_end='2018-12-31T00:00:00')
        self.assertNotIn(historical['id'], self.reader.present(current_bundle))
        self.assertEqual(current_only['answer'], '')
        # 同一份写入：D 组读得到、C 组读不到，差别只来自"是否允许历史"
        self.assertEqual(current_only['bundle_version_ids'], [self.current['id']])
        self.assertGreaterEqual(current_only['excluded_historical_versions'], 1)
        pair, pair_bundle = self.ask('pair', window_end='2018-12-31T00:00:00')
        self.assertIn(historical['id'], self.reader.present(pair_bundle))
        self.assertEqual(pair['answer'], 'Edouard Philippe')

    # -- (d) observation_cutoff 在两模式下语义一致 ----------------------------
    def test_observation_cutoff_semantics_are_identical_across_modes(self):
        cutoff = '2019-01-01T00:00:00'
        for read_mode in ('pair', 'trajectory'):
            result, bundle = self.ask(read_mode, observation_cutoff=cutoff)
            self.assertEqual(result['observation_cutoff'], cutoff)
            # 晚于截止的 current 绝不可读
            self.assertNotIn(self.current['id'], self.reader.present(bundle))
            self.assertNotIn(self.current['id'], result['bundle_version_ids'])
            self.assertTrue(all(time <= cutoff for time in result['bundle_version_times']))
            self.assertTrue(result['future_current_suppressed'])
            self.assertEqual(result['hidden_future_versions'], 1)
        # current_only 在截止时刻之后没有 current 可读 → 显式 no_state_match，
        # 不退回历史（历史本来就是本模式排除的东西），也不越界读未来
        blocked, blocked_bundle = self.ask('current_only', observation_cutoff=cutoff)
        self.assertIsNone(blocked_bundle)
        self.assertTrue(blocked['no_state_match'])
        self.assertEqual(blocked['answer'], '')
        self.assertTrue(blocked['future_current_suppressed'])
        self.assertEqual(blocked['hidden_future_versions'], 1)
        self.assertEqual(blocked['read_latent_slots'], 0)
        self.assertEqual(blocked['excluded_historical_versions'], 2)
        # 截止晚于 current 时两模式都能读，且 current_only 仍然只读 current
        late = '2020-06-01T00:00:00'
        trajectory, trajectory_bundle = self.ask('trajectory', observation_cutoff=late)
        current_only, current_bundle = self.ask('current_only', observation_cutoff=late)
        self.assertIn(self.current['id'], self.reader.present(trajectory_bundle))
        self.assertIn(self.current['id'], self.reader.present(current_bundle))
        self.assertEqual(self.reader.present(current_bundle), {self.current['id']})
        self.assertEqual(trajectory['read_latent_slots'] - current_only['read_latent_slots'],
                         2 * CODEC_SLOTS)
        self.assertFalse(current_only['future_current_suppressed'])
        self.assertEqual(current_only['excluded_historical_versions'], 2)
        self.assertEqual(self.store.serialize(), self.snapshot)

    # -- 未知 read_mode 必须显式报错（不给静默降级） ---------------------------
    def test_unknown_read_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            self.ask('current')
        self.assertEqual(self.store.serialize(), self.snapshot)


class PairModeRegressionTest(unittest.TestCase):
    """pair 口径零回归：字段与既有实现一致，只是多了 C/D 审计字段。"""

    def setUp(self):
        self.reader = VersionAwareReader()
        self.store = IncrementalCurrentStore(make_codec(), key_dimensions=SEMANTIC_DIMENSIONS)
        self.runtime = PredictedRuntime(
            self.reader, make_codec(), VersionMarkers(HIDDEN), DeterministicIndex(),
            self.store, StateIndex(), router=StateRouter(top_k=5),
            resolver=UpdateResolver(recency_update=True, recency_min_confidence=0.0),
            extractor=FactExtractor(), query_parser=QueryParser(),
            time_mode='window_end', allow_as_of=False)
        counter = PublicEventCounter()
        for pair in (FR_2016, FR_2018, FR_2020):
            self.runtime.write_predicted(public_event(counter.next_id(), pair))
        self.entries = list(self.store.entries)

    def ask(self, **kwargs):
        return self.runtime.answer_predicted(
            {'id': 'q1', 'question': QUESTION, 'time': '2021-01-01T00:00:00',
             'as_of': '2021-01-01T00:00:00', 'answers': [], 'latest': True, 'type': 'latest'},
            window_end='2021-01-01T00:00:00', **kwargs)

    def test_default_read_mode_is_still_pair(self):
        result = self.ask()
        self.assertEqual(result['read_mode'], 'pair')
        self.assertEqual(result['bundle_layout'], ['current', 'historical'])
        self.assertEqual(result['trajectory_length'], 2)
        self.assertEqual(result['bundle_version_ids'],
                         [self.entries[-1]['id'], self.entries[-2]['id']])
        self.assertEqual(result['selected_version_time'], '2020-01-01T00:00:00')
        self.assertEqual(result['version_role'], 'current')
        self.assertFalse(result['no_state_match'])
        # pair 只留一个历史位 → 更早的历史被排除，审计字段如实记账
        self.assertEqual(result['excluded_historical_versions'], 1)
        self.assertEqual(result['read_latent_slots'], MEMORY_SLOTS + CODEC_SLOTS)

    def test_pair_is_the_trajectory_restricted_to_one_history_slot(self):
        pair = self.ask(read_mode='pair')
        trajectory = self.ask(read_mode='trajectory')
        # 两者共用同一份 canonical 布局（current 在首位，历史按时间升序）
        self.assertEqual(pair['bundle_version_ids'][0], trajectory['bundle_version_ids'][0])
        self.assertEqual(pair['bundle_layout'], ['current', 'historical'])
        self.assertEqual(trajectory['bundle_layout'], ['current', 'historical', 'historical'])
        self.assertEqual(trajectory['bundle_version_ids'][1:],
                         [entry['id'] for entry in self.entries[:-1]])   # 历史按时间升序
        self.assertIn(pair['bundle_version_ids'][1], trajectory['bundle_version_ids'])
        self.assertGreater(trajectory['read_latent_slots'], pair['read_latent_slots'])
        self.assertEqual(pair['excluded_historical_versions'], 1)
        self.assertEqual(trajectory['excluded_historical_versions'], 0)


if __name__ == '__main__':
    unittest.main()
