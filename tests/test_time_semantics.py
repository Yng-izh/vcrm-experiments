from __future__ import annotations
# t25：统一时间语义（target_time 来自题面 / observation_cutoff 来自提问时刻）
#
# 核心回归：LongMemEval 的 "Where did I live in 2018?" —— 相关证据在 2023 年的
# 会话里才被用户讲述。旧实现把题面锚点（2018-12-31）同时当成 observation_cutoff，
# 于是 2023 年写入的证据被可见性硬上界裁掉，整条答案消失。
# 默认 window_end 路径下 cutoff 必须取 question_date（2023 年），target 才是 2018。
#
# 另外钉住三件事：
#   1) question_date 缺失 → cutoff=None + time_source='missing_question_date'（不猜）；
#   2) 调用方给出的公开截止（episode 观测窗口末刻）优先于 question_date；
#   3) TempLAMA 旧口径（target_time == cutoff）只在显式 allow_as_of 路径保留。
import unittest

from vcrm.data_views import public_query
from vcrm.query_time import (
    QUERY_TIME_MODES,
    TIME_SOURCE_AS_OF_LEGACY,
    TIME_SOURCE_CALLER,
    TIME_SOURCE_MISSING_QUESTION_DATE,
    TIME_SOURCE_QUESTION_DATE,
    QueryTimeBounds,
    QueryTimeResolver,
    resolve_query_time,
)

QUESTION_2018 = 'Where did I live in 2018?'
QUESTION_DATE_2023 = '2023/06/15 (Thu) 10:30'      # LongMemEval 的原始时间格式
EVIDENCE_TOLD_IN_2023 = '2023-05-20T00:00:00'      # 2023 年才讲述的 2018 事实
CUTOFF_2018 = '2018-12-31T23:59:59'                # 旧实现误用的"历史锚截止"


class LongMemEvalCutoffTest(unittest.TestCase):
    """默认 window_end 路径：历史锚只做 target，绝不做 cutoff。"""

    def setUp(self):
        self.bounds = resolve_query_time(QUESTION_2018, QUESTION_DATE_2023)

    def test_target_time_comes_from_question_text(self):
        self.assertEqual(self.bounds.target_time, CUTOFF_2018)   # 2018 年窗口末刻
        self.assertEqual(self.bounds.time_anchor, 'question_text_anchor')

    def test_cutoff_comes_from_question_date(self):
        self.assertEqual(self.bounds.observation_cutoff, '2023-06-15T10:30:00')
        self.assertEqual(self.bounds.time_mode, 'window_end')
        self.assertEqual(self.bounds.time_source, TIME_SOURCE_QUESTION_DATE)
        self.assertFalse(self.bounds.allow_as_of)
        self.assertNotEqual(self.bounds.observation_cutoff, self.bounds.target_time)

    def test_2023_evidence_is_not_excluded(self):
        """核心断言：2023 年讲述的 2018 证据在默认路径下仍然可见。"""
        self.assertTrue(self.bounds.visible(EVIDENCE_TOLD_IN_2023))
        # 对照：旧口径（cutoff=历史锚）会把它整条裁掉
        legacy = QueryTimeBounds(target_time=CUTOFF_2018,
                                 observation_cutoff=CUTOFF_2018,
                                 time_mode='as_of', time_source=TIME_SOURCE_AS_OF_LEGACY,
                                 allow_as_of=True)
        self.assertFalse(legacy.visible(EVIDENCE_TOLD_IN_2023))

    def test_selection_still_uses_target_time(self):
        """可见性放宽不等于版本选择放宽：选择上界仍是 min(cutoff, target)。"""
        self.assertEqual(self.bounds.selection_bound, CUTOFF_2018)
        self.assertTrue(self.bounds.visible('2019-03-01T00:00:00'))

    def test_question_without_anchor_reads_latest_visible(self):
        bounds = resolve_query_time('What is my current employer?', QUESTION_DATE_2023)
        self.assertIsNone(bounds.target_time)                     # 无锚 → 读最新可见
        self.assertEqual(bounds.observation_cutoff, '2023-06-15T10:30:00')

    def test_public_query_gets_target_not_cutoff(self):
        """public_query 的 time 通道承载 target_time（下游做版本选择上界）。"""
        view = public_query({'question': QUESTION_2018},
                            time_mode='window_end', window_end=self.bounds.target_time)
        self.assertEqual(view['time'], CUTOFF_2018)
        self.assertEqual(view['time_source'], 'window_end')
        # cutoff 不经这个参数传递：它是读取层的可见性硬上界
        self.assertNotEqual(view['time'], self.bounds.observation_cutoff)


class MissingQuestionDateTest(unittest.TestCase):
    """question_date 缺失：cutoff=None（不设截止）并显式标记来源，绝不猜。"""

    def test_cutoff_is_none_and_marked(self):
        bounds = resolve_query_time(QUESTION_2018, None)
        self.assertIsNone(bounds.observation_cutoff)
        self.assertEqual(bounds.time_source, TIME_SOURCE_MISSING_QUESTION_DATE)
        self.assertIn('missing_question_date', bounds.reason)
        self.assertEqual(bounds.target_time, CUTOFF_2018)          # target 仍可解析
        # cutoff=None = 既有"读全部已写入"行为
        self.assertTrue(bounds.visible(EVIDENCE_TOLD_IN_2023))
        self.assertTrue(bounds.visible('2099-01-01T00:00:00'))

    def test_resolver_level_matches(self):
        bounds = QueryTimeResolver().resolve_bounds(QUESTION_2018, asked_at=None)
        self.assertIsNone(bounds.observation_cutoff)
        self.assertEqual(bounds.time_source, TIME_SOURCE_MISSING_QUESTION_DATE)


class CallerCutoffPrecedenceTest(unittest.TestCase):
    """调用方给的公开截止（episode 观测窗口末刻）优先，仍绝不取 target_time。"""

    def test_caller_cutoff_wins(self):
        bounds = resolve_query_time(QUESTION_2018, QUESTION_DATE_2023,
                                    observation_cutoff='2023-12-31T23:59:59')
        self.assertEqual(bounds.observation_cutoff, '2023-12-31T23:59:59')
        self.assertEqual(bounds.time_source, TIME_SOURCE_CALLER)

    def test_caller_cutoff_used_when_question_date_missing(self):
        bounds = resolve_query_time(QUESTION_2018, None,
                                    observation_cutoff='2023-12-31T23:59:59')
        self.assertEqual(bounds.observation_cutoff, '2023-12-31T23:59:59')
        self.assertEqual(bounds.time_source, TIME_SOURCE_CALLER)
        self.assertIn('cutoff_from_caller', bounds.reason)

    def test_caller_cannot_override_with_historical_anchor(self):
        """把历史锚当 cutoff 传进来时，审计字段会把它标成 caller（可见的误用）。"""
        bounds = resolve_query_time(QUESTION_2018, QUESTION_DATE_2023,
                                    observation_cutoff=CUTOFF_2018)
        self.assertEqual(bounds.time_source, TIME_SOURCE_CALLER)
        self.assertFalse(bounds.visible(EVIDENCE_TOLD_IN_2023))   # 误用可被检出


class AsOfLegacyCompatTest(unittest.TestCase):
    """TempLAMA 旧口径（target_time == cutoff）只在显式 allow_as_of 下保留。"""

    QUESTION = 'In 2016: The head of government of France is Manuel Valls'

    def test_as_of_requires_explicit_allow(self):
        with self.assertRaises(ValueError):
            resolve_query_time(self.QUESTION, '2016-12-31T00:00:00', time_mode='as_of')
        with self.assertRaises(ValueError):
            QueryTimeResolver().resolve_bounds(self.QUESTION, '2016-12-31T00:00:00',
                                               time_mode='as_of', allow_as_of=False)

    def test_as_of_keeps_target_equals_cutoff(self):
        bounds = resolve_query_time(self.QUESTION, '2016-12-31T00:00:00',
                                    time_mode='as_of', allow_as_of=True)
        self.assertEqual(bounds.target_time, '2016-12-31T23:59:59')
        self.assertEqual(bounds.observation_cutoff, bounds.target_time)
        self.assertEqual(bounds.time_source, TIME_SOURCE_AS_OF_LEGACY)
        self.assertTrue(bounds.allow_as_of)
        self.assertEqual(bounds.time_mode, 'as_of')

    def test_as_of_without_anchor_falls_back_to_question_date(self):
        bounds = resolve_query_time('What is the capital of France?',
                                    '2016-12-31T00:00:00',
                                    time_mode='as_of', allow_as_of=True)
        self.assertIsNone(bounds.target_time)
        # 无锚时 cutoff 退回提问时刻本身（原样保留，不编造区间末刻）
        self.assertEqual(bounds.observation_cutoff, '2016-12-31T00:00:00')
        self.assertIn('as_of_without_target', bounds.reason)

    def test_unknown_time_mode_rejected(self):
        with self.assertRaises(ValueError):
            resolve_query_time(self.QUESTION, None, time_mode='latest')
        self.assertEqual(QUERY_TIME_MODES, ('window_end', 'as_of'))

    def test_default_path_is_not_as_of(self):
        """默认路径必须是 window_end，且 cutoff 与 target 分离。"""
        bounds = resolve_query_time(self.QUESTION, '2016-12-31T00:00:00')
        self.assertEqual(bounds.time_mode, 'window_end')
        self.assertFalse(bounds.allow_as_of)
        self.assertNotEqual(bounds.observation_cutoff, bounds.target_time)


class AuditFieldTest(unittest.TestCase):
    def test_as_dict_exposes_both_sources(self):
        payload = resolve_query_time(QUESTION_2018, QUESTION_DATE_2023).as_dict()
        for key in ('target_time', 'observation_cutoff', 'selection_bound',
                    'time_mode', 'time_source', 'allow_as_of'):
            self.assertIn(key, payload)
        self.assertEqual(payload['time_source'], TIME_SOURCE_QUESTION_DATE)

    def test_public_query_vocabulary_unchanged(self):
        """t13 契约冻结：query_time 的 sidecar 口径不扩 public_query 的词表。"""
        from vcrm import data_views
        self.assertEqual(tuple(data_views.TIME_SOURCES),
                         ('as_of', 'window_end', 'unknown'))
        self.assertNotIn(TIME_SOURCE_MISSING_QUESTION_DATE, data_views.TIME_SOURCES)
        self.assertEqual(tuple(data_views.PUBLIC_QUERY_KEYS),
                         ('question', 'time', 'time_source'))


if __name__ == '__main__':
    unittest.main()
