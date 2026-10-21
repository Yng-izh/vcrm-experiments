"""QueryTimeResolver 单测（t7 P4 交付）：历史时间锚优先，其次提问时刻。

覆盖四类必需场景（last year / 2019 / in March 2022 / 无时间表达）+ 冻结契约的
边界：窗口末刻是区间**末刻**（select_version 的上界，不是起点）、不可唯一折算
的时间词单列 ambiguous、未来年份不猜、无提问时刻不编造年份。

stdlib unittest only；不依赖 torch / GPU / 网络。
"""
from __future__ import annotations

import unittest

from vcrm.query_time import (
    QueryTimeResolver,
    TIME_ANCHOR_ASK_TIME,
    TIME_ANCHOR_QUESTION_TEXT,
    end_of_day,
    end_of_month,
    end_of_year,
    parse_ask_time,
)

ASK = '2024-05-06T09:30:00'


class AskTimeFallbackTest(unittest.TestCase):
    """无时间表达 → 提问时刻即观测窗口末刻（不是"无时间"、也不是历史目标时间）。"""

    def setUp(self):
        self.resolver = QueryTimeResolver()

    def test_no_temporal_expression_uses_ask_time(self):
        result = self.resolver.resolve('What is my dog called?', ASK)
        self.assertEqual(result.window_end, ASK)
        self.assertEqual(result.time_anchor, TIME_ANCHOR_ASK_TIME)
        self.assertFalse(result.anchor_ambiguous)
        self.assertEqual(result.anchor_kind, 'none')
        self.assertEqual(result.reason, 'no_temporal_expression')
        self.assertIsNone(result.anchor_text)

    def test_ask_time_is_not_rounded_or_shifted(self):
        result = self.resolver.resolve('How many siblings do I have?', '2023/12/31 (Sun) 23:45')
        self.assertEqual(result.window_end, '2023-12-31T23:45:00')
        self.assertEqual(result.time_anchor, TIME_ANCHOR_ASK_TIME)

    def test_missing_ask_time_stays_unknown(self):
        result = self.resolver.resolve('What is my dog called?', None)
        self.assertIsNone(result.window_end)          # public_query → time_source='unknown'
        self.assertEqual(result.time_anchor, TIME_ANCHOR_ASK_TIME)
        self.assertEqual(result.reason, 'ask_time_unavailable')

    def test_assume_ask_year_is_opt_in(self):
        resolver = QueryTimeResolver(assume_ask_year=2022)
        result = resolver.resolve('What is my dog called?', None)
        self.assertEqual(result.window_end, '2022-12-31T23:59:59')
        self.assertEqual(result.time_anchor, TIME_ANCHOR_ASK_TIME)


class HistoricalAnchorTest(unittest.TestCase):
    """四类必需场景：last year / 2019 / in March 2022 / 无时间表达。"""

    def setUp(self):
        self.resolver = QueryTimeResolver()

    def test_last_year(self):
        result = self.resolver.resolve('What did I buy last year?', ASK)
        self.assertEqual(result.window_end, '2023-12-31T23:59:59')
        self.assertEqual(result.time_anchor, TIME_ANCHOR_QUESTION_TEXT)
        self.assertFalse(result.anchor_ambiguous)
        self.assertEqual(result.anchor_kind, 'last_year')
        self.assertNotEqual(result.window_end, ASK)

    def test_explicit_year(self):
        result = self.resolver.resolve('Which city did I live in in 2019?', ASK)
        self.assertEqual(result.window_end, '2019-12-31T23:59:59')
        self.assertEqual(result.time_anchor, TIME_ANCHOR_QUESTION_TEXT)
        self.assertEqual(result.anchor_kind, 'year')
        self.assertEqual(result.anchor_text, '2019')

    def test_month_and_year(self):
        result = self.resolver.resolve('Where did I live in March 2022?', ASK)
        self.assertEqual(result.window_end, '2022-03-31T23:59:59')
        self.assertEqual(result.anchor_kind, 'month_year')
        self.assertEqual(result.time_anchor, TIME_ANCHOR_QUESTION_TEXT)

    def test_month_year_wins_over_bare_year(self):
        # "March 2022" 不能被降级成整年末刻 2022-12-31（否则会选到 12 月的版本）
        result = self.resolver.resolve('In March 2022 I adopted a cat, what did I name it?', ASK)
        self.assertEqual(result.window_end, '2022-03-31T23:59:59')

    def test_iso_day_anchor(self):
        result = self.resolver.resolve('What happened on 2019-03-05?', ASK)
        self.assertEqual(result.window_end, '2019-03-05T23:59:59')
        self.assertEqual(result.anchor_kind, 'iso_day')

    def test_years_ago(self):
        result = self.resolver.resolve('I started a new job three years ago', ASK)
        self.assertEqual(result.window_end, '2021-12-31T23:59:59')
        self.assertEqual(result.anchor_kind, 'years_ago')

    def test_this_year(self):
        result = self.resolver.resolve('What did I do this year?', ASK)
        self.assertEqual(result.window_end, '2024-12-31T23:59:59')
        self.assertEqual(result.anchor_kind, 'this_year')

    def test_anchor_never_falls_back_to_ask_time(self):
        """硬保证：只要文本里有可解析锚，window_end 必须来自锚而不是提问时刻。"""
        for question in ('last year?', 'in 2019?', 'in March 2022?', 'three years ago?',
                         'on 2019-03-05?', 'this year?'):
            result = self.resolver.resolve(question, ASK)
            self.assertEqual(result.time_anchor, TIME_ANCHOR_QUESTION_TEXT, question)
            self.assertNotEqual(result.window_end, ASK, question)


class WindowEndIsIntervalEndTest(unittest.TestCase):
    """窗口末刻 = 区间末刻（select_version 的上界），不是区间起点/等值匹配。"""

    def test_end_of_year_and_month_helpers(self):
        self.assertEqual(end_of_year(2019), '2019-12-31T23:59:59')
        self.assertEqual(end_of_month(2022, 3), '2022-03-31T23:59:59')
        self.assertEqual(end_of_month(2020, 2), '2020-02-29T23:59:59')  # 闰年
        self.assertEqual(end_of_month(2019, 2), '2019-02-28T23:59:59')
        self.assertEqual(end_of_day(2019, 3, 5), '2019-03-05T23:59:59')

    def test_same_year_anchor_points_at_december_not_january(self):
        result = QueryTimeResolver().resolve('What was my job in 2019?', ASK)
        self.assertEqual(result.window_end, '2019-12-31T23:59:59')
        self.assertGreater(result.window_end, '2019-03-01T00:00:00')
        self.assertGreater(result.window_end, '2019-11-01T00:00:00')
        self.assertLess(result.window_end, '2020-01-01T00:00:00')


class AmbiguityTest(unittest.TestCase):
    """不可唯一折算的时间词 → 仍用 ask_time，但标 ambiguous 并单列统计。"""

    def setUp(self):
        self.resolver = QueryTimeResolver()

    def test_relative_word_is_ambiguous_but_uses_ask_time(self):
        result = self.resolver.resolve('I recently moved, where do I live?', ASK)
        self.assertEqual(result.window_end, ASK)
        self.assertEqual(result.time_anchor, TIME_ANCHOR_ASK_TIME)
        self.assertTrue(result.anchor_ambiguous)
        self.assertEqual(result.anchor_kind, 'relative_ambiguous')
        self.assertEqual(result.reason, 'anchor_not_unique:recently')
        self.assertEqual(result.anchor_text, 'recently')

    def test_vague_duration_is_not_fabricated(self):
        for question in ('a while ago I moved', 'a few years ago I moved',
                         'a couple of years ago I moved', 'lately I feel tired'):
            result = self.resolver.resolve(question, ASK)
            self.assertTrue(result.anchor_ambiguous, question)
            self.assertEqual(result.window_end, ASK, question)
            self.assertEqual(result.time_anchor, TIME_ANCHOR_ASK_TIME, question)

    def test_multiple_anchors_use_latest_but_flag_ambiguous(self):
        result = self.resolver.resolve(
            'In 2019 I lived in Paris, and in 2022 I moved to Rome. Where do I live?', ASK)
        self.assertEqual(result.window_end, '2022-12-31T23:59:59')
        self.assertTrue(result.anchor_ambiguous)
        self.assertEqual(result.reason, 'multiple_anchors')
        # 仍是锚（不是 ask_time）：硬保证优先
        self.assertEqual(result.time_anchor, TIME_ANCHOR_QUESTION_TEXT)

    def test_future_anchor_is_not_trusted(self):
        result = self.resolver.resolve('In 2030 I will move to Rome', '2020-01-01T00:00:00')
        self.assertEqual(result.window_end, '2020-01-01T00:00:00')
        self.assertTrue(result.anchor_ambiguous)
        self.assertEqual(result.reason, 'anchor_after_ask_time')

    def test_relative_anchor_without_ask_time_does_not_guess(self):
        result = self.resolver.resolve('What did I do last year?', None)
        self.assertIsNone(result.window_end)
        self.assertTrue(result.anchor_ambiguous)
        self.assertEqual(result.reason, 'anchor_needs_ask_time')


class ParseAskTimeTest(unittest.TestCase):
    def test_formats(self):
        self.assertEqual(parse_ask_time('2024-05-06T09:30:00').isoformat(), ASK)
        self.assertEqual(parse_ask_time('2024/05/06 (Mon) 09:30').isoformat(), ASK)
        self.assertEqual(parse_ask_time('2024-05-06').isoformat(), '2024-05-06T00:00:00')
        self.assertEqual(parse_ask_time('2024').isoformat(), '2024-12-31T23:59:59')

    def test_unparseable_returns_none(self):
        self.assertIsNone(parse_ask_time(None))
        self.assertIsNone(parse_ask_time(''))
        self.assertIsNone(parse_ask_time('not a date'))

    def test_resolve_rejects_empty_question(self):
        with self.assertRaises(ValueError):
            QueryTimeResolver().resolve('   ', ASK)


class PublicQueryContractTest(unittest.TestCase):
    """window_end 直接喂 public_query：as_of 通道不参与，unknown 不回退。"""

    def test_window_end_flows_into_public_query(self):
        from vcrm.data_views import public_query
        resolved = QueryTimeResolver().resolve('What did I do in 2019?', ASK)
        view = public_query({'question': 'q', 'time': ASK, 'as_of': ASK},
                            time_mode='window_end', window_end=resolved.window_end)
        self.assertEqual(view, {'question': 'q', 'time': resolved.window_end,
                                'time_source': 'window_end'})
        self.assertNotEqual(view['time'], ASK)   # 提问时刻没有被当成历史目标时间

    def test_none_window_end_is_unknown_not_question_date(self):
        from vcrm.data_views import public_query
        resolved = QueryTimeResolver().resolve('What is my dog called?', None)
        view = public_query({'question': 'q', 'time': ASK}, time_mode='window_end',
                            window_end=resolved.window_end)
        self.assertIsNone(view['time'])
        self.assertEqual(view['time_source'], 'unknown')


if __name__ == '__main__':
    unittest.main()
