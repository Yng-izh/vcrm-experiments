"""QueryTimeResolver：把问题文本里的历史时间锚折算成「观测窗口末刻」。

为什么需要单独一层（t13 冻结契约 + t3 泄漏审计 + t15 时间戳口径）：

* ``vcrm.data_views.public_query`` 只负责**关掉 oracle 时间通道**：Predicted 口径
  ``time_mode='window_end'`` 下它绝不读 ``query['time']`` / ``query['as_of']``，
  只接受调用方显式传入的 ``window_end``。
* ``window_end`` 从哪来由本模块决定，**不是** ``question_date`` 的直接透传：
  1. 问题文本里有可唯一折算的历史时间锚（"in 2019" / "in March 2022" /
     "last year" / "three years ago" / "2019-03-05"）→ 必须用锚折算出的窗口
     末刻。这是**硬保证**：这条路径上不存在落到 ``ask_time`` 的分支。
  2. 没有时间表达 → 用提问时刻（ask_time）作为观测窗口末刻。
  3. 有时间词但无法唯一折算（"recently" / "a while ago" / "a few years ago"）
     → 仍用 ask_time，但 ``anchor_ambiguous=True``，单独统计，不混入正常口径。
* 窗口末刻 = 被指代区间的**末刻**，它是 ``store.select_version`` 的上界
  （``entry['time'] <= window_end``），不是等值匹配：问 "what was X in 2019"
  取的是 ``2019-12-31T23:59:59``，所以同一年内 2019-03 与 2019-11 两个版本
  会选到 11 月那个，而不是 3 月那个（边界用例见
  ``tests/test_lme_latent.py::WindowEndUpperBoundTest``）。
* 禁止把 question_date 当历史目标时间：``ask_time`` 只作为观测窗口末刻的
  fallback，永远不被当作「目标版本时间戳」使用（那会把时间维度免费解掉）。

与 ``vcrm.data_views`` 的分工：

* ``time_source``（'window_end' / 'unknown'）由 ``public_query`` 产出，进
  ``PUBLIC_QUERY_KEYS``；
* 本模块的 ``time_anchor``（'question_text_anchor' / 'ask_time'）与
  ``anchor_ambiguous`` / ``anchor_text`` 是 **sidecar 子口径**，只写进我们自己的
  jsonl / summary.json，不进入 ``PUBLIC_QUERY_KEYS`` / ``TIME_SOURCES``
  （t13 契约冻结：扩词表会让 t6 的 ``--time-mode`` 语义跟着漂）。
"""
from __future__ import annotations

import calendar
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: 历史时间锚来自问题文本本身。
TIME_ANCHOR_QUESTION_TEXT = 'question_text_anchor'
#: 没有可用锚，退回提问时刻（观测窗口末刻）。
TIME_ANCHOR_ASK_TIME = 'ask_time'
#: sidecar 子口径词表（不进 PUBLIC_QUERY_KEYS）。
TIME_ANCHORS: Sequence[str] = (TIME_ANCHOR_QUESTION_TEXT, TIME_ANCHOR_ASK_TIME)

_MONTHS: Dict[str, int] = {
    'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
    'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12,
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'jun': 6, 'jul': 7, 'aug': 8,
    'sep': 9, 'sept': 9, 'oct': 10, 'nov': 11, 'dec': 12,
}
_MONTH_RE = '|'.join(sorted(_MONTHS, key=len, reverse=True))

_WORD_NUMBERS: Dict[str, int] = {
    'a': 1, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6,
    'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'a couple of': 2, 'a few': 3,
    'several': 4,
}

_ISO_DAY = re.compile(r'\b(?P<year>(?:19|20)\d{2})-(?P<month>\d{2})-(?P<day>\d{2})\b')
_MONTH_YEAR = re.compile(
    r'\b(?:(?:in|during|back in|as of|since|around|by|of)\s+)?'
    rf'(?P<month>{_MONTH_RE})\.?\s*,?\s+(?P<year>(?:19|20)\d{{2}})\b', re.I)
_YEAR = re.compile(r'\b(?P<year>(?:19|20)\d{2})\b')
_LAST_YEAR = re.compile(r'\b(?:last|previous|the previous|the past)\s+year\b', re.I)
# 只认明确数字/数词（"three years ago" / "a year ago"）；"a few years ago" /
# "a couple of years ago" / "several years ago" 属于不可唯一折算，留给 _AMBIGUOUS，
# 不在这里编造一个具体年份。
_YEARS_AGO = re.compile(
    r'\b(?P<n>\d{1,2}|a|one|two|three|four|five|six|seven|eight|nine|ten)'
    r'\s+years?\s+ago\b', re.I)
_THIS_YEAR = re.compile(r'\bthis\s+year\b', re.I)
_AMBIGUOUS = re.compile(
    r'\b(recently|lately|these\s+days|a\s+while\s+ago|a\s+while\s+back|some\s+time\s+ago|'
    r'sometime|not\s+long\s+ago|the\s+other\s+day|earlier(?:\s+on)?|a\s+few\s+years\s+ago|'
    r'a\s+couple\s+of\s+years\s+ago|several\s+years\s+ago|in\s+the\s+past|previously|'
    r'used\s+to|at\s+the\s+time)\b', re.I)

_ASK_FORMATS = ('%Y/%m/%d (%a) %H:%M', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S',
                '%Y/%m/%d', '%Y-%m-%d')


def _iso(value: datetime) -> str:
    return value.strftime('%Y-%m-%dT%H:%M:%S')


def end_of_year(year: int) -> str:
    """该年最后一个可观测时刻（窗口末刻，不是区间起点）。"""
    return f'{year:04d}-12-31T23:59:59'


def end_of_month(year: int, month: int) -> str:
    last = calendar.monthrange(year, month)[1]
    return f'{year:04d}-{month:02d}-{last:02d}T23:59:59'


def end_of_day(year: int, month: int, day: int) -> str:
    return f'{year:04d}-{month:02d}-{day:02d}T23:59:59'


def parse_ask_time(value: Any) -> Optional[datetime]:
    """把提问时刻（question_date / ISO / 'YYYY'）解析成 naive datetime；失败返回 None。"""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if re.fullmatch(r'\d{4}', text):
        return datetime(int(text), 12, 31, 23, 59, 59)
    try:
        parsed = datetime.fromisoformat(text.replace('Z', '+00:00'))
        return parsed.replace(tzinfo=None)
    except ValueError:
        pass
    for pattern in _ASK_FORMATS:
        try:
            return datetime.strptime(text, pattern)
        except ValueError:
            pass
    return None


def _overlaps(span: Tuple[int, int], others: Sequence[Tuple[int, int]]) -> bool:
    start, end = span
    return any(start < other_end and other_start < end for other_start, other_end in others)


def _absolute_anchors(text: str) -> List[Dict[str, Any]]:
    """绝对时间锚（不依赖提问时刻）：ISO 日期 > 月+年 > 单独年份。"""
    anchors: List[Dict[str, Any]] = []
    claimed: List[Tuple[int, int]] = []
    for match in _ISO_DAY.finditer(text):
        try:
            value = datetime(int(match['year']), int(match['month']), int(match['day']))
        except ValueError:
            continue
        anchors.append({'kind': 'iso_day', 'text': match.group(0),
                        'window_end': end_of_day(value.year, value.month, value.day),
                        'year': value.year})
        claimed.append(match.span())
    for match in _MONTH_YEAR.finditer(text):
        if _overlaps(match.span(), claimed):
            continue
        month = _MONTHS[match['month'].lower().rstrip('.')]
        year = int(match['year'])
        anchors.append({'kind': 'month_year', 'text': match.group(0),
                        'window_end': end_of_month(year, month), 'year': year})
        claimed.append(match.span())
    for match in _YEAR.finditer(text):
        if _overlaps(match.span(), claimed):
            continue
        year = int(match['year'])
        anchors.append({'kind': 'year', 'text': match.group(0),
                        'window_end': end_of_year(year), 'year': year})
        claimed.append(match.span())
    return anchors


def _relative_anchors(text: str, ask: Optional[datetime]) -> List[Dict[str, Any]]:
    """相对时间锚：需要提问时刻才能折算；ask 未知时 window_end=None。"""
    anchors: List[Dict[str, Any]] = []
    for match in _LAST_YEAR.finditer(text):
        year = None if ask is None else ask.year - 1
        anchors.append({'kind': 'last_year', 'text': match.group(0),
                        'window_end': None if year is None else end_of_year(year),
                        'year': year})
    for match in _YEARS_AGO.finditer(text):
        raw = re.sub(r'\s+', ' ', match['n'].strip().lower())
        count = _WORD_NUMBERS.get(raw)
        if count is None:
            try:
                count = int(raw)
            except ValueError:
                continue
        year = None if ask is None else ask.year - count
        anchors.append({'kind': 'years_ago', 'text': match.group(0),
                        'window_end': None if year is None else end_of_year(year),
                        'year': year})
    for match in _THIS_YEAR.finditer(text):
        year = None if ask is None else ask.year
        anchors.append({'kind': 'this_year', 'text': match.group(0),
                        'window_end': None if year is None else end_of_year(year),
                        'year': year})
    return anchors


@dataclass(frozen=True)
class TimeResolution:
    """一次查询时间解析的结果（``window_end`` 直接喂给 ``public_query``）。"""

    window_end: Optional[str]
    time_anchor: str
    anchor_ambiguous: bool = False
    anchor_text: Optional[str] = None
    anchor_kind: str = 'none'
    reason: str = ''
    ask_time: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {'window_end': self.window_end, 'time_anchor': self.time_anchor,
                'anchor_ambiguous': self.anchor_ambiguous, 'anchor_text': self.anchor_text,
                'anchor_kind': self.anchor_kind, 'reason': self.reason,
                'ask_time': self.ask_time}


class QueryTimeResolver:
    """确定性的查询时间解析器：问题文本历史锚优先，其次提问时刻。

    ``resolve(question, asked_at)`` 返回 :class:`TimeResolution`；
    ``window_end`` 为 None 表示连提问时刻都不可用（评测侧记 time_source='unknown'，
    ``select_version(state, None)`` 即"无时间约束 = 最新可见版本"）。
    """

    def __init__(self, *, assume_ask_year: Optional[int] = None) -> None:
        if assume_ask_year is not None and (not isinstance(assume_ask_year, int)
                                            or isinstance(assume_ask_year, bool)):
            raise ValueError('assume_ask_year must be an int or None')
        self.assume_ask_year = assume_ask_year

    def resolve(self, question: str, asked_at: Any = None) -> TimeResolution:
        if not isinstance(question, str) or not question.strip():
            raise ValueError('QueryTimeResolver.resolve expects a non-empty question str')
        ask = parse_ask_time(asked_at)
        if ask is None and self.assume_ask_year is not None:
            ask = datetime(self.assume_ask_year, 12, 31, 23, 59, 59)
        ask_iso = None if ask is None else _iso(ask)

        absolute = _absolute_anchors(question)
        relative = _relative_anchors(question, ask)
        ambiguous_word = _AMBIGUOUS.search(question)

        # 1) 绝对锚优先（硬保证：有锚就绝不落到 ask_time）
        if absolute:
            ends = sorted({anchor['window_end'] for anchor in absolute})
            chosen = max(absolute, key=lambda anchor: (anchor['window_end'], anchor['kind']))
            ambiguous = len(ends) > 1 or bool(relative)
            if ask is not None and chosen['year'] is not None and chosen['year'] > ask.year:
                # 未来年份不是"历史目标时间"：不猜，退回观测窗口末刻并标 ambiguous。
                return TimeResolution(None if ask_iso is None else ask_iso,
                                      TIME_ANCHOR_ASK_TIME, True, chosen['text'],
                                      chosen['kind'], 'anchor_after_ask_time', ask_iso)
            reason = 'multiple_anchors' if ambiguous else f"anchor:{chosen['kind']}"
            return TimeResolution(ends[-1], TIME_ANCHOR_QUESTION_TEXT, ambiguous,
                                  chosen['text'], chosen['kind'], reason, ask_iso)

        # 2) 相对锚（last year / N years ago / this year），同样不落到 ask_time 的等值语义
        resolvable = [anchor for anchor in relative if anchor['window_end']]
        if resolvable:
            ends = sorted({anchor['window_end'] for anchor in resolvable})
            chosen = max(resolvable, key=lambda anchor: anchor['window_end'])
            ambiguous = len(ends) > 1
            if ask is not None and chosen['year'] is not None and chosen['year'] > ask.year:
                return TimeResolution(ask_iso, TIME_ANCHOR_ASK_TIME, True, chosen['text'],
                                      chosen['kind'], 'anchor_after_ask_time', ask_iso)
            reason = 'multiple_anchors' if ambiguous else f"anchor:{chosen['kind']}"
            return TimeResolution(ends[-1], TIME_ANCHOR_QUESTION_TEXT, ambiguous,
                                  chosen['text'], chosen['kind'], reason, ask_iso)
        if relative:
            # 有相对锚但拿不到提问时刻 → 折算不出来，显式 ambiguous，不猜年份。
            return TimeResolution(None, TIME_ANCHOR_ASK_TIME, True, relative[0]['text'],
                                  relative[0]['kind'], 'anchor_needs_ask_time', ask_iso)

        # 3) 有时间词但无法唯一折算 → 仍用 ask_time，但单列 ambiguous 桶
        if ambiguous_word:
            word = ambiguous_word.group(0)
            return TimeResolution(ask_iso, TIME_ANCHOR_ASK_TIME, True, word,
                                  'relative_ambiguous',
                                  f'anchor_not_unique:{word.lower()}', ask_iso)

        # 4) 无时间表达 → 提问时刻即观测窗口末刻
        if ask is None:
            return TimeResolution(None, TIME_ANCHOR_ASK_TIME, False, None, 'none',
                                  'ask_time_unavailable', None)
        return TimeResolution(ask_iso, TIME_ANCHOR_ASK_TIME, False, None, 'none',
                              'no_temporal_expression', ask_iso)


__all__ = [
    'QueryTimeResolver',
    'TIME_ANCHORS',
    'TIME_ANCHOR_ASK_TIME',
    'TIME_ANCHOR_QUESTION_TEXT',
    'TimeResolution',
    'end_of_day',
    'end_of_month',
    'end_of_year',
    'parse_ask_time',
]
