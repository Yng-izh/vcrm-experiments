"""从 TempLAMA 事件版本链构造"跨版本诊断题"（受控诊断，不是官方分数）。

为什么必须自己造题
------------------
TempLAMA 官方查询全部是同一个模板的单版本填空（``<value> is <pred> of <entity>``），
时间信息只出现在 ``as_of``（金标版本时间戳，泄漏）或观测窗口末刻（同一 episode
内全部塌到同一年）。**它们的答案永远只依赖一个版本**，因此结构上无法区分
"读 2 个版本"和"读全部版本"——用官方题做实验一，A/B 两臂必然打平。

本模块从事件的版本链出发，构造四类必须依赖版本关系的题（全部为英文，与
TempLAMA 事件文本同语言、同模板族）：

=============== ============================================ ==============
family          题面（模板）                                  金标答案
=============== ============================================ ==============
recall_year     ``In {year}: {cloze}``                        该年版本的值
ordinal_back    ``Two changes before {year}, ... {cloze}``   倒数第 3 个版本的值
count_values    ``How many different values ... between a b`` 区间内不同值的个数
compare_years   ``In {a} and in {b}, was the value the same``  yes / no
=============== ============================================ ==============

``cloze`` 就是金标 ``key_text``（形如 ``_X_ is the head coach of VfL Wolfsburg.``），
即 reader 在 TempLAMA 上已经见过的填空模板。

四类的分工（这是实验一的读法）
------------------------------
* ``recall_year``：题面自带年份，A 臂（pair）用题面年份就能选对版本 → **A 不吃亏**。
  它测的是"多塞几个版本会不会干扰"，是"trajectory 会不会更差"的探针。
* ``ordinal_back`` / ``count_values`` / ``compare_years``：结构上必须同时看到
  多个版本（序关系 / 全部版本 / 两个版本）。A 臂只有 current + 1 个历史版本，
  再公平也无法完成 → 测的是"trajectory 到底能不能用起来"。

口径声明：这些是受控诊断题，**不是 TempLAMA 官方分数**，不能用来声称开放域泛化。
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

FAMILIES: Tuple[str, ...] = ('recall_year', 'ordinal_back', 'count_values', 'compare_years')
VALUE_FAMILIES: Tuple[str, ...] = ('recall_year', 'ordinal_back')

_YEAR_PREFIX = re.compile(r'^In\s+(\d{4})\s*:\s*')
_VALUE_SPLIT = re.compile(r'\s*/\s*')
_WORD_NUMBERS = {
    'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6,
    'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12,
}
_DIGITS = re.compile(r'\b(\d{1,3})\b')
_POLARITY = re.compile(r'\b(yes|no|true|false|same|different)\b', re.I)
_PUNCT = re.compile(r'[^a-z0-9 ]+')


# ---------------------------------------------------------------------------
# 事件 → (年份, 值)
# ---------------------------------------------------------------------------
def split_template(key_text: str) -> Optional[Tuple[str, str]]:
    """把 ``_X_ is the head coach of VfL Wolfsburg.`` 拆成 (_X_ 前缀, 后缀)。

    事件文本 = ``In {year}: `` + 前缀 + 值 + 后缀。因此只要事件文本同时以
    ``前缀`` 开头、以 ``后缀`` 结尾，中间那段就是值 —— 与 ``_X_`` 出现在句首还是
    句末无关（``V. V. S. Laxman plays for _X_.`` 也能正确取值）。
    """
    text = str(key_text or '')
    if '_X_' not in text:
        return None
    pre, _, post = text.partition('_X_')
    return pre, post


def value_of(event: Mapping[str, Any], template: Tuple[str, str]) -> Optional[str]:
    """事件文本 → 该版本的值（多值时保留 ``A / B`` 原样）。"""
    pre, post = template
    body = str(event.get('text') or '')
    match = _YEAR_PREFIX.match(body)
    if match:
        body = body[match.end():]
    if not body.startswith(pre):
        return None
    if post and not body.endswith(post):
        return None
    value = body[len(pre):len(body) - len(post)] if post else body[len(pre):]
    value = value.strip().rstrip('.').strip()
    return value or None


def value_set(value: str) -> Tuple[str, ...]:
    """``A / B`` → ('A', 'B')；用于集合比较与"不同值个数"。"""
    parts = [part.strip() for part in _VALUE_SPLIT.split(str(value or ''))]
    return tuple(sorted({part for part in parts if part}))


def norm_set(values: Sequence[str]) -> frozenset:
    """规范化后的取值集合 —— **所有比较都必须用它**。

    同一实体在句中与句末的写法不同（``Southport F.C.`` vs 句末被剥掉句点的
    ``Southport F.C``），直接比较原始字符串会漏掉重合，让"读成相邻版本也算对"
    的松判混进来。判分侧 ``matches_value`` 本来就先规范化，这里保持一致。
    """
    return frozenset(normalize(value) for value in values if normalize(value))


def timeline_of(episode: Mapping[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """episode → 该状态的版本链（时间升序）。

    返回 ``None`` 表示这条 episode 不能用来造题（事件缺 ``key_text`` / 时间不是
    可解析的年份 / 取值失败）。任何一条失败就整条丢弃，绝不猜。
    """
    events = list(episode.get('events') or [])
    if not events:
        return None
    keys = {str(event.get('key_text') or '') for event in events}
    if len(keys) != 1:
        return None
    template = split_template(next(iter(keys)))
    if template is None:
        return None
    timeline: List[Dict[str, Any]] = []
    for event in events:
        time = str(event.get('time') or '')
        match = re.match(r'^(\d{4})', time)
        if not match:
            return None
        value = value_of(event, template)
        if value is None:
            return None
        values = value_set(value)
        timeline.append({'time': time, 'year': int(match.group(1)),
                         'value': value, 'values': values,
                         'norm': norm_set(values), 'event_id': event.get('id')})
    timeline.sort(key=lambda entry: (entry['time'], str(entry['event_id'])))
    if len({entry['year'] for entry in timeline}) != len(timeline):
        return None                       # 同一年多个事件：年份→版本不是函数
    return timeline


def key_text_of(episode: Mapping[str, Any]) -> str:
    return str((episode.get('events') or [{}])[0].get('key_text') or '')


def inline_record(key_text: str) -> str:
    """题面里引用记录模板时的写法：``_X_`` → ``the value``（避免题面里出现占位符）。"""
    text = str(key_text or '').replace('_X_', 'the value').strip()
    return re.sub(r'\bthe\s+the\b', 'the', text)


# ---------------------------------------------------------------------------
# 四类题的构造
# ---------------------------------------------------------------------------
def _base(episode: Mapping[str, Any], family: str, timeline: Sequence[Mapping[str, Any]],
          key_text: str, question: str, answers: Sequence[str], gold_years: Sequence[int],
          required_times: Sequence[str], pair_target_time: Optional[str]) -> Dict[str, Any]:
    return {
        'id': f"{episode['id']}::{family}",
        'family': family,
        'episode_id': episode['id'],
        'state_key': str((episode.get('events') or [{}])[0].get('state_key') or ''),
        'question': question,
        'answers': [answer for answer in answers if answer],
        'gold_years': list(gold_years),
        'required_version_times': list(required_times),
        'pair_target_time': pair_target_time,
        'n_versions': len(timeline),
        'version_times': [entry['time'] for entry in timeline],
        'version_values': [entry['value'] for entry in timeline],
        'newest_values': list(timeline[-1]['values']) if timeline else [],
        'key_text': key_text,
    }


def _recall_year(episode, timeline, key_text, wanted_index) -> Optional[Dict[str, Any]]:
    """某一年的值（题面自带年份 → pair 用题面年份就能选对版本）。"""
    if len(timeline) < 2:
        return None
    entry = timeline[wanted_index]
    question = f"In {entry['year']}: {key_text}"
    return _base(episode, 'recall_year', timeline, key_text, question,
                 list(entry['values']), [entry['year']], [entry['time']],
                 f"{entry['year']:04d}-12-31T23:59:59")


def _ordinal_back(episode, timeline, key_text, steps: int = 2) -> Optional[Dict[str, Any]]:
    """倒数第 steps+1 个版本的值（"两次变更之前"），需要版本序关系。"""
    if len(timeline) < steps + 2:
        return None
    gold_index = len(timeline) - 1 - steps
    gold = timeline[gold_index]
    # 金标版本必须与更新的两个版本取值**完全不相交**：
    # ① "两次变更之前"这句话本身要求中间真的换了值；
    # ② 只要有一个名字重复出现，reader 读成相邻版本也会被判对（松判），
    #    这条题就不再区分"读对序关系"和"读错版本"。
    newer = [entry['norm'] for entry in timeline[gold_index + 1:]]
    if any(newer_values & gold['norm'] for newer_values in newer):
        return None
    anchor = timeline[-1]
    record = inline_record(key_text)
    question = (f"Two changes before {anchor['year']}, which value filled the blank _X_ "
                f"in the record '{record}'? Answer with the value.")
    return _base(episode, 'ordinal_back', timeline, key_text, question,
                 list(gold['values']), [gold['year']], [gold['time']],
                 f"{anchor['year']:04d}-12-31T23:59:59")


def _count_values(episode, timeline, key_text) -> Optional[Dict[str, Any]]:
    """区间内一共出现过多少个不同的值（需要全部版本）。"""
    if len(timeline) < 3:
        return None
    seen = set()
    for entry in timeline:
        seen |= entry['norm']
    if len(seen) < 3:
        return None
    first, last = timeline[0], timeline[-1]
    record = inline_record(key_text)
    question = (f"Between {first['year']} and {last['year']}, how many different values "
                f"appear in the record '{record}'? Answer with a single number.")
    return _base(episode, 'count_values', timeline, key_text, question,
                 [str(len(seen))], [first['year'], last['year']],
                 [entry['time'] for entry in timeline],
                 f"{first['year']:04d}-12-31T23:59:59")


def _compare_years(episode, timeline, key_text, want_same: bool) -> Optional[Dict[str, Any]]:
    """两个年份的值是否相同（需要同时看到两个版本）。"""
    if len(timeline) < 3:
        return None
    record = inline_record(key_text)
    best = None
    for left in range(len(timeline) - 2):
        for right in range(left + 1, len(timeline) - 1):
            same = timeline[left]['norm'] == timeline[right]['norm']
            if same != bool(want_same):
                continue
            # 优先"中间还夹着别的版本"的一对：pair 更不可能同时装下。
            gap = right - left
            if best is None or gap > best[2]:
                best = (left, right, gap)
    if best is None:
        return None
    left, right, _ = best
    a, b = timeline[left], timeline[right]
    question = (f"In {a['year']} and in {b['year']}, was the value in the record "
                f"'{record}' the same? Answer yes or no.")
    return _base(episode, 'compare_years', timeline, key_text, question,
                 ['yes' if a['norm'] == b['norm'] else 'no'],
                 [a['year'], b['year']], [a['time'], b['time']],
                 f"{a['year']:04d}-12-31T23:59:59")


def candidates_for(episode: Mapping[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """一条 episode 能支撑的全部候选题（按 family 分组）。

    同一状态在同一 family 下最多出一题，保证"每条题对应一个独立状态"，
    配对统计不引入同状态内相关性。
    """
    timeline = timeline_of(episode)
    if timeline is None or not (3 <= len(timeline) <= 8):
        return {}
    key_text = key_text_of(episode)
    out: Dict[str, List[Dict[str, Any]]] = {family: [] for family in FAMILIES}

    # recall_year：金标必须是历史版本，且该取值在**其它任何版本里都不出现** ——
    # ① 与 current 不相交，"答成 current" 才能被识别为错误；
    # ② 与相邻年份也不相交，否则 reader 读成隔壁年份也会被判对（松判），
    #    这条题就测不出"能不能定位到那一年"。取最早满足条件的年份。
    newest_values = timeline[-1]['norm']
    for index in range(len(timeline) - 1):
        others = set()
        for other_index, other in enumerate(timeline):
            if other_index != index:
                others |= other['norm']
        if timeline[index]['norm'] & (newest_values | others):
            continue
        candidate = _recall_year(episode, timeline, key_text, index)
        if candidate:
            out['recall_year'].append(candidate)
        break

    for builder, family in ((_ordinal_back, 'ordinal_back'),
                            (_count_values, 'count_values')):
        candidate = builder(episode, timeline, key_text)
        if candidate:
            out[family].append(candidate)

    for want_same in (False, True):
        candidate = _compare_years(episode, timeline, key_text, want_same)
        if candidate:
            out['compare_years'].append(candidate)
    return out


def build_questions(episodes: Sequence[Mapping[str, Any]], limit: int,
                    per_family: Optional[int] = None) -> List[Dict[str, Any]]:
    """轮转取题：四个 family 轮流取，每个状态只贡献一道题。

    ``per_family`` 缺省时按 ``limit // len(FAMILIES)`` 分配（80 → 每族 20）。
    compare_years 的 yes/no 在两支候选里交替取，避免答案全落在 'no' 上。
    """
    quota = per_family if per_family else max(1, limit // len(FAMILIES))
    pools = {family: [] for family in FAMILIES}
    for episode in episodes:
        for family, items in candidates_for(episode).items():
            pools[family].extend(items)
    for family in ('compare_years',):
        pools[family] = _interleave_labels(pools[family])

    used_states: set = set()
    chosen: List[Dict[str, Any]] = []
    cursors = {family: 0 for family in FAMILIES}
    counts = {family: 0 for family in FAMILIES}
    while len(chosen) < limit:
        progressed = False
        for family in FAMILIES:
            if len(chosen) >= limit or counts[family] >= quota:
                continue
            pool = pools[family]
            while cursors[family] < len(pool):
                candidate = pool[cursors[family]]
                cursors[family] += 1
                if candidate['state_key'] in used_states:
                    continue
                used_states.add(candidate['state_key'])
                counts[family] += 1
                chosen.append(candidate)
                progressed = True
                break
        if not progressed:
            break
    return chosen


def _interleave_labels(items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    yes = [item for item in items if item['answers'] == ['yes']]
    no = [item for item in items if item['answers'] == ['no']]
    merged: List[Dict[str, Any]] = []
    for index in range(max(len(yes), len(no))):
        for bucket in (no, yes):
            if index < len(bucket):
                merged.append(bucket[index])
    return merged


# ---------------------------------------------------------------------------
# 打分（与 family 匹配的判分规则；两臂用同一套规则）
# ---------------------------------------------------------------------------
def normalize(text: Any) -> str:
    return ' '.join(_PUNCT.sub(' ', str(text or '').lower()).split())


def matches_value(answer: Any, value: Any) -> bool:
    """值匹配：规范化后完全相等，或金标值作为子串出现在回答里。"""
    got, want = normalize(answer), normalize(value)
    if not want:
        return False
    return got == want or f' {want} ' in f' {got} '


def extract_number(text: Any) -> Optional[int]:
    got = normalize(text)
    match = _DIGITS.search(got)
    if match:
        return int(match.group(1))
    for token in got.split():
        if token in _WORD_NUMBERS:
            return _WORD_NUMBERS[token]
    return None


def extract_polarity(text: Any) -> Optional[str]:
    match = _POLARITY.search(str(text or ''))
    if not match:
        return None
    token = match.group(1).lower()
    if token in ('yes', 'true', 'same'):
        return 'yes'
    return 'no'


def score_answer(family: str, answer: Any, question: Mapping[str, Any]) -> Dict[str, Any]:
    """family 对应的判分：返回 ``{'correct', 'rule', 'strict_em'}``。

    规则在 A/B 两臂之间完全一致；唯一按 family 变化的只是"什么算对"。
    """
    answers = list(question.get('answers') or [])
    text = str(answer or '')
    if family in VALUE_FAMILIES:
        strict = int(any(normalize(text) == normalize(gold) for gold in answers))
        return {'correct': int(any(matches_value(text, gold) for gold in answers)),
                'rule': 'value_match', 'strict_em': strict}
    if family == 'count_values':
        gold = extract_number(answers[0]) if answers else None
        got = extract_number(text)
        return {'correct': int(gold is not None and got == gold),
                'rule': 'number', 'strict_em': int(normalize(text) == normalize(answers[0]))}
    if family == 'compare_years':
        gold = extract_polarity(answers[0]) if answers else None
        got = extract_polarity(text)
        return {'correct': int(gold is not None and got == gold),
                'rule': 'polarity', 'strict_em': int(normalize(text) == normalize(answers[0]))}
    raise ValueError(f'unknown family {family!r}')


def answered_with_current(answer: Any, question: Mapping[str, Any]) -> bool:
    """"把历史答案答成了 current 版本的值"（只对取值型 family 有意义）。"""
    if question.get('family') not in VALUE_FAMILIES:
        return False
    newest = list(question.get('newest_values') or [])
    gold = list(question.get('answers') or [])
    if not newest or norm_set(newest) == norm_set(gold):
        return False
    if any(matches_value(answer, value) for value in gold):
        return False
    return any(matches_value(answer, value) for value in newest)


__all__ = [
    'FAMILIES',
    'VALUE_FAMILIES',
    'answered_with_current',
    'build_questions',
    'candidates_for',
    'extract_number',
    'extract_polarity',
    'inline_record',
    'matches_value',
    'norm_set',
    'normalize',
    'score_answer',
    'split_template',
    'timeline_of',
    'value_of',
    'value_set',
]
