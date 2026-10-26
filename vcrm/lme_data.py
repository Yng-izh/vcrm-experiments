from __future__ import annotations
"""Shared LongMemEval-S loader: ONE official data view for every pipeline.

官方 cleaned schema（LongMemEval-S / longmemeval_s_cleaned.json）里，
``haystack_sessions`` 是 **turn 列表的列表**（每个 session 是一串
``{"role", "content"}``），并与并行的 ``haystack_session_ids`` / ``haystack_dates``
两个数组按下标一一对应。任何把 session 当 dict 处理、或把三个数组直接
``zip()`` 截断的写法都会读错数据 —— 本模块是唯一的解析入口：

* ``parse_item(row)`` 解析单题，``load_lme_s(path)`` 解析整份文件
  （JSON 数组或 JSONL 自动识别）；
* 三个数组长度不一致时**显式抛 ValueError**，绝不静默截断；
* 只投影公开字段（question / type / answers / abstention / question_date /
  sessions）；gold evidence（``answer_session_ids`` 等）即使出现在原始行里
  也会被丢弃，绝不进入 Episode —— ``assert_public_episode()`` 负责事后复核；
* session 按 ``date`` 升序**稳定**排序（同 date 保原序），两条管线拿到同一写入顺序。

Text-RAG（``lme_s_text_rag/run_text_rag.py``）与 latent（``lme_s_latent/run_e3.py``）
都必须经由本模块取数，避免"两个 baseline 读入不同内容"。

两种输入格式（``load_lme_s`` 自动探测，官方 cleaned schema 是金标准）
--------------------------------------------------------------------
``cleaned``（官方 longmemeval_s_cleaned.json）：一行一题，含
``question_id / question / question_type / question_date / answer`` 与三个
按下标对齐的 ``haystack_session_ids / haystack_dates / haystack_sessions`` 数组。
``prepared``（本仓库 ``data/prepared/longmemeval_test.json``，由
``vcrm.data.prepare()`` 从 cleaned 派生）：一行一题，顶层为
``{id, dataset, split, group, events[], queries[]}``，**没有** haystack_* 三数组，
因此不能直接喂给 ``parse_item``。

prepared → cleaned 字段审计表（对齐 ``vcrm/data.py::longmemeval()`` 的既有语义）：

| prepared                                   | cleaned                    | 说明 |
|--------------------------------------------|----------------------------|------|
| ``queries[0].id``（回退 ``group``，再回退 ``id`` 去掉 ``longmemeval:`` 前缀） | ``question_id`` | 一题一 query；多 query 显式报错 |
| ``queries[0].question``                    | ``question``               | 缺失显式报错 |
| ``queries[0].type``                        | ``question_type``          | 官方六类之一 |
| ``queries[0].answers[0]``                  | ``answer``                 | 0 个 → 无答案；>1 个显式报错 |
| ``queries[0].question_date`` / ``.time``   | ``question_date``          | ``longmemeval()`` 里 query.time 就是 question_date |
| ``queries[0].abstention``                  | （派生）                    | 官方路径按 question_id 的 ``_abs`` 后缀派生，保持一致 |
| ``events[i].id``（回退 ``session_{i:06d}``） | ``haystack_session_ids[i]`` | prepared 无 session_id，用 event id 作稳定派生 id |
| ``events[i].time``                         | ``haystack_dates[i]``      | 缺失**显式报错**，绝不猜日期 |
| ``events[i].text`` 逐行 ``role: content``   | ``haystack_sessions[i]``   | 每行一个 turn；不带 role 前缀的续行并入上一 turn 的 content |

prepared 的 ``events`` 本身就是"一 event = 一 haystack session"（``longmemeval()``
就是这样生成的），所以聚合边界取 event 边界，不额外猜测。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .data import timestamp as normalize_timestamp
from .io import read_records

#: 官方六类任务（顺序与 docs/LME_S_OFFICIAL_EVAL_SPEC.md 一致）。
LME_TASK_TYPES: Tuple[str, ...] = ('single-session-user', 'single-session-assistant',
                                   'single-session-preference', 'temporal-reasoning',
                                   'knowledge-update', 'multi-session')

#: 官方 cleaned LongMemEval-S 题数。
OFFICIAL_S_SIZE = 500

#: 官方六类题数（LongMemEval 论文/README 公布值）。仅在
#: ``LME_STRICT_OFFICIAL_COUNTS=1`` 时由门控测试逐类严格比对。
OFFICIAL_S_TYPE_COUNTS: Dict[str, int] = {
    'single-session-user': 70,
    'single-session-assistant': 56,
    'single-session-preference': 30,
    'multi-session': 133,
    'temporal-reasoning': 133,
    'knowledge-update': 78,
}

#: 允许从原始行读取的字段（白名单）；其它字段一律丢弃。
PUBLIC_ITEM_FIELDS: Tuple[str, ...] = ('question_id', 'question', 'question_type',
                                       'question_date', 'answer',
                                       'haystack_session_ids', 'haystack_dates',
                                       'haystack_sessions')

#: 解析必需字段（缺任何一个都显式报错；question_date / question_type / answer 可缺）。
REQUIRED_ITEM_FIELDS: Tuple[str, ...] = ('question_id', 'question',
                                         'haystack_session_ids', 'haystack_dates',
                                         'haystack_sessions')

#: 公开 turn / session 字段。
PUBLIC_TURN_FIELDS: Tuple[str, ...] = ('role', 'content')
PUBLIC_SESSION_FIELDS: Tuple[str, ...] = ('session_id', 'date', 'raw_date', 'turns')

#: 只属于评测侧（gold evidence / Oracle 路由）的字段：任何输出里出现即视为泄漏。
GOLD_ONLY_FIELDS = frozenset({
    'answer_session_ids',   # gold evidence：直接给出答案所在 session
    'evidence_session_ids',
    'evidence',
    'has_answer',
    'state_key',
    'key_text',
    'entity',
    'group',
    'latest',
    'episode',
})


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class Turn:
    """一条对话轮次（官方 schema：role + content）。"""

    role: str
    content: str

    def __post_init__(self) -> None:
        _require(isinstance(self.role, str) and self.role != '', 'Turn.role must be a non-empty str')
        _require(isinstance(self.content, str), 'Turn.content must be str')

    def render(self) -> str:
        return f'{self.role}: {self.content}'

    def to_dict(self) -> Dict[str, str]:
        return {'role': self.role, 'content': self.content}


@dataclass(frozen=True)
class Session:
    """一个 haystack session：session_id + 归一化 date + 原始 date + turn 列表。

    ``date`` 是 ``vcrm.data.timestamp`` 归一化后的 ISO8601（排序/比较用）；
    ``raw_date`` 保留官方原串，供 text-RAG 的 ``[date]`` 前缀原样渲染
    （不改变既有 prompt 文本）。
    """

    session_id: str
    date: str
    turns: Tuple[Turn, ...]
    raw_date: str = ''

    def __post_init__(self) -> None:
        _require(isinstance(self.session_id, str) and self.session_id != '',
                 'Session.session_id must be a non-empty str')
        _require(isinstance(self.date, str) and self.date != '',
                 'Session.date must be a non-empty ISO8601 str')
        _require(isinstance(self.turns, tuple), 'Session.turns must be a tuple of Turn')

    @property
    def text(self) -> str:
        """按原始顺序渲染：``role: content`` 逐行。"""
        return '\n'.join(turn.render() for turn in self.turns)

    def render(self, prefix_date: bool = False) -> str:
        """渲染为检索文档文本；``prefix_date`` 复刻 text-RAG 的 ``[date]`` 前缀。"""
        body = self.text
        if not prefix_date:
            return body
        return f'[{self.raw_date or self.date}] {body}'

    def to_dict(self) -> Dict[str, Any]:
        return {'session_id': self.session_id, 'date': self.date, 'raw_date': self.raw_date,
                'turns': [turn.to_dict() for turn in self.turns]}


@dataclass(frozen=True)
class Episode:
    """一道 LongMemEval-S 题目的公开视图（无 gold evidence、无 Oracle 字段）。"""

    question_id: str
    question: str
    question_type: str
    answers: Tuple[str, ...]
    abstention: bool
    question_date: Optional[str]
    sessions: Tuple[Session, ...] = field(default_factory=tuple)

    @property
    def type(self) -> str:
        """官方字段名 ``question_type`` 的短别名（评测分组用）。"""
        return self.question_type

    def to_dict(self) -> Dict[str, Any]:
        return {'question_id': self.question_id, 'question': self.question,
                'type': self.question_type, 'question_type': self.question_type,
                'answers': list(self.answers), 'abstention': self.abstention,
                'question_date': self.question_date,
                'sessions': [session.to_dict() for session in self.sessions]}

    def session_texts(self, prefix_date: bool = False) -> List[str]:
        """按写入顺序给出每个 session 的渲染文本（text-RAG 直接用）。"""
        return [session.render(prefix_date=prefix_date) for session in self.sessions]


def _parse_turn(turn: Any, where: str) -> Turn:
    _require(isinstance(turn, Mapping),
             f'{where}: turn must be a mapping with role/content, got {type(turn).__name__}')
    missing = [key for key in PUBLIC_TURN_FIELDS if key not in turn]
    _require(not missing, f'{where}: turn is missing required field(s) {missing}')
    return Turn(role=turn['role'], content=turn['content'])


def _parse_session(index: int, session_id: Any, date: Any, turns: Any,
                   question_id: Any) -> Session:
    where = f'question {question_id!r} session[{index}]'
    _require(isinstance(session_id, str) and session_id != '',
             f'{where}: session_id must be a non-empty str, got {session_id!r}')
    _require(isinstance(date, str) and date.strip() != '',
             f'{where}: date must be a non-empty str, got {date!r}')
    # 官方 schema：haystack_sessions 的元素是 turn 列表，不是 dict。
    _require(isinstance(turns, (list, tuple)),
             f'{where}: haystack_sessions[{index}] must be a list of turns, '
             f'got {type(turns).__name__} — the official cleaned schema stores turns, '
             f'not a session dict')
    parsed = tuple(_parse_turn(turn, f'{where} turn[{i}]') for i, turn in enumerate(turns))
    return Session(session_id=session_id, date=normalize_timestamp(date),
                   turns=parsed, raw_date=str(date))


def parse_item(row: Mapping[str, Any]) -> Episode:
    """解析单题（官方 cleaned schema）。长度不一致 → ValueError，绝不静默截断。"""
    _require(isinstance(row, Mapping),
             f'LongMemEval item must be a mapping, got {type(row).__name__}')
    missing = [key for key in REQUIRED_ITEM_FIELDS if key not in row]
    _require(not missing,
             f'LongMemEval item {row.get("question_id")!r} is missing required field(s) {missing}')

    ids = list(row['haystack_session_ids'])
    dates = list(row['haystack_dates'])
    raw_sessions = list(row['haystack_sessions'])
    _require(len(ids) == len(dates) == len(raw_sessions),
             f'misaligned session arrays for question {row.get("question_id")!r}: '
             f'haystack_session_ids={len(ids)} haystack_dates={len(dates)} '
             f'haystack_sessions={len(raw_sessions)} — refusing to zip-truncate')

    sessions = [_parse_session(index, session_id, date, turns, row.get('question_id'))
                for index, (session_id, date, turns) in enumerate(zip(ids, dates, raw_sessions))]
    # 稳定排序：同 date 保持原始相对顺序（Python sorted 保证 stable）。
    sessions.sort(key=lambda session: session.date)

    answer = row.get('answer')
    answers: Tuple[str, ...] = () if answer is None else (str(answer),)
    question_date = row.get('question_date')
    question_id = str(row['question_id'])
    episode = Episode(question_id=question_id,
                      question=str(row['question']),
                      question_type=str(row.get('question_type') or ''),
                      answers=answers,
                      abstention=question_id.endswith('_abs'),
                      question_date=None if question_date in (None, '') else normalize_timestamp(question_date),
                      sessions=tuple(sessions))
    assert_public_episode(episode, f'parse_item({question_id!r})')
    return episode


#: 官方 cleaned schema 的判别键。
CLEANED_MARKER = 'haystack_sessions'
#: prepared 派生格式的判别键（vcrm.data.prepare() 的输出）。
PREPARED_MARKERS: Tuple[str, ...] = ('events', 'queries')
#: prepared 文本里合法的对话角色（决定一行是否开启新 turn）。
KNOWN_ROLES: Tuple[str, ...] = ('user', 'assistant', 'system')
#: prepared 行允许的 dataset 值（None/'' 表示未标注，按 longmemeval 处理）。
PREPARED_DATASETS: Tuple[str, ...] = ('longmemeval',)


def detect_lme_s_format(row: Mapping[str, Any]) -> str:
    """判别单行格式：'cleaned' | 'prepared'。两套标记同时出现 → 显式报错（不混猜）。"""
    _require(isinstance(row, Mapping),
             f'LongMemEval-S item must be a mapping, got {type(row).__name__}')
    has_cleaned = CLEANED_MARKER in row
    has_prepared = any(marker in row for marker in PREPARED_MARKERS)
    _require(not (has_cleaned and has_prepared),
             f'ambiguous LongMemEval-S row: has both {CLEANED_MARKER!r} (cleaned) and '
             f'{list(PREPARED_MARKERS)} (prepared) — refusing to guess the format')
    if has_cleaned:
        return 'cleaned'
    if has_prepared:
        return 'prepared'
    missing = [CLEANED_MARKER, *PREPARED_MARKERS]
    raise ValueError(f'unrecognized LongMemEval-S row {row.get("question_id") or row.get("id")!r}: '
                     f'missing every format marker {missing}; keys={sorted(map(str, row))}')


def turns_from_prepared_text(text: Any, where: str) -> List[Dict[str, str]]:
    """把 prepared 的事件文本还原成 turn 列表（``role: content`` 逐行）。

    带合法 role 前缀（user/assistant/system）的行开启新 turn；不带前缀的行视为
    上一条 turn 的续行（prepared 用 ``\\n`` 拼接 turn，content 内换行会落到这里）；
    空行跳过。首行就没有 role 前缀 → 显式报错（不猜说话人）。
    """
    _require(isinstance(text, str) and text.strip() != '',
             f'{where}: event text must be a non-empty str, got {type(text).__name__}')
    turns: List[Dict[str, str]] = []
    for line in text.split('\n'):
        if not line.strip():
            continue
        role, separator, content = line.partition(': ')
        if separator and role.strip().lower() in KNOWN_ROLES:
            turns.append({'role': role.strip(), 'content': content})
        elif turns:
            turns[-1]['content'] = f"{turns[-1]['content']}\n{line}"
        else:
            raise ValueError(f'{where}: cannot parse turn line {line!r} '
                             f'(no leading role from {list(KNOWN_ROLES)})')
    _require(bool(turns), f'{where}: event text produced no turns')
    return turns


def adapt_prepared_item(row: Mapping[str, Any]) -> Dict[str, Any]:
    """prepared 派生行（events/queries）→ 官方 cleaned schema 行。

    字段对应关系见模块 docstring 的审计表；语义对齐 ``vcrm/data.py::longmemeval()``：
    一 event = 一 haystack session，query.time 就是 question_date。
    缺 event date / 缺 question / 多 query / 多 answer 一律显式 ValueError（不猜）。
    """
    _require(isinstance(row, Mapping),
             f'prepared LongMemEval-S item must be a mapping, got {type(row).__name__}')
    dataset = row.get('dataset')
    _require(dataset in (None, '') or str(dataset) in PREPARED_DATASETS,
             f'prepared item {row.get("id")!r}: dataset {dataset!r} is not '
             f'{list(PREPARED_DATASETS)} — wrong adapter for this dataset')

    queries = row.get('queries')
    _require(isinstance(queries, (list, tuple)) and len(queries) == 1,
             f'prepared item {row.get("id")!r}: expected exactly 1 query '
             f'(one prepared episode = one question), got '
             f'{len(queries) if isinstance(queries, (list, tuple)) else type(queries).__name__}')
    query = queries[0]
    _require(isinstance(query, Mapping),
             f'prepared item {row.get("id")!r}: query must be a mapping')
    question = query.get('question')
    _require(isinstance(question, str) and question.strip() != '',
             f'prepared item {row.get("id")!r}: query is missing a non-empty '
             f"'question' field")

    question_id = query.get('id') or row.get('group') or row.get('id')
    _require(question_id not in (None, ''),
             f'prepared item {row.get("id")!r}: cannot resolve question_id '
             f"(no query['id'], group or id)")
    question_id = str(question_id)
    if question_id.startswith('longmemeval:'):
        question_id = question_id.split(':', 1)[1]

    answers = query.get('answers')
    answers = [] if answers is None else list(answers)
    _require(len(answers) <= 1,
             f'prepared question {question_id!r}: expected at most 1 answer '
             f'(cleaned schema is single-answer), got {len(answers)}: {answers!r}')
    answer = None if not answers else answers[0]

    events = row.get('events')
    _require(isinstance(events, (list, tuple)),
             f'prepared item {question_id!r}: events must be a list, '
             f'got {type(events).__name__}')

    session_ids, session_dates, session_turns = [], [], []
    for index, event in enumerate(events):
        where = f'prepared question {question_id!r} event[{index}]'
        _require(isinstance(event, Mapping), f'{where}: event must be a mapping')
        date = event.get('time') or event.get('date')
        # 不猜日期：prepared 里没有时间就是数据问题，必须显式失败。
        _require(isinstance(date, str) and date.strip() != '',
                 f'{where}: event has no usable date (looked for time/date); '
                 f'refusing to invent a timestamp')
        session_id = event.get('session_id') or event.get('id') or f'session_{index:06d}'
        session_ids.append(str(session_id))
        session_dates.append(str(date))
        session_turns.append(turns_from_prepared_text(event.get('text'), where))

    cleaned = {
        'question_id': question_id,
        'question': question,
        'question_type': query.get('type') or query.get('question_type') or '',
        'question_date': query.get('question_date') or query.get('time'),
        'answer': answer,
        'haystack_session_ids': session_ids,
        'haystack_dates': session_dates,
        'haystack_sessions': session_turns,
    }
    # 适配产物本身就是公开视图：gold/Oracle 字段若被误带进来会在这里失败。
    assert_public_episode(cleaned, f'adapt_prepared_item({question_id!r})')
    return cleaned


def load_lme_s(path: str | Path) -> List[Episode]:
    """读取整份 LongMemEval-S 文件（JSON 数组或 JSONL，格式自动探测）。

    * 官方 cleaned 行（含 ``haystack_sessions``）→ 直接 ``parse_item``（语义不变）；
    * prepared 派生行（含 ``events``/``queries``）→ ``adapt_prepared_item`` 再 ``parse_item``；
    * 同一文件内混用两种格式、或两种标记都不匹配 → 显式 ValueError。
    """
    records = read_records(path)
    _require(isinstance(records, list) and bool(records),
             f'no LongMemEval-S items found in {path}')
    formats = {detect_lme_s_format(row) for row in records}
    _require(len(formats) == 1,
             f'{path}: mixed LongMemEval-S formats {sorted(formats)} in one file')
    if formats == {'prepared'}:
        records = [adapt_prepared_item(row) for row in records]
    return [parse_item(row) for row in records]


def _walk_keys(node: Any, path: str) -> Iterable[Tuple[str, Any]]:
    if isinstance(node, Mapping):
        for key, value in node.items():
            yield f'{path}.{key}', key
            yield from _walk_keys(value, f'{path}.{key}')
    elif isinstance(node, (list, tuple)):
        for index, value in enumerate(node):
            yield from _walk_keys(value, f'{path}[{index}]')
    elif isinstance(node, Episode):
        yield from _walk_keys(node.to_dict(), path)
    elif isinstance(node, Session):
        yield from _walk_keys(node.to_dict(), path)
    elif isinstance(node, Turn):
        yield from _walk_keys(node.to_dict(), path)


def assert_public_episode(obj: Any, where: str = 'episode') -> Any:
    """复核 Episode/其 dict 形式只含公开字段；出现 gold evidence 或未知字段即 ValueError。"""
    allowed = set(PUBLIC_ITEM_FIELDS) | set(PUBLIC_SESSION_FIELDS) | set(PUBLIC_TURN_FIELDS)
    allowed |= {'type', 'question_type', 'answers', 'abstention', 'question_date',
                'sessions', 'question', 'question_id', 'session_id', 'date',
                'raw_date', 'turns', 'role', 'content'}
    for path, key in _walk_keys(obj, where):
        if key in GOLD_ONLY_FIELDS:
            raise ValueError(f'{path}: gold/Oracle field {key!r} must never reach an Episode')
        if key not in allowed:
            raise ValueError(f'{path}: field {key!r} is not a public LongMemEval-S field')
    return obj


def task_type_counts(episodes: Sequence[Episode]) -> Dict[str, int]:
    """按官方六类统计题数（评测/门控测试共用）。"""
    counts = {name: 0 for name in LME_TASK_TYPES}
    for episode in episodes:
        counts[episode.question_type] = counts.get(episode.question_type, 0) + 1
    return counts


__all__ = [
    'CLEANED_MARKER',
    'Episode',
    'GOLD_ONLY_FIELDS',
    'KNOWN_ROLES',
    'LME_TASK_TYPES',
    'OFFICIAL_S_SIZE',
    'OFFICIAL_S_TYPE_COUNTS',
    'PREPARED_DATASETS',
    'PREPARED_MARKERS',
    'PUBLIC_ITEM_FIELDS',
    'PUBLIC_SESSION_FIELDS',
    'PUBLIC_TURN_FIELDS',
    'REQUIRED_ITEM_FIELDS',
    'Session',
    'Turn',
    'adapt_prepared_item',
    'assert_public_episode',
    'detect_lme_s_format',
    'load_lme_s',
    'parse_item',
    'task_type_counts',
    'turns_from_prepared_text',
]
