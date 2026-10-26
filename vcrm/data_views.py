from __future__ import annotations
"""P0 boundary: the single oracle-free data interface for the Predicted path.

写视图（write）只允许 {'id', 'text', 'time'}；查询视图（query）只允许
{'question', 'time', 'time_source'}。两者都不携带 state_key / key_text / entity /
group / answers / latest / episode 等 Oracle 字段，因此 Predicted 路径在结构上
无法读到“目标 episode / 目标状态 / 参考答案”。

时间通道同样受管（t3 泄漏审计结论）：
* Predicted 正式口径 time_mode='window_end' 必须由调用方显式传入 episode 观测窗口
  末刻 window_end；未提供时输出 time=None、time_source='unknown'，**禁止静默回退**
  到 query['time'] / query['as_of']。
* **Predicted 正式口径不得使用 as_of。** 在 TempLAMA 上
  query['time'] == query['as_of'] == 金标版本 event 的时间戳
  （见 vcrm/data.py:39,44,47-50），一旦走 as_of，store.select_version 的版本选择
  就被时间戳直接解掉，Predicted 指标失去意义。as_of 仅允许用于受控对照与
  时间问答（temporal QA）协议口径，且必须按 time_source 分口径统计、不得与
  window_end 口径混算。

vcrm.data.ingestion_view() / query_view() 一行不改，继续作为 Oracle 对照组
（以及既有 ICAE 训练/评测路径）的唯一入口。本模块是与之并行的新入口，
二者不可互相混用。
"""

from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

#: 公开写视图允许的键（严格白名单）。
PUBLIC_WRITE_KEYS: Sequence[str] = ('id', 'text', 'time')
#: 公开查询视图允许的键（严格白名单）。
PUBLIC_QUERY_KEYS: Sequence[str] = ('question', 'time', 'time_source')

#: 查询时间语义。'window_end' 是 Predicted 正式口径；'as_of' 仅限受控对照/时间问答。
TIME_MODES: Sequence[str] = ('as_of', 'window_end')

#: 查询视图 time 字段的来源标记；评测器据此分口径统计。
TIME_SOURCES: Sequence[str] = ('as_of', 'window_end', 'unknown')

#: 禁止出现在任何 Predicted 视图中的 Oracle 字段。
ORACLE_FIELDS = frozenset({
    'state_key',   # Oracle：目标状态（entity:relation）直接给答案
    'key_text',    # Oracle：检索键原文，等价于答案线索
    'entity',      # Oracle：episode 实体
    'group',       # Oracle：episode 分组（entity:relation）
    'answers',     # Oracle：参考答案
    'latest',      # Oracle：是否为最新版本
    'episode',     # Oracle：episode 句柄/对象
    'as_of',       # Oracle：查询目标时间点
    'is_update',   # Oracle：是否更新的标注
    'target_event',  # Oracle：目标事件指针
})

def _as_mapping(obj: Any, where: str) -> Mapping[str, Any]:
    if isinstance(obj, Mapping):
        return obj
    raise ValueError(f'{where}: expected a mapping, got {type(obj).__name__}')


def public_write_event(event: Mapping[str, Any], public_id: str) -> Dict[str, str]:
    """把一条原始事件投影成 Oracle-free 写视图。

    只读取 event 的 text / time（若事件自带 Oracle 字段则被静默丢弃），
    输出的 id 完全由调用方给的 public_id 决定（通常是 PublicEventCounter 的编号），
    与源 event['id'] 无任何字符串关系。
    """
    source = _as_mapping(event, 'public_write_event')
    if not isinstance(public_id, str) or not public_id.strip():
        raise ValueError('public_write_event: public_id must be a non-empty str')
    try:
        text, time = source['text'], source['time']
    except KeyError as exc:
        raise ValueError(f'public_write_event: event is missing required field {exc.args[0]!r}') from None
    view = {'id': public_id, 'text': text, 'time': time}
    assert_no_oracle(view, 'public_write_event', allowed=PUBLIC_WRITE_KEYS)
    return view


def public_query(query: Mapping[str, Any], time_mode: str = 'window_end',
                 window_end: Optional[str] = None) -> Dict[str, Any]:
    """把一条原始查询投影成 Oracle-free 查询视图：question / time / time_source。

    time_mode:
      'window_end'（Predicted 正式默认）— time 只来自显式传入的 window_end
          （该 episode 观测窗口末刻 / 数据集声明的“当前时刻”）。window_end 未提供时
          time=None、time_source='unknown'，**绝不**静默回退到 query['time'] 或
          query['as_of']（二者在 TempLAMA 上等于金标版本时间戳）。
      'as_of'（仅受控对照 / 时间问答口径）— time 取 query['time']，time_source='as_of'。

    返回值 {'question', 'time', 'time_source'}；time_source ∈ TIME_SOURCES，
    评测器据此分口径统计，as_of 口径结果不得与 window_end 口径混算。

    t25 时间语义审计（本函数**不区分** target/cutoff，语义由调用方决定）：

    * ``time`` 在这里是**目标时间**（``target_time``）通道 —— 下游
      ``PredictedRuntime`` 用它做版本选择上界（``select_version(state, target_time)``）。
      它应当来自问题文本解析（``vcrm.query_time.resolve_query_time(...).target_time``），
      无历史锚时传 ``window_end=None``（= 读最新可见版本）。
    * **可见性截止**（``observation_cutoff``）**不能**经这个参数传递：
      ``public_query`` 不接收它，它由 ``answer_predicted(observation_cutoff=...)``
      直接进读取层。cutoff 只允许来自公开合法信息（``question_date`` / 提问时刻 /
      episode 观测窗口末刻），**绝不允许**传 ``target_time``。
    * ``time_mode='as_of'`` 是 TempLAMA 旧口径（金标版本时间戳 == 观测时刻），
      只允许在 ``vcrm.query_time.resolve_bounds(..., time_mode='as_of',
      allow_as_of=True)`` 显式开启的对照路径上使用。
    * 行为与 t13 契约一致：本函数**不**做任何时间推断，也不读 ``query['as_of']``。
    """
    source = _as_mapping(query, 'public_query')
    if time_mode not in TIME_MODES:
        raise ValueError(
            f'public_query: time_mode must be one of {TIME_MODES}, got {time_mode!r}')
    if 'question' not in source:
        raise ValueError("public_query: query is missing required field 'question'")

    if time_mode == 'as_of':
        # 受控对照/时间问答口径：只在这里允许读 query['time']。
        if 'time' not in source:
            raise ValueError("public_query: time_mode='as_of' requires query['time']")
        time, time_source = source['time'], 'as_of'
    else:
        if window_end is None:
            # 禁止静默回退：宁可显式 unknown，也不把金标时间戳塞给版本选择。
            time, time_source = None, 'unknown'
        else:
            if not isinstance(window_end, str) or not window_end.strip():
                raise ValueError(
                    'public_query: window_end must be a non-empty ISO8601 str, '
                    f'got {window_end!r}')
            time, time_source = window_end, 'window_end'

    view = {'question': source['question'], 'time': time, 'time_source': time_source}
    assert_no_oracle(view, 'public_query', allowed=PUBLIC_QUERY_KEYS)
    return view


class PublicEventCounter:
    """与 Wikidata entity/relation 完全无关的公开事件编号器。

    编号形如 input_000001，只保证在单次运行内单调递增、全局唯一；
    它不携带实体、关系、episode 或时间信息，因此不可能被反推成 Oracle 指针。
    """

    def __init__(self, start: int = 1, prefix: str = 'input') -> None:
        if not isinstance(start, int) or isinstance(start, bool) or start < 1:
            raise ValueError(f'PublicEventCounter.start must be a positive int, got {start!r}')
        if not isinstance(prefix, str) or not prefix.strip():
            raise ValueError('PublicEventCounter.prefix must be a non-empty str')
        self._next = start
        self.prefix = prefix

    @property
    def next_index(self) -> int:
        return self._next

    def next_id(self) -> str:
        public_id = f'{self.prefix}_{self._next:06d}'
        self._next += 1
        return public_id

    # 便于直接当作生成器使用
    __call__ = next_id

    def reset(self, start: int = 1) -> None:
        if not isinstance(start, int) or isinstance(start, bool) or start < 1:
            raise ValueError(f'PublicEventCounter.start must be a positive int, got {start!r}')
        self._next = start

    def __repr__(self) -> str:  # pragma: no cover - 诊断用
        return f'PublicEventCounter(prefix={self.prefix!r}, next_index={self._next})'


def _iter_child_keys(node: Any, path: str) -> Iterator[tuple]:
    """递归遍历，产出 (key_path, key) 对。"""
    if isinstance(node, Mapping):
        for key, value in node.items():
            yield f'{path}.{key}', key
            yield from _iter_child_keys(value, f'{path}.{key}')
    elif isinstance(node, (list, tuple, set, frozenset)):
        for index, value in enumerate(node):
            yield from _iter_child_keys(value, f'{path}[{index}]')


def assert_no_oracle(obj: Any, where: str = 'object', allowed: Optional[Iterable[str]] = None) -> Any:
    """白名单校验器：Predicted 视图里出现任何 Oracle 字段即抛 ValueError。

    where:   出错信息里的位置标签（例如 'public_write_event'）
    allowed: 若给出，则额外要求所有键都落在该白名单内（严格模式）

    返回 obj 本身，便于 `view = assert_no_oracle(public_write_event(...), 'ingest')`。
    """
    allowed_set = None if allowed is None else frozenset(allowed)
    for path, key in _iter_child_keys(obj, where):
        if isinstance(key, str) and key in ORACLE_FIELDS:
            raise ValueError(
                f'{path}: Oracle field {key!r} is forbidden in Predicted views '
                f'(forbidden: {sorted(ORACLE_FIELDS)})')
        if allowed_set is not None and key not in allowed_set:
            raise ValueError(
                f'{path}: key {key!r} is not in the public whitelist {sorted(allowed_set)}')
    return obj


def assert_public_id_independent(public_id: str, source_id: object) -> str:
    """断言公开编号与源事件 id 不存在字符串包含关系（防 Oracle 回溯）。

    短 id（< 4 字符）无法做有意义的包含判断，直接跳过，避免误报。
    """
    if not isinstance(public_id, str):
        raise ValueError(f'public_id must be str, got {type(public_id).__name__}')
    if not isinstance(source_id, str) or len(source_id) < 4:
        return public_id
    if source_id in public_id or public_id in source_id:
        raise ValueError(
            f'public_id {public_id!r} is not independent of source event id {source_id!r}')
    return public_id


__all__ = [
    'ORACLE_FIELDS',
    'PUBLIC_QUERY_KEYS',
    'PUBLIC_WRITE_KEYS',
    'PublicEventCounter',
    'TIME_MODES',
    'TIME_SOURCES',
    'assert_no_oracle',
    'assert_public_id_independent',
    'public_query',
    'public_write_event',
]
