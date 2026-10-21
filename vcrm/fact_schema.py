from __future__ import annotations
# P0 boundary: oracle-free fact / state-decision schema.
#
# 状态身份（state identity）= subject + attribute。
# `value` 与 `observed_at` **不**参与状态身份判定：同一 (subject, attribute) 在不同时间
# 观察到不同 value，仍然属于同一个状态；到底算新建、更新还是重复，由 UpdateResolver
# 依据历史状态决定（CREATE / UPDATE / DUPLICATE / DEFER）。
# 时间只用于排序与“截至某时刻可见性”的裁剪，不作为状态身份的一部分。
#
# 本模块刻意不包含任何 Oracle 概念（state_key / key_text / entity / relation /
# group / answers / latest / target episode）。上游 Predicted 视图必须先把原始
# episode 收敛成 AtomicFact 才能进入关联与存储层。

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

#: 允许的状态决策动作；任何其它取值都视为非法（见 StateDecision.__post_init__）。
STATE_ACTIONS: Tuple[str, ...] = ('CREATE', 'UPDATE', 'DUPLICATE', 'DEFER')

#: 需要已有 state 才能成立的动作。
STATE_ACTIONS_REQUIRING_STATE: Tuple[str, ...] = ('UPDATE', 'DUPLICATE')

StateAction = Literal['CREATE', 'UPDATE', 'DUPLICATE', 'DEFER']


def _require_text(value: object, field: str, allow_empty: bool = True) -> str:
    if not isinstance(value, str):
        raise ValueError(f'AtomicFact.{field} must be str, got {type(value).__name__}')
    if not allow_empty and not value.strip():
        raise ValueError(f'AtomicFact.{field} must be non-empty')
    return value


@dataclass(frozen=True)
class AtomicFact:
    """一条从 Predicted 视图抽取出的原子事实，无任何 Oracle 字段。

    id:          事实自身标识（由 PublicEventCounter 之类的公开编号生成，与源 event['id'] 无关）
    text:        事实的自然语言表述
    observed_at: 观察到该事实的时间（ISO8601 字符串），仅用于排序/裁剪
    subject:     状态身份的左半部分
    attribute:   状态身份的右半部分
    value:       该时刻的取值（不参与状态身份）
    source_id:   产生该事实的公开事件编号（可追溯到公开输入，不是 Oracle 指针）
    """

    id: str
    text: str
    observed_at: str
    subject: str
    attribute: str
    value: str
    source_id: str

    def __post_init__(self) -> None:
        _require_text(self.id, 'id', allow_empty=False)
        _require_text(self.text, 'text')
        _require_text(self.observed_at, 'observed_at', allow_empty=False)
        _require_text(self.subject, 'subject', allow_empty=False)
        _require_text(self.attribute, 'attribute', allow_empty=False)
        _require_text(self.value, 'value')
        _require_text(self.source_id, 'source_id')

    @property
    def state_identity(self) -> Tuple[str, str]:
        """状态身份 = (subject, attribute)；value 与时间不参与。"""
        return (self.subject, self.attribute)

    def identity_key(self) -> str:
        """状态身份的可比较字符串形式，仅供索引/调试使用。"""
        return f'{self.subject}\u241f{self.attribute}'


@dataclass(frozen=True)
class StateDecision:
    """对一条 AtomicFact 的状态层决策，不含任何 Oracle 依据。

    action:      'CREATE' | 'UPDATE' | 'DUPLICATE' | 'DEFER'
    state_id:    目标状态编号；UPDATE/DUPLICATE 必填，CREATE/DEFER 可为 None
    confidence:  [0, 1] 区间的决策置信度
    reason:      人类可读的决策依据（仅来自公开可见信息）
    """

    action: StateAction
    state_id: Optional[str] = None
    confidence: float = 0.0
    reason: str = ''

    def __post_init__(self) -> None:
        if not isinstance(self.action, str) or self.action not in STATE_ACTIONS:
            raise ValueError(
                f'StateDecision.action must be one of {STATE_ACTIONS}, got {self.action!r}')
        if isinstance(self.confidence, bool) or not isinstance(self.confidence, (int, float)):
            raise ValueError(
                f'StateDecision.confidence must be a number in [0, 1], got {type(self.confidence).__name__}')
        if not (0.0 <= float(self.confidence) <= 1.0):
            raise ValueError(
                f'StateDecision.confidence must be within [0, 1], got {self.confidence!r}')
        if self.state_id is not None and not isinstance(self.state_id, str):
            raise ValueError(
                f'StateDecision.state_id must be str or None, got {type(self.state_id).__name__}')
        if not isinstance(self.reason, str):
            raise ValueError(
                f'StateDecision.reason must be str, got {type(self.reason).__name__}')
        if self.action in STATE_ACTIONS_REQUIRING_STATE and not self.state_id:
            raise ValueError(
                f'StateDecision.action={self.action} requires a non-empty state_id')

    @property
    def creates_state(self) -> bool:
        return self.action == 'CREATE'

    @property
    def touches_existing_state(self) -> bool:
        return self.action in STATE_ACTIONS_REQUIRING_STATE


__all__ = [
    'AtomicFact',
    'StateAction',
    'StateDecision',
    'STATE_ACTIONS',
    'STATE_ACTIONS_REQUIRING_STATE',
]
