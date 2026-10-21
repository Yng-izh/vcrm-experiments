from __future__ import annotations
# P2: Oracle-free state index.
#
# 状态索引只认「从公开文本抽出来的事实」，完全不知道 state_key / key_text /
# entity / group / answers / latest / episode 这些 Oracle 概念。state_id 一律由
# 系统自己分配（state_%06d 或 uuid4），与数据集里的任何标签、实体名、relation id
# 都没有编码关系。
#
# 状态身份 = (subject_key, attribute_key)。value 与时间不参与身份：
# 同一个 (subject, attribute) 在不同时间取不同 value 仍是同一个状态，究竟是
# CREATE / UPDATE / DUPLICATE / DEFER 由 vcrm.update_resolver 判定。
#
# 本模块同时提供：
#   - 规范化函数（normalize_subject / normalize_attribute / value_fingerprint）：
#     这是**语言表面形式**归一化，不含任何数据集 relation id → attribute 的映射表。
#   - StateIndex.commit_with(store, ...)：与 IncrementalCurrentStore 的事务钩子。
#     store 写成功之后才更新索引；索引更新失败则回滚 store 并把索引恢复原样。

import hashlib
import re
import unicodedata
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

#: semantic_key 的维度（与 vcrm.index.SemanticIndex(dimensions=128) 对齐）。
SEMANTIC_DIMENSIONS = 128

_ARTICLES: Tuple[str, ...] = ('the', 'a', 'an')
_WS = re.compile(r'\s+')
_NON_ALNUM = re.compile(r'[^a-z0-9]+')

#: 语言表面形式 → 规范属性名。**不是**数据集 relation id 映射表：
#: 表里只有英文短语，没有任何 Q/P id、没有 state_key 片段。
ATTRIBUTE_ALIASES: Dict[str, str] = {
    'capital': 'capital',
    'capital city': 'capital',
    'head of government': 'head_of_government',
    'head of state': 'head_of_state',
    'head of the government': 'head_of_government',
    'head of the state': 'head_of_state',
    'prime minister': 'prime_minister',
    'president': 'president',
    'monarch': 'monarch',
    'mayor': 'mayor',
    'population': 'population',
    'area': 'area',
    'currency': 'currency',
    'official language': 'official_language',
    'official languages': 'official_language',
    'language': 'official_language',
    'languages': 'official_language',
    'largest city': 'largest_city',
    'spouse': 'spouse',
    'employer': 'employer',
    'owner': 'owner',
    'director': 'director',
    'chief executive officer': 'ceo',
    'ceo': 'ceo',
    'place of birth': 'place_of_birth',
    'date of birth': 'date_of_birth',
    'country of citizenship': 'country_of_citizenship',
    'citizenship': 'country_of_citizenship',
    'member of political party': 'member_of_political_party',
    'political party': 'member_of_political_party',
    'residence': 'residence',
    'alma mater': 'alma_mater',
    'occupation': 'occupation',
    'genre': 'genre',
}

#: ---------------------------------------------------------------------------
#: 属性基数本体表（三档）。键是**语言归一化后的属性字符串**（由英文表面形式派生，
#: 见 normalize_attribute），**严禁**由数据集 relation ID 映射生成。
#:
#: EXCLUSIVE    同一时刻单值：新值出现即意味着旧值失效（仍需要 rule5 的显式替代
#:              表达或 rule5b 的时间更晚证据才会真正 UPDATE）
#: MULTI_VALUED 可并存：第二个取值只能 DEFER，永远不覆盖
#: UNKNOWN      未列出：一律 DEFER（严格基线 unknown_only 会把全部属性当 UNKNOWN）
#:
#: 注意：把 employer 判成「同一时刻单值」是一条**本体论假设**（现实中可以兼职），
#: 属于可消融项；消融方式就是 cardinality_policy='unknown_only'。
#: ---------------------------------------------------------------------------
CARDINALITY_EXCLUSIVE = frozenset({
    'employer', 'occupation', 'position_held', 'spouse', 'head_of_government',
    'head_of_state', 'country_of_citizenship', 'residence', 'place_of_birth',
    'date_of_birth', 'alma_mater', 'religion', 'gender', 'capital', 'currency',
    'official_language', 'leader', 'director', 'ceo', 'owner', 'parent_organization',
})

CARDINALITY_MULTI_VALUED = frozenset({
    'languages_spoken', 'children', 'award_received', 'member_of_sports_team',
    'cast_member', 'genre', 'field_of_work', 'notable_work', 'sibling',
    'participant', 'composer', 'screenwriter', 'author',
})

#: 三档名称与可选策略。
CARDINALITY_BUCKETS: Tuple[str, ...] = ('EXCLUSIVE', 'MULTI_VALUED', 'UNKNOWN')
CARDINALITY_POLICIES: Tuple[str, ...] = ('ontology', 'unknown_only')

#: 向后兼容别名（t5/t15 的调用方与测试用这两个名字）。
EXCLUSIVE_ATTRIBUTES = CARDINALITY_EXCLUSIVE
MULTI_VALUED_ATTRIBUTES = CARDINALITY_MULTI_VALUED


def utcnow() -> str:
    """当前 UTC 时间（秒精度 ISO8601），索引的时间戳只用于排序。"""
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def normalize_text(value: Any) -> str:
    """语言表面形式归一化：NFKC + 去首尾标点 + 折叠空白 + 小写。"""
    if value is None:
        return ''
    text = unicodedata.normalize('NFKC', str(value))
    text = text.replace('\u241f', ' ')
    text = text.replace('\u2019', "'")
    text = _WS.sub(' ', text).strip()
    text = text.strip(' .,;:!?\'"()[]{}')
    return text.lower()


def _strip_leading_articles(text: str) -> str:
    changed = True
    while changed and text:
        changed = False
        for article in _ARTICLES:
            if text.startswith(article + ' '):
                text = text[len(article) + 1:]
                changed = True
        if text.startswith('the '):
            text = text[4:]
            changed = True
    return text


def _slug(text: str) -> str:
    return _NON_ALNUM.sub('_', text).strip('_')


def normalize_subject(subject: Any) -> str:
    """主体规范键：去冠词/所有格，空白与标点折叠成下划线。"""
    text = _strip_leading_articles(normalize_text(subject))
    if text.endswith("'s"):
        text = text[:-2].strip()
    return _slug(text) or 'unknown_subject'


def normalize_attribute(attribute: Any) -> str:
    """属性规范键：先查语言别名表，再做通用 snake_case 归一化。"""
    text = _strip_leading_articles(normalize_text(attribute))
    if text.endswith("'s"):
        text = text[:-2].strip()
    canonical = ATTRIBUTE_ALIASES.get(text)
    if canonical:
        return canonical
    return _slug(text) or 'unknown_attribute'


def canonical_value(value: Any) -> str:
    """取值的等价规范形：小写、去括号补充、去千分位逗号、去冠词、标点转空格。

    "1,234,567" 与 "1234567"、"The Paris" 与 "paris" 在这里收敛成同一个规范形，
    因此 UpdateResolver 的「值等价 → DUPLICATE」判定不依赖任何数据集标注。
    """
    text = normalize_text(value)
    text = re.sub(r'\([^)]*\)', ' ', text)
    text = text.replace(',', '')
    text = re.sub(r'\b(?:the|a|an)\b', ' ', text)
    text = _NON_ALNUM.sub(' ', text)
    return _WS.sub(' ', text).strip()


def attribute_cardinality(attribute: Any, policy: str = 'ontology') -> str:
    """返回属性的基数档位：EXCLUSIVE / MULTI_VALUED / UNKNOWN。

    policy='ontology'      使用上面的三档本体表（默认）
    policy='unknown_only'  严格基线：全部当 UNKNOWN（即全部 DEFER），用于消融
    """
    if policy not in CARDINALITY_POLICIES:
        raise ValueError(f'unknown cardinality policy {policy!r}; expected {CARDINALITY_POLICIES}')
    if policy == 'unknown_only':
        return 'UNKNOWN'
    key = normalize_attribute(attribute)
    if key in CARDINALITY_EXCLUSIVE:
        return 'EXCLUSIVE'
    if key in CARDINALITY_MULTI_VALUED:
        return 'MULTI_VALUED'
    return 'UNKNOWN'


def value_fingerprint(value: Any) -> str:
    """取值的等价指纹（只依赖规范形，不依赖任何数据集标注）。"""
    return hashlib.sha1(canonical_value(value).encode('utf-8')).hexdigest()[:16]


def semantic_dimensions_ok(key: Optional[Sequence[float]]) -> bool:
    return key is None or len(key) == SEMANTIC_DIMENSIONS


def _fact_field(fact: Any, name: str, default: str = '') -> Any:
    """同时接受 AtomicFact 与普通 mapping，保持索引层对 schema 的松耦合。"""
    if isinstance(fact, Mapping):
        return fact.get(name, default)
    return getattr(fact, name, default)


# ---------------------------------------------------------------------------
# 事务错误
# ---------------------------------------------------------------------------

class TransactionError(RuntimeError):
    """事务钩子的基类错误。"""


class StoreCommitError(TransactionError):
    """Current store 写入失败：索引保持原样，调用方可以安全重试。"""


class IndexCommitError(TransactionError):
    """store 已写入但索引更新失败：已尽力回滚 store。"""


# ---------------------------------------------------------------------------
# 记录
# ---------------------------------------------------------------------------

@dataclass
class StateRecord:
    """一条状态记录（subject + attribute 决定身份，value 只留指纹）。

    aliases:             同一主体/属性的表面别名（用于宽松匹配，不参与身份哈希）
    semantic_key:        128 维语义键（来自 SemanticIndex.key，可为 None）
    value_fingerprint:   当前版本取值的等价指纹
    current_version_id:  当前版本编号（由 store 分配，索引只记录指针）
    current_observed_at: 当前版本的观察时间（来自 fact.observed_at，仅用于时间先后比较）
    """

    state_id: str
    subject_key: str
    attribute_key: str
    aliases: List[str] = field(default_factory=list)
    semantic_key: Optional[Tuple[float, ...]] = None
    value_fingerprint: str = ''
    current_version_id: str = ''
    created_at: str = ''
    updated_at: str = ''
    current_observed_at: str = ''

    def __post_init__(self) -> None:
        if not isinstance(self.state_id, str) or not self.state_id.strip():
            raise ValueError('StateRecord.state_id must be a non-empty str')
        if not isinstance(self.subject_key, str) or not self.subject_key.strip():
            raise ValueError('StateRecord.subject_key must be a non-empty str')
        if not isinstance(self.attribute_key, str) or not self.attribute_key.strip():
            raise ValueError('StateRecord.attribute_key must be a non-empty str')
        if self.semantic_key is not None:
            self.semantic_key = tuple(float(x) for x in self.semantic_key)
            if not semantic_dimensions_ok(self.semantic_key):
                raise ValueError(
                    f'StateRecord.semantic_key must have {SEMANTIC_DIMENSIONS} dims, '
                    f'got {len(self.semantic_key)}')
        if self.aliases is None:
            self.aliases = []
        self.aliases = [str(a) for a in self.aliases]

    # -- 身份 -------------------------------------------------------------
    @property
    def state_identity(self) -> Tuple[str, str]:
        return (self.subject_key, self.attribute_key)

    def identity_key(self) -> str:
        return f'{self.subject_key}\u241f{self.attribute_key}'

    def add_alias(self, alias: Any) -> None:
        text = normalize_text(alias)
        if text and text not in self.aliases:
            self.aliases.append(text)

    def matches_subject(self, subject: Any) -> bool:
        key = normalize_subject(subject)
        if key == self.subject_key:
            return True
        return normalize_text(subject) in self.aliases

    def matches_attribute(self, attribute: Any) -> bool:
        key = normalize_attribute(attribute)
        if key == self.attribute_key:
            return True
        return normalize_text(attribute) in self.aliases

    # -- 快照（commit_with 回滚用） ---------------------------------------
    def snapshot(self) -> Tuple[str, str, str, str]:
        return (self.value_fingerprint, self.current_version_id, self.updated_at,
                self.current_observed_at)

    def restore(self, snapshot: Tuple[str, str, str, str]) -> None:
        (self.value_fingerprint, self.current_version_id, self.updated_at,
         self.current_observed_at) = snapshot

    # -- 序列化 -----------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            'state_id': self.state_id,
            'subject_key': self.subject_key,
            'attribute_key': self.attribute_key,
            'aliases': list(self.aliases),
            'semantic_key': None if self.semantic_key is None else list(self.semantic_key),
            'value_fingerprint': self.value_fingerprint,
            'current_version_id': self.current_version_id,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'current_observed_at': self.current_observed_at,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> 'StateRecord':
        return cls(
            state_id=payload['state_id'],
            subject_key=payload['subject_key'],
            attribute_key=payload['attribute_key'],
            aliases=list(payload.get('aliases') or []),
            semantic_key=payload.get('semantic_key'),
            value_fingerprint=payload.get('value_fingerprint', ''),
            current_version_id=payload.get('current_version_id', ''),
            created_at=payload.get('created_at', ''),
            updated_at=payload.get('updated_at', ''),
            current_observed_at=payload.get('current_observed_at', ''),
        )


@dataclass
class PendingMerge:
    """DEFER 的独立待确认记录：只挂在这里，绝不改动 Current。"""

    pending_id: str
    fact_id: str
    subject_key: str
    attribute_key: str
    value: str
    candidate_state_id: Optional[str] = None
    reason: str = ''
    confidence: float = 0.0
    created_at: str = ''
    source_id: str = ''

    def to_dict(self) -> Dict[str, Any]:
        return {
            'pending_id': self.pending_id,
            'fact_id': self.fact_id,
            'subject_key': self.subject_key,
            'attribute_key': self.attribute_key,
            'value': self.value,
            'candidate_state_id': self.candidate_state_id,
            'reason': self.reason,
            'confidence': float(self.confidence),
            'created_at': self.created_at,
            'source_id': self.source_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> 'PendingMerge':
        return cls(
            pending_id=payload['pending_id'],
            fact_id=payload.get('fact_id', ''),
            subject_key=payload.get('subject_key', ''),
            attribute_key=payload.get('attribute_key', ''),
            value=payload.get('value', ''),
            candidate_state_id=payload.get('candidate_state_id'),
            reason=payload.get('reason', ''),
            confidence=float(payload.get('confidence', 0.0) or 0.0),
            created_at=payload.get('created_at', ''),
            source_id=payload.get('source_id', ''),
        )


# ---------------------------------------------------------------------------
# 索引
# ---------------------------------------------------------------------------

class StateIndex:
    """无 Oracle 的状态索引。

    id 分配完全由系统决定：默认 state_000001 递增；也可以传
    id_factory=lambda: 'state_' + uuid.uuid4().hex[:12] 走 uuid4 方案。
    """

    def __init__(self, *, start: int = 1, id_prefix: str = 'state',
                 id_factory: Optional[Callable[[], str]] = None,
                 semantic_dimensions: int = SEMANTIC_DIMENSIONS,
                 now: Optional[Callable[[], str]] = None) -> None:
        if id_factory is not None and not callable(id_factory):
            raise ValueError('StateIndex.id_factory must be callable or None')
        if not isinstance(start, int) or isinstance(start, bool) or start < 1:
            raise ValueError(f'StateIndex.start must be a positive int, got {start!r}')
        self.id_prefix = id_prefix
        self.semantic_dimensions = int(semantic_dimensions)
        self._id_factory = id_factory
        self._next = start
        self._now = now or utcnow
        self._records: Dict[str, StateRecord] = {}
        self._order: List[str] = []
        self._pending: Dict[str, PendingMerge] = {}
        self._pending_next = 1
        self._version_next = 1

    # -- 基本访问 ---------------------------------------------------------
    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, state_id: object) -> bool:
        return state_id in self._records

    def __iter__(self):
        return iter(self.records)

    @property
    def records(self) -> List[StateRecord]:
        return [self._records[sid] for sid in self._order]

    @property
    def pending(self) -> List[PendingMerge]:
        return list(self._pending.values())

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def get(self, state_id: str) -> StateRecord:
        try:
            return self._records[state_id]
        except KeyError:
            raise KeyError(f'unknown state_id {state_id!r}') from None

    def find(self, subject: Any, attribute: Any) -> Optional[StateRecord]:
        subject_key, attribute_key = normalize_subject(subject), normalize_attribute(attribute)
        for record in self.records:
            if record.subject_key == subject_key and record.attribute_key == attribute_key:
                return record
        return None

    def candidates(self, subject_key: Optional[str] = None,
                   attribute_key: Optional[str] = None, *,
                   limit: Optional[int] = None) -> List[StateRecord]:
        """按主体/属性过滤候选状态（别名也算命中）；两个都 None 时返回全部。"""
        subject_norm = None if subject_key is None else normalize_subject(subject_key)
        attribute_norm = None if attribute_key is None else normalize_attribute(attribute_key)
        out: List[StateRecord] = []
        for record in self.records:
            if subject_norm is not None and record.subject_key != subject_norm:
                if normalize_text(subject_key) not in record.aliases:
                    continue
            if attribute_norm is not None and record.attribute_key != attribute_norm:
                if normalize_text(attribute_key) not in record.aliases:
                    continue
            out.append(record)
        if limit is not None:
            out = out[:int(limit)]
        return out

    # -- 写入 -------------------------------------------------------------
    def new_state_id(self) -> str:
        if self._id_factory is not None:
            state_id = self._id_factory()
            if not isinstance(state_id, str) or not state_id.strip():
                raise ValueError('id_factory must return a non-empty str')
            if state_id in self._records:
                raise ValueError(f'id_factory produced a duplicate state_id {state_id!r}')
            return state_id
        state_id = f'{self.id_prefix}_{self._next:06d}'
        self._next += 1
        return state_id

    def build_record(self, fact: Any, *, semantic_key: Optional[Sequence[float]] = None,
                     aliases: Optional[Sequence[str]] = None, version_id: str = '',
                     state_id: Optional[str] = None,
                     now: Optional[str] = None) -> StateRecord:
        """构造一条 StateRecord 但**不插入索引**（store 事务的预备对象）。

        apply_decision 的 CREATE 路径需要先拿到 state_id 去写 store，store 成功之后
        才把记录插入索引；因此这里把「构造」与「插入」分开。
        """
        subject, attribute = _fact_field(fact, 'subject'), _fact_field(fact, 'attribute')
        value = _fact_field(fact, 'value')
        if not str(subject).strip() or not str(attribute).strip():
            raise ValueError('StateIndex.build_record requires non-empty subject and attribute')
        subject_key, attribute_key = normalize_subject(subject), normalize_attribute(attribute)
        existing = self.find(subject_key, attribute_key)
        if existing is not None:
            raise ValueError(
                f'state for {subject_key!r}/{attribute_key!r} already exists as {existing.state_id}')
        stamp = now or self._now()
        return StateRecord(
            state_id=state_id if state_id is not None else self.new_state_id(),
            subject_key=subject_key,
            attribute_key=attribute_key,
            aliases=list(dict.fromkeys(
                [normalize_text(subject), normalize_text(attribute)]
                + [normalize_text(a) for a in (aliases or [])]
            )),
            semantic_key=None if semantic_key is None else tuple(float(x) for x in semantic_key),
            value_fingerprint=value_fingerprint(value),
            current_version_id=version_id if version_id is not None else _fact_field(fact, 'id', ''),
            created_at=stamp,
            updated_at=stamp,
            current_observed_at=str(_fact_field(fact, 'observed_at', '') or ''),
        )

    def add(self, fact: Any, *, semantic_key: Optional[Sequence[float]] = None,
            aliases: Optional[Sequence[str]] = None, version_id: Optional[str] = None,
            state_id: Optional[str] = None, now: Optional[str] = None) -> StateRecord:
        """由一条事实建立新状态并插入索引（state_id 可预分配，供 store 事务用）。"""
        if state_id is not None and state_id in self._records:
            raise ValueError(f'state_id {state_id!r} already exists')
        record = self.build_record(fact, semantic_key=semantic_key, aliases=aliases,
                                   version_id=version_id if version_id is not None else '',
                                   state_id=state_id, now=now)
        self._records[record.state_id] = record
        self._order.append(record.state_id)
        return record

    def discard(self, state_id: str) -> bool:
        """撤销一条刚插入的记录（store 事务失败时的索引侧回滚）。"""
        if state_id not in self._records:
            return False
        del self._records[state_id]
        self._order = [sid for sid in self._order if sid != state_id]
        return True

    def add_keys(self, subject: Any, attribute: Any, value: str = '', *,
                 semantic_key: Optional[Sequence[float]] = None,
                 aliases: Optional[Sequence[str]] = None,
                 version_id: Optional[str] = None,
                 observed_at: str = '',
                 now: Optional[str] = None) -> StateRecord:
        """便捷入口：直接用 subject/attribute 字符串建状态（测试与脚手架用）。"""
        fact = {'id': f'fact_{self._version_next:06d}', 'subject': subject,
                'attribute': attribute, 'value': value, 'observed_at': observed_at}
        self._version_next += 1
        return self.add(fact, semantic_key=semantic_key, aliases=aliases,
                        version_id=version_id, now=now)

    def commit(self, state_id: str, version_id: str, *,
               value_fingerprint: Optional[str] = None,
               observed_at: Optional[str] = None,
               now: Optional[str] = None) -> StateRecord:
        """把已有状态的当前版本指针推进到 version_id。"""
        record = self.get(state_id)
        if not isinstance(version_id, str) or not version_id.strip():
            raise ValueError('StateIndex.commit requires a non-empty version_id str')
        if now is not None and not isinstance(now, str):
            raise TypeError(f'StateIndex.commit now must be str or None, got {type(now).__name__}')
        record.current_version_id = version_id
        record.value_fingerprint = (
            record.value_fingerprint if value_fingerprint is None else value_fingerprint)
        if observed_at is not None:
            record.current_observed_at = str(observed_at)
        record.updated_at = now or self._now()
        return record

    def merge_pending(self, pending_id: str, *, state_id: Optional[str] = None,
                      version_id: Optional[str] = None,
                      value_fingerprint: Optional[str] = None,
                      observed_at: Optional[str] = None,
                      now: Optional[str] = None) -> StateRecord:
        """人工/规则确认后，把一条 DEFER 待确认记录合并进状态层。"""
        try:
            pending = self._pending.pop(pending_id)
        except KeyError:
            raise KeyError(f'unknown pending_id {pending_id!r}') from None
        target_id = state_id or pending.candidate_state_id
        if target_id is None:
            record = self.add_keys(
                pending.subject_key, pending.attribute_key, pending.value,
                version_id=version_id or pending.fact_id, observed_at=observed_at or '', now=now)
            return record
        return self.commit(target_id, version_id or pending.fact_id,
                           value_fingerprint=value_fingerprint,
                           observed_at=observed_at, now=now)

    def add_pending(self, fact: Any, *, candidate_state_id: Optional[str] = None,
                    reason: str = '', confidence: float = 0.0,
                    now: Optional[str] = None) -> PendingMerge:
        """存一条独立待确认记录（DEFER 的落点），不触碰任何 Current 版本。"""
        if candidate_state_id is not None and candidate_state_id not in self._records:
            raise KeyError(f'unknown candidate_state_id {candidate_state_id!r}')
        pending = PendingMerge(
            pending_id=f'pending_{self._pending_next:06d}',
            fact_id=str(_fact_field(fact, 'id', '')),
            subject_key=normalize_subject(_fact_field(fact, 'subject')),
            attribute_key=normalize_attribute(_fact_field(fact, 'attribute')),
            value=str(_fact_field(fact, 'value', '')),
            candidate_state_id=candidate_state_id,
            reason=str(reason),
            confidence=float(confidence),
            created_at=now or self._now(),
            source_id=str(_fact_field(fact, 'source_id', '')),
        )
        self._pending_next += 1
        self._pending[pending.pending_id] = pending
        return pending

    # -- 事务钩子 ---------------------------------------------------------
    def commit_with(self, store: Any, state_id: str, fact: Any = None, *,
                    apply: Optional[Callable[[Any, StateRecord, Any], Any]] = None,
                    rollback: Optional[Callable[[Any, Any], Any]] = None,
                    value_fingerprint: Optional[str] = None,
                    observed_at: Optional[str] = None,
                    memory: Any = None,
                    semantic_key: Optional[Sequence[float]] = None,
                    decision: Any = None,
                    store_kwargs: Optional[Mapping[str, Any]] = None,
                    now: Optional[str] = None) -> str:
        """与 IncrementalCurrentStore 的事务钩子。

        契约：
          1. store 侧写入成功（返回版本号或 {'version_id': ...}）后，才更新索引；
          2. store 写入抛异常 → 索引保持原样，抛 StoreCommitError；
          3. store 成功但索引更新失败 → 尽力回滚 store（store.rollback / 传入的
             rollback 回调）并把记录恢复原样，抛 IndexCommitError。

        apply:    (store, record, fact) -> version，默认调用
                  store.commit(record=..., fact=..., **extras)
        rollback: (store, result) -> None，默认尝试 store.rollback(result)

        真实 IncrementalCurrentStore 强制要求 memory=（原始 full memory）与
        semantic_key=，因此默认 apply 会把 memory / semantic_key / decision /
        store_kwargs 里**非 None** 的项原样透传给 store.commit。
        """
        record = self.get(state_id)
        snapshot = record.snapshot()
        try:
            if apply is not None:
                result = apply(store, record, fact)
            else:
                extras = _store_extras(record, memory=memory, semantic_key=semantic_key,
                                       decision=decision, store_kwargs=store_kwargs)
                result = store.commit(record=record, fact=fact, **extras)
        except Exception as exc:  # noqa: BLE001 - 任何 store 失败都必须原样保留索引
            record.restore(snapshot)
            raise StoreCommitError(
                f'current store commit failed for {state_id}: {exc}') from exc
        try:
            version_id = _coerce_version_id(result)
            if now is not None and not isinstance(now, str):
                raise TypeError(f'now must be str or None, got {type(now).__name__}')
            record.current_version_id = version_id
            if value_fingerprint is not None:
                record.value_fingerprint = value_fingerprint
            if observed_at is not None:
                record.current_observed_at = str(observed_at)
            record.updated_at = now or self._now()
        except Exception as exc:  # noqa: BLE001 - 索引失败必须回滚 store
            record.restore(snapshot)
            rollback_note = self._rollback_store(store, result, rollback)
            raise IndexCommitError(
                f'index update failed after store commit for {state_id}: {exc}; {rollback_note}') from exc
        return version_id

    @staticmethod
    def _rollback_store(store: Any, result: Any,
                        rollback: Optional[Callable[[Any, Any], Any]]) -> str:
        try:
            if rollback is not None:
                rollback(store, result)
                return 'store rollback executed via callback'
            handler = getattr(store, 'rollback', None)
            if callable(handler):
                handler(result)
                return 'store rollback executed via store.rollback'
            return 'no store rollback handler available'
        except Exception as exc:  # noqa: BLE001 - 回滚失败只记录，不掩盖原始错误
            return f'store rollback itself failed: {exc}'

    # -- 序列化 -----------------------------------------------------------
    def serialize(self) -> Dict[str, Any]:
        return {
            'version': 1,
            'id_prefix': self.id_prefix,
            'semantic_dimensions': self.semantic_dimensions,
            'next_index': self._next,
            'pending_next': self._pending_next,
            'records': [r.to_dict() for r in self.records],
            'pending': [p.to_dict() for p in self._pending.values()],
        }

    @classmethod
    def load(cls, payload: Mapping[str, Any]) -> 'StateIndex':
        index = cls(
            start=int(payload.get('next_index', 1) or 1),
            id_prefix=payload.get('id_prefix', 'state'),
            semantic_dimensions=int(payload.get('semantic_dimensions', SEMANTIC_DIMENSIONS)),
        )
        for item in payload.get('records') or []:
            record = StateRecord.from_dict(item)
            if record.state_id in index._records:
                raise ValueError(f'duplicate state_id {record.state_id!r} in serialized index')
            index._records[record.state_id] = record
            index._order.append(record.state_id)
        for item in payload.get('pending') or []:
            pending = PendingMerge.from_dict(item)
            index._pending[pending.pending_id] = pending
        index._pending_next = int(payload.get('pending_next', len(index._pending) + 1) or 1)
        return index


def _store_extras(record: StateRecord, *, memory: Any = None,
                  semantic_key: Optional[Sequence[float]] = None,
                  decision: Any = None,
                  store_kwargs: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """组装要透传给 store.commit 的关键字参数（只放非 None 的项）。"""
    extras: Dict[str, Any] = dict(store_kwargs or {})
    key = semantic_key if semantic_key is not None else getattr(record, 'semantic_key', None)
    if key is not None:
        extras.setdefault('semantic_key', key)
    if memory is not None:
        extras.setdefault('memory', memory)
    if decision is not None:
        extras.setdefault('decision', decision)
    return extras


def _coerce_version_id(result: Any) -> str:
    """把 store 的返回值收敛成版本号字符串；不合法就抛 TypeError。"""
    if isinstance(result, Mapping):
        result = result.get('version_id')
    if not isinstance(result, str) or not result.strip():
        raise TypeError(
            'store must return a non-empty version_id str (or {"version_id": ...}), '
            f'got {result!r}')
    return result


__all__ = [
    'ATTRIBUTE_ALIASES',
    'CARDINALITY_BUCKETS',
    'CARDINALITY_EXCLUSIVE',
    'CARDINALITY_MULTI_VALUED',
    'CARDINALITY_POLICIES',
    'attribute_cardinality',
    'canonical_value',
    'EXCLUSIVE_ATTRIBUTES',
    'IndexCommitError',
    'MULTI_VALUED_ATTRIBUTES',
    'PendingMerge',
    'SEMANTIC_DIMENSIONS',
    'StateIndex',
    'StateRecord',
    'StoreCommitError',
    'TransactionError',
    'normalize_attribute',
    'normalize_subject',
    'normalize_text',
    'utcnow',
    'value_fingerprint',
]
