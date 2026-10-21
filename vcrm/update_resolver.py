from __future__ import annotations
# P2: Oracle-free update resolution — 六条确定性规则。
#
#   rule1  无候选                          -> CREATE
#   rule2  候选主体全部不同（拒绝合并）      -> CREATE
#   rule3  同主体但不同属性                  -> CREATE
#   rule4  同主体同属性且值等价              -> DUPLICATE
#   rule5  同主体同属性 + 单值排他属性 + 显式「替代」表达 -> UPDATE
#   rule5b 同主体同属性 + 单值排他属性 + 新事实观察时间严格更晚 -> UPDATE
#          （**显式策略开关** recency_update，默认 False；打开后才生效）
#   rule5c 同主体同属性 + time-scoped 属性（SNAPSHOT_SUPERSEDES）+ 新事实是
#          **带年份前缀的年度快照**且时间严格更晚 -> UPDATE
#          （**显式策略开关** snapshot_update，默认 True；只对快照句生效，
#           日常陈述不受影响，累积型多值属性如 children/award_received 永远 DEFER）
#   rule6  多值 / 否定 / 冲突 / 不确定       -> DEFER（存独立待确认记录，不改 Current）
#
# rule5 与 rule5b 的 reason 前缀分开：'explicit_replacement:' 与
# 'exclusive_superseded_by_recency:'，评测器可以按口径统计 UPDATE 的来源。
# recency_update 默认关闭：逐年陈述句本身不构成「替代」证据，打开它等于接受
# 「后出现的陈述覆盖先出现的陈述」这一策略假设。
#
# 排他性 = 属性基数本体表（vcrm.state_index.CARDINALITY_EXCLUSIVE）∩ 文本里的显式替代
# 表达（rule5）或观察时间更晚（rule5b），**不是**余弦阈值：余弦只用来排序候选，
# 不参与 CREATE/UPDATE/DUPLICATE 的判定。
#
# 基数三档由 cardinality_policy 决定：'ontology'（默认，用本体表）/
# 'unknown_only'（严格基线：全部属性当 UNKNOWN，全部 DEFER，用于消融）。
# 每个决策的 reason 尾部都带 `|cardinality=<EXCLUSIVE|MULTI_VALUED|UNKNOWN>`，
# 逐样本可查，便于评测按口径统计。
#
# 整个模块不读 state_key / key_text / entity / group / answers / latest / episode。

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from .fact_schema import AtomicFact, StateDecision
from .state_index import (
    CARDINALITY_POLICIES,
    EXCLUSIVE_ATTRIBUTES,
    MULTI_VALUED_ATTRIBUTES,
    SNAPSHOT_SUPERSEDES,
    IndexCommitError,
    StateIndex,
    StateRecord,
    StoreCommitError,
    _coerce_version_id,
    attribute_cardinality,
    canonical_value,
    normalize_attribute,
    normalize_subject,
    normalize_text,
    value_fingerprint,
)

#: 显式「替代」表达：只有出现这些词才允许覆盖一个单值排他属性的旧值。
REPLACEMENT_MARKERS = re.compile(
    r'\b(?:now|instead|replaces?|replaced|succeeds?|succeeded|formerly|previously|'
    r'changed\s+(?:to|from)|switched\s+to|updated\s+to|steps?\s+down|stepped\s+down|'
    r'resigned|took\s+over|is\s+the\s+new|became|becomes)\b', re.I)

#: 年度快照语义：事实文本以显式时间前缀开头（"In 2016: ..." / "As of 2016, ..." / "2016: ..."）。
#: rule5c 只对这类输入生效——它表达的是"2016 年的快照"而不是"日常陈述"。
SNAPSHOT_PREFIX = re.compile(
    r'^\s*(?:(?:in|during|as\s+of|back\s+in|on|by|for)\s+(?:19|20)\d{2}\b'
    r'|(?:19|20)\d{2}\s*[:,])', re.I)

#: 否定/撤回表达：一律 DEFER，不做覆盖也不做合并。
NEGATION_MARKERS = re.compile(
    r"(?:\bnot\b|\bnever\b|\bno\s+longer\b|\bwithout\b|\bdenies\b|\bdenied\b|"
    r"\bisn't\b|\baren't\b|\bwasn't\b|\bweren't\b|\bdoesn't\b|\bdidn't\b|"
    r"\bfalse\b|\brumou?r\b|\bincorrect\b|\bwrong\b)", re.I)

UNRESOLVED_SUBJECTS = frozenset({'<unresolved:subject>'})
UNRESOLVED_ATTRIBUTES = frozenset({'<unresolved:attribute>', 'unknown_attribute'})

#: 规则标签（供日志/测试断言使用）。默认口径 = 六条（rule1..rule4 + rule5 + rule6），
#: rule5b 是默认关闭的第七条。
RULES: Dict[str, str] = {
    'rule1': 'no_candidate_create',
    'rule2': 'subject_mismatch_no_merge',
    'rule3': 'same_subject_different_attribute_create',
    'rule4': 'value_equivalent_duplicate',
    'rule5': 'exclusive_attribute_explicit_replacement_update',
    'rule5b': 'exclusive_superseded_by_recency_update',
    'rule5c': 'dated_snapshot_supersedes_time_scoped_attribute',
    'rule6': 'defer_pending_review',
}


def parse_timestamp(value: Any) -> Optional[datetime]:
    """解析 observed_at；无法解析（含 'unknown'/空串/非 ISO）时返回 None。"""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if re.fullmatch(r'(?:19|20)\d{2}', text):
        return datetime(int(text), 1, 1)
    candidate = text.replace('Z', '+00:00')
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.replace(tzinfo=None)
    return parsed


def _fact_field(fact: Any, name: str, default: str = '') -> str:
    if isinstance(fact, dict):
        return str(fact.get(name, default) or default)
    return str(getattr(fact, name, default) or default)


def _value_atoms(value: Any) -> set:
    """把取值拆成原子（逗号/斜杠/分号分隔），用于宽松等价判断。"""
    text = canonical_value(value)
    atoms = {canonical_value(part) for part in re.split(r'[,/;]|\band\b', text)}
    return {atom for atom in atoms if atom}


def values_equivalent(left: Any, right: Any) -> bool:
    """值等价判定：规范形一致，或数字相同，或原子集合有交集。"""
    left_canonical, right_canonical = canonical_value(left), canonical_value(right)
    if not left_canonical or not right_canonical:
        return False
    if left_canonical == right_canonical:
        return True
    if left_canonical.replace(' ', '').isdigit() and right_canonical.replace(' ', '').isdigit():
        return int(left_canonical.replace(' ', '')) == int(right_canonical.replace(' ', ''))
    atoms_left, atoms_right = _value_atoms(left), _value_atoms(right)
    return bool(atoms_left & atoms_right)


class UpdateResolver:
    """六条规则的确定性实现（无训练、无阈值调参）。"""

    def __init__(self, *,
                 exclusive_attributes: Optional[Sequence[str]] = None,
                 multi_valued_attributes: Optional[Sequence[str]] = None,
                 replacement_markers: Optional[re.Pattern] = None,
                 negation_markers: Optional[re.Pattern] = None,
                 create_confidence: float = 0.9,
                 duplicate_confidence: float = 0.95,
                 update_confidence: float = 0.8,
                 defer_confidence: float = 0.3,
                 cardinality_policy: str = 'ontology',
                 recency_update: bool = False,
                 recency_confidence_floor: float = 0.5,
                 recency_confidence_ceiling: float = 0.85,
                 recency_gap_days_full: float = 365.0,
                 recency_min_confidence: float = 0.0,
                 snapshot_update: bool = True) -> None:
        if cardinality_policy not in CARDINALITY_POLICIES:
            raise ValueError(
                f'unknown cardinality_policy {cardinality_policy!r}; expected {CARDINALITY_POLICIES}')
        self.cardinality_policy = cardinality_policy
        self.exclusive_attributes = frozenset(
            EXCLUSIVE_ATTRIBUTES if exclusive_attributes is None else exclusive_attributes)
        self.multi_valued_attributes = frozenset(
            MULTI_VALUED_ATTRIBUTES if multi_valued_attributes is None else multi_valued_attributes)
        #: 逐样本审计：(attribute_key, bucket, action)。供评测按基数口径统计。
        self.audit: List[Dict[str, str]] = []
        self.replacement_markers = replacement_markers or REPLACEMENT_MARKERS
        self.negation_markers = negation_markers or NEGATION_MARKERS
        self.create_confidence = float(create_confidence)
        self.duplicate_confidence = float(duplicate_confidence)
        self.update_confidence = float(update_confidence)
        self.defer_confidence = float(defer_confidence)
        # rule5b 显式策略开关：默认关闭，不改变任何现有保守语义
        self.recency_update = bool(recency_update)
        self.recency_confidence_floor = float(recency_confidence_floor)
        self.recency_confidence_ceiling = float(recency_confidence_ceiling)
        self.recency_gap_days_full = float(recency_gap_days_full)
        self.recency_min_confidence = float(recency_min_confidence)
        # rule5c 显式策略开关：只对"带年份前缀的年度快照"生效，默认开启；
        # 置 False 即回到"非排他属性一律 DEFER"的旧行为。
        self.snapshot_update = bool(snapshot_update)

    # -- 判定辅助 ---------------------------------------------------------
    @staticmethod
    def _is_unresolved(fact: Any) -> bool:
        subject = _fact_field(fact, 'subject')
        attribute = normalize_attribute(_fact_field(fact, 'attribute'))
        value = _fact_field(fact, 'value')
        return (subject in UNRESOLVED_SUBJECTS or attribute in UNRESOLVED_ATTRIBUTES
                or not normalize_text(value))

    @staticmethod
    def _same_subject(fact: Any, record: StateRecord) -> bool:
        subject = _fact_field(fact, 'subject')
        if subject in UNRESOLVED_SUBJECTS:
            return False
        return normalize_subject(subject) == record.subject_key or record.matches_subject(subject)

    @staticmethod
    def _same_attribute(fact: Any, record: StateRecord) -> bool:
        attribute = _fact_field(fact, 'attribute')
        if normalize_attribute(attribute) in UNRESOLVED_ATTRIBUTES:
            return False
        return (normalize_attribute(attribute) == record.attribute_key
                or record.matches_attribute(attribute))

    def cardinality_of(self, attribute: Any) -> str:
        """属性落哪一档：EXCLUSIVE / MULTI_VALUED / UNKNOWN（受 cardinality_policy 控制）。"""
        key = normalize_attribute(attribute)
        if self.cardinality_policy == 'unknown_only':
            return 'UNKNOWN'
        if key in self.multi_valued_attributes:
            return 'MULTI_VALUED'
        if key in self.exclusive_attributes:
            return 'EXCLUSIVE'
        return attribute_cardinality(key, 'ontology')

    def cardinality_summary(self) -> Dict[str, int]:
        """审计汇总：{bucket: 决策次数}，供评测分口径。"""
        summary = {bucket: 0 for bucket in ('EXCLUSIVE', 'MULTI_VALUED', 'UNKNOWN')}
        for item in self.audit:
            summary[item['bucket']] = summary.get(item['bucket'], 0) + 1
        return summary

    def _record_audit(self, attribute: Any, bucket: str, action: str) -> None:
        self.audit.append({'attribute': normalize_attribute(attribute),
                           'bucket': bucket, 'action': action})

    def _negated(self, fact: Any) -> bool:
        return bool(self.negation_markers.search(_fact_field(fact, 'text')))

    def _replacement_explicit(self, fact: Any) -> bool:
        return bool(self.replacement_markers.search(_fact_field(fact, 'text')))

    def _recency_confidence(self, gap_days: float) -> float:
        """可校准的 rule5b 置信度：底分 + 时间间隔的饱和增益（值不等价已由 rule4 保证）。

        floor / ceiling / gap_days_full / min_confidence 都是构造参数，
        正式实验前必须在独立验证集上校准门槛，不得硬编码。
        """
        span = max(self.recency_gap_days_full, 1.0)
        ratio = min(max(gap_days, 0.0) / span, 1.0)
        span_confidence = (self.recency_confidence_floor
                           + (self.recency_confidence_ceiling - self.recency_confidence_floor) * ratio)
        return max(0.0, min(span_confidence, 1.0))

    def _recency_update(self, fact: Any, target: StateRecord,
                        attribute: str) -> Optional[StateDecision]:
        """rule5b：单值排他属性 + 新事实观察时间严格更晚 + 值不等价 ⇒ 时间性覆盖。

        返回 None 表示「本条不适用」，由调用方落 DEFER（不猜）。
        """
        if not self.recency_update:
            return None
        bucket = self.cardinality_of(attribute)
        if bucket == 'MULTI_VALUED':                       # 多值属性永不触发
            return None
        if bucket != 'EXCLUSIVE':                          # 排他性未知不触发
            return None
        new_time = parse_timestamp(_fact_field(fact, 'observed_at'))
        old_time = parse_timestamp(getattr(target, 'current_observed_at', ''))
        if new_time is None or old_time is None:           # 不可解析 → 不猜
            return None
        if new_time <= old_time:                           # 相等或更早 → 不猜
            return None
        confidence = self._recency_confidence((new_time - old_time).total_seconds() / 86400.0)
        if confidence < self.recency_min_confidence:       # 校准门槛未过 → 不猜
            return None
        return StateDecision('UPDATE', target.state_id, confidence,
                             f'exclusive_superseded_by_recency:{attribute}|cardinality={bucket}')

    @staticmethod
    def _is_dated_snapshot(fact: Any) -> bool:
        """输入是否是一条"年度快照"（文本以显式年份前缀开头）。"""
        return bool(SNAPSHOT_PREFIX.match(_fact_field(fact, 'text')))

    def _snapshot_update(self, fact: Any, target: StateRecord,
                         attribute: str, bucket: str) -> Optional[StateDecision]:
        """rule5c：年度快照 + time-scoped 属性 + 新快照严格更晚 ⇒ 覆盖旧快照。

        只处理 rule5/rule5b 覆盖不到的属性（SNAPSHOT_SUPERSEDES，见 state_index），
        因此排他属性上的口径仍由 explicit_replacement / recency 统计，不混算。
        返回 None 表示「本条不适用」，由调用方按原规则继续（通常是 DEFER）。
        """
        if not self.snapshot_update:
            return None
        if attribute not in SNAPSHOT_SUPERSEDES:
            return None
        if not self._is_dated_snapshot(fact):              # 日常陈述不按快照处理
            return None
        new_time = parse_timestamp(_fact_field(fact, 'observed_at'))
        old_time = parse_timestamp(getattr(target, 'current_observed_at', ''))
        if new_time is None or old_time is None:           # 不可解析 → 不猜
            return None
        if new_time <= old_time:                           # 相等或更早 → 不猜
            return None
        confidence = self._recency_confidence((new_time - old_time).total_seconds() / 86400.0)
        if confidence < self.recency_min_confidence:       # 校准门槛未过 → 不猜
            return None
        return StateDecision('UPDATE', target.state_id, confidence,
                             f'snapshot_superseded:{attribute}|cardinality={bucket}')

    # -- 主入口 -----------------------------------------------------------
    def resolve(self, new_fact: Any, candidates: Sequence[StateRecord],
                state_index: StateIndex) -> StateDecision:
        """六条规则 + 可选的 rule5b，并对每条决策记录属性基数口径。"""
        decision = self._resolve(new_fact, candidates, state_index)
        attribute = _fact_field(new_fact, 'attribute')
        self._record_audit(attribute, self.cardinality_of(attribute), decision.action)
        return decision

    def _resolve(self, new_fact: Any, candidates: Sequence[StateRecord],
                 state_index: StateIndex) -> StateDecision:
        candidates = list(candidates or [])

        if self._is_unresolved(new_fact):
            return self._defer(new_fact, [], state_index,
                               'rule6:unresolved_identity_or_value', confidence=0.2)

        # rule1：没有任何候选 → 新状态
        if not candidates:
            return StateDecision('CREATE', None, self.create_confidence,
                                 'rule1:no_candidate')

        # rule2：候选主体全部不同 → 拒绝合并，建新状态
        same_subject = [c for c in candidates if self._same_subject(new_fact, c)]
        if not same_subject:
            return StateDecision('CREATE', None, self.create_confidence,
                                 'rule2:subject_mismatch_no_merge')

        # rule3：同主体、属性都不同 → 建新状态（错误合并反例的落点）
        same_identity = [c for c in same_subject if self._same_attribute(new_fact, c)]
        if not same_identity:
            return StateDecision('CREATE', None, self.create_confidence,
                                 'rule3:same_subject_different_attribute')

        # rule6（冲突）：同一身份下多个候选且取值指纹不一致 → 不猜
        fingerprints = {c.value_fingerprint for c in same_identity}
        if len(same_identity) > 1 and len(fingerprints) > 1:
            return self._defer(new_fact, same_identity, state_index,
                               'rule6:conflicting_candidates')

        target = same_identity[0]

        # rule6（否定）：否定/撤回表达一律 DEFER，即使值看起来等价
        if self._negated(new_fact):
            return self._defer(new_fact, [target], state_index, 'rule6:negation')

        # rule4：值等价 → DUPLICATE（不产生新版本）
        # 索引只存取值指纹（规范形 sha1），因此等价判定 = 指纹一致。
        new_fingerprint = value_fingerprint(_fact_field(new_fact, 'value'))
        if target.value_fingerprint and new_fingerprint == target.value_fingerprint:
            return StateDecision('DUPLICATE', target.state_id,
                                 self.duplicate_confidence,
                                 'rule4:value_equivalent')

        attribute = normalize_attribute(_fact_field(new_fact, 'attribute'))
        bucket = self.cardinality_of(attribute)

        # rule5c（年度快照覆盖）：带年份前缀的快照 + time-scoped 属性 + 严格更晚 → UPDATE。
        # 放在基数门槛之前，因为成员关系这类属性本体上是 MULTI_VALUED，但在"逐年快照"
        # 语义下同一年只能有一个取值；累积型多值不在 SNAPSHOT_SUPERSEDES 里，仍然 DEFER。
        if attribute in SNAPSHOT_SUPERSEDES:
            snapshot_decision = self._snapshot_update(new_fact, target, attribute, bucket)
            if snapshot_decision is not None:
                return snapshot_decision

        # rule6（多值属性）：第二个不同取值不能覆盖
        if bucket == 'MULTI_VALUED':
            return self._defer(new_fact, [target], state_index,
                               f'rule6:multi_valued_attribute|cardinality={bucket}')

        # rule6（基数未知）：不在本体表里的属性不擅自覆盖（严格基线全部走这里）
        if bucket != 'EXCLUSIVE':
            return self._defer(new_fact, [target], state_index,
                               f'rule6:cardinality_unknown|cardinality={bucket}')

        # rule5：单值排他属性 + 显式替代表达 → UPDATE（口径：explicit_replacement）
        if self._replacement_explicit(new_fact):
            return StateDecision('UPDATE', target.state_id, self.update_confidence,
                                 f'explicit_replacement:{attribute}|cardinality={bucket}')

        # rule5b：单值排他属性 + 观察时间严格更晚 → UPDATE
        # （显式策略开关 recency_update，默认关闭）
        if self.recency_update:
            recency_decision = self._recency_update(new_fact, target, attribute)
            if recency_decision is not None:
                return recency_decision
            return self._defer(new_fact, [target], state_index,
                               f'rule6:recency_not_applicable|cardinality={bucket}')

        # rule6（无显式替代表达）：不按相似度硬覆盖
        return self._defer(new_fact, [target], state_index,
                           f'rule6:no_explicit_replacement|cardinality={bucket}')

    def _defer(self, fact: Any, targets: Sequence[StateRecord],
               state_index: StateIndex, reason: str,
               confidence: Optional[float] = None) -> StateDecision:
        candidate_id = targets[0].state_id if targets else None
        pending = state_index.add_pending(
            fact, candidate_state_id=candidate_id, reason=reason,
            confidence=self.defer_confidence if confidence is None else confidence)
        return StateDecision(
            'DEFER', None,
            self.defer_confidence if confidence is None else confidence,
            f'{reason}|pending={pending.pending_id}')


def resolve(new_fact: Any, candidates: Sequence[StateRecord], state_index: StateIndex,
            resolver: Optional[UpdateResolver] = None) -> StateDecision:
    """模块级便捷入口：默认规则集的 resolve(new_fact, candidates, state_index)。"""
    return (resolver or UpdateResolver()).resolve(new_fact, candidates, state_index)


def _store_decision(decision: StateDecision, state_id: str) -> StateDecision:
    """store 适配层要求 decision.state_id == record.state_id。

    CREATE/DEFER 时 state_id 由系统在写入时分配（解析层的 StateDecision 里是 None），
    这里生成一份 state_id 对齐后的副本，动作/置信度/理由原样保留。
    """
    if decision.state_id == state_id:
        return decision
    return StateDecision(decision.action, state_id, decision.confidence, decision.reason)


def _defer_store_record(decision: StateDecision, fact: Any, state_index: StateIndex,
                        semantic_key: Optional[Sequence[float]]) -> StateRecord:
    """DEFER 写 store 时需要一条 record 承载 state_id（pending 记录要保留原始 full memory）。

    优先用决策/候选状态的 state_id；都没有时用一条**不插入索引**的预备记录，
    避免为一条未确认的事实建出哨兵状态。
    """
    state_id = decision.state_id
    if not state_id:
        fact_id = _fact_field(fact, 'id')
        for pending in reversed(state_index.pending):
            if pending.fact_id == fact_id and pending.candidate_state_id:
                state_id = pending.candidate_state_id
                break
    if state_id and state_id in state_index:
        return state_index.get(state_id)
    try:
        return state_index.build_record(fact, semantic_key=semantic_key,
                                        state_id=state_id or None)
    except ValueError:
        return StateRecord(state_id=state_id or state_index.new_state_id(),
                           subject_key=normalize_subject(_fact_field(fact, 'subject')),
                           attribute_key=normalize_attribute(_fact_field(fact, 'attribute')),
                           semantic_key=None if semantic_key is None else tuple(
                               float(x) for x in semantic_key),
                           current_version_id='',
                           current_observed_at=_fact_field(fact, 'observed_at'))


def apply_decision(decision: StateDecision, fact: Any, state_index: StateIndex,
                   store: Any = None, *, memory: Any = None,
                   semantic_key: Optional[Sequence[float]] = None,
                   store_kwargs: Optional[Dict[str, Any]] = None,
                   now: Optional[str] = None) -> Optional[StateRecord]:
    """把决策落到状态层（唯一写入点）。

    四条动作都会写 store（store is not None 时）：
      CREATE     store 先落地（provisional record），成功后再插索引；
                 store 失败 → 索引不变；索引插入失败 → 回滚 store
      UPDATE     走 commit_with 事务钩子（store 成功才推进索引指针）
      DUPLICATE  只写 store 的 alias 元数据（duplicates），版本与索引都不变
      DEFER      写 store 的 pending（保留原始 full memory），Current 与索引都不变

    真实 IncrementalCurrentStore 的 commit() 对**四种动作都强制要求** memory=
    （原始 full memory）与 semantic_key=，因此这里必须全部透传：
    memory / semantic_key 会原样交给
    store.commit(record=..., fact=..., memory=..., semantic_key=..., decision=...)。
    store_kwargs 用于给具体 store 追加额外关键字参数。
    """
    if not isinstance(decision, StateDecision):
        raise TypeError(f'apply_decision expects StateDecision, got {type(decision).__name__}')
    fact_id = _fact_field(fact, 'id')
    fingerprint = value_fingerprint(_fact_field(fact, 'value'))
    observed_at = _fact_field(fact, 'observed_at') or None
    extras = dict(store_kwargs or {})

    def store_payload(state_id: Optional[str] = None) -> Dict[str, Any]:
        payload = dict(extras)
        if semantic_key is not None:
            payload.setdefault('semantic_key', semantic_key)
        if memory is not None:
            payload.setdefault('memory', memory)
        payload.setdefault(
            'decision',
            decision if not state_id else _store_decision(decision, state_id))
        return payload

    # ---- 无 store：只维护索引（最小路径，测试/离线用）----
    if store is None:
        if decision.action == 'CREATE':
            return state_index.add(fact, semantic_key=semantic_key,
                                   version_id=fact_id or None, now=now)
        if decision.action == 'DUPLICATE':
            return state_index.get(decision.state_id)
        if decision.action == 'UPDATE':
            record = state_index.get(decision.state_id)
            state_index.commit(record.state_id, fact_id or record.current_version_id,
                               value_fingerprint=fingerprint,
                               observed_at=observed_at, now=now)
            return record
        return None

    # ---- CREATE：store 先落地，索引后落地 ----
    if decision.action == 'CREATE':
        provisional = state_index.build_record(fact, semantic_key=semantic_key)
        payload = store_payload(provisional.state_id)
        try:
            result = store.commit(record=provisional, fact=fact, **payload)
        except Exception as exc:  # noqa: BLE001 - store 失败时索引必须保持原样
            raise StoreCommitError(
                f'current store CREATE failed for {provisional.state_id}: {exc}') from exc
        try:
            version_id = _coerce_version_id(result)
            return state_index.add(fact, semantic_key=semantic_key,
                                   state_id=provisional.state_id,
                                   version_id=version_id, now=now)
        except Exception as exc:  # noqa: BLE001 - 索引失败时回滚 store
            note = StateIndex._rollback_store(store, result, None)
            raise IndexCommitError(
                f'index CREATE failed after store commit for {provisional.state_id}: {exc}; {note}'
            ) from exc

    # ---- UPDATE：事务钩子（store 成功才更新索引）----
    if decision.action == 'UPDATE':
        record = state_index.get(decision.state_id)
        state_index.commit_with(store, record.state_id, fact,
                                value_fingerprint=fingerprint,
                                observed_at=observed_at,
                                memory=memory, semantic_key=semantic_key,
                                decision=decision, store_kwargs=extras, now=now)
        return record

    # ---- DUPLICATE：只写 alias 元数据，版本/索引不变 ----
    if decision.action == 'DUPLICATE':
        record = state_index.get(decision.state_id)
        try:
            store.commit(record=record, fact=fact, **store_payload(record.state_id))
        except Exception as exc:  # noqa: BLE001
            raise StoreCommitError(
                f'current store DUPLICATE failed for {record.state_id}: {exc}') from exc
        return record

    # ---- DEFER：写 pending，Current 与索引都不变 ----
    record = _defer_store_record(decision, fact, state_index, semantic_key)
    try:
        store.commit(record=record, fact=fact, **store_payload(record.state_id))
    except Exception as exc:  # noqa: BLE001
        raise StoreCommitError(
            f'current store DEFER failed for {record.state_id}: {exc}') from exc
    return None


__all__ = [
    'NEGATION_MARKERS',
    'REPLACEMENT_MARKERS',
    'RULES',
    'SNAPSHOT_PREFIX',
    'UNRESOLVED_ATTRIBUTES',
    'UNRESOLVED_SUBJECTS',
    'UpdateResolver',
    'apply_decision',
    'parse_timestamp',
    'resolve',
    'values_equivalent',
]
