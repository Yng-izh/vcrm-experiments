"""证据记忆（P0/P1）：与状态更新解耦、永远可读取的一层。

为什么必须单独存在一层（这是 LongMemEval 全量结果的直接归因）

    旧写入路径是「会话文本 → LLMFactExtractor → 原子三元组 → 只有抽出来的证据
    才编码进 latent memory」。抽取规则写死"不跨句消解指代"，所以真实对话里
    "我去年换了工作。新公司在西雅图，入职时管理六个人。" 这类必须靠上下文才成立
    的信息，压成 (subject, attribute, value) 之后关系就断了；而抽不出来的部分
    （开放域属性、DEFER 的候选值、互相冲突的表述）**根本不进可读取记忆**。
    TempLAMA 的分布（一个主体、一个属性、一条随时间更新的取值序列）掩盖了这一点。

本模块的分工：

    EvidenceMemory   保存**语义自足的证据单元**（带说话人、前后轮上下文、
                     事件时间），无论事实抽取成功与否都写入。
    StateIndex       仍在证据之上维护 (subject, attribute) → 版本的关联，
                     负责"谁是 current、谁是历史版本"。

也就是说：原子事实只负责**版本裁决**，不再决定"原始证据有没有资格进记忆"。
DEFER 仍然是一个合法的版本裁决结果（不改 current 指针），但**取消它对记忆可读性
的影响**——证据在写入时就已经进 EvidenceMemory，随时可检索。

读取顺序（P1）：query → 证据检索 → 状态关联与版本选择 → 多证据 bundle → reader。
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .state_index import normalize_attribute, normalize_subject

#: 检索用停用词：只去掉纯功能词，不动数字/年份/专名。
STOPWORDS = frozenset("""
a an the and or but if then than that this these those there here of in on at to for from by with
without into over under again further once about as is are was were be been being do does did doing
have has had having i me my we our you your he him his she her it its they them their what which who
whom when where why how all any both each few more most other some such no nor not only own same so
too very can will just should now
""".split())

_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9'_-]*")


def tokenize(text: Any) -> Tuple[str, ...]:
    """小写分词 + 去停用词；保留年份与专名（它们才是 LongMemEval 的判别信号）。"""
    return tuple(token for token in _TOKEN.findall(str(text).lower())
                 if token not in STOPWORDS and len(token) > 1)


@dataclass
class EvidenceUnit:
    """一条可独立读取的对话证据单元。"""

    evidence_id: str
    session_id: str
    time: str                       # 事件被观察到的时间（= session date），可见性锚
    text: str                       # 保上下文的证据文本（含相邻轮次）
    focus: str                      # 单元中心轮次文本（打分主体）
    speaker: str
    turn_index: int
    fact_ids: Tuple[str, ...] = ()
    state_ids: Tuple[str, ...] = ()
    subjects: Tuple[str, ...] = ()
    attributes: Tuple[str, ...] = ()
    unresolved: bool = False        # 本单元有抽不出身份/落不了状态的内容
    tokens: Tuple[str, ...] = ()
    code: Optional[Any] = None      # 压缩后的 latent code（[slots, dim]），可为 None
    memory: Optional[Any] = None    # 未压缩的 full memory（conditioned codec 时用）
    slots: int = 0

    def __post_init__(self) -> None:
        if not self.tokens:
            self.tokens = tokenize(f'{self.focus} {self.text}')

    @property
    def readable(self) -> bool:
        """证据不依赖事实抽取是否成功：只要写进来了就可读。"""
        return bool(self.text.strip())

    def to_dict(self) -> Dict[str, Any]:
        return {'evidence_id': self.evidence_id, 'session_id': self.session_id,
                'time': self.time, 'text': self.text, 'focus': self.focus,
                'speaker': self.speaker, 'turn_index': self.turn_index,
                'fact_ids': list(self.fact_ids), 'state_ids': list(self.state_ids),
                'subjects': list(self.subjects), 'attributes': list(self.attributes),
                'unresolved': self.unresolved, 'slots': self.slots}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> 'EvidenceUnit':
        return cls(evidence_id=str(payload['evidence_id']),
                   session_id=str(payload['session_id']), time=str(payload['time']),
                   text=str(payload['text']), focus=str(payload.get('focus') or ''),
                   speaker=str(payload.get('speaker') or ''),
                   turn_index=int(payload.get('turn_index') or 0),
                   fact_ids=tuple(payload.get('fact_ids') or ()),
                   state_ids=tuple(payload.get('state_ids') or ()),
                   subjects=tuple(payload.get('subjects') or ()),
                   attributes=tuple(payload.get('attributes') or ()),
                   unresolved=bool(payload.get('unresolved')),
                   slots=int(payload.get('slots') or 0))


def iter_turns(text: str) -> List[Tuple[str, str, int, int]]:
    """把会话文本切成 (speaker, utterance, char_start, char_end)，保留字符区间。

    字符区间是事实 ↔ 证据单元的**精确**联系方式：``FactExtractor.provenance``
    给出事实在原始 session 文本里的 [start, end)，靠它把事实挂到真正覆盖它的那一轮
    上，而不是"整段会话的第一条"。
    """
    raw_text = str(text)
    turns: List[Tuple[str, str, int, int]] = []
    offset = 0
    for raw in raw_text.splitlines(keepends=True):
        line_start = offset
        offset += len(raw)
        stripped = raw.strip()
        if not stripped:
            continue
        lead = len(raw) - len(raw.lstrip())
        start = line_start + lead
        match = re.match(r"^([A-Za-z][A-Za-z _'-]{0,24}):\s*(.+)$", stripped)
        if match:
            said = match.group(2).strip()
            said_start = start + stripped.index(':') + 1
            said_start = said_start + (len(stripped[stripped.index(':') + 1:])
                                       - len(stripped[stripped.index(':') + 1:].lstrip()))
            turns.append((match.group(1).strip(), said, said_start,
                          start + len(stripped)))
        elif turns:
            speaker, said, said_start, _ = turns[-1]
            turns[-1] = (speaker, f'{said} {stripped}', said_start, start + len(stripped))
        else:
            turns.append(('', stripped, start, start + len(stripped)))
    return turns or [('', raw_text.strip(), 0, len(raw_text))]


def split_turns(text: str) -> List[Tuple[str, str]]:
    """把会话文本切成 (speaker, utterance)；不认识的格式整段当一轮。"""
    return [(speaker, said) for speaker, said, _, _ in iter_turns(text)]


class EvidenceMemory:
    """证据单元仓库 + 轻量检索（纯词法打分，不引入新依赖、不占 GPU）。

    检索打分刻意不用学习到的 encoder：`longmemeval` 的查询词（年份、地点、专名）
    在词法层面就已经很强，而这一层要解决的问题是"证据有没有进记忆"，不是"检索
    模型好不好"。等真到了"证据在 bundle 里但 reader 答错"再换检索器。
    """

    def __init__(self, *, max_units_per_session: int = 12, context_turns: int = 1,
                 min_tokens: int = 2, max_units_total: Optional[int] = None,
                 write_unlinked: bool = True) -> None:
        if max_units_per_session < 1:
            raise ValueError('max_units_per_session must be >= 1')
        if context_turns < 0:
            raise ValueError('context_turns must be >= 0')
        self.max_units_per_session = int(max_units_per_session)
        self.context_turns = int(context_turns)
        self.min_tokens = int(min_tokens)
        self.max_units_total = max_units_total
        self.write_unlinked = bool(write_unlinked)
        self.units: List[EvidenceUnit] = []
        self._by_session: Dict[str, List[int]] = {}
        self._idf: Dict[str, float] = {}
        self._dirty = True
        self.dropped_units = 0

    # -- write path -------------------------------------------------------
    def write_session(self, event: Mapping[str, Any], *,
                      facts: Sequence[Any] = (),
                      unresolved: Sequence[Any] = (),
                      spans: Optional[Mapping[str, Tuple[int, int]]] = None
                      ) -> List[EvidenceUnit]:
        """把一个 session 写成证据单元；**不依赖**事实抽取是否成功。

        ``facts`` / ``unresolved`` / ``spans`` 只用于把证据与事实、状态以及
        "这里曾抽不出来"标记联系起来，不参与"要不要写"的判断。
        ``spans`` = {fact_id: (char_start, char_end)}，来自
        ``FactExtractor.provenance``；没有它时退回"文本是否出现在上下文里"。
        """
        session_id = str(event.get('id') or '')
        when = str(event.get('time') or 'unknown')
        turns = iter_turns(event.get('text') or '')
        spans = spans or {}
        units: List[EvidenceUnit] = []
        window = self.context_turns
        pending = list(facts)
        for index, (speaker, utterance, start, end) in enumerate(turns):
            tokens = tokenize(utterance)
            if len(tokens) < self.min_tokens:
                continue
            low, high = max(0, index - window), min(len(turns), index + window + 1)
            context_start = turns[low][2]
            context_end = turns[high - 1][3]
            context = str(event.get('text') or '')[context_start:context_end]
            own = [fact for fact in pending
                   if self._fact_covers(fact, spans, context, start, end)]
            pending = [fact for fact in pending if fact not in own]
            units.append(EvidenceUnit(
                evidence_id=f'{session_id}:u{index}',
                session_id=session_id, time=when, text=context or utterance,
                focus=utterance, speaker=speaker, turn_index=index,
                fact_ids=tuple(fact.id for fact in own),
                subjects=tuple(sorted({normalize_subject(fact.subject) for fact in own
                                       if getattr(fact, 'subject', '')})),
                attributes=tuple(sorted({normalize_attribute(fact.attribute) for fact in own
                                         if getattr(fact, 'attribute', '')})),
                unresolved=any(item in own for item in unresolved)))
        # 兜底：无 provenance、且事实文本没逐字出现（模型改写过），仍要保证这条
        # 事实有一条可读证据可挂（否则状态就有版本、却检索不到对应证据）。
        if pending and units:
            head = units[0]
            head.fact_ids = tuple(head.fact_ids) + tuple(fact.id for fact in pending)
            head.unresolved = head.unresolved or any(item in pending for item in unresolved)
        if not units:
            # 整段会话都切不出足够的 token（极短 utterance）：整段作为一条证据。
            text = str(event.get('text') or '').strip()
            if text:
                units = [EvidenceUnit(evidence_id=f'{session_id}:u0', session_id=session_id,
                                      time=when, text=text, focus=text, speaker='',
                                      turn_index=0,
                                      fact_ids=tuple(fact.id for fact in facts),
                                      unresolved=bool(unresolved))]
        if len(units) > self.max_units_per_session:
            # 保留信息量最大（token 最多）的若干条，其余计数丢弃，不静默。
            self.dropped_units += len(units) - self.max_units_per_session
            units = sorted(units, key=lambda unit: (-len(unit.tokens), unit.turn_index))
            units = sorted(units[:self.max_units_per_session],
                           key=lambda unit: unit.turn_index)
        if self.max_units_total is not None:
            room = self.max_units_total - len(self.units)
            if room <= 0:
                self.dropped_units += len(units)
                return []
            if len(units) > room:
                self.dropped_units += len(units) - room
                units = units[:room]
        for unit in units:
            self._by_session.setdefault(unit.session_id, []).append(len(self.units))
            self.units.append(unit)
        self._dirty = True
        return units

    @staticmethod
    def _fact_covers(fact: Any, spans: Mapping[str, Tuple[int, int]], context: str,
                     start: int, end: int) -> bool:
        """该事实是否属于这一轮：优先按 provenance 字符区间，其次按文本包含。"""
        span = spans.get(str(getattr(fact, 'id', '')))
        if span is not None:
            return span[0] < end and span[1] > start
        text_value = str(getattr(fact, 'text', '') or '')
        return bool(text_value) and text_value in context

    def attach_memories(self, units: Sequence[EvidenceUnit],
                        memories: Sequence[Any]) -> None:
        """把写路径批量编码出来的 memory 挂到证据单元上（顺序一一对应）。"""
        for unit, memory in zip(units, memories):
            if memory is None:
                continue
            try:
                unit.memory = memory.detach().cpu()
            except AttributeError:              # 非 tensor（stub reader）时原样保留
                unit.memory = memory

    def link_state(self, fact_id: str, state_id: Optional[str]) -> None:
        """把事实 → 状态的联系回填到证据单元上（DEFER 时 state_id 为 None）。"""
        if not fact_id:
            return
        normalised = None if state_id is None else str(state_id)
        for unit in self.units:
            if fact_id not in unit.fact_ids:
                continue
            if normalised and normalised not in unit.state_ids:
                unit.state_ids = unit.state_ids + (normalised,)
            if normalised is None:
                unit.unresolved = True

    def link_receipts(self, receipts: Iterable[Mapping[str, Any]]) -> None:
        for receipt in receipts or ():
            self.link_state(str(receipt.get('fact_id') or ''),
                            receipt.get('predicted_state_id'))

    def compress(self, codec: Any, *, anchor_of=None) -> int:
        """把证据的 full memory 压成 code（16-slot 量级），返回压缩条数。

        ``anchor_of(state_id)`` 给出条件化 codec 需要的 anchor 记忆；E3（独立
        codec）不需要 anchor。压缩失败只跳过该条（保留 text，证据仍然可读）。
        """
        import torch

        compressed = 0
        for unit in self.units:
            if unit.memory is None or unit.code is not None:
                continue
            anchor = None
            if getattr(codec, 'conditioned', False):
                state_id = unit.state_ids[0] if unit.state_ids else None
                anchor = anchor_of(state_id) if (anchor_of and state_id) else None
                if anchor is None:
                    continue        # 条件化 codec 无 anchor 时宁可不压，不猜
            memory = unit.memory
            if not hasattr(memory, 'ndim') or memory.ndim != 2:
                continue
            try:
                with torch.no_grad():
                    code = codec.encode(memory, None if anchor is None else anchor)
            except (TypeError, ValueError, RuntimeError):
                continue
            code = code.detach().cpu()
            unit.code = code
            unit.memory = None
            unit.slots = int(code.shape[0])
            compressed += 1
        return compressed

    # -- read path --------------------------------------------------------
    def _refresh_idf(self) -> None:
        if not self._dirty:
            return
        document_frequency: Dict[str, int] = {}
        for unit in self.units:
            for token in set(unit.tokens):
                document_frequency[token] = document_frequency.get(token, 0) + 1
        total = max(1, len(self.units))
        self._idf = {token: math.log((total + 1) / (count + 0.5))
                     for token, count in document_frequency.items()}
        self._dirty = False

    def retrieve(self, question: str, *, k: int = 4,
                 subject: Optional[str] = None, attribute: Optional[str] = None,
                 observed_until: Optional[str] = None,
                 allowed_sessions: Optional[Sequence[str]] = None
                 ) -> List[Tuple[float, EvidenceUnit]]:
        """证据优先检索：返回 [(score, unit)]，按分数降序。

        ``observed_until`` 是硬上界（可见性）：晚于它的证据一律不返回。
        ``allowed_sessions`` 用于"只在给定 session 里找"，None = 全部。
        """
        if k < 1 or not self.units:
            return []
        self._refresh_idf()
        query_tokens = tokenize(question)
        wanted_subject = normalize_subject(subject) if subject else ''
        wanted_attribute = normalize_attribute(attribute) if attribute else ''
        allowed = None if allowed_sessions is None else set(map(str, allowed_sessions))
        scored: List[Tuple[float, EvidenceUnit]] = []
        for unit in self.units:
            if observed_until is not None and unit.time > str(observed_until):
                continue
            if allowed is not None and unit.session_id not in allowed:
                continue
            overlap = 0.0
            for token in query_tokens:
                if token in unit.tokens:
                    overlap += self._idf.get(token, 1.0)
            if overlap <= 0.0 and not (wanted_subject or wanted_attribute):
                continue
            length_penalty = 1.0 / math.sqrt(1.0 + len(unit.tokens) / 16.0)
            score = overlap * length_penalty
            if wanted_subject and wanted_subject in unit.subjects:
                score += 1.5
            if wanted_attribute and wanted_attribute in unit.attributes:
                score += 1.0
            if unit.state_ids:
                score += 0.25          # 已能挂上状态的证据优先（可叠版本信息）
            if score <= 0.0:
                continue
            scored.append((score, unit))
        scored.sort(key=lambda pair: (-pair[0], pair[1].time, pair[1].turn_index))
        return scored[:k]

    # -- helpers ----------------------------------------------------------
    def state_ids(self, units: Sequence[EvidenceUnit]) -> List[str]:
        seen: Dict[str, None] = {}
        for unit in units:
            for state_id in unit.state_ids:
                seen.setdefault(state_id, None)
        return list(seen)

    def session_ids(self, units: Sequence[EvidenceUnit]) -> List[str]:
        seen: Dict[str, None] = {}
        for unit in units:
            seen.setdefault(unit.session_id, None)
        return list(seen)

    def readable_text(self, units: Sequence[EvidenceUnit]) -> str:
        """B 组（state_text）读取用：把证据单元拼成带日期/说话人的文本块。"""
        blocks, seen = [], set()
        for unit in units:
            if unit.evidence_id in seen:
                continue
            seen.add(unit.evidence_id)
            blocks.append(f"DATE: {unit.time}\n{unit.text}")
        return '\n\n'.join(blocks)

    def stats(self) -> Dict[str, Any]:
        linked = sum(1 for unit in self.units if unit.state_ids)
        return {'evidence_units': len(self.units),
                'evidence_sessions': len(self._by_session),
                'evidence_linked_units': linked,
                'evidence_unlinked_units': len(self.units) - linked,
                'evidence_unresolved_units': sum(1 for unit in self.units if unit.unresolved),
                'evidence_compressed_units': sum(1 for unit in self.units if unit.code is not None),
                'evidence_dropped_units': self.dropped_units}

    def serialize(self) -> Dict[str, Any]:
        return {'max_units_per_session': self.max_units_per_session,
                'context_turns': self.context_turns,
                'dropped_units': self.dropped_units,
                'units': [unit.to_dict() for unit in self.units]}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> 'EvidenceMemory':
        memory = cls(max_units_per_session=int(payload.get('max_units_per_session') or 12),
                     context_turns=int(payload.get('context_turns') or 1))
        memory.dropped_units = int(payload.get('dropped_units') or 0)
        for item in payload.get('units') or ():
            unit = EvidenceUnit.from_dict(item)
            memory._by_session.setdefault(unit.session_id, []).append(len(memory.units))
            memory.units.append(unit)
        memory._dirty = True
        return memory


__all__ = ['EvidenceMemory', 'EvidenceUnit', 'split_turns', 'tokenize', 'STOPWORDS']
