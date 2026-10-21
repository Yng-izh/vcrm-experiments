from __future__ import annotations
# P2: Oracle-free fact extraction.
#
# 输入只允许 P0 的公开写视图 {'id', 'text', 'time'}（vcrm.data_views.public_write_event）。
# 抽取是**确定性规则**：主体识别 + 模板/依存式属性规范化 + 取值抽取 + 时间解析。
# 属性一律由自然语言表面形式归一化得到（vcrm.state_index.normalize_attribute），
# 绝不使用数据集 relation id 映射，也不读 state_key / key_text / entity / group /
# answers / latest / episode。
#
# TempLAMA 单事实优先：一条事件通常只有一句「In <year>: <attribute> of <subject> is <value>」。
# LongMemEval 多事实：先按行（session turn）切段，再按句子/子句切分，用
# reference_resolver 钩子做代词消解，并把来源 session / turn / 时间 / 文本区间
# 记进 FactProvenance。
#
# 抽不出 subject / attribute 时**不猜合并**：产出一条带哨兵键的低置信度事实，
# 放进 extractor 自己的 unresolved 独立记忆，并把失败原因写进诊断；
# 后续 UpdateResolver 会把这类事实判成 DEFER。

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .data_views import PUBLIC_WRITE_KEYS, assert_no_oracle, assert_public_id_independent
from .fact_schema import AtomicFact
from .state_index import normalize_attribute, normalize_subject, normalize_text

#: 抽不出主体/属性时的哨兵键（非空，但语义上明确是「未知」）。
UNRESOLVED_SUBJECT = '<unresolved:subject>'
UNRESOLVED_ATTRIBUTE = '<unresolved:attribute>'
#: 时间无法确定时的占位（AtomicFact.observed_at 必须非空）。
UNKNOWN_TIME = 'unknown'

PRONOUNS = frozenset({
    'i', 'we', 'you', 'he', 'she', 'they', 'it', 'him', 'her', 'them', 'us',
    'his', 'hers', 'their', 'its', 'my', 'our', 'your',
    'this', 'that', 'these', 'those',
})

_MONTHS = {
    'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
    'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12,
}

_WS = re.compile(r'\s+')
_LINE = re.compile(r'[^\n]+')
_ROLE_PREFIX = re.compile(r'^\s*(?P<role>[A-Za-z][A-Za-z ._-]{0,23}?)\s*:\s*(?P<content>.+)$')
_SENTENCE = re.compile(r'[^.!?]+[.!?]*')
_TIME_PREFIX = re.compile(
    r'^\s*(?:in|during|as of|back in|on)\s+(?P<stamp>[^:,;]{2,32}?)\s*[:,]\s*', re.I)
_COPULA = re.compile(
    r'\s+(?:is|was|are|were|became|becomes|has been|have been|had been)\s+', re.I)
_POSSESSIVE = re.compile(r"^(?P<subject>.+?)['\u2019]s\s+(?P<attribute>.+)$")
_INVERSE = re.compile(r'^(?:the\s+)?(?P<attribute>[a-z][a-z \-]*?)\s+of\s+(?P<owner>.+)$', re.I)
_POSSESSIVE_PRONOUN = re.compile(
    r'^(?P<pronoun>his|her|hers|their|theirs|its|my|mine|our|ours|your|yours)\s+(?P<attribute>.+)$',
    re.I)

#: 动词模板 → 规范属性名（模板式属性规范化，与 relation id 无关）。
_VERB_TEMPLATES: Tuple[Tuple[re.Pattern, str], ...] = (
    (re.compile(r'^(?P<subject>.+?)\s+(?:works at|works for|is employed by|joined)\s+(?P<value>.+)$', re.I), 'employer'),
    (re.compile(r'^(?P<subject>.+?)\s+(?:lives in|moved to|is based in|relocated to)\s+(?P<value>.+)$', re.I), 'residence'),
    (re.compile(r'^(?P<subject>.+?)\s+(?:is married to|got married to|married)\s+(?P<value>.+)$', re.I), 'spouse'),
    (re.compile(r'^(?P<subject>.+?)\s+(?:was born in|is from|comes from)\s+(?P<value>.+)$', re.I), 'place_of_birth'),
    (re.compile(r'^(?P<subject>.+?)\s+(?:studied at|graduated from)\s+(?P<value>.+)$', re.I), 'alma_mater'),
)


#: 省略主语的动词短语模板（接续前一句的主体）。
_LEADING_VERB_TEMPLATES: Tuple[Tuple[re.Pattern, str], ...] = (
    (re.compile(r'^(?:works at|works for|lives in|moved to|is based in|relocated to)\s+(?P<value>.+)$', re.I), 'residence'),
    (re.compile(r'^(?:is married to|married)\s+(?P<value>.+)$', re.I), 'spouse'),
    (re.compile(r'^(?:was born in|is from|comes from)\s+(?P<value>.+)$', re.I), 'place_of_birth'),
    (re.compile(r'^(?:studied at|graduated from)\s+(?P<value>.+)$', re.I), 'alma_mater'),
)


@dataclass(frozen=True)
class FactProvenance:
    """一条事实的来源定位：公开事件编号 + session/turn + 时间 + 文本区间。"""

    fact_id: str
    source_id: str
    session: str
    turn: str
    observed_at: str
    char_start: int
    char_end: int
    confidence: float
    reasons: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ExtractionDiagnostic:
    """单条事实的抽取诊断：置信度 + 失败/降级原因。"""

    fact_id: str
    clause: str
    confidence: float
    reasons: Tuple[str, ...] = ()

    @property
    def failed(self) -> bool:
        return bool(self.reasons) and self.confidence < 0.5


@dataclass
class ExtractionResult:
    facts: List[AtomicFact] = field(default_factory=list)
    provenance: List[FactProvenance] = field(default_factory=list)
    diagnostics: List[ExtractionDiagnostic] = field(default_factory=list)
    unresolved: List[AtomicFact] = field(default_factory=list)

    def provenance_of(self, fact_id: str) -> Optional[FactProvenance]:
        for item in self.provenance:
            if item.fact_id == fact_id:
                return item
        return None


@dataclass
class ExtractionContext:
    """抽取过程中的滚动上下文（代词消解的锚点，只在公开文本内传递）。"""

    source_id: str = ''
    observed_at: str = UNKNOWN_TIME
    role: str = ''
    last_subject: Optional[str] = None
    last_attribute: Optional[str] = None


ReferenceResolver = Callable[[str, ExtractionContext], Optional[str]]


class DefaultReferenceResolver:
    """默认代词消解钩子：把 he/she/they/it/their... 绑到同一事件里最近出现的主体。

    这是可替换的钩子，不是模型；外部可以传入更强的 coref 实现。
    """

    def __call__(self, mention: str, context: ExtractionContext) -> Optional[str]:
        key = normalize_text(mention)
        if key.endswith("'s"):
            key = key[:-2]
        if key not in PRONOUNS:
            return None
        return context.last_subject


def parse_observed_at(text: str, fallback: str = UNKNOWN_TIME) -> str:
    """从文本里解析观察时间（年/ISO 日期/英文日期），解析不到就用 fallback。"""
    if isinstance(text, str):
        iso = re.search(r'\b((?:19|20)\d{2})-(\d{2})-(\d{2})\b', text)
        if iso:
            return f'{iso.group(1)}-{iso.group(2)}-{iso.group(3)}T00:00:00'
        year = re.search(r'\b((?:19|20)\d{2})\b', text)
        named = re.search(
            r'\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|'
            r'September|October|November|December)\s+((?:19|20)\d{2})\b', text, re.I)
        if named:
            month = _MONTHS[named.group(2).lower()]
            return f'{named.group(3)}-{month:02d}-{int(named.group(1)):02d}T00:00:00'
        if year:
            return f'{year.group(1)}-01-01T00:00:00'
    return fallback or UNKNOWN_TIME


def _clean_surface(text: str) -> str:
    value = _WS.sub(' ', str(text)).strip()
    value = value.strip(' \t,;:.!?"\'\u201c\u201d\u2018\u2019()[]')
    value = re.sub(r'\s+(?:and|but|while|although)$', '', value, flags=re.I)
    # 取值尾部的时间状语交给 observed_at，不留在 value 里
    value = re.sub(r'\s+(?:in|on|at|during|since|as\s+of)\s+(?:19|20)\d{2}.*$', '', value, flags=re.I)
    return value.strip()


def _is_pronoun(mention: str) -> bool:
    key = normalize_text(mention)
    if key.endswith("'s"):
        key = key[:-2]
    return key in PRONOUNS


class FactExtractor:
    """确定性事实抽取器（TempLAMA 单事实 + LongMemEval 多事实骨架）。

    enforce_public_view=True 时，输入事件必须是 P0 的公开写视图白名单，
    一旦出现 Oracle 字段立即抛 ValueError（这是结构性防泄漏，不靠约定）。
    """

    def __init__(self, *, reference_resolver: Optional[ReferenceResolver] = None,
                 enforce_public_view: bool = True,
                 max_facts: int = 128,
                 split_multi_fact: bool = True,
                 fact_id_prefix: str = 'fact') -> None:
        self.reference_resolver = reference_resolver or DefaultReferenceResolver()
        self.enforce_public_view = bool(enforce_public_view)
        self.max_facts = int(max_facts)
        self.split_multi_fact = bool(split_multi_fact)
        self.fact_id_prefix = fact_id_prefix
        self._fact_next = 1
        self._provenance: Dict[str, FactProvenance] = {}
        self._unresolved: List[AtomicFact] = []
        self.last_result: Optional[ExtractionResult] = None

    # -- 公开 API ---------------------------------------------------------
    def extract(self, event: Mapping[str, Any]) -> List[AtomicFact]:
        """把一条公开写视图事件抽成若干 AtomicFact（含低置信度的未解析事实）。"""
        return list(self.extract_detailed(event).facts)

    def extract_detailed(self, event: Mapping[str, Any]) -> ExtractionResult:
        if not isinstance(event, Mapping):
            raise ValueError(f'FactExtractor.extract expects a mapping, got {type(event).__name__}')
        if self.enforce_public_view:
            assert_no_oracle(event, 'FactExtractor.extract', allowed=PUBLIC_WRITE_KEYS)
        source_id = str(event.get('id', ''))
        text = event.get('text', '')
        if not isinstance(text, str):
            raise ValueError('FactExtractor.extract: event["text"] must be str')
        event_time = str(event.get('time', '') or '')
        result = ExtractionResult()
        context = ExtractionContext(source_id=source_id,
                                    observed_at=event_time or UNKNOWN_TIME)
        for segment in self._iter_segments(text):
            for sentence, start, end in self._iter_sentences(segment.text):
                for clause, cstart, cend in self._iter_clauses(sentence, segment.start + start):
                    if len(result.facts) >= self.max_facts:
                        break
                    self._extract_clause(
                        clause=clause, char_start=cstart, char_end=cend,
                        context=context, segment_role=segment.role,
                        event_time=event_time, result=result)
        self.last_result = result
        return result

    def provenance(self, fact_id: str) -> Optional[FactProvenance]:
        return self._provenance.get(fact_id)

    @property
    def unresolved(self) -> List[AtomicFact]:
        """独立记忆：抽不出 subject/attribute 的事实留在这里，不进状态层。"""
        return list(self._unresolved)

    def serialize_provenance(self) -> List[Dict[str, Any]]:
        return [
            {'fact_id': p.fact_id, 'source_id': p.source_id, 'session': p.session,
             'turn': p.turn, 'observed_at': p.observed_at,
             'char_start': p.char_start, 'char_end': p.char_end,
             'confidence': p.confidence, 'reasons': list(p.reasons)}
            for p in self._provenance.values()
        ]

    # -- 切分 -------------------------------------------------------------
    @dataclass
    class Segment:
        role: str
        text: str
        start: int
        end: int

    def _iter_segments(self, text: str) -> List['FactExtractor.Segment']:
        segments: List['FactExtractor.Segment'] = []
        for line in _LINE.finditer(text):
            raw = line.group(0)
            if not raw.strip():
                continue
            role, content = '', raw
            if self.split_multi_fact:
                matched = _ROLE_PREFIX.match(raw)
                if matched:
                    role, content = matched.group('role').strip().lower(), matched.group('content')
                    offset = raw.find(content)
                else:
                    offset = 0
            else:
                offset = 0
            segments.append(FactExtractor.Segment(
                role=role, text=content,
                start=line.start() + max(offset, 0),
                end=line.start() + max(offset, 0) + len(content)))
        if not segments:
            segments.append(FactExtractor.Segment(role='', text=text, start=0, end=len(text)))
        return segments

    def _iter_sentences(self, text: str) -> List[Tuple[str, int, int]]:
        if not self.split_multi_fact:
            return [(text, 0, len(text))]
        out: List[Tuple[str, int, int]] = []
        for match in _SENTENCE.finditer(text):
            piece = match.group(0)
            if not piece.strip():
                continue
            out.append((piece, match.start(), match.end()))
        return out or [(text, 0, len(text))]

    def _iter_clauses(self, sentence: str, offset: int) -> List[Tuple[str, int, int]]:
        out: List[Tuple[str, int, int]] = []
        if not self.split_multi_fact:
            return [(sentence, offset, offset + len(sentence))]
        for part in re.split(r'(?<=;)\s+|\s+and\s+|\s+but\s+', sentence):
            if not part or not part.strip():
                continue
            clause = part.strip()
            position = sentence.find(clause)
            if position < 0:
                position = 0
            out.append((clause, offset + position, offset + position + len(clause)))
        return out or [(sentence, offset, offset + len(sentence))]

    # -- 单子句解析 -------------------------------------------------------
    def _extract_clause(self, *, clause: str, char_start: int, char_end: int,
                        context: ExtractionContext, segment_role: str,
                        event_time: str, result: ExtractionResult) -> None:
        stripped, time_hint, inner_offset = self._strip_time_prefix(clause)
        char_start += inner_offset
        observed_at = parse_observed_at(time_hint or stripped, event_time or UNKNOWN_TIME)
        parsed = self._parse_clause(stripped, context)
        subject, attribute, value, confidence, reasons = parsed
        if subject and not _is_pronoun(subject):
            context.last_subject = subject
        if attribute:
            context.last_attribute = attribute
        context.observed_at = observed_at
        context.role = segment_role

        fact_id = self._new_fact_id()
        fact = AtomicFact(
            id=fact_id,
            text=_clean_surface(clause),
            observed_at=observed_at or UNKNOWN_TIME,
            subject=subject or UNRESOLVED_SUBJECT,
            attribute=attribute or UNRESOLVED_ATTRIBUTE,
            value=value or '',
            source_id=context.source_id,
        )
        if fact.subject == UNRESOLVED_SUBJECT or fact.attribute == UNRESOLVED_ATTRIBUTE:
            reasons = tuple(reasons) + ('unresolved_identity',)
            self._unresolved.append(fact)
        provenance = FactProvenance(
            fact_id=fact_id,
            source_id=context.source_id,
            session=context.source_id,
            turn=f'{segment_role}#{len(result.facts)}' if segment_role else f'#{len(result.facts)}',
            observed_at=fact.observed_at,
            char_start=char_start,
            char_end=char_end,
            confidence=float(confidence),
            reasons=tuple(reasons),
        )
        self._provenance[fact_id] = provenance
        result.facts.append(fact)
        result.provenance.append(provenance)
        result.diagnostics.append(ExtractionDiagnostic(
            fact_id=fact_id, clause=_clean_surface(clause),
            confidence=float(confidence), reasons=tuple(reasons)))
        if fact in self._unresolved:
            result.unresolved.append(fact)

    def _new_fact_id(self) -> str:
        fact_id = f'{self.fact_id_prefix}_{self._fact_next:06d}'
        self._fact_next += 1
        return fact_id

    @staticmethod
    def _strip_time_prefix(clause: str) -> Tuple[str, str, int]:
        matched = _TIME_PREFIX.match(clause)
        if not matched:
            return clause, '', 0
        return clause[matched.end():], matched.group('stamp'), matched.end()

    def _parse_clause(self, clause: str, context: ExtractionContext
                      ) -> Tuple[Optional[str], Optional[str], str, float, Tuple[str, ...]]:
        """返回 (subject, attribute, value, confidence, reasons)。"""
        text = _clean_surface(clause)
        if not text:
            return None, None, '', 0.0, ('empty_clause',)

        # 1) 动词模板："Alice works at Acme"
        for pattern, attribute in _VERB_TEMPLATES:
            matched = pattern.match(text)
            if matched:
                subject = self._resolve_subject(matched.group('subject'), context)
                value = _clean_surface(matched.group('value'))
                if subject and value:
                    return subject, normalize_attribute(attribute), value, 0.6, ('verb_template',)
                break

        # 2) 系表结构："<attribute> of <subject> is <value>" / "<subject>'s <attribute> is <value>"
        copula = _COPULA.search(text)
        if copula:
            left, right = text[:copula.start()].strip(), text[copula.end():].strip()
            value = _clean_surface(right)
            possessive = _POSSESSIVE.match(left)
            if possessive:
                subject = self._resolve_subject(possessive.group('subject'), context)
                attribute = normalize_attribute(possessive.group('attribute'))
                if subject and attribute and value:
                    return subject, attribute, value, 0.9, ('possessive_template',)
            if ' of ' in left:
                attribute_text, subject_text = left.rsplit(' of ', 1)
                subject = self._resolve_subject(subject_text, context)
                attribute = normalize_attribute(attribute_text)
                if subject and attribute and value:
                    return subject, attribute, value, 0.9, ('of_template',)
            # 3) 反向系表："Paris is the capital of France"
            inverse = _INVERSE.match(right)
            if inverse:
                subject = self._resolve_subject(inverse.group('owner'), context)
                attribute = normalize_attribute(inverse.group('attribute'))
                if subject and attribute and left:
                    return subject, attribute, _clean_surface(left), 0.75, ('inverse_template',)
            # 3.5) 物主代词 + 属性："Her employer is Acme" → 主体由代词消解钩子给出
            possessive_pronoun = _POSSESSIVE_PRONOUN.match(left)
            if possessive_pronoun:
                subject = self._resolve_subject(possessive_pronoun.group('pronoun'), context)
                attribute = normalize_attribute(possessive_pronoun.group('attribute'))
                if attribute and value:
                    if subject:
                        return subject, attribute, value, 0.85, ('possessive_pronoun_template',)
                    return None, attribute, value, 0.2, ('subject_unresolved',)
            # 4) 只有属性 + 值：靠代词消解钩子补主体
            attribute = normalize_attribute(left)
            subject = self._resolve_subject(context.last_subject or '', context)
            if attribute and value and subject:
                return subject, attribute, value, 0.4, ('attribute_only_subject_from_context',)
            return None, attribute if attribute else None, value, 0.2, ('subject_unresolved',)

        # 5) 省略主语的动词短语："... and lives in Berlin"
        for pattern, attribute in _LEADING_VERB_TEMPLATES:
            matched = pattern.match(text)
            if matched:
                value = _clean_surface(matched.group('value'))
                subject = self._resolve_subject(context.last_subject or '', context)
                if subject and value:
                    return (subject, normalize_attribute(attribute), value, 0.45,
                            ('elided_subject_from_context',))
                break

        return None, None, text, 0.15, ('no_pattern_match',)

    def _resolve_subject(self, mention: str, context: ExtractionContext) -> Optional[str]:
        text = _clean_surface(mention)
        if not text:
            return None
        if _is_pronoun(text):
            resolved = self.reference_resolver(text, context)
            return _clean_surface(resolved) if resolved else None
        return text


__all__ = [
    'DefaultReferenceResolver',
    'ExtractionContext',
    'ExtractionDiagnostic',
    'ExtractionResult',
    'FactExtractor',
    'FactProvenance',
    'PRONOUNS',
    'UNKNOWN_TIME',
    'UNRESOLVED_ATTRIBUTE',
    'UNRESOLVED_SUBJECT',
    'parse_observed_at',
]
