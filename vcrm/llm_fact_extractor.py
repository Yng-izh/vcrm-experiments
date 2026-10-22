"""通用模型事实抽取前端（实验二 B 臂；模型本身是占位，接口已冻结）。

为什么要另开一个前端而不是删掉旧的：

    vcrm/fact_extractor.py 是确定性硬规则，句式模板是按 TempLAMA 的问句/陈述句
    适配出来的。它在该数据集上覆盖率高，但换一个数据集（自由对话、多跳、开放域）
    会 0 覆盖、全部落 DEFER——这既不是"抽取能力"，也不能作为跨数据集证据。

    实验二只换这一个变量做对照：

        A  vcrm.fact_extractor.FactExtractor   现有硬规则   + 同一 E3
        B  vcrm.llm_fact_extractor.LLMFactExtractor  通用模型抽取 + 同一 E3

    因此本模块**必须与 FactExtractor 同接口**（``extract`` / ``extract_detailed``
    / ``provenance`` / ``unresolved``），产出同一种 ``AtomicFact``，让状态层、
    版本层、读取层完全不动。

时间口径（不可妥协）：
    一条事实的 ``observed_at`` 一律取**事件被观察到的时间**（event['time']），
    不取模型从文本里猜出来的日期。理由是可见性：``store.select_version`` /
    ``select_trajectory`` 用这个时间做版本排序与 as-of 裁剪，模型猜出来的日期
    一旦晚于真实观测时刻，就会把未来版本提前变成"可见"。模型返回的
    ``observed_at`` 只记进 provenance 供诊断，不进状态层。

接口（占位模型的契约）::

    class LLMClient(Protocol):
        def complete(self, prompt: str, *, system: str | None = None,
                     max_tokens: int = 512) -> str: ...

真实模型接入时只要实现 ``complete`` 并传进 ``LLMFactExtractor(client=...)``，
其余代码一行不用改。``PlaceholderLLM`` 会显式抛 ``NotImplementedError``，
绝不静默退回规则抽取（否则 A/B 对照会退化成同一个前端跑两遍）。
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

from .data_views import PUBLIC_WRITE_KEYS, assert_no_oracle
from .fact_extractor import (
    UNKNOWN_TIME,
    UNRESOLVED_ATTRIBUTE,
    UNRESOLVED_SUBJECT,
    ExtractionDiagnostic,
    ExtractionResult,
    FactProvenance,
)
from .fact_schema import AtomicFact
from .state_index import normalize_attribute, normalize_text

#: 抽取器产出的属性一律过一遍 normalize_attribute，保证与写入侧/查询侧同一个
#: 状态身份键（否则 "head coach" 与 "head_coach" 会裂成两个状态）。

FACT_EXTRACTION_SYSTEM = (
    'You extract atomic facts from text for a versioned memory store. '
    'Answer with JSON only, no prose, no markdown fences.')

FACT_EXTRACTION_PROMPT = """\
Extract every atomic fact stated in the TEXT below.

An atomic fact is one (subject, attribute, value) triple:
  * subject   - the entity the fact is about, as written in the text (no pronouns)
  * attribute - the property/relation, a short lower_snake_case noun phrase
  * value     - the value the subject has for that attribute, verbatim from the text
  * confidence- 0.0-1.0, how clearly the text states it
  * evidence  - the shortest substring of TEXT that supports the fact

Rules:
  * one clause = at most one fact; do not merge two different attributes
  * do not infer facts that are not stated
  * do not resolve pronouns across sentences
  * if the text states the same attribute twice with different values, emit two facts
  * if no fact can be extracted, return {{"facts": []}}

Respond with exactly:
{{"facts": [{{"subject": "...", "attribute": "...", "value": "...",
            "confidence": 0.0, "evidence": "..."}}]}}

TEXT:
{text}
"""

_FENCE = re.compile(r'^\s*```(?:json)?\s*|\s*```\s*$', re.I)


class LLMClient(Protocol):
    """模型客户端的唯一契约：把提示词变成一段文本。"""

    def complete(self, prompt: str, *, system: Optional[str] = None,
                 max_tokens: int = 512) -> str:  # pragma: no cover - 协议声明
        ...


class PlaceholderLLM:
    """占位客户端：真实模型未接入前显式失败。

    故意**不**实现任何降级路径。降级到规则抽取会让实验二的 A/B 对照失效，
    比直接报错糟糕得多。
    """

    def __init__(self, name: str = 'placeholder') -> None:
        self.name = name

    def complete(self, prompt: str, *, system: Optional[str] = None,
                 max_tokens: int = 512) -> str:
        raise NotImplementedError(
            f'LLM frontend is not wired yet (client={self.name!r}). Implement '
            'LLMClient.complete() and pass it as llm_client=/client=; this module '
            'deliberately refuses to fall back to the rule-based extractor.')


def parse_json_payload(text: str) -> Optional[Any]:
    """容错解析模型输出：去 markdown 围栏，取第一个 JSON 值，失败返回 None。"""
    if not isinstance(text, str) or not text.strip():
        return None
    cleaned = _FENCE.sub('', text.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    for opener, closer in (('{', '}'), ('[', ']')):
        start, end = cleaned.find(opener), cleaned.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start:end + 1])
            except json.JSONDecodeError:
                continue
    return None


def _fact_items(payload: Any) -> List[Mapping[str, Any]]:
    """把模型输出规整成 fact 列表（兼容 {"facts": [...]} 与裸列表）。"""
    if isinstance(payload, Mapping):
        items = payload.get('facts') or payload.get('fact') or []
    elif isinstance(payload, list):
        items = payload
    else:
        items = []
    return [item for item in items if isinstance(item, Mapping)]


class LLMFactExtractor:
    """通用模型抽取器；与 FactExtractor 同接口，可整体替换。

    ``max_facts`` 与 ``min_confidence`` 是硬闸门：模型输出超量或低置信的条目
    直接丢弃并记进 diagnostics，不做"猜着补全"。
    """

    def __init__(self, client: Optional[LLMClient] = None, *, max_facts: int = 32,
                 min_confidence: float = 0.0, enforce_public_view: bool = True,
                 max_tokens: int = 1024) -> None:
        if max_facts < 1:
            raise ValueError('max_facts must be >= 1')
        if not 0.0 <= float(min_confidence) <= 1.0:
            raise ValueError('min_confidence must be within [0, 1]')
        self.client = client if client is not None else PlaceholderLLM()
        self.max_facts = int(max_facts)
        self.min_confidence = float(min_confidence)
        self.enforce_public_view = bool(enforce_public_view)
        self.max_tokens = int(max_tokens)
        self._provenance: Dict[str, FactProvenance] = {}
        self._unresolved: List[AtomicFact] = []
        self.last_result: Optional[ExtractionResult] = None
        self.last_raw_response: Optional[str] = None

    # -- 与 FactExtractor 相同的入口 --------------------------------------
    def extract(self, event: Mapping[str, Any]) -> List[AtomicFact]:
        return self.extract_detailed(event).facts

    def extract_detailed(self, event: Mapping[str, Any]) -> ExtractionResult:
        if not isinstance(event, Mapping):
            raise ValueError(
                f'LLMFactExtractor.extract expects a mapping, got {type(event).__name__}')
        if self.enforce_public_view:
            assert_no_oracle(event, 'LLMFactExtractor.extract', allowed=PUBLIC_WRITE_KEYS)
        source_id = str(event.get('id', ''))
        text = event.get('text', '')
        if not isinstance(text, str):
            raise ValueError('LLMFactExtractor: event["text"] must be str')
        # 观测时间是事件属性，不是模型输出：见模块 docstring 的时间口径。
        observed_at = str(event.get('time', '') or '') or UNKNOWN_TIME
        result = ExtractionResult()
        self.last_raw_response = None
        if not text.strip():
            return result
        prompt = FACT_EXTRACTION_PROMPT.format(text=text)
        try:
            raw = self.client.complete(prompt, system=FACT_EXTRACTION_SYSTEM,
                                       max_tokens=self.max_tokens)
        except NotImplementedError:
            raise
        except Exception as exc:  # noqa: BLE001 - 模型故障 → 显式诊断，不静默吞
            result.diagnostics.append(ExtractionDiagnostic(
                fact_id=f'{source_id}#error', clause=text[:200], confidence=0.0,
                reasons=(f'llm_call_failed:{type(exc).__name__}',)))
            self.last_result = result
            return result
        self.last_raw_response = raw
        payload = parse_json_payload(raw)
        if payload is None:
            result.diagnostics.append(ExtractionDiagnostic(
                fact_id=f'{source_id}#error', clause=text[:200], confidence=0.0,
                reasons=('llm_parse_failed',)))
            self.last_result = result
            return result
        for index, item in enumerate(_fact_items(payload)):
            if len(result.facts) >= self.max_facts:
                break
            fact = self._fact_from_item(item, source_id=source_id, index=index,
                                        observed_at=observed_at, text=text,
                                        result=result)
            if fact is not None:
                result.facts.append(fact)
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

    # -- 内部 --------------------------------------------------------------
    def _fact_from_item(self, item: Mapping[str, Any], *, source_id: str, index: int,
                        observed_at: str, text: str,
                        result: ExtractionResult) -> Optional[AtomicFact]:
        subject = str(item.get('subject') or '').strip()
        attribute = str(item.get('attribute') or '').strip()
        value = str(item.get('value') or '').strip()
        evidence = str(item.get('evidence') or '').strip()
        try:
            confidence = float(item.get('confidence', 1.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = min(1.0, max(0.0, confidence))
        fact_id = f'{source_id}#f{index:03d}'
        if confidence < self.min_confidence:
            result.diagnostics.append(ExtractionDiagnostic(
                fact_id=fact_id, clause=evidence or text[:200], confidence=confidence,
                reasons=('below_min_confidence',)))
            return None
        if not subject or not attribute:
            result.diagnostics.append(ExtractionDiagnostic(
                fact_id=fact_id, clause=evidence or text[:200], confidence=confidence,
                reasons=('missing_subject_or_attribute',)))
            unresolved = AtomicFact(
                id=fact_id, text=evidence or text[:200], observed_at=observed_at,
                subject=normalize_text(subject) or UNRESOLVED_SUBJECT,
                attribute=(normalize_attribute(attribute)
                           if attribute else UNRESOLVED_ATTRIBUTE),
                value=value, source_id=source_id)
            self._unresolved.append(unresolved)
            result.unresolved.append(unresolved)
            return None
        if normalize_text(subject) == 'unknown' or not normalize_attribute(attribute):
            result.diagnostics.append(ExtractionDiagnostic(
                fact_id=fact_id, clause=evidence or text[:200], confidence=confidence,
                reasons=('degenerate_identity',)))
            return None
        fact = AtomicFact(
            id=fact_id, text=evidence or text[:200], observed_at=observed_at,
            subject=subject, attribute=normalize_attribute(attribute),
            value=value, source_id=source_id)
        start = text.find(evidence) if evidence else -1
        self._provenance[fact_id] = FactProvenance(
            fact_id=fact_id, source_id=source_id, session='', turn='',
            observed_at=observed_at,
            char_start=0 if start < 0 else start,
            char_end=len(text) if start < 0 else start + len(evidence),
            confidence=confidence, reasons=('llm_extraction',))
        return fact


__all__ = [
    'FACT_EXTRACTION_PROMPT',
    'FACT_EXTRACTION_SYSTEM',
    'LLMClient',
    'LLMFactExtractor',
    'PlaceholderLLM',
    'parse_json_payload',
]
