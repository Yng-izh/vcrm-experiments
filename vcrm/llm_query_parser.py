"""通用模型查询解析前端（实验二 B 臂的查询侧；模型是占位，接口已冻结）。

为什么查询侧不能只做"换个说法解析成单一 (subject, attribute)"：

    LongMemEval 这类自由问答里大量问题是多跳/多事实的（"我上次提到的那家餐厅
    现在的主厨是谁"），一个状态装不下答案。只把问句硬压成单一状态，会把
    "解析不出"错报成"记忆里没有"，测的其实是解析器的表达力，不是记忆系统。

    所以本模块产出**一组**查询身份（``parse_multi``），让 ``PredictedRuntime``
    可以一次检索/读取多个相关状态（``multi_state=True``）；``parse()`` 保留单身份
    口径，供既有评测路径继续使用，两者不冲突。

与 ``vcrm.runtime.QueryParser``（规则版）的关系：
    同接口、可整体替换。规则版只认 "X lives in _X_" 这类模板；本版把问句交给
    通用模型，输出同样是 ``{'resolved', 'question', 'subject', 'attribute',
    'reason'}``，额外多一个 ``identities`` 列表。

边界（不可越过）：
    * 只看 ``question`` 文本，不看 query 的 time/as_of/answers/state_key；
    * 不产出时间：目标时间/可见截止由 ``vcrm.query_time`` 决定，避免把时间维度
      从问句里"解析"出来当成金标；
    * ``PlaceholderLLM`` 会显式抛错，不静默退回规则解析。

真实模型接入只需实现 ``complete(prompt, system=..., max_tokens=...) -> str``。
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

from .fact_extractor import UNRESOLVED_ATTRIBUTE, UNRESOLVED_SUBJECT
from .llm_fact_extractor import (
    LLMClient,
    PlaceholderLLM,
    parse_json_payload,
)
from .runtime import QUERY_MASK_TOKENS, QUERY_PLACEHOLDER
from .state_index import normalize_attribute, normalize_text

QUERY_PARSE_SYSTEM = (
    'You map a natural-language question onto memory lookup keys. '
    'Answer with JSON only, no prose, no markdown fences.')

QUERY_PARSE_PROMPT = """\
A memory store is keyed by (subject, attribute) pairs: subject = the entity the
question is about, attribute = the property/relation being asked for.

QUESTION: {question}

List every lookup key needed to answer it, most important first. For a multi-hop
or multi-fact question, list one key per hop / per fact instead of merging them.
Use the literal entity name from the question. Never put the answer in a key.

Respond with exactly:
{{"identities": [{{"subject": "...", "attribute": "...", "confidence": 0.0}}]}}

If the question cannot be mapped to any (subject, attribute) pair, return
{{"identities": []}}.
"""

#: 掩码占位符（TempLAMA 的 `_X_`）先换成中性词再交给模型。
MASK_TOKENS: Tuple[str, ...] = tuple(QUERY_MASK_TOKENS)


def mask_question(question: str) -> str:
    """把问题里的掩码占位符替换成中性词（模型不该去猜掩码里是什么）。"""
    text = question
    for token in MASK_TOKENS:
        text = text.replace(token, QUERY_PLACEHOLDER)
    return text


def _identity_items(payload: Any) -> List[Mapping[str, Any]]:
    if isinstance(payload, Mapping):
        items = payload.get('identities') or payload.get('facts') or []
    elif isinstance(payload, list):
        items = payload
    else:
        items = []
    return [item for item in items if isinstance(item, Mapping)]


class LLMQueryParser:
    """通用模型查询解析器；与 ``vcrm.runtime.QueryParser`` 同接口。

    ``max_identities`` 是多身份上限（默认 3，与 ``PredictedRuntime.max_query_states``
    对齐）；超出部分截断并记进 ``last_truncated``，不做无界展开。
    """

    def __init__(self, client: Optional[LLMClient] = None, *, max_identities: int = 3,
                 min_confidence: float = 0.0, max_tokens: int = 512) -> None:
        if int(max_identities) < 1:
            raise ValueError('max_identities must be >= 1')
        if not 0.0 <= float(min_confidence) <= 1.0:
            raise ValueError('min_confidence must be within [0, 1]')
        self.client = client if client is not None else PlaceholderLLM()
        self.max_identities = int(max_identities)
        self.min_confidence = float(min_confidence)
        self.max_tokens = int(max_tokens)
        self.last_raw_response: Optional[str] = None
        self.last_truncated = 0

    # -- 与 QueryParser 相同的入口 ----------------------------------------
    def parse(self, question: str) -> Dict[str, Any]:
        """单身份口径：取置信度最高的那一个，供既有评测路径直接替换使用。"""
        result = self.parse_multi(question)
        identities = result.get('identities') or []
        if not identities:
            return {'resolved': False, 'question': result['question'],
                    'subject': UNRESOLVED_SUBJECT, 'attribute': UNRESOLVED_ATTRIBUTE,
                    'reason': result.get('reason') or 'unresolved_identity',
                    'identities': []}
        subject, attribute = identities[0]
        return {'resolved': True, 'question': result['question'], 'subject': subject,
                'attribute': attribute, 'reason': '', 'identities': identities}

    def parse_multi(self, question: str) -> Dict[str, Any]:
        """多身份口径：返回 ``identities`` 列表（多跳/多事实问题不合并）。"""
        if not isinstance(question, str) or not question.strip():
            raise ValueError('LLMQueryParser.parse expects a non-empty question str')
        text = mask_question(question)
        self.last_truncated = 0
        try:
            raw = self.client.complete(
                QUERY_PARSE_PROMPT.format(question=text),
                system=QUERY_PARSE_SYSTEM, max_tokens=self.max_tokens)
        except NotImplementedError:
            raise
        except Exception as exc:  # noqa: BLE001 - 模型故障 → 显式未解析，不猜
            self.last_raw_response = None
            return {'resolved': False, 'question': question,
                    'subject': UNRESOLVED_SUBJECT, 'attribute': UNRESOLVED_ATTRIBUTE,
                    'reason': f'llm_call_failed:{type(exc).__name__}', 'identities': []}
        self.last_raw_response = raw
        payload = parse_json_payload(raw)
        if payload is None:
            return {'resolved': False, 'question': question,
                    'subject': UNRESOLVED_SUBJECT, 'attribute': UNRESOLVED_ATTRIBUTE,
                    'reason': 'llm_parse_failed', 'identities': []}
        identities: List[Tuple[str, str]] = []
        seen = set()
        for item in _identity_items(payload):
            subject = str(item.get('subject') or '').strip()
            attribute = str(item.get('attribute') or '').strip()
            try:
                confidence = float(item.get('confidence', 1.0))
            except (TypeError, ValueError):
                confidence = 0.0
            if confidence < self.min_confidence or not subject or not attribute:
                continue
            if normalize_text(subject) in ('', 'unknown'):
                continue
            key = (normalize_text(subject), normalize_attribute(attribute))
            if key in seen:
                continue
            seen.add(key)
            identities.append((subject, normalize_attribute(attribute)))
        if len(identities) > self.max_identities:
            self.last_truncated = len(identities) - self.max_identities
            identities = identities[:self.max_identities]
        if not identities:
            return {'resolved': False, 'question': question,
                    'subject': UNRESOLVED_SUBJECT, 'attribute': UNRESOLVED_ATTRIBUTE,
                    'reason': 'unresolved_identity', 'identities': []}
        subject, attribute = identities[0]
        return {'resolved': True, 'question': question, 'subject': subject,
                'attribute': attribute, 'reason': '', 'identities': identities,
                'truncated': self.last_truncated}


__all__ = [
    'LLMQueryParser',
    'MASK_TOKENS',
    'QUERY_PARSE_PROMPT',
    'QUERY_PARSE_SYSTEM',
    'mask_question',
]
