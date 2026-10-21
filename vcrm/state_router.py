from __future__ import annotations
# P2: Oracle-free candidate retrieval + routing.
#
# 两阶段检索：
#   阶段 1（硬过滤）：主体命中 或 属性命中。刻意同时保留「同主体不同属性」与
#                    「不同主体同属性」两类候选，交给 UpdateResolver 显式拒绝，
#                    避免用相似度阈值把错误的合并/拆分悄悄做掉。
#   阶段 2（辅助排序）：128 维 semantic_key 余弦相似度；没有语义键时退化为
#                    词面 Jaccard，仅用于排序，不用于判定合并。
#
# 本模块**不训练** EntityHead / RelationHead，也不修改 SemanticIndex 的权重与结构：
# 需要语义键时由调用方传入 semantic_key_fn（典型实现是 SemanticIndex.key 的闭包），
# 需要排序器时传入 ranker（典型实现是 SemanticIndex.rank）。

import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .fact_schema import AtomicFact, StateDecision
from .state_index import (
    SEMANTIC_DIMENSIONS,
    StateIndex,
    StateRecord,
    normalize_attribute,
    normalize_subject,
    normalize_text,
)

SemanticKeyFn = Callable[[Any], Optional[Sequence[float]]]
Ranker = Callable[[Sequence[Any]], List[Tuple[float, Dict[str, Any]]]]


def cosine_similarity(left: Optional[Sequence[float]],
                      right: Optional[Sequence[float]]) -> Optional[float]:
    """128 维语义键的余弦相似度；任一侧缺失则返回 None（不猜）。"""
    if left is None or right is None:
        return None
    if len(left) != len(right):
        raise ValueError(
            f'semantic keys must share dimensions, got {len(left)} vs {len(right)}')
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    norm_left = math.sqrt(sum(float(a) * float(a) for a in left))
    norm_right = math.sqrt(sum(float(b) * float(b) for b in right))
    if not norm_left or not norm_right:
        return None
    return dot / (norm_left * norm_right)


def _tokens(*values: Any) -> set:
    tokens = set()
    for value in values:
        for token in normalize_text(value).replace('_', ' ').split():
            if len(token) > 1:
                tokens.add(token)
    return tokens


def lexical_similarity(left_tokens: Iterable[str], right_tokens: Iterable[str]) -> float:
    left_set, right_set = set(left_tokens), set(right_tokens)
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / float(len(left_set | right_set))


def containment_similarity(left_tokens: Iterable[str], right_tokens: Iterable[str]) -> float:
    """包含系数：小集合被大集合覆盖的比例（"united states" ⊂ "united states of america"）。"""
    left_set, right_set = set(left_tokens), set(right_tokens)
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / float(min(len(left_set), len(right_set)))


def _fact_field(fact: Any, name: str, default: str = '') -> str:
    if isinstance(fact, dict):
        return str(fact.get(name, default) or default)
    return str(getattr(fact, name, default) or default)


def _record_tokens(record: StateRecord) -> set:
    return _tokens(record.subject_key, record.attribute_key, *record.aliases)


def _fact_tokens(fact: Any) -> set:
    return _tokens(_fact_field(fact, 'subject'), _fact_field(fact, 'attribute'),
                   _fact_field(fact, 'value'), _fact_field(fact, 'text'))


def _dedupe(records: Iterable[StateRecord]) -> List[StateRecord]:
    seen: Dict[str, StateRecord] = {}
    for record in records:
        seen.setdefault(record.state_id, record)
    return list(seen.values())


def stage1_filter(fact: Any, state_index: StateIndex, *,
                  allow_lexical_subject_fallback: bool = True,
                  min_lexical_overlap: float = 0.6) -> List[StateRecord]:
    """阶段 1：主体/属性硬过滤（别名也算命中）。"""
    subject_key = normalize_subject(_fact_field(fact, 'subject'))
    attribute_key = normalize_attribute(_fact_field(fact, 'attribute'))
    pool = _dedupe(
        state_index.candidates(subject_key=subject_key)
        + state_index.candidates(attribute_key=attribute_key))
    if pool or not allow_lexical_subject_fallback:
        return pool
    subject_tokens = _tokens(subject_key)
    fallback = [
        record for record in state_index.records
        if containment_similarity(subject_tokens, _tokens(record.subject_key)) >= min_lexical_overlap
    ]
    return fallback


def stage2_rank(fact: Any, records: Sequence[StateRecord], *,
                semantic_key: Optional[Sequence[float]] = None,
                semantic_key_fn: Optional[SemanticKeyFn] = None,
                ranker: Optional[Ranker] = None) -> List[Tuple[float, StateRecord]]:
    """阶段 2：语义键余弦辅助排序（缺失时退化为词面 Jaccard）。"""
    key = semantic_key
    if key is None and semantic_key_fn is not None:
        key = semantic_key_fn(fact)
    if key is not None:
        if len(key) != SEMANTIC_DIMENSIONS:
            raise ValueError(
                f'semantic key must have {SEMANTIC_DIMENSIONS} dims, got {len(key)}')

    scored: List[Tuple[float, StateRecord]] = []
    if ranker is not None:
        entries = [{'id': r.state_id, 'key': r.semantic_key, 'state': r.state_id}
                   for r in records if r.semantic_key is not None]
        ranked_ids = {entry['id']: score for score, entry in ranker(entries)}
        fact_tokens = _fact_tokens(fact)
        for record in records:
            score = ranked_ids.get(record.state_id)
            if score is None:
                score = lexical_similarity(fact_tokens, _record_tokens(record))
            scored.append((float(score), record))
    else:
        fact_tokens = _fact_tokens(fact)
        for record in records:
            similarity = cosine_similarity(key, record.semantic_key)
            if similarity is None:
                similarity = lexical_similarity(fact_tokens, _record_tokens(record))
            scored.append((float(similarity), record))
    scored.sort(key=lambda item: (-item[0], item[1].state_id))
    return scored


def retrieve_candidates(fact: Any, state_index: StateIndex, top_k: int = 5, *,
                        semantic_key: Optional[Sequence[float]] = None,
                        semantic_key_fn: Optional[SemanticKeyFn] = None,
                        ranker: Optional[Ranker] = None,
                        allow_lexical_subject_fallback: bool = True,
                        min_lexical_overlap: float = 0.6) -> List[StateRecord]:
    """两阶段候选检索：先硬过滤，再用语义键余弦辅助排序，返回 top_k 条。"""
    if not isinstance(state_index, StateIndex):
        raise TypeError('retrieve_candidates expects a StateIndex')
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError(f'top_k must be a positive int, got {top_k!r}')
    pool = stage1_filter(fact, state_index,
                         allow_lexical_subject_fallback=allow_lexical_subject_fallback,
                         min_lexical_overlap=min_lexical_overlap)
    if not pool:
        return []
    ranked = stage2_rank(fact, pool, semantic_key=semantic_key,
                         semantic_key_fn=semantic_key_fn, ranker=ranker)
    return [record for _, record in ranked[:top_k]]


def route(fact: Any, state_index: StateIndex, resolver: Any, top_k: int = 5, **kwargs: Any) -> StateDecision:
    """检索候选 → 交给 UpdateResolver → 返回 StateDecision（只做决策，不写 Current）。"""
    if resolver is None or not hasattr(resolver, 'resolve'):
        raise TypeError('route expects a resolver with a .resolve(fact, candidates, index) method')
    candidates = retrieve_candidates(fact, state_index, top_k=top_k, **kwargs)
    decision = resolver.resolve(fact, candidates, state_index)
    if not isinstance(decision, StateDecision):
        raise TypeError(
            f'resolver must return StateDecision, got {type(decision).__name__}')
    return decision


class StateRouter:
    """StateIndex + 语义键函数 + resolver 的轻量封装（无参数训练）。"""

    def __init__(self, *, semantic_key_fn: Optional[SemanticKeyFn] = None,
                 ranker: Optional[Ranker] = None, top_k: int = 5,
                 allow_lexical_subject_fallback: bool = True,
                 min_lexical_overlap: float = 0.6) -> None:
        self.semantic_key_fn = semantic_key_fn
        self.ranker = ranker
        self.top_k = int(top_k)
        self.allow_lexical_subject_fallback = allow_lexical_subject_fallback
        self.min_lexical_overlap = float(min_lexical_overlap)

    def retrieve_candidates(self, fact: Any, state_index: StateIndex,
                            top_k: Optional[int] = None,
                            semantic_key: Optional[Sequence[float]] = None) -> List[StateRecord]:
        return retrieve_candidates(
            fact, state_index, top_k=self.top_k if top_k is None else top_k,
            semantic_key=semantic_key, semantic_key_fn=self.semantic_key_fn,
            ranker=self.ranker,
            allow_lexical_subject_fallback=self.allow_lexical_subject_fallback,
            min_lexical_overlap=self.min_lexical_overlap)

    def route(self, fact: Any, state_index: StateIndex, resolver: Any,
              top_k: Optional[int] = None,
              semantic_key: Optional[Sequence[float]] = None) -> StateDecision:
        candidates = self.retrieve_candidates(
            fact, state_index, top_k=top_k, semantic_key=semantic_key)
        decision = resolver.resolve(fact, candidates, state_index)
        if not isinstance(decision, StateDecision):
            raise TypeError(
                f'resolver must return StateDecision, got {type(decision).__name__}')
        return decision


__all__ = [
    'Ranker',
    'SemanticKeyFn',
    'StateRouter',
    'containment_similarity',
    'cosine_similarity',
    'lexical_similarity',
    'retrieve_candidates',
    'route',
    'stage1_filter',
    'stage2_rank',
]
