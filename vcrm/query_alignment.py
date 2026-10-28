"""查询侧主体/属性对齐层：把 LLM 解析出的身份对齐到**写入侧真实存在的状态键空间**。

为什么需要（t32 GPU 冒烟抓到的 blocker）：

    写侧 ``LLMFactExtractor`` 的 subject 是文本里的具体实体
    （``'1997 jeep wrangler'``、``'engine'``）；
    读侧 ``LLMQueryParser`` 对 "How many engineers do I lead ..." 给出
    ``subject='I'`` + ``attribute='number of engineers led when I just started my
    new role as Senior Software Engineer'``。

    两侧归一化后仍永不相交 → ``identity_matched`` 恒空 → ``no_state_match=True``
    → latent 组答案 100% 为空。语义相似度召回照样能返回候选，所以**不报错、
    只静默答空**，这正是最危险的一类失败。

对齐策略（逐级降级，每一级都记账，绝不静默猜）：

    1. ``exact``             subject_key 直接在写入侧词表里
    2. ``user_alias``        查询侧是代词（I/my/me/…）→ 找写入侧已有的
                             user/self 类主体键（可能是 'user'、'i'、'the_user'）
    3. ``subject_fuzzy``     主体键 token 包含度匹配（'jeep wrangler 1997' vs
                             '1997 jeep wrangler'）
    4. ``attribute_global``  **主体怎么都对不上时**，用属性词表在全库范围内找
                             最佳属性匹配（属性词表很小，且属性是"问什么"的
                             真实信号）——这是 t32 例子里唯一能救回的路径
    5. ``unmatched``         保持原样并显式记账（调用方可据此判 no_state_match）

设计边界：

    * 只读 ``StateIndex.records`` 的公开字段（subject_key / attribute_key），
      不读金标、不读 state_key、不改写入侧任何行为；
    * 只做**键空间对齐**，不做"猜答案"：对齐后仍然走正常的路由/版本选择/读取；
    * 每一步都留 ``source`` / ``score``，便于事后审计"这条答案是对齐救回来的
      还是本来就能命中"；
    * 词表为空时原样返回（CPU 冒烟/空库不报错）。
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .state_index import normalize_attribute, normalize_subject

#: 代词/自身指代表面形式 → 归一化后用于识别"这是在问用户自己"。
USER_SUBJECT_ALIASES = frozenset({
    'i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours',
    'user', 'the_user', 'this_user', 'the_speaker', 'speaker', 'self',
})
#: 写入侧可能采用的"用户实体"键（LLM 抽取器可能写 'user' 也可能写 'i'）。
USER_ENTITY_KEYS = frozenset({'user', 'the_user', 'this_user', 'i', 'me',
                              'speaker', 'the_speaker', 'self'})
#: 对齐来源（审计字段取值，顺序即降级顺序）。
ALIGNMENT_SOURCES: Tuple[str, ...] = ('exact', 'user_alias', 'subject_fuzzy',
                                      'attribute_global', 'unmatched',
                                      'no_vocabulary')

DEFAULT_MIN_SUBJECT_SCORE = 0.5
DEFAULT_MIN_ATTRIBUTE_SCORE = 0.6

#: 属性匹配时忽略的高频虚词（否则 "of/is/the" 会把任意两条属性拉到一起）。
_STOPWORDS = frozenset({'a', 'an', 'the', 'of', 'is', 'are', 'was', 'were', 'be',
                        'to', 'in', 'on', 'at', 'for', 'and', 'or', 'that', 'this',
                        'what', 'which', 'who', 'how', 'many', 'much', 'do', 'does',
                        'did', 'my', 'i', 'me', 'you', 'your', 'it', 'its'})


def _tokens(key: str) -> frozenset:
    return frozenset(part for part in str(key).split('_')
                     if part and part not in _STOPWORDS)


def containment_score(query_key: str, target_key: str) -> float:
    """target 被 query 覆盖的比例（query 常是整句，target 是短键）。

    ``containment_score('number_of_engineers_led_now', 'engineers_led') == 1.0``
    """
    target = _tokens(target_key)
    if not target:
        return 0.0
    return len(_tokens(query_key) & target) / len(target)


def match_subject_key(subject: Any, known_subjects: Iterable[str],
                      aliases: Sequence[str] = (), *,
                      min_score: float = DEFAULT_MIN_SUBJECT_SCORE
                      ) -> Tuple[Optional[str], str, float]:
    """把查询主体匹配到**已知主体键集合**里的一个键（router 硬门复用同一套归一化）。

    返回 ``(matched_key_or_None, source, score)``；``source`` 与对齐层同一套取值
    （exact / user_alias / subject_fuzzy / unmatched）。匹配不上时返回 ``None``，
    调用方据此拒掉候选，而不是靠相似度阈值蒙混。
    """
    known = [key for key in dict.fromkeys(known_subjects) if key]
    if not known:
        return None, 'unmatched', 0.0
    key = normalize_subject(subject)
    if key in known:
        return key, 'exact', 1.0
    candidate_keys = [normalize_subject(alias) for alias in aliases if str(alias).strip()]
    if key in USER_SUBJECT_ALIASES or any(c in USER_SUBJECT_ALIASES for c in candidate_keys):
        for alias_key in [key] + candidate_keys:
            if alias_key in USER_ENTITY_KEYS and alias_key in known:
                return alias_key, 'user_alias', 1.0
        for subject_key in known:
            if subject_key in USER_ENTITY_KEYS:
                return subject_key, 'user_alias', 1.0
    best_key, best_score = None, 0.0
    for subject_key in known:
        score = max(containment_score(key, subject_key), containment_score(subject_key, key))
        if score > best_score:
            best_key, best_score = subject_key, score
    if best_key is not None and best_score >= min_score:
        return best_key, 'subject_fuzzy', best_score
    return None, 'unmatched', best_score


class StateVocabularyAligner:
    """把查询身份对齐到写入侧状态键空间；``bind()`` 可逐 episode 重新绑定。"""

    def __init__(self, state_index: Any = None, *,
                 min_subject_score: float = DEFAULT_MIN_SUBJECT_SCORE,
                 min_attribute_score: float = DEFAULT_MIN_ATTRIBUTE_SCORE) -> None:
        self.min_subject_score = float(min_subject_score)
        self.min_attribute_score = float(min_attribute_score)
        self.state_index: Any = None
        self._subject_labels: Dict[str, str] = {}
        self._attributes: Dict[str, List[str]] = {}
        self._built_for: Optional[int] = None
        if state_index is not None:
            self.bind(state_index)

    # -- 词表 -------------------------------------------------------------
    def bind(self, state_index: Any) -> 'StateVocabularyAligner':
        """绑定（或重新绑定）一个 StateIndex，并重建词表快照。

        注意：StateIndex 是**边写边长**的，构造时为空、写入后才有关键。因此
        ``align()`` 会在 ``len(records)`` 变化时自动重建（见 ``_refresh``）；
        只在构造期 bind 一次会得到空词表 → 全部降级成 no_vocabulary（t33 复验
        的 GPU 现场就是这么炸的）。
        """
        self.state_index = state_index
        self._built_for = None
        subjects: Dict[str, str] = {}
        attributes: Dict[str, List[str]] = {}
        for record in getattr(state_index, 'records', []) or []:
            subject_key = getattr(record, 'subject_key', None)
            attribute_key = getattr(record, 'attribute_key', None)
            if not subject_key:
                continue
            subjects.setdefault(subject_key, subject_key)
            if attribute_key:
                attributes.setdefault(subject_key, [])
                if attribute_key not in attributes[subject_key]:
                    attributes[subject_key].append(attribute_key)
        self._subject_labels = subjects
        self._attributes = attributes
        self._built_for = len(getattr(state_index, 'records', []) or [])
        return self

    def _refresh(self) -> None:
        """写入增长后按需重建词表（O(n) 且仅在记录数变化时发生）。"""
        if self.state_index is None:
            return
        try:
            current = len(self.state_index.records)
        except Exception:                      # 只读诊断对象，不可数时保持现有词表
            return
        if self._built_for != current:
            self.bind(self.state_index)

    @property
    def vocabulary(self) -> Dict[str, List[str]]:
        return {key: list(self._attributes.get(key, [])) for key in self._subject_labels}

    @property
    def empty(self) -> bool:
        return not self._subject_labels

    # -- 对齐 -------------------------------------------------------------
    def _match_subject(self, subject: Any, aliases: Sequence[str]) -> Tuple[str, str, float]:
        matched, source, score = match_subject_key(
            subject, self._subject_labels, aliases, min_score=self.min_subject_score)
        if matched is None:                    # 对齐层保持「原样保留」，不做丢弃
            return normalize_subject(subject), 'unmatched', score
        return matched, source, score

    def _match_attribute(self, subject_key: str, attribute: Any) -> Tuple[str, str, float]:
        key = normalize_attribute(attribute)
        known = self._attributes.get(subject_key, [])
        if key in known:
            return key, 'exact', 1.0
        best_key, best_score = key, 0.0
        for candidate in known:
            score = containment_score(key, candidate)
            if score > best_score:
                best_key, best_score = candidate, score
        if best_score >= self.min_attribute_score:
            return best_key, 'subject_fuzzy', best_score
        return key, 'unmatched', best_score

    def _match_attribute_globally(self, attribute: Any) -> Tuple[Optional[str], str, float]:
        """主体对不上时的兜底：全库按属性词表找最佳匹配（t32 关键路径）。"""
        key = normalize_attribute(attribute)
        best: Tuple[Optional[str], str, float] = (None, key, 0.0)
        for subject_key, known in self._attributes.items():
            for candidate in known:
                score = containment_score(key, candidate)
                if score > best[2]:
                    best = (subject_key, candidate, score)
        if best[0] is not None and best[2] >= self.min_attribute_score:
            return best
        return None, key, best[2]

    def align_identity(self, subject: Any, attribute: Any,
                       aliases: Sequence[str] = ()) -> Dict[str, Any]:
        self._refresh()
        raw_subject, raw_attribute = str(subject or ''), str(attribute or '')
        if self.empty:
            return {'subject': raw_subject, 'attribute': raw_attribute,
                    'subject_key': normalize_subject(raw_subject),
                    'attribute_key': normalize_attribute(raw_attribute),
                    'source': 'no_vocabulary', 'subject_source': 'no_vocabulary',
                    'attribute_source': 'no_vocabulary', 'score': 0.0}
        subject_key, subject_source, subject_score = self._match_subject(raw_subject, aliases)
        attribute_key, attribute_source, attribute_score = self._match_attribute(
            subject_key, raw_attribute)
        if subject_source == 'unmatched' or attribute_source == 'unmatched':
            global_subject, global_attribute, global_score = self._match_attribute_globally(
                raw_attribute)
            if global_subject is not None:
                subject_key, attribute_key = global_subject, global_attribute
                subject_source = attribute_source = 'attribute_global'
                subject_score = attribute_score = global_score
        return {'subject': subject_key, 'attribute': attribute_key,
                'subject_key': subject_key, 'attribute_key': attribute_key,
                'source': subject_source if subject_source == attribute_source
                else f'{subject_source}+{attribute_source}',
                'subject_source': subject_source, 'attribute_source': attribute_source,
                'score': round(max(subject_score, attribute_score), 4),
                'input_subject': raw_subject, 'input_attribute': raw_attribute}

    def align(self, identities: Sequence[Sequence[str]],
              aliases: Optional[Sequence[Sequence[str]]] = None
              ) -> Tuple[List[Tuple[str, str]], List[Dict[str, Any]]]:
        """把 [(subject, attribute), …] 对齐成状态键元组，并返回逐条审计报告。"""
        alias_rows: Sequence[Sequence[str]] = aliases or []
        aligned: List[Tuple[str, str]] = []
        report: List[Dict[str, Any]] = []
        seen = set()
        for position, identity in enumerate(identities):
            subject = identity[0] if len(identity) > 0 else ''
            attribute = identity[1] if len(identity) > 1 else ''
            row_aliases = alias_rows[position] if position < len(alias_rows) else ()
            record = self.align_identity(subject, attribute, row_aliases)
            key = (record['subject_key'], record['attribute_key'])
            report.append(record)
            if key in seen or record['source'] == 'unmatched':
                continue
            seen.add(key)
            aligned.append((record['subject_key'], record['attribute_key']))
        return aligned, report


def align_identities(identities: Iterable[Sequence[str]], state_index: Any,
                     *, aliases: Optional[Sequence[Sequence[str]]] = None,
                     min_subject_score: float = DEFAULT_MIN_SUBJECT_SCORE,
                     min_attribute_score: float = DEFAULT_MIN_ATTRIBUTE_SCORE
                     ) -> Tuple[List[Tuple[str, str]], List[Dict[str, Any]]]:
    """一次性对齐（无状态便利函数，测试与离线审计用）。"""
    aligner = StateVocabularyAligner(state_index, min_subject_score=min_subject_score,
                                     min_attribute_score=min_attribute_score)
    return aligner.align(list(identities), aliases=aliases)


__all__ = [
    'ALIGNMENT_SOURCES',
    'DEFAULT_MIN_ATTRIBUTE_SCORE',
    'DEFAULT_MIN_SUBJECT_SCORE',
    'StateVocabularyAligner',
    'USER_ENTITY_KEYS',
    'USER_SUBJECT_ALIASES',
    'align_identities',
    'containment_score',
    'match_subject_key',
]
