from __future__ import annotations
"""P3 state-level evaluator: three modes over one frozen E3 codec + reader.

口径（务必先读）：
    oracle_state     用金标 state_key 直接选状态 —— **只复现上界**，不是自动关联证据。
    predicted_state  episode 内混合事实池（本 episode + 邻近 episode 的事件），
                     状态身份完全由 FactExtractor/StateRouter/UpdateResolver 预测。
    global_e2e       所有选中 episode 的事实混进同一个全局状态空间，端到端路由 + 读取。

**predicted_state / global_e2e 才是自动关联证据**；oracle_state 只用来给出"如果状态
身份全对能拿到多少"的天花板，任何报告都必须把三者分列，不得混算。

时间口径（t13 冻结契约 + t3 泄漏审计）：
    --time-mode window_end（默认，Predicted 正式口径）
        目标时间只来自该 episode 观测窗口末刻 max(event.time)（公开可得），经
        vcrm.data_views.public_query(time_mode='window_end', window_end=...) 注入。
        窗口末刻缺失时 time=None / time_source='unknown' → select_version 走
        "无时间约束 = 最新可见版本"，绝不回退 query['time'] / query['as_of']。
    --time-mode as_of（仅受控对照 / 时间问答）
        直接使用 query['time']（TempLAMA 上 == query['as_of'] == 金标版本时间戳）。
        该口径等于把"哪个版本是答案"交给 select_version，**只能当对照**，
        summary.json 里与 window_end 口径分列，禁止混算。

UPDATE 来源口径（t15）：
    --recency-update 默认 off = 保守基线：逐年陈述句（"In 2016/2018/2020: X is Y"）
    除第一年外全部落 DEFER，版本链建不起来，predicted_state 测不到任何 UPDATE。
    on = rule5b（单值排他属性 + 观察时间严格更晚）才允许覆盖，是 TempLAMA 上唯一
    能形成多版本状态的配置。summary.json 按 reason 前缀分列
    `explicit_replacement:` 与 `exclusive_superseded_by_recency:`。
    recency_min_confidence 未在独立验证集校准前不得设非零值当结论（校准项在放行清单里）。

本脚本只做 CPU 可跑的通路 + 指标；GPU 全量运行需 captain 批准（命令清单见报告）。
"""
import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from vcrm.codec import SlotCodec, VersionMarkers
from vcrm.data_views import PublicEventCounter, TIME_MODES, public_query, public_write_event
from vcrm.fact_extractor import FactExtractor
from vcrm.fact_schema import StateDecision
from vcrm.index import SemanticIndex
from vcrm.io import read_json, write_json
from vcrm.metrics import exact, f1 as token_f1
from vcrm.runtime import PredictedRuntime, QueryParser, validate_decision_log
from vcrm.state_index import SEMANTIC_DIMENSIONS, StateIndex, value_fingerprint
from vcrm.state_router import StateRouter
from vcrm.store import IncrementalCurrentStore
from vcrm.update_resolver import UpdateResolver

MODES = ('oracle_state', 'predicted_state', 'global_e2e')
_TOKEN = re.compile(r'[a-z]+')
_STOPWORDS = frozenset('a an the is are was were of in on at to for and or it that '
                       'this with by as from'.split())
UPDATE_SOURCE_PREFIXES = ('explicit_replacement', 'exclusive_superseded_by_recency')


# ---------------------------------------------------------------------------
# components: reader / codec / markers / semantic index
# ---------------------------------------------------------------------------
class StubReader:
    """CPU-only deterministic stand-in for FrozenReader（绝不冒充真实 ICAE 结果）。

    同一文本 → 同一 latent（blake2b 定种），因此可以验证"选了哪个版本/哪个状态"，
    但不产生任何语言答案；summary.json 会显式标注 reader='stub'。
    """

    def __init__(self, hidden: int = 32, slots: int = 8) -> None:
        self.cfg = {'read_latent_slots': None, 'state_similarity': 0.8}
        self.device = torch.device('cpu')
        self.hidden, self.slots = int(hidden), int(slots)

    def event_text(self, event: Mapping[str, Any]) -> str:
        return f"DATE: {event['time']}\n{event['text']}"

    def encode(self, text: str, training: bool = False) -> torch.Tensor:
        """词袋式确定性 latent：同文本同向量，词重叠越多余弦越近（可解释，非 ICAE）。"""
        vector = torch.zeros(self.hidden)
        for token in _TOKEN.findall(str(text).lower()):
            if token in _STOPWORDS:
                continue
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            value = int.from_bytes(digest, 'little')
            vector[value % self.hidden] += 1.0 if (value // self.hidden) % 2 else -1.0
        norm = float(vector.norm())
        if norm > 0:
            vector = vector / norm
        return vector.unsqueeze(0).repeat(self.slots, 1)

    def encode_event(self, event: Mapping[str, Any], training: bool = False) -> torch.Tensor:
        return self.encode(self.event_text(event), training)

    def question(self, query: Mapping[str, Any]) -> str:
        date = query.get('time')
        return (f'Current date: {date}\n' if date else '') + str(query['question'])

    def answer(self, query: Mapping[str, Any], bundle: Optional[torch.Tensor] = None,
               text: str = '') -> Dict[str, Any]:
        if bundle is None:
            return {'answer': '', 'read_latent_slots': 0}
        checksum = hashlib.blake2b(
            bundle.detach().cpu().float().numpy().tobytes(), digest_size=8).hexdigest()
        return {'answer': f'stub:{checksum}', 'read_latent_slots': int(bundle.shape[0]),
                'bundle_checksum': checksum}


class Components:
    def __init__(self, reader, codec, markers, index, label: str) -> None:
        self.reader, self.codec, self.markers, self.index = reader, codec, markers, index
        self.label = label


def local_e3_shaped_codec(dimension: int, slots: int = 16, width: int = 256,
                          heads: int = 4) -> Tuple[SlotCodec, str]:
    """未训练的 E3 同构 codec（服务器权重不可达时的 CPU 通路，绝不冒充真实权重）。"""
    torch.manual_seed(42)
    codec = SlotCodec(dimension, slots, width, heads, False, 0., 42,
                      'residual', None, 'full', 8.0)
    return codec.eval().requires_grad_(False), (
        f'local_e3_config(dimension={dimension},slots={slots},width={width},heads={heads},'
        'value_path=full,rms_cap=8.0; UNTRAINED)')


def build_components(args: argparse.Namespace) -> Components:
    if args.reader == 'stub':
        reader = StubReader(hidden=args.hidden, slots=args.memory_slots)
        codec, label = local_e3_shaped_codec(args.hidden, min(16, max(1, args.memory_slots // 2)),
                                             width=min(256, args.hidden), heads=2)
        markers = VersionMarkers(args.hidden)
        index = SemanticIndex(args.hidden, SEMANTIC_DIMENSIONS)
        return Components(reader, codec, markers, index, f'stub+{label}')
    from vcrm.runtime import SlotMemoryRuntime
    config, training_root, _ = _load_v6(args.arch_exp)
    runtime = SlotMemoryRuntime.from_checkpoint(config, training_root)
    return Components(runtime.reader, runtime.codec, runtime.markers, runtime.index,
                      f'icae+checkpoint:{training_root}')


def _load_v6(exp: str):
    import importlib.util
    path = REPO_ROOT / 'probe' / 'v6_loader.py'
    spec = importlib.util.spec_from_file_location('vcrm_probe_v6_loader', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_v6(exp)


# ---------------------------------------------------------------------------
# metrics accumulator
# ---------------------------------------------------------------------------
class Metrics:
    def __init__(self) -> None:
        self.tp = self.fp = self.fn = 0
        self.pred_states = 0
        self.false_merged_states = 0
        self.gold_states = 0
        self.false_split_states = 0
        self.time_source: Counter = Counter()
        self.decisions: Counter = Counter()
        self.update_sources: Counter = Counter()
        self.decision_log: List[Dict[str, Any]] = []
        self._seen_decisions: set = set()
        self.rows: List[Dict[str, Any]] = []
        self.writes = 0
        self.write_seconds = 0.0
        self.store_bytes = 0
        self.pending_bytes = 0
        self.pending_count = 0
        self.diagnostic_correct = 0
        self.diagnostic_n = 0
        self.slots: List[int] = []
        self.read_seconds: List[float] = []

    def add_facts(self, facts: Sequence[Mapping[str, Any]]) -> None:
        usable = [f for f in facts if f.get('state_id') and f.get('gold')]
        for i in range(len(usable)):
            for j in range(i + 1, len(usable)):
                left, right = usable[i], usable[j]
                same_pred = left['state_id'] == right['state_id']
                same_gold = left['gold'] == right['gold']
                if same_pred and same_gold:
                    self.tp += 1
                elif same_pred:
                    self.fp += 1
                elif same_gold:
                    self.fn += 1
        by_state: Dict[str, set] = defaultdict(set)
        by_gold: Dict[str, set] = defaultdict(set)
        for fact in usable:
            by_state[fact['state_id']].add(fact['gold'])
            by_gold[fact['gold']].add(fact['state_id'])
        self.pred_states += len(by_state)
        self.false_merged_states += sum(1 for golds in by_state.values() if len(golds) > 1)
        self.gold_states += len(by_gold)
        self.false_split_states += sum(1 for states in by_gold.values() if len(states) > 1)

    def add_decisions(self, decisions: Iterable[Mapping[str, Any]],
                      only_fact_ids: Optional[Iterable[str]] = None) -> None:
        # F3：predicted_state 每个 episode 用的是"本 episode + 邻近"整池 runtime，直接累加会
        # 把同一批决策重复计 1+neighbours 次；这里按 (event_id, fact_id) 去重，并且（给定
        # only_fact_ids 时）只累计本 episode 的事实，保证 decision_counts 与写入事实数一致。
        allowed = None if only_fact_ids is None else set(only_fact_ids)
        seen: set = set()          # 单次调用内去重；跨 episode 的同名 fact_id 是不同事实
        for record in decisions:
            if allowed is not None and record.get('fact_id') not in allowed:
                continue
            key = (record.get('event_id'), record.get('fact_id'))
            if key in seen:
                continue
            seen.add(key)
            self.decision_log.append(dict(record))
            self.decisions[record['decision']] += 1
            if record['decision'] == 'UPDATE':
                prefix = str(record.get('reason', '')).split(':', 1)[0]
                self.update_sources[prefix or 'unknown'] += 1
            else:
                self.update_sources['not_update'] += 1

    def add_store(self, store: Any) -> None:
        self.store_bytes += len(store.serialize())
        pending = getattr(store, 'pending', None) or []
        self.pending_count += len(pending)
        self.pending_bytes += sum(int(record['payload'].nbytes) for record in pending)

    def add_row(self, row: Dict[str, Any]) -> None:
        self.rows.append(row)
        self.time_source[row['time_source']] += 1
        self.slots.append(int(row.get('read_latent_slots') or 0))
        if row.get('diagnostic_version_correct') is not None and row['gold_type'] != 'latest':
            self.diagnostic_n += 1
            self.diagnostic_correct += int(bool(row['diagnostic_version_correct']))
        if row.get('read_seconds') is not None:
            self.read_seconds.append(float(row['read_seconds']))

    def summary(self, args: argparse.Namespace, components: Components,
                episodes: int) -> Dict[str, Any]:
        precision = self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0
        recall = self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        answered = [row for row in self.rows if not row['no_state_match']]
        # F2：主表 em/f1 只统计 latent 路径；parametric 回退单列，两者不得混算。
        latent = [row for row in answered if not row.get('hybrid_fallback')]
        # 回退行按定义是 no_state_match=True 的行（不进 answered），所以从全表筛。
        fallback = [row for row in self.rows if row.get('hybrid_fallback')]
        em = sum(row['em'] for row in latent) / len(latent) if latent else 0.0
        f1_score = sum(row['f1'] for row in latent) / len(latent) if latent else 0.0
        fallback_em = sum(row['em'] for row in fallback) / len(fallback) if fallback else None
        fallback_f1 = sum(row['f1'] for row in fallback) / len(fallback) if fallback else None
        history = [row for row in self.rows
                   if row['gold_type'] != 'latest' and row.get('selected_version_time')]
        latest = [row for row in self.rows
                  if row['gold_type'] == 'latest' and row.get('selected_version_time')]
        recall_rows = [row for row in self.rows if row.get('recall_at_1') is not None]
        return {
            'tag': args.tag,
            'mode': args.mode,
            'time_mode': args.time_mode,
            'time_source_distribution': dict(sorted(self.time_source.items())),
            'time_source_vocabulary': ['as_of', 'window_end', 'unknown'],
            'recency_update': bool(args.recency_update),
            'recency_min_confidence': float(args.recency_min_confidence),
            'update_source_counts': {key: int(self.update_sources.get(key, 0))
                                     for key in UPDATE_SOURCE_PREFIXES + ('not_update',)},
            'decision_counts': dict(sorted(self.decisions.items())),
            'reader': args.reader,
            'components': components.label,
            'arch_exp': args.arch_exp,
            'budget_mode': args.budget_mode,
            'episodes': episodes,
            'queries': len(self.rows),
            'writes': self.writes,
            'state_pairwise': {'precision': precision, 'recall': recall, 'f1': f1,
                               'tp': self.tp, 'fp': self.fp, 'fn': self.fn},
            'false_merge_rate': (self.false_merged_states / self.pred_states
                                 if self.pred_states else 0.0),
            'false_split_rate': (self.false_split_states / self.gold_states
                                 if self.gold_states else 0.0),
            'query_state_recall_at_1': (sum(1 for row in recall_rows if row['recall_at_1'])
                                        / len(recall_rows) if recall_rows else None),
            'query_state_recall_at_3': (sum(1 for row in recall_rows if row['recall_at_3'])
                                        / len(recall_rows) if recall_rows else None),
            'query_state_recall_n': len(recall_rows),
            'history_version_accuracy': (sum(1 for row in history if row['version_correct'])
                                         / len(history) if history else None),
            'history_version_accuracy_upper_bound_reason': (
                'window_end_exceeds_all_event_times' if args.time_mode == 'window_end' else None),
            'history_probe_mode': args.history_probe_mode,
            'diagnostic': ({'diagnostic_only': True,
                            'target_time_source': 'gold_version_time(query.as_of)',
                            'history_version_accuracy': (
                                self.diagnostic_correct / self.diagnostic_n
                                if self.diagnostic_n else None),
                            'n': self.diagnostic_n,
                            'note': 'diagnostic_only：只量“若时间已知”的历史版本上界，'
                                    '不得并入主表，也不得当作自动关联证据'}
                           if args.history_probe_mode == 'event_time' else None),
            'defer_store_memory': bool(args.defer_store_memory),
            'pending_count': self.pending_count,
            'pending_bytes': self.pending_bytes,
            'history_version_n': len(history),
            'latest_version_accuracy': (sum(1 for row in latest if row['version_correct'])
                                        / len(latest) if latest else None),
            'latest_version_n': len(latest),
            'em': em,
            'metric_scope': 'em/f1 仅 latent 路径（hybrid_fallback=False）；parametric 回退单列，不混算',
            'hybrid_fallback': bool(args.hybrid_fallback),
            'hybrid_fallback_count': len(fallback),
            'hybrid_fallback_rate': (len(fallback) / len(self.rows)) if self.rows else 0.0,
            'parametric_fallback': {'count': len(fallback), 'em': fallback_em, 'f1': fallback_f1,
                                    'note': 'parametric 回退答案未经 latent 读取，不计入主表 em/f1'},
            'f1': f1_score,
            'answer_n': len(answered),
            'no_state_match_rate': (1 - len(answered) / len(self.rows)) if self.rows else 0.0,
            'mean_write_seconds': (self.write_seconds / self.writes) if self.writes else 0.0,
            'mean_read_seconds': (sum(self.read_seconds) / len(self.read_seconds)
                                  if self.read_seconds else None),
            'store_bytes': self.store_bytes,
            'mean_read_latent_slots': (sum(self.slots) / len(self.slots)) if self.slots else 0.0,
            'notes': [
                'oracle_state 只是上界复现；predicted_state / global_e2e 才是自动关联证据。',
                'time_mode=window_end 与 time_mode=as_of 必须分列统计，禁止混算；'
                'TempLAMA 上 as_of == query.time == 金标版本时间戳。',
                'recency_update=off 是保守基线（版本链建不起来）；on 是 TempLAMA 上唯一'
                '能形成多版本状态的配置。两者都要报。',
                'recency_min_confidence 未在独立验证集校准前不得设非零值当作结论。',
            ],
        }


# ---------------------------------------------------------------------------
# data helpers
# ---------------------------------------------------------------------------
def load_episodes(args: argparse.Namespace) -> List[Dict[str, Any]]:
    path = Path(args.data_root) / f'{args.split}.json'
    if not path.exists():
        raise FileNotFoundError(f'missing prepared split: {path}')
    episodes = read_json(path)
    episodes = episodes[args.shard::args.shards]
    return episodes[:args.limit] if args.limit else episodes


def window_end_of(episode: Mapping[str, Any]) -> Optional[str]:
    """episode 观测窗口末刻 = 已观测事件的最大时间（公开可得，不含任何金标字段）。"""
    times = [event['time'] for event in episode.get('events') or []]
    return max(times) if times else None


def primary_states(facts: Sequence[Mapping[str, Any]]) -> Dict[str, str]:
    """gold state_key → 写入后占据最多的预测 state_id（评估用，不参与路由）。"""
    counts: Dict[str, Counter] = defaultdict(Counter)
    for fact in facts:
        if fact.get('state_id') and fact.get('gold'):
            counts[fact['gold']][fact['state_id']] += 1
    return {gold: counter.most_common(1)[0][0] for gold, counter in counts.items()}


def make_resolver(args: argparse.Namespace) -> UpdateResolver:
    return UpdateResolver(recency_update=bool(args.recency_update),
                          recency_min_confidence=float(args.recency_min_confidence))


def new_runtime(args: argparse.Namespace, components: Components) -> PredictedRuntime:
    codec = components.codec
    store = IncrementalCurrentStore(codec, key_dimensions=SEMANTIC_DIMENSIONS,
                                    budget_mode=args.budget_mode)
    return PredictedRuntime(components.reader, codec, components.markers, components.index,
                            store, StateIndex(), router=StateRouter(top_k=args.top_k),
                            resolver=make_resolver(args), extractor=FactExtractor(),
                            query_parser=QueryParser(), time_mode=args.time_mode,
                            allow_as_of=(args.time_mode == 'as_of'),
                            hybrid_fallback=args.hybrid_fallback,
                            defer_store_memory=args.defer_store_memory, top_k=args.top_k)


def write_episode(runtime: PredictedRuntime, episode: Mapping[str, Any],
                  counter: PublicEventCounter) -> Tuple[List[Dict[str, Any]], int, float]:
    """把 episode 的事件写成公开写视图后写入运行时；返回事实台账（含评估用金标）。"""
    facts, writes, seconds = [], 0, 0.0
    for event in episode.get('events') or []:
        public_id = counter.next_id()
        view = public_write_event(event, public_id)   # 只保留 id/text/time
        result = runtime.write_predicted(view)        # 路由路径只见公开视图
        writes += 1
        seconds += float(result['write_seconds'])
        for record in result['receipts']:
            facts.append({'fact_id': record['fact_id'],
                          'state_id': record['predicted_state_id'],
                          'gold': event.get('state_key'),      # 评估台账，不进路由
                          'episode': episode['id'],
                          'decision': record['decision']})
    return facts, writes, seconds


def score_query(runtime: PredictedRuntime, episode: Mapping[str, Any],
                query: Mapping[str, Any], state_id: Optional[str],
                result: Mapping[str, Any], gold_states: Mapping[str, str],
                args: argparse.Namespace,
                diagnostic: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    answers = list(query.get('answers') or [])
    prediction = str(result.get('answer') or '')
    gold_type = str(query.get('type') or '')
    gold_version_time = query.get('as_of') or query.get('time')
    candidates = [candidate['state_id'] for candidate in result.get('candidates') or []]
    gold_state_id = gold_states.get(str(query.get('state_key') or ''))
    selected_time = result.get('selected_version_time')
    return {
        'tag': args.tag, 'mode': args.mode, 'time_mode': args.time_mode,
        'episode': episode['id'], 'query_id': query.get('id'),
        'question': query.get('question'), 'time_source': result.get('time_source'),
        'target_time': result.get('target_time'),
        'state_id': result.get('state_id') if state_id is None else state_id,
        'no_state_match': bool(result.get('no_state_match')),
        'selected_version_id': result.get('selected_version_id'),
        'selected_version_time': selected_time,
        'version_role': result.get('version_role'),
        'answer': prediction,
        'em': exact(prediction, answers) if answers else 0,
        'hybrid_fallback': bool(result.get('hybrid_fallback')),
        'fallback': result.get('fallback'),
        'f1': token_f1(prediction, answers) if answers else 0.0,
        'gold_type': gold_type,
        'gold_state': query.get('state_key'),          # 仅评估
        'gold_version_time': gold_version_time,        # 仅评估
        'recall_at_1': None if gold_state_id is None else int(
            bool(candidates[:1]) and candidates[0] == gold_state_id),
        'recall_at_3': None if gold_state_id is None else int(gold_state_id in candidates[:3]),
        'version_correct': bool(selected_time) and selected_time == gold_version_time,
        'read_latent_slots': result.get('read_latent_slots', 0),
        'read_seconds': (result.get('timings') or {}).get('read_seconds'),
        'diagnostic_selected_version_time': (diagnostic or {}).get('selected_version_time'),
        'diagnostic_version_correct': (None if diagnostic is None else bool(
            diagnostic.get('selected_version_time')) and
            diagnostic.get('selected_version_time') == gold_version_time),
    }


def diagnostic_read(runtime: PredictedRuntime, state_id: Optional[str],
                    query: Mapping[str, Any], window_end: Optional[str],
                    args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    """diagnostic_only：用金标版本时间做上界读一次，只量“若时间已知”的历史准确率。"""
    if args.history_probe_mode != 'event_time' or state_id is None:
        return None
    if str(query.get('type') or '') == 'latest':
        return None
    gold_time = query.get('as_of') or query.get('time')
    if not gold_time:
        return None
    view = public_query(query, time_mode='window_end', window_end=window_end)
    reader_query = {'question': view['question'], 'time': gold_time}
    return runtime.read_state(state_id, view, target_time=gold_time,
                              reader_query=reader_query)


# ---------------------------------------------------------------------------
# modes
# ---------------------------------------------------------------------------
def run_predicted_state(args, episodes, components) -> Metrics:
    metrics = Metrics()
    for index, episode in enumerate(episodes):
        runtime = new_runtime(args, components)
        pool = [episode] + _neighbours(episodes, index, args.neighbours)
        counter = PublicEventCounter()
        facts: List[Dict[str, Any]] = []
        for member in pool:
            written, writes, seconds = write_episode(runtime, member, counter)
            facts.extend(written)
            metrics.writes += writes
            metrics.write_seconds += seconds
        gold_states = primary_states(facts)
        window_end = window_end_of(episode)
        for query in episode.get('queries') or []:
            result = runtime.answer_predicted(query, window_end=window_end)
            metrics.add_row(score_query(
                runtime, episode, query, None, result, gold_states, args,
                diagnostic_read(runtime, result.get('state_id'), query, window_end, args)))
        own = [fact for fact in facts if fact['episode'] == episode['id']]
        metrics.add_facts(own)
        metrics.add_decisions(runtime.decisions, {fact['fact_id'] for fact in own})
        metrics.add_store(runtime.store)
    return metrics


def run_global_e2e(args, episodes, components) -> Metrics:
    metrics = Metrics()
    runtime = new_runtime(args, components)
    counter = PublicEventCounter()
    facts: List[Dict[str, Any]] = []
    for episode in episodes:
        written, writes, seconds = write_episode(runtime, episode, counter)
        facts.extend(written)
        metrics.writes += writes
        metrics.write_seconds += seconds
    gold_states = primary_states(facts)
    for episode in episodes:
        window_end = window_end_of(episode)
        for query in episode.get('queries') or []:
            result = runtime.answer_predicted(query, window_end=window_end)
            metrics.add_row(score_query(
                runtime, episode, query, None, result, gold_states, args,
                diagnostic_read(runtime, result.get('state_id'), query, window_end, args)))
    metrics.add_facts(facts)
    metrics.add_decisions(runtime.decisions)
    metrics.add_store(runtime.store)
    return metrics


def run_oracle_state(args, episodes, components) -> Metrics:
    """上界复现：状态身份直接取金标 state_key，其余（读版本 + 读取）与 Predicted 完全一致。"""
    metrics = Metrics()
    runtime = new_runtime(args, components)
    counter = PublicEventCounter()
    gold_to_state: Dict[str, str] = {}
    oracle_facts: List[Dict[str, Any]] = []
    for episode in episodes:
        for event in episode.get('events') or []:
            view = public_write_event(event, counter.next_id())
            gold = str(event.get('state_key') or '')       # Oracle 对照：仅此处读取
            if not gold:
                raise ValueError('oracle_state needs the gold state_key on every event')
            entity, _, relation = gold.partition(':')
            memory = components.reader.encode_event({'time': view['time'], 'text': view['text']})
            key = components.index.key(memory)
            state_id = gold_to_state.get(gold)
            if state_id is None:
                state_id = runtime.state_index.new_state_id()
                gold_to_state[gold] = state_id
                decision = StateDecision('CREATE', state_id, 1.0, 'oracle_state:gold_key')
            else:
                decision = StateDecision('UPDATE', state_id, 1.0, 'oracle_state:gold_key')
            version_event = {'id': view['id'], 'time': view['time'], 'text': view['text']}
            receipt = runtime.store.ingest_decision(version_event, memory, key, decision)
            if decision.action == 'CREATE':
                # 用 add(state_id=...) 保留预分配的编号（add_keys 会另发一个新编号）
                runtime.state_index.add(
                    {'id': receipt['id'], 'subject': entity, 'attribute': relation,
                     'value': view['text'], 'observed_at': view['time']},
                    semantic_key=key, version_id=receipt['id'], state_id=state_id)
            else:
                runtime.state_index.commit(state_id, receipt['id'], observed_at=view['time'],
                                           value_fingerprint=value_fingerprint(view['text']))
            metrics.writes += 1
            oracle_facts.append({'fact_id': view['id'], 'state_id': state_id, 'gold': gold,
                                 'episode': episode['id'], 'decision': decision.action})
    metrics.add_facts(oracle_facts)
    for episode in episodes:
        window_end = window_end_of(episode)
        for query in episode.get('queries') or []:
            view = public_query(query, time_mode=args.time_mode, window_end=window_end)
            state_id = gold_to_state.get(str(query.get('state_key') or ''))
            if state_id is None:
                result = {'answer': '', 'state_id': None, 'no_state_match': True,
                          'selected_version_id': None, 'selected_version_time': None,
                          'version_role': None, 'read_latent_slots': 0,
                          'time_source': view['time_source'], 'target_time': view['time'],
                          'candidates': [], 'timings': {}}
            else:
                result = runtime.read_state(state_id, view, target_time=view['time'])
                # 上界口径：关联步骤按定义完全正确，候选列表就是金标状态本身。
                result = result | {'time_source': view['time_source'],
                                   'target_time': view['time'],
                                   'candidates': [{'state_id': state_id, 'rank': 1}]}
            metrics.add_row(score_query(
                runtime, episode, query, state_id, result, gold_to_state, args,
                diagnostic_read(runtime, state_id, query, window_end, args)))
    metrics.add_decisions([])
    metrics.add_store(runtime.store)
    return metrics


def _neighbours(episodes: Sequence[Mapping[str, Any]], index: int,
                count: int) -> List[Mapping[str, Any]]:
    if count <= 0:
        return []
    picked = []
    for offset in range(1, count + 1):
        for candidate in (index - offset, index + offset):
            if 0 <= candidate < len(episodes):
                picked.append(episodes[candidate])
    return picked[:count]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='P3 state-level evaluator: oracle_state / predicted_state / global_e2e')
    parser.add_argument('--mode', choices=MODES, default='predicted_state')
    parser.add_argument('--time-mode', choices=TIME_MODES, default='window_end',
                        help='window_end = Predicted 正式口径；as_of = 仅受控对照')
    parser.add_argument('--data-root', default='/share/latent2/vcrm_icae/data/prepared')
    parser.add_argument('--split', default='templama_test')
    parser.add_argument('--limit', type=int, default=50)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--tag', default='state_eval')
    parser.add_argument('--out', default=str(REPO_ROOT / 'probe' / 'runs'))
    parser.add_argument('--budget-mode', default='retain_all', choices=['retain_all'])
    parser.add_argument('--recency-update', action='store_true', default=False,
                        help='打开 rule5b（观察时间更晚的排他属性覆盖）；默认 off = 保守基线')
    parser.add_argument('--recency-min-confidence', type=float, default=0.0,
                        help='rule5b 置信度门槛；未校准前保持 0.0')
    parser.add_argument('--hybrid-fallback', action='store_true', default=False,
                        help='无候选时允许 parametric 回退（默认关；开启后答案单列，不混入主表 em/f1）')
    parser.add_argument('--defer-store-memory', action='store_true', default=False,
                        help='DEFER 时把原始 full memory 也写进 store.pending（默认关；'
                             '每条 ≈1MB，且永不参与 select_version/bundle）')
    parser.add_argument('--history-probe-mode', choices=['window_end', 'event_time'],
                        default='window_end',
                        help='event_time = diagnostic_only：用金标版本时间量“若时间已知”的历史上界')
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--neighbours', type=int, default=4,
                        help='predicted_state 混合事实池的邻近 episode 数')
    parser.add_argument('--reader', choices=['icae', 'stub'], default='icae')
    parser.add_argument('--arch-exp', default='E3', choices=['E1', 'E2', 'E3', 'E4'])
    parser.add_argument('--hidden', type=int, default=4096, help='stub reader / local codec 维度')
    parser.add_argument('--memory-slots', type=int, default=128)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = parse_args(argv)
    started = time.perf_counter()
    episodes = load_episodes(args)
    components = build_components(args)
    runner = {'predicted_state': run_predicted_state, 'global_e2e': run_global_e2e,
              'oracle_state': run_oracle_state}[args.mode]
    metrics = runner(args, episodes, components)
    summary = metrics.summary(args, components, len(episodes))
    summary['wall_seconds'] = time.perf_counter() - started
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.tag}_{args.mode}_{args.time_mode}_{args.limit}"
    # F4：运行时决策日志（严格 11 字段、无 gold）与评分行（含 gold）分文件落盘。
    scores_path = out / f'{stem}.scores.jsonl'
    decisions_path = out / f'{stem}.decisions.jsonl'
    with scores_path.open('w') as handle:
        for row in metrics.rows:
            handle.write(json.dumps(row | {'gold_fields_present': True}, ensure_ascii=False) + '\n')
    with decisions_path.open('w') as handle:
        for record in metrics.decision_log:
            validate_decision_log(record, where='decisions.jsonl')
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
    summary['outputs'] = {'scores': str(scores_path), 'decisions': str(decisions_path),
                          'summary': str(out / f'{stem}.summary.json')}
    write_json(out / f'{stem}.summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == '__main__':
    main()
