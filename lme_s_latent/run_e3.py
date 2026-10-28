"""P4 LongMemEval-S latent pipeline: E3 ICAE latent memory + frozen reader.

**只产出 hypothesis；不做答案抽取/归一化/评分。** 官方 LongMemEval judge
（t12 spec / t14 接线）才是唯一评分口径；本脚本里的 em/f1 只是分级错误分析用的
diagnostic，不是结果。

口径（务必先读）：

1) **memory namespace = 本题 haystack。** 每题一个全新的 runtime
   （新 StateIndex / 新 IncrementalCurrentStore / 新 PublicEventCounter），只写
   本题的 haystack session，绝不跨题共享索引，因此不存在跨题泄漏，也不存在
   session_id 重名错配。合法 namespace 只由题目自身的输入定义，**不等同于
   Oracle episode**（LongMemEval-S 没有 episode 标签，答案会话未知）。

2) **时间通道（t13 冻结 + t3 审计 + t15 时间戳口径）。** 查询只经
   ``vcrm.data_views.public_query(time_mode='window_end', window_end=...)``：
   ``window_end`` 由 ``vcrm.query_time.QueryTimeResolver`` 产出 —— 问题文本里有
   可唯一折算的历史时间锚就用锚折算出的**区间末刻**，否则才用提问时刻；
   ``question_date`` 从不被当作历史目标时间，也从不直接透传成 ``as_of``。
   观察时间（fact.observed_at）取自 session 日期，不取索引写入墙钟。

3) **写入走 Predicted 无 Oracle 链路。** FactExtractor（多事实拆分 + 指代消解 +
   属性归一化）→ StateRouter → UpdateResolver → IncrementalCurrentStore，
   全部通过 ``vcrm.runtime.PredictedRuntime.write_predicted``（公开写视图
   白名单 id/text/time）。

4) **分级错误分析**（逐样本写进 jsonl，汇总进 summary.json）：
   ``fact_extraction_failed`` / ``update_deferred`` / ``state_match_failed``
   / ``version_select_failed`` / ``reader_wrong``，外加 ``correct``、
   ``abstention_unscored``、``no_gold`` 三个显式分桶。判定按上游优先（最早失败的
   阶段），永不把上游失败混进 ``reader_wrong``。``update_deferred`` 单列
   「证据抽出来了但决策落 DEFER」：recency_update=off 时这一桶膨胀，就是长对话
   演化事实覆盖不足的度量（t15 要求）。
   ``correct`` / ``reader_wrong`` 的切分用的是 ``vcrm.metrics.exact/f1`` —— 与
   ``probe/run_state_eval.py`` **同一套函数**，不是第二套评分实现；它只用于分桶，
   权威分口径来自 P3 评测器（``--hybrid-fallback`` 默认 off）与官方 judge。

5) **本轮不跑 GPU、不跑官方评测。** ``--reader icae`` 需要冻结 E3 checkpoint +
   CUDA（由用户放行后才能执行）；CPU 通路用 ``--reader stub``（确定性假 reader，
   绝不冒充 ICAE 结果，summary 里显式标注 reader 与 components）。

已知限制（P4 正式跑之前必须知道）：
   ``QueryParser``（t6）是模板式解析，只认 "X lives in _X_" 这类结构；LongMemEval
   的自由问答大多解析不出 (subject, attribute) → ``no_state_match``，分级桶会大量
   落在 ``state_match_failed``。这是查询侧解析能力的限制，不是写入/版本层的问题，
   也是 P4 全量跑之前必须先决定的口径问题（见 README 的阻塞清单）。

用法（CPU 冒烟，5 题，stub reader）::

    python3 lme_s_latent/run_e3.py --data data/longmemeval_s_cleaned.json \\
        --reader stub --limit 5 --tag lme_s_latent_smoke

产出：
    <runs-dir>/<tag>/<question_id>.json   每题一条（官方链路可直接消费）
    <runs-dir>/<tag>.jsonl                逐样本诊断行（含分级错误字段）
    <runs-dir>/<tag>_summary.json         汇总（分口径统计）
    <runs-dir>/<tag>_errors.json          逐题异常（断点续跑保留）
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vcrm.codec import VersionMarkers
from vcrm.data import timestamp as normalize_timestamp
from vcrm.data_views import PublicEventCounter, public_write_event
from vcrm.index import SemanticIndex
from vcrm.io import read_records, write_json
from vcrm.metrics import exact, f1 as token_f1, normalize
from vcrm.query_time import QueryTimeResolver, TIME_ANCHORS
from vcrm.query_alignment import StateVocabularyAligner
from vcrm.runtime import PredictedRuntime, build_frontend
from vcrm.state_index import SEMANTIC_DIMENSIONS, StateIndex, normalize_subject
from vcrm.state_router import StateRouter
from vcrm.store import IncrementalCurrentStore
from vcrm.update_resolver import UpdateResolver

#: 分级错误分析的四条必需桶 + t15 要求的「本该 UPDATE 却落 DEFER」桶（上游优先排列）。
#: update_deferred 单列：长对话里"上个月搬去上海、去年搬到北京"这类演化叙述在
#: recency_update=off（默认）时全部落 DEFER，版本链建不起来 —— 这是演化事实覆盖
#: 不足的度量，绝不能混进 reader_wrong。
ERROR_BUCKETS: Sequence[str] = ('fact_extraction_failed', 'update_deferred',
                                'state_match_failed', 'version_select_failed',
                                'reader_wrong')
#: 额外显式分桶：正确样本 / 弃答样本 / 无金标样本（永不并入上面四桶）。
EXTRA_BUCKETS: Sequence[str] = ('correct', 'abstention_unscored', 'no_gold')
ALL_BUCKETS: Sequence[str] = ERROR_BUCKETS + EXTRA_BUCKETS
#: UPDATE 来源分口径（reason 前缀，t15 契约）。
UPDATE_SOURCE_PREFIXES: Sequence[str] = ('explicit_replacement',
                                         'exclusive_superseded_by_recency',
                                         'snapshot_superseded')


# ---------------------------------------------------------------------------
# data: LongMemEval-S parsing (same field contract as vcrm.data.longmemeval)
# ---------------------------------------------------------------------------
def render_session(turns: Sequence[Mapping[str, Any]]) -> str:
    """把一轮 session 渲染成文本；role 前缀**幂等**（P4-F1 修复）。

    ``vcrm.fact_extractor`` 每段只剥**一层** role 前缀
    （``fact_extractor.py`` 的 ``_ROLE_PREFIX`` 分支）。若这里对已带前缀的 content
    再叠一层，就会得到 ``user: user: Alice ...``，抽取器剥完剩 ``user: Alice``
    → ``subject_key='user_alice'``，而查询侧 ``QueryParser`` 给的是 ``'alice'``，
    两边对不上（路由靠语义相似度还能蒙到候选，但主体口径已经错了）。

    真实 LongMemEval 的 ``content`` 不带 role 前缀，但已渲染过的 transcript 会被
    重投（复跑、拼接、人工改数据），所以这里必须幂等：先把 content 里**自身 role**
    的重复前缀全部剥掉，再加且只加一层。回归用例见
    ``tests/test_lme_latent.py::RolePrefixIdempotencyTest``。
    """
    lines: List[str] = []
    for turn in turns:
        role = str(turn.get('role') or '')
        content = str(turn.get('content') or '')
        if role:
            pattern = re.compile(rf'^\s*{re.escape(role)}\s*:\s*', re.IGNORECASE)
            while pattern.match(content):
                content = pattern.sub('', content, count=1)
            content = f'{role}: {content}'
        lines.append(content)
    return '\n'.join(lines)


def parse_item(row: Mapping[str, Any]) -> Dict[str, Any]:
    """单题解析：haystack 三个对齐数组 → 按时间排序的 session 列表。

    字段契约与 ``vcrm.data.longmemeval()`` 完全一致（同一套 ``timestamp`` 归一化、
    同样的 role/content 渲染、同样的 question/answer/type/abstention 投影）；差别只
    在于本函数允许任意条数（``vcrm.data.longmemeval`` 硬要求 500 条），以便
    5 题冒烟。等价性由 ``tests/test_lme_latent.py::ParseParityTest`` 用 500 条合成
    数据逐字段对齐 ``vcrm.data.longmemeval()`` 钉死。
    """
    ids = list(row['haystack_session_ids'])
    dates = list(row['haystack_dates'])
    sessions = list(row['haystack_sessions'])
    if not len(ids) == len(dates) == len(sessions):
        raise ValueError(f'misaligned session ids/dates for {row.get("question_id")!r}')
    question_id = str(row['question_id'])
    parsed = []
    for session_id, date, turns in zip(ids, dates, sessions):
        parsed.append({'session_id': str(session_id),
                       'time': normalize_timestamp(date),
                       'text': render_session(turns)})
    parsed.sort(key=lambda session: session['time'])
    answer = row.get('answer')
    return {'question_id': question_id,
            'question': str(row['question']),
            'question_date': normalize_timestamp(row['question_date']),
            'question_type': str(row.get('question_type') or ''),
            'answers': [] if answer is None else [str(answer)],
            'abstention': question_id.endswith('_abs'),
            'sessions': parsed}


def load_items(path: str) -> List[Dict[str, Any]]:
    """读取 LongMemEval-S 数据文件（JSON 数组或 JSONL，自动识别）。"""
    records = read_records(path)
    if not isinstance(records, list) or not records:
        raise ValueError(f'no LongMemEval items found in {path}')
    return [parse_item(row) for row in records]


# ---------------------------------------------------------------------------
# components: reader / codec / markers / semantic index
# ---------------------------------------------------------------------------
class Components:
    """一次运行共用的编码组件（每题共享权重，但状态与存储每题独立）。"""

    def __init__(self, reader: Any, codec: Any, markers: Any, index: Any, label: str) -> None:
        self.reader, self.codec, self.markers, self.index = reader, codec, markers, index
        self.label = label


def build_components(args: argparse.Namespace) -> Components:
    if args.reader == 'stub':
        # 复用 P3 评测器（probe/run_state_eval.py）的同一份 CPU 假 reader / 同构 codec：
        # 单一来源，避免两份 stub 漂移；label 明确写 UNTRAINED，绝不冒充真实权重。
        from probe.run_state_eval import StubReader, local_e3_shaped_codec
        reader = StubReader(hidden=args.hidden, slots=args.memory_slots)
        codec, label = local_e3_shaped_codec(
            args.hidden, min(16, max(1, args.memory_slots // 2)),
            width=min(256, args.hidden), heads=2)
        markers = VersionMarkers(args.hidden)
        index = SemanticIndex(args.hidden, SEMANTIC_DIMENSIONS)
        return Components(reader, codec, markers, index, f'stub+{label}')
    from vcrm.runtime import SlotMemoryRuntime
    config, training_root, _ = _load_v6(args.arch_exp)
    runtime = SlotMemoryRuntime.from_checkpoint(config, training_root)
    return Components(runtime.reader, runtime.codec, runtime.markers, runtime.index,
                      f'icae+checkpoint:{training_root}')


def _load_v6(arch_exp: str):
    import importlib.util
    path = REPO_ROOT / 'probe' / 'v6_loader.py'
    spec = importlib.util.spec_from_file_location('vcrm_lme_v6_loader', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_v6(arch_exp)


def new_runtime(args: argparse.Namespace, components: Components) -> PredictedRuntime:
    """每题一个新的 runtime：状态索引 / Current 存储 / 事件编号全部独立。

    前端由 --frontend 决定（rules = 现有硬规则；llm = 通用模型前端，占位）。
    实验二的受控对照只换这一个变量，E3 / 状态层 / 版本层 / 读取层完全不动。
    """
    store = IncrementalCurrentStore(components.codec, key_dimensions=SEMANTIC_DIMENSIONS,
                                    budget_mode='retain_all')
    resolver = UpdateResolver(recency_update=bool(args.recency_update),
                              recency_min_confidence=float(args.recency_min_confidence))
    frontend = getattr(args, 'frontend', 'rules')
    # T32-F1：llm 前端的查询身份必须对齐到写入侧真实状态键空间，否则
    # subject='I' 与 subject='1997 jeep wrangler' 永不相交 → 静默答空。
    parser_kwargs = ({'max_identities': int(getattr(args, 'max_query_states', 3)),
                      'aligner': StateVocabularyAligner()}
                     if frontend == 'llm' else {})
    extractor, query_parser = build_frontend(
        frontend, llm_client=build_llm_client(args),
        query_parser_kwargs=parser_kwargs)
    state_index = StateIndex()
    aligner = getattr(query_parser, 'aligner', None)
    if aligner is not None:               # 每题独立 runtime → 逐题重建词表
        aligner.bind(state_index)
    return PredictedRuntime(components.reader, components.codec, components.markers,
                            components.index, store, state_index,
                            router=StateRouter(top_k=args.top_k), resolver=resolver,
                            extractor=extractor, query_parser=query_parser,
                            time_mode='window_end', allow_as_of=False,
                            defer_store_memory=bool(args.defer_store_memory),
                            hybrid_fallback=bool(args.hybrid_fallback),
                            as_of_future_fallback=bool(args.as_of_future_fallback),
                            top_k=args.top_k, frontend=frontend,
                            read_mode=getattr(args, 'read_mode', 'pair'),
                            multi_state=bool(getattr(args, 'multi_state', False)),
                            max_query_states=int(getattr(args, 'max_query_states', 3)))


def build_llm_client(args: argparse.Namespace):
    """构造模型客户端：给 ``--llm-config`` 就接 OpenAI 兼容客户端 + 磁盘缓存。

    配置格式与 ``vcrm/judge.py`` 同构（三个键名都接受 judge 别名）::

        {"model": "...", "base_url": "https://.../v1", "key_env": "OPENAI_API_KEY",
         "max_tokens": 512, "temperature": 0}

    没给 ``--llm-config`` 时保持旧语义：``--llm-model placeholder``（默认）→ None
    （``build_frontend`` 会退化成 ``PlaceholderLLM``，任何调用显式抛错），其它模型名
    仍然显式失败——绝不静默退回规则抽取。缓存目录默认 ``<runs-dir>/<tag>/llm_cache``，
    保证 B/C/D 三组读到同一份模型输出。
    """
    config_path = getattr(args, 'llm_config', None)
    if config_path:
        from vcrm.llm_client import build_llm_client as _build_client
        cache_root = (getattr(args, 'llm_cache_root', None)
                      or str(Path(args.runs_dir) / args.tag / 'llm_cache'))
        args.llm_cache_root = cache_root
        return _build_client(config_path, cache_root=cache_root)
    model = getattr(args, 'llm_model', None)
    if not model or model == 'placeholder':
        return None
    raise SystemExit(
        f'--llm-model {model!r} 需要配 --llm-config PATH 才能构造客户端'
        '（JSON: {model, base_url, key_env, max_tokens?, temperature?}）；'
        '在此之前 --frontend llm 会显式失败，绝不静默退回规则抽取。')


# ---------------------------------------------------------------------------
# write path: 本题 haystack → 公开写视图 → PredictedRuntime
# ---------------------------------------------------------------------------
def write_item(runtime: PredictedRuntime, item: Mapping[str, Any],
               counter: PublicEventCounter, max_sessions: int = 0) -> Dict[str, Any]:
    """把本题 haystack 逐 session 写入；返回事实台账（供分级错误分析用）。"""
    sessions = item['sessions'][:max_sessions] if max_sessions else item['sessions']
    texts: Dict[str, str] = {}
    ledger: List[Dict[str, Any]] = []
    writes, seconds = 0, 0.0
    for session in sessions:
        public_id = counter.next_id()
        view = public_write_event({'text': session['text'], 'time': session['time']},
                                  public_id)
        texts[public_id] = session['text']
        before = len(runtime.decisions)
        result = runtime.write_predicted(view)
        writes += 1
        seconds += float(result['write_seconds'])
        for record in runtime.decisions[before:]:
            provenance = runtime.extractor.provenance(record['fact_id'])
            clause = ''
            observed_at = session['time']
            if provenance is not None:
                source = texts.get(provenance.source_id, '')
                clause = source[provenance.char_start:provenance.char_end]
                observed_at = provenance.observed_at
            ledger.append({'fact_id': record['fact_id'],
                           'state_id': record['predicted_state_id'],
                           'decision': record['decision'],
                           'version_id': record['new_version_id'],
                           'observed_at': observed_at,
                           'session_id': session['session_id'],
                           'clause': clause})
    return {'ledger': ledger, 'writes': writes, 'write_seconds': seconds,
            'sessions': len(sessions)}


# ---------------------------------------------------------------------------
# graded error analysis
# ---------------------------------------------------------------------------
def contains_answer(text: Any, answers: Sequence[str]) -> bool:
    """金标答案的内容词是否全部出现在候选文本里（分级分析用的代理判定）。"""
    haystack = set(normalize(text).split())
    if not haystack:
        return False
    for answer in answers:
        tokens = set(normalize(answer).split())
        if tokens and tokens <= haystack:
            return True
    return False


def classify_error(item: Mapping[str, Any], result: Mapping[str, Any],
                   ledger: Sequence[Mapping[str, Any]], em: int) -> str:
    """上游优先的分级判定：最早失败的阶段决定桶，绝不把上游失败记成 reader_wrong。

    桶序（上游 → 下游）：
        fact_extraction_failed → update_deferred → state_match_failed
        → version_select_failed → reader_wrong
    """
    if not item['answers']:
        return 'no_gold'
    if item['abstention']:
        return 'abstention_unscored'
    if em:
        return 'correct'
    if not ledger:
        return 'fact_extraction_failed'
    answers = item['answers']
    evidence = [entry for entry in ledger if contains_answer(entry['clause'], answers)]
    if not evidence:
        # 答案证据从未被抽成事实 → 抽取阶段失败（不是路由/读取的错）。
        return 'fact_extraction_failed'
    attached = [entry for entry in evidence if entry['state_id']]
    if not attached:
        # 证据抽出来了，但决策落 DEFER（没有 current 版本）→ 演化事实没进版本链。
        # recency_update=off 时这一桶会显著膨胀，正是"长对话演化覆盖不足"的度量。
        return 'update_deferred'
    state_id = result.get('state_id')
    if result.get('no_state_match') or not state_id:
        return 'state_match_failed'
    own = [entry for entry in attached if entry['state_id'] == state_id]
    if not own:
        # 路由到了别的状态：正确证据所在状态没被选中。
        return 'state_match_failed'
    selected = result.get('selected_version_id')
    if not any(entry['version_id'] == selected for entry in own):
        # 状态对了，但选中的版本不含证据（时间上界/版本链问题）。
        return 'version_select_failed'
    return 'reader_wrong'


def update_source_counts(decisions: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    counts = {prefix: 0 for prefix in UPDATE_SOURCE_PREFIXES}
    counts['not_update'] = 0
    for record in decisions:
        if record['decision'] == 'UPDATE':
            prefix = str(record.get('reason', '')).split(':', 1)[0] or 'unknown'
            counts[prefix] = counts.get(prefix, 0) + 1
        else:
            counts['not_update'] += 1
    return counts


# ---------------------------------------------------------------------------
# rows / summary
# ---------------------------------------------------------------------------
def official_row(item: Mapping[str, Any], row: Mapping[str, Any]) -> Dict[str, Any]:
    """每题一条：官方链路（make_official_eval_input.py）只需要 question_id + hypothesis。

    其余字段是诊断附加项，官方脚本只读这两个键，多出来的键不影响链路。
    """
    return {'question_id': item['question_id'],
            'question': item['question'],
            'question_type': item['question_type'],
            'hypothesis': row['hypothesis'],
            'time_mode': row['time_mode'],
            'time_source': row['time_source'],
            'time_anchor': row['time_anchor'],
            'anchor_ambiguous': row['anchor_ambiguous'],
            'window_end': row['window_end'],
            'state_id': row['state_id'],
            'selected_version_id': row['selected_version_id'],
            'selected_version_time': row['selected_version_time'],
            'no_state_match': row['no_state_match'],
            'hybrid_fallback': row['hybrid_fallback'],
            'fallback_used': row['fallback_used'],
            'error_bucket': row['error_bucket'],
            'diagnostic_em': row['diagnostic_em'],
            'read_latent_slots': row['read_latent_slots'],
            'prompt_tokens': row['prompt_tokens'],
            'read_seconds': row['read_seconds']}


def build_row(item: Mapping[str, Any], resolved: Any, result: Mapping[str, Any],
              written: Mapping[str, Any], hypothesis: str, em: int, f1_score: float,
              decisions: Sequence[Mapping[str, Any]], bucket: str,
              args: argparse.Namespace) -> Dict[str, Any]:
    timings = result.get('timings') or {}
    candidates = [{'state_id': candidate.get('state_id'), 'rank': candidate.get('rank'),
                   'similarity': candidate.get('similarity')}
                  for candidate in result.get('candidates') or []]
    return {'question_id': item['question_id'],
            'question': item['question'],
            'question_type': item['question_type'],
            'hypothesis': hypothesis,
            'time_mode': 'window_end',
            'time_source': result.get('time_source'),
            'time_anchor': resolved.time_anchor,
            'anchor_ambiguous': bool(resolved.anchor_ambiguous),
            'anchor_text': resolved.anchor_text,
            'anchor_kind': resolved.anchor_kind,
            'anchor_reason': resolved.reason,
            'window_end': resolved.window_end,
            'ask_time': item['question_date'],
            'state_id': result.get('state_id'),
            'no_state_match': bool(result.get('no_state_match')),
            'selected_version_id': result.get('selected_version_id'),
            'selected_version_time': result.get('selected_version_time'),
            'version_role': result.get('version_role'),
            'read_mode': result.get('read_mode', getattr(args, 'read_mode', 'pair')),
            'frontend': getattr(args, 'frontend', 'rules'),
            'observation_cutoff': result.get('observation_cutoff'),
            'trajectory_length': int(result.get('trajectory_length') or 0),
            'hidden_future_versions': int(result.get('hidden_future_versions') or 0),
            'future_current_suppressed': bool(result.get('future_current_suppressed')),
            'multi_state_ids': list(result.get('multi_state_ids') or []),
            'candidates': candidates,
            'n_states': len({entry['state_id'] for entry in written['ledger']
                             if entry['state_id']}),
            'n_facts': len(written['ledger']),
            'n_sessions': written['sessions'],
            'decision_counts': dict(sorted(Counter(record['decision']
                                                   for record in decisions).items())),
            'update_source_counts': update_source_counts(decisions),
            'error_bucket': bucket,
            'diagnostic_em': int(em),
            'diagnostic_f1': float(f1_score),
            'read_latent_slots': int(result.get('read_latent_slots') or 0),
            'prompt_tokens': result.get('prompt_tokens'),
            'read_seconds': timings.get('read_seconds'),
            'write_seconds': written['write_seconds'],
            'recency_update': bool(args.recency_update),
            'recency_min_confidence': float(args.recency_min_confidence),
            'hybrid_fallback': bool(args.hybrid_fallback),          # 本次运行的分口径开关
            'fallback_used': bool(result.get('hybrid_fallback')),   # 本题是否真走了回退
            'as_of_future_fallback': bool(args.as_of_future_fallback),
            'version_after_target_time': bool(result.get('version_after_target_time'))}


def summarise(args: argparse.Namespace, components: Components, rows: Sequence[Mapping[str, Any]],
              errors: Mapping[str, str], started: float) -> Dict[str, Any]:
    buckets = Counter(row['error_bucket'] for row in rows)
    time_sources = Counter(str(row['time_source']) for row in rows)
    anchors = Counter(row['time_anchor'] for row in rows)
    ambiguous = sum(1 for row in rows if row['anchor_ambiguous'])
    decisions: Counter = Counter()
    update_sources: Counter = Counter()
    for row in rows:
        decisions.update(row['decision_counts'])
        update_sources.update(row['update_source_counts'])
    # hybrid_fallback 分口径：回退行本身就是 no_state_match=True 的行，必须从**全表**
    # 筛（从 answered 筛会永远得 0）。主表 diagnostic 只统计 latent 行。
    fallback_rows = [row for row in rows if row['fallback_used']]
    latent_rows = [row for row in rows if not row['fallback_used']]
    scored = [row for row in latent_rows if row['error_bucket'] not in ('no_gold',)]
    answered = [row for row in rows if not row['no_state_match']]
    read_seconds = [float(row['read_seconds']) for row in rows
                    if row.get('read_seconds') is not None]
    write_seconds = [float(row['write_seconds']) for row in rows]
    return {
        'tag': args.tag,
        'dataset': 'longmemeval_s',
        'protocol': 'latent_e3_predicted_state',
        'data': str(args.data),
        'reader': args.reader,
        'components': components.label,
        'arch_exp': args.arch_exp if args.reader == 'icae' else None,
        'items': len(rows),
        'errors': len(errors),
        'limit': args.limit,
        'shard': args.shard,
        'shards': args.shards,
        'max_sessions': args.max_sessions,
        'time_mode': 'window_end',
        'time_source_distribution': dict(sorted(time_sources.items())),
        'time_source_vocabulary': ['window_end', 'unknown'],
        'time_anchor_distribution': {anchor: int(anchors.get(anchor, 0))
                                     for anchor in TIME_ANCHORS},
        'anchor_ambiguous_count': ambiguous,
        'anchor_ambiguous_rate': (ambiguous / len(rows)) if rows else 0.0,
        'recency_update': bool(args.recency_update),
        'recency_min_confidence': float(args.recency_min_confidence),
        'defer_store_memory': bool(args.defer_store_memory),
        'hybrid_fallback': bool(args.hybrid_fallback),
        'as_of_future_fallback': bool(args.as_of_future_fallback),
        'version_after_target_time_count': sum(1 for row in rows
                                               if row['version_after_target_time']),
        'decision_counts': dict(sorted(decisions.items())),
        'update_source_counts': {key: int(update_sources.get(key, 0))
                                 for key in UPDATE_SOURCE_PREFIXES + ('not_update',)},
        'error_buckets': {bucket: int(buckets.get(bucket, 0)) for bucket in ALL_BUCKETS},
        'error_bucket_order': list(ERROR_BUCKETS),
        'no_state_match_rate': (1 - len(answered) / len(rows)) if rows else 0.0,
        'hybrid_fallback_count': len(fallback_rows),
        'hybrid_fallback_rate': (len(fallback_rows) / len(rows)) if rows else 0.0,
        'parametric_fallback': {
            'count': len(fallback_rows),
            'em': (sum(row['diagnostic_em'] for row in fallback_rows) / len(fallback_rows)
                   if fallback_rows else None),
            'f1': (sum(row['diagnostic_f1'] for row in fallback_rows) / len(fallback_rows)
                   if fallback_rows else None),
            'note': 'hybrid_fallback=off 时恒为空；开启后是"无记忆参数化读取"，'
                    '单独分口径，不混进 latent 主表。'},
        'metric_scope': 'diagnostic em/f1 仅 latent 路径（fallback_used=False）；'
                        'parametric 回退单列，不混算',
        'diagnostic': {'em': (sum(row['diagnostic_em'] for row in scored) / len(scored)
                              if scored else None),
                       'f1': (sum(row['diagnostic_f1'] for row in scored) / len(scored)
                              if scored else None),
                       'n': len(scored),
                       'note': 'diagnostic only：exact/f1 直接复用 vcrm.metrics 的同一套'
                               '函数（与 probe/run_state_eval.py 同源），只用于判定'
                               'correct/reader_wrong 分桶，不另算一套评分；权威分口径来自'
                               ' probe/run_state_eval.py（--hybrid-fallback 默认 off）'
                               '与官方 LongMemEval judge（t12/t14）。'},
        'mean_write_seconds': (sum(write_seconds) / len(write_seconds)
                               if write_seconds else 0.0),
        'mean_read_seconds': (sum(read_seconds) / len(read_seconds)
                              if read_seconds else None),
        'mean_read_latent_slots': (sum(row['read_latent_slots'] for row in rows) / len(rows)
                                   if rows else 0.0),
        'wall_seconds': time.perf_counter() - started,
        'notes': [
            'memory namespace = 本题 haystack（每题独立 runtime，不跨题共享索引）。',
            'as_of_future_fallback 默认 off：window_end 是硬上界，不读窗口之后写入的版本'
            '（runtime 默认 True 是为 TempLAMA 口径，本管线显式关闭并记录）。',
            'time_mode 只有 window_end：window_end = QueryTimeResolver 的产物'
            '（历史锚末刻优先，其次提问时刻），question_date 从不作为历史目标时间。',
            'time_anchor / anchor_ambiguous 是 sidecar 子口径，不进 PUBLIC_QUERY_KEYS；'
            'anchor_ambiguous=True 的样本单列，不混入正常口径。',
            'update_deferred 单列「本该 UPDATE 却落 DEFER」；recency_update=off 时'
            '该桶膨胀，不并入 reader_wrong。',
            'reader=stub 时 components 明确标注 UNTRAINED，结果不可作为论文数字。',
            'QueryParser 是模板式解析：LongMemEval 自由问答大多 no_state_match，'
            '这是查询侧解析限制，P4 全量跑前需先定口径。',
        ],
    }


# ---------------------------------------------------------------------------
# self-check: 主体口径对账（P4-F3）
# ---------------------------------------------------------------------------
def self_check(args: argparse.Namespace, items: Sequence[Mapping[str, Any]],
               components: Components, resolver: QueryTimeResolver) -> Dict[str, Any]:
    """打印「写入侧 state subject_key」vs「查询侧 QueryParser subject」对账表。

    用途：role 前缀叠加、主体归一化漂移会让两侧主体口径分叉 —— 路由靠语义相似度
    还能蒙到候选，但状态身份已经错了，hypothesis 会悄悄变空。这个自查就是提前
    把分叉打出来。**hypothesis 为空 = 链路失败，不是"低分"**，因此有空 hypothesis
    时本函数返回 ok=False（CLI 退出码 1）。
    """
    rows = []
    for item in items:
        runtime = new_runtime(args, components)
        written = write_item(runtime, item, PublicEventCounter(), args.max_sessions)
        resolved = resolver.resolve(item['question'], asked_at=item['question_date'])
        parsed = runtime.query_parser.parse(item['question'])
        result = runtime.answer_predicted({'question': item['question']},
                                          window_end=resolved.window_end)
        hypothesis = str(result.get('answer') or '')
        subjects = sorted({record.subject_key for record in runtime.state_index.records})
        # 状态身份用归一化主体键（state_index.normalize_subject）；QueryParser 给的是
        # 表面形式，所以对账必须先把查询侧也归一化，否则永远 False（假阴性）。
        query_key = normalize_subject(parsed['subject']) if parsed['subject'] else ''
        rows.append({'question_id': item['question_id'],
                     'query_subject': parsed['subject'],
                     'query_subject_key': query_key,
                     'query_attribute': parsed['attribute'],
                     'query_resolved': bool(parsed['resolved']),
                     'state_subjects': subjects,
                     'subject_aligned': bool(query_key) and query_key in subjects,
                     'n_states': len(runtime.state_index),
                     'n_facts': len(written['ledger']),
                     'hypothesis_chars': len(hypothesis),
                     'empty_hypothesis': not hypothesis,
                     'time_anchor': resolved.time_anchor,
                     'window_end': resolved.window_end})
    empty = [row['question_id'] for row in rows if row['empty_hypothesis']]
    unaligned = [row['question_id'] for row in rows if not row['subject_aligned']]
    print('[self-check] question_id | query_subject | state_subjects | aligned | hyp_chars')
    for row in rows:
        print(f"[self-check] {row['question_id']} | {row['query_subject']!r} "
              f"→ {row['query_subject_key']!r} | {row['state_subjects']} | "
              f"{row['subject_aligned']} | {row['hypothesis_chars']}")
    summary = {'items': len(rows), 'empty_hypothesis': empty, 'subject_unaligned': unaligned,
               'ok': not empty,
               'note': 'hypothesis 为空 = 链路失败（主体口径分叉/无状态命中），'
                       '不得只当低分处理；subject_unaligned 行需先修渲染或归一化。',
               'rows': rows}
    print(json.dumps({key: summary[key] for key in ('items', 'empty_hypothesis',
                                                    'subject_unaligned', 'ok')},
                     ensure_ascii=False))
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--data', required=True,
                        help='LongMemEval-S 数据文件（JSON 数组或 JSONL）')
    parser.add_argument('--tag', default='lme_s_latent_e3')
    parser.add_argument('--runs-dir', default=str(REPO_ROOT / 'lme_s_text_rag' / 'runs'),
                        help='与 make_official_eval_input.py 的 --runs-dir 对齐（默认 '
                             'lme_s_text_rag/runs，既有官方链路直接可用）')
    parser.add_argument('--limit', type=int, default=0, help='0 = 全部')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--max-sessions', type=int, default=0,
                        help='0 = 全部 session；>0 只用于冒烟，summary 会记录该值')
    parser.add_argument('--no-resume', action='store_true',
                        help='默认 resume：已存在 <runs-dir>/<tag>/<qid>.json 的题跳过')
    parser.add_argument('--reader', choices=['stub', 'icae'], default='stub',
                        help='stub = CPU 确定性假 reader（默认）；icae = 冻结 E3 ICAE，'
                             '需 checkpoint + CUDA，须用户放行')
    parser.add_argument('--arch-exp', default='E3', choices=['E1', 'E2', 'E3', 'E4'])
    parser.add_argument('--hidden', type=int, default=32, help='stub reader / 同构 codec 维度')
    parser.add_argument('--memory-slots', type=int, default=8, help='stub reader slot 数')
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--recency-update', action='store_true', default=False,
                        help='打开 UpdateResolver 的 recency 策略（默认 off = 保守基线）')
    parser.add_argument('--recency-min-confidence', type=float, default=0.0,
                        help='未在独立验证集校准前不得设非零值当结论')
    parser.add_argument('--defer-store-memory', action='store_true', default=False,
                        help='DEFER 时把原始 full memory 写进 store.pending（默认关）')
    parser.add_argument('--as-of-future-fallback', action='store_true', default=False,
                        help='窗口末端早于该状态首个版本时是否回退读最新版本。默认 **off**：'
                             'LongMemEval 的历史锚（如 in 2019）是硬上界，回退会把 2023 的'
                             '事实读成 2019 的答案；runtime 默认 True 是为 TempLAMA 口径，'
                             '本管线显式关掉并记录')
    parser.add_argument('--self-check', action='store_true', default=False,
                        help='只做主体口径对账：打印 QueryParser subject vs state '
                             'subject_key，不写 runs 产物；有空 hypothesis 时退出码 1')
    parser.add_argument('--hybrid-fallback', action='store_true', default=False,
                        help='透传给 PredictedRuntime 的 hybrid_fallback 分口径开关'
                             '（默认 off；口径与 probe/run_state_eval.py 的同名开关一致）')
    parser.add_argument('--frontend', choices=['rules', 'llm'], default='rules',
                        help='rules = 现有确定性硬规则（默认）；llm = 通用模型抽取/解析'
                             '（实验二 B 臂；模型未接入时显式失败，不静默退回规则）')
    parser.add_argument('--llm-model', default='placeholder',
                        help='模型标识；placeholder = 未接入（--frontend llm 会直接报错）')
    parser.add_argument('--llm-config', default=None,
                        help='OpenAI 兼容客户端配置路径 {model, base_url, key_env, '
                             'max_tokens?, temperature?}；给了就接真实客户端 + 磁盘缓存')
    parser.add_argument('--llm-cache-root', default=None,
                        help='LLM 输出缓存目录（默认 <runs-dir>/<tag>/llm_cache）')
    parser.add_argument('--multi-state', action='store_true', default=False,
                        help='允许一次读取多个相关状态（多跳/多事实问题；默认关）')
    parser.add_argument('--max-query-states', type=int, default=3,
                        help='多状态读取的状态数上限（同时作为解析器的 identities 上限）')
    parser.add_argument('--read-mode', choices=['pair', 'trajectory'], default='pair',
                        help='pair = 当时可见最新版本 + 1 个历史版本（默认）；'
                             'trajectory = 当时可见的全部版本')
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = parse_args(argv)
    started = time.perf_counter()

    items = load_items(args.data)
    items.sort(key=lambda item: item['question_id'])
    if args.limit:
        items = items[:args.limit]
    todo = items[args.shard::args.shards]

    components = build_components(args)
    resolver = QueryTimeResolver()
    if args.self_check:
        report = self_check(args, items[:args.limit or len(items)], components, resolver)
        return report if report['ok'] else sys.exit(1)

    runs_dir = Path(args.runs_dir)
    out_dir = runs_dir / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    errors_path = runs_dir / f'{args.tag}_errors.json'
    errors: Dict[str, str] = (json.loads(errors_path.read_text())
                              if errors_path.exists() else {})
    done = set() if args.no_resume else {path.stem for path in out_dir.glob('*.json')}
    todo = [item for item in todo if item['question_id'] not in done]
    print(f'[lme-latent] items={len(items)} done={len(done)} todo={len(todo)} '
          f'recency_update={bool(args.recency_update)} reader={args.reader}', flush=True)

    rows: List[Dict[str, Any]] = []

    for index, item in enumerate(todo):
        question_id = item['question_id']
        try:
            # 每题独立 namespace：索引 / 存储 / 编号器全部新建，绝不跨题共享。
            runtime = new_runtime(args, components)
            counter = PublicEventCounter()
            written = write_item(runtime, item, counter, args.max_sessions)
            resolved = resolver.resolve(item['question'], asked_at=item['question_date'])
            # t25 时间语义：target_time 与 observation_cutoff 来源彻底分开。
            #   target_time         = 问题文本解析出的历史锚（无锚 → None = 读最新可见）
            #   observation_cutoff  = question_date（提问时刻，公开合法信息）
            # **不能**再把历史锚当 cutoff：LongMemEval 里 2018 年的事实常在 2023 年的
            # 会话里才被讲述，用 2018 锚当截止会把唯一相关证据排除掉。
            bounds = resolver.resolve_bounds(item['question'],
                                             asked_at=item['question_date'])
            result = runtime.answer_predicted(
                {'question': item['question']}, window_end=bounds.target_time,
                read_mode=getattr(args, 'read_mode', 'pair'),
                observation_cutoff=bounds.observation_cutoff)
            hypothesis = str(result.get('answer') or '')
            em = exact(hypothesis, item['answers']) if item['answers'] else 0
            f1_score = token_f1(hypothesis, item['answers']) if item['answers'] else 0.0
            bucket = classify_error(item, result, written['ledger'], em)
            row = build_row(item, resolved, result, written, hypothesis, em, f1_score,
                            runtime.decisions, bucket, args)
            # t25 审计字段：target 与 cutoff 各自的来源（sidecar，不进官方两键）
            row['target_time'] = bounds.target_time
            row['query_time_mode'] = bounds.time_mode
            row['query_time_source'] = bounds.time_source
            row['query_time_allow_as_of'] = bounds.allow_as_of
            write_json(out_dir / f'{question_id}.json', official_row(item, row))
            rows.append(row)
            print(f"[lme-latent] {index + 1}/{len(todo)} {question_id} "
                  f"anchor={row['time_anchor']} states={row['n_states']} "
                  f"bucket={bucket} em={em}", flush=True)
        except Exception as exc:  # noqa: BLE001 - 逐题隔离
            errors[question_id] = repr(exc)
            write_json(errors_path, errors)
            print(f'[lme-latent] ERROR {question_id}: {exc!r}', flush=True)

    summary = summarise(args, components, rows, errors, started)
    jsonl_path = runs_dir / f'{args.tag}.jsonl'
    with jsonl_path.open('w') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    write_json(runs_dir / f'{args.tag}_summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == '__main__':
    main()
