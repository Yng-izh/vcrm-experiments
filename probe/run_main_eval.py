"""统一评测入口：论文主实验四组（A Text-RAG / B State-aligned Text / C Current-only / D Versioned）。

一组 = 一个 ``--group``，四组共用同一批 question_id、同一份官方数据、同一套输出协议::

    <runs-root>/<tag>/<group>/predictions.jsonl   逐题一行（含失败行）
    <runs-root>/<tag>/<group>/summary.json        组内汇总（分母/耗时/字节）
    <runs-root>/<tag>/<group>_judged.jsonl        二段式判分产物（可选）

四组定义（只改"记忆怎么读"，写入链路与数据完全一致）：

* ``text_rag``      —— A：纯文本检索（本题 haystack 内的词法检索 + 文本 reader），
  不经过 latent 状态层，作为 Text-RAG 基线。
* ``state_text``    —— B：关联层选状态/版本，但**读原始 session 文本**交给文本 reader。
  隔离"状态层选得准不准"与"latent 读得准不准"。
* ``current_only``  —— C：latent，``read_mode='current_only'``（只读当前版本）。
* ``versioned``     —— D：latent，``read_mode='trajectory'``（当前 + 紧邻历史版本）。

协议硬约束（全部落成常量/CLI 校验，改协议必须改这里）：

1. **分母永不缩水**：所有题都进 predictions.jsonl；失败的题 ``answer=""`` 且带
   ``error`` 字段，绝不从分母里消失。
2. **逐题只用本题 haystack**（t7 机制）：每题一个全新 runtime/索引，不跨题共享。
3. **写入顺序 = session date 升序**（``vcrm.lme_data`` 加载器保证）。
4. **cutoff 口径（t25）**：latent 与 text 组都用 ``question_date`` 作观测上界，
   历史时间锚只通过 ``QueryTimeResolver`` 折算成 ``window_end``；四个审计字段
   ``query_time_source`` / ``query_cutoff_source`` / ``query_window_end`` /
   ``query_time_anchor`` 逐题落盘，**不把"对话末刻"当默认等价**。
5. **``as_of_future_fallback=False`` 强制**：LongMemEval 的 window_end 是硬上界，
   回退读窗口之后的版本会把"2019 年"答成 2023 年的事实；该值写进 config 快照。
6. **字节口径**：latent 组 ``write_bytes = store.accounting()['total_bytes']``
   （payload + 索引键 + 元数据 + 序列化头，不含模型参数）；text 组记该题检索索引
   落盘字节。C/D 写入路径相同 → 同一题 write_bytes 必须逐字节相等（单测钉死）。
7. **B/C/D 共享前端缓存**：``--llm-config`` 时三组共用同一个
   ``--llm-cache-root``（默认 ``<runs-root>/<tag>/llm_cache``），杜绝前端随机性差异。

判分两段式（复用 ``probe/judge_dialogue_rows.py`` 的官方 prompt，绝不字符串近似）：

    # 第一段：产出 predictions（不需要任何 judge 凭据）
    python3 probe/run_main_eval.py --group versioned --data <S_JSON> --tag main_v1
    # 第二段：拿 judge 配置后补分
    python3 probe/run_main_eval.py --group versioned --data <S_JSON> --tag main_v1 \\
        --judge-config judge.json

本轮只做代码 + CPU 单测（stub reader/answerer），不跑 GPU、不跑官方评测。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vcrm.codec import VersionMarkers                      # noqa: E402
from vcrm.data_views import PublicEventCounter, public_write_event  # noqa: E402
from vcrm.fact_extractor import FactExtractor              # noqa: E402
from vcrm.index import SemanticIndex                       # noqa: E402
from vcrm.io import file_digest, write_json                # noqa: E402
from vcrm.lme_data import Episode, load_lme_s              # noqa: E402
from vcrm.query_alignment import StateVocabularyAligner     # noqa: E402
from vcrm.query_time import QueryTimeResolver              # noqa: E402
from vcrm.runtime import PredictedRuntime, QueryParser      # noqa: E402
from vcrm.state_index import SEMANTIC_DIMENSIONS, StateIndex, normalize_subject  # noqa: E402
from vcrm.state_router import ROUTER_MODES, StateRouter      # noqa: E402
from vcrm.store import IncrementalCurrentStore             # noqa: E402
from vcrm.update_resolver import UpdateResolver            # noqa: E402

#: 四组（A/B/C/D）—— 组名即目录名，禁止随意新增/改名（论文表格直接对齐）。
GROUPS: Tuple[str, ...] = ('text_rag', 'state_text', 'current_only', 'versioned')
#: 走 latent 存储/读取路径的组。
LATENT_GROUPS: Tuple[str, ...] = ('current_only', 'versioned')
#: C/D 的读取口径（t24）；B 用 pair 只为取"选中的版本"，实际读文本。
GROUP_READ_MODE: Dict[str, str] = {'state_text': 'pair',
                                   'current_only': 'current_only',
                                   'versioned': 'trajectory'}
#: predictions.jsonl 的字段契约（顺序固定，缺一个都不算合格行）。
PREDICTION_FIELDS: Tuple[str, ...] = (
    'question_id', 'question', 'type', 'answer', 'abstention', 'group',
    'checkpoint_sha256', 'data_sha256', 'config', 'n_sessions',
    'read_slots', 'input_tokens', 'write_bytes', 'write_seconds', 'read_seconds',
    'query_time_source', 'query_cutoff_source', 'query_window_end', 'query_time_anchor',
)
#: t25 的四个 query_time 审计字段（单独导出，便于外部校验）。
QUERY_AUDIT_FIELDS: Tuple[str, ...] = ('query_time_source', 'query_cutoff_source',
                                       'query_window_end', 'query_time_anchor')
#: 观测上界来源口径（本轮固定 question_date；"对话末刻"不是等价默认）。
CUTOFF_SOURCES: Tuple[str, ...] = ('question_date',)

#: 字节口径的可判定条件（T32-F7）。write_bytes 逐字节相等**只在**同机 + 同序
#: （先跑一组把前端缓存填满）+ CPU 确定性编码下可判定；GPU 上编码浮点非确定性会
#: 让 semantic_key 微变，进而翻转路由/压缩判定，状态数与字节数都会不同
#: （t32 实测 C/D 3/5 不等）。GPU 对比请改用 decision_counts / state_count。
WRITE_BYTES_COMPARE_NOTE = (
    'not_comparable: 单组运行无法自证跨组字节相等；逐字节相等仅在「同机 + 同序'
    '（先跑一组填满前端缓存）+ CPU 确定性编码」下可判定。GPU 上编码浮点非确定性会'
    '翻转路由/压缩判定（t32 实测 C/D write_bytes 3/5 不等），此时唯一诚实锚点是'
    'decision_counts 与 state_count 一致。')
#: 跨组对比锚点（GPU 可用）：同一写入代码路径下的决策计数与状态数。
COMPARABLE_ANCHORS: Tuple[str, ...] = ('decision_counts', 'state_count')

#: 文本组检索索引实现：lexical = 依赖零的词法基线；hybrid = 复用 text-RAG 的
#: BM25+BGE+RRF（需要 rank_bm25/sentence_transformers，CPU 单测不用）。
TEXT_INDEX_MODES: Tuple[str, ...] = ('lexical', 'hybrid')


# ---------------------------------------------------------------------------
# 摘要/哈希
# ---------------------------------------------------------------------------
def sha256_of(path: Optional[str]) -> str:
    """文件/目录哈希；未提供时返回 'none'（不猜、不留空）。"""
    if not path:
        return 'none'
    target = Path(path)
    if target.is_dir():
        import hashlib
        hasher = hashlib.sha256()
        for child in sorted(p for p in target.rglob('*') if p.is_file()):
            hasher.update(child.name.encode())
            hasher.update(file_digest(child).encode())
        return hasher.hexdigest()
    return file_digest(target)


def tokenize(text: str) -> List[str]:
    return re.findall(r'[a-z0-9]+', str(text).lower())


# ---------------------------------------------------------------------------
# 文本组（A / B 的读取端）
# ---------------------------------------------------------------------------
class StubTextAnswerer:
    """确定性假文本 reader：CPU 单测用，绝不冒充真实模型输出。"""

    name = 'stub_text_reader'

    def answer(self, prompt: str) -> Dict[str, Any]:
        return {'answer': f'stub-text:{len(prompt)}', 'prompt_tokens': len(prompt.split())}


class VLLMTextAnswerer:
    """vLLM /v1/completions 文本 reader（需 GPU 服务；本轮不跑）。"""

    name = 'vllm_text_reader'

    def __init__(self, url: str, model: str, max_new_tokens: int = 256,
                 retries: int = 6, timeout: float = 600.0,
                 key_env: Optional[str] = None) -> None:
        self.url, self.model = url, model
        self.max_new_tokens, self.retries, self.timeout = max_new_tokens, retries, timeout
        #: Bearer key 的环境变量名（t32-F6）。None/'' = 显式关闭鉴权（本地无鉴权 vLLM）。
        self.key_env = key_env

    def auth_header(self) -> Dict[str, str]:
        """``Authorization: Bearer <key>``。

        ``key_env`` 为空 → 不带头（向后兼容本地无鉴权服务）；给了 key_env 但环境变量
        缺失 → ``RuntimeError``，与 ``vcrm/llm_client.py`` 同语义：不静默发无鉴权请求
        （否则 401 会被重试循环吞成"服务不可用"，把配置错误伪装成网络故障）。
        """
        if not self.key_env:
            return {}
        key = os.environ.get(self.key_env)
        if not key:
            raise RuntimeError(
                f'vllm credentials missing: environment variable {self.key_env!r} is not set; '
                'refusing to send an unauthenticated request (no silent fallback)')
        return {'Authorization': 'Bearer ' + key}

    def answer(self, prompt: str) -> Dict[str, Any]:
        # key 在重试循环外解析：缺 key 立即抛，不会被 6 次重试吞掉。
        headers = {'Content-Type': 'application/json'} | self.auth_header()
        payload = {'model': self.model, 'prompt': prompt, 'temperature': 0.0,
                   'max_tokens': self.max_new_tokens, 'seed': 42}
        last: Optional[Exception] = None
        for attempt in range(self.retries):
            try:
                started = time.time()
                request = urllib.request.Request(
                    self.url, data=json.dumps(payload).encode(), headers=headers)
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = json.loads(response.read())
                return {'answer': str(body['choices'][0]['text']).strip(),
                        'prompt_tokens': int(body.get('usage', {}).get('prompt_tokens', 0)),
                        'seconds': time.time() - started}
            except Exception as exc:                        # noqa: BLE001
                last = exc
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f'vllm call failed after {self.retries} retries: {last}')


class LexicalItemIndex:
    """依赖零的本题内词法检索（A 组的 CPU 基线；每题独立，不跨题共享）。"""

    def __init__(self, texts: Sequence[str]) -> None:
        self.texts = list(texts)
        self._tokens = [tokenize(text) for text in self.texts]

    def search(self, query: str, topk: int) -> List[int]:
        wanted = set(tokenize(query))
        scored = []
        for index, tokens in enumerate(self._tokens):
            overlap = len(wanted & set(tokens))
            if overlap:
                scored.append((overlap, -index, index))
        scored.sort(reverse=True)
        return [index for _, _, index in scored[:topk]]

    def size_bytes(self) -> int:
        return sum(len(text.encode('utf-8')) for text in self.texts)


def build_text_index(mode: str, texts: Sequence[str], item: Mapping[str, Any]) -> Any:
    if mode == 'lexical':
        return LexicalItemIndex(texts)
    # hybrid：复用 lme_s_text_rag 修正后的 per-item 检索（BM25+BGE+RRF）。
    import importlib.util
    path = REPO_ROOT / 'lme_s_text_rag' / 'run_text_rag.py'
    spec = importlib.util.spec_from_file_location('lme_text_rag_for_main_eval', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_item_index(item)


def render_text_prompt(question: str, sessions: Sequence[Mapping[str, Any]]) -> str:
    """文本组提示词：只放本题被选中的 session 原文（含日期），不含任何金标字段。"""
    blocks = [f"[{session['date']}] {session['text']}" for session in sessions]
    return ('You are answering a question about the user using only the sessions below.\n'
            + '\n\n'.join(blocks) + f'\n\nQuestion: {question}\nAnswer:')


def episode_to_item(episode: Episode) -> Dict[str, Any]:
    """Episode → text-RAG 口径的 item（与 lme_s_text_rag.normalize_item 同形）。"""
    return {'question_id': episode.question_id,
            'question': episode.question,
            'question_type': episode.question_type,
            'sessions': [{'session_id': session.session_id, 'date': session.date,
                          'text': session.text} for session in episode.sessions]}


# ---------------------------------------------------------------------------
# latent 组（C / D）与 B 的写入端
# ---------------------------------------------------------------------------
class Components:
    def __init__(self, reader: Any, codec: Any, markers: Any, index: Any, label: str,
                 checkpoint_root: Optional[str] = None,
                 checkpoint_requested: Optional[str] = None) -> None:
        self.reader, self.codec, self.markers, self.index = reader, codec, markers, index
        self.label = label
        #: 实际加载权重所在的目录（--checkpoint 或 VCRM_TRAINING_ROOT 或 v6_loader 默认）。
        #: 曾经的问题：--checkpoint 只被 sha256 一下写进 summary，真实权重却由
        #: v6_loader / VCRM_TRAINING_ROOT 决定，两者不一致时结果无法归因。现在这个
        #: 字段是**权威来源**，并且会被写进每一行 row 的 config 快照。
        self.checkpoint_root = checkpoint_root
        self.checkpoint_requested = checkpoint_requested


def build_components(args: argparse.Namespace) -> Components:
    if args.reader == 'stub':
        from probe.run_state_eval import StubReader, local_e3_shaped_codec
        reader = StubReader(hidden=args.hidden, slots=args.memory_slots)
        codec, label = local_e3_shaped_codec(
            args.hidden, min(16, max(1, args.memory_slots // 2)),
            width=min(256, args.hidden), heads=2)
        return Components(reader, codec, VersionMarkers(args.hidden),
                          SemanticIndex(args.hidden, SEMANTIC_DIMENSIONS), f'stub+{label}')
    from vcrm.runtime import SlotMemoryRuntime
    import importlib.util
    path = REPO_ROOT / 'probe' / 'v6_loader.py'
    spec = importlib.util.spec_from_file_location('vcrm_main_eval_v6_loader', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config, default_root, _ = module.load_v6(args.arch_exp)
    training_root = resolve_training_root(args, default_root, module)
    runtime = SlotMemoryRuntime.from_checkpoint(config, training_root)
    from vcrm.reader_batch import attach_batch_encode
    attach_batch_encode(runtime.reader)          # reader.py 受 sha256 锁定，只挂实例
    args.checkpoint_resolved = training_root     # 进 config 快照（可归因）
    return Components(runtime.reader, runtime.codec, runtime.markers, runtime.index,
                      f'icae+checkpoint:{training_root}', checkpoint_root=training_root,
                      checkpoint_requested=getattr(args, 'checkpoint', None))


def resolve_training_root(args: argparse.Namespace, default_root: str, module: Any) -> str:
    """决定**真正加载哪份权重**，并在冲突时大声失败（不再静默错配）。

    旧行为的真实风险：``--checkpoint`` 只被 ``sha256_of`` 一下写进 summary，权重却
    由 ``probe/v6_loader.load_v6`` 的默认目录（E3）或 ``VCRM_TRAINING_ROOT`` 决定。
    于是"报告里的 checkpoint"与"跑出来的权重"可以完全不是同一份。

    现在的优先级（第一个显式来源胜出）：``--checkpoint`` > ``VCRM_TRAINING_ROOT``
    > ``v6_loader`` 的默认训练目录。若两个显式来源同时存在且不一致 → 直接报错
    （除非显式 ``--allow-checkpoint-mismatch``），并把最终目录写进 row 的 config。
    """
    requested = getattr(args, 'checkpoint', None)
    env_root = os.environ.get('VCRM_TRAINING_ROOT')
    chosen = requested or env_root or default_root
    chosen = str(Path(chosen).expanduser().resolve())
    if requested and env_root:
        env_path = Path(env_root).expanduser()
        if env_path.exists() and env_path.resolve() != chosen:
            message = (f'--checkpoint {chosen} 与 VCRM_TRAINING_ROOT {env_path.resolve()} '
                       f'不一致：权重只会从 --checkpoint 加载，但环境变量会误导复现')
            if not getattr(args, 'allow_checkpoint_mismatch', False):
                raise SystemExit(message + '；确需如此请加 --allow-checkpoint-mismatch')
            print(f'[main-eval] WARNING {message}', flush=True)
    if not Path(chosen, 'manifest.json').exists():
        raise SystemExit(f'checkpoint 目录 {chosen} 缺少 manifest.json —— 拒绝加载，'
                         '避免把"没训练好的目录"当成冻结权重')
    return chosen


def build_frontend_pair(args: argparse.Namespace) -> Tuple[Any, Any, Any]:
    """返回 (extractor, query_parser, llm_client)；llm 前端需 --llm-config（CLI 已校验）。

    ``llm`` 前端的查询解析器带一个 ``StateVocabularyAligner``（T32-F1）：模型给出的
    subject/attribute 会逐 episode 对齐到写入侧真实状态键空间，避免
    ``subject='I'`` vs ``subject='1997 jeep wrangler'`` 这类永不相交导致的静默答空。
    """
    if args.frontend == 'rules':
        return FactExtractor(), QueryParser(), None
    from vcrm.runtime import build_frontend
    from vcrm.llm_client import CachedLLMClient, build_llm_client
    client = CachedLLMClient(build_llm_client(args.llm_config),
                             cache_root=args.llm_cache_root)
    extractor, parser = build_frontend(
        'llm', llm_client=client,
        query_parser_kwargs={'max_identities': 3, 'aligner': StateVocabularyAligner()})
    return extractor, parser, client


def new_runtime(args: argparse.Namespace, components: Components,
                extractor: Any, query_parser: Any, read_mode: str) -> PredictedRuntime:
    store = IncrementalCurrentStore(components.codec, key_dimensions=SEMANTIC_DIMENSIONS,
                                    budget_mode='retain_all')
    resolver = UpdateResolver(recency_update=bool(args.recency_update),
                              recency_min_confidence=float(args.recency_min_confidence))
    state_index = StateIndex()
    aligner = getattr(query_parser, 'aligner', None)
    if aligner is not None:                # 逐 episode 重建词表（每题独立 runtime）
        aligner.bind(state_index)
    return PredictedRuntime(components.reader, components.codec, components.markers,
                            components.index, store, state_index,
                            router=StateRouter(top_k=args.top_k,
                                               router_mode=getattr(args, 'router_mode',
                                                                   'identity')),
                            resolver=resolver,
                            extractor=extractor, query_parser=query_parser,
                            time_mode='window_end', allow_as_of=False,
                            hybrid_fallback=bool(args.hybrid_fallback),
                            defer_store_memory=bool(args.defer_store_memory),
                            as_of_future_fallback=False,      # LongMemEval：window_end 是硬上界
                            read_mode=read_mode, top_k=args.top_k,
                            # P1：多状态读取 + 证据优先（两个开关都可单独关掉做消融）
                            multi_state=bool(getattr(args, 'multi_state', False)),
                            max_query_states=int(getattr(args, 'max_query_states', 3)),
                            evidence_first=bool(getattr(args, 'evidence_first', False)),
                            evidence_top_k=int(getattr(args, 'evidence_top_k', 4)),
                            evidence_context_turns=int(getattr(args, 'evidence_context_turns', 1)),
                            evidence_max_units_per_session=(
                                int(args.evidence_max_units_per_session)
                                if int(getattr(args, 'evidence_max_units_per_session', 0)) > 0
                                else None),
                            # P2：历史目标时刻写进题面（与训练同一口径）
                            target_in_question=(
                                getattr(args, 'time_semantics', 'unified') == 'unified'))


def decision_counts(ledger: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    """写入决策计数（CREATE/UPDATE/DUPLICATE/DEFER）——GPU 上的跨组对比锚点。"""
    counts: Dict[str, int] = {}
    for entry in ledger:
        key = str(entry.get('decision') or 'unknown')
        counts[key] = counts.get(key, 0) + 1
    return counts


def write_episode(runtime: PredictedRuntime, episode: Episode) -> Dict[str, Any]:
    """把本题 haystack 逐 session（date 升序）写入；返回事实台账。"""
    counter = PublicEventCounter()
    texts: Dict[str, str] = {}
    ledger: List[Dict[str, Any]] = []
    session_map: Dict[str, str] = {}                     # public_id → episode session_id
    seconds = 0.0
    for session in episode.sessions:                     # 加载器已按 date 升序
        public_id = counter.next_id()
        texts[public_id] = session.text
        session_map[public_id] = session.session_id
        before = len(runtime.decisions)
        result = runtime.write_predicted(
            public_write_event({'text': session.text, 'time': session.date}, public_id))
        seconds += float(result['write_seconds'])
        for record in runtime.decisions[before:]:
            provenance = runtime.extractor.provenance(record['fact_id'])
            clause = ''
            if provenance is not None:
                clause = texts.get(provenance.source_id, '')[provenance.char_start:
                                                             provenance.char_end]
            ledger.append({'fact_id': record['fact_id'],
                           'state_id': record['predicted_state_id'],
                           'decision': record['decision'],
                           'version_id': record['new_version_id'],
                           'session_id': session.session_id,
                           'clause': clause})
    return {'ledger': ledger, 'write_seconds': seconds, 'n_sessions': len(episode.sessions),
            'session_map': session_map}


def resolve_window(args: argparse.Namespace, episode: Episode,
                   resolver: QueryTimeResolver) -> Any:
    """把一次查询拆成「目标时间」与「观测截止」两条边界（P2 统一口径）。

    unified（默认）：``target_time`` 只来自问题文本的历史锚（没有则 None = 读最新
    可见版本），``observation_cutoff`` = question_date。两者分别喂给
    ``answer_predicted(window_end=..., observation_cutoff=...)``，与训练侧
    ``vcrm.training.trajectory_samples`` 的时间语义一致。

    legacy：旧行为——历史锚直接被当成 ``window_end``，同时不再传
    ``observation_cutoff``（= 无可见性截止），历史年份被渲染成 reader 的
    ``Current date``。保留它只为复现 t34 之前的数字。
    """
    asked_at = episode.question_date
    if args.cutoff_source != 'question_date':            # pragma: no cover - 未来口径
        raise ValueError(f'unsupported cutoff source: {args.cutoff_source!r}')
    if getattr(args, 'time_semantics', 'unified') == 'legacy':
        resolution = resolver.resolve(episode.question, asked_at=asked_at)
        return resolution, resolution.window_end, None
    bounds = resolver.resolve_bounds(episode.question, asked_at=asked_at)
    resolution = resolver.resolve(episode.question, asked_at=asked_at)
    return resolution, bounds.target_time, bounds.observation_cutoff


# ---------------------------------------------------------------------------
# 四组执行体：统一返回 {'answer', 'read_slots', 'input_tokens',
#                        'write_bytes', 'write_seconds', 'read_seconds', 'extra'}
# ---------------------------------------------------------------------------
def run_text_group(args: argparse.Namespace, episode: Episode,
                   answerer: Any) -> Dict[str, Any]:
    """A：本题内文本检索 + 文本 reader，完全不经过 latent 状态层。"""
    item = episode_to_item(episode)
    started = time.perf_counter()
    index = build_text_index(args.text_index, [session['text'] for session in item['sessions']],
                             item)
    if hasattr(index, 'search'):
        picked = index.search(episode.question, args.top_k)
        chosen = [item['sessions'][position] for position in picked]
    else:                                                # hybrid 路径返回 (index, hits)
        index, hits = index
        lookup = {session['session_id']: session for session in item['sessions']}
        chosen = [lookup[session_id] for session_id in hits if session_id in lookup]
    write_bytes = int(index.size_bytes() if hasattr(index, 'size_bytes')
                      else len(json.dumps(item['sessions']).encode('utf-8')))
    prompt = render_text_prompt(episode.question, chosen)
    write_seconds = time.perf_counter() - started
    reply = answerer.answer(prompt)
    return {'answer': str(reply.get('answer') or ''),
            'read_slots': None,
            'input_tokens': int(reply.get('prompt_tokens') or 0),
            'write_bytes': write_bytes,
            'write_seconds': write_seconds,
            'read_seconds': float(reply.get('seconds') or 0.0),
            'extra': {'retrieved_session_ids': [session['session_id'] for session in chosen],
                      'text_index': args.text_index}}


def run_state_text_group(args: argparse.Namespace, episode: Episode, components: Components,
                         extractor: Any, query_parser: Any, resolver: QueryTimeResolver,
                         answerer: Any) -> Dict[str, Any]:
    """B：关联层选状态/版本，读**原始 session 文本**给文本 reader。"""
    runtime = new_runtime(args, components, extractor, query_parser, 'pair')
    written = write_episode(runtime, episode)
    resolution, target_time, observation_cutoff = resolve_window(args, episode, resolver)
    parsed = runtime.query_parser.parse(episode.question)
    chosen: List[Mapping[str, Any]] = []
    selected_version = None
    state_id = None
    reader_query = runtime._reader_query(
        {'question': episode.question, 'time': target_time, 'time_source': 'window_end'},
        observation_time=observation_cutoff, target_time=target_time)
    if parsed['resolved']:
        from vcrm.fact_schema import AtomicFact
        query_fact = AtomicFact(id='query', text=episode.question,
                                observed_at=target_time or 'unknown',
                                subject=parsed['subject'], attribute=parsed['attribute'],
                                value='', source_id='query')
        semantic_key = runtime.index.key(
            runtime.reader.encode(runtime.reader.question(reader_query)))
        for record in runtime.router.retrieve_candidates(
                query_fact, runtime.state_index, semantic_key=semantic_key):
            candidate = runtime.store.select_version(record.state_id, target_time)
            if candidate is not None:
                state_id, selected_version = record.state_id, candidate
                break
    wanted: set = set()
    if selected_version is not None:
        wanted |= {entry['session_id'] for entry in written['ledger']
                   if entry['version_id'] == selected_version['id']}
    # B 组也享受 P0/P1 的证据层：状态没命中时不再"空上下文 → 参数化幻觉"，
    # 而是把证据检索到的原始 session 交给文本 reader。
    evidence_units = []
    evidence_session_ids: List[str] = []
    if bool(getattr(args, 'evidence_first', False)) and int(getattr(args, 'evidence_top_k', 0)):
        hits = runtime.evidence.retrieve(
            episode.question, k=int(args.evidence_top_k),
            subject=parsed.get('subject'), attribute=parsed.get('attribute'),
            observed_until=observation_cutoff)
        evidence_units = [unit for _, unit in hits]
        # 证据单元记的是公开写入编号（event['id']），B 组要的是 episode 的
        # haystack session_id，必须经 write_episode 给出的映射反查。
        session_map = written['session_map']
        evidence_session_ids = [session_map[session_id]
                                for session_id in runtime.evidence.session_ids(evidence_units)
                                if session_id in session_map]
        wanted |= set(evidence_session_ids)
    if wanted:
        chosen = [session for session in episode.sessions
                  if session.session_id in wanted]
    write_bytes = int(runtime.store.accounting()['total_bytes'])
    extra = {'state_id': state_id,
             'selected_version_id': (selected_version or {}).get('id'),
             'read_session_ids': [session.session_id for session in chosen],
             'no_state_match': selected_version is None and not evidence_units,
             'evidence_units': len(evidence_units),
             'evidence_session_ids': evidence_session_ids,
             'parametric_only': False,
             'alignment': getattr(query_parser, 'last_alignment', []),
             'decision_counts': decision_counts(written['ledger']),
             'state_count': len(runtime.state_index.records),
             'retrieval_audit': getattr(runtime.router, 'last_retrieval_audit', {}),
             'n_facts': len(written['ledger'])}
    if not chosen:
        # T32-F2：连证据都没检索到任何 session 时，上下文才是真空的 —— 此时调文本
        # reader 只会得到参数化幻觉（实测 846–974 字、prompt 仅 36 token）。显式留空
        # 并单独记账，绝不让"空上下文长答案"混进 B 组非空率。
        # 反过来，只要证据检索到了 session（哪怕状态层没命中版本），就必须照常读取：
        # 旧逻辑在 `selected_version is None` 时就返回空，把证据层的召回直接扔掉了。
        extra['parametric_only'] = True
        return {'answer': '', 'read_slots': None, 'input_tokens': 0,
                'write_bytes': write_bytes, 'write_seconds': written['write_seconds'],
                'read_seconds': 0.0, 'extra': extra,
                'resolution': resolution, 'window_end': target_time,
                'observation_cutoff': observation_cutoff}
    prompt = render_text_prompt(episode.question, [
        {'date': session.date, 'text': session.text} for session in chosen])
    reply = answerer.answer(prompt)
    return {'answer': str(reply.get('answer') or ''),
            'read_slots': None,
            'input_tokens': int(reply.get('prompt_tokens') or 0),
            'write_bytes': write_bytes,
            'write_seconds': written['write_seconds'],
            'read_seconds': float(reply.get('seconds') or 0.0),
            'extra': extra,
            'resolution': resolution, 'window_end': target_time,
            'observation_cutoff': observation_cutoff}


def run_latent_group(args: argparse.Namespace, episode: Episode, components: Components,
                     extractor: Any, query_parser: Any, resolver: QueryTimeResolver,
                     read_mode: str) -> Dict[str, Any]:
    """C/D：latent 写入 + latent 读取（唯一变量 = read_mode）。"""
    runtime = new_runtime(args, components, extractor, query_parser, read_mode)
    written = write_episode(runtime, episode)
    resolution, target_time, observation_cutoff = resolve_window(args, episode, resolver)
    result = runtime.answer_predicted({'question': episode.question}, window_end=target_time,
                                      observation_cutoff=observation_cutoff)
    timings = result.get('timings') or {}
    evidence_stats = runtime.evidence.stats()
    return {'answer': str(result.get('answer') or ''),
            'read_slots': int(result.get('read_latent_slots') or 0),
            'input_tokens': None,
            'write_bytes': int(runtime.store.accounting()['total_bytes']),
            'write_seconds': written['write_seconds'],
            'read_seconds': float(timings.get('read_seconds') or 0.0),
            'extra': {'state_id': result.get('state_id'),
                      'selected_version_id': result.get('selected_version_id'),
                      'version_role': result.get('version_role'),
                      'no_state_match': bool(result.get('no_state_match')),
                      'hybrid_fallback': bool(result.get('hybrid_fallback')),
                      'read_mode': read_mode,
                      'parametric_only': bool(result.get('hybrid_fallback')),
                      'alignment': getattr(runtime.query_parser, 'last_alignment', []),
                      'decision_counts': decision_counts(written['ledger']),
                      'state_count': len(runtime.state_index.records),
                      'retrieval_audit': getattr(runtime.router, 'last_retrieval_audit', {}),
                      'attribute_fallback': bool(result.get('attribute_fallback')),
                      'multi_state': bool(result.get('multi_state')),
                      'multi_state_ids': result.get('multi_state_ids') or [],
                      'bundle_version_ids': result.get('bundle_version_ids') or [],
                      'bundle_evidence_ids': result.get('bundle_evidence_ids') or [],
                      'evidence_latent_slots': int(result.get('evidence_latent_slots') or 0),
                      'evidence_first': bool(result.get('evidence_first')),
                      'evidence_units': int(result.get('evidence_units') or 0),
                      'evidence_state_ids': result.get('evidence_state_ids') or [],
                      'evidence_only_read': bool(result.get('evidence_only_read')),
                      'evidence_stats': evidence_stats,
                      'n_facts': len(written['ledger'])},
            'resolution': resolution, 'window_end': target_time,
            'observation_cutoff': observation_cutoff}


# ---------------------------------------------------------------------------
# rows / summary
# ---------------------------------------------------------------------------
def prediction_row(args: argparse.Namespace, episode: Episode, outcome: Mapping[str, Any],
                   data_sha: str, checkpoint_sha: str) -> Dict[str, Any]:
    resolution = outcome.get('resolution')
    row = {'question_id': episode.question_id,
           'question': episode.question,
           'type': episode.question_type,
           'answer': str(outcome.get('answer') or ''),
           'abstention': bool(episode.abstention),
           'group': args.group,
           'checkpoint_sha256': checkpoint_sha,
           'data_sha256': data_sha,
           'config': config_snapshot(args),
           'n_sessions': len(episode.sessions),
           'read_slots': outcome.get('read_slots'),
           'input_tokens': outcome.get('input_tokens'),
           'write_bytes': int(outcome.get('write_bytes') or 0),
           'write_seconds': float(outcome.get('write_seconds') or 0.0),
           'read_seconds': float(outcome.get('read_seconds') or 0.0),
           'query_time_source': 'window_end',
           'query_time_semantics': getattr(args, 'time_semantics', 'unified'),
           'query_cutoff_source': args.cutoff_source,
           'query_window_end': outcome.get('window_end'),
           'query_target_time': outcome.get('window_end'),
           'query_observation_cutoff': outcome.get('observation_cutoff'),
           'checkpoint_resolved': getattr(args, 'checkpoint_resolved', None),
           'query_time_anchor': getattr(resolution, 'time_anchor', None)}
    row.update(outcome.get('extra') or {})
    return row


def config_snapshot(args: argparse.Namespace) -> Dict[str, Any]:
    """完整 CLI 快照（协议要求：可复现，含所有口径开关）。"""
    snapshot = {key: value for key, value in sorted(vars(args).items())
                if key not in ('judge_config', 'judge_cache')}
    snapshot['as_of_future_fallback'] = False        # 硬约束，不接受 CLI 覆盖
    snapshot['read_mode'] = GROUP_READ_MODE.get(args.group)
    # 真正加载权重的目录（--checkpoint > VCRM_TRAINING_ROOT > v6_loader 默认）。
    # 与 --checkpoint 不同时这里能一眼看出来，避免"报告 checkpoint ≠ 实跑权重"。
    snapshot['checkpoint_resolved'] = getattr(args, 'checkpoint_resolved', None)
    return snapshot


def failed_row(args: argparse.Namespace, episode: Episode, error: str,
               data_sha: str, checkpoint_sha: str) -> Dict[str, Any]:
    """失败题也进文件：answer="" + error，分母永不缩水。"""
    row = prediction_row(args, episode, {'answer': ''}, data_sha, checkpoint_sha)
    row['error'] = error
    return row


def _sum_decision_counts(rows: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    totals: Dict[str, int] = {}
    for row in rows:
        for key, value in (row.get('decision_counts') or {}).items():
            totals[str(key)] = totals.get(str(key), 0) + int(value)
    return totals


def compare_write_bytes(left_rows: Sequence[Mapping[str, Any]],
                        right_rows: Sequence[Mapping[str, Any]], *,
                        same_machine_and_order: bool = True,
                        deterministic_encoding: bool = True) -> Dict[str, Any]:
    """跨组对比（T32-F7）：字节口径只在同机同序 + 确定性编码时可判定。

    不可判定时**绝不返回 True/False 假象**，而是 ``comparable=False`` + 原因；
    GPU 上改用 ``decision_counts`` / ``state_count`` 作为锚点。
    """
    left = {row['question_id']: int(row.get('write_bytes') or 0) for row in left_rows}
    right = {row['question_id']: int(row.get('write_bytes') or 0) for row in right_rows}
    if set(left) != set(right):
        return {'comparable': False, 'equal': None,
                'reason': 'question_id sets differ between groups'}
    if not (same_machine_and_order and deterministic_encoding):
        counts_equal = all((left_row.get('decision_counts') or {}) ==
                           (right_row.get('decision_counts') or {})
                           for left_row, right_row in
                           zip(sorted(left_rows, key=lambda row: row['question_id']),
                               sorted(right_rows, key=lambda row: row['question_id'])))
        states_equal = all(int(left_row.get('state_count') or 0) ==
                           int(right_row.get('state_count') or 0)
                           for left_row, right_row in
                           zip(sorted(left_rows, key=lambda row: row['question_id']),
                               sorted(right_rows, key=lambda row: row['question_id'])))
        return {'comparable': False, 'equal': None, 'reason': WRITE_BYTES_COMPARE_NOTE,
                'anchors': {'decision_counts_equal': counts_equal,
                            'state_count_equal': states_equal}}
    per_item = [{'question_id': qid, 'left': left[qid], 'right': right[qid],
                 'equal': left[qid] == right[qid]} for qid in sorted(left)]
    return {'comparable': True, 'equal': all(item['equal'] for item in per_item),
            'per_item': per_item,
            'reason': 'byte-identical' if all(item['equal'] for item in per_item)
            else 'divergent even under deterministic settings: inspect encoder/cache state'}


def summarise(args: argparse.Namespace, rows: Sequence[Mapping[str, Any]],
              components_label: str, started: float,
              llm_client: Any = None) -> Dict[str, Any]:
    failed = [row['question_id'] for row in rows if row.get('error')]
    empty = [row['question_id'] for row in rows if not row.get('answer') and not row.get('error')]
    return {'tag': args.tag, 'group': args.group, 'dataset': 'longmemeval_s',
             'components': components_label, 'items': len(rows),
             'failed': len(failed), 'failed_ids': failed,
             'empty_answers': len(empty), 'empty_answer_ids': empty,
             'cutoff_source': args.cutoff_source,
             'as_of_future_fallback': False,
             'read_mode': GROUP_READ_MODE.get(args.group),
             'frontend': args.frontend, 'text_index': getattr(args, 'text_index', None),
             'llm_cache_root': args.llm_cache_root,
             'mean_write_bytes': (sum(row['write_bytes'] for row in rows) / len(rows)
                                  if rows else 0.0),
             'mean_write_seconds': (sum(row['write_seconds'] for row in rows) / len(rows)
                                    if rows else 0.0),
             'mean_read_seconds': (sum(row['read_seconds'] for row in rows) / len(rows)
                                   if rows else 0.0),
             'mean_read_slots': (sum(row['read_slots'] or 0 for row in rows) / len(rows)
                                 if rows else 0.0),
             'mean_input_tokens': (sum(row['input_tokens'] or 0 for row in rows) / len(rows)
                                   if rows else 0.0),
             'no_state_match': sum(1 for row in rows if row.get('no_state_match')),
             'decision_counts': _sum_decision_counts(rows),
             'state_count_total': sum(int(row.get('state_count') or 0) for row in rows),
             'state_count_mean': (sum(int(row.get('state_count') or 0) for row in rows)
                                  / len(rows) if rows else 0.0),
             'router_mode': getattr(args, 'router_mode', 'identity'),
             'write_bytes_equal_across_groups': 'not_comparable',
             'write_bytes_compare_reason': WRITE_BYTES_COMPARE_NOTE,
             'cross_group_anchors': list(COMPARABLE_ANCHORS),
             'time_semantics': getattr(args, 'time_semantics', 'unified'),
             'multi_state': bool(getattr(args, 'multi_state', False)),
             'evidence_first': bool(getattr(args, 'evidence_first', False)),
             'evidence_top_k': int(getattr(args, 'evidence_top_k', 0)),
             'checkpoint_resolved': getattr(args, 'checkpoint_resolved', None),
             'evidence_units_total': sum(int(row.get('evidence_units') or 0) for row in rows),
             'evidence_in_bundle': sum(1 for row in rows
                                       if (row.get('bundle_evidence_ids') or [])),
             'evidence_only_read': sum(1 for row in rows if row.get('evidence_only_read')),
             'multi_state_reads': sum(1 for row in rows if row.get('multi_state')),
             'parametric_only': sum(1 for row in rows if row.get('parametric_only')),
             'nonempty_answer': sum(1 for row in rows if str(row.get('answer') or '').strip()),
             'llm_cache': (llm_client.stats() if hasattr(llm_client, 'stats') else None),
             'wall_seconds': time.perf_counter() - started,
             'predictions': str(predictions_path(args))}


def predictions_path(args: argparse.Namespace) -> Path:
    return Path(args.runs_root) / args.tag / args.group / 'predictions.jsonl'


# ---------------------------------------------------------------------------
# 判分（二段式，复用 judge_dialogue_rows）
# ---------------------------------------------------------------------------
def judge_predictions(args: argparse.Namespace, episodes: Sequence[Episode],
                      rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    import importlib.util
    path = REPO_ROOT / 'probe' / 'judge_dialogue_rows.py'
    spec = importlib.util.spec_from_file_location('judge_rows_for_main_eval', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from vcrm.judge import load_config_file
    config = load_config_file(args.judge_config)
    cache = Path(args.judge_cache)
    gold = {episode.question_id: episode for episode in episodes}
    judged: List[Dict[str, Any]] = []
    for row in rows:
        episode = gold[row['question_id']]
        cell = {'answer': row['answer']}
        if row.get('error'):
            cell['error'] = row['error']
        proxy = {'type': episode.question_type, 'question': episode.question,
                 'answers': list(episode.answers), 'abstention': episode.abstention,
                 'pair': dict(cell), 'trajectory': dict(cell)}
        judged.append(dict(module.judge_row(proxy, config, cache))['pair']
                      | {'question_id': row['question_id'], 'group': args.group})
    out = predictions_path(args).parent / 'judged.jsonl'
    with out.open('w') as handle:
        for row in judged:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    scored = [row for row in judged if row.get('correct') is not None]
    summary = {'tag': args.tag, 'group': args.group, 'rows': str(out),
               'n': len(judged), 'scored': len(scored),
               'unscored': [row['question_id'] for row in judged
                            if row.get('correct') is None],
               'accuracy': (sum(row['correct'] for row in scored) / len(scored)
                            if scored else None),
               'scoring_rule': 'official_longmemeval_prompt',
               'judge_model': (scored[0].get('judge_model') if scored else None)}
    write_json(predictions_path(args).parent / 'judged_summary.json', summary)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--group', required=True, choices=list(GROUPS))
    parser.add_argument('--data', required=True,
                        help='官方 cleaned LongMemEval-S 数据（JSON 数组或 JSONL）')
    parser.add_argument('--tag', default='main_v1')
    parser.add_argument('--runs-root', default=str(REPO_ROOT / 'probe' / 'runs'))
    parser.add_argument('--limit', type=int, default=0, help='0 = 全部')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--reader', choices=['stub', 'icae'], default='stub',
                        help='latent 组 reader；icae 需冻结 E3 checkpoint + CUDA（本轮不跑）')
    parser.add_argument('--arch-exp', default='E3', choices=['E1', 'E2', 'E3', 'E4'])
    parser.add_argument('--checkpoint', default=None,
                        help='冻结 checkpoint 目录（权威：权重就从这里加载，并写进 '
                             'checkpoint_sha256 / checkpoint_resolved）；不给则依次退回 '
                             'VCRM_TRAINING_ROOT 与 v6_loader 默认目录')
    parser.add_argument('--hidden', type=int, default=32)
    parser.add_argument('--memory-slots', type=int, default=8)
    parser.add_argument('--text-reader', choices=['stub', 'vllm'], default='stub')
    parser.add_argument('--vllm-url', default='http://localhost:8001/v1/completions')
    parser.add_argument('--vllm-model', default='mistral-7b')
    parser.add_argument('--router-mode', choices=ROUTER_MODES, default='identity',
                        help='候选召回口径：identity = 默认两阶段硬过滤（行为不变）；'
                             'semantic = 语义 top-k + 主体硬门 + 属性软排序（P4 实验变量）')
    parser.add_argument('--vllm-key-env', default='VCRM_LLM_KEY',
                        help='Bearer key 的环境变量名（t32-F6；默认 VCRM_LLM_KEY）；'
                             "传空串显式关闭鉴权（本地无鉴权 vLLM）")
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--text-index', choices=list(TEXT_INDEX_MODES), default='lexical')
    parser.add_argument('--cutoff-source', choices=list(CUTOFF_SOURCES),
                        default='question_date')
    parser.add_argument('--frontend', choices=['rules', 'llm'], default='rules')
    parser.add_argument('--llm-config', default=None)
    parser.add_argument('--llm-cache-root', default=None,
                        help='B/C/D 共享前端缓存目录（默认 <runs-root>/<tag>/llm_cache）')
    parser.add_argument('--recency-update', action='store_true', default=False)
    parser.add_argument('--recency-min-confidence', type=float, default=0.0)
    parser.add_argument('--defer-store-memory', action='store_true', default=False)
    parser.add_argument('--hybrid-fallback', action='store_true', default=False)
    # ---- P0/P1/P2 读取路径改造开关（默认全开；--legacy-read-path 一次性回旧口径）--
    parser.add_argument('--multi-state', dest='multi_state', action='store_true', default=True,
                        help='允许多个状态一起读（P1；默认开）：一个 (subject, attribute) '
                             '装不下的多跳/多事实问题不再答空')
    parser.add_argument('--no-multi-state', dest='multi_state', action='store_false',
                        help='关掉多状态读取（= 旧行为）')
    parser.add_argument('--max-query-states', type=int, default=3)
    parser.add_argument('--evidence-first', dest='evidence_first', action='store_true',
                        default=True,
                        help='查询先检索证据记忆，再让状态索引决定版本（P0/P1；默认开）')
    parser.add_argument('--no-evidence-first', dest='evidence_first', action='store_false',
                        help='关掉证据优先读取（= 只走状态路由的旧行为）')
    parser.add_argument('--evidence-top-k', type=int, default=4)
    parser.add_argument('--evidence-context-turns', type=int, default=1,
                        help='证据单元保留几轮上下文（0 = 只留当前轮）')
    parser.add_argument('--evidence-max-units-per-session', type=int, default=0,
                        help='每 session 最多保留多少证据单元；0 = 不截断（默认，P0 不丢证据）')
    parser.add_argument('--time-semantics', choices=['unified', 'legacy'], default='unified',
                        help='unified（默认）：Current date = 观测截止、历史年份只进题面，'
                             '与训练同口径；legacy：历史锚直接当 window_end 注入 reader')
    parser.add_argument('--legacy-read-path', action='store_true', default=False,
                        help='等价于 --no-multi-state --no-evidence-first --time-semantics legacy，'
                             '只用于复现改造前的数字')
    parser.add_argument('--allow-checkpoint-mismatch', action='store_true', default=False,
                        help='允许 --checkpoint 与 VCRM_TRAINING_ROOT 不一致（默认直接报错）')
    parser.add_argument('--judge-config', default=None,
                        help='给就判分（第二段）；不给则只产出 predictions')
    parser.add_argument('--judge-cache', default=str(REPO_ROOT / 'probe' / 'runs'
                                                     / 'judge_cache'))
    parser.add_argument('--defer-judge', action='store_true', default=True,
                        help='默认只产出 predictions（两段式第一段）')
    args = parser.parse_args(argv)
    if args.frontend == 'llm' and not args.llm_config:
        parser.error('--frontend llm 需要 --llm-config PATH（缺配置不静默退回规则抽取）')
    if args.llm_cache_root is None:
        args.llm_cache_root = str(Path(args.runs_root) / args.tag / 'llm_cache')
    if args.shards < 1:
        parser.error('--shards must be >= 1')
    if args.legacy_read_path:
        args.multi_state, args.evidence_first = False, False
        args.time_semantics = 'legacy'
    if args.reader == 'icae' and not args.checkpoint \
            and not os.environ.get('VCRM_TRAINING_ROOT'):
        parser.error('--reader icae 需要 --checkpoint PATH 或 VCRM_TRAINING_ROOT'
                     '（否则无法确定加载的是哪份权重，结果不可复现/不可审计）')
    return args


def build_answerer(args: argparse.Namespace) -> Any:
    if args.text_reader == 'stub':
        return StubTextAnswerer()
    return VLLMTextAnswerer(args.vllm_url, args.vllm_model, args.max_new_tokens,
                            key_env=args.vllm_key_env)


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = parse_args(argv)
    started = time.perf_counter()
    episodes = load_lme_s(args.data)
    episodes.sort(key=lambda episode: episode.question_id)
    if args.limit:
        episodes = episodes[:args.limit]
    todo = episodes[args.shard::args.shards]
    data_sha = sha256_of(args.data)
    checkpoint_sha = sha256_of(args.checkpoint) if args.reader == 'icae' else 'none'
    components_label = 'text' if args.group == 'text_rag' else 'latent'

    resolver = QueryTimeResolver()
    answerer = build_answerer(args)
    components = extractor = query_parser = llm_client = None
    if args.group != 'text_rag':
        components = build_components(args)
        extractor, query_parser, llm_client = build_frontend_pair(args)
        components_label = components.label

    rows: List[Dict[str, Any]] = []
    for position, episode in enumerate(todo, start=1):
        try:
            if args.group == 'text_rag':
                outcome = run_text_group(args, episode, answerer)
            elif args.group == 'state_text':
                outcome = run_state_text_group(args, episode, components, extractor,
                                               query_parser, resolver, answerer)
            else:
                outcome = run_latent_group(args, episode, components, extractor,
                                           query_parser, resolver,
                                           GROUP_READ_MODE[args.group])
            rows.append(prediction_row(args, episode, outcome, data_sha, checkpoint_sha))
            print(f"[main-eval:{args.group}] {position}/{len(todo)} {episode.question_id} "
                  f"answer_chars={len(rows[-1]['answer'])} "
                  f"write_bytes={rows[-1]['write_bytes']}", flush=True)
        except Exception as exc:                          # noqa: BLE001 —— 失败进分母
            rows.append(failed_row(args, episode, f'{type(exc).__name__}: {exc}',
                                   data_sha, checkpoint_sha))
            print(f'[main-eval:{args.group}] ERROR {episode.question_id}: {exc!r}', flush=True)

    out = predictions_path(args)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    summary = summarise(args, rows, components_label, started, llm_client=llm_client)
    write_json(out.parent / 'summary.json', summary)
    if args.judge_config:
        summary['judge'] = judge_predictions(args, todo, rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == '__main__':
    main()
