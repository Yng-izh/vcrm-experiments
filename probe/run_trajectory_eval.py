#!/usr/bin/env python
"""实验一：Pair 读取 vs 全轨迹（trajectory）读取 —— 逐题配对对照。

问题（只问一件事）：
    "把当时已经可见的全部历史版本都交给 reader" 是否比
    "当时可见的最新版本 + 一个历史版本" 读得更好？

如果 trajectory 不优于 pair，就不继续开发超预算淘汰、轨迹压缩之类的机制。

题目来源（--question-source，默认 constructed）
------------------------------------------------
TempLAMA 官方查询是单版本填空，答案永远只依赖一个版本，结构上测不出"跨版本"。
默认用 ``probe/cross_version_questions.py`` 从事件版本链构造的四类诊断题：

    recall_year     某年的值（题面自带年份 → pair 用年份就能选对版本，A 不吃亏）
    ordinal_back    两次变更之前的值（需要版本序关系）
    count_values    区间内不同值的个数（需要全部版本）
    compare_years   两个年份是否相同（需要同时看到两个版本）

这些是**受控诊断题，不是 TempLAMA 官方分数**，不能用来声称开放域泛化。

受控变量（除了 read_mode 全部相同）
-----------------------------------
同一批 episode / 同一个事实池 / 同一个 E3 reader+codec / 同一状态身份来源
（默认金标 state_key，见 --identity）/ 同一个 observation_cutoff / 同一题面 /
同一个 pair 选择上界（见下方"对 A 臂的公平性"）。唯一差别是 read_state 的 read_mode。

时间的两条边界（t16，本实验必须先理清的东西）
---------------------------------------------
* ``observation_cutoff``：episode 观测窗口末刻 = **系统当时可能已经观测到的最后时刻**。
  它是可见性硬上界，且 ``allow_future_fallback`` 被强制关闭 —— store 里存着更晚的
  current 也不会被塞进 reader（结果里 ``future_current_suppressed`` /
  ``hidden_future_versions`` 记录被挡掉的未来信息量）。
* ``target_time``：问题文本指向的历史时刻，由 ``QueryTimeResolver.resolve_bounds``
  从题面折算（"In 2017" → 2017-12-31T23:59:59）。它只用于**选哪个版本**，
  不用于可见性。

对 A 臂的公平性
---------------
A 臂允许用题面里的明确时间信息选历史版本（否则就是把基线故意做弱）：
``pair_target_time`` 取该题"信息量最大的单个版本"——
recall_year = 金标年份；ordinal_back = 锚点年份；count_values / compare_years = 区间
最早年份（比 resolver 取最晚年份更有利于 A）。resolver 原样输出另存
``resolver_target_time`` 备查。

指标（只留三条 + 一条可达性）
-----------------------------
主指标   ``accuracy``：family 对应的判分（值匹配 / 数字 / 极性），两臂同一套规则
        （注：定长 latent 预算下"答出率"恒为 1，噪声率没有意义，故不报）。
指标2    ``mean_read_latent_slots``：平均读取 slot 数（预算代价）
指标3    ``wrong_current_answer_rate``：金标是历史版本、却答成了当时最新版本的值
        （只在取值型 family、且 current 取值≠金标的题上算）
可达性   ``all_required_in_bundle_rate``：题目所需的全部版本是否都在 bundle 里
        （pair 只有 2 个版本，结构上就低；这是解释 accuracy 差异的中间量）

GPU 纪律：真实 E3 固定 cuda:0，必须用 CUDA_VISIBLE_DEVICES 钉住；GPU 0/1 禁用。

用法：

    CUDA_VISIBLE_DEVICES=9 PYTHONPATH=.:probe env_py310/bin/python -u \
        probe/run_trajectory_eval.py --limit 80 --arch-exp E3 \
        --out probe/runs/traj_80 --tag traj_80

    # 只看题目/预算，不加载 7B reader（CPU 可跑）
    ... --plan-only
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vcrm.metrics import paired_bootstrap
from vcrm.query_time import QueryTimeResolver

DEFAULT_DATA_ROOT = '/share/latent2/vcrm_icae/data/prepared'
MODES = ('pair', 'trajectory')
FORBIDDEN_PHYSICAL_GPUS = ('0', '1')
IDENTITIES = ('gold', 'predicted')


def load_module(name: str, filename: str):
    path = REPO_ROOT / 'probe' / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_evaluator():
    return load_module('vcrm_probe_run_state_eval', 'run_state_eval.py')


def load_questions():
    return load_module('vcrm_probe_cross_version_questions', 'cross_version_questions.py')


def assert_gpu_policy() -> Dict[str, Any]:
    raw = os.environ.get('CUDA_VISIBLE_DEVICES')
    if raw is None or not raw.strip():
        raise SystemExit(
            '真实 E3 需要 GPU：请显式设置 CUDA_VISIBLE_DEVICES（例如 CUDA_VISIBLE_DEVICES=9）。\n'
            'GPU 0/1 为本工作区禁用设备。只做管线自检可加 --plumbing-stub。')
    ids = [part.strip() for part in raw.split(',') if part.strip()]
    bad = [value for value in ids if value in FORBIDDEN_PHYSICAL_GPUS]
    if bad:
        raise SystemExit(f'CUDA_VISIBLE_DEVICES={raw} 含禁用设备 {bad}：GPU 0/1 不可用')
    return {'cuda_visible_devices': raw, 'physical_gpu_ids': ids}


def git_commit() -> Optional[str]:
    try:
        out = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=str(REPO_ROOT),
                             capture_output=True, text=True, timeout=10)
    except Exception:                                     # noqa: BLE001
        return None
    return (out.stdout.strip() or None) if out.returncode == 0 else None


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='实验一：Pair 读取 vs 全轨迹读取（逐题配对，受控 read_mode）')
    parser.add_argument('--limit', type=int, default=80,
                        help='配对问题数上限（默认 80；建议 50-100）')
    parser.add_argument('--per-family', type=int, default=0,
                        help='每个 family 的题数上限（0 = limit // 4）')
    parser.add_argument('--question-source', choices=('constructed', 'templama'),
                        default='constructed',
                        help='constructed = 从版本链构造的跨版本诊断题（默认）；'
                             'templama = TempLAMA 官方查询（单版本填空，测不出跨版本）')
    parser.add_argument('--data-root', default=DEFAULT_DATA_ROOT)
    parser.add_argument('--split', default='templama_val',
                        help='TempLAMA validation（data-root 下的 templama_val.json）')
    parser.add_argument('--episodes', type=int, default=0,
                        help='只扫前 N 个 episode（0 = 全部；题目按 family 轮转取满 limit）')
    parser.add_argument('--episode-offset', type=int, default=0,
                        help='跳过前 N 个 episode，用于换一批未用过的验证状态')
    parser.add_argument('--neighbours', type=int, default=4,
                        help='混合事实池的邻近 episode 数（两臂共用同一池）')
    parser.add_argument('--identity', choices=IDENTITIES, default='gold',
                        help='状态身份来源：gold = 金标 state_key（默认，隔离读取阶段）；'
                             'predicted = 规则抽取 + 解析的预测身份')
    parser.add_argument('--min-versions', type=int, default=3)
    parser.add_argument('--max-versions', type=int, default=8)
    parser.add_argument('--max-slots', type=int, default=256,
                        help='轨迹读取的 decoder slot 预算（E3: 128 + k*16 + 2k）')
    parser.add_argument('--include-latest', action='store_true', default=False,
                        help='仅 --question-source templama 生效：保留 type=latest')
    parser.add_argument('--target-time', choices=['none', 'resolver'], default='none',
                        help='none = 用题目自带的 pair_target_time（默认）；'
                             'resolver = 全部改用 QueryTimeResolver 折算的上界')
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--recency-update', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--recency-min-confidence', type=float, default=0.0)
    parser.add_argument('--arch-exp', default='E3', choices=('E1', 'E2', 'E3', 'E4'))
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--merge', action='store_true', default=False,
                        help='合并各 shard 的 rows 并重算报告（CPU，不加载 reader）')
    parser.add_argument('--plumbing-stub', action='store_true', default=False,
                        help='只用 stub reader 自检管线（CPU、无 GPU，分数无效）')
    parser.add_argument('--plan-only', action='store_true', default=False,
                        help='只做造题与预算预检，不加载 reader、不跑实验')
    parser.add_argument('--tag', default='traj_eval')
    parser.add_argument('--out', default=str(REPO_ROOT / 'probe' / 'runs' / 'traj_eval'))
    return parser.parse_args(argv)


def eval_argv(args: argparse.Namespace, out: Path) -> List[str]:
    """构造 run_state_eval 的参数命名空间（复用它的 reader/池/存储默认值）。"""
    argv = ['--mode', 'predicted_state', '--time-mode', 'window_end',
            '--data-root', args.data_root, '--split', args.split,
            '--limit', '0', '--tag', args.tag, '--out', str(out),
            '--budget-mode', 'retain_all', '--top-k', str(args.top_k),
            '--neighbours', str(args.neighbours), '--arch-exp', args.arch_exp,
            '--reader', 'stub' if args.plumbing_stub else 'icae',
            '--history-probe-mode', 'window_end',
            '--recency-min-confidence', str(args.recency_min_confidence)]
    if args.recency_update:
        argv.append('--recency-update')
    return argv


def load_episodes(mod, args: argparse.Namespace) -> List[Mapping[str, Any]]:
    path = Path(args.data_root) / f'{args.split}.json'
    if not path.exists():
        available = sorted(p.stem for p in Path(args.data_root).glob('*.json'))
        raise SystemExit(f'缺少准备好的 split：{path}\n可用 split：{available}')
    episodes = mod.read_json(path)
    if args.episode_offset:                    # 跳过已参与分析的验证状态
        episodes = episodes[args.episode_offset:]
    if args.episodes:
        episodes = episodes[:args.episodes]
    if args.shards > 1:
        episodes = episodes[args.shard::args.shards]
    if not episodes:
        raise SystemExit(f'{args.split} 没有可用 episode')
    return episodes


# ---------------------------------------------------------------------------
# 造题
# ---------------------------------------------------------------------------
def build_probe_questions(cvq, mod, episodes, args) -> List[Dict[str, Any]]:
    """构造配对题；每题绑定一个 episode（一个独立状态）。"""
    if args.question_source == 'templama':
        return _templama_questions(mod, episodes, args)
    questions = cvq.build_questions(episodes, args.limit,
                                    per_family=args.per_family or None)
    for question in questions:
        if not (args.min_versions <= question['n_versions'] <= args.max_versions):
            raise SystemExit(
                f"造题越界：{question['id']} 有 {question['n_versions']} 个版本，"
                f'不在 [{args.min_versions}, {args.max_versions}]')
    return questions


def _templama_questions(mod, episodes, args) -> List[Dict[str, Any]]:
    """旧口径：TempLAMA 官方查询（单版本填空），仅作对照保留。"""
    questions: List[Dict[str, Any]] = []
    for episode in episodes:
        if len(questions) >= args.limit:
            break
        cutoff = mod.window_end_of(episode)
        for query in episode.get('queries') or []:
            if len(questions) >= args.limit:
                break
            if not args.include_latest and str(query.get('type') or '') == 'latest':
                continue
            gold_time = query.get('as_of') or query.get('time')
            questions.append({
                'id': f"{episode['id']}::{query.get('id')}", 'family': 'templama',
                'episode_id': episode['id'], 'state_key': str(query.get('state_key') or ''),
                'question': str(query.get('question') or ''),
                'answers': list(query.get('answers') or []),
                'gold_years': [], 'required_version_times': [gold_time] if gold_time else [],
                'pair_target_time': None, 'n_versions': None, 'version_times': [],
                'version_values': [], 'newest_values': [], 'key_text': '',
                'observation_cutoff': cutoff})
    return questions


# ---------------------------------------------------------------------------
# 写入：两臂共用同一个事实池与同一份状态身份
# ---------------------------------------------------------------------------
def write_pool_gold(mod, runtime, pool) -> Dict[str, str]:
    """金标 state_key 当状态身份写池（隔离读取阶段，读取差异只剩 read_mode）。"""
    counter = mod.PublicEventCounter()
    gold_to_state: Dict[str, str] = {}
    for member in pool:
        for event in member.get('events') or []:
            view = mod.public_write_event(event, counter.next_id())
            gold = str(event.get('state_key') or '')
            if not gold:
                raise ValueError('--identity gold 需要每个事件都带金标 state_key')
            entity, _, relation = gold.partition(':')
            memory = runtime.reader.encode_event({'time': view['time'], 'text': view['text']})
            key = runtime.index.key(memory)
            state_id = gold_to_state.get(gold)
            if state_id is None:
                state_id = runtime.state_index.new_state_id()
                gold_to_state[gold] = state_id
                decision = mod.StateDecision('CREATE', state_id, 1.0, 'oracle_state:gold_key')
            else:
                decision = mod.StateDecision('UPDATE', state_id, 1.0, 'oracle_state:gold_key')
            version_event = {'id': view['id'], 'time': view['time'], 'text': view['text']}
            receipt = runtime.store.ingest_decision(version_event, memory, key, decision)
            if decision.action == 'CREATE':
                runtime.state_index.add(
                    {'id': receipt['id'], 'subject': entity, 'attribute': relation,
                     'value': view['text'], 'observed_at': view['time']},
                    semantic_key=key, version_id=receipt['id'], state_id=state_id)
            else:
                runtime.state_index.commit(
                    state_id, receipt['id'], observed_at=view['time'],
                    value_fingerprint=mod.value_fingerprint(view['text']))
    return gold_to_state


def write_pool_predicted(mod, runtime, pool) -> Dict[str, str]:
    """规则前端预测状态身份（只在 --identity predicted 时用）。"""
    counter = mod.PublicEventCounter()
    facts: List[Dict[str, Any]] = []
    for member in pool:
        written, _, _ = mod.write_episode(runtime, member, counter)
        facts.extend(written)
    return mod.primary_states(facts)


# ---------------------------------------------------------------------------
# 单题评估：两臂读同一个状态，只有 read_mode 不同
# ---------------------------------------------------------------------------
def evaluate_question(cvq, mod, runtime, resolver, episode, question, state_id, cutoff,
                      args) -> Dict[str, Any]:
    view = mod.public_query({'question': question['question']},
                            time_mode='window_end', window_end=cutoff)
    bounds = resolver.resolve_bounds(question['question'], observation_cutoff=cutoff)
    pair_target = (bounds.target_time if args.target_time == 'resolver'
                   else question['pair_target_time'])
    visible = runtime.store.visible_versions(state_id, cutoff)
    required = [time_ for time_ in question['required_version_times'] if time_]
    row: Dict[str, Any] = {
        'tag': args.tag, 'family': question['family'], 'episode': episode['id'],
        'question_id': question['id'], 'question': question['question'],
        'answers': question['answers'], 'state_key': question['state_key'],
        'gold_years': question['gold_years'],
        'observation_cutoff': cutoff, 'pair_target_time': pair_target,
        'resolver_target_time': bounds.target_time,
        'resolver_time_anchor': bounds.time_anchor,
        'resolver_anchor_ambiguous': bounds.anchor_ambiguous,
        'n_versions': len(visible),
        'visible_version_times': [entry['time'] for entry in visible],
        'visible_version_values': question['version_values'],
        'newest_values': question['newest_values'],
        'required_version_times': required,
    }
    for mode in MODES:
        result = runtime.read_state(
            state_id, view, target_time=pair_target, read_mode=mode,
            observation_cutoff=cutoff, max_slots=args.max_slots,
            max_versions=args.max_versions)
        answer = str(result.get('answer') or '')
        bundle_times = list(result.get('bundle_version_times') or [])
        scored = cvq.score_answer(question['family'], answer, question)
        values = list(question.get('version_values') or [])
        gold_indices = {index for index, value in enumerate(values)
                        if any(cvq.matches_value(gold, value) for gold in question['answers'])}
        matched = next((index for index, value in enumerate(values)
                        if cvq.matches_value(answer, value)), None)
        # count_values / compare_years 的答案不是版本值（数字、yes/no），
        # 这两类没有"答成了哪个年份"可言，wrong_year 恒为 0。
        if not values or not gold_indices:
            matched, wrong_year, distance = None, 0, 0
        else:
            wrong_year = int(matched is not None and matched not in gold_indices)
            distance = min(abs(matched - index) for index in gold_indices) if wrong_year else 0
        row[mode] = {
            'answer': answer,
            'correct': int(scored['correct']),
            'strict_em': int(scored['strict_em']),
            'scoring_rule': scored['rule'],
            'wrong_current': int(cvq.answered_with_current(answer, question)),
            'matched_version_index': matched, 'wrong_year': wrong_year,
            'wrong_year_distance': distance,
            'read_latent_slots': int(result.get('read_latent_slots') or 0),
            'trajectory_length': int(result.get('trajectory_length') or 0),
            'bundle_version_times': bundle_times,
            'selected_version_time': result.get('selected_version_time'),
            'all_required_in_bundle': bool(required) and all(
                time_ in bundle_times for time_ in required),
            'future_current_suppressed': bool(result.get('future_current_suppressed')),
            'hidden_future_versions': int(result.get('hidden_future_versions') or 0),
            'read_seconds': (result.get('timings') or {}).get('read_seconds'),
        }
    row['wrong_current_applicable'] = int(
        question['family'] in cvq.VALUE_FAMILIES
        and bool(question['newest_values'])
        and cvq.norm_set(question['newest_values'])
        != cvq.norm_set(question['answers']))
    return row


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------
def _mean(values: Sequence[float]) -> Optional[float]:
    values = [float(value) for value in values]
    return (sum(values) / len(values)) if values else None


def _rate(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if not denominator:
        return None
    return float(numerator or 0.0) / float(denominator)


def block(rows: Sequence[Mapping[str, Any]], mode: str) -> Dict[str, Any]:
    cells = [row[mode] for row in rows]
    applicable = [row for row in rows if row['wrong_current_applicable']]
    return {
        'n': len(cells),
        'accuracy': _mean([cell['correct'] for cell in cells]),
        'strict_em': _mean([cell['strict_em'] for cell in cells]),
        'all_required_in_bundle_rate': _rate(
            sum(1 for cell in cells if cell['all_required_in_bundle']), len(cells)),
        'wrong_current_answer_rate': _rate(
            sum(1 for row in applicable if row[mode]['wrong_current']), len(applicable)),
        'wrong_current_applicable': len(applicable),
        'wrong_year_rate': _rate(sum(1 for cell in cells if cell['wrong_year']), len(cells)),
        'wrong_year_distance_mean': (
            sum(cell['wrong_year_distance'] for cell in cells if cell['wrong_year'])
            / max(1, sum(1 for cell in cells if cell['wrong_year']))),
        'mean_read_latent_slots': _mean([cell['read_latent_slots'] for cell in cells]),
        'mean_trajectory_length': _mean([cell['trajectory_length'] for cell in cells]),
        'mean_read_seconds': _mean([cell['read_seconds'] for cell in cells
                                    if cell['read_seconds'] is not None]),
        'future_current_suppressed': sum(1 for cell in cells
                                         if cell['future_current_suppressed']),
        'hidden_future_versions': sum(cell['hidden_future_versions'] for cell in cells),
    }


def paired(rows: Sequence[Mapping[str, Any]], label: str = 'all') -> Dict[str, Any]:
    wins = losses = ties = 0
    deltas: List[float] = []
    for row in rows:
        a, b = float(row['pair']['correct']), float(row['trajectory']['correct'])
        deltas.append(b - a)
        if b > a:
            wins += 1
        elif b < a:
            losses += 1
        else:
            ties += 1
    delta = _mean(deltas)
    if delta is None:
        verdict = 'no_data'
    elif delta > 0 and wins > losses:
        verdict = 'trajectory_better'
    elif delta < 0 and losses > wins:
        verdict = 'pair_better'
    else:
        verdict = 'tie'
    bootstrap = paired_bootstrap(
        [float(row['trajectory']['correct']) for row in rows],
        [float(row['pair']['correct']) for row in rows],
        [str(row['family']) for row in rows]) if rows else {'ci95': None, 'delta': None}
    return {'group': label, 'n_paired': len(rows), 'trajectory_wins': wins,
            'pair_wins': losses, 'ties': ties, 'mean_delta_accuracy': delta,
            'ci95_by_family': bootstrap.get('ci95'),
            'verdict': verdict}


def summarise(rows: Sequence[Mapping[str, Any]], args: argparse.Namespace,
              components_label: Optional[str], gpu: Optional[Mapping[str, Any]],
              wall: float) -> Dict[str, Any]:
    blocks = {mode: block(rows, mode) for mode in MODES}
    families = sorted({str(row['family']) for row in rows})
    return {
        'tag': args.tag,
        'git_commit': git_commit(),
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'reader': 'stub' if args.plumbing_stub else 'icae',
        'components': components_label,
        'citable': not args.plumbing_stub,
        'protocol': {
            'dataset': args.split, 'data_root': args.data_root,
            'question_source': args.question_source,
            'questions': len(rows), 'identity': args.identity,
            'neighbours': args.neighbours, 'arch_exp': args.arch_exp,
            'min_versions': args.min_versions, 'max_versions': args.max_versions,
            'max_slots': args.max_slots, 'target_time': args.target_time,
            'include_latest': bool(args.include_latest),
            'observation_cutoff': 'episode window_end (max event time)',
            'shard': args.shard, 'shards': args.shards, 'gpu': dict(gpu or {}),
            'families': families,
        },
        'modes': blocks,
        'by_family': {
            family: {
                'n': sum(1 for row in rows if row['family'] == family),
                'pair_accuracy': _mean([row['pair']['correct'] for row in rows
                                        if row['family'] == family]),
                'trajectory_accuracy': _mean([row['trajectory']['correct'] for row in rows
                                              if row['family'] == family]),
                'pair_all_required_in_bundle_rate': _rate(
                    sum(1 for row in rows if row['family'] == family
                        and row['pair']['all_required_in_bundle']),
                    sum(1 for row in rows if row['family'] == family)),
                'trajectory_all_required_in_bundle_rate': _rate(
                    sum(1 for row in rows if row['family'] == family
                        and row['trajectory']['all_required_in_bundle']),
                    sum(1 for row in rows if row['family'] == family)),
            } for family in families},
        'paired': paired(rows),
        'paired_by_family': {family: paired([row for row in rows
                                             if row['family'] == family], family)
                             for family in families},
        'reading_notes': [
            '这些是从 TempLAMA validation 事件版本链构造的受控诊断题，不是官方分数，'
            '不得用于声称开放域泛化。',
            '两臂共用同一事实池、同一状态身份、同一 cutoff、同一题面、同一 pair 选择上界，'
            '唯一变量是 read_state 的 read_mode。',
            'recall_year 的题面自带年份，pair 用年份即可选对版本 —— 这一族是"trajectory '
            '会不会被多版本干扰"的探针；ordinal_back / count_values / compare_years '
            '结构上必须同时看到多个版本。',
            'all_required_in_bundle_rate 是可达性：pair 只装 2 个版本，跨版本题的金标版本'
            '可能根本不在 bundle 里；accuracy 的差异必须和这一列一起读。',
            'wrong_current_answer_rate 只在取值型 family（recall_year / ordinal_back）'
            '且 current 取值≠金标的题上算，分母见 wrong_current_applicable。',
            '定长 latent 预算下 reader 永远会输出一个答案，答出率恒为 1，故不报噪声率。',
            'stub reader 的分数无语言意义（plumbing_stub=True 时不得引用任何分数）。',
        ],
        'wall_seconds': wall,
    }


def render_markdown(summary: Mapping[str, Any]) -> str:
    blocks, pair = summary['modes'], summary['paired']
    proto = summary['protocol']

    def fmt(value: Any) -> str:
        if value is None:
            return 'n/a'
        if isinstance(value, float):
            return f'{value:.4f}'
        if isinstance(value, (list, tuple)):
            return '[' + ', '.join(fmt(item) for item in value) + ']'
        return str(value)

    rows = [
        ('题数', 'n'),
        ('跨版本题准确率（主指标）', 'accuracy'),
        ('严格 EM', 'strict_em'),
        ('所需版本全在 bundle 内', 'all_required_in_bundle_rate'),
        ('历史答成 current 的比例', 'wrong_current_answer_rate'),
        ('（该指标分母）', 'wrong_current_applicable'),
        ('平均读取 slots', 'mean_read_latent_slots'),
        ('平均轨迹长度', 'mean_trajectory_length'),
        ('平均读取秒', 'mean_read_seconds'),
        ('未来 current 被挡次数', 'future_current_suppressed'),
        ('被挡未来版本总数', 'hidden_future_versions'),
    ]
    lines = [f"# 实验一：Pair vs 全轨迹读取（{summary['tag']}）", '',
             f"- git: `{summary['git_commit']}` | reader: `{summary['reader']}` | "
             f"components: `{summary['components']}` | citable={summary['citable']}",
             f"- 数据: `{proto['dataset']}`，题目来源={proto['question_source']}，"
             f"题数={proto['questions']}，identity={proto['identity']}，"
             f"neighbours={proto['neighbours']}，版本数 {proto['min_versions']}–"
             f"{proto['max_versions']}，max_slots={proto['max_slots']}",
             f"- 可见性截止：{proto['observation_cutoff']}；future fallback 已强制关闭",
             '', '| 指标 | A: pair | B: trajectory | Δ(B−A) |', '| --- | --- | --- | --- |']
    for label, key in rows:
        a, b = blocks['pair'].get(key), blocks['trajectory'].get(key)
        delta = (b - a) if isinstance(a, float) and isinstance(b, float) else None
        lines.append(f'| {label} | {fmt(a)} | {fmt(b)} | {fmt(delta)} |')

    lines += ['', '## 分族准确率', '',
              '| family | n | A: pair | B: trajectory | Δ | pair 可达性 | traj 可达性 |',
              '| --- | --- | --- | --- | --- | --- | --- |']
    for family, cell in summary['by_family'].items():
        delta = (cell['trajectory_accuracy'] - cell['pair_accuracy']
                 if cell['trajectory_accuracy'] is not None
                 and cell['pair_accuracy'] is not None else None)
        lines.append(
            f"| {family} | {cell['n']} | {fmt(cell['pair_accuracy'])} | "
            f"{fmt(cell['trajectory_accuracy'])} | {fmt(delta)} | "
            f"{fmt(cell['pair_all_required_in_bundle_rate'])} | "
            f"{fmt(cell['trajectory_all_required_in_bundle_rate'])} |")

    lines += ['', '## 配对结论', '',
              f"- 配对题数：{pair['n_paired']}",
              f"- trajectory 胜 / pair 胜 / 平：{pair['trajectory_wins']} / "
              f"{pair['pair_wins']} / {pair['ties']}",
              f"- 平均准确率差（B − A）：{fmt(pair['mean_delta_accuracy'])}，"
              f"按 family 聚类的 bootstrap 95% CI：{fmt(pair['ci95_by_family'])}",
              f"- **判定：{pair['verdict']}**", '',
              '分族配对：', '',
              '| family | n | A | B | Δ | 判定 |', '| --- | --- | --- | --- | --- | --- |']
    for family, cell in summary['paired_by_family'].items():
        lines.append(f"| {family} | {cell['n_paired']} | "
                     f"{fmt(summary['by_family'][family]['pair_accuracy'])} | "
                     f"{fmt(summary['by_family'][family]['trajectory_accuracy'])} | "
                     f"{fmt(cell['mean_delta_accuracy'])} | {cell['verdict']} |")
    lines += ['', '## 读表约束', '']
    lines += [f'- {note}' for note in summary['reading_notes']]
    return '\n'.join(lines) + '\n'


def merge_main(args: argparse.Namespace) -> Dict[str, Any]:
    out = Path(args.out)
    paths = sorted(out.glob(f'{args.tag}_rows_*.jsonl'))
    if not paths:
        raise SystemExit(f'{out} 下没有 {args.tag}_rows_*.jsonl，先跑各 shard')
    rows: List[Dict[str, Any]] = []
    for path in paths:
        for line in path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    summary = summarise(rows, args, components_label=None, gpu=None, wall=0.0)
    summary['merged_from'] = [str(path) for path in paths]
    mod = load_evaluator()
    mod.write_json(out / f'{args.tag}_summary.json', summary)
    (out / f'{args.tag}_report.md').write_text(render_markdown(summary))
    print(render_markdown(summary))
    return summary


def plan_main(cvq, mod, episodes, args, out: Path) -> Dict[str, Any]:
    questions = build_probe_questions(cvq, mod, episodes, args)
    counts: Dict[str, int] = {}
    for question in questions:
        counts[question['family']] = counts.get(question['family'], 0) + 1
    plan = {
        'tag': args.tag, 'plan_only': True, 'split': args.split,
        'question_source': args.question_source,
        'episodes_scanned': len(episodes), 'questions': len(questions),
        'by_family': counts,
        'max_slots': args.max_slots,
        'slot_math': 'E3: current 128 + k*16 + (k+1)*2 <= 256 → 最多 8 个版本',
        'filters': {'versions': [args.min_versions, args.max_versions],
                    'identity': args.identity},
        'samples': [{'family': question['family'], 'question': question['question'],
                     'answers': question['answers'],
                     'n_versions': question['n_versions'],
                     'pair_target_time': question['pair_target_time'],
                     'version_times': question['version_times'],
                     'version_values': question['version_values']}
                    for question in questions[:8]],
        'entry_command': ' '.join([
            'CUDA_VISIBLE_DEVICES=<gpu≠0,1>', 'PYTHONPATH=.:probe',
            'env_py310/bin/python', '-u', 'probe/run_trajectory_eval.py',
            '--limit', str(args.limit), '--arch-exp', args.arch_exp,
            '--out', str(out), '--tag', args.tag]),
        'note': 'plan_only：未加载 reader、未跑实验、未写任何 summary。',
    }
    out.mkdir(parents=True, exist_ok=True)
    mod.write_json(out / f'{args.tag}_plan.json', plan)
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    return plan


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = parse_args(argv)
    if args.merge:
        return merge_main(args)
    if args.min_versions < 1 or args.max_versions < args.min_versions:
        raise SystemExit('非法版本数区间')
    if args.shards < 1 or not (0 <= args.shard < args.shards):
        raise SystemExit(f'非法分片：shard={args.shard} shards={args.shards}')
    mod = load_evaluator()
    cvq = load_questions()
    out = Path(args.out)
    episodes = load_episodes(mod, args)
    print(json.dumps({'split': args.split, 'episodes_in_shard': len(episodes),
                      'limit_questions': args.limit, 'shard': args.shard,
                      'shards': args.shards, 'identity': args.identity,
                      'question_source': args.question_source},
                     ensure_ascii=False, indent=2))

    if args.plan_only:
        return plan_main(cvq, mod, episodes, args, out)

    questions = build_probe_questions(cvq, mod, episodes, args)
    if not questions:
        raise SystemExit('没有题目通过筛选：放宽 --min-versions/--max-versions 或换 --split')

    gpu = None if args.plumbing_stub else assert_gpu_policy()
    out.mkdir(parents=True, exist_ok=True)
    run_args = mod.parse_args(eval_argv(args, out))
    started = time.perf_counter()
    components = mod.build_components(run_args)
    if not args.plumbing_stub and not str(components.label).startswith('icae+'):
        raise SystemExit(f'期望真实 ICAE 组件，实际 {components.label!r}：拒绝把 stub 当结论')
    resolver = QueryTimeResolver()

    grouped: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
    index_of = {episode['id']: index for index, episode in enumerate(episodes)}
    for question in questions:
        grouped.setdefault(question['episode_id'], []).append(question)

    rows: List[Dict[str, Any]] = []
    for done, (episode_id, batch) in enumerate(grouped.items(), start=1):
        episode = episodes[index_of[episode_id]]
        runtime = mod.new_runtime(run_args, components)
        pool = [episode] + mod._neighbours(episodes, index_of[episode_id], args.neighbours)
        state_map = (write_pool_gold(mod, runtime, pool) if args.identity == 'gold'
                     else write_pool_predicted(mod, runtime, pool))
        cutoff = mod.window_end_of(episode)
        for question in batch:
            state_id = state_map.get(question['state_key'])
            if state_id is None:
                print(f"[traj] skip {question['id']}：池内没有该状态", file=sys.stderr)
                continue
            rows.append(evaluate_question(cvq, mod, runtime, resolver, episode, question,
                                          state_id, cutoff, args))
        print(f'[traj] episode {done}/{len(grouped)} {episode_id} '
              f'questions={len(batch)} rows={len(rows)}', file=sys.stderr)
    if not rows:
        raise SystemExit('没有任何题目被评估：检查 --identity 与 state_key 是否匹配')

    wall = time.perf_counter() - started
    summary = summarise(rows, args, components.label, gpu, wall)
    suffix = f'_s{args.shard}of{args.shards}' if args.shards > 1 else ''
    rows_path = out / f'{args.tag}_rows{suffix}.jsonl'
    with rows_path.open('w') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    summary['outputs'] = {'rows': str(rows_path),
                          'summary': str(out / f'{args.tag}_summary.json'),
                          'report': str(out / f'{args.tag}_report.md')}
    mod.write_json(out / f'{args.tag}_summary.json', summary)
    markdown = render_markdown(summary)
    (out / f'{args.tag}_report.md').write_text(markdown)
    print(markdown)
    return summary


if __name__ == '__main__':
    main()
