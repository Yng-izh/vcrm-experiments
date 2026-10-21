#!/usr/bin/env python
"""P3 自动关联损失探针：小规模真实 E3（recency_update 开）对照 oracle 上界。

三个 pass 跑**同一批 episode**、同一 reader/codec、同一时间口径、同一读取路径：

    predicted      run_state_eval.run_predicted_state
                   状态身份完全由 FactExtractor / StateRouter / UpdateResolver 预测
                   → 唯一的自动关联证据。
    oracle_pooled  本脚本新增：与 predicted **逐字段同构**的事实池（目标 episode +
                   同样数量的邻近 episode，同一个 mod._neighbours）、同一个 window_end
                   口径、同一个 read_state 路径，只把状态身份换成金标 state_key。
                   → 与 predicted 只差一个变量，两者之差就是"自动关联造成的损失"。
    oracle_global  run_state_eval.run_oracle_state（全部 episode 塞进单一全局状态空间）
                   → 第二参照；池结构与 predicted 不同，禁止与 predicted 配对求损失。

为什么必须新增 oracle_pooled：
    既有 run_oracle_state 把 N 个 episode 写进同一个 runtime，而 predicted_state 每个
    episode 重建 runtime、池里只有 1+neighbours 个 episode。池规模、金标状态数、候选
    密度、store 规模全都不同，两数之差混入了池结构差异，不能当作自动关联损失。
    oracle_pooled 保持池结构不变，只替换状态身份来源。

多卡：--shard/--shards 把抽样后的 episode 交错切片（episodes[shard::shards]），
    每个 shard 独立跑三个 pass 并写 bundle，最后 --merge 合并 counters + rows 重算报告。
    注意：shard 内的池邻居取自该 shard 的 episode 子集，所以单个 shard 的绝对指标与
    不分片的整跑略有差异；predicted vs oracle_pooled 的配对关系在 shard 内完全保持。

口径（不得混算，summary 里已分列）：
    em/f1 = 条件口径（仅 latent 路径）；端到端看 end_to_end_em / answer_coverage。
    latest EM 单独报（TempLAMA 上 latest 查询 = 该 (entity, relation) 最后一年）。
    window_end 口径下 history_version_accuracy 结构性为 0（窗口末刻晚于全部事件）。
    recency_update=on 是 TempLAMA 上唯一能形成多版本状态的配置；update 来源按
    explicit_replacement: / exclusive_superseded_by_recency: 分列。

GPU 纪律：真实 reader 固定 cuda:0，设备必须在启动前用 CUDA_VISIBLE_DEVICES 钉住；
    未设置或含 GPU 0/1 时直接退出（GPU 0/1 为本工作区禁用设备）。

用法：

    # 两卡分片（卡 3 / 卡 9），各跑 50 个 episode 的三个 pass
    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=.:probe env_py310/bin/python -u \
        probe/run_auto_association_loss.py --limit 100 --neighbours 4 \
        --recency-update --top-k 5 --arch-exp E3 --shard 0 --shards 2 \
        --out probe/runs/p3_100ep --tag p3_100ep
    CUDA_VISIBLE_DEVICES=9 ... --shard 1 --shards 2 ...

    # 合并两个 shard 的 bundle 并重算报告（CPU，无 GPU）
    CUDA_VISIBLE_DEVICES= PYTHONPATH=.:probe env_py310/bin/python -u \
        probe/run_auto_association_loss.py --merge --out probe/runs/p3_100ep --tag p3_100ep

    # 只做预检 + 打印命令，不加载 7B reader、不占 GPU
    ... --plan-only
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_DATA_ROOT = '/share/latent2/vcrm_icae/data/prepared'
MODES = ('predicted', 'oracle_pooled', 'oracle_global')
EVAL_MODES = {'predicted': 'predicted_state', 'oracle_pooled': 'oracle_state',
              'oracle_global': 'oracle_state'}
FORBIDDEN_PHYSICAL_GPUS = ('0', '1')      # 本工作区禁用设备
DECISION_KEYS = ('CREATE', 'UPDATE', 'DUPLICATE', 'DEFER')
UPDATE_SOURCE_KEYS = ('explicit_replacement', 'exclusive_superseded_by_recency', 'not_update')


# ---------------------------------------------------------------------------
# 载入既有评测器（probe/ 不是包，沿用 run_state_eval._load_v6 的 importlib 约定）
# ---------------------------------------------------------------------------
def load_evaluator():
    path = REPO_ROOT / 'probe' / 'run_state_eval.py'
    spec = importlib.util.spec_from_file_location('vcrm_probe_run_state_eval', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# 安全闸门
# ---------------------------------------------------------------------------
def assert_gpu_policy() -> Dict[str, Any]:
    """真实 E3 必须在启动前钉住设备；GPU 0/1 禁用。"""
    raw = os.environ.get('CUDA_VISIBLE_DEVICES')
    if raw is None or not raw.strip():
        raise SystemExit(
            '真实 E3 需要 GPU：请显式设置 CUDA_VISIBLE_DEVICES（例如 CUDA_VISIBLE_DEVICES=3）。\n'
            '本脚本拒绝在未指定设备时加载 7B reader；GPU 0/1 为本工作区禁用设备。\n'
            '只做管线自检可加 --plumbing-stub（结果标记 reader=stub，不得当结论）。')
    ids = [part.strip() for part in raw.split(',') if part.strip()]
    bad = [value for value in ids if value in FORBIDDEN_PHYSICAL_GPUS]
    if bad:
        raise SystemExit(f'CUDA_VISIBLE_DEVICES={raw} 含禁用设备 {bad}：GPU 0/1 不可用')
    if len(ids) > 1:
        print(f'[warn] CUDA_VISIBLE_DEVICES={raw} 含多个设备；FrozenReader 固定用 cuda:0 '
              f'（= 列表第一个 {ids[0]}）', file=sys.stderr)
    return {'cuda_visible_devices': raw, 'physical_gpu_ids': ids}


def git_commit() -> Optional[str]:
    try:
        out = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=str(REPO_ROOT),
                             capture_output=True, text=True, timeout=10)
    except Exception:                                     # noqa: BLE001 - provenance 缺失不阻断
        return None
    return (out.stdout.strip() or None) if out.returncode == 0 else None


# ---------------------------------------------------------------------------
# 参数与数据采样
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='P3 100-episode 自动关联损失探针（真实 E3 + oracle 上界，三 pass 配对）')
    parser.add_argument('--limit', type=int, default=100, help='episode 数（默认 100）')
    parser.add_argument('--sample', choices=['stride', 'head'], default='stride',
                        help='stride = 在全体 episode 上等距抽样（默认）；head = 取前 limit 个')
    parser.add_argument('--data-root', default=DEFAULT_DATA_ROOT)
    parser.add_argument('--split', default='templama_test')
    parser.add_argument('--modes', default=','.join(MODES),
                        help=f'要跑的 pass，逗号分隔，可选 {MODES}')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1,
                        help='分片数；>1 时用 episodes[shard::shards] 交错切片')
    parser.add_argument('--merge', action='store_true', default=False,
                        help='合并各 shard 的 bundle 并重算报告（CPU，不加载 reader）')
    parser.add_argument('--time-mode', choices=('window_end', 'as_of'), default='window_end',
                        help='window_end = Predicted 正式口径；as_of = 仅受控对照（泄漏口径）')
    parser.add_argument('--recency-update', action=argparse.BooleanOptionalAction, default=True,
                        help='rule5b（观察时间更晚的排他属性覆盖）；本探针默认开')
    parser.add_argument('--recency-min-confidence', type=float, default=0.0)
    parser.add_argument('--neighbours', type=int, default=4,
                        help='混合事实池的邻近 episode 数（predicted 与 oracle_pooled 共用）')
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--arch-exp', default='E3', choices=('E1', 'E2', 'E3', 'E4'))
    parser.add_argument('--tag', default='p3_100ep')
    parser.add_argument('--out', default=str(REPO_ROOT / 'probe' / 'runs' / 'p3_100ep'))
    parser.add_argument('--hybrid-fallback', action='store_true', default=False,
                        help='无候选时允许 parametric 回退（答案单列，不进主表 em/f1）')
    parser.add_argument('--defer-store-memory', action='store_true', default=False)
    parser.add_argument('--history-probe-mode', choices=('window_end', 'event_time'),
                        default='window_end',
                        help='event_time = diagnostic_only：用金标版本时间量上界')
    parser.add_argument('--plumbing-stub', action='store_true', default=False,
                        help='只用 stub reader 自检管线（CPU、无 GPU，分数无效）')
    parser.add_argument('--plan-only', action='store_true', default=False,
                        help='只做数据/权重预检并打印命令，不加载 reader、不跑实验')
    return parser.parse_args(argv)


def build_eval_argv(args: argparse.Namespace, mode: str, out: Path) -> List[str]:
    """所有 pass 共用同一份参数，只有 --mode 不同 → 池结构/口径逐字段一致。"""
    argv = ['--mode', mode, '--time-mode', args.time_mode, '--data-root', args.data_root,
            '--split', args.split, '--limit', str(args.limit), '--tag', args.tag,
            '--out', str(out), '--budget-mode', 'retain_all',
            '--top-k', str(args.top_k), '--neighbours', str(args.neighbours),
            '--arch-exp', args.arch_exp, '--reader', 'stub' if args.plumbing_stub else 'icae',
            '--history-probe-mode', args.history_probe_mode,
            '--recency-min-confidence', str(args.recency_min_confidence)]
    if args.recency_update:
        argv.append('--recency-update')
    if args.hybrid_fallback:
        argv.append('--hybrid-fallback')
    if args.defer_store_memory:
        argv.append('--defer-store-memory')
    return argv


def sample_episodes(mod, args: argparse.Namespace) -> Tuple[List[Mapping[str, Any]], int, int]:
    path = Path(args.data_root) / f'{args.split}.json'
    if not path.exists():
        raise SystemExit(f'缺少准备好的 split：{path}')
    all_episodes = mod.read_json(path)
    if args.sample == 'stride':
        stride = max(1, len(all_episodes) // max(1, args.limit))
        picked = all_episodes[::stride][:args.limit]
    else:
        stride, picked = 1, all_episodes[:args.limit]
    if len(picked) < args.limit:
        raise SystemExit(f'{args.split} 只有 {len(picked)} 个 episode（需 {args.limit}）')
    missing = [ep['id'] for ep in picked
               if any('state_key' not in event for event in ep.get('events') or [])]
    if missing:
        raise SystemExit(f'这些 episode 的事件缺金标 state_key，oracle 对照无法构造：{missing[:3]}')
    return picked, len(all_episodes), stride


def pool_stats(mod, episodes: Sequence[Mapping[str, Any]], neighbours: int) -> Dict[str, Any]:
    """预算：每个 pass 要写多少事件、读多少次。"""
    writes = queries = latest = 0
    pool_events = []
    for index, episode in enumerate(episodes):
        pool = [episode] + mod._neighbours(episodes, index, neighbours)
        count = sum(len(member.get('events') or []) for member in pool)
        writes += count
        pool_events.append(count)
        queries += len(episode.get('queries') or [])
        latest += sum(1 for q in episode.get('queries') or [] if q.get('type') == 'latest')
    return {'episodes': len(episodes), 'pool_writes_per_pass': writes,
            'queries_per_pass': queries, 'latest_queries': latest,
            'mean_pool_events': (sum(pool_events) / len(pool_events)) if pool_events else 0.0,
            'max_pool_events': max(pool_events) if pool_events else 0}


# ---------------------------------------------------------------------------
# oracle_pooled：与 predicted 同池、同读取路径，只有状态身份来自金标
# ---------------------------------------------------------------------------
def run_oracle_pooled(mod, args: argparse.Namespace,
                      episodes: Sequence[Mapping[str, Any]], components) -> Any:
    metrics = mod.Metrics()
    gold_identity: Counter = Counter()
    for index, episode in enumerate(episodes):
        runtime = mod.new_runtime(args, components)
        pool = [episode] + mod._neighbours(episodes, index, args.neighbours)
        counter = mod.PublicEventCounter()
        gold_to_state: Dict[str, str] = {}
        pool_facts: List[Dict[str, Any]] = []
        for member in pool:
            for event in member.get('events') or []:
                view = mod.public_write_event(event, counter.next_id())
                gold = str(event.get('state_key') or '')
                if not gold:
                    raise ValueError('oracle_pooled needs the gold state_key on every event')
                entity, _, relation = gold.partition(':')
                memory = components.reader.encode_event(
                    {'time': view['time'], 'text': view['text']})
                key = components.index.key(memory)
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
                    # 用 add(state_id=...) 保留预分配编号（add_keys 会另发新编号）
                    runtime.state_index.add(
                        {'id': receipt['id'], 'subject': entity, 'attribute': relation,
                         'value': view['text'], 'observed_at': view['time']},
                        semantic_key=key, version_id=receipt['id'], state_id=state_id)
                else:
                    runtime.state_index.commit(
                        state_id, receipt['id'], observed_at=view['time'],
                        value_fingerprint=mod.value_fingerprint(view['text']))
                gold_identity[decision.action] += 1
                metrics.writes += 1
                pool_facts.append({'fact_id': view['id'], 'state_id': state_id, 'gold': gold,
                                   'episode': member['id'], 'decision': decision.action})
        window_end = mod.window_end_of(episode)
        for query in episode.get('queries') or []:
            view = mod.public_query(query, time_mode=args.time_mode, window_end=window_end)
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
            metrics.add_row(mod.score_query(
                runtime, episode, query, state_id, result, gold_to_state, args,
                mod.diagnostic_read(runtime, state_id, query, window_end, args)))
        metrics.add_pool_facts(pool_facts)
        metrics.add_facts([fact for fact in pool_facts if fact['episode'] == episode['id']])
        metrics.add_decisions([])          # oracle 不产生 resolver 决策日志
        metrics.add_store(runtime.store)
    metrics.decisions.update(gold_identity)   # 计数来自金标身份分配，非 resolver
    return metrics


# ---------------------------------------------------------------------------
# 落盘：官方 per-pass 文件（沿用 run_state_eval 命名）+ 可分片合并的 bundle
# ---------------------------------------------------------------------------
def dump_pass(mod, args: argparse.Namespace, metrics, components,
              episodes: Sequence[Mapping[str, Any]], mode: str, out: Path,
              started: float) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    summary = metrics.summary(args, components, len(episodes))
    summary['mode'] = mode
    summary['evaluator'] = 'probe/run_state_eval.py'
    summary['wall_seconds'] = time.perf_counter() - started
    summary['decision_counts_source'] = ('gold_identity' if mode.startswith('oracle')
                                         else 'resolver')
    stem = f'{args.tag}_{mode}_{args.time_mode}_{args.limit}'
    scores_path = out / f'{stem}.scores.jsonl'
    decisions_path = out / f'{stem}.decisions.jsonl'
    with scores_path.open('w') as handle:
        for row in metrics.rows:
            handle.write(json.dumps(row | {'gold_fields_present': True}, ensure_ascii=False) + '\n')
    with decisions_path.open('w') as handle:
        for record in metrics.decision_log:
            mod.validate_decision_log(record, where='decisions.jsonl')
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
    summary['outputs'] = {'scores': str(scores_path), 'decisions': str(decisions_path),
                          'summary': str(out / f'{stem}.summary.json')}
    mod.write_json(out / f'{stem}.summary.json', summary)
    return summary, metrics.rows


def bundle_from(metrics, mode: str, wall_seconds: float, episodes: int,
                protocol: Mapping[str, Any]) -> Dict[str, Any]:
    """可跨 shard 相加的最小状态：计数类进 counters，逐题明细进 rows。"""
    counters = {
        'episodes': episodes, 'queries': len(metrics.rows), 'writes': metrics.writes,
        'wall_seconds': wall_seconds, 'write_seconds': metrics.write_seconds,
        'tp': metrics.tp, 'fp': metrics.fp, 'fn': metrics.fn,
        'pred_states': metrics.pred_states, 'false_merged_states': metrics.false_merged_states,
        'gold_states': metrics.gold_states, 'false_split_states': metrics.false_split_states,
        'pool_pools': metrics.pool_pools,
        'pool_cross_gold_pairs': metrics.pool_cross_gold_pairs,
        'pool_merged_pairs': metrics.pool_merged_pairs,
        'pool_same_gold_pairs': metrics.pool_same_gold_pairs,
        'pool_split_pairs': metrics.pool_split_pairs,
        'store_bytes': metrics.store_bytes,
        'read_seconds_sum': float(sum(metrics.read_seconds)),
        'read_seconds_n': len(metrics.read_seconds),
        'slots_sum': int(sum(metrics.slots)), 'slots_n': len(metrics.slots),
        'decisions': dict(metrics.decisions),
        'update_sources': dict(metrics.update_sources),
    }
    return {'mode': mode, 'counters': counters, 'rows': metrics.rows,
            'protocol': dict(protocol)}


def merge_bundles(paths: Sequence[Path]) -> Dict[str, Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for path in paths:
        bundle = json.loads(path.read_text())
        mode = bundle['mode']
        target = merged.get(mode)
        if target is None:
            merged[mode] = {'mode': mode, 'counters': dict(bundle['counters']),
                            'rows': list(bundle['rows']), 'protocol': bundle['protocol'],
                            'sources': [str(path)]}
            continue
        for key, value in bundle['counters'].items():
            if isinstance(value, dict):
                bucket = target['counters'].setdefault(key, {})
                for sub, sub_value in value.items():
                    bucket[sub] = bucket.get(sub, 0) + sub_value
            else:
                target['counters'][key] = target['counters'].get(key, 0) + value
        target['rows'].extend(bundle['rows'])
        target['sources'].append(str(path))
    return merged


# ---------------------------------------------------------------------------
# 指标提取 / 损失分解
# ---------------------------------------------------------------------------
def _mean(values: Sequence[float]) -> Optional[float]:
    values = [float(value) for value in values]
    return (sum(values) / len(values)) if values else None


def _rate(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if not denominator:
        return None
    return float(numerator or 0.0) / float(denominator)


def _e2e_value(row: Mapping[str, Any], key: str) -> float:
    """与 run_state_eval.Metrics.summary 的 _e2e 完全同口径。"""
    if row.get('no_state_match') and not str(row.get('answer') or '').strip():
        return 0.0
    if row.get('hybrid_fallback'):
        return 0.0
    return float(row[key])


def latest_block(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    latest = [row for row in rows if row.get('gold_type') == 'latest']
    latent = [row for row in latest
              if not row.get('no_state_match') and not row.get('hybrid_fallback')]
    versioned = [row for row in latest if row.get('selected_version_time')]
    return {'n': len(latest),
            'em_conditional': _mean([row['em'] for row in latent]),
            'em_end_to_end': _mean([_e2e_value(row, 'em') for row in latest]),
            'f1_conditional': _mean([row['f1'] for row in latent]),
            'version_accuracy': _mean([float(bool(row.get('version_correct')))
                                       for row in versioned]),
            'note': 'em_conditional 只在 no_state_match=False 的 latest 行上算（会高估）；'
                    'em_end_to_end 分母 = 全部 latest 行，是可与 oracle 对齐的口径。'}


def attribution_block(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """latest 查询的损失分解：关联错 / 版本选错 / 读取出错。"""
    latest = [row for row in rows if row.get('gold_type') == 'latest']
    n = len(latest)

    def association_ok(row: Mapping[str, Any]) -> bool:
        gold_state_id = row.get('gold_state_id')
        return bool(gold_state_id) and row.get('state_id') == gold_state_id

    associated = [row for row in latest if association_ok(row)]
    missed = [row for row in latest if not association_ok(row)]
    version_ok = [row for row in associated if row.get('version_correct')]
    em_all = _mean([row['em'] for row in latest])
    em_given = _mean([row['em'] for row in associated])
    association_rate = _rate(len(associated), n)
    return {'n': n,
            'association_ok': len(associated),
            'association_miss': len(missed),
            'association_rate': association_rate,
            'em_all': em_all,
            'em_given_association_ok': em_given,
            'version_accuracy_given_association_ok': _rate(len(version_ok), len(associated)),
            'em_given_version_ok': _mean([row['em'] for row in version_ok]),
            'association_attributable_loss_upper': (
                (1 - association_rate) * em_given
                if association_rate is not None and em_given is not None else None),
            'note': 'em_given_association_ok 是条件量；association_attributable_loss_upper = '
                    '(1 - association_rate) * em_given_association_ok，表示"关联全对且下游质量'
                    '不变"能补回的上限，不是逐题反事实。'}


def summarize_bundle(bundle: Mapping[str, Any]) -> Dict[str, Any]:
    counters = bundle['counters']
    rows = bundle['rows']
    mode = bundle['mode']
    tp, fp, fn = counters.get('tp', 0), counters.get('fp', 0), counters.get('fn', 0)
    precision = _rate(tp, tp + fp)
    recall = _rate(tp, tp + fn)
    pairwise_f1 = (2 * precision * recall / (precision + recall)
                   if precision is not None and recall is not None and (precision + recall)
                   else None)
    if mode == 'oracle_global':
        # run_oracle_state 不登记决策计数；给 None 而不是 0，避免读成"没有 CREATE/UPDATE"。
        decision_counts: Dict[str, Any] = {key: None for key in DECISION_KEYS}
        update_sources: Any = None
    else:
        decisions = counters.get('decisions') or {}
        decision_counts = {key: int(decisions.get(key, 0)) for key in DECISION_KEYS}
        sources = counters.get('update_sources') or {}
        update_sources = {key: int(sources.get(key, 0)) for key in UPDATE_SOURCE_KEYS}
    latent = [row for row in rows
              if not row.get('no_state_match') and not row.get('hybrid_fallback')]
    history = [row for row in rows
               if row.get('gold_type') != 'latest' and row.get('selected_version_time')]
    recall_rows = [row for row in rows if row.get('recall_at_1') is not None]
    return {
        'mode': mode,
        'reader': bundle['protocol'].get('reader'),
        'components': bundle['protocol'].get('components'),
        'recency_update': bundle['protocol'].get('recency_update'),
        'episodes': counters.get('episodes'),
        'queries': len(rows),
        'writes': counters.get('writes'),
        'wall_seconds': counters.get('wall_seconds'),
        'decision_counts': decision_counts,
        'update_source_counts': update_sources,
        'state_pairwise': {'precision': precision, 'recall': recall, 'f1': pairwise_f1,
                           'tp': tp, 'fp': fp, 'fn': fn},
        'false_merge_rate': _rate(counters.get('false_merged_states'),
                                  counters.get('pred_states')),
        'false_split_rate': _rate(counters.get('false_split_states'),
                                  counters.get('gold_states')),
        'pool_false_merge_rate': _rate(counters.get('pool_merged_pairs'),
                                       counters.get('pool_cross_gold_pairs')),
        'pool_false_split_rate': _rate(counters.get('pool_split_pairs'),
                                       counters.get('pool_same_gold_pairs')),
        'pool_pools': counters.get('pool_pools'),
        'pool_pairs_cross_gold': counters.get('pool_cross_gold_pairs'),
        'query_state_recall_at_1': _mean([float(row['recall_at_1']) for row in recall_rows]),
        'query_state_recall_at_3': _mean([float(row['recall_at_3']) for row in recall_rows]),
        'query_state_recall_n': len(recall_rows),
        'history_version_accuracy': _mean([float(bool(row.get('version_correct')))
                                           for row in history]),
        'history_version_accuracy_upper_bound_reason': (
            'window_end_exceeds_all_event_times'
            if bundle['protocol'].get('time_mode') == 'window_end' else None),
        'em_conditional': _mean([row['em'] for row in latent]),
        'f1_conditional': _mean([row['f1'] for row in latent]),
        'end_to_end_em': _mean([_e2e_value(row, 'em') for row in rows]),
        'end_to_end_f1': _mean([_e2e_value(row, 'f1') for row in rows]),
        'answer_coverage': _rate(sum(1 for row in rows if not row.get('no_state_match')),
                                 len(rows)),
        'no_state_match_rate': _rate(sum(1 for row in rows if row.get('no_state_match')),
                                     len(rows)),
        'mean_read_seconds': _rate(counters.get('read_seconds_sum'),
                                   counters.get('read_seconds_n')),
        'mean_read_latent_slots': _rate(counters.get('slots_sum'), counters.get('slots_n')),
        'mean_write_seconds': _rate(counters.get('write_seconds'), counters.get('writes')),
        'store_bytes_per_episode': _rate(counters.get('store_bytes'), counters.get('episodes')),
        'latest': latest_block(rows),
        'latest_attribution': attribution_block(rows),
    }


def _delta(oracle: Optional[float], predicted: Optional[float]) -> Optional[float]:
    if oracle is None or predicted is None:
        return None
    return float(oracle) - float(predicted)


def build_report(tag: str, provenance: Mapping[str, Any], protocol: Mapping[str, Any],
                 bundles: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    blocks = {mode: summarize_bundle(bundle) for mode, bundle in bundles.items()}
    predicted = blocks.get('predicted') or {}
    oracle = blocks.get('oracle_pooled') or {}

    def pick(block: Mapping[str, Any], *path: str) -> Optional[float]:
        value: Any = block
        for key in path:
            if not isinstance(value, Mapping):
                return None
            value = value.get(key)
        return None if value is None else float(value)

    latest_gap = _delta(pick(oracle, 'latest', 'em_end_to_end'),
                        pick(predicted, 'latest', 'em_end_to_end'))
    oracle_latest = pick(oracle, 'latest', 'em_end_to_end')
    loss = {
        'paired_against': 'oracle_pooled（同池、同窗口口径、同读取路径，只换状态身份来源）',
        'delta_latest_em_end_to_end': latest_gap,
        'delta_latest_em_conditional': _delta(pick(oracle, 'latest', 'em_conditional'),
                                              pick(predicted, 'latest', 'em_conditional')),
        'delta_end_to_end_em': _delta(pick(oracle, 'end_to_end_em'),
                                      pick(predicted, 'end_to_end_em')),
        'delta_em_conditional': _delta(pick(oracle, 'em_conditional'),
                                       pick(predicted, 'em_conditional')),
        'delta_query_state_recall_at_1': _delta(pick(oracle, 'query_state_recall_at_1'),
                                                pick(predicted, 'query_state_recall_at_1')),
        'delta_answer_coverage': _delta(pick(oracle, 'answer_coverage'),
                                        pick(predicted, 'answer_coverage')),
        'relative_loss_latest_em': ((latest_gap / oracle_latest)
                                    if latest_gap is not None and oracle_latest else None),
        'predicted_association_attributable_loss_upper': pick(
            predicted, 'latest_attribution', 'association_attributable_loss_upper'),
        'note': 'delta = oracle_pooled - predicted，正数 = 自动关联造成的损失。'
                'oracle_global 池结构不同（全部 episode 一个状态空间），只作第二参照，'
                '不参与 delta。',
    }
    return {
        'tag': tag,
        'provenance': dict(provenance),
        'protocol': dict(protocol),
        'modes': blocks,
        'loss': loss,
        'reading_notes': [
            'predicted 是唯一可当自动关联证据的 pass；oracle_* 只是上界。',
            'em/f1 为条件口径；跨 pass 比较请用 end_to_end_em / latest.em_end_to_end。',
            'window_end 口径下 history_version_accuracy 结构性为 0，不作结论。',
            'recency_update=on 时 UPDATE 才可能形成多版本；update_source_counts 里 '
            'exclusive_superseded_by_recency 的数量说明 rule5b 实际生效多少条。',
            'stub reader 的 EM 无语言意义（plumbing_stub=True 时不得引用任何分数）。',
        ],
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    modes = report['modes']
    order = [mode for mode in MODES if mode in modes]
    protocol = report['protocol']
    loss = report['loss']

    def fmt(value: Any) -> str:
        if value is None:
            return 'n/a'
        if isinstance(value, bool):
            return 'on' if value else 'off'
        if isinstance(value, float):
            return f'{value:.4f}'
        return str(value)

    def path(*keys):
        def getter(block):
            value: Any = block
            for key in keys:
                if not isinstance(value, Mapping):
                    return None
                value = value.get(key)
            return value
        return getter

    def row(label: str, getter, delta=None) -> str:
        return ('| ' + label + ' | '
                + ' | '.join(fmt(getter(modes[mode])) for mode in order)
                + f' | {fmt(delta)} |')

    lines = [f"# P3 自动关联损失探针：{report['tag']}", '',
             f"- git: `{report['provenance'].get('git_commit')}`",
             f"- 数据: `{protocol.get('dataset')}`，抽样 {protocol.get('sample')}"
             f"（stride={protocol.get('sample_stride')}），"
             f"episode={protocol.get('episodes')}，neighbours={protocol.get('neighbours')}，"
             f"top_k={protocol.get('top_k')}",
             f"- recency_update={fmt(protocol.get('recency_update'))}，"
             f"time_mode={protocol.get('time_mode')}，arch={protocol.get('arch_exp')}，"
             f"shards={protocol.get('shards')}",
             f"- components: `{protocol.get('components')}`",
             '', '| 指标 | ' + ' | '.join(order) + ' | Δ(oracle_pooled−predicted) |',
             '| --- | ' + ' | '.join('---' for _ in order) + ' | --- |',
             row('CREATE', path('decision_counts', 'CREATE')),
             row('UPDATE', path('decision_counts', 'UPDATE')),
             row('DUPLICATE', path('decision_counts', 'DUPLICATE')),
             row('DEFER', path('decision_counts', 'DEFER')),
             row('update:explicit_replacement',
                 path('update_source_counts', 'explicit_replacement')),
             row('update:exclusive_superseded_by_recency',
                 path('update_source_counts', 'exclusive_superseded_by_recency')),
             row('query→state recall@1', path('query_state_recall_at_1'),
                 loss.get('delta_query_state_recall_at_1')),
             row('query→state recall@3', path('query_state_recall_at_3')),
             row('state pairwise P', path('state_pairwise', 'precision')),
             row('state pairwise R', path('state_pairwise', 'recall')),
             row('state pairwise F1', path('state_pairwise', 'f1')),
             row('false_merge_rate（目标 episode 口径）', path('false_merge_rate')),
             row('false_merge_rate（整池口径）', path('pool_false_merge_rate')),
             row('false_split_rate（整池口径）', path('pool_false_split_rate')),
             row('latest EM（条件口径）', path('latest', 'em_conditional'),
                 loss.get('delta_latest_em_conditional')),
             row('latest EM（端到端口径）', path('latest', 'em_end_to_end'),
                 loss.get('delta_latest_em_end_to_end')),
             row('latest version accuracy', path('latest', 'version_accuracy')),
             row('history version accuracy', path('history_version_accuracy')),
             row('EM（条件口径）', path('em_conditional'), loss.get('delta_em_conditional')),
             row('F1（条件口径）', path('f1_conditional')),
             row('端到端 EM', path('end_to_end_em'), loss.get('delta_end_to_end_em')),
             row('端到端 F1', path('end_to_end_f1')),
             row('answer coverage', path('answer_coverage'), loss.get('delta_answer_coverage')),
             row('平均读取秒', path('mean_read_seconds')),
             row('平均 latent slots', path('mean_read_latent_slots')),
             row('wall seconds', path('wall_seconds')),
             '', '## latest 损失分解（关联 / 版本 / 读取）', '',
             '| 指标 | ' + ' | '.join(order) + ' |',
             '| --- | ' + ' | '.join('---' for _ in order) + ' |']
    for label, getter in (
            ('n_latest', path('latest_attribution', 'n')),
            ('association_ok', path('latest_attribution', 'association_ok')),
            ('association_miss', path('latest_attribution', 'association_miss')),
            ('association_rate', path('latest_attribution', 'association_rate')),
            ('em_all', path('latest_attribution', 'em_all')),
            ('em_given_association_ok', path('latest_attribution', 'em_given_association_ok')),
            ('version_accuracy_given_association_ok',
             path('latest_attribution', 'version_accuracy_given_association_ok')),
            ('em_given_version_ok', path('latest_attribution', 'em_given_version_ok')),
            ('association_attributable_loss_upper',
             path('latest_attribution', 'association_attributable_loss_upper'))):
        lines.append(f'| {label} | '
                     + ' | '.join(fmt(getter(modes[mode])) for mode in order) + ' |')
    lines += ['', '## 损失结论', '',
              f"- 配对口径：{loss['paired_against']}",
              f"- latest EM 端到端损失：{fmt(loss.get('delta_latest_em_end_to_end'))}"
              f"（相对 {fmt(loss.get('relative_loss_latest_em'))}）",
              f"- 端到端 EM 损失：{fmt(loss.get('delta_end_to_end_em'))}",
              f"- 关联可解释的上限："
              f"{fmt(loss.get('predicted_association_attributable_loss_upper'))}",
              '', '## 读表约束', '']
    lines += [f'- {note}' for note in report['reading_notes']]
    return '\n'.join(lines) + '\n'


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def _protocol(args: argparse.Namespace, sample: Mapping[str, Any], budget: Mapping[str, Any],
              components_label: Optional[str]) -> Dict[str, Any]:
    return {'dataset': args.split, 'data_root': args.data_root,
            'sample': args.sample, 'sample_stride': sample['stride'],
            'episodes_in_split': sample['total'],
            'first_episode': sample['first'], 'last_episode': sample['last'],
            'limit': args.limit, 'neighbours': args.neighbours, 'top_k': args.top_k,
            'time_mode': args.time_mode, 'recency_update': bool(args.recency_update),
            'recency_min_confidence': args.recency_min_confidence,
            'arch_exp': args.arch_exp, 'reader': 'stub' if args.plumbing_stub else 'icae',
            'components': components_label, 'shards': args.shards,
            'budget': budget,
            'cost_note': '每个 pass 独立重建 store；池内 store 只含 1+neighbours 个 episode 的'
                         '状态。budget 是本 shard 的实际写/读次数。'}


def merge_main(args: argparse.Namespace) -> Dict[str, Any]:
    mod = load_evaluator()
    out = Path(args.out)
    paths = sorted(out.glob(f'{args.tag}_bundle_*.json'))
    if not paths:
        raise SystemExit(f'{out} 下没有 {args.tag}_bundle_*.json，先跑各 shard')
    bundles = merge_bundles(paths)
    protocol = next(iter(bundles.values()))['protocol']
    provenance = {'git_commit': git_commit(),
                  'merged_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                  'sources': {mode: bundle['sources'] for mode, bundle in bundles.items()},
                  'argv': sys.argv}
    report = build_report(args.tag, provenance, protocol, bundles)
    mod.write_json(out / f'{args.tag}_report.json', report)
    markdown = render_markdown(report)
    (out / f'{args.tag}_report.md').write_text(markdown)
    print(markdown)
    return report


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = parse_args(argv)
    if args.merge:
        return merge_main(args)
    out = Path(args.out)
    modes = [mode.strip() for mode in args.modes.split(',') if mode.strip()]
    unknown = [mode for mode in modes if mode not in MODES]
    if unknown:
        raise SystemExit(f'未知 pass {unknown}，可选 {MODES}')
    if args.shards < 1 or not (0 <= args.shard < args.shards):
        raise SystemExit(f'非法分片：shard={args.shard} shards={args.shards}')
    if args.time_mode == 'as_of':
        print('[warn] time_mode=as_of 是泄漏口径（as_of == 金标版本时间戳），只能当受控对照。',
              file=sys.stderr)

    mod = load_evaluator()
    sampled, total, stride = sample_episodes(mod, args)
    episodes = sampled[args.shard::args.shards] if args.shards > 1 else sampled
    budget = pool_stats(mod, episodes, args.neighbours)
    sample = {'total': total, 'stride': stride, 'budget': budget,
              'first': episodes[0]['id'], 'last': episodes[-1]['id']}

    # 权重/配置预检（只读 json，不加载 7B reader）
    if not args.plumbing_stub:
        try:
            config, training_root, manifest = mod._load_v6(args.arch_exp)
        except Exception as exc:                       # noqa: BLE001 - 预检失败即终止
            raise SystemExit(f'{args.arch_exp} 权重预检失败：{exc}')
        print(f'[preflight] {args.arch_exp}: {training_root} | codec_value_path='
              f'{config.get("codec_value_path")} | val manifest={manifest}', file=sys.stderr)

    print(json.dumps({'shard': args.shard, 'shards': args.shards,
                      'episodes': len(episodes),
                      'first_episode': episodes[0]['id'],
                      'last_episode': episodes[-1]['id'],
                      'budget': budget}, ensure_ascii=False, indent=2))

    if args.plan_only:
        plan = {'tag': args.tag, 'plan_only': True, 'shard': args.shard,
                'shards': args.shards, 'protocol': sample,
                'entry_commands': [
                    ' '.join([f'CUDA_VISIBLE_DEVICES=<gpu{index}>', 'env_py310/bin/python', '-u',
                              'probe/run_auto_association_loss.py',
                              '--limit', str(args.limit),
                              '--neighbours', str(args.neighbours),
                              '--recency-update', '--top-k', str(args.top_k),
                              '--arch-exp', args.arch_exp,
                              '--shard', str(index), '--shards', str(args.shards),
                              '--out', str(args.out), '--tag', args.tag])
                    for index in range(args.shards)]
                + [' '.join(['CUDA_VISIBLE_DEVICES=', 'env_py310/bin/python', '-u',
                             'probe/run_auto_association_loss.py', '--merge',
                             '--out', str(args.out), '--tag', args.tag])],
                'equivalent_per_mode': [
                    {'mode': mode,
                     'cmd': ' '.join(['env_py310/bin/python', '-u', 'probe/run_state_eval.py']
                                     + build_eval_argv(args, EVAL_MODES[mode], out))}
                    for mode in modes],
                'note': 'plan_only：未加载 reader、未跑实验、未写任何 summary。'
                        'equivalent_per_mode 只是同参数的等价单跑写法；oracle_pooled 必须在 '
                        'run_auto_association_loss.py 内执行（run_state_eval 没有该模式）。'}
        out.mkdir(parents=True, exist_ok=True)
        mod.write_json(out / f'{args.tag}_plan_s{args.shard}.json', plan)
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return plan

    gpu = None if args.plumbing_stub else assert_gpu_policy()
    out.mkdir(parents=True, exist_ok=True)

    eval_args = mod.parse_args(build_eval_argv(args, 'predicted_state', out))
    started = time.perf_counter()
    components = mod.build_components(eval_args)
    print(f'[components] {components.label} | load_seconds='
          f'{time.perf_counter() - started:.1f}', file=sys.stderr)
    if not args.plumbing_stub and not str(components.label).startswith('icae+'):
        raise SystemExit(f'期望真实 ICAE 组件，实际 {components.label!r}：拒绝把 stub 结果当结论')

    protocol = _protocol(args, sample, budget, components.label)
    for mode in modes:
        started = time.perf_counter()
        if mode == 'predicted':
            run_args = eval_args
            metrics = mod.run_predicted_state(run_args, episodes, components)
        elif mode == 'oracle_global':
            run_args = mod.parse_args(build_eval_argv(args, 'oracle_state', out))
            metrics = mod.run_oracle_state(run_args, episodes, components)
        else:
            run_args = mod.parse_args(build_eval_argv(args, 'oracle_state', out))
            run_args.mode = 'oracle_pooled'
            metrics = run_oracle_pooled(mod, run_args, episodes, components)
        wall = time.perf_counter() - started
        summary, rows = dump_pass(mod, run_args, metrics, components, episodes, mode, out,
                                  started)
        bundle = bundle_from(metrics, mode, wall, len(episodes), protocol)
        mod.write_json(out / f'{args.tag}_bundle_s{args.shard}_{mode}.json', bundle)
        print(f'[shard {args.shard}/{args.shards}] {mode}: episodes={len(episodes)} '
              f'queries={len(rows)} writes={summary["writes"]} '
              f'em={summary["em"]:.4f} e2e_em={summary["end_to_end_em"]:.4f} '
              f'recall@1={summary["query_state_recall_at_1"]} wall={wall:.1f}s',
              file=sys.stderr)
    print(f'[shard {args.shard}/{args.shards}] done: {len(modes)} pass, '
          f'bundle -> {out}/{args.tag}_bundle_s{args.shard}_*.json', file=sys.stderr)
    return {'shard': args.shard, 'shards': args.shards, 'modes': modes,
            'episodes': len(episodes), 'out': str(out)}


if __name__ == '__main__':
    main()
