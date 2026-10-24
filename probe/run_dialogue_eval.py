"""自然对话数据集（LongMemEval-S）上的端到端评测。

与 run_trajectory_eval.py 的差别（这是刻意设计的，不是漏掉）
------------------------------------------------------------
* TempLAMA 实验一是**受控对照**：金标 state_key 写池、两臂只差 read_mode。
  LongMemEval 没有金标状态，所以这里走管线自己的抽取+路由（write_predicted），
  状态身份也是被测对象的一部分 —— 端到端口径，结论只能说"管线整体"。
* 题目用官方 LongMemEval-S 问题（1 题/条），**不造题**：对话没有 key_text，
  cross_version_questions 的模板造题在这里不成立。
* 判分用 vcrm.judge 的官方 LongMemEval prompt + 自定义 judge（带缓存）。
  judge 不可用时该题记 score_error，**绝不做字符串匹配替代**（与仓库的
  "no exact-match substitution" 规则一致）；report 会把两组分数分开报。

用法（服务器，GPU 9）
--------------------
    CUDA_VISIBLE_DEVICES=9 PYTHONPATH=.:probe env_py310/bin/python \\
        probe/run_dialogue_eval.py --split longmemeval_test --limit 30 \\
        --question-types temporal-reasoning knowledge-update \\
        --out probe/runs/dialogue_v1 --tag dialogue_v1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vcrm.judge import score as judge_score          # noqa: E402

MODES = ('pair', 'trajectory')
FORBIDDEN_PHYSICAL_GPUS = ('0', '1')
KNOWN_TYPES = ('single-session-user', 'single-session-assistant', 'single-session-preference',
               'multi-session', 'temporal-reasoning', 'knowledge-update')


def load_module(name: str, filename: str):
    import importlib.util
    path = REPO_ROOT / 'probe' / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# 参数
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--data-root', default='/share/latent2/vcrm_icae/data/prepared')
    parser.add_argument('--split', default='longmemeval_test')
    parser.add_argument('--episode-offset', type=int, default=0)
    parser.add_argument('--episodes', type=int, default=0, help='取前 N 条 episode（0=全部）')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--limit', type=int, default=30, help='最多评多少道题')
    parser.add_argument('--question-types', nargs='*', default=None,
                        choices=KNOWN_TYPES, help='只评这些官方类型（缺省=全部）')
    parser.add_argument('--arch-exp', default='E3', choices=['E1', 'E2', 'E3', 'E4'])
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--neighbours', type=int, default=4,
                        help='写池时附带多少条相邻 episode 的对话（上下文压力）')
    parser.add_argument('--max-slots', type=int, default=256)
    parser.add_argument('--max-versions', type=int, default=8)
    parser.add_argument('--plumbing-stub', action='store_true')
    parser.add_argument('--defer-judge', action='store_true',
                        help='跑评测时不调 judge，把回答原文存进 rows，判分留给 '
                             'probe/judge_dialogue_rows.py（rows 需要 judge 三要素才合法，'
                             '所以必须同时给 --judge-config/--env 指定路径，写进 summary 备用）')
    parser.add_argument('--judge-config', default='',
                        help='含 judge_model/judge_base_url/judge_key_env 的 json；'
                             '缺省读环境变量 VCRM_JUDGE_CONFIG，再缺省用 arch_exp 的 v6 config')
    parser.add_argument('--judge-cache', default=str(REPO_ROOT / 'probe' / 'runs' / 'judge_cache'))
    parser.add_argument('--tag', default='dialogue_eval')
    parser.add_argument('--out', default=str(REPO_ROOT / 'probe' / 'runs' / 'dialogue_eval'))
    parser.add_argument('--plan-only', action='store_true')
    return parser.parse_args(argv)


def eval_argv(args: argparse.Namespace, out: Path) -> List[str]:
    """复用 run_state_eval 的组件/运行时默认值。"""
    mod = load_module('vcrm_probe_run_state_eval', 'run_state_eval.py')
    return ['--mode', 'predicted_state', '--time-mode', 'window_end',
            '--data-root', args.data_root, '--split', args.split,
            '--limit', '0', '--tag', args.tag, '--out', str(out),
            '--budget-mode', 'retain_all', '--top-k', str(args.top_k),
            '--neighbours', str(args.neighbours), '--arch-exp', args.arch_exp,
            '--reader', 'stub' if args.plumbing_stub else 'icae',
            '--history-probe-mode', 'window_end']


def load_episodes(mod, args: argparse.Namespace) -> List[Mapping[str, Any]]:
    path = Path(args.data_root) / f'{args.split}.json'
    if not path.exists():
        available = sorted(p.stem for p in Path(args.data_root).glob('*.json'))
        raise SystemExit(f'缺少准备好的 split：{path}\n可用 split：{available}')
    episodes = [e for e in mod.read_json(path) if e.get('queries')]
    if args.question_types:
        wanted = set(args.question_types)
        episodes = [e for e in episodes if str(e['queries'][0].get('type')) in wanted]
    if args.episode_offset:
        episodes = episodes[args.episode_offset:]
    if args.episodes:
        episodes = episodes[:args.episodes]
    if args.shards > 1:
        episodes = episodes[args.shard::args.shards]
    if not episodes:
        raise SystemExit('筛选后没有可用 episode：放宽 --question-types / --episode-offset')
    return episodes


def load_judge_config(args: argparse.Namespace) -> Dict[str, str]:
    path = args.judge_config or os.environ.get('VCRM_JUDGE_CONFIG', '')
    if path and Path(path).exists():
        config = json.loads(Path(path).read_text())
    elif path:
        # defer-judge 冒烟测试常用假路径：只记账不拒绝，判分反正在第二段
        print(f'[dialogue] judge 配置 {path} 不存在（--defer-judge 模式下仅记录）', file=sys.stderr)
        config = {}
    else:
        import importlib.util
        loader = REPO_ROOT / 'probe' / 'v6_loader.py'
        spec = importlib.util.spec_from_file_location('vcrm_probe_v6_loader_judge', loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        config, _, _ = module.load_v6(args.arch_exp)
    missing = [k for k in ('judge_model', 'judge_base_url', 'judge_key_env') if not config.get(k)]
    if missing and not args.defer_judge:
        raise SystemExit(f'judge 配置缺字段 {missing}：用 --judge-config 指定或补 VCRM_JUDGE_CONFIG')
    config['longmemeval_root'] = config.get('longmemeval_root') or str(REPO_ROOT)
    return config


# ---------------------------------------------------------------------------
# 写入（端到端：管线自己抽事实、自己路由状态）
# ---------------------------------------------------------------------------
def write_pool(mod, runtime, pool) -> Dict[str, Any]:
    counter = mod.PublicEventCounter()
    facts: List[Dict[str, Any]] = []
    writes, seconds = 0, 0.0
    for member in pool:
        written, count, cost = mod.write_episode(runtime, member, counter)
        facts.extend(written)
        writes += count
        seconds += cost
    return {'facts': facts, 'writes': writes, 'seconds': seconds,
            'states': mod.primary_states(facts)}


# ---------------------------------------------------------------------------
# 单题评估：同一次写入，两臂只差 read_mode（与实验一相同的唯一变量原则）
# ---------------------------------------------------------------------------
def evaluate_question(runtime, episode, query, window_end, judge_config, cache_root,
                      args, write_stats) -> Dict[str, Any]:
    result_by_mode: Dict[str, Mapping[str, Any]] = {}
    for mode in MODES:
        result_by_mode[mode] = runtime.answer_predicted(
            query, window_end=window_end, read_mode=mode, observation_cutoff=window_end)
    judged: Dict[str, Any] = {}
    if args.defer_judge:
        # 离线判分模式：回答原文照存，判分字段全部留 None，等 judge_dialogue_rows.py 补
        for mode, result in result_by_mode.items():
            judged[mode] = {'answer': str(result.get('answer') or ''), 'correct': None,
                            'scoring_rule': 'deferred_judge', 'judge_model': None}
    else:
        for mode, result in result_by_mode.items():
            answer = str(result.get('answer') or '')
            try:
                verdict = judge_score(judge_config, query, answer, cache_root)
                judged[mode] = {'answer': answer, 'correct': int(verdict['accuracy']),
                                'scoring_rule': 'official_longmemeval_prompt',
                                'judge_model': verdict['judge_model']}
            except Exception as exc:                # noqa: BLE001 —— judge 失败必须显式记账
                judged[mode] = {'answer': answer, 'correct': None,
                                'scoring_rule': 'score_error', 'error': f'{type(exc).__name__}: {exc}'}
    row: Dict[str, Any] = {
        'tag': args.tag, 'episode': episode['id'],
        'question_id': str(query.get('id')), 'question': str(query.get('question')),
        'type': str(query.get('type')), 'answers': list(query.get('answers') or []),
        'abstention': bool(query.get('abstention')),
        'observation_cutoff': window_end,
        'n_sessions': len(episode.get('events') or []),
        'n_states_in_pool': len(write_stats['states']),
        'n_facts_in_pool': len(write_stats['facts']),
        'runtime_stats': runtime.stats(),
        'write_seconds': round(write_stats['seconds'], 3),
    }
    for mode in MODES:
        result, verdict = result_by_mode[mode], judged[mode]
        row[mode] = {
            **verdict,
            'state_id': result.get('state_id'), 'no_state_match': bool(result.get('no_state_match')),
            'parsed_subject': result.get('parsed_subject'),
            'parsed_attribute': result.get('parsed_attribute'),
            'selected_version_id': result.get('selected_version_id'),
            'version_role': result.get('version_role'),
            'read_latent_slots': int(result.get('read_latent_slots') or 0),
            'trajectory_length': int(result.get('trajectory_length') or 0),
            'time_source': result.get('time_source'),
            'read_seconds': round(float((result.get('timings') or {}).get('read_seconds') or 0), 3),
        }
    row['arms_agree'] = int(judged[MODES[0]]['answer'] == judged[MODES[1]]['answer'])
    return row


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------
def summarise(rows: Sequence[Mapping[str, Any]], args: argparse.Namespace,
              components_label: str, wall: float) -> Dict[str, Any]:
    scored = [row for row in rows if all(row[mode]['correct'] is not None for mode in MODES)]
    errors = [row for row in rows if row not in scored]
    pending = args.defer_judge and not scored           # defer 模式：全部等离线判分
    if pending:
        errors = []

    def block(items: Sequence[Mapping[str, Any]], mode: str) -> Dict[str, Any]:
        if not items:
            return {'n': 0}
        wins = sum(1 for row in items
                   if row[mode]['correct'] and not row['trajectory' if mode == 'pair' else 'pair']['correct'])
        return {'n': len(items), 'accuracy': sum(row[mode]['correct'] for row in items) / len(items),
                'wins': wins,
                'no_state_match': sum(1 for row in items if row[mode]['no_state_match']),
                'mean_slots': sum(row[mode]['read_latent_slots'] for row in items) / len(items),
                'mean_traj_len': sum(row[mode]['trajectory_length'] for row in items) / len(items)}

    by_type: Dict[str, Dict[str, Any]] = {}
    for kind in sorted({str(row['type']) for row in scored}):
        subset = [row for row in scored if str(row['type']) == kind]
        by_type[kind] = {mode: block(subset, mode) for mode in MODES}
    pair_wins = sum(1 for row in scored if row['pair']['correct'] > row['trajectory']['correct'])
    traj_wins = sum(1 for row in scored if row['trajectory']['correct'] > row['pair']['correct'])
    return {
        'tag': args.tag, 'protocol': 'dialogue_end_to_end_v1',
        'split': args.split, 'question_types': args.question_types or 'all',
        'components': components_label, 'rows': len(rows), 'scored_rows': len(scored),
        'judge_errors': len(errors), 'judge_deferred': bool(args.defer_judge),
        'error_detail': [f"{row['question_id']}: {row['pair'].get('error')}" for row in errors][:5],
        'overall': {mode: block(scored, mode) for mode in MODES},
        'by_type': by_type,
        'paired': {'pair_wins': pair_wins, 'trajectory_wins': traj_wins,
                   'ties': len(scored) - pair_wins - traj_wins},
        'arms_agree_rate': (sum(row['arms_agree'] for row in rows) / len(rows)) if rows else None,
        'wall_seconds': round(wall, 1),
        'caveats': [
            '端到端口径：状态身份由管线自己预测，不是金标写池；结论属于"管线整体"。',
            '判分用官方 LongMemEval prompt + 自定义 judge（非官方 GPT-4o 分数）。',
            'LongMemEval-S 问题多不依赖版本更替；这里量的是"对话流入记忆后还能不能读出来"。',
        ],
    }


def render_markdown(summary: Mapping[str, Any]) -> str:
    lines = ['# 自然对话端到端评测（LongMemEval-S）', '']
    lines.append(f"- components: `{summary['components']}` | 题数={summary['rows']}"
                 f"（判分成功 {summary['scored_rows']}，judge 失败 {summary['judge_errors']}）"
                 f" | 类型过滤={summary['question_types']}")
    for mode, label in (('pair', 'A: pair'), ('trajectory', 'B: trajectory')):
        blk = summary['overall'][mode]
        if blk.get('n'):
            lines.append(f"- {label}: acc={blk['accuracy']:.4f} (n={blk['n']})"
                         f" no_state_match={blk['no_state_match']}"
                         f" slots={blk['mean_slots']:.1f} traj_len={blk['mean_traj_len']:.2f}")
    lines += ['', '| 类型 | n | pair | trajectory | Δ |', '| --- | --- | --- | --- | --- |']
    for kind, block_by_mode in summary['by_type'].items():
        p, t = block_by_mode['pair'], block_by_mode['trajectory']
        if not p.get('n'):
            continue
        lines.append(f"| {kind} | {p['n']} | {p['accuracy']:.4f} | {t['accuracy']:.4f} "
                     f"| {t['accuracy'] - p['accuracy']:+.4f} |")
    paired = summary['paired']
    lines += ['', f"- 配对：trajectory 胜 {paired['trajectory_wins']} / pair 胜 {paired['pair_wins']}"
              f" / 平 {paired['ties']}（仅计 judge 成功的题）",
              f"- 两臂答案一致率：{summary['arms_agree_rate']}"]
    if summary['judge_errors']:
        lines.append(f"- **judge 失败 {summary['judge_errors']} 题，未做任何替代判分**，样例："
                     + '; '.join(summary['error_detail']))
    lines += ['', '## 口径', '']
    lines += [f'- {note}' for note in summary['caveats']]
    return '\n'.join(lines) + '\n'


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = parse_args(argv)
    mod = load_module('vcrm_probe_run_state_eval', 'run_state_eval.py')
    episodes = load_episodes(mod, args)
    print(json.dumps({'split': args.split, 'episodes_in_shard': len(episodes),
                      'limit': args.limit, 'types': args.question_types or 'all'},
                     ensure_ascii=False, indent=2))
    if args.plan_only:
        return {'plan_only': True, 'episodes': len(episodes),
                'types': sorted({str(e['queries'][0].get('type')) for e in episodes})}
    if not args.plumbing_stub:
        gpu_env = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        bad = [part for part in gpu_env.split(',') if part.strip() in FORBIDDEN_PHYSICAL_GPUS]
        if not gpu_env.strip() or bad:
            raise SystemExit('真实组件需要显式 CUDA_VISIBLE_DEVICES（GPU 0/1 禁用）')
    judge_config = load_judge_config(args)
    Path(args.judge_cache).mkdir(parents=True, exist_ok=True)

    run_args = mod.parse_args(eval_argv(args, Path(args.out)))
    started = time.perf_counter()
    components = mod.build_components(run_args)
    if not args.plumbing_stub and not str(components.label).startswith('icae+'):
        raise SystemExit(f'期望真实 ICAE 组件，实际 {components.label!r}')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for index, episode in enumerate(episodes, start=1):
        if len(rows) >= args.limit:
            break
        runtime = mod.new_runtime(run_args, components)
        pool = [episode] + mod._neighbours(episodes, index - 1, args.neighbours)
        write_stats = write_pool(mod, runtime, pool)
        window_end = mod.window_end_of(episode)
        for query in episode.get('queries') or []:
            if len(rows) >= args.limit:
                break
            rows.append(evaluate_question(runtime, episode, query, window_end,
                                          judge_config, args.judge_cache, args, write_stats))
        print(f'[dialogue] {index}/{len(episodes)} {episode["id"]} rows={len(rows)}',
              file=sys.stderr)
    if not rows:
        raise SystemExit('没有任何题被评估')

    summary = summarise(rows, args, components.label, time.perf_counter() - started)
    mod.write_json(out / f'{args.tag}_summary.json', summary)
    with (out / f'{args.tag}_rows.jsonl').open('w') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    (out / f'{args.tag}_report.md').write_text(render_markdown(summary))
    print(render_markdown(summary))
    return summary


if __name__ == '__main__':
    main()
