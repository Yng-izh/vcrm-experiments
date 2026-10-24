"""离线判分：给 run_dialogue_eval.py --defer-judge 产出的 rows 补官方 judge 分数。

先跑评测、后判分（用户授权的两段式）：
    # 第一段（不需要任何 judge 凭据）
    probe/run_dialogue_eval.py --defer-judge --limit 20 ...
    # 第二段（拿到 judge 配置后）
    probe/judge_dialogue_rows.py --rows probe/runs/dialogue_v1/dialogue_v1_rows.jsonl \
        --judge-config judge.json

只调用官方 LongMemEval prompt + judge 模型，不做任何字符串匹配替代；
某一题 judge 失败就保持未判分状态并在报告里列出，绝不猜。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vcrm.judge import score as judge_score      # noqa: E402

MODES = ('pair', 'trajectory')


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--rows', required=True, help='run_dialogue_eval 产出的 *_rows.jsonl')
    parser.add_argument('--judge-config', required=True)
    parser.add_argument('--judge-cache', default=str(REPO_ROOT / 'probe' / 'runs' / 'judge_cache'))
    parser.add_argument('--tag', default=None, help='输出文件名前缀（缺省 = <rows词干>_judged）')
    return parser.parse_args(argv)


def judge_row(row: Mapping[str, Any], config: Mapping[str, Any], cache: Path) -> Mapping[str, Any]:
    query = {'type': row.get('type'), 'question': row.get('question'),
             'answers': row.get('answers') or [''], 'abstention': row.get('abstention')}
    out = dict(row)
    for mode in MODES:
        cell = dict(row[mode])
        if cell.get('correct') is not None and cell.get('scoring_rule') == 'official_longmemeval_prompt':
            out[mode] = cell                       # 已判过分（如重跑），保留
            continue
        try:
            verdict = judge_score(config, query, str(cell.get('answer') or ''), cache)
            cell.update({'correct': int(verdict['accuracy']),
                         'scoring_rule': 'official_longmemeval_prompt',
                         'judge_model': verdict['judge_model']})
            if 'error' in cell:
                del cell['error']
        except Exception as exc:                   # noqa: BLE001 —— 失败显式记账，绝不替代
            cell.update({'correct': None, 'scoring_rule': 'score_error',
                         'error': f'{type(exc).__name__}: {exc}'})
        out[mode] = cell
    return out


def summarise(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    scored = [row for row in rows
              if all(row[mode].get('correct') is not None for mode in MODES)]
    failed = len(rows) - len(scored)
    pair_wins = sum(1 for row in scored if row['pair']['correct'] > row['trajectory']['correct'])
    traj_wins = sum(1 for row in scored if row['trajectory']['correct'] > row['pair']['correct'])
    by_type: Dict[str, Dict[str, Any]] = {}
    for kind in sorted({str(row['type']) for row in scored}):
        subset = [row for row in scored if str(row['type']) == kind]
        by_type[kind] = {mode: sum(row[mode]['correct'] for row in subset) / len(subset)
                         for mode in MODES} | {'n': len(subset)}
    return {'rows': len(rows), 'scored_rows': len(scored), 'judge_errors': failed,
            'overall': {mode: (sum(row[mode]['correct'] for row in scored) / len(scored))
                        if scored else None for mode in MODES},
            'paired': {'pair_wins': pair_wins, 'trajectory_wins': traj_wins,
                       'ties': len(scored) - pair_wins - traj_wins},
            'by_type': by_type}


def render(summary: Mapping[str, Any], components: str, tag: str) -> str:
    lines = [f'# 自然对话端到端评测（{tag}，离线判分）', '',
             f"- components: `{components}` | 题数={summary['rows']}"
             f"（判分成功 {summary['scored_rows']}，judge 失败 {summary['judge_errors']}）"]
    for mode, label in (('pair', 'A: pair'), ('trajectory', 'B: trajectory')):
        if summary['overall'][mode] is not None:
            lines.append(f'- {label}: acc={summary["overall"][mode]:.4f}')
    lines += ['', '| 类型 | n | pair | trajectory | Δ |', '| --- | --- | --- | --- | --- |']
    for kind, cell in summary['by_type'].items():
        lines.append(f"| {kind} | {cell['n']} | {cell['pair']:.4f} | {cell['trajectory']:.4f} "
                     f"| {cell['trajectory'] - cell['pair']:+.4f} |")
    paired = summary['paired']
    lines += ['', f"- 配对：trajectory 胜 {paired['trajectory_wins']} / pair 胜 {paired['pair_wins']}"
              f" / 平 {paired['ties']}"]
    if summary['judge_errors']:
        lines.append(f"- **judge 失败 {summary['judge_errors']} 题，未做任何替代判分**")
    lines += ['', '- 口径：端到端（状态身份由管线预测）；官方 LongMemEval prompt + 自定义 judge，'
              '非官方 GPT-4o 分数；LongMemEval-S 问题多不依赖版本更替。']
    return '\n'.join(lines) + '\n'


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    rows_path = Path(args.rows)
    rows: List[Dict[str, Any]] = [json.loads(line) for line in
                                  rows_path.read_text().splitlines() if line.strip()]
    config = json.loads(Path(args.judge_config).read_text())
    config.setdefault('longmemeval_root', str(REPO_ROOT))
    Path(args.judge_cache).mkdir(parents=True, exist_ok=True)

    stem = args.tag or f'{rows_path.stem}_judged'
    out_dir = rows_path.parent
    judged: List[Dict[str, Any]] = []
    started = time.perf_counter()
    for index, row in enumerate(rows, start=1):
        judged.append(judge_row(row, config, Path(args.judge_cache)))
        print(f'[judge] {index}/{len(rows)} {row.get("question_id")}', file=sys.stderr)
    summary = summarise(judged)
    summary['components'] = f"see {rows_path.parent.name}"
    summary['judge_seconds'] = round(time.perf_counter() - started, 1)
    (out_dir / f'{stem}_rows.jsonl').write_text(
        ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in judged))
    (out_dir / f'{stem}_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2))
    (out_dir / f'{stem}_report.md').write_text(
        render(summary, summary['components'], stem))
    print(json.dumps({k: summary[k] for k in ('rows', 'scored_rows', 'judge_errors', 'overall',
                                              'paired')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
