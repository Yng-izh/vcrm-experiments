from __future__ import annotations
"""真实存储字节测量：逐 episode / 逐写入 dump payload / index / metadata 明细。

用途：第四步"先测真实消耗再定预算"。本脚本**只测量、不训练、不评测**，也不读
测试集 state_key 参与任何决策——它只是把 `IncrementalCurrentStore` 的
`accounting()` 前后差值记下来（公式见 docs/store_budget.md）。

字节口径（每次写入的增量，三项之和恒等于真实字节增量）：
    delta_total = delta_memory_bytes + delta_index_key_bytes + delta_metadata_bytes
    payload  = latent 载荷（current 128×dim×2 B / 历史 code 16×dim×2 B / pending 一整条 full）
    index    = 语义 key（key_dimensions × 2 B）
    metadata = header_json + magic + 长度字段 + aux 段（含 pending 描述、duplicates、rejections）

用法：
    env_py310/bin/python probe/measure_store_bytes.py --episodes 50 \\
        --dataset /path/to/longmemeval_s.json --out runs/store_bytes.json

    # 没有数据集时可用合成写入计划（同样精确测量字节，只是 latent 内容是随机的）
    env_py310/bin/python probe/measure_store_bytes.py --episodes 20 --out /tmp/store_bytes.json

    # 顺带量一条"相同预算下 drop_oldest 会淘汰多少"（只读统计，不改默认口径）
    env_py310/bin/python probe/measure_store_bytes.py --episodes 20 --budget-fraction 0.6 \\
        --out /tmp/store_bytes.json
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from vcrm.codec import SlotCodec
from vcrm.fact_schema import StateDecision
from vcrm.index import key_vector
from vcrm.store import IncrementalCurrentStore

# E3 正式配置（与 vcrm/codec.py 的 SlotCodec 参数一致；只用于构造同构 codec）
DEFAULT_DIMENSION = 4096
DEFAULT_CODEC_SLOTS = 16
DEFAULT_KEY_DIMENSIONS = 128


def build_codec(dimension: int, codec_slots: int) -> SlotCodec:
    """同构 E3 codec：本脚本只量字节，不需要训练权重（payload 大小与权重无关）。"""
    torch.manual_seed(7)
    return SlotCodec(dimension, codec_slots, 256, 4, False, 0., 7, 'residual', None,
                     'full', 8.0).eval().requires_grad_(False)


def full_memory(seed: int, dimension: int, slots: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return (torch.randn(slots, dimension, generator=generator) * .5).half().float()


def synthetic_plan(episodes: int, facts_per_episode: int, states: int) -> List[Dict[str, Any]]:
    """合成写入计划：每个 episode 内多轮陈述同一批状态（制造 CREATE→UPDATE 版本链）。"""
    plan = []
    for episode in range(episodes):
        events = []
        for index in range(facts_per_episode):
            # 全局单调递增序号：跨 episode 复用的状态也必须按时间顺序写入
            seq = episode * facts_per_episode + index
            state = f's{seq % states:04d}'
            year = 2016 + (seq // states)
            events.append({'id': f'e{episode:04d}_f{index:04d}', 'state': state,
                           'time': f'{year}-01-01T00:00:00'})
        plan.append({'episode_id': f'synthetic_{episode:04d}', 'events': events})
    return plan


def dataset_plan(path: Path, episodes: int) -> List[Dict[str, Any]]:
    """从 LongMemEval 风格 JSON 读 episode → 写入计划。

    只用**公开**字段（事件 id / 时间 / 文本），状态身份用 (subject, attribute) 的
    粗粒度哈希占位：本脚本量的是字节，不做自动关联质量评估。
    """
    raw = json.loads(path.read_text())
    if isinstance(raw, dict):
        raw = raw.get('episodes') or raw.get('data') or []
    plan = []
    for index, episode in enumerate(list(raw)[:episodes]):
        events = []
        for order, event in enumerate(episode.get('haystack_events') or episode.get('events') or []):
            text = str(event.get('text') or event.get('content') or '')
            words = [word for word in text.replace(':', ' ').split() if word][:2]
            state = ':'.join(words) if words else f'state_{order:04d}'
            events.append({'id': f'{episode.get("id", f"ep{index:04d}")}_{order:04d}',
                           'state': state,
                           'time': str(event.get('time') or '2016-01-01T00:00:00')})
        if events:
            plan.append({'episode_id': str(episode.get('id', f'ep{index:04d}')), 'events': events})
    return plan


def kind_bytes(store: IncrementalCurrentStore) -> Dict[str, int]:
    """按条目类型拆 payload 字节：current(full) / historical(code) / pending。"""
    current = sum(int(e['payload'].nbytes) for e in store.entries if e['kind'] == 'full')
    historical = sum(int(e['payload'].nbytes) for e in store.entries if e['kind'] == 'code')
    pending = sum(int(r['payload'].nbytes) for r in store.pending)
    return {'current': current, 'historical': historical, 'pending': pending}


def measure(plan: Sequence[Dict[str, Any]], *, dimension: int, memory_slots: int,
            codec_slots: int, key_dimensions: int,
            pending_every: int = 0) -> Dict[str, Any]:
    codec = build_codec(dimension, codec_slots)
    store = IncrementalCurrentStore(codec, key_dimensions=key_dimensions)   # retain_all
    semantic_key = key_vector('measure store bytes', key_dimensions)
    episodes, seed = [], 0
    for episode in plan:
        per_write = []
        episode_totals = {'payload': 0, 'index': 0, 'metadata': 0, 'total': 0,
                          'current': 0, 'historical': 0, 'pending': 0}
        for event in episode['events']:
            before = store.accounting()
            kinds_before = kind_bytes(store)
            state = event['state']
            # --pending-every N：每第 N 条写成 DEFER，制造真实的 pending 字节
            deferred = bool(pending_every) and (seed + 1) % pending_every == 0
            if deferred:
                action = 'DEFER'
            else:
                action = 'CREATE' if store.current(state) is None else 'UPDATE'
            seed += 1
            receipt = store.ingest_decision(
                {'id': event['id'], 'time': event['time']},
                full_memory(seed, dimension, memory_slots), semantic_key,
                StateDecision(action, state, .4 if deferred else .9,
                              'measure_pending' if deferred else 'measure'))
            after = store.accounting()
            delta = {name: after[key] - before[key] for name, key in
                     (('payload', 'memory_bytes'), ('index', 'index_key_bytes'),
                      ('metadata', 'metadata_and_format_bytes'), ('total', 'total_bytes'))}
            # 三项之和必须等于真实字节增量（守卫用的是真实 serialize 值）
            assert delta['payload'] + delta['index'] + delta['metadata'] == delta['total'], (
                'byte accounting must add up', delta)
            kinds_after = kind_bytes(store)
            kinds_delta = {kind: kinds_after[kind] - kinds_before[kind]
                           for kind in ('current', 'historical', 'pending')}
            per_write.append({'id': event['id'], 'action': action, 'state': state,
                              'delta_bytes': delta['total'],
                              'payload_bytes': delta['payload'],
                              'index_bytes': delta['index'],
                              'metadata_bytes': delta['metadata'],
                              'payload_current_bytes': kinds_delta['current'],
                              'payload_historical_bytes': kinds_delta['historical'],
                              'payload_pending_bytes': kinds_delta['pending'],
                              'total_bytes_after': after['total_bytes']})
            for name in ('payload', 'index', 'metadata', 'total'):
                episode_totals[name] += delta[name]
            for name in ('current', 'historical', 'pending'):
                episode_totals[name] += kinds_delta[name]
        episodes.append({'episode_id': episode['episode_id'], 'writes': len(per_write),
                         'episode_write_bytes': episode_totals, 'per_write': per_write})
    writes = sum(episode['writes'] for episode in episodes)
    totals = {name: sum(episode['episode_write_bytes'][name] for episode in episodes)
              for name in ('payload', 'index', 'metadata', 'total',
                           'current', 'historical', 'pending')}
    resident = kind_bytes(store)
    resident_total = sum(resident.values()) or 1
    states = len({event['state'] for episode in plan for event in episode['events']})
    summary = {'episodes': len(episodes), 'writes': writes, 'states': states,
               'payload': totals['payload'], 'index': totals['index'],
               'metadata': totals['metadata'], 'total': totals['total'],
               'bytes_per_episode_mean': totals['total'] / max(1, len(episodes)),
               'bytes_per_write_mean': totals['total'] / max(1, writes),
               'per_state_mean': totals['total'] / max(1, states),
               'store_total_bytes': store.accounting()['total_bytes'],
               'resident_payload_by_kind': resident,
               'resident_payload_share': {kind: resident[kind] / resident_total
                                          for kind in resident},
               'written_payload_by_kind': {kind: totals[kind]
                                           for kind in ('current', 'historical', 'pending')},
               'written_payload_share': {kind: totals[kind] / (totals['payload'] or 1)
                                         for kind in ('current', 'historical', 'pending')},
               'floor_bytes_currents_only': states * (memory_slots * dimension * 2
                                                      + key_dimensions * 2),
               'final_accounting': store.accounting()}
    return {'config': {'dimension': dimension, 'memory_slots': memory_slots,
                       'codec_slots': codec_slots, 'key_dimensions': key_dimensions,
                       'budget_mode': store.budget_mode},
            'episodes': episodes, 'summary': summary}


def drop_oldest_probe(plan: Sequence[Dict[str, Any]], budget: int, *, dimension: int,
                      memory_slots: int, codec_slots: int, key_dimensions: int) -> Dict[str, Any]:
    """同一份写入序列在固定字节预算下的淘汰账（只统计，不改 retain_all 口径）。"""
    codec = build_codec(dimension, codec_slots)
    store = IncrementalCurrentStore(codec, key_dimensions=key_dimensions,
                                    byte_budget=budget, budget_mode='drop_oldest')
    semantic_key = key_vector('measure store bytes', key_dimensions)
    seed = 0
    for episode in plan:
        for event in episode['events']:
            state = event['state']
            action = 'CREATE' if store.current(state) is None else 'UPDATE'
            seed += 1
            try:
                store.ingest_decision({'id': event['id'], 'time': event['time']},
                                      full_memory(seed, dimension, memory_slots), semantic_key,
                                      StateDecision(action, state, .4 if deferred else .9,
                              'measure_pending' if deferred else 'measure'))
            except ValueError as error:      # 地板都装不下：记录后停止（不静默吞掉）
                return {'byte_budget': budget, 'rejected': True, 'error': str(error),
                        'evictions': store.evictions, 'accounting': store.accounting()}
    return {'byte_budget': budget, 'rejected': False,
            'evicted_ids': store.accounting()['evicted_ids'],
            'evicted_bytes': store.accounting()['evicted_bytes'],
            'evictions': store.evictions, 'accounting': store.accounting()}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description='Measure real incremental store bytes per episode')
    parser.add_argument('--episodes', type=int, default=20, help='episode 数（默认 20）')
    parser.add_argument('--out', type=Path, required=True, help='输出 JSON 路径')
    parser.add_argument('--dataset', type=Path, default=None,
                        help='可选：LongMemEval 风格 JSON；缺省用合成写入计划')
    parser.add_argument('--facts-per-episode', type=int, default=12, help='合成计划的每 episode 事实数')
    parser.add_argument('--states', type=int, default=4, help='合成计划的状态数')
    parser.add_argument('--dimension', type=int, default=DEFAULT_DIMENSION)
    parser.add_argument('--memory-slots', type=int, default=128, help='current 的 slot 数')
    parser.add_argument('--codec-slots', type=int, default=DEFAULT_CODEC_SLOTS)
    parser.add_argument('--key-dimensions', type=int, default=DEFAULT_KEY_DIMENSIONS)
    parser.add_argument('--pending-every', type=int, default=0,
                        help='每第 N 条写成 DEFER（制造 pending 字节，默认 0=不产生 pending）')
    parser.add_argument('--budget-fraction', type=float, default=None,
                        help='可选：按 summary.total × fraction 作为固定预算，量 drop_oldest 淘汰账')
    args = parser.parse_args(argv)

    plan = (dataset_plan(args.dataset, args.episodes) if args.dataset
            else synthetic_plan(args.episodes, args.facts_per_episode, args.states))
    if not plan:
        raise SystemExit('no events to measure: check --dataset / --facts-per-episode')

    report = measure(plan, dimension=args.dimension, memory_slots=args.memory_slots,
                     codec_slots=args.codec_slots, key_dimensions=args.key_dimensions,
                     pending_every=args.pending_every)
    if args.budget_fraction is not None:
        budget = int(report['summary']['total'] * args.budget_fraction)
        report['drop_oldest'] = drop_oldest_probe(
            plan, budget, dimension=args.dimension, memory_slots=args.memory_slots,
            codec_slots=args.codec_slots, key_dimensions=args.key_dimensions)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))

    summary = report['summary']
    print(f'episodes={summary["episodes"]} writes={summary["writes"]} states={summary["states"]}')
    print(f'bytes: payload={summary["payload"]} index={summary["index"]} '
          f'metadata={summary["metadata"]} total={summary["total"]}')
    print(f'payload_by_kind={summary["resident_payload_by_kind"]} '
          f'share={ {k: round(v, 4) for k, v in summary["resident_payload_share"].items()} }')
    print(f'per_episode={summary["bytes_per_episode_mean"]:.0f} '
          f'per_write={summary["bytes_per_write_mean"]:.0f} '
          f'floor={summary["floor_bytes_currents_only"]}')
    if 'drop_oldest' in report:
        print(f'drop_oldest: budget={report["drop_oldest"]["byte_budget"]} '
              f'evicted={len(report["drop_oldest"].get("evicted_ids", []))} '
              f'rejected={report["drop_oldest"]["rejected"]}')
    print(f'wrote {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
