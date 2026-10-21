from __future__ import annotations
# P2 脚手架（只写不跑）：从 TempLAMA **train** 划分构造「同状态」监督数据。
#
# 目的：给 vcrm.update_resolver 的 rule4/rule5 提供可校准的「同状态判定」训练/验证信号。
#
# 铁律：
#   - 只读 templama/train.json，绝不读 test 划分，也绝不使用 state_key / key_text /
#     answers / latest / episode 等目标侧字段；
#   - relation 字段只用于「同主体不同属性」困难负样本的取样，**不**作为 attribute 来源；
#     attribute 一律由 vcrm.fact_extractor 从自然语言表面形式归一化得到；
#   - 实体级切分：train/val 的实体集合互斥（哈希分桶），保证阈值在独立验证集上校准。
#
# 用法（本轮不执行）：
#   python probe/build_state_supervision.py --data-root /share/latent2/vcrm_icae/data \
#       --out probe/runs/state_supervision.jsonl

import argparse
import hashlib
import json
import pathlib
import sys
from typing import Dict, Iterable, List, Optional, Tuple

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from vcrm.fact_extractor import FactExtractor  # noqa: E402
from vcrm.state_index import normalize_attribute, normalize_subject, value_fingerprint  # noqa: E402
from vcrm.update_resolver import values_equivalent  # noqa: E402

#: 允许读取的划分；test 一律拒绝。
ALLOWED_SPLITS = ('train',)


def read_jsonl(path: pathlib.Path) -> Iterable[dict]:
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def templama_event_text(row: dict) -> str:
    """还原 vcrm.data.templama 的事件文本形态（In <date>: <query 填空>）。

    只使用 query / date / answer.name：不用 relation，也不用任何目标侧字段。
    """
    answers = [item['name'] for item in row['answer']]
    filled = row['query'].replace('_X_', ' / '.join(answers))
    return f"In {row['date']}: {filled}"


def entity_bucket(entity: str, val_ratio: int = 5) -> str:
    """实体级哈希分桶：保证同一实体的所有年份落在同一侧（train 或 val）。"""
    digest = hashlib.blake2b(entity.encode('utf-8'), digest_size=8).hexdigest()
    return 'val' if int(digest, 16) % val_ratio == 0 else 'train'


def extract_stream(rows: List[dict], extractor: FactExtractor) -> List[dict]:
    """把 train 行抽成带时间的事实流（按 entity 分组，保持时间顺序）。"""
    by_entity: Dict[str, List[dict]] = {}
    for row in rows:
        entity = str(row['id']).split('_')[0]
        event = {'id': f'input_{len(by_entity):06d}_{row["id"]}', 'text': templama_event_text(row),
                 'time': f'{row["date"]}-01-01T00:00:00'}
        facts = extractor.extract(event)
        for fact in facts:
            if fact.subject.startswith('<unresolved'):
                continue
            by_entity.setdefault(entity, []).append({
                'entity': entity,
                'subject_key': normalize_subject(fact.subject),
                'attribute_key': normalize_attribute(fact.attribute),
                'value': fact.value,
                'value_fingerprint': value_fingerprint(fact.value),
                'observed_at': fact.observed_at,
                'text': fact.text,
                'source_id': fact.source_id,
            })
    for entity in by_entity:
        by_entity[entity].sort(key=lambda item: item['observed_at'])
    return [fact for entity in sorted(by_entity) for fact in by_entity[entity]]


def build_pairs(facts: List[dict]) -> List[dict]:
    """构造正样本 / 重复样本 / 困难负样本 / 随机负样本。"""
    pairs: List[dict] = []
    seen = set()

    def emit(left: dict, right: dict, label: int, kind: str) -> None:
        key = (left['source_id'], left['value'], right['source_id'], right['value'], kind)
        if key in seen:
            return
        seen.add(key)
        pairs.append({
            'text_a': left['text'],
            'text_b': right['text'],
            'subject_a': left['subject_key'], 'attribute_a': left['attribute_key'],
            'subject_b': right['subject_key'], 'attribute_b': right['attribute_key'],
            'value_a': left['value'], 'value_b': right['value'],
            'label': label,
            'kind': kind,
        })

    by_identity: Dict[Tuple[str, str], List[dict]] = {}
    by_subject: Dict[str, List[dict]] = {}
    by_attribute: Dict[str, List[dict]] = {}
    for fact in facts:
        by_identity.setdefault((fact['subject_key'], fact['attribute_key']), []).append(fact)
        by_subject.setdefault(fact['subject_key'], []).append(fact)
        by_attribute.setdefault(fact['attribute_key'], []).append(fact)

    for identity, series in by_identity.items():
        for previous, current in zip(series, series[1:]):
            if previous['value_fingerprint'] == current['value_fingerprint']:
                emit(previous, current, 1, 'duplicate_same_value')       # 正样本：同值重复
            elif values_equivalent(previous['value'], current['value']):
                emit(previous, current, 1, 'duplicate_equivalent_value')
            else:
                emit(previous, current, 1, 'update_same_identity')       # 正样本：同状态换值

    # 困难负样本 1：同主体、不同属性
    for subject, series in by_subject.items():
        attributes = sorted({item['attribute_key'] for item in series})
        for i, attribute in enumerate(attributes):
            for other in attributes[i + 1:]:
                left = next(item for item in series if item['attribute_key'] == attribute)
                right = next(item for item in series if item['attribute_key'] == other)
                emit(left, right, 0, 'hard_negative_same_subject')

    # 困难负样本 2：不同主体、同属性（尽量挑同值，逼模型只看主体）
    for attribute, series in by_attribute.items():
        subjects = sorted({item['subject_key'] for item in series})
        for i, subject in enumerate(subjects):
            for other in subjects[i + 1:]:
                left = next(item for item in series if item['subject_key'] == subject)
                right = next(item for item in series if item['subject_key'] == other)
                emit(left, right, 0, 'hard_negative_same_attribute')

    # 随机负样本：主体与属性都不同
    flat = [item for item in facts]
    for index in range(0, min(len(flat) - 1, 512), 2):
        left, right = flat[index], flat[index + 1]
        if (left['subject_key'], left['attribute_key']) != (right['subject_key'], right['attribute_key']):
            emit(left, right, 0, 'random_negative')
    return pairs


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description='Build same-state supervision from TempLAMA train')
    parser.add_argument('--data-root', default='/share/latent2/vcrm_icae/data')
    parser.add_argument('--split', default='train', choices=ALLOWED_SPLITS)
    parser.add_argument('--out', default='probe/runs/state_supervision.jsonl')
    parser.add_argument('--stats', default='probe/runs/state_supervision_stats.json')
    args = parser.parse_args(argv)

    if args.split not in ALLOWED_SPLITS:
        raise SystemExit(f'split {args.split!r} is not allowed: 只允许 {ALLOWED_SPLITS}')

    source = pathlib.Path(args.data_root) / 'templama' / f'{args.split}.json'
    rows = list(read_jsonl(source))
    extractor = FactExtractor()
    facts = extract_stream(rows, extractor)
    pairs = build_pairs(facts)

    # 实体级切分：同一实体不跨 train/val（阈值校准必须落在独立实体上）
    for pair in pairs:
        pair['split'] = entity_bucket(pair['subject_a'])

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8') as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, ensure_ascii=False) + '\n')

    stats = {
        'source': str(source),
        'split': args.split,
        'rows': len(rows),
        'facts': len(facts),
        'pairs': len(pairs),
        'by_kind': {kind: sum(1 for p in pairs if p['kind'] == kind)
                    for kind in sorted({p['kind'] for p in pairs})},
        'by_split': {name: sum(1 for p in pairs if p['split'] == name)
                     for name in ('train', 'val')},
        'unresolved_facts_dropped': len(extractor.unresolved),
        'note': ('threshold calibration must use the val split only; '
                 'never calibrate on the test split, never hardcode 0.75'),
    }
    stats_path = pathlib.Path(args.stats)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
