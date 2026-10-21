from __future__ import annotations
# P2 脚手架（只写不跑）：冻结编码器 + 轻量二分类头，判断两条事实是否同一状态。
#
# 铁律：
#   - 编码器**冻结**（不训练、不微调，不碰 ICAE / E3 Codec / Frozen Reader 的权重）；
#   - 只训练一个轻量二分类头（LayerNorm + MLP），输入是冻结特征；
#   - 判定阈值**必须在独立验证集上校准**（实体级互斥切分），
#     禁止硬编码 0.75，也不允许用测试集调阈值；
#   - 本轮只交付代码，不执行训练、不跑 GPU。
#
# 用法（本轮不执行）：
#   python probe/build_state_supervision.py --out probe/runs/state_supervision.jsonl
#   python probe/train_same_state.py --data probe/runs/state_supervision.jsonl \
#       --out probe/runs/same_state_head.pt --report probe/runs/same_state_calibration.json
#
# 说明：默认特征器是确定性哈希特征（CPU 可跑通），只用于验证脚手架管线；
# 论文数字必须换成 --encoder-checkpoint 指向的冻结 ICAE 编码器。

import argparse
import json
import math
import pathlib
import sys
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from torch import nn  # noqa: E402

from vcrm.index import key_vector  # noqa: E402

#: 阈值校准策略：只允许在独立验证集上按目标 FPR 选阈值。
CALIBRATION_POLICY = (
    'Threshold MUST be calibrated on the entity-disjoint validation split '
    '(maximise F1 subject to FPR <= max_fpr). Never calibrate on the test split, '
    'never hardcode 0.75.'
)
FORBIDDEN_HARDCODED_THRESHOLD = 0.75
FEATURE_DIMENSIONS = 128


def read_jsonl(path: pathlib.Path) -> List[dict]:
    rows = []
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def hashed_features(text: str, dimensions: int = FEATURE_DIMENSIONS) -> torch.Tensor:
    """确定性哈希特征（CPU 调试用；不是论文用的冻结编码器）。"""
    vector = key_vector(text, dimensions).astype('float32')
    return torch.from_numpy(vector)


class FrozenEncoder:
    """冻结编码器包装：默认哈希特征，可选加载 ICAE 权重（只做前向）。"""

    def __init__(self, checkpoint: Optional[str] = None, dimensions: int = FEATURE_DIMENSIONS):
        self.dimensions = dimensions
        self.module = None
        if checkpoint:
            # 延迟导入，避免 CPU 单测环境强依赖 GPU 栈
            from vcrm.index import SemanticIndex
            state = torch.load(checkpoint, map_location='cpu')
            hidden = int(state.get('hidden', 4096)) if isinstance(state, dict) else 4096
            self.module = SemanticIndex(hidden=hidden, dimensions=dimensions)
            payload = state.get('state_dict', state) if isinstance(state, dict) else state
            self.module.load_state_dict(payload, strict=False)
            self.module.eval()
            for parameter in self.module.parameters():
                parameter.requires_grad_(False)

    @torch.no_grad()
    def __call__(self, text: str) -> torch.Tensor:
        if self.module is None:
            return hashed_features(text, self.dimensions)
        memory = torch.zeros(1, self.module.hidden)
        key = self.module.key(memory)
        return torch.from_numpy(key.astype('float32'))


class SameStateHead(nn.Module):
    """轻量二分类头：输入两条事实的冻结特征差/积/和。"""

    def __init__(self, dimensions: int = FEATURE_DIMENSIONS, hidden: int = 64):
        super().__init__()
        self.norm = nn.LayerNorm(dimensions * 4)
        self.net = nn.Sequential(
            nn.Linear(dimensions * 4, hidden), nn.GELU(),
            nn.Linear(hidden, 1))

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        features = torch.cat([left, right, (left - right).abs(), left * right], dim=-1)
        return self.net(self.norm(features)).squeeze(-1)


def build_dataset(rows: Sequence[dict], encoder: FrozenEncoder) -> Tuple[torch.Tensor, ...]:
    features, labels = [], []
    for row in rows:
        left = encoder(row['text_a'])
        right = encoder(row['text_b'])
        features.append(torch.cat([left, right, (left - right).abs(), left * right]))
        labels.append(float(row['label']))
    if not features:
        raise SystemExit('empty supervision split')
    return torch.stack(features), torch.tensor(labels, dtype=torch.float32)


def split_rows(rows: Sequence[dict]) -> Tuple[List[dict], List[dict]]:
    train = [row for row in rows if row.get('split') == 'train']
    val = [row for row in rows if row.get('split') == 'val']
    if not train or not val:
        raise SystemExit('need both train and val pairs; run build_state_supervision.py first')
    return train, val


def roc_auc(scores: Sequence[float], labels: Sequence[float]) -> float:
    positives = [s for s, y in zip(scores, labels) if y >= 0.5]
    negatives = [s for s, y in zip(scores, labels) if y < 0.5]
    if not positives or not negatives:
        return float('nan')
    wins = sum(1.0 for p in positives for n in negatives if p > n)
    ties = sum(0.5 for p in positives for n in negatives if p == n)
    return (wins + ties) / (len(positives) * len(negatives))


def calibrate_threshold(scores: Sequence[float], labels: Sequence[float],
                        max_fpr: float = 0.1) -> Dict[str, float]:
    """在独立验证集上按「F1 最大且 FPR <= max_fpr」选阈值（不硬编码）。"""
    best: Dict[str, float] = {'threshold': 0.0, 'f1': -1.0, 'fpr': 1.0, 'tpr': 0.0}
    for step in range(1, 100):
        threshold = step / 100.0
        tp = sum(1 for s, y in zip(scores, labels) if s >= threshold and y >= 0.5)
        fp = sum(1 for s, y in zip(scores, labels) if s >= threshold and y < 0.5)
        fn = sum(1 for s, y in zip(scores, labels) if s < threshold and y >= 0.5)
        tn = sum(1 for s, y in zip(scores, labels) if s < threshold and y < 0.5)
        fpr = fp / max(fp + tn, 1)
        tpr = tp / max(tp + fn, 1)
        precision = tp / max(tp + fp, 1)
        f1 = 0.0 if tp == 0 else 2 * precision * tpr / max(precision + tpr, 1e-9)
        if fpr <= max_fpr and f1 > best['f1']:
            best = {'threshold': threshold, 'f1': f1, 'fpr': fpr, 'tpr': tpr}
    if abs(best['threshold'] - FORBIDDEN_HARDCODED_THRESHOLD) < 1e-12:
        best['warning'] = ('calibrated threshold coincides with the forbidden hardcoded 0.75; '
                           're-check the calibration split before using it')
    return best


def train(args: argparse.Namespace) -> int:
    rows = read_jsonl(pathlib.Path(args.data))
    train_rows, val_rows = split_rows(rows)
    encoder = FrozenEncoder(args.encoder_checkpoint)
    features, labels = build_dataset(train_rows, encoder)
    val_features, val_labels = build_dataset(val_rows, encoder)

    head = SameStateHead(args.dimensions, args.hidden)
    optimiser = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-2)
    loss_fn = nn.BCEWithLogitsLoss()

    best_state, best_auc = None, -1.0
    for epoch in range(args.epochs):
        head.train()
        permutation = torch.randperm(features.shape[0])
        total = 0.0
        for start in range(0, features.shape[0], args.batch_size):
            batch = permutation[start:start + args.batch_size]
            optimiser.zero_grad()
            logits = head(features[batch, :args.dimensions], features[batch, args.dimensions:2 * args.dimensions])
            loss = loss_fn(logits, labels[batch])
            loss.backward()
            optimiser.step()
            total += float(loss) * len(batch)
        head.eval()
        with torch.no_grad():
            val_logits = head(val_features[:, :args.dimensions], val_features[:, args.dimensions:2 * args.dimensions])
            scores = torch.sigmoid(val_logits).tolist()
        auc = roc_auc(scores, val_labels.tolist())
        print(f'epoch {epoch}: loss={total / max(len(labels), 1):.4f} val_auc={auc:.4f}')
        if auc > best_auc:
            best_auc, best_state = auc, {k: v.clone() for k, v in head.state_dict().items()}

    if best_state is not None:
        head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        val_logits = head(val_features[:, :args.dimensions], val_features[:, args.dimensions:2 * args.dimensions])
        scores = torch.sigmoid(val_logits).tolist()
    calibration = calibrate_threshold(scores, val_labels.tolist(), args.max_fpr)
    calibration.update({'val_auc': best_auc, 'policy': CALIBRATION_POLICY,
                        'val_pairs': len(val_rows), 'train_pairs': len(train_rows)})

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': head.state_dict(), 'threshold': calibration['threshold'],
                'dimensions': args.dimensions, 'hidden': args.hidden,
                'calibration': calibration}, out)
    report = pathlib.Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(calibration, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(calibration, ensure_ascii=False, indent=2))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description='Train a frozen-encoder same-state head (not run in P2)')
    parser.add_argument('--data', default='probe/runs/state_supervision.jsonl')
    parser.add_argument('--out', default='probe/runs/same_state_head.pt')
    parser.add_argument('--report', default='probe/runs/same_state_calibration.json')
    parser.add_argument('--encoder-checkpoint', default=None,
                        help='冻结 ICAE 编码器权重（缺省用确定性哈希特征，仅调试）')
    parser.add_argument('--dimensions', type=int, default=FEATURE_DIMENSIONS)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max-fpr', type=float, default=0.1)
    parser.add_argument('--threshold', type=float, default=None,
                        help='禁止使用：阈值必须在独立验证集校准')
    args = parser.parse_args(argv)
    if args.threshold is not None:
        raise SystemExit(
            f'--threshold is not allowed: {CALIBRATION_POLICY} '
            f'(hardcoded {FORBIDDEN_HARDCODED_THRESHOLD} is explicitly forbidden)')
    return train(args)


if __name__ == '__main__':
    raise SystemExit(main())
