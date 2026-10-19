#!/usr/bin/env python3
"""Split templama episodes into N write-shards (md5(ep_id) % N), order-preserving.

Usage: python3 split_write_shards.py --num-shards 4
Writes runs/write_shard{k}of{N}_episodes.txt (one ep id per line) for k in 0..N-1.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--num-shards", type=int, default=4)
    ap.add_argument("--exclude-episodes-from", default=None,
                    help="file of episode ids already written (e.g. smoke episodes); "
                         "they keep their shard assignment but are excluded from shard files")
    args = ap.parse_args()

    eps = json.loads((ROOT / f"data/prepared/templama_{args.split}.json").read_text())
    ep_ids = sorted(ep["id"] for ep in eps)  # order-preserving, deterministic
    exclude = set()
    if args.exclude_episodes_from:
        exclude = {l.strip() for l in open(args.exclude_episodes_from) if l.strip()}

    shards = {k: [] for k in range(args.num_shards)}
    for eid in ep_ids:
        shards[int(hashlib.md5(eid.encode()).hexdigest(), 16) % args.num_shards].append(eid)

    out_dir = HERE / "runs"
    out_dir.mkdir(exist_ok=True)
    for k in range(args.num_shards):
        keep = [e for e in shards[k] if e not in exclude]
        p = out_dir / f"write_shard{k}of{args.num_shards}_episodes.txt"
        p.write_text("\n".join(keep) + "\n")
        print(f"shard {k}: {len(keep)} episodes (excluded {len(shards[k]) - len(keep)})",
              file=sys.stderr)
    total = sum(len(v) for v in shards.values())
    print(f"TOTAL {total} episodes across {args.num_shards} shards", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
