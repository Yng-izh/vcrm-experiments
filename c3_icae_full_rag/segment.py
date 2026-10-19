"""C3 offline segmentation stats (CONTRACT_C3_ICAE_FULL_RAG.md §1.4, frozen).

For every episode in data/prepared/templama_test.json: tokenize the full
history event stream (Mistral tokenizer, same tokenizer as the ICAE encoder).
Total <=5,120 tokens -> single segment, no split. Total >5,120 -> greedy
sequential packing at EVENT boundaries (never truncate an event text);
segments keep original event order and verbatim texts joined by "\n".

Label-free: reads only ep["id"] and ep["events"][*]["text"/"time"]. Oracle
fields (answers/state_key/as_of/latest) are never touched (AST-audited, see
ast_audit in run_probe.py; this file's read path uses an explicit field
allowlist).

Output: c3_icae_full_rag/segment_stats.json
  {
    "contract": "CONTRACT_C3_ICAE_FULL_RAG.md",
    "tokenizer": "models/Mistral-7B-Instruct-v0.2",
    "encoder_max_tokens": 5120,
    "slot_budget_segments": 8,
    "episodes_total": N,
    "episodes_over_limit": n_over,          # >5,120 tokens -> segmented
    "episodes_evicted": n_evicted,          # segments_total > 8 -> FIFO eviction
    "eviction_rate": ...,                   # episodes_evicted / episodes_total
    "segments_total": S,                    # sum of segments over all episodes
    "segments_distribution": {"1": ..., "2": ...},
    "evicted_segments_total": ...,          # segments dropped by FIFO eviction
    "oversize_events": [],                  # single events >5,120 tokens (must be empty)
    "episodes": {ep_id: {"tokens_total", "segments_total", "segments_kept",
                          "segments_evicted", "segment_token_counts"}}
  }
The per-episode segment plans (segment_token_counts) are the input for the
offline encode cache (step 2); the cache is keyed by segment text, not by plan.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path("/share/latent2/vcrm_icae")
OUT_DIR = ROOT / "c3_icae_full_rag"
STATS_PATH = OUT_DIR / "segment_stats.json"

ENCODER_MAX_TOKENS = 5120   # contract §1.1 (== config.json encoder_max_tokens)
SLOT_BUDGET_SEGMENTS = 8    # contract §1.4: 1,024 slots = 8 x 128


def sha16(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def segment_episode(events, tok):
    """Greedy event-boundary packing. events: [{time, text}] time-ascending.
    Returns (segment_texts, oversize_flags). No event text is ever truncated."""
    segments, cur, cur_tokens = [], [], 0
    oversize = []
    for ev in events:
        text = ev["text"]
        n = len(tok(text, add_special_tokens=True)["input_ids"])
        if n > ENCODER_MAX_TOKENS:
            # contract §1.2: stop and report; never crop
            oversize.append({"time": str(ev.get("time")), "tokens": n,
                             "text_head": text[:120]})
            continue  # recorded, excluded from packing (run is invalid if nonempty)
        joiner = len(tok("\n", add_special_tokens=False)["input_ids"]) if cur else 0
        if cur and cur_tokens + joiner + n > ENCODER_MAX_TOKENS:
            segments.append("\n".join(cur))
            cur, cur_tokens = [text], n
        else:
            cur.append(text)
            cur_tokens += joiner + n
    if cur:
        segments.append("\n".join(cur))
    return segments, oversize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    t0 = time.time()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(ROOT / "models/Mistral-7B-Instruct-v0.2"))

    episodes = json.loads((ROOT / f"data/prepared/templama_{args.split}.json").read_text())
    print(f"[seg] episodes={len(episodes)}", flush=True)

    # label-free read path: these fields are never read; assert we do not
    # touch the oracle fields anywhere (read path uses id/events only).
    FORBIDDEN = {"answers", "state_key", "as_of", "latest"}

    episodes_out = {}
    dist = {}
    n_over = n_evicted = evicted_segments = segments_total = 0
    oversize_all = []
    for i, ep in enumerate(episodes):
        assert not (set(ep) & FORBIDDEN), f"oracle field at episode level: {set(ep) & FORBIDDEN}"
        # events carry a state_key field in the source file; it is never read.
        # Guard only against oracle fields that must not even be present on
        # the routing path (answers/as_of/latest), per contract §7.
        EV_FORBIDDEN = {"answers", "as_of", "latest"}
        events = sorted(ep["events"], key=lambda e: str(e["time"]))
        for ev in events:
            assert not (set(ev) & EV_FORBIDDEN), f"oracle field in event: {set(ev) & EV_FORBIDDEN}"
        segments, oversize = segment_episode(events, tok)
        oversize_all.extend(oversize)
        total_tokens = sum(len(tok(s, add_special_tokens=True)["input_ids"])
                           for s in segments)
        st = len(segments)
        kept = min(st, SLOT_BUDGET_SEGMENTS)
        seg_evicted = st - kept
        # FIFO oldest-first eviction; token counts of DROPPED (oldest) segments
        dropped_tokens = [len(tok(s, add_special_tokens=True)["input_ids"])
                          for s in segments[:seg_evicted]]
        episodes_out[ep["id"]] = {
            "tokens_total": total_tokens,
            "segments_total": st,
            "segments_kept": kept,
            "segments_evicted": seg_evicted,
            "segment_token_counts": [len(tok(s, add_special_tokens=True)["input_ids"])
                                     for s in segments],
            "evicted_segment_token_counts": dropped_tokens,
        }
        segments_total += st
        dist[st] = dist.get(st, 0) + 1
        n_over += st > 1
        n_evicted += seg_evicted > 0
        evicted_segments += seg_evicted
        if (i + 1) % 2000 == 0:
            print(f"[seg] {i+1}/{len(episodes)} done", flush=True)

    stats = {
        "contract": "CONTRACT_C3_ICAE_FULL_RAG.md",
        "split": args.split,
        "tokenizer": "models/Mistral-7B-Instruct-v0.2",
        "encoder_max_tokens": ENCODER_MAX_TOKENS,
        "slot_budget_segments": SLOT_BUDGET_SEGMENTS,
        "episodes_total": len(episodes),
        "episodes_over_limit": n_over,
        "episodes_evicted": n_evicted,
        "eviction_rate": round(n_evicted / len(episodes), 6),
        "segments_total": segments_total,
        "segments_distribution": {str(k): v for k, v in sorted(dist.items())},
        "evicted_segments_total": evicted_segments,
        "oversize_events": oversize_all,
        "elapsed_seconds": round(time.time() - t0, 1),
        "episodes": episodes_out,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    STATS_PATH.write_text(json.dumps(stats, indent=1))
    print(f"[seg] wrote {STATS_PATH}")
    print(f"[seg] episodes={len(episodes)} over_limit={n_over} evicted={n_evicted} "
          f"eviction_rate={stats['eviction_rate']} segments_total={segments_total} "
          f"oversize_events={len(oversize_all)}")


if __name__ == "__main__":
    sys.exit(main())
