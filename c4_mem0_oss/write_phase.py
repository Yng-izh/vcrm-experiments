#!/usr/bin/env python3
"""Phase A: build Mem0 memory store from TempLAMA test episodes (contract C4 §2/§3).

Writes only {id, time, text} event views — zero oracle contact
(answers/as_of/latest/state_key never touched on the write path).
Cost bookkeeping per contract §3 (USD 200 cap, 80% alert, extrapolation check).
Run with latent2 python, PYTHONPATH=c4_mem0_oss/venv_target.
"""
import argparse
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE / "venv_target"))

PIN_SHA = "94c3fe9f238f3dbf29c9ce98643bd71eb13077cd"
WRITER_MODEL = "gpt-4o-mini"            # contract §9 rev-5: gpt-4o-mini @ ccode relay (OpenAI-compatible)
WRITER_BASE_URL = "https://api.ccode.vip/v1"
# gpt-4o-mini pricing (USD per 1M tokens, official list; relay may differ — summary uses ledger actuals)
COST_PER_1M_IN = 0.15
COST_PER_1M_OUT = 0.60
BUDGET_USD = 200.0
ALERT_FRAC = 0.5    # §9 rev-7: relay-endpoint stability unknown, early-warning at 50% per captain
INGESTION_FIELDS = ("id", "time", "text")  # event whitelist (contract §6)

# oracle audit (contract §6): the eval-side field names {answers, as_of, latest,
# state_key} must not appear as string constants or attribute accesses in this
# module — the write path sees only INGESTION_FIELDS. AST audit enforces this.


def ingestion_view(ep):
    """Episode -> flat event list, whitelist fields only."""
    out = []
    for ev in ep["events"]:
        out.append({k: ev[k] for k in INGESTION_FIELDS if k in ev})
    return out


def estimate_cost(ptoks, ctoks):
    return (ptoks * COST_PER_1M_IN + ctoks * COST_PER_1M_OUT) / 1e6



def estimate_tokens(prompt_text, model):
    """Fallback token estimate when mem0 add() returns no usage (6B-v3).
    Uses tiktoken o200k_base (gpt-4o family tokenizer); ±10% vs real counts,
    acceptable for a USD-budget guard (never silently zero)."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("o200k_base")
        return len(enc.encode(prompt_text)), 60  # ~60 out tokens per fact write
    except Exception:
        return max(1, len(prompt_text) // 4), 60  # ~4 chars/token last resort


# Relay-endpoint resilience (503/429/5xx observed to kill whole shards):
# exponential backoff with jitter; after MAX_RETRIES consecutive failures the
# episode is marked failed in errors.jsonl and the process MOVES ON — a single
# bad ep must never cost the remaining ~1000.
RETRY_STATUS_HINTS = ("503", "429", "500", "502", "504", "overloaded",
                      "cpu overloaded", "rate limit", "timeout", "timed out")
MAX_RETRIES = 5
BACKOFF_BASE = 30.0  # seconds: 30/60/120/240/480 + jitter


def _transient(exc):
    s = str(exc).lower()
    return any(h in s for h in RETRY_STATUS_HINTS)


def add_with_retry(memory, text, user_id, metadata):
    """memory.add with exponential backoff on transient relay errors.
    Returns (result, attempts) or raises RuntimeError after MAX_RETRIES."""
    last = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            res = memory.add(text, user_id=user_id, metadata=metadata)
            return res, attempt
        except Exception as exc:
            last = exc
            if attempt == MAX_RETRIES or not _transient(exc):
                raise
            wait = BACKOFF_BASE * (2 ** (attempt - 1)) + random.uniform(0, 10)
            print(f"[retry] transient error attempt {attempt}/{MAX_RETRIES} "
                  f"({type(exc).__name__}: {str(exc)[:120]}) — sleep {wait:.0f}s",
                  flush=True)
            time.sleep(wait)
    raise RuntimeError(f"unreachable: {last}")


def error_path(tag):
    return HERE / "runs" / f"{tag}_errors.jsonl"


def log_error(tag, ep_id, event_id, exc):
    with error_path(tag).open("a") as f:
        f.write(json.dumps({"episode_id": ep_id, "event_id": event_id,
                            "error": f"{type(exc).__name__}: {exc}",
                            "ts": time.time()}) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit-episodes", type=int, default=None)
    ap.add_argument("--episodes-from", default=None,
                    help="file with episode ids (one per line) to restrict writing to")
    ap.add_argument("--tag", default="write")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    out_dir = HERE / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = out_dir / f"{args.tag}_cost_ledger.jsonl"
    # Event-level resume: a ledger row is written per ingested event, so the
    # done-set is the set of (episode_id, event_id) pairs actually written.
    # An episode is todo if ANY of its events is missing; on rerun only the
    # missing events are re-written (no whole-episode redo).
    done_pairs = set()
    if args.resume and ledger_path.exists():
        for l in ledger_path.open():
            if not l.strip():
                continue
            rec = json.loads(l)
            if rec.get("kind") == "episode_complete":
                continue
            if rec.get("event_id") is not None:
                done_pairs.add((rec["episode_id"], rec["event_id"]))
    done_eps = {ep_id for ep_id, _ in done_pairs}

    eps = json.loads((ROOT / f"data/prepared/templama_{args.split}.json").read_text())
    if args.episodes_from:
        want = {l.strip() for l in open(args.episodes_from) if l.strip()}
        eps = [ep for ep in eps if ep["id"] in want]
    if args.limit_episodes:
        eps = eps[: args.limit_episodes]
    todo = []
    for ep in eps:
        need = [ev for ev in ingestion_view(ep) if (ep["id"], ev["id"]) not in done_pairs]
        if need:
            todo.append((ep, need))
    n_events_done = len(done_pairs)
    print(f"[write] episodes={len(eps)} done_eps={len(done_eps)} "
          f"events_done={n_events_done} todo_eps={len(todo)}", flush=True)

    from mem0 import Memory
    # Parallel-write isolation: mem0 builds migrations_qdrant (a locked Qdrant
    # store) under MEM0_DIR (default ~/.mem0). Per-process MEM0_DIR + a
    # per-process history.db key prevent the global migration lock contention
    # that killed shards 1-3 at startup.
    config = {
        "history_db_path": str(HERE / (os.environ.get(
            "QDRANT_PATH", "qdrant_data").replace("qdrant_data", "history_", 1) + ".db")),
        "llm": {
            "provider": "openai",
            "config": {"model": WRITER_MODEL, "temperature": 0.0,
                       "openai_base_url": WRITER_BASE_URL,
                       "api_key": os.environ["DEEPSEEK_API_KEY"],  # env-injected, never written
                       },
        },
        "embedder": {
            "provider": "huggingface",
            "config": {"model": str(ROOT / "models/bge-base-en-v1.5"),
                       "model_kwargs": {"device": "cpu"}},
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                # 4-process parallel write: QDRANT_PATH env isolates each
                # shard's store (qdrant local mode holds an exclusive dir lock;
                # merged afterwards by merge_qdrant_shards.py).
                "path": str(HERE / os.environ.get("QDRANT_PATH", "qdrant_data")),
                "collection_name": "c4_mem0_oss",
                "embedding_model_dims": 768,  # BGE dim; QdrantConfig default 1536 (6B-v3)
            },
        },
        "version": "v1.1",
    }
    memory = Memory.from_config(config)

    total_in = total_out = 0
    cost_usd = 0.0
    n_ok = n_failed_eps = 0
    t0 = time.time()
    ledger = ledger_path.open("a")
    n = -1
    for n, (ep, events) in enumerate(todo):
        try:
            # one add() per MISSING event text only (event-level resume);
            # mem0 dedupes/consolidates internally
            for ev in events:
                prompt_text = f"[{ev['time']}] {ev['text']}"
                res, _attempts = add_with_retry(
                    memory, prompt_text,
                    user_id=ep["id"],
                    metadata={"time": ev["time"], "event_id": ev["id"]},
                )
                # mem0 returns {'results': [...], 'relations': [...]} — token usage is
                # tracked via the llm wrapper; fall back to response metadata if present
                # 6B-v3: mem0 add() response carries NO usage key. Real usage is
                # recorded only if mem0 surfaces it; otherwise estimate with the
                # writer tokenizer and tag provenance "estimated" so the ledger
                # is never silently zero (USD 200 guard depends on it).
                usage = res.get("usage") if isinstance(res, dict) else None
                if usage:
                    pt, ct = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
                    provenance = "reported"
                else:
                    pt, ct = estimate_tokens(prompt_text, WRITER_MODEL)
                    provenance = "estimated"
                total_in += pt
                total_out += ct
                cost_usd += estimate_cost(pt, ct)
                ledger.write(json.dumps({
                    "episode_id": ep["id"], "event_id": ev["id"],
                    "prompt_tokens": pt, "completion_tokens": ct,
                    "cost_usd": round(estimate_cost(pt, ct), 8), "usage_provenance": provenance,
                    "writer_model": WRITER_MODEL, "pin_sha": PIN_SHA,
                }) + "\n")
            # all events of this episode present in the ledger now -> one
            # episode_complete marker row (kind field distinguishes it from
            # per-event cost rows; carries no cost).
            ledger.write(json.dumps({
                "kind": "episode_complete",
                "episode_id": ep["id"], "n_events": len(events),
                "ts": time.time(),
            }) + "\n")
            ledger.flush()
            n_ok += 1
        except Exception as exc:
            # event-level resume semantics: failed events are not in the
            # ledger, so a rerun redoes only the missing events
            n_failed_eps += 1
            log_error(args.tag, ep["id"], None, exc)
            traceback.print_exc()
            print(f"[error] episode {ep['id']} failed after retries "
                  f"({n_failed_eps} failed so far) — continuing", flush=True)
        if cost_usd > BUDGET_USD * ALERT_FRAC:
            print(f"[COST-ALERT] {cost_usd:.2f} USD > 50% of {BUDGET_USD} — stop and report captain",
                  flush=True)
            break
        if (n + 1) % 20 == 0 or n + 1 == len(todo):
            el = time.time() - t0
            eta = el / (n + 1) * (len(todo) - n - 1)
            print(f"[write] {n+1}/{len(todo)} ok={n_ok} failed={n_failed_eps} "
                  f"elapsed {el:.0f}s eta {eta/60:.1f}min "
                  f"in={total_in} out={total_out} cost=${cost_usd:.4f}", flush=True)
    ledger.close()

    # extrapolation check (contract §3)
    n_done = len(done_eps) + min(n + 1, len(todo))
    if n_done:
        per_ep = cost_usd / max(1, n + 1)
        full_est = per_ep * 4057
        print(f"[write] done {n_done} eps, cost so far ${cost_usd:.4f}, "
              f"per-episode ${per_ep:.6f}, FULL-RUN EXTRAPOLATION ${full_est:.2f} "
              f"(cap {BUDGET_USD}, {'OVER' if full_est > BUDGET_USD else 'within'})", flush=True)
        (out_dir / f"{args.tag}_cost_summary.json").write_text(json.dumps({
            "pin_sha": PIN_SHA, "writer_model": WRITER_MODEL,
            "episodes_done": n_done, "prompt_tokens": total_in,
            "completion_tokens": total_out, "cost_usd": cost_usd,
            "per_episode_usd": per_ep, "full_run_extrapolation_usd": full_est,
            "budget_usd": BUDGET_USD,
        }, indent=2))


if __name__ == "__main__":
    main()
