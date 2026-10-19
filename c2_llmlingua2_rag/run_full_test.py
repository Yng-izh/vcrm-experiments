"""C2-t3: full-test run (34,963 queries) on compressed test corpus.

C1 pipeline bitwise-reused; only event text is compressed. Three vllm
instances (GPU 9/0/4, ports 8001/8000/8004 — user ruling after host
reaper killed port-8001 instance twice), shard by hash(query_id) % 3,
16 concurrent requests per shard (ThreadPoolExecutor).
Row-level resumable; progress logged every 500 rows; async latency is NOT
per-request wall time — read_seconds recorded as null, wall time in summary.
"""
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path("/share/latent2/vcrm_icae/c1_hybrid_rag")
ROOT = Path("/share/latent2/vcrm_icae")
C2 = ROOT / "c2_llmlingua2_rag"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from corpus import load_corpus  # noqa: E402
from inject import build_facts, build_prompt, chat_wrap  # noqa: E402
from reader import read_answer  # noqa: E402  (URL overridden below)
from retrieval import RetrievalIndex, build_query_text  # noqa: E402
from scoring import aggregate, extract_answer, score_query  # noqa: E402

SPLIT = "test"
PROTOCOL = "csr_v1"
RUNS = C2 / "runs_csr_v1"
OUT = RUNS / "c2_test.jsonl"
SUMMARY = RUNS / "c2_test_summary.json"
N_SHARDS = int(os.environ.get("C2_SHARDS", "3"))
CONCURRENCY = int(os.environ.get("C2_CONCURRENCY", "16"))
PORTS = {0: 8001, 1: 8000, 2: 8004}  # GPU 9 / 0 / 4

_readers_lock = threading.Lock()
_readers = {}


def _reader_for(shard):
    """A reader module instance per shard with its own port (C1 reader logic,
    per-instance URL; C1 module file itself is NOT modified)."""
    with _readers_lock:
        if shard not in _readers:
            import importlib
            import urllib.request
            m = importlib.import_module("reader")
            m = importlib.reload(m)  # fresh module object per shard
            m.VLLM_URL = f"http://localhost:{PORTS[shard]}/v1/completions"

            def read_answer(prompt, max_retries=6, timeout=600,
                            _m=m):
                return _read_one(_m, prompt, max_retries, timeout)
            _readers[shard] = read_answer
        return _readers[shard]


def _read_one(m, prompt, max_retries, timeout):
    import urllib.request
    payload = {"model": m.MODEL, "prompt": prompt, "temperature": 0.0,
               "max_tokens": 128, "seed": 42}
    last = None
    for attempt in range(max_retries):
        try:
            t0 = time.time()
            req = urllib.request.Request(
                m.VLLM_URL, data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                out = json.loads(r.read())
            return (out["choices"][0]["text"].strip(),
                    out["usage"]["prompt_tokens"], time.time() - t0)
        except Exception as e:
            last = e
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"vllm call failed after {max_retries} retries: {last}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--query-ids", default="",
                    help="JSON file {\"query_ids\": [...]} restricting todo to those qids")
    ap.add_argument("--tag", default="c2_test",
                    help="output tag; OUT/SUMMARY become runs_csr_v1/{tag}.jsonl")
    ap.add_argument("--no-index", action="store_true",
                    help="skip building index (all shards done; summary only)")
    args = ap.parse_args()

    global OUT, SUMMARY
    RUNS.mkdir(parents=True, exist_ok=True)
    OUT = RUNS / f"{args.tag}.jsonl"
    SUMMARY = RUNS / f"{args.tag}_summary.json"

    doc = json.loads((C2 / "corpus_test_compressed.json").read_text())
    compression_rate = doc["compression_rate"]
    events = []
    for ep_i, ep in enumerate(doc["episodes"]):
        for ev_i, ev in enumerate(ep["events"]):
            events.append({
                "event_id": f"e{ep_i:05d}_{ev_i:04d}",
                "orig_id": ev["id"],
                "episode_id": ep["id"],
                "time": str(ev["time"]).strip(),
                "text": ev["text"],           # compressed; metadata intact
            })
    events_by_id = {e["event_id"]: e for e in events}
    print(f"[corpus] {len(events)} compressed events, rate={compression_rate}",
          flush=True)

    eps = json.loads((ROOT / f"data/prepared/templama_{SPLIT}.json").read_text())
    qs, ep_of = {}, {}
    for ep in eps:
        for q in ep["queries"]:
            qs[q["id"]] = q
            ep_of[q["id"]] = ep["id"]
    all_ids = sorted(qs)
    ans_by_key = {}
    for q in qs.values():
        ans_by_key.setdefault(q["state_key"], set()).add(
            json.dumps(q["answers"], sort_keys=True))
    conflicting_meta = {
        qid: len(ans_by_key[q["state_key"]]) > 1 for qid in qs}
    latest_meta = {qid: q["latest"] for qid, q in qs.items()}
    print(f"[queries] {len(all_ids)} total", flush=True)

    done = set()
    stale = 0
    RUNS.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("protocol") != PROTOCOL:
                    stale += 1
                    continue
                done.add(r["query_id"])
    if stale:
        print(f"[resume] skipped {stale} rows with protocol != {PROTOCOL}",
              flush=True)
    if args.query_ids:
        allowed = set(json.loads(Path(args.query_ids).read_text())["query_ids"])
        todo = [i for i in all_ids if i in allowed]
    else:
        todo = [i for i in all_ids if i not in done]
    print(f"[queries] {len(done)} done, {len(todo)} todo", flush=True)
    if not todo:
        print("[run] nothing to do", flush=True)
        return

    t_all = time.time()
    # shards share one index (read-only retrieve is thread-safe: BM25 python
    # dict + numpy matmul; guarded by GIL, no mutation)
    index = RetrievalIndex(events, device=args.device,
                           bge_path=str(ROOT / "models/bge-base-en-v1.5"),
                           cache_path=str(C2 / f"emb_test_compressed_{args.device}.npy"),
                           batch_size=args.batch_size)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(ROOT / "models/Mistral-7B-Instruct-v0.2"))
    print(f"[index] ready {time.time()-t_all:.0f}s", flush=True)

    shard_todo = {s: [q for q in todo if int(hashlib.md5(q.encode()).hexdigest(), 16) % N_SHARDS == s]
                  for s in range(N_SHARDS)}
    print("[shards] " + " ".join(f"s{s}:{len(v)}" for s, v in shard_todo.items()),
          flush=True)

    write_lock = threading.Lock()
    out_f = OUT.open("a")
    counters = {s: 0 for s in range(N_SHARDS)}
    errors = []
    t0 = time.time()
    last_report = [t0]

    def run_shard(shard):
        read_answer = _reader_for(shard)
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
            futures = {}

            def work(qid):
                q = qs[qid]
                query_text = build_query_text(q)
                retrieved = index.retrieve(query_text, topk=10,
                                           episode_id=ep_of[qid])
                fact_texts, hits = build_facts(events_by_id, retrieved)
                prompt = chat_wrap(build_prompt(query_text, fact_texts), tok)
                answer, prompt_tokens, _ = read_answer(prompt)
                extracted = extract_answer(answer)
                s = score_query(answer, q["answers"], extracted=extracted)
                return {
                    "query_id": qid, "protocol": PROTOCOL,
                    "query_text": query_text,
                    "episode_id": ep_of[qid],
                    "question": q["question"], "type": q["type"],
                    "latest": q["latest"], "as_of": q["as_of"],
                    "answers": q["answers"],
                    "conflicting": conflicting_meta[qid],
                    "retrieved_event_ids": [r["event_id"] for r in retrieved],
                    "fact_texts": fact_texts, "prompt_tokens": prompt_tokens,
                    "raw_answer": answer, "extracted_answer": extracted,
                    "em_raw": s["em_raw"], "f1_raw": s["f1_raw"],
                    "em_extracted": s["em_extracted"],
                    "f1_extracted": s["f1_extracted"],
                    "read_seconds": None,  # async: not per-request measurable
                    "shard": shard,
                }

            for qid in shard_todo[shard]:
                futures[ex.submit(work, qid)] = qid
            for fut in futures:
                qid = futures[fut]
                try:
                    row = fut.result()
                except Exception as e:
                    errors.append((qid, repr(e)[:200]))
                    continue
                with write_lock:
                    out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out_f.flush()
                    counters[shard] += 1
                    done_n = sum(counters.values())
                    if done_n % 500 == 0:
                        el = time.time() - t0
                        thr = done_n / el
                        print(f"[run] {done_n}/{len(todo)} elapsed {el:.0f}s "
                              f"thr {thr:.1f} rows/s "
                              f"eta {el/done_n*(len(todo)-done_n)/60:.0f}min "
                              f"errors {len(errors)}", flush=True)
                        if el - (last_report[0] - t0) > 1800:
                            last_report[0] = t0 + el

    threads = [threading.Thread(target=run_shard, args=(s,))
               for s in range(N_SHARDS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    out_f.close()

    wall = time.time() - t_all
    print(f"[done] wall {wall:.0f}s errors {len(errors)}", flush=True)
    if errors:
        RUNS.mkdir(parents=True, exist_ok=True)
        (RUNS / "c2_test_errors.json").write_text(
            json.dumps(errors, indent=1))
        print(f"[ERRORS] {len(errors)} rows failed — rerun to resume", flush=True)
        return
    rows = [json.loads(l) for l in OUT.read_text().splitlines() if l.strip()]
    summary = aggregate(rows)
    summary.update({
        "tag": args.tag, "protocol": PROTOCOL,
        "compression_rate": compression_rate,
        "n_events": len(events),
        "wall_seconds": round(wall, 1),
        "n_shards": N_SHARDS,
        "concurrency_per_shard": CONCURRENCY,
        "mean_read_seconds_note": "not measured (async concurrency; "
                                  "per-request wall time is not attributable)",
        "latest_n_meta": sum(1 for r in rows if r["latest"]),
        "conflict_n_meta": sum(1 for r in rows if r["conflicting"]),
    })
    SUMMARY.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary.get(k) for k in
                      ("n_total", "hist_em", "hist_f1", "latest_em",
                       "conflict_f1", "mean_prompt_tokens")}, indent=2))


if __name__ == "__main__":
    main()
