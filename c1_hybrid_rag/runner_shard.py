"""Sharded concurrent C1 test runner (speedup; frozen scoring/contract intact).

Speedup design (captain-approved 2026-10-17):
- shard = int(query_id) % n_shards; each shard served by one vllm instance
  (8001 = shard 0, 8002-8005 = shards 1-4 on GPUs 2/3/4/5).
- retrieval (BM25+BGE encode cached on GPU 9) stays single-process; prompts are
  built centrally, reader calls fan out concurrently (asyncio + aiohttp, 16
  per instance).
- Resumable at row level: reads ALL shard files c1_test.shard{k}.jsonl, skips
  done query_ids. Merged c1_test.jsonl produced at completion (dedup by
  query_id; completeness 34,963 checked before summary).
- Frozen semantics unchanged: greedy/max 128/seed 42 per request, ex28
  extraction, four score columns, time-asc injection.
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from corpus import load_corpus                       # noqa: E402
from inject import build_facts, build_prompt, chat_wrap  # noqa: E402
from reader_async import read_answers_async          # noqa: E402
from retrieval import RetrievalIndex                 # noqa: E402
from scoring import aggregate, extract_answer, score_query  # noqa: E402


def load_queries(split):
    eps = json.loads((ROOT / f"data/prepared/templama_{split}.json").read_text())
    qs = {}
    ep_of = {}
    for ep in eps:
        for q in ep["queries"]:
            qs[q["id"]] = q
            ep_of[q["id"]] = ep["id"]
    return qs, ep_of


N_SHARDS = [6]


def shard_of(qid):
    """Stable shard assignment: hash-based (query ids are alphanumeric)."""
    return hash(qid) % N_SHARDS[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--tag", default="c1_test")
    ap.add_argument("--n-shards", type=int, default=6)
    ap.add_argument("--base-port", type=int, default=8001,
                    help="shard k uses port base_port+perm(k); see --ports")
    ap.add_argument("--ports", default="8001,8006,8007,8008,8004,8005",
                    help="comma list of vllm ports, one per shard (len == n-shards)")
    ap.add_argument("--concurrency", type=int, default=16,
                    help="in-flight reader requests per vllm instance")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--chunk", type=int, default=64,
                    help="queries per retrieval/reader batch cycle")
    ap.add_argument("--log-every", type=int, default=10, help="cycles")
    args = ap.parse_args()

    out_dir = HERE / "runs"
    out_dir.mkdir(exist_ok=True)
    shard_paths = [out_dir / f"{args.tag}.shard{k}.jsonl" for k in range(args.n_shards)]
    merged_path = out_dir / f"{args.tag}.jsonl"
    summary_path = out_dir / f"{args.tag}_summary.json"

    # corpus (cached)
    corpus_path = out_dir / f"corpus_{args.split}.json"
    events = load_corpus(corpus_path)
    events_by_id = {e["event_id"]: e for e in events}
    print(f"[corpus] loaded {len(events)} events", flush=True)

    qs, ep_of = load_queries(args.split)
    ids = list(qs)

    # resume: union of done ids across shards + legacy merged file
    done = set()
    for p in shard_paths + [merged_path]:
        if p.exists():
            for line in p.read_text().splitlines():
                if line.strip():
                    done.add(json.loads(line)["query_id"])
    todo = [i for i in ids if i not in done]
    print(f"[queries] {len(ids)} selected, {len(done)} done, {len(todo)} todo "
          f"(shards={args.n_shards}, concurrency={args.concurrency})", flush=True)

    N_SHARDS[0] = args.n_shards
    shard_ports = [int(x) for x in args.ports.split(",")]
    assert len(shard_ports) == args.n_shards, "ports length must equal n-shards"
    shard_urls = [f"http://localhost:{p}/v1/completions" for p in shard_ports]

    # index (GPU via CUDA_VISIBLE_DEVICES outside)
    emb_cache = out_dir / f"emb_{args.split}_{args.device}.npy"
    t_enc = time.time()
    index = RetrievalIndex(events, device=args.device,
                           bge_path=str(ROOT / "models/bge-base-en-v1.5"),
                           cache_path=str(emb_cache), batch_size=args.batch_size)
    print(f"[index] encode device={args.device} batch={args.batch_size} "
          f"{time.time() - t_enc:.1f}s", flush=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(ROOT / "models/Mistral-7B-Instruct-v0.2"))

    files = [p.open("a") for p in shard_paths]
    n_done_cycle = 0
    t_start = time.time()
    try:
        for cyc_start in range(0, len(todo), args.chunk):
            batch_ids = todo[cyc_start:cyc_start + args.chunk]
            # retrieval + prompt build (CPU/GPU-lite, central)
            prompts, metas = [], []
            for qid in batch_ids:
                q = qs[qid]
                retrieved = index.retrieve(q["question"], topk=10)
                fact_texts, _ = build_facts(events_by_id, retrieved)
                prompts.append(chat_wrap(build_prompt(q["question"], fact_texts), tok))
                metas.append((qid, q, [r["event_id"] for r in retrieved], fact_texts))
            # fan out: each query goes to its shard's endpoint
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            async def fan():
                sems = {u: asyncio.Semaphore(args.concurrency) for u in shard_urls}

                async def one(u, p):
                    from reader_async import read_answer_async
                    return await read_answer_async(session, u, p, sems[u])
                import aiohttp
                async with aiohttp.ClientSession(
                        connector=aiohttp.TCPConnector(limit=args.concurrency * args.n_shards)) as session:
                    tasks = [one(shard_urls[shard_of(qid)], p)
                             for (qid, _, _, _), p in zip(metas, prompts)]
                    return await asyncio.gather(*tasks)
            results = loop.run_until_complete(fan())
            loop.close()

            for (qid, q, ev_ids, fact_texts), (answer, ptoks) in zip(metas, results):
                extracted = extract_answer(answer)
                s = score_query(answer, q["answers"], extracted=extracted)
                row = {
                    "query_id": qid, "episode_id": ep_of[qid],
                    "question": q["question"], "type": q["type"], "latest": q["latest"],
                    "as_of": q["as_of"], "answers": q["answers"],
                    "conflicting": True,  # placeholder; recomputed in summary
                    "retrieved_event_ids": ev_ids, "fact_texts": fact_texts,
                    "prompt_tokens": ptoks,
                    "raw_answer": answer, "extracted_answer": extracted,
                    "em_raw": s["em_raw"], "f1_raw": s["f1_raw"],
                    "em_extracted": s["em_extracted"], "f1_extracted": s["f1_extracted"],
                    "read_seconds": None,
                }
                files[shard_of(qid)].write(json.dumps(row, ensure_ascii=False) + "\n")
            for f in files:
                f.flush()
            n_done_cycle += len(batch_ids)
            cyc = cyc_start // args.chunk + 1
            if cyc % args.log_every == 0 or cyc_start + args.chunk >= len(todo):
                el = time.time() - t_start
                eta = el / n_done_cycle * (len(todo) - n_done_cycle)
                last_ex = results[-1][0][:40]
                print(f"[run] {n_done_cycle}/{len(todo)} elapsed {el:.0f}s eta {eta/60:.1f}min "
                      f"last={last_ex!r}", flush=True)
    finally:
        for f in files:
            f.close()

    # merge (dedup by query_id); legacy serial-run file included first
    seen, merged = set(), []
    merge_sources = [merged_path] + shard_paths
    for p in merge_sources:
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["query_id"] not in seen:
                seen.add(r["query_id"])
                merged.append(r)
    merged_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in merged))
    missing = set(ids) - seen
    print(f"[merge] {len(merged)} unique rows, missing {len(missing)}", flush=True)

    if not missing:
        summary = aggregate(merged)
        summary.update({
            "tag": args.tag, "n_events": len(events),
            "runner": "sharded", "n_shards": args.n_shards,
            "reader_concurrency_per_instance": args.concurrency,
            "shard_urls": shard_urls,
            "speedup_note": ("concurrent async reader (aiohttp) + multi-way "
                             "shard by query_id mod n_shards; per-request "
                             "frozen semantics unchanged (greedy/max128/"
                             "seed42); scoring frozen (ex28, four columns)"),
        })
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"[summary] written {summary_path}", flush=True)


if __name__ == "__main__":
    main()
