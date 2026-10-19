"""C1 end-to-end runner (CONTRACT_C1_HYBRID_RAG.md).

Pipeline per query (no oracle: retrieval sees only the masked question):
  question -> BM25 top-50 + BGE dense top-50 -> RRF k=60 -> top-10
           -> sort by time asc (latest last) -> prompt -> mistral-7b (greedy)
           -> EM/F1 via vcrm/metrics.py -> JSONL row.

Usage:
  python runner.py --split val --limit 481 --query-ids runs/refine_v2_suite/validation_481.json
  python runner.py --split test                # full 34,963
Resumable: skips query_ids already present in the output JSONL.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from corpus import build_corpus, load_corpus          # noqa: E402
from inject import build_facts, build_prompt, chat_wrap  # noqa: E402
from reader import read_answer                         # noqa: E402
from retrieval import RetrievalIndex, build_query_text     # noqa: E402
from scoring import aggregate, extract_answer, score_query  # noqa: E402

PROTOCOL = "csr_v1"  # Controlled State Routing rev1


def load_queries(split):
    eps = json.loads((ROOT / f"data/prepared/templama_{split}.json").read_text())
    qs = {}
    ep_of = {}
    for ep in eps:
        for q in ep["queries"]:
            qs[q["id"]] = q
            ep_of[q["id"]] = ep["id"]
    return qs, ep_of


def conflict_flags(qs, ep_of):
    """Metadata definition (unified with C2/C4): conflicting = the same
    state_key has >= 2 distinct answer sets across its queries."""
    answers_by_key = {}
    for q in qs.values():
        key = q["state_key"]
        sig = tuple(sorted(q["answers"]))
        answers_by_key.setdefault(key, set()).add(sig)
    multi = {k for k, sigs in answers_by_key.items() if len(sigs) >= 2}
    return {qid: q["state_key"] in multi for qid, q in qs.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--query-ids", default=None, help="JSON file with query_ids list (481 diag)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--log-every", type=int, default=25)
    args = ap.parse_args()

    tag = args.tag or f"c1_{args.split}" + (f"_{args.limit or 'diag'}" if args.limit or args.query_ids else "")
    out_dir = HERE / "runs_csr_v1"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{tag}.jsonl"
    summary_path = out_dir / f"{tag}_summary.json"

    # --- corpus (global, per split; 481 diag uses val corpus) ---
    corpus_path = out_dir / f"corpus_{args.split}.json"
    if corpus_path.exists():
        events = load_corpus(corpus_path)
        print(f"[corpus] loaded {len(events)} events from {corpus_path}")
        write_seconds = None
    else:
        events, write_seconds, sha = build_corpus(ROOT / f"data/prepared/templama_{args.split}.json", corpus_path)
        print(f"[corpus] built {len(events)} events, sha256 {sha[:16]}, write {write_seconds:.2f}s")
    events_by_id = {e["event_id"]: e for e in events}

    # --- queries ---
    qs, ep_of = load_queries(args.split)
    if args.query_ids:
        diag = json.loads(Path(args.query_ids).read_text())
        ids = list(diag["query_ids"])
    else:
        ids = list(qs)
    if args.limit:
        ids = ids[: args.limit]

    # resume (csr_v1: only rows carrying protocol==csr_v1 count as done;
    # any existing row with a foreign/missing protocol refuses resume)
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("protocol") != PROTOCOL:
                    raise SystemExit(
                        f"[resume] refusing to resume: {out_path} contains "
                        f"rows with protocol={r.get('protocol')!r} "
                        f"(expected {PROTOCOL!r})")
                done.add(r["query_id"])
    todo = [i for i in ids if i not in done]
    print(f"[queries] {len(ids)} selected, {len(done)} done, {len(todo)} todo -> {out_path}")

    if not todo:
        rows = [json.loads(l) for l in out_path.read_text().splitlines() if l.strip()]
        finish_summary(rows, qs, ep_of, summary_path, args, write_seconds, len(events),
                       encode_seconds=None)
        return

    # --- index (GPU 9 enforced outside via CUDA_VISIBLE_DEVICES) ---
    emb_cache = out_dir / f"emb_{args.split}_{args.device}.npy"
    t_enc = time.time()
    index = RetrievalIndex(events, device=args.device,
                           bge_path=str(ROOT / "models/bge-base-en-v1.5"),
                           cache_path=str(emb_cache), batch_size=args.batch_size)
    encode_seconds = time.time() - t_enc
    print(f"[index] encode device={args.device} batch={args.batch_size} {encode_seconds:.1f}s", flush=True)

    # tokenizer only for chat template (CPU, no model weights loaded via vllm path:
    # we need the template; load from the Mistral dir with transformers AutoTokenizer)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(ROOT / "models/Mistral-7B-Instruct-v0.2"))

    flags = conflict_flags(qs, ep_of)
    t_start = time.time()
    with out_path.open("a") as f:
        for n, qid in enumerate(todo):
            q = qs[qid]
            query_text = build_query_text(q)
            retrieved = index.retrieve(query_text, topk=10,
                                       episode_id=ep_of[qid])
            fact_texts, hits = build_facts(events_by_id, retrieved)
            prompt = chat_wrap(build_prompt(query_text, fact_texts), tok)
            answer, prompt_tokens, read_seconds = read_answer(prompt)
            extracted = extract_answer(answer)
            s = score_query(answer, q["answers"], extracted=extracted)
            # wrong_version: prediction matches a non-as-of version's answer
            # (resolved post-hoc in analysis; placeholder kept for schema parity)
            row = {
                "query_id": qid, "episode_id": ep_of[qid],
                "protocol": PROTOCOL,
                "query_text": query_text,
                "question": q["question"], "type": q["type"], "latest": q["latest"],
                "as_of": q["as_of"], "answers": q["answers"], "conflicting": flags[qid],
                "retrieved_event_ids": [r["event_id"] for r in retrieved],
                "fact_texts": fact_texts, "prompt_tokens": prompt_tokens,
                "raw_answer": answer, "extracted_answer": extracted,
                "em_raw": s["em_raw"], "f1_raw": s["f1_raw"],
                "em_extracted": s["em_extracted"], "f1_extracted": s["f1_extracted"],
                "read_seconds": round(read_seconds, 3),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if (n + 1) % args.log_every == 0 or n + 1 == len(todo):
                el = time.time() - t_start
                eta = el / (n + 1) * (len(todo) - n - 1)
                print(f"[run] {n+1}/{len(todo)} elapsed {el:.0f}s eta {eta/60:.1f}min "
                       f"last_em={s['em_extracted']} last_f1={s['f1_extracted']:.2f}", flush=True)

    rows = [json.loads(l) for l in out_path.read_text().splitlines() if l.strip()]
    finish_summary(rows, qs, ep_of, summary_path, args, write_seconds, len(events),
                   encode_seconds=encode_seconds)


def finish_summary(rows, qs, ep_of, summary_path, args, write_seconds, n_events,
                   encode_seconds=None):
    got = {r["query_id"] for r in rows}
    summary = aggregate(rows)
    corpus_bytes = None
    cp = HERE / "runs_csr_v1" / f"corpus_{args.split}.json"
    if cp.exists():
        corpus_bytes = cp.stat().st_size
    summary.update({
        "tag": args.tag or f"c1_{args.split}",
        "protocol": PROTOCOL,
        "n_events": n_events,
        "persistent_mb": round((corpus_bytes or 0) / 1e6, 2),
        "write_seconds": write_seconds,
        "encode_device": args.device,
        "encode_batch_size": args.batch_size,
        "encode_seconds": encode_seconds,
        "complete": None,  # set by caller context below
    })
    # completeness against this run's selected ids is checked by check_completeness()
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[summary] {summary_path}")
    print(json.dumps({k: summary[k] for k in
                      ("n_total", "hist_n", "hist_f1", "latest_n", "latest_em",
                       "conflict_n", "conflict_f1", "mean_prompt_tokens", "mean_read_seconds")},
                     indent=2))


def check_completeness(tag, split, query_ids_file=None, limit=None):
    """Contract §5: run invalid unless all selected queries present."""
    out_path = HERE / "runs_csr_v1" / f"{tag}.jsonl"
    rows = [json.loads(l) for l in out_path.read_text().splitlines() if l.strip()]
    got = {r["query_id"] for r in rows}
    if query_ids_file:
        ids = set(json.loads(Path(query_ids_file).read_text())["query_ids"])
    else:
        eps = json.loads((ROOT / f"data/prepared/templama_{split}.json").read_text())
        ids = {q["id"] for ep in eps for q in ep["queries"]}
        if limit:
            ids = set(list(ids)[:limit])
    missing = ids - got
    return {"complete": not missing, "n_expected": len(ids), "n_got": len(got),
            "n_missing": len(missing), "missing": sorted(missing)[:20]}


if __name__ == "__main__":
    main()
