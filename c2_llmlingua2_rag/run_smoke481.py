"""C2-t2: 481 smoke on LLMLingua-2-compressed val corpus.

Contract C2 §0: bitwise reuse of C1 pipeline; ONLY the event text is the
compressed version. Corpus/events built from corpus_val_compressed.json
(text = compressed, text_original kept; metadata intact), then C1
RetrievalIndex / inject / reader / scoring run unchanged.
"""
import json
import sys
import time
from pathlib import Path

HERE = Path("/share/latent2/vcrm_icae/c1_hybrid_rag")
ROOT = Path("/share/latent2/vcrm_icae")
C2 = ROOT / "c2_llmlingua2_rag"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from corpus import load_corpus  # noqa: E402
from inject import build_facts, build_prompt, chat_wrap  # noqa: E402
from reader import read_answer  # noqa: E402
from retrieval import RetrievalIndex  # noqa: E402
from scoring import aggregate, extract_answer, score_query  # noqa: E402

SPLIT = "val"
QUERY_IDS = json.loads(
    (ROOT / "runs/refine_v2_suite/validation_481.json").read_text())["query_ids"]
OUT = C2 / "runs/c2_val481.jsonl"
SUMMARY = C2 / "runs/c2_val481_summary.json"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    # --- compressed corpus -> C1 event schema (text = compressed) ---
    doc = json.loads((C2 / "corpus_val_compressed.json").read_text())
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
    print(f"[corpus] {len(events)} compressed events, rate={compression_rate}")

    # --- queries (frozen 481 diag set) ---
    eps = json.loads((ROOT / f"data/prepared/templama_{SPLIT}.json").read_text())
    qs, ep_of = {}, {}
    for ep in eps:
        for q in ep["queries"]:
            qs[q["id"]] = q
            ep_of[q["id"]] = ep["id"]
    # metadata conflict definition (main table): answers differ across versions
    ans_by_key = {}
    for q in qs.values():
        ans_by_key.setdefault(q["state_key"], set()).add(
            json.dumps(q["answers"], sort_keys=True))
    conflicting_meta = {
        qid: len(ans_by_key[q["state_key"]]) > 1 for qid in qs}

    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["query_id"])
    todo = [i for i in QUERY_IDS if i not in done]
    print(f"[queries] {len(QUERY_IDS)} selected, {len(done)} done, {len(todo)} todo")

    # --- index over compressed text (C1 module unchanged) ---
    cache = C2 / f"emb_val_compressed_{args.device}.npy"
    t_enc = time.time()
    index = RetrievalIndex(events, device=args.device,
                           bge_path=str(ROOT / "models/bge-base-en-v1.5"),
                           cache_path=str(cache), batch_size=args.batch_size)
    encode_seconds = time.time() - t_enc
    print(f"[index] encode {encode_seconds:.1f}s")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(ROOT / "models/Mistral-7B-Instruct-v0.2"))

    t0 = time.time()
    with OUT.open("a") as f:
        for n, qid in enumerate(todo):
            q = qs[qid]
            retrieved = index.retrieve(q["question"], topk=10)
            fact_texts, hits = build_facts(events_by_id, retrieved)
            prompt = chat_wrap(build_prompt(q["question"], fact_texts), tok)
            answer, prompt_tokens, read_seconds = read_answer(prompt)
            extracted = extract_answer(answer)
            s = score_query(answer, q["answers"], extracted=extracted)
            row = {
                "query_id": qid, "episode_id": ep_of[qid],
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
                "read_seconds": round(read_seconds, 3),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if (n + 1) % 50 == 0 or n + 1 == len(todo):
                el = time.time() - t0
                print(f"[run] {n+1}/{len(todo)} elapsed {el:.0f}s "
                      f"eta {el/(n+1)*(len(todo)-n-1)/60:.1f}min", flush=True)

    rows = [json.loads(l) for l in OUT.read_text().splitlines() if l.strip()]
    summary = aggregate(rows)
    c1 = json.loads((HERE / "runs/c1_val481_summary.json").read_text())
    # per-query diff vs C1 481 (same query ids, same frozen metrics)
    c1_rows = {}
    for line in (HERE / "runs/c1_val481.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            c1_rows[r["query_id"]] = r
    diff_f1 = [r["f1_extracted"] - c1_rows[r["query_id"]]["f1_extracted"]
               for r in rows if r["query_id"] in c1_rows]
    summary.update({
        "tag": "c2_val481",
        "compression_rate": compression_rate,
        "n_events": len(events),
        "corpus_bytes_compressed": (C2 / "corpus_val_compressed.json").stat().st_size,
        "corpus_bytes_original": (HERE / "runs/corpus_val.json").stat().st_size,
        "encode_device": args.device,
        "encode_seconds": encode_seconds,
        "mean_read_seconds_note": "serial smoke; per-row timing measured",
        "c1_comparison": {
            "c1_hist_em": c1.get("hist_em"),
            "c1_hist_f1": c1.get("hist_f1"),
            "c2_hist_em": summary.get("hist_em"),
            "c2_hist_f1": summary.get("hist_f1"),
            "n_paired": len(diff_f1),
            "mean_f1_delta_c2_minus_c1": round(sum(diff_f1) / len(diff_f1), 6),
            "n_improved": sum(d > 0 for d in diff_f1),
            "n_degraded": sum(d < 0 for d in diff_f1),
            "note": "record-only, no tuning (contract C2 §3.2)",
        },
        "read_latency_note": "serial 481 smoke: per-row read_seconds measured; "
                             "full-test shard runs carry C1's async caveat",
    })
    SUMMARY.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in
                      ("n_total", "hist_em", "hist_f1", "conflict_n",
                       "conflict_f1", "mean_prompt_tokens")}, indent=2))


if __name__ == "__main__":
    main()
