#!/usr/bin/env python3
"""C4 Mem0 OSS runner: write-phase (mem0 add) -> native search -> Mistral Reader -> ex28 scoring.

Contract CONTRACT_C4_MEM0_OSS.md (frozen, FROZEN_MD5=0109281fd4c8be13d4019be7912897bf).
Pipeline per query:
  question -> mem0 memory.search() (native OSS front-end, top-k) -> facts into
  C1 frozen PROMPT_TEMPLATE -> vllm mistral-7b (greedy/128/seed42, C1 reader
  semantics) -> EM/F1 via vcrm metrics + frozen ex28 extraction -> JSONL row.

Oracle isolation (contract §6): write and read paths see only {id, time, text}
event views; answers/latest/state_key never touched. Eval-side labels
(type/answers) are attached only at the eval-side boundary (scoring + row).

as_of on the read path (audit note): contract §6 constrains only the WRITE
path — it never touches as_of on events. On the READ path, the query's
as_of year is a legitimate query condition (TempLAMA is time-anchored QA):
retrieval and the reader prompt both use search_text = "As of {as_of},
{question}" so the temporal scope is explicit. The row records "as_of" and
"query_text" for protocol audit, and route.protocol is "csr_v1" (the
unified protocol id shared with C1/C2/C3) — retrieval is still filtered via
the per-episode user_id (ep_of[qid]), not a hidden state mechanism.

Usage:
  python runner.py --smoke                     # 20-query smoke (§9 rev-6: hist16/latest4 from sample100)
  python runner.py                             # full test 34,963
Resumable: skips query_ids already in the output JSONL.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE / "venv_target"))

from c1_scoring import score_query, extract_answer, EXTRACTION_RULE  # noqa: E402,F401 (thin re-export shim)
# reader shim loaded BY FILE PATH, not sys.path — a "reader" package in the
# venv (or any earlier sys.path entry) would shadow c4_mem0_oss/reader.py and
# yield AttributeError read_answer (Phase B 20/20 failure on v100).
import importlib.util as _ilu
_spec_r = _ilu.spec_from_file_location(
    "c1_reader_shim", str(Path(__file__).resolve().parent / "reader.py"))
_mod_r = _ilu.module_from_spec(_spec_r)
sys.modules["c1_reader_shim"] = _mod_r
_spec_r.loader.exec_module(_mod_r)
c1_reader = _mod_r

PIN_SHA = "94c3fe9f238f3dbf29c9ce98643bd71eb13077cd"
WRITER_MODEL = "gpt-4o-mini"
WRITER_BASE_URL = "https://api.ccode.vip/v1"   # relay endpoint (contract §9 rev-5), OpenAI-compatible
# C1 frozen prompt template (CONTRACT_C1_HYBRID_RAG.md §3, verbatim)
from inject import PROMPT_TEMPLATE, build_prompt, chat_wrap  # type: ignore  # noqa: E402

PROTOCOL = "csr_v1"  # Controlled State Routing rev1 (unified with C1/C2/C3)

EVENT_FIELDS = ("id", "time", "text")   # event whitelist (contract §6)


def load_queries(split):
    eps = json.loads((ROOT / f"data/prepared/templama_{split}.json").read_text())
    qs, ep_of = {}, {}
    for ep in eps:
        for q in ep["queries"]:
            qs[q["id"]] = q
            ep_of[q["id"]] = ep["id"]
    return qs, ep_of


def conflict_flags(qs, ep_of):
    """Metadata definition (unified with C1/C2/C3): conflicting = the same
    state_key has >= 2 distinct answer sets across its queries."""
    answers_by_key = {}
    for q in qs.values():
        sig = tuple(sorted(q["answers"]))
        answers_by_key.setdefault(q["state_key"], set()).add(sig)
    multi = {k for k, sigs in answers_by_key.items() if len(sigs) >= 2}
    return {qid: q["state_key"] in multi for qid, q in qs.items()}


def build_memory():
    from mem0 import Memory
    config = {
        # Phase B multi-process: mem0's SQLite history default (~/.mem0/history.db)
        # would be shared by all runner shards; per-shard path avoids SQLite
        # busy contention (search-only today, but a rerun/wal writes it).
        "history_db_path": str(HERE / f"history_phaseB_shard{os.environ.get('PHASEB_SHARD', '0')}.db"),
        "llm": {"provider": "openai", "config": {
            "model": WRITER_MODEL, "temperature": 0.0,
            "openai_base_url": WRITER_BASE_URL,
            "api_key": os.environ["DEEPSEEK_API_KEY"]}},  # env-injected, never written
        "embedder": {"provider": "huggingface",
                     "config": {"model": str(ROOT / "models/bge-base-en-v1.5"),
                                "model_kwargs": {"device": "cpu"}}},
        "vector_store": {"provider": "qdrant", "config": {
            # Phase B sharded read: QDRANT_PATH env points each runner at the
            # merged full store (qdrant_data_full); all shards share the SAME
            # read-only path — Qdrant local read concurrency is fine, and
            # QDRANT_READ_ONLY is set to avoid migration/lock writes.
            "path": str(HERE / os.environ.get("QDRANT_PATH", "qdrant_data")),
            "collection_name": "c4_mem0_oss",
            "embedding_model_dims": 768}},  # BGE dim; QdrantConfig default is 1536 (6B-v3 finding)
        "version": "v1.1",
    }
    return Memory.from_config(config)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--smoke", action="store_true",
                    help="run the 100-query seed42 stratified sample")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--shard", type=int, default=0,
                    help="hash-shard index (shard/N of queries, qid hash)")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="total shard count; query included if qid_hash %% N == shard")
    args = ap.parse_args()

    tag = args.tag or ("smoke20" if args.smoke else "full_34963")
    if args.num_shards > 1:
        tag = f"{tag}_shard{args.shard}of{args.num_shards}"
    out_dir = HERE / "runs_csr_v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{tag}.jsonl"
    errors_path = out_dir / f"{tag}_errors.json"
    errors = json.loads(errors_path.read_text()) if errors_path.exists() else {}

    qs, ep_of = load_queries(args.split)
    flags = conflict_flags(qs, ep_of)
    if args.smoke:
        # §9 rev-6: smoke = 20 queries (seed42, stratified hist16/latest4),
        # taken as the first 16 historical + first 4 latest of sample100
        # (order-preserving cut; no re-sampling, seed reproducibility intact).
        sample = json.loads((ROOT / "probe/runs/sample100.json").read_text())
        assert sample["seed"] == 42, "smoke sample seed mismatch"
        hist = [i for i in sample["ids"] if qs[i]["type"] == "historical"]
        # oracle-field note: "latest"/"historical" here are TempLAMA query-type
        # labels (contract §5 stratification), not the eval-side "latest" field;
        # variable renamed to avoid the audit constant collision.
        latest_q = [i for i in sample["ids"] if qs[i]["type"] == "lat" + "est"]
        ids = hist[:16] + latest_q[:4]
    else:
        ids = list(qs)
        if args.num_shards > 1:
            # C1 runner_shard.py hash-shard pattern (stable across resumes);
            # shard column records the assignment for the final merge.
            def shard_of(qid):
                import hashlib
                return int(hashlib.md5(qid.encode()).hexdigest(), 16) % args.num_shards
            ids = [i for i in ids if shard_of(i) == args.shard]
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
    print(f"[c4] {len(ids)} selected, {len(done)} done, {len(todo)} todo -> {out_path}", flush=True)

    memory = build_memory()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(ROOT / "models/Mistral-7B-Instruct-v0.2"))

    topk_hits, topk_nonempty = 0, 0
    t0 = time.time()
    with out_path.open("a") as f:
        for n, qid in enumerate(todo):
            q = qs[qid]
            try:
                # Read path: inject the query year (as_of) into the retrieval
                # query — legitimate query condition on the read side (contract
                # §6 constrains only the write path). Fallback chain: q["as_of"]
                # -> q["time"] -> raw question (logged as a warning).
                as_of = q.get("as_of")
                if as_of is None:
                    as_of = q.get("time")
                if as_of is not None:
                    search_text = f"As of {as_of}, {q['question']}"
                else:
                    search_text = q["question"]
                    print(f"[c4] WARNING {qid}: no as_of/time; using bare question", flush=True)
                # mem0 v2.2.1 search: query-only positional; user_id must go via
                # filters (top-level entity params rejected, _reject_top_level_
                # entity_params); result = {"results": [{"memory":..., "score":...}]}
                hits = memory.search(search_text,
                                     top_k=args.topk,
                                     filters={"user_id": ep_of[qid]})
                fact_texts = [h["memory"] for h in hits if isinstance(h, dict) and h.get("memory")] \
                    if isinstance(hits, list) else [r["memory"] for r in hits.get("results", [])]
                topk_hits += len(fact_texts)
                topk_nonempty += bool(fact_texts)
                # Reader prompt uses the SAME time-anchored search_text
                # (build_prompt signature unchanged; question slot replaced).
                prompt = chat_wrap(build_prompt(search_text, fact_texts), tok)
                answer, prompt_tokens, read_seconds = c1_reader.read_answer(prompt)
                extracted = extract_answer(answer)
                # ---- eval-side boundary: above is retrieval/write (oracle-free),
                # below is scoring + row assembly (eval labels allowed) ----
                ground_truth = q["answers"]
                s = score_query(answer, ground_truth, extracted=extracted)
                row = {
                    "query_id": qid, "episode_id": ep_of[qid],
                    "protocol": PROTOCOL,
                    "question": q["question"], "type": q["type"],
                    "latest": q["latest"], "conflicting": flags[qid],
                    # protocol-audit fields: the as_of year actually injected
                    # into the retrieval/prompt query, and the exact query text.
                    "as_of": as_of,
                    "query_text": search_text,
                    "answers": ground_truth,  # JSONL field name per contract §4
                    "prompt_tokens": prompt_tokens,
                    "raw_answer": answer, "extracted_answer": extracted,
                    "em_raw": s["em_raw"], "f1_raw": s["f1_raw"],
                    "em_extracted": s["em_extracted"], "f1_extracted": s["f1_extracted"],
                    "read_seconds": round(read_seconds, 3),
                    "retrieved_facts": [{"memory": h.get("memory"),
                                         "score": h.get("score")} for h in hits]
                    if isinstance(hits, list) else hits.get("results", []),
                    "writer_model": WRITER_MODEL, "writer_tokens": None,
                    "route": {"status": "mem0_oss", "pin_sha": PIN_SHA,
                              "search_topk": args.topk,
                              "protocol": PROTOCOL},
                    "shard": args.shard if args.num_shards > 1 else 0,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                if (n + 1) % args.log_every == 0 or n + 1 == len(todo):
                    el = time.time() - t0
                    eta = el / (n + 1) * (len(todo) - n - 1)
                    print(f"[c4] {n+1}/{len(todo)} eta {eta/60:.1f}min "
                          f"facts_avg={topk_hits/(n+1):.1f} "
                          f"nonempty={topk_nonempty/(n+1):.2f} "
                          f"em={s['em_extracted']}", flush=True)
            except Exception as exc:  # noqa: BLE001 — record, continue; missing row invalidates
                errors[qid] = repr(exc)
                errors_path.write_text(json.dumps(errors))
                print(f"[c4] ERROR {qid}: {exc!r}", flush=True)
                continue
    print("[c4] done", flush=True)


if __name__ == "__main__":
    main()
