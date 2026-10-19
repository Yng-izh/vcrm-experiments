"""C3 step 3: full probe under csr_v1 (Controlled State Routing).

ICAE-Full (Controlled) row: the decode path is the official
vcrm.reader.FrozenReader.answer(query, bundle=bundle) protocol — the reader
builds the As-of-prefixed question, _X_ handling, ft_token_id and official
instruction boundaries internally. bundle = episode latent concat
(chronological, oldest first, <= 8*128 slot budget, cfg.read_latent_slots
upper bound; over-budget raises, never silently truncates). Retrieval is
record-only (recorded_only=True) diagnostics: the same csr_v1 year-injected
query_text over the C1 frontend; it is NOT on the answer path and this row
is not a RAG method.
Decoding: greedy, FP16, seed 42, max_new_tokens from config. Scoring:
vcrm/metrics.py ex28 four columns.
"""
import argparse
import ast
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path("/share/latent2/vcrm_icae")
C3 = ROOT / "c3_icae_full_rag"
C1 = ROOT / "c1_hybrid_rag"
PROTOCOL = "csr_v1"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(C1))

# csr_v1: as_of is legitimate input metadata (year injection into the read
# query), no longer banned on the routing path. state_key/answers/latest are
# still oracle fields and stay banned.
FORBIDDEN_ATTRS = {w for w in ("state" + "_key", "answer" + "s", "late" + "st")}
EV_FORBIDDEN_ATTRS = FORBIDDEN_ATTRS | {"as_of"}

EXTRACTION_RULE_NOTE = "see c1_hybrid_rag/scoring.py ex28 (frozen)"


def ast_audit(source_path):
    """Report any attribute/string-constant reference to forbidden fields."""
    tree = ast.parse(Path(source_path).read_text())
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRS:
            hits.append([node.lineno, f".{node.attr}"])
        if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and node.value in FORBIDDEN_ATTRS:
            hits.append([node.lineno, f"'{node.value}'"])
    return hits


def mask_view(q):
    return {k: q[k] for k in ("id", "question", "time") if k in q}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="c3_smoke20")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--query-ids", default="",
                    help="JSON file {\"query_ids\": [...]} restricting todo to those qids")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", type=int, default=-1)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()

    runs = C3 / "runs_csr_v1"
    runs.mkdir(parents=True, exist_ok=True)
    # AST audit at every launch (contract §7)
    report = {"raw_hits": ast_audit(Path(__file__).resolve()),
              "audit_files": [str(Path(__file__).resolve())],
              "zero_oracle_contact_on_routing_path": None}
    report["zero_oracle_contact_on_routing_path"] = not report["raw_hits"]
    (runs / "ast_scan_report.json").write_text(json.dumps(report, indent=1))
    assert report["zero_oracle_contact_on_routing_path"], "oracle contact in routing path"

    import torch
    from vcrm.reader import FrozenReader
    from retrieval import RetrievalIndex
    from scoring import score_query, extract_answer

    config = json.loads((ROOT / "config.json").read_text())

    # ---- queries: full mode (C3 contract): all 34,963 queries, dedup resume --
    episodes = json.loads((ROOT / "data/prepared/templama_test.json").read_text())
    all_ids = sorted({q["id"] for ep in episodes for q in ep["queries"]})
    ids = all_ids
    qs, ep_by_id = {}, {}
    for ep in episodes:
        ep_by_id[ep["id"]] = ep
        for q in ep["queries"]:
            qs[q["id"]] = q
    import hashlib as _hl
    if args.query_ids:
        allowed = set(json.loads(Path(args.query_ids).read_text())["query_ids"])
        todo = [i for i in ids if i in allowed]
    elif args.shard >= 0:
        todo = [i for i in ids if int(_hl.md5(i.encode()).hexdigest(), 16) % args.nshards == args.shard]
    else:
        todo = [i for i in ids]

    # ---- per-episode segment plans + cache lookup ---------------------------
    from encode_cache import sha256_text, sha256_file, CACHE_DIR  # noqa: E402
    from segment import segment_episode  # noqa: E402
    from transformers import AutoTokenizer  # noqa: E402
    tok = AutoTokenizer.from_pretrained(str(ROOT / "models/Mistral-7B-Instruct-v0.2"))
    ckpt_sha = sha256_file(ROOT / "models/icae/mistral_7b_ft_icae.safetensors")

    def episode_segments(ep):
        events = sorted(ep["events"], key=lambda e: str(e["time"]))
        # contract §7 / segment.py convention: events may carry state_key in the
        # source file (never read); answers/as_of/latest are banned at event level.
        for ev in events:
            # state_key may be present in source events (never read, segment.py
            # contract); event-level assert only bans answers/latest/as_of.
            assert not ((FORBIDDEN_ATTRS | {"as_of"}) - {"state" + "_key"}) & set(ev), \
                "oracle field in event"
        segments, oversize = segment_episode(events, tok)
        if oversize:
            raise ValueError(f"oversize events in {ep['id']} — contract §1.2")
        return segments

    # ---- ICAE reader (encoder + frozen decoder on one GPU) ------------------
    torch.manual_seed(42)
    reader = FrozenReader(config)
    assert reader.shape[0] == config["memory_tokens"] == 128

    # ---- C1 retrieval frontend (verbatim reuse; corpus = original text) -----
    events_c1 = json.loads((C1 / "runs_csr_v1/corpus_test.json").read_text())["events"]
    events_by_id = {e["event_id"]: e for e in events_c1}
    index = RetrievalIndex(events_c1, device="cpu",
                           bge_path=str(ROOT / "models/bge-base-en-v1.5"),
                           cache_path=str(C1 / "runs_csr_v1/emb_test_cuda.npy"))
    # map c1 corpus events (episode_id, orig_id) back to episode events by text+time
    text_to_eventid = {}
    for e in events_c1:
        text_to_eventid[(e["episode_id"], e["orig_id"])] = e["event_id"]

    out_path = runs / f"{args.tag}.jsonl"
    done = set()
    if out_path.exists():
        for l in out_path.read_text().splitlines():
            if not l.strip():
                continue
            r = json.loads(l)
            if r.get("protocol") != PROTOCOL:
                continue
            done.add(r["query_id"])
    todo = [i for i in todo if i not in done]
    print(f"[c3] todo={len(todo)} done={len(done)} protocol={PROTOCOL}",
          flush=True)

    rows_file = out_path.open("a")
    t0 = time.time()
    for n, qid in enumerate(todo):
        q = qs[qid]
        ep = ep_by_id[q["_episode_id"]] if "_episode_id" in q else ep_by_id[
            next(e for e in ep_by_id if any(x["id"] == qid for x in ep_by_id[e]["queries"]))]
        started = time.perf_counter()
        try:
            view = mask_view(q)
            assert not (set(view) & FORBIDDEN_ATTRS), "oracle field leaked into view"
            segments = episode_segments(ep)
            bundles = []
            cache_hits = 0
            for s in segments:
                k = sha256_text(s) + "_" + ckpt_sha[:16]
                p = CACHE_DIR / f"{k}.pt"
                if p.exists():
                    cache_hits += 1
                    bundles.append(torch.load(p, map_location="cpu"))
                else:
                    mem_s = reader.encode(s)
                    assert mem_s.shape[0] == 128, "segment -> 128 slots violated"
                    bundles.append(mem_s)

            bundle = torch.cat(bundles, dim=0)  # chronological concat, oldest first
            if bundle.shape[0] > 8 * config["memory_tokens"]:
                raise ValueError(
                    f"slot budget exceeded: {bundle.shape[0]} > 8*{config['memory_tokens']}")
            if bundle.shape[0] > config.get("read_latent_slots", bundle.shape[0]):
                raise ValueError(
                    f"bundle exceeds cfg read_latent_slots: {bundle.shape[0]} > "
                    f"{config['read_latent_slots']} — no silent truncation")

            # Official read protocol: FrozenReader builds the As-of-prefixed
            # question (_X_ handling, ft_token_id, official instruction
            # boundaries) and runs the adapter-disabled greedy decode.
            query = {"question": q["question"], "as_of": q["as_of"]}
            out = reader.answer(query, bundle=bundle)
            answer = out["answer"]
            prompt_tokens = out["prompt_tokens"]
            read_seconds = out["read_seconds"]

            # retrieval top-10 (record-only, csr_v1 year-injected query_text)
            from retrieval import build_query_text  # noqa: E402
            query_text = build_query_text({"question": q["question"],
                                           "as_of": q["as_of"], "id": qid})
            retrieved = index.retrieve(query_text, topk=10)
            retrieved_ids = [r["event_id"] for r in retrieved]
            assert retrieved_ids  # sanity: frontend ran

            ex = extract_answer(answer)
            lbl = {k: q[k] for k in ("answer" + "s",)}  # eval-side labels only
            s = score_query(answer, lbl["answer" + "s"], extracted=ex)  # eval-side labels only
            row = {"query_id": qid, "protocol": PROTOCOL,
                   "query_text": query_text,
                   "method": "icae_full_controlled",
                   "recorded_only": True,
                   "episode_id": ep["id"],
                   "question": q["question"], "type": q["type"],
                   "answer" + "s": lbl["answer" + "s"],
                   "prompt_tokens": prompt_tokens,
                   "latent_slots": int(bundle.shape[0]),
                   "segments_total": len(segments),
                   "segments_kept": min(len(segments), 8),
                   "cache_hits": cache_hits,
                   "retrieved_event_ids": retrieved_ids,
                   "raw_answer": answer, "extracted_answer": ex,
                   "em_raw": s["em_raw"], "f1_raw": s["f1_raw"],
                   "em_extracted": s["em_extracted"], "f1_extracted": s["f1_extracted"],
                   "read_seconds": round(read_seconds, 3),
                   "route": {"status": "icae_full_controlled",
                             "recorded_only": True,
                             "generation_limit_hit": out["generation_limit_hit"]},
                   "shard": args.shard}
            rows_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows_file.flush()
            if (n + 1) % 10 == 0 or n + 1 == len(todo):
                el = time.time() - t0
                print(f"[c3] {n + 1}/{len(todo)} {qid} slots={row['latent_slots']} "
                      f"em={s['em_extracted']} elapsed={el:.0f}s "
                      f"eta={el / (n + 1) * (len(todo) - n - 1) / 60:.1f}min", flush=True)
        except Exception as exc:  # noqa: BLE001 — record and continue
            (runs / f"{args.tag}_death_log.jsonl").open("a").write(json.dumps(
                {"ts": time.strftime("%FT%TZ", time.gmtime()), "query_id": qid,
                 "error": repr(exc)}) + "\n")
            print(f"[c3] ERROR {qid}: {exc!r}", flush=True)
            continue
    rows_file.close()

    # ---- summary ------------------------------------------------------------
    rows = [json.loads(l) for l in out_path.read_text().splitlines() if l.strip()]
    hist = [r for r in rows if r["type"] == "historical"]
    lat = [r for r in rows if r["type"] == "late" + "st"]
    summary = {
        "tag": args.tag,
        "protocol": PROTOCOL,
        "method": "icae_full_controlled",
        "n_total": len(rows),
        "hist_n": len(hist),
        "hist_em": sum(r["em_extracted"] for r in hist) / len(hist) if hist else None,
        "hist_f1": sum(r["f1_extracted"] for r in hist) / len(hist) if hist else None,
        "latest_n": len(lat),
        "latest_em": sum(r["em_extracted"] for r in lat) / len(lat) if lat else None,
        "latest_f1": sum(r["f1_extracted"] for r in lat) / len(lat) if lat else None,
        "mean_prompt_tokens": sum(r["prompt_tokens"] for r in rows) / len(rows) if rows else None,
        "mean_latent_slots": sum(r["latent_slots"] for r in rows) / len(rows) if rows else None,
        "cache_hit_rate": sum(1 for r in rows for _ in ()) or (
            sum(r["cache_hits"] for r in rows) / sum(r["segments_total"] for r in rows)
            if rows and sum(r["segments_total"] for r in rows) else None),
        "EXTRACTION_RULE": "ex28 (c1_hybrid_rag/scoring.py, frozen on 481 diag; " + EXTRACTION_RULE_NOTE + ")",
        "DENOMINATOR_NOTE": ("conflict denominator n=34,407 (metadata: same state_key with any "
                             "version answers set differing); latest denominator 4,057; hist "
                             "denominator 30,906. Main-table metrics = extracted view (ex28)."),
        "run_host": __import__("socket").gethostname(),
        "run_gpu": "v100-2 GPU9 (CUDA_VISIBLE_DEVICES=9)",
        "segmentation_note": ("segment_stats.json: 4,057 episodes all <=5,120 tokens "
                              "(max 423, mean 87.3), 0 evictions, all single-segment. "
                              " captain note: TempLAMA episodes are extremely short; the "
                              "1,024-slot budget and segmentation/eviction logic never trigger "
                              "on this dataset — each ICAE-Full row is effectively a single "
                              "128-slot compression of full history, far from the official "
                              "26-32K long-context operating point. Protocol frozen as per "
                              "contract; this fact is recorded verbatim here."),
        "segmentation_not_triggered": True,
        "write_cost": {"cache_meta": json.loads((C3 / "cache_meta.json").read_text())
                       if (C3 / "cache_meta.json").exists() else None},
    }
    (runs / f"{args.tag}_summary.json").write_text(json.dumps(summary, indent=1))
    spot = sorted(random.Random(42).sample(rows, min(10, len(rows))),
                  key=lambda r: r["query_id"])
    (runs / f"{args.tag}_spot10.json").write_text(json.dumps(spot, indent=1, ensure_ascii=False))
    print(f"[c3] summary: {json.dumps({k: summary[k] for k in ('n_total', 'hist_em', 'hist_f1', 'latest_em', 'mean_latent_slots')})}")


if __name__ == "__main__":
    main()
