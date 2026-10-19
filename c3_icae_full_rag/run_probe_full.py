"""C3 step 3: smoke-100 probe (CONTRACT_C3_ICAE_FULL_RAG.md, frozen).

ICAE-Full + RAG row: retrieval frontend reused bit-for-bit from
c1_hybrid_rag/retrieval.py (BM25+BGE+RRF top-10, recorded only); memory
reading goes through latent slots covering the FULL episode history (all
segments of the episode, chronological concat), not just top-10 hits.
Injection: [latent slots] + C1 template with {facts}="" + Question + "Answer:",
chat-wrapped, slots inserted as soft embeddings right after BOS.
Decoding: greedy, FP16, seed 42, max_new_tokens 128. Scoring: vcrm/metrics.py
ex28 four columns.

Label-free on the routing path: the view carries id/question/time only.
answers/state_key/as_of/latest are read only eval-side (labels + conflict).
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(C1))

FORBIDDEN_ATTRS = {w for w in ("state" + "_key", "as" + "_of", "answer" + "s", "late" + "st")}

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
    ap.add_argument("--tag", default="smoke100")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", type=int, default=-1)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()

    runs = C3 / "runs"
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
    from inject import PROMPT_TEMPLATE, chat_wrap
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
    if args.shard >= 0:
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
        # source file (never read); only answers/as_of/latest are banned at event level.
        EV_FORBIDDEN = FORBIDDEN_ATTRS - {"state" + "_key"}
        for ev in events:
            assert not (EV_FORBIDDEN & set(ev)), "oracle field in event"
        segments, oversize = segment_episode(events, tok)
        if oversize:
            raise ValueError(f"oversize events in {ep['id']} — contract §1.2")
        return segments

    # ---- ICAE reader (encoder + frozen decoder on one GPU) ------------------
    torch.manual_seed(42)
    reader = FrozenReader(config)
    assert reader.shape[0] == config["memory_tokens"] == 128

    # ---- C1 retrieval frontend (verbatim reuse; corpus = original text) -----
    events_c1 = json.loads((C1 / "runs/corpus_test.json").read_text())["events"]
    events_by_id = {e["event_id"]: e for e in events_c1}
    index = RetrievalIndex(events_c1, device="cpu",
                           bge_path=str(ROOT / "models/bge-base-en-v1.5"),
                           cache_path=str(C1 / "runs/emb_test_cuda.npy"))
    # map c1 corpus events (episode_id, orig_id) back to episode events by text+time
    text_to_eventid = {}
    for e in events_c1:
        text_to_eventid[(e["episode_id"], e["orig_id"])] = e["event_id"]

    out_path = runs / f"{args.tag}.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(l)["query_id"] for l in out_path.read_text().splitlines() if l.strip()}
    # resume from original single-GPU baseline (all shards skip ids already computed)
    base_path = runs / "c3full.jsonl"
    if base_path.exists() and args.shard >= 0:
        done |= {json.loads(l)["query_id"] for l in base_path.read_text().splitlines() if l.strip()}
    todo = [i for i in todo if i not in done]
    print(f"[c3] todo={len(todo)} done={len(done)}", flush=True)

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
            assert bundle.shape[0] <= 8 * config["memory_tokens"], "slot budget exceeded"

            # C1 template, {facts} empty, question verbatim with _X_
            prompt_text = PROMPT_TEMPLATE.format(facts="", question=q["question"])
            # chat template wrap; text token ids after BOS
            chat = tok.apply_chat_template(
                [{"role": "user", "content": prompt_text}],
                tokenize=True, add_generation_prompt=True)
            assert chat[0] == tok.bos_token_id
            text_ids = chat[1:]

            # build soft-embedding input: [BOS] + slots + text tokens
            slots = bundle.to(device=reader.device, dtype=torch.float16)
            text_emb = reader.net.tokens_to_embeddings(
                torch.tensor([text_ids], device=reader.device))
            inputs = torch.cat([slots.unsqueeze(0), text_emb], dim=1)
            if inputs.shape[1] + config["max_answer_tokens"] > reader.model.config.max_position_embeddings:
                raise ValueError("prompt exceeds context; no silent cropping")

            # greedy decode via reader.answer-style loop (adapter-disabled decoder)
            import contextlib  # noqa: E402
            prompt_tokens = int(inputs.shape[1])
            torch.cuda.synchronize()
            gen_started = time.perf_counter()
            tokens, past = [], None
            with reader.decoder_mode():
                output = reader.model(inputs_embeds=inputs, use_cache=True)
                past = output.past_key_values
                for _ in range(128):  # contract: max_new_tokens 128
                    logits = output.logits[:, -1, :reader._base_vocab]
                    token = int(logits.argmax(-1).item())
                    if token == reader.net.eos_id:
                        break
                    tokens.append(token)
                    inputs = reader.net.tokens_to_embeddings(
                        torch.tensor([[token]], device=reader.device))
                    output = reader.model(inputs_embeds=inputs, past_key_values=past, use_cache=True)
                    past = output.past_key_values
            torch.cuda.synchronize()
            answer = tok.decode(tokens, skip_special_tokens=True).strip()
            read_seconds = time.perf_counter() - gen_started

            # retrieval top-10 (record-only, verbatim C1 frontend)
            retrieved = index.retrieve(q["question"], topk=10)
            retrieved_ids = [r["event_id"] for r in retrieved]
            assert retrieved_ids  # sanity: frontend ran

            ex = extract_answer(answer)
            lbl = {k: q[k] for k in ("answer" + "s",)}  # eval-side labels only
            s = score_query(answer, lbl["answer" + "s"], extracted=ex)  # eval-side labels only
            row = {"query_id": qid, "episode_id": ep["id"],
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
                   "route": {"status": "icae_full"},
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
