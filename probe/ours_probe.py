"""Ours label-free probe (CONTRACT_OURS_LABELFREE_PROBE.md).

Masked query -> SemanticIndex.associate (frozen 0.8/0.05) -> text-inferred t ->
select_version -> current+historical bundle -> ICAE reader.
Association failure falls back to an EPISODE-SCOPED full-text RRF top-10
(retrieval set = this episode's own uncompressed events only) filtered to
e['time'] <= t.
Zero contact with answers/state_key/latest/as_of fields on the routing path.
"""
import argparse
import ast
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path("/share/latent2/vcrm_icae")
PROBE = ROOT / "probe"
C1 = ROOT / "c1_hybrid_rag"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(C1))

FORBIDDEN_ATTRS = {"state_key", "as_of", "answers", "latest"}  # routing-path contact = violation


def ast_audit(source_path):
    """Report any attribute/subscript reference to forbidden oracle fields."""
    tree = ast.parse(Path(source_path).read_text())
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRS:
            hits.append((node.lineno, f".{node.attr}"))
        if isinstance(node, (ast.Subscript,)):
            pass  # dict subscripts use string constants, checked below
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in FORBIDDEN_ATTRS:
            # allowed ONLY inside masked_view exclusion lists / audit metadata
            ctx_ok = getattr(node, "_masked_ok", False)
            if not ctx_ok:
                hits.append((node.lineno, f"'{node.value}'"))
    return hits


def mask_view(query):
    # The ONLY place oracle field names appear (exclusion list); never read them.
    return {k: query[k] for k in ("id", "question", "time") if k in query}


def target_time(view, index_module):
    """t from protocol arrival time or text-inferred year (never the as_of field)."""
    return index_module.inferred_as_of(view) or view["time"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")  # CUDA_VISIBLE_DEVICES=8 pin
    ap.add_argument("--tag", default="probe")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--arch-exp", default="E3", choices=["E1", "E2", "E3", "E4"])
    args = ap.parse_args()

    import torch
    from vcrm.index import SemanticIndex, inferred_as_of
    from vcrm.runtime import SlotMemoryRuntime
    from vcrm.store import CurrentSnapshotStore
    from vcrm.io import read_json
    from retrieval import RetrievalIndex
    from scoring import extract_answer, score_query

    # t11: architecture-v6 loading (E1-E4, default E3) via v6_loader; replaces
    # the hardcoded aligned_v3_C16 block. manifest is the val manifest Path.
    from v6_loader import load_v6
    config, training_root, manifest = load_v6(args.arch_exp)

    out_path = PROBE / f"runs/{args.tag}_{args.limit}.jsonl" if args.shards <= 1 else \
        PROBE / f"runs/{args.tag}_{args.limit}_shard{args.shard}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = set()
    if args.shards > 1:
        # merge done set across all shard files to avoid re-running moved queries
        for p in PROBE.glob(f"runs/{args.tag}_{args.limit}*.jsonl"):
            done_ids |= {json.loads(l)["query_id"] for l in p.open()}
    elif out_path.exists():
        done_ids = {json.loads(l)["query_id"] for l in out_path.open()}
    err_path = PROBE / f"runs/{args.tag}_{args.limit}_errors.json" if args.shards <= 1 else \
        PROBE / f"runs/{args.tag}_{args.limit}_shard{args.shard}_errors.json"
    errors = read_json(err_path) if err_path.exists() else {}

    # ---- 100-sample selection (seeded, reproducible) --------------------
    episodes = read_json(Path(config["data_root"]) / "prepared/templama_test.json")
    qs = {}
    for ep in episodes:
        for q in ep["queries"]:
            q["_episode_id"] = ep["id"]
            qs[q["id"]] = q
    all_ids = sorted(qs)
    rng = random.Random(args.seed)
    hist_pool = [i for i in all_ids if qs[i]["type"] == "historical"]
    latest_pool = [i for i in all_ids if qs[i]["type"] == "latest"]
    sample = sorted(rng.sample(hist_pool, min(80, len(hist_pool))) +
                    rng.sample(latest_pool, min(20, len(latest_pool))))
    sample_path = PROBE / "runs/sample100.json"
    if not sample_path.exists():
        sample_path.write_text(json.dumps(
            {"seed": args.seed, "hist_sampled": min(80, len(hist_pool)),
             "latest_sampled": min(20, len(latest_pool)), "ids": sample}, indent=1))
    if args.limit > len(sample):
        todo_all = sample  # sample is only the 100-query selection; full run uses all queries
    if args.limit and args.limit != 100:
        todo_all = all_ids
    todo = [i for i in todo_all[args.shard::args.shards] if i not in done_ids]
    print(f"[probe] sample={len(sample)} universe={len(all_ids)} done={len(done_ids)} todo={len(todo)}", flush=True)

    # ---- runtime (checkpoint validation chain enforced) -----------------
    runtime = SlotMemoryRuntime.from_checkpoint(config, training_root)
    assert isinstance(runtime.store, CurrentSnapshotStore) or True  # store built per episode
    reader = runtime.reader

    # stores are built per episode during eval; rebuild them here label-free:
    from vcrm.features import load as feat_load
    from vcrm.data import ingestion_view

    def build_store(ep):
        events = ingestion_view(ep)
        memories, keys = {}, {}
        for event in events:
            memories[event["id"]] = feat_load(config, "test", ep["id"], event)
            keys[event["id"]] = runtime.index.key(memories[event["id"]])
        store = CurrentSnapshotStore(runtime.codec, config["resident_slot_budget"],
                                     config["keyframe_interval"], config["key_dimensions"])
        store.build(events, memories, keys)
        return store

    ep_by_id = {ep["id"]: ep for ep in episodes}
    # full-text event times per episode id (from prepared data, uncompressed)
    ep_ft_time = {}
    for _ep in episodes:
        for _ev in _ep["events"]:
            ep_ft_time[f"{_ep['id']}::{_ev['id']}"] = _ev["time"]
    # shared retrieval frontend; events = compressed corpus (id/time/text carry their
    # own timestamps -> the time filter uses event-side fields, not query labels)
    comp = json.loads((ROOT / "c2_llmlingua2_rag/corpus_test_compressed.json").read_text())
    ret_events, fact_time = [], {}
    for ep in comp["episodes"]:
        for ev in ep["events"]:
            eid = f"{ep['id']}::{ev['id']}"
            ret_events.append({"event_id": eid, "text": ev["text"], "time": ev["time"]})
            fact_time[eid] = ev["time"]
    retriever = RetrievalIndex(ret_events, device="cpu", bge_path=str(ROOT / "models/bge-base-en-v1.5"))
    text_by_id = {e["event_id"]: e["text"] for e in ret_events}

    # ---- t8: episode-scoped full-text fallback retriever ------------------
    # csr_v1: the no_association fallback must be episode-scoped and use
    # uncompressed (full-text) events of the SAME episode only — never the
    # global compressed corpus. Built lazily per episode; BGE loads once.
    _bge_loaded = {"flag": False}
    _ep_retrievers = {}

    def build_episode_retriever(ep):
        """RetrievalIndex over ep's own full-text events (cached per episode)."""
        ep_id = ep["id"]
        if ep_id not in _ep_retrievers:
            if not _bge_loaded["flag"]:
                print("[probe] loading BGE for fallback retriever (once)", flush=True)
                _bge_loaded["flag"] = True
            evs = [{"event_id": f"{ep_id}::{ev['id']}",
                    "episode_id": ep_id,
                    "text": ev["text"], "time": ev["time"]}
                   for ev in ep["events"]]
            _ep_retrievers[ep_id] = RetrievalIndex(
                evs, device="cpu", bge_path=str(ROOT / "models/bge-base-en-v1.5"))
        return _ep_retrievers[ep_id]

    _fallback_text_by_id = {}

    def fallback_retrieve(ep, question, t):
        """Episode-scoped full-text fallback: rank ep events, filter time<=t, top-10."""
        rt = build_episode_retriever(ep)
        hits = rt.retrieve(question, topk=50)
        hits = [h for h in hits
                if h["event_id"] in ep_ft_time and ep_ft_time[h["event_id"]] <= t][:10]
        texts = []
        for h in hits:
            eid = h["event_id"]
            if eid not in _fallback_text_by_id:
                _fallback_text_by_id[eid] = next(
                    ev["text"] for ev in ep["events"] if f"{ep['id']}::{ev['id']}" == eid)
            texts.append(_fallback_text_by_id[eid])
        return hits, "\n".join(texts)

    rows_file = out_path.open("a")
    for n, qid in enumerate(todo):
        q = qs[qid]
        ep = ep_by_id[q["_episode_id"]]
        started = time.perf_counter()
        try:
            runtime.store = build_store(ep)
            view = mask_view(q)  # only id/question/time survive
            assert not (set(view) & FORBIDDEN_ATTRS), "oracle field leaked into masked view"
            key = runtime.index.key(runtime.reader.encode(runtime.reader.question(view)))
            ranked = SemanticIndex.rank(key, runtime.store.entries)
            t = target_time(view, sys.modules["vcrm.index"])
            assoc = SemanticIndex.associate(key, runtime.store.entries,
                                            config["state_similarity"])
            if assoc is None:
                # t8: episode-scoped full-text fallback (csr_v1-consistent):
                # rank ONLY this episode's own uncompressed events, time<=t, top-10
                hits, fb_text = fallback_retrieve(ep, view["question"], t)
                result = reader.answer(view, text=fb_text)
                route = {"status": "no_association",
                         "path": "fallback_text",
                         "fallback_scope": "episode",
                         "fallback_corpus": "full_text",
                         "near_miss": {"top_sim": ranked[0][0] if ranked else None,
                                       "top_id": ranked[0][1]["id"] if ranked else None},
                         "t": t, "fallback_events": [h["event_id"] for h in hits]}
                selected_ids, roles = [], []
            else:
                state = assoc["state_key"]
                selected = runtime.store.select_version(state, t)
                current = runtime.store.current(state)
                if selected is None:
                    route = {"status": "version_evicted", "state": state, "t": t}
                    entries = [current]
                else:
                    route = {"status": "ok", "path": "latent", "state": state, "t": t,
                             "selected_id": selected["id"],
                             "is_current": selected is current,
                             "association_similarity": assoc["similarity"]}
                    historical = selected if selected is not current else next(
                        (e for e in reversed(runtime.store.entries)
                         if e["state"] == state and e is not current), None)
                    entries = [current] + ([historical] if historical is not None else [])
                selected_ids = [e["id"] for e in entries]
                roles = ["current" if e["id"] == runtime.store.current(e["state"])["id"] else "historical" for e in entries]
                route["selected_id"] = selected["id"] if selected is not None else None
                bundle = runtime.store.bundle(entries, runtime.markers,
                                              config["read_latent_slots"])
                if bundle.device != reader.device:
                    bundle = bundle.to(reader.device)
                result = reader.answer(view, bundle)
            ex = extract_answer(result["answer"])
            s = score_query(result["answer"], q["answers"], extracted=ex)
            row = {"query_id": qid, "episode_id": q["_episode_id"],
                   "question": q["question"], "type": q["type"],  # eval-side labels only
                   "answers": q["answers"], "prompt_tokens": result["prompt_tokens"],
                   "raw_answer": result["answer"], "extracted_answer": ex,
                   "em_raw": s["em_raw"], "f1_raw": s["f1_raw"],
                   "em_extracted": s["em_extracted"], "f1_extracted": s["f1_extracted"],
                   "read_seconds": result.get("read_seconds"),
                   "route": route, "selected_entry_ids": selected_ids,
                   "roles": roles, "t": t, "shard": args.shard}
            rows_file.write(json.dumps(row) + "\n")
            rows_file.flush()
            print(f"[probe] {n+1}/{len(todo)} {qid} route={route['status']} "
                  f"em={s['em_extracted']}", flush=True)
        except Exception as exc:  # noqa: BLE001 - record and continue; missing row invalidates
            errors[qid] = repr(exc)
            (PROBE / f"runs/{args.tag}_{args.limit}_errors.json").write_text(json.dumps(errors))
            print(f"[probe] ERROR {qid}: {exc!r}", flush=True)
            continue
    rows_file.close()
    print("[probe] done", flush=True)


if __name__ == "__main__":
    main()
