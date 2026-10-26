#!/usr/bin/env python3
"""LongMemEval-S text-RAG baseline: session-level hybrid retrieval + frozen reader.

PROTOCOL NOTE (differs fundamentally from the TempLAMA csr_v1 probes):
- LongMemEval-S has NO episode partition, NO as_of year, NO state_key, and NO
  oracle fields to mask. Retrieval is over the WHOLE haystack: every haystack
  session is rendered into ONE retrieval document (with a date prefix to help
  temporal-reasoning questions), queried by the question text, top-k by RRF.
  Do NOT port the episode-scoped retrieval logic from probe/ours_probe.py —
  it would be wrong here (this comment exists so nobody "fixes" it back).

SCOPE SEMANTICS (P0 fix, see tests/test_lme_scope.py):
- The only legal *memory namespace* for a question is THAT question's own
  haystack (`item["sessions"]`). It is NOT the Oracle episode: LongMemEval-S
  ships no episode labels, and the answer-bearing session is unknown here, so
  the namespace must be defined purely from the question's own input.
- Retrieval is therefore PER-ITEM: `build_item_index()` builds a fresh
  RetrievalIndex over exactly one item's haystack, inside the per-item loop.
  There is no global/shared index anymore, so cross-question leakage is
  structurally impossible rather than merely filtered:
    * BM25 IDF and the dense matrix are computed only over this item's
      sessions — other questions' sessions cannot influence this ranking
      (not even statistically, which a shared index would still do);
    * document ids are namespaced as "<question_id>::<session_id>", so two
      questions that reuse the same session_id ("S1") map to distinct docs,
      and each id resolves only inside its own item's `session_by_id`;
    * `retrieve_for_item()` hard-asserts every returned id carries this
      item's "<question_id>::" prefix and exists in this item's haystack, so
      a leak (or a stale/cached index) fails loudly instead of silently
      writing a foreign session into the prompt or into
      retrieved_session_ids.
- Cost: one BGE encode pass per item (~one haystack of sessions each) instead
  of one pass over the whole corpus; the frozen SentenceTransformer weights
  are loaded once and reused across items (`model=` reuse) so the overhead is
  encode time only, never model reload.
- This script produces HYPOTHESES ONLY. LongMemEval-S is free-form QA scored
  by an official judge; we do NOT extract, normalize, or score answers here.
  Scoring fields/naming are owned by t12's spec; t14 wires them up.

Retrieval: reuses c1_hybrid_rag/retrieval.RetrievalIndex (BM25Okapi + BGE
models/bge-base-en-v1.5 + RRF), bge on CPU.

Reader (frozen, TempLAMA-main-table-consistent): Mistral-7B-Instruct-v0.2
greedy via the C1 vLLM OpenAI shim (c1_hybrid_rag/reader.read_answer), with
max_tokens raised to 256 by default (--max-new-tokens) because free-form QA
needs more than the TempLAMA 128 budget; temperature=0, seed=42 unchanged.
vLLM server must be started separately, e.g.:
  python3 -m vllm.entrypoints.openai.api_server \
    --model /share/latent2/vcrm_icae/models/Mistral-7B-Instruct-v0.2 \
    --served-model-name mistral-7b --dtype half --gpu-memory-utilization 0.85 \
    --max-model-len 32768 --port 8001

Usage:
  python run_text_rag.py --data <LongMemEval-S haystack json> \
      --tag lme_s_smoke --limit 50 --shard 0 --shards 1

Input format (LongMemEval-S / official haystack file, one JSON per line or a
JSON list — auto-detected): each item is an official cleaned row with
question_id, question, question_type, question_date, answer, and the three
index-aligned arrays haystack_session_ids / haystack_dates / haystack_sessions
(the last one is a list of turn lists, NOT session dicts). Parsing is delegated
to the shared loader `vcrm.lme_data.parse_item`, so this baseline and the latent
pipeline read exactly the same sessions in exactly the same order; misaligned
arrays raise instead of being zip-truncated. Paths are NOT hardcoded (pass
--data).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path("/share/latent2/vcrm_icae")
PROBE = ROOT / "c1_hybrid_rag"
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PROBE))
sys.path.insert(0, str(REPO_ROOT))

from vcrm.lme_data import parse_item as parse_lme_item  # noqa: E402  (shared loader)


def normalize_item(item):
    """Adapter: official cleaned row -> {question_id, question, question_type, sessions}.

    Parsing is delegated to `vcrm.lme_data.parse_item` — the SAME loader the
    latent pipeline uses — so both baselines see identical sessions in identical
    order. The official cleaned schema stores `haystack_sessions` as a list of
    TURN LISTS aligned by index with `haystack_session_ids` / `haystack_dates`;
    the loader raises on misalignment instead of zip-truncating.

    sessions: list of {session_id, date, text} sorted by date ascending (stable).
    Each turn is rendered as 'role: content' (same rendering as the latent
    pipeline); the session text is the date-prefixed concatenation of its turns
    in original order, where the date prefix keeps the official raw date string.
    """
    episode = parse_lme_item(item)
    sessions = [{"session_id": session.session_id,
                 "date": session.date,
                 "text": session.render(prefix_date=True)}
                for session in episode.sessions]
    return {"question_id": episode.question_id, "question": episode.question,
            "question_type": episode.question_type, "sessions": sessions}


def load_data(path):
    text = Path(path).read_text()
    try:
        data = json.loads(text)
        items = data if isinstance(data, list) else data.get("items", [data])
    except json.JSONDecodeError:
        items = [json.loads(l) for l in text.splitlines() if l.strip()]
    return [normalize_item(it) for it in items]


def render_prompt(question, sessions):
    """Chronological context block; same Context:/question framing as C1."""
    ctx = "\n\n".join(s["text"] for s in sessions)
    return f"Context:\n{ctx}\n\n{question}"


# ---------------------------------------------------------------------------
# Per-item scope: the legal memory namespace is this item's own haystack.
# ---------------------------------------------------------------------------
def event_id_of(question_id, session_id):
    """Namespaced doc id: "<question_id>::<session_id>".

    The question_id prefix is what makes a session id collision across
    questions harmless — "S1" of question A and "S1" of question B are two
    different documents, and each is only reachable from its own item index.
    """
    return f"{question_id}::{session_id}"


def session_id_of(event_id):
    """Inverse of event_id_of (split on the FIRST separator only)."""
    return event_id.split("::", 1)[1]


def session_by_id(item):
    """session_id -> session for THIS item only (first occurrence wins).

    Duplicate session_ids inside one haystack are collapsed to their first
    (chronologically earliest, since normalize_item sorts by date) occurrence —
    same behaviour as the previous shared-index implementation, kept so that
    retrieved_session_ids/retrieved_session_dates stay comparable.
    """
    out = {}
    for s in item["sessions"]:
        out.setdefault(s["session_id"], s)
    return out


def build_item_events(item):
    """Retrieval docs for ONE item: exactly its own haystack, nothing else."""
    qid = item["question_id"]
    events, seen = [], set()
    for s in item["sessions"]:
        eid = event_id_of(qid, s["session_id"])
        if eid in seen:
            continue  # duplicate session_id inside this haystack: keep first
        seen.add(eid)
        events.append({"event_id": eid, "text": s["text"],
                       "time": s["date"] or "1970-01-01"})
    return events


def build_item_index(item, index_cls=None, model=None, bge_path=None, device="cpu"):
    """Build a retrieval index over EXACTLY this item's haystack.

    Returns (index, model). `model` is threaded back in on the next item so the
    frozen BGE weights are loaded once per process instead of once per item
    (only the encode pass is per-item, which is the intended cost of true
    per-item isolation).

    `index_cls` is injectable for unit tests (defaults to the frozen
    c1_hybrid_rag RetrievalIndex so retrieval math is unchanged).
    """
    if index_cls is None:
        from retrieval import RetrievalIndex as index_cls  # c1_hybrid_rag (frozen)
    events = build_item_events(item)
    kwargs = {"device": device}
    if bge_path is not None:
        kwargs["bge_path"] = str(bge_path)
    if model is None:
        idx = index_cls(events, **kwargs)
    else:
        # RetrievalIndex constructs SentenceTransformer inside __init__; swap in
        # the already-loaded instance for this call only (no math is touched).
        import sentence_transformers
        original = sentence_transformers.SentenceTransformer
        sentence_transformers.SentenceTransformer = lambda *a, **k: model
        try:
            idx = index_cls(events, **kwargs)
        finally:
            sentence_transformers.SentenceTransformer = original
    return idx, getattr(idx, "model", model)


def retrieve_for_item(index, item, topk):
    """Top-k for ONE item, validated against that item's namespace.

    Returns (hit_ids, selected_sessions). Any hit that is not namespaced to this
    question_id, or whose session_id is absent from this haystack, raises — a
    cross-question hit must never reach the prompt or the output row.
    """
    qid = item["question_id"]
    prefix = f"{qid}::"
    hits = index.retrieve(item["question"], topk=topk)
    hit_ids = [h["event_id"] for h in hits]
    known = session_by_id(item)
    foreign = [h for h in hit_ids
               if not h.startswith(prefix) or session_id_of(h) not in known]
    if foreign:
        raise RuntimeError(
            f"cross-question retrieval leak for {qid}: {foreign[:3]} "
            f"(namespace = this item's haystack only)")
    chosen = {session_id_of(h) for h in hit_ids}
    # keep original chronological order among retrieved sessions
    sel = [s for s in item["sessions"] if s["session_id"] in chosen]
    return hit_ids, sel


def build_row(item, hit_ids, session_lookup, hypothesis, prompt_tokens, read_seconds):
    """Output row; field names/order are consumed by make_official_eval_input.py."""
    return {
        "question_id": item["question_id"],
        "question": item["question"],
        "question_type": item["question_type"],
        "hypothesis": hypothesis,
        "retrieved_session_ids": hit_ids,
        "retrieved_session_dates": [session_lookup[session_id_of(h)]["date"]
                                    for h in hit_ids],
        "prompt_tokens": prompt_tokens,
        "read_seconds": read_seconds,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True,
                    help="path to LongMemEval-S haystack/questions file (NOT hardcoded; see README)")
    ap.add_argument("--tag", default="lme_s_text_rag")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--device", default="cuda:0",
                    help="only used to pin vLLM placement out-of-band; retrieval BGE always runs on CPU")
    ap.add_argument("--vllm-url", default="http://localhost:8001/v1/completions")
    args = ap.parse_args()

    # RetrievalIndex (c1_hybrid_rag/retrieval.py, BM25+BGE+RRF) is imported
    # lazily inside build_item_index so the module stays importable without the
    # retrieval deps (unit tests inject a fake index_cls).
    import reader as c1_reader           # c1_hybrid_rag/reader.py vLLM shim
    c1_reader.VLLM_URL = args.vllm_url
    orig_read = c1_reader.read_answer

    def read_answer(prompt, max_retries=6, timeout=600):
        payload_max = args.max_new_tokens
        import json as _json
        import time as _time
        import urllib.request as _ur
        payload = {"model": c1_reader.MODEL, "prompt": prompt,
                   "temperature": 0.0, "max_tokens": payload_max, "seed": 42}
        last = None
        for attempt in range(max_retries):
            try:
                t0 = _time.time()
                req = _ur.Request(c1_reader.VLLM_URL,
                                  data=_json.dumps(payload).encode(),
                                  headers={"Content-Type": "application/json"})
                with _ur.urlopen(req, timeout=timeout) as r:
                    out = _json.loads(r.read())
                return (out["choices"][0]["text"].strip(),
                        out["usage"]["prompt_tokens"], _time.time() - t0)
            except Exception as e:
                last = e
                _time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"vllm call failed after {max_retries} retries: {last}")

    items = load_data(args.data)
    items.sort(key=lambda it: it["question_id"])
    if args.limit:
        items = items[:args.limit]
    todo = items[args.shard::args.shards]

    out_dir = HERE / "runs" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    err_path = out_dir.parent / f"{args.tag}_errors.json"
    errors = json.loads(err_path.read_text()) if err_path.exists() else {}

    # resume: merge done set across shard dirs of the same tag
    done = set()
    for d in (HERE / "runs").glob(f"{args.tag}"):
        for f in d.glob("*.json"):
            done.add(f.stem)
    todo = [it for it in todo if it["question_id"] not in done]
    print(f"[lme] items={len(items)} done={len(done)} todo={len(todo)}", flush=True)

    # ---- PER-ITEM retrieval index (namespace = this item's haystack) ----
    # No shared corpus index is built: see the SCOPE SEMANTICS header note for
    # why per-item indexing (not id filtering over a global index) is what makes
    # cross-question leakage impossible here.
    bge_path = ROOT / "models/bge-base-en-v1.5"
    shared_model = None
    print("[lme] retrieval: per-item haystack index (no shared index)", flush=True)

    for n, it in enumerate(todo):
        qid = it["question_id"]
        started = time.perf_counter()
        try:
            index, shared_model = build_item_index(
                it, model=shared_model, bge_path=bge_path, device="cpu")
            hit_ids, sel = retrieve_for_item(index, it, args.topk)
            prompt = render_prompt(it["question"], sel)
            hypothesis, prompt_tokens, read_seconds = read_answer(prompt)
            row = build_row(it, hit_ids, session_by_id(it), hypothesis,
                            prompt_tokens, read_seconds)
            (out_dir / f"{qid}.json").write_text(json.dumps(row, indent=1))
            print(f"[lme] {n + 1}/{len(todo)} {qid} type={it['question_type']} "
                  f"docs={len(it['sessions'])} tok={prompt_tokens} sec={read_seconds:.1f}",
                  flush=True)
        except Exception as exc:  # noqa: BLE001 - per-sample isolation
            errors[qid] = repr(exc)
            err_path.write_text(json.dumps(errors))
            print(f"[lme] ERROR {qid}: {exc!r}", flush=True)
            continue
    print("[lme] done", flush=True)


if __name__ == "__main__":
    main()
