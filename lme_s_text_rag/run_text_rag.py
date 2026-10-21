#!/usr/bin/env python3
"""LongMemEval-S text-RAG baseline: session-level hybrid retrieval + frozen reader.

PROTOCOL NOTE (differs fundamentally from the TempLAMA csr_v1 probes):
- LongMemEval-S has NO episode partition, NO as_of year, NO state_key, and NO
  oracle fields to mask. Retrieval is over the WHOLE haystack: every haystack
  session is rendered into ONE retrieval document (with a date prefix to help
  temporal-reasoning questions), queried by the question text, top-k by RRF.
  Do NOT port the episode-scoped retrieval logic from probe/ours_probe.py —
  it would be wrong here (this comment exists so nobody "fixes" it back).
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
JSON list — auto-detected): each item has at least
  question_id, question, question_type, and haystack sessions. Session field
names follow the official release (question.statement / sessions with
session_id, session_date, turns) but a per-dataset adapter is isolated in
`normalize_item` — adjust there if the schema differs; paths are NOT
hardcoded (pass --data).
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path("/share/latent2/vcrm_icae")
PROBE = ROOT / "c1_hybrid_rag"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PROBE))


def normalize_item(item):
    """Adapter: return {question_id, question, question_type, sessions:[...]}.

    sessions: list of {session_id, date, text} sorted by date ascending.
    Each turn is rendered as 'USER: ...' / 'ASSISTANT: ...'; the session text
    is the date-prefixed concatenation of its turns in original order.
    """
    qid = item.get("question_id") or item.get("qid") or item.get("id")
    qtype = item.get("question_type") or item.get("type") or ""
    question = item.get("question") or item.get("question_text") or ""
    raw_sessions = item.get("sessions") or item.get("haystack_sessions") or item.get("context") or []
    sessions = []
    for i, s in enumerate(raw_sessions):
        if isinstance(s, str):  # already-rendered fallback
            sessions.append({"session_id": s[:32] or f"s{i}", "date": "", "text": s})
            continue
        sid = s.get("session_id") or s.get("id") or f"session_{i}"
        date = s.get("session_date") or s.get("date") or s.get("timestamp") or ""
        turns = s.get("turns") or s.get("messages") or s.get("session") or []
        lines = []
        for t in turns:
            if isinstance(t, str):
                lines.append(t)
                continue
            role = (t.get("role") or t.get("speaker") or "user").upper()
            content = t.get("content") or t.get("text") or ""
            lines.append(f"{role}: {content}")
        text = f"[{date}] " + "\n".join(lines) if date else "\n".join(lines)
        sessions.append({"session_id": str(sid), "date": str(date), "text": text})
    sessions.sort(key=lambda s: s["date"])  # chronological order for the prompt
    return {"question_id": str(qid), "question": question,
            "question_type": qtype, "sessions": sessions}


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

    from retrieval import RetrievalIndex  # c1_hybrid_rag/retrieval.py (BM25+BGE+RRF)
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

    # ---- haystack-level retrieval index (NO episode scoping — see header) ----
    all_sessions, session_meta = [], {}
    for it in items:
        for s in it["sessions"]:
            key = f"{it['question_id']}::{s['session_id']}"
            if key in session_meta:
                continue
            session_meta[key] = s
            all_sessions.append({"event_id": key, "text": s["text"], "time": s["date"] or "1970-01-01"})
    print(f"[lme] retrieval docs={len(all_sessions)}", flush=True)
    retriever = RetrievalIndex(all_sessions, device="cpu",
                               bge_path=str(ROOT / "models/bge-base-en-v1.5"))

    for n, it in enumerate(todo):
        qid = it["question_id"]
        started = time.perf_counter()
        try:
            hits = retriever.retrieve(it["question"], topk=args.topk)
            hit_ids = [h["event_id"] for h in hits]
            # keep original chronological order among retrieved sessions
            chosen = {h["event_id"].split("::", 1)[1] for h in hits}
            sel = [s for s in it["sessions"] if s["session_id"] in chosen]
            prompt = render_prompt(it["question"], sel)
            hypothesis, prompt_tokens, read_seconds = read_answer(prompt)
            row = {
                "question_id": qid,
                "question": it["question"],
                "question_type": it["question_type"],
                "hypothesis": hypothesis,
                "retrieved_session_ids": hit_ids,
                "retrieved_session_dates": [session_meta[h]["date"] for h in hit_ids],
                "prompt_tokens": prompt_tokens,
                "read_seconds": read_seconds,
            }
            (out_dir / f"{qid}.json").write_text(json.dumps(row, indent=1))
            print(f"[lme] {n + 1}/{len(todo)} {qid} type={it['question_type']} "
                  f"tok={prompt_tokens} sec={read_seconds:.1f}", flush=True)
        except Exception as exc:  # noqa: BLE001 - per-sample isolation
            errors[qid] = repr(exc)
            err_path.write_text(json.dumps(errors))
            print(f"[lme] ERROR {qid}: {exc!r}", flush=True)
            continue
    print("[lme] done", flush=True)


if __name__ == "__main__":
    main()
