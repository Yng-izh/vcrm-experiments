"""C1 corpus construction (CONTRACT_C1_HYBRID_RAG.md §2, frozen).

Global single corpus: pool ALL events of ALL episodes of the given templama
split file. Unit = one event {event_id, time, text}; text verbatim; no
chunking/rewriting. Build never uses answers / state_key / as_of / episode
membership for retrieval (episode attribution is post-hoc analysis only).
"""
import hashlib
import json
import statistics
import time
from pathlib import Path


def build_corpus(templama_path, out_path):
    """Returns (events, write_seconds, sha256). event_id = f"e{ep:05d}_{ev:04d}"
    (deterministic; original event id kept as `orig_id` for error analysis)."""
    t0 = time.time()
    episodes = json.loads(Path(templama_path).read_text())
    if not isinstance(episodes, list):
        raise ValueError(f"expected list of episodes, got {type(episodes).__name__}")
    events = []
    for ep_i, ep in enumerate(episodes):
        for ev_i, ev in enumerate(ep.get("events", [])):
            text, t = ev.get("text"), ev.get("time")
            if not text or not t:
                raise ValueError(f"bad event ep{ep_i} ev{ev_i}: {list(ev)}")
            events.append({
                "event_id": f"e{ep_i:05d}_{ev_i:04d}",
                "orig_id": ev["id"],
                "episode_id": ep["id"],   # post-hoc analysis only, not used by retrieval
                "time": str(t).strip(),
                "text": text,
            })
    write_seconds = time.time() - t0
    blob = json.dumps({"source": str(templama_path), "events": events}, ensure_ascii=False)
    Path(out_path).write_text(blob)
    sha = hashlib.sha256(Path(out_path).read_bytes()).hexdigest()
    return events, write_seconds, sha


def load_corpus(path):
    return json.loads(Path(path).read_text())["events"]


def corpus_stats(events):
    return {
        "n_events": len(events),
        "mean_tokens": round(statistics.mean(len(e["text"].split()) for e in events), 2),
        "bytes": Path(events if isinstance(events, (str, Path)) else "").stat().st_size
        if isinstance(events, (str, Path)) else None,
    }
