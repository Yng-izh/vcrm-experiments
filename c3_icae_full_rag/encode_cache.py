"""C3 step 2: offline segment encode cache (CONTRACT_C3_ICAE_FULL_RAG.md §2).

Encodes every episode's segments with the official ICAE checkpoint via the
audited vcrm/reader.FrozenReader path (same tokenizer as the ICAE encoder).
Cache key = sha256(segment_text) + "_" + sha256(checkpoint file)[:16].
Value: FP16 tensor [memory_tokens, d_model] saved at
c3_icae_full_rag/segment_cache/<key>.pt (atomic tmp+rename).
Idempotent: existing keys are skipped, so interrupted runs resume safely.

Label-free: reads only ep["id"] and ep["events"][*]["text"/"time"]. Oracle
fields (answers/state_key/as_of/latest) are never read (AST-audited in
run_probe.py; this file's read path uses an explicit allowlist).

Write Cost = wall time of the encoding loop (contract §4 cost column),
recorded in cache_meta.json together with the checkpoint/vcrm-reader sha256
invalidation chain.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path("/share/latent2/vcrm_icae")
C3 = ROOT / "c3_icae_full_rag"
CACHE_DIR = C3 / "segment_cache"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(C3))

FORBIDDEN_ATTRS = {"state_key", "as_of", "answers", "latest"}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(t):
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def collect_plans(tok):
    """Re-derive segment plans from segment_stats-consistent logic: same
    sorted-events order and segment_episode packing as step 1."""
    from segment import segment_episode
    episodes = json.loads((ROOT / "data/prepared/templama_test.json").read_text())
    plans = []
    for ep in episodes:
        assert not ({"answers", "as_of", "latest"} & set(ep)), "oracle field at episode level"
        events = sorted(ep["events"], key=lambda e: str(e["time"]))
        for ev in events:
            assert not ({"answers", "as_of", "latest"} & set(ev)), "oracle field in event"
        segments, oversize = segment_episode(events, tok)
        if oversize:
            raise SystemExit(f"[cache] oversize events in {ep['id']}: run invalid per contract §1.2")
        plans.append({"episode_id": ep["id"], "segments": segments})
    return plans


def main():
    import torch
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(ROOT / "models/Mistral-7B-Instruct-v0.2"))
    plans = collect_plans(tok)
    unique = sorted({s for p in plans for s in p["segments"]})
    ckpt = ROOT / "models/icae/mistral_7b_ft_icae.safetensors"
    ckpt_sha = sha256_file(ckpt)
    keys = {s: sha256_text(s) + "_" + ckpt_sha[:16] for s in unique}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    todo = [s for s in unique if not (CACHE_DIR / f"{keys[s]}.pt").exists()]
    print(f"[cache] episodes={len(plans)} unique_segments={len(unique)} "
          f"cached={len(unique) - len(todo)} todo={len(todo)}", flush=True)

    encode_seconds = 0.0
    if todo:
        from vcrm.reader import FrozenReader
        config = json.loads((ROOT / "config.json").read_text())
        if sha256_file(ROOT / "upstream/icae/code/icae_v2/modeling_icae_multi_span.py") \
                != config["icae_source_sha256"]:
            raise SystemExit("[cache] ICAE source hash mismatch")
        reader = FrozenReader(config)
        t0 = time.time()
        for n, s in enumerate(todo):
            mem = reader.encode(s)  # FP16 [mem_size, dim] on CPU, <=5120 tokens enforced
            # contract §1.2: each segment -> exactly 128 slots. The frozen reader
            # internally splits at compute_num_segments (~512 tok/128 slots), so a
            # cacheable segment must be <= that internal width; assert, never crop.
            assert mem.shape[0] == 128, f"segment produced {mem.shape[0]} slots, expected 128"

            k = keys[s]
            tmp = CACHE_DIR / f".{k}.tmp"
            torch.save(mem, tmp)
            tmp.rename(CACHE_DIR / f"{k}.pt")
            if (n + 1) % 200 == 0 or n + 1 == len(todo):
                el = time.time() - t0
                print(f"[cache] {n + 1}/{len(todo)} elapsed {el:.0f}s "
                      f"eta {el / (n + 1) * (len(todo) - n - 1) / 60:.1f}min", flush=True)
        encode_seconds = time.time() - t0
        del reader
        torch.cuda.empty_cache()

    meta = {
        "contract": "CONTRACT_C3_ICAE_FULL_RAG.md",
        "checkpoint": str(ckpt),
        "checkpoint_sha256": ckpt_sha,
        "vcrm_reader_sha256": sha256_file(ROOT / "vcrm/reader.py"),
        "icae_source_sha256": json.loads((ROOT / "config.json").read_text())["icae_source_sha256"],
        "tokenizer": "models/Mistral-7B-Instruct-v0.2",
        "dtype": "float16",
        "slots_per_segment": json.loads((ROOT / "config.json").read_text())["memory_tokens"],
        "unique_segments": len(unique),
        "newly_encoded": len(todo),
        "encode_seconds": round(encode_seconds, 1),
        "cache_bytes": sum(p.stat().st_size for p in CACHE_DIR.glob("*.pt")),
    }
    (C3 / "cache_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[cache] done: {json.dumps({k: meta[k] for k in ('unique_segments', 'newly_encoded', 'encode_seconds', 'cache_bytes')})}")


if __name__ == "__main__":
    main()
