"""Merge 8 Ours shards -> full-run metrics, consistent with C1-C4 report definitions."""
import json, glob
from collections import Counter
from pathlib import Path

ROOT = Path("/share/latent2/vcrm_icae")
RUNS = ROOT / "probe/runs"

rows = []
for p in sorted(glob.glob(str(RUNS / "ours_full_34963_shard*.jsonl"))):
    n = 0
    for l in open(p):
        l = l.strip()
        if not l:
            continue
        rows.append(json.loads(l))
        n += 1
    print(f"{Path(p).name}: {n}")

# dedup by query_id (keep last)
by_qid = {}
for r in rows:
    by_qid[r["query_id"]] = r
print("raw rows:", len(rows), "unique qids:", len(by_qid))

# reference ids / metadata from C4 merged (same test split, has conflicting/latest flags)
ref = {}
for l in open(ROOT / "c4_mem0_oss/runs_csr_v1/c4full_merged.jsonl"):
    l = l.strip()
    if not l:
        continue
    r = json.loads(l)
    ref[r["query_id"]] = {"conflicting": r.get("conflicting"), "latest": r.get("latest"), "type": r.get("type")}
print("ref qids:", len(ref))

missing = sorted(set(ref) - set(by_qid))
extra = sorted(set(by_qid) - set(ref))
print("missing:", len(missing), "extra:", len(extra))
if missing[:5]:
    print("missing sample:", missing[:5])
if extra[:5]:
    print("extra sample:", extra[:5])

# type agreement check
mismatch = [q for q in by_qid if q in ref and (by_qid[q]["type"] == "latest") != bool(ref[q]["latest"])]
print("type/latest mismatches:", len(mismatch))

# empty answer check
empty = sum(1 for r in by_qid.values() if not str(r.get("extracted_answer", "")).strip())
print("empty extracted_answer rows:", empty)

R = list(by_qid.values())


def mean(rs, k):
    return round(sum(float(r[k]) for r in rs) / len(rs), 4) if rs else None


def block(rs):
    return {"n": len(rs), "em": mean(rs, "em_extracted"), "f1": mean(rs, "f1_extracted")}


hist = [r for r in R if r["type"] == "historical"]
latest = [r for r in R if r["type"] == "latest"]
conf = [r for r in R if r.get("query_id") in ref and ref[r["query_id"]]["conflicting"] is True]
nonconf = [r for r in R if r.get("query_id") in ref and ref[r["query_id"]]["conflicting"] is False]

status = Counter(r.get("route", {}).get("status") for r in R)
ok = [r for r in R if r.get("route", {}).get("status") == "ok"]
noassoc = [r for r in R if r.get("route", {}).get("status") == "no_association"]

summary = {
    "tag": "ours_full",
    "n_rows": len(R),
    "protocol_note": "Ours = label-free routing probe; protocol recorded per row via route.status (csr_v1 control state routing)",
    "overall": block(R),
    "hist": block(hist),
    "latest": block(latest),
    "conflict": block(conf),
    "nonconflict": block(nonconf),
    "route": {
        "counts": dict(status),
        "ok_rate": round(len(ok) / len(R), 4),
        "no_association_rate": round(len(noassoc) / len(R), 4),
    },
    "no_association_subset": block(noassoc),
    "ok_subset": block(ok),
    "efficiency": {
        "mean_prompt_tokens": round(sum(r["prompt_tokens"] for r in R) / len(R), 2),
        "mean_read_seconds": round(sum(r["read_seconds"] for r in R) / len(R), 4),
    },
    "is_current_rate": round(sum(1 for r in ok if r["route"].get("is_current")) / len(ok), 4) if ok else None,
}
(RUNS / "ours_full_summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1, ensure_ascii=False))
