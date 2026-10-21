import json, glob
from pathlib import Path
from collections import Counter

ROOT = Path("/share/latent2/vcrm_icae"); RUNS = ROOT / "probe/runs"
R = {}
for p in sorted(glob.glob(str(RUNS / "ours_v6_34963_shard*.jsonl"))):
    for l in open(p):
        if l.strip():
            r = json.loads(l); R[r["query_id"]] = r
ref = {}
for l in open(ROOT / "c4_mem0_oss/runs_csr_v1/c4full_merged.jsonl"):
    if l.strip():
        r = json.loads(l); ref[r["query_id"]] = {"conflicting": r.get("conflicting")}
c1 = {}
for l in open(ROOT / "c1_hybrid_rag/runs_csr_v1/c1_test.jsonl"):
    if l.strip():
        r = json.loads(l); c1[r["query_id"]] = r

def m(rs, k):
    return round(sum(float(x[k]) for x in rs) / len(rs), 4) if rs else None

def b(rs):
    return "n=%d em=%s f1=%s" % (len(rs), m(rs, "em_extracted"), m(rs, "f1_extracted"))

A = list(R.values())
print("rows", len(A), "unique", len(R), "missing", len(set(ref) - set(R)), "extra", len(set(R) - set(ref)))
print("empty_extracted", sum(1 for r in A if not str(r.get("extracted_answer", "")).strip()))
print("route.path", dict(Counter(r["route"].get("path") for r in A)))
lat = [r for r in A if r["route"].get("path") == "latent"]
fb = [r for r in A if r["route"].get("path") == "fallback_text"]
for name, rs in (("ALL", A), ("LATENT", lat), ("FALLBACK", fb)):
    print("--- %s %s" % (name, b(rs)))
    for t in ("historical", "latest"):
        print("    %s: %s" % (t, b([x for x in rs if x["type"] == t])))
    print("    conflict: %s" % b([x for x in rs if ref.get(x["query_id"], {}).get("conflicting") is True]))
print("mean_prompt_tokens", round(sum(r["prompt_tokens"] for r in A) / len(A), 2),
      "mean_read_s", round(sum(r["read_seconds"] for r in A) / len(A), 4))
latok = [r for r in A if r["type"] == "latest" and r["route"].get("path") == "latent"]
print("latest latent is_current", dict(Counter(r["route"].get("is_current") for r in latok)))
cm = [q for q in R if q in c1]
print("paired vs C1: ours", m([R[q] for q in cm], "em_extracted"), m([R[q] for q in cm], "f1_extracted"),
      "c1", m([c1[q] for q in cm], "em_extracted"), m([c1[q] for q in cm], "f1_extracted"))
w = sum(1 for q in cm if R[q]["f1_extracted"] > c1[q]["f1_extracted"])
l = sum(1 for q in cm if R[q]["f1_extracted"] < c1[q]["f1_extracted"])
print("f1 win/loss/tie", w, l, len(cm) - w - l)
print("empty rows:", [(r["query_id"], r["raw_answer"][:40], r["route"].get("path")) for r in A if not str(r.get("extracted_answer", "")).strip()])
