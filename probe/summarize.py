"""Probe summary + route audit + C1 record-only comparison."""
import json
import glob
import sys
from collections import Counter
from pathlib import Path

ROOT = Path("/share/latent2/vcrm_icae")
RUNS = ROOT / "probe/runs"

rows = [json.loads(l) for l in (RUNS / "dry_5.jsonl").open()]
errors = json.loads((RUNS / "dry_5_errors.json").read_text()) if (RUNS / "dry_5_errors.json").exists() else {}


def mean(rs, k):
    return round(sum(r[k] for r in rs) / len(rs), 4) if rs else None


hist = [r for r in rows if r["type"] == "historical"]
latest = [r for r in rows if r["type"] == "latest"]
ok = [r for r in rows if r["route"]["status"] == "ok"]
no_assoc = [r for r in rows if r["route"]["status"] == "no_association"]
evicted = [r for r in rows if r["route"]["status"] == "version_evicted"]

# --- cross-as_of audit: same state, different t must select different version (captain fix:
# judge on selected_id, not the reading set [current]+[historical] which overlaps for adjacent versions)
by_state = {}
for r in ok:
    by_state.setdefault(r["route"]["state"], []).append(r)
groups = []
for state, rs in by_state.items():
    ts = {}
    for r in rs:
        ts.setdefault(r["t"], r["route"].get("selected_id"))
    if len(ts) >= 2:
        distinct = len(set(ts.values())) == len(ts)
        groups.append({"state": state, "n_times": len(ts),
                       "distinct_reading": distinct, "t_values": sorted(ts),
                       "selected_per_t": ts,
                       "note": "identical selected_id with different t means the attribute has no later update within horizon; legal behavior, reported as is_current_true"})
audit = {"route_status_counts": dict(Counter(r["route"]["status"] for r in rows)),
         "cross_asof_groups": groups,
         "groups_with_distinct_reading": sum(g["distinct_reading"] for g in groups),
         "groups_total": len(groups),
         "note": "states sampled once per as_of show single-time groups; distinct_reading verified where >=2 times present"}

# --- C1 record-only same-query comparison ---
c1 = {}
for p in glob.glob(str(ROOT / "c1_hybrid_rag/runs/c1_test*.shard*.jsonl")):
    for l in open(p):
        row = json.loads(l)
        c1[row["query_id"]] = row
paired = []
for r in rows:
    c = c1.get(r["query_id"])
    if c:
        paired.append({"query_id": r["query_id"], "type": r["type"],
                       "ours_f1": r["f1_extracted"], "c1_f1": c["f1_extracted"],
                       "ours_em": r["em_extracted"], "c1_em": c["em_extracted"]})
delta = (round(mean(paired, "ours_f1") - mean(paired, "c1_f1"), 4)
         if paired else None)

summary = {
    "n_expected": 100, "n_rows": len(rows), "n_errors": len(errors), "errors": errors,
    "route": {"ok": len(ok), "no_association": len(no_assoc),
              "version_evicted": len(evicted),
              "route_hit_rate": round(len(ok) / len(rows), 4),
              "no_association_rate": round(len(no_assoc) / len(rows), 4),
              "version_evicted_rate": round(len(evicted) / len(rows), 4)},
    "metrics": {
        "hist": {"n": len(hist), "em": mean(hist, "em_extracted"), "f1": mean(hist, "f1_extracted")},
        "latest": {"n": len(latest), "em": mean(latest, "em_extracted")},
        "overall_em": mean(rows, "em_extracted"), "overall_f1": mean(rows, "f1_extracted"),
        "mean_prompt_tokens": round(sum(r["prompt_tokens"] for r in rows) / len(rows), 2),
        "read_seconds_measured": True},
    "no_association_subset": {"n": len(no_assoc), "em": mean(no_assoc, "em_extracted"),
                              "f1": mean(no_assoc, "f1_extracted"),
                              "note": "time-filtered RRF fallback (e.time <= t); subset reported separately per contract §3.2"},
    "c1_record_only": {"n_paired": len(paired),
                       "ours_mean_f1": mean(paired, "ours_f1") if paired else None,
                       "c1_mean_f1": mean(paired, "c1_f1") if paired else None,
                       "delta_f1": delta,
                       "note": "record-only; no tuning against C1"},
    "route_audit": audit,
    "reference_footnotes": {
        "hist_upper": "E3 historical_f1 0.9237 (oracle routing, post-hoc val, F1口径)",
        "latest_upper": "baseline_D latest_accuracy 0.9244 (accuracy口径, n=172)"},
    "sampling": json.loads((RUNS / "sample100.json").read_text()) | {"realized_types": dict(Counter(r["type"] for r in rows))},
}
(RUNS / "dry_summary.json").write_text(json.dumps(summary, indent=1))
(RUNS / "dry_route_audit.json").write_text(json.dumps(audit, indent=1))
(RUNS / "dry_c1_paired.json").write_text(json.dumps(paired, indent=1))
print(json.dumps({k: summary[k] for k in ("route", "metrics", "c1_record_only")}, indent=1))
print("audit groups distinct:", audit["groups_with_distinct_reading"], "/", audit["groups_total"])
