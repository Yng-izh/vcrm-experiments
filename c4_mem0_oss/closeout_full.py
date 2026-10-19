#!/usr/bin/env python3
"""Full-run closeout: qid reconciliation, concat c4_test.jsonl, 4-column summary.

Runs LOCALLY on the 4 shard jsonl files fetched from the server.
Contract: c4_test.jsonl must contain every test query exactly once (34,963).
Scoring columns already embedded per-row by runner (c1_scoring) — verified here
by recomputing the aggregate; no platform scores anywhere.
Usage: python3 closeout_full.py runs/full_34963_shard{0..3}of4.jsonl ...
       [--ledger-glob 'runs/*_cost_ledger.jsonl']
       [--expected-events-json FILE]

Write-coverage audit (non-blocking): cross-checks the write-phase cost ledgers
(smoke ∪ full union) against the expected event list. If neither the prepared
data nor --expected-events-json is available locally, the audit prints a SKIP
note and the rest of the closeout proceeds unchanged.
"""
import argparse
import json
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
N_QUERIES = 34963


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def write_coverage_audit(ledger_glob, expected_json):
    """Write-coverage audit over cost ledgers (smoke ∪ full union).

    Returns (report_dict, summary_excerpt) or (None, None) when the audit
    cannot run (no expected-event list available) — non-blocking by design.
    """
    ledgers = sorted(HERE.glob(ledger_glob)) if ledger_glob else []
    if not ledgers:
        print("[coverage] SKIP: no cost ledgers found "
              f"({ledger_glob!r} under {HERE}); write-coverage audit not run")
        return None, None

    # per-episode events seen in the ledgers (smoke ∪ full union)
    ledger_events = {}   # episode_id -> set(event_id)
    ledger_files = {}    # episode_id -> set(ledger file names containing it)
    for lf in ledgers:
        for line in lf.open():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("kind") == "episode_complete":
                continue
            if rec.get("event_id") is None:
                continue
            ledger_events.setdefault(rec["episode_id"], set()).add(rec["event_id"])
            ledger_files.setdefault(rec["episode_id"], set()).add(lf.name)
    episodes_with_any_ledger = set(ledger_events)

    # expected event list: explicit file first, then prepared data
    expected = {}  # episode_id -> set(event_id)
    if expected_json:
        expected = {ep_id: set(ev_ids)
                    for ep_id, ev_ids in json.loads(Path(expected_json).read_text()).items()}
        print(f"[coverage] expected events from {expected_json}")
    elif (ROOT / "data/prepared/templama_test.json").exists():
        eps = json.loads((ROOT / "data/prepared/templama_test.json").read_text())
        expected = {ep["id"]: {ev["id"] for ev in ep["events"]} for ep in eps}
        print("[coverage] expected events from data/prepared/templama_test.json")
    else:
        print("[coverage] SKIP: no --expected-events-json and prepared data not local; "
              "write-coverage audit not run (upload templama_test.json or pass "
              "--expected-events-json {episode_id: [event_id, ...]})")
        return None, None

    complete = {ep for ep, evs in expected.items() if ledger_events.get(ep, set()) >= evs}
    incomplete = {}
    for ep, evs in expected.items():
        seen = ledger_events.get(ep, set())
        missing = sorted(evs - seen)
        if missing:
            incomplete[ep] = {"expected": len(evs), "seen": len(seen),
                              "missing_event_ids": missing,
                              "in_ledger_files": sorted(ledger_files.get(ep, []))}

    # union coverage: an episode counted once even if written by BOTH the smoke
    # and the full phase (ledger glob spans both). 4038 vs 4057 note: the full
    # phase covers 4038 prepared episodes; the 19-episode gap is expected to be
    # explained by smoke-phase-only episodes present in the smoke ledger (the
    # smoke sample draws from val/sample100, not the test split) — the report
    # records both numbers so the delta is auditable instead of hand-waved.
    report = {
        "ledger_glob": ledger_glob,
        "ledger_files": [lf.name for lf in ledgers],
        "expected_episodes": len(expected),
        "episodes_with_any_ledger": len(episodes_with_any_ledger),
        "episodes_complete": len(complete),
        "episodes_incomplete": len(incomplete),
        "incomplete_detail": incomplete,
        "union_note": "episodes_with_any_ledger is the smoke∪full union; an "
                      "episode in both ledgers counts once. If "
                      "episodes_with_any_ledger - episodes_complete != 0, the "
                      "incomplete_detail lists every missing event_id.",
    }
    out = HERE / "runs" / "write_coverage.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"[coverage] expected={report['expected_episodes']} "
          f"with_any_ledger={report['episodes_with_any_ledger']} "
          f"complete={report['episodes_complete']} incomplete={report['episodes_incomplete']}"
          f" -> {out}")
    excerpt = {
        "expected_episodes": report["expected_episodes"],
        "episodes_with_any_ledger": report["episodes_with_any_ledger"],
        "episodes_complete": report["episodes_complete"],
        "episodes_incomplete": report["episodes_incomplete"],
    }
    return report, excerpt


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("shards", nargs="*", help="4 shard result jsonl files")
    ap.add_argument("--ledger-glob", default="runs/*_cost_ledger.jsonl",
                    help="glob of cost ledgers (smoke ∪ full union) relative to c4_mem0_oss/")
    ap.add_argument("--expected-events-json", default=None,
                    help="JSON file {episode_id: [event_id, ...]} with the expected event list")
    args = ap.parse_args(argv)
    shard_files = sorted(args.shards)
    assert len(shard_files) == 4, f"need 4 shard jsonl, got {len(shard_files)}"
    rows, qids = [], set()
    for f in shard_files:
        cnt = 0
        for line in Path(f).open():
            r = json.loads(line)
            assert r["query_id"] not in qids, f"duplicate qid {r['query_id']} in {f}"
            qids.add(r["query_id"])
            rows.append(r)
            cnt += 1
        print(f"[closeout] {f}: {cnt} rows md5={md5(f)}")
    print(f"[closeout] total rows={len(rows)} unique qids={len(qids)}")

    # reconcile against the prepared test split (qid list mirror if prepared
    # json is not local; the mirror itself is dumped from templama_test.json)
    prep = HERE / "runs" / "templama_test_qids.json"
    if prep.exists():
        all_q = set(json.loads(prep.read_text()))
    else:
        eps = json.loads((ROOT / "data/prepared/templama_test.json").read_text())
        all_q = {q["id"] for ep in eps for q in ep["queries"]}
    missing = all_q - qids
    extra = qids - all_q
    print(f"[closeout] prepared={len(all_q)} missing={len(missing)} extra={len(extra)}")
    assert not missing and not extra and len(qids) == N_QUERIES, "RECONCILIATION FAILED"

    # concat in shard order -> contract artifact
    out = HERE / "c4_test.jsonl"
    with out.open("w") as f:
        for r in rows:  # stable: shard0..3, original order within each
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[closeout] wrote {out.name} md5={md5(out)}")

    # errors check
    errs = {}
    for i, f in enumerate(shard_files):
        ef = Path(f).with_name(Path(f).stem + "_errors.json")
        if ef.exists():
            e = json.loads(ef.read_text())
            if e:
                errs[f] = len(e)
    print(f"[closeout] errors: {errs or 'none'}")

    # 4-column aggregate (c1_scoring columns embedded per-row by runner)
    # write-coverage audit first (non-blocking: result may be None)
    _cov_report, cov_excerpt = write_coverage_audit(args.ledger_glob, args.expected_events_json)
    n = len(rows)
    col = lambda k: sum(r[k] for r in rows) / n
    types = {}
    for r in rows:
        types.setdefault(r["type"], []).append(r)
    summary = {
        "EXTRACTION_RULE": "ex28",
        "DENOMINATOR_NOTE": f"full test: {n} queries, all scored (no exclusions)",
        "PLATFORM_SCORE_BAN": "Mem0 platform scores (94.4% LongMemEval / 92.5% LoCoMo) "
                              "are NOT cited anywhere (contract prohibition)",
        "pin_sha": rows[0]["route"]["pin_sha"],
        "writer_model": rows[0]["writer_model"],
        # protocol naming (audit): retrieval is filtered per-episode via the
        # mem0 user_id; the route is named honestly, not as a state oracle.
        "protocol": "controlled_state_routing",
        "n_scored": n, "n_errored": 0 if not errs else errs,
        "scores": {"em_raw": col("em_raw"), "f1_raw": col("f1_raw"),
                   "em_extracted": col("em_extracted"), "f1_extracted": col("f1_extracted")},
        "scores_by_type": {t: {"n": len(rs),
                               "em_raw": sum(x["em_raw"] for x in rs) / len(rs),
                               "f1_raw": sum(x["f1_raw"] for x in rs) / len(rs),
                               "em_extracted": sum(x["em_extracted"] for x in rs) / len(rs),
                               "f1_extracted": sum(x["f1_extracted"] for x in rs) / len(rs)}
                           for t, rs in types.items()},
        "retrieval": {"facts_avg": sum(len(r["retrieved_facts"]) for r in rows) / n,
                      "nonempty_rate": sum(bool(r["retrieved_facts"]) for r in rows) / n},
        "shards": {f: md5(f) for f in shard_files},
        "write_coverage": cov_excerpt if cov_excerpt is not None else {
            "status": "skipped",
            "reason": "no ledgers or no expected-event list available locally"},
    }
    out_s = HERE / "runs" / "full_run_summary.json"
    out_s.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[closeout] summary written {out_s}")
    print(json.dumps(summary["scores"], indent=2))


if __name__ == "__main__":
    sys.exit(main())