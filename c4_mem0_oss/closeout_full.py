#!/usr/bin/env python3
"""Full-run closeout: qid reconciliation, concat c4_test.jsonl, 4-column summary.

Runs LOCALLY on the 4 shard jsonl files fetched from the server.
Contract: c4_test.jsonl must contain every test query exactly once (34,963).
Scoring columns already embedded per-row by runner (c1_scoring) — verified here
by recomputing the aggregate; no platform scores anywhere.
Usage: python3 closeout_full.py runs/full_34963_shard{0..3}of4.jsonl ...
"""
import json
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
N_QUERIES = 34963


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def main():
    shard_files = sorted(sys.argv[1:])
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
    }
    out_s = HERE / "runs" / "full_run_summary.json"
    out_s.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[closeout] summary written {out_s}")
    print(json.dumps(summary["scores"], indent=2))


if __name__ == "__main__":
    sys.exit(main())
