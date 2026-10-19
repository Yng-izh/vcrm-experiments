#!/usr/bin/env python3
"""Build C4 smoke-20 summary + 10-query human-audit sample (contract §4/§5, §9 rev-6).

Reads runs/smoke20.jsonl + runs/smoke20_cost_summary.json; emits:
  runs/smoke20_summary.json  — EXTRACTION_RULE, DENOMINATOR_NOTE, writer model,
                               per-query realized writer cost (ledger actuals),
                               4-column EM/F1 scores, retrieval stats
  runs/smoke20_audit10.json  — 10-query human-audit sample (stratified 8h/2l,
                               includes raw/extracted answers + retrieved facts)
Run anywhere (no server deps): python3 summarize_smoke.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"

EXTRACTION_RULE = "ex28"          # frozen C1 extraction (contract §4)
DENOMINATOR_NOTE = "smoke20: 20 queries (seed42, hist16/latest4 from sample100, §9 rev-6); " \
                   "denominator for all rates = 20 minus errored queries; full-run denominators " \
                   "remain 4,057/30,906/34,407 per contract §4 (smoke does not redefine them)"
PLATFORM_SCORE_BAN = "Mem0 platform scores (94.4% LongMemEval / 92.5% LoCoMo) are NOT cited " \
                     "anywhere in this summary (contract prohibition)"


def main():
    rows = [json.loads(l) for l in (RUNS / "smoke20.jsonl").open() if l.strip()]
    errs = json.loads((RUNS / "smoke20_errors.json").read_text()) if (RUNS / "smoke20_errors.json").exists() else {}
    cost = json.loads((RUNS / "smoke20_cost_summary.json").read_text()) if (RUNS / "smoke20_cost_summary.json").exists() else {}
    ledger = [json.loads(l) for l in (RUNS / "smoke20_cost_ledger.jsonl").open()] \
        if (RUNS / "smoke20_cost_ledger.jsonl").exists() else []

    n = len(rows)
    n_err = len(errs)
    denom = n  # scored denominator = successfully completed rows
    col = lambda k: sum(r[k] for r in rows) / denom if denom else None
    types = {}
    for r in rows:
        types.setdefault(r["type"], []).append(r)

    summary = {
        "EXTRACTION_RULE": EXTRACTION_RULE,
        "DENOMINATOR_NOTE": DENOMINATOR_NOTE,
        "PLATFORM_SCORE_BAN": PLATFORM_SCORE_BAN,
        "pin_sha": rows[0]["route"]["pin_sha"] if rows else None,
        "writer_model": rows[0]["writer_model"] if rows else None,
        "search_topk": rows[0]["route"]["search_topk"] if rows else None,
        "n_selected": 20, "n_scored": n, "n_errored": n_err,
        "errors": errs,
        "scores": {
            "em_raw": col("em_raw"), "f1_raw": col("f1_raw"),
            "em_extracted": col("em_extracted"), "f1_extracted": col("f1_extracted"),
        },
        "scores_by_type": {
            t: {"n": len(rs),
                "em_raw": sum(r["em_raw"] for r in rs) / len(rs),
                "f1_raw": sum(r["f1_raw"] for r in rs) / len(rs),
                "em_extracted": sum(r["em_extracted"] for r in rs) / len(rs),
                "f1_extracted": sum(r["f1_extracted"] for r in rs) / len(rs)}
            for t, rs in types.items()},
        "retrieval": {
            "facts_avg": sum(len(r["retrieved_facts"]) for r in rows) / denom if denom else 0,
            "nonempty_rate": sum(bool(r["retrieved_facts"]) for r in rows) / denom if denom else 0,
        },
        "read_latency_s_avg": sum(r["read_seconds"] for r in rows) / denom if denom else 0,
        "writer_cost": {
            "episodes_written": cost.get("episodes_done"),
            "cost_usd_smoke_write": cost.get("cost_usd"),
            "full_run_extrapolation_usd": cost.get("full_run_extrapolation_usd"),
            "budget_usd": cost.get("budget_usd"),
            "usage_provenance": sorted({l.get("usage_provenance") for l in ledger}) or ["unknown"],
            "per_query_realized_usd": (sum(l["cost_usd"] for l in ledger) / denom) if denom and ledger else None,
            "note": "smoke write cost covers the 20 sampled queries' episodes only; "
                    "per-query write cost = smoke write cost / 20 (upper bound: episodes "
                    "are shared across queries in full run)",
        },
    }
    (RUNS / "smoke20_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print("[summary] written runs/smoke20_summary.json")

    # 10-query human-audit sample: first 8 historical + first 2 latest (order-preserving)
    hist = [r for r in rows if r["type"] == "historical"]
    lat = [r for r in rows if r["type"] != "historical"]
    audit = hist[:8] + lat[:2]
    (RUNS / "smoke20_audit10.json").write_text(json.dumps({
        "purpose": "human spot-check sample (contract §5)",
        "n": len(audit),
        "queries": [{"query_id": r["query_id"], "type": r["type"],
                     "question": r["question"], "ground_truth": r["answers"],
                     "raw_answer": r["raw_answer"], "extracted_answer": r["extracted_answer"],
                     "em_raw": r["em_raw"], "f1_raw": r["f1_raw"],
                     "em_extracted": r["em_extracted"], "f1_extracted": r["f1_extracted"],
                     "retrieved_facts": [f["memory"] for f in r["retrieved_facts"]][:5]}
                    for r in audit]}, indent=2, ensure_ascii=False))
    print(f"[summary] written runs/smoke20_audit10.json ({len(audit)} queries)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
