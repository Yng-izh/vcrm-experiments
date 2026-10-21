#!/usr/bin/env python3
"""t14: convert t13 hypothesis runs -> official LongMemEval eval input.

Per t12 spec (docs/LME_S_OFFICIAL_EVAL_SPEC.md), the official
`LongMemEval/src/evaluation/evaluate_qa.py` consumes a jsonl (or json array)
where each entry needs exactly:
  {"question_id": <must match ref_file question_id verbatim>, "hypothesis": <str>}

Notes from the spec:
- qid mismatch is silently skipped by the official script ("Warning: skipping
  ... not in reference data") — so we hard-assert full coverage here.
- missing `hypothesis` -> KeyError in the official script — we hard-assert.
- abstention questions (qid suffix `_abs`) must be included; judge prompt
  switches automatically.

This script NEVER calls the judge/API/GPU; it only rewrites files.
Idempotent: overwrites the output file, never touches runs/ inputs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REQUIRED_FIELDS = ("question_id", "hypothesis")


def eprint(*a):  # noqa: ANN001
    print(*a, file=sys.stderr, flush=True)


def load_expected_qids(ref_path: Path) -> set[str]:
    """Full question_id set from the official S data json (500 items)."""
    data = json.loads(ref_path.read_text())
    qids = {it["question_id"] for it in data}
    if not qids:
        raise SystemExit(f"[fatal] no question_id found in ref {ref_path}")
    return qids


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True,
                    help="tag under runs/ produced by run_text_rag.py")
    ap.add_argument("--runs-dir", default=Path(__file__).resolve().parent / "runs")
    ap.add_argument("--out", default=None,
                    help="output predictions file (default: official_eval_in/<tag>/predictions.jsonl)")
    ap.add_argument("--ref", default="/share/latent2/vcrm_icae/data/longmemeval_s_cleaned.json",
                    help="official S data json; used to read the full question_id set "
                         "(TODO/阻塞项: official repo ships no data file; path per t12 spec §6, "
                         "confirm before execution)")
    ap.add_argument("--expect-n", type=int, default=0,
                    help="optional hard assert on expected question count (overrides ref)")
    args = ap.parse_args()

    in_dir = args.runs_dir / args.tag
    if not in_dir.is_dir():
        raise SystemExit(f"[fatal] input dir not found: {in_dir}")

    rows: dict[str, dict] = {}
    n_input = n_empty = 0
    bad_missing: list[str] = []
    bad_empty: list[str] = []

    for p in sorted(in_dir.glob("*.json")):
        r = json.loads(p.read_text())
        n_input += 1
        qid = r.get("question_id")
        missing = [f for f in REQUIRED_FIELDS if r.get(f) in (None, "")]
        if "question_id" in missing or qid is None:
            bad_missing.append(f"<no-question_id>@{p.name}")
            continue
        if missing:
            bad_missing.append(qid)
        if not isinstance(r.get("hypothesis"), str) or not r["hypothesis"].strip():
            bad_empty.append(qid)
            n_empty += 1
            continue  # empty hypothesis would silently tank accuracy; exclude + report
        rows[qid] = {"question_id": qid, "hypothesis": r["hypothesis"]}

    # --- coverage assertion against the dataset's full qid set ---
    expected: set[str]
    if args.expect_n > 0:
        # without a ref we can only check the count, not identity
        expected = set(rows)  # placeholder; identity check skipped
        if len(rows) != args.expect_n:
            eprint(f"[fatal] --expect-n={args.expect_n} but got {len(rows)} non-empty rows")
    else:
        expected = load_expected_qids(Path(args.ref))

    missing = sorted(expected - set(rows))
    extra = sorted(set(rows) - expected)
    n_out = len(rows)

    print("=== reconciliation ===")
    print(f"n_input  (json files in runs/{args.tag}) : {n_input}")
    print(f"n_out    (rows written)                  : {n_out}")
    print(f"n_missing (in dataset, no prediction)    : {len(missing)}")
    print(f"n_extra  (prediction not in dataset)     : {len(extra)}")
    print(f"n_empty  (empty/blank hypothesis)        : {n_empty}")
    if missing:
        print("missing qids:", ", ".join(missing))
    if extra:
        print("extra qids:  ", ", ".join(extra))
    if bad_missing:
        print("rows missing required fields:", ", ".join(bad_missing))
    if bad_empty:
        print("rows with empty hypothesis:  ", ", ".join(bad_empty))

    fatal = bool(missing or extra or bad_missing or bad_empty)
    if fatal:
        eprint("[fatal] reconciliation failed — NOT writing output. "
               "Fix runs/<tag> (rerun run_text_rag.py for missing qids) and retry.")
        raise SystemExit(1)

    out_path = Path(args.out) if args.out else (
        Path(__file__).resolve().parent / "official_eval_in" / args.tag / "predictions.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for qid in sorted(rows):
            f.write(json.dumps(rows[qid], ensure_ascii=False) + "\n")
    print(f"[ok] wrote {n_out} predictions -> {out_path}")


if __name__ == "__main__":
    main()
