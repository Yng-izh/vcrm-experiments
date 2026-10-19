"""C1 scoring: reuses vcrm/metrics.py EM/F1 (contract §5). Adds conflict/stale
bookkeeping per v2 §3.2 rules, with explicit denominators. Adds the frozen
post-hoc answer-extraction rule (extract_answer, pure & deterministic)."""
import re
import sys
from pathlib import Path

# reuse vcrm/metrics.py — do not reimplement
_VCRM = Path(__file__).resolve().parent.parent / "vcrm"
if str(_VCRM) not in sys.path:
    sys.path.insert(0, str(_VCRM))
from metrics import exact as em_score      # noqa: E402
from metrics import f1 as f1_score         # noqa: E402

# ---- frozen post-hoc extraction rule (rule id: ex28) ------------------------
# Frozen on 481 diag: EM 0.2308 / F1 0.3943 (raw: EM 0.0042 / F1 0.1471).
# Deterministic pure function; applied post-hoc to raw_answer. Main-table
# metrics use the extracted view (Table-1 note: "post-hoc extraction").
_EXTRACTION_RULE_ID = "ex28"
_QUOTE = '\u201c\u201d\u2018\u2019\'"'


def extract_answer(raw):
    """Deterministic post-hoc extraction from a free-form reader answer.
    Pure function of `raw` — same input always yields the same output."""
    t = raw.strip()
    m = re.search(r"answer(?:\s+is|\s*:)\s*(.+)", t, re.I)
    if m:
        seg = m.group(1)
    else:
        parts = [p for p in re.split(r"(?<!\b[A-Za-z])\.(?!\w)(?:\s|$)", t) if p.strip()]
        if not parts:
            return ""
        seg = parts[0]
        # prefer a later sentence carrying the verb pattern, unless the first
        # sentence already contains "answer"
        for p in parts:
            if re.search(r"\b(?:played|plays|works|worked)\s+for\b|\bmember of\b", p, re.I) \
                    and not re.search(r"\banswer\b", seg, re.I):
                seg = p
                break
    q = re.match(r'^["' + _QUOTE + r'](.+?)["' + _QUOTE + r']', seg)
    if q:
        seg = q.group(1)
    seg = re.sub(r"^(the (most )?(recent|consistent) (fact given is that|answer( for the question)?"
                 r"( based on the (given |provided )?(historical )?facts)?( is| would be)?)"
                 r"|the (correct )?answer( to the question)? is|answer is|answer:)\s*",
                 "", seg, flags=re.I)
    seg = re.sub(r"^(In|On|At|By) \d{4}[,]?\s*", "", seg)
    # verb-object pattern first (before year-tail cleanup destroys it)
    m2 = re.search(r"\b(?:plays|works|worked|played)\s+for\s+(.+)$", seg, flags=re.I)
    if m2:
        seg = m2.group(1)
    else:
        m2b = re.search(r"\b(?:member of|part of)\s+(.+)$", seg, flags=re.I)
        if m2b:
            seg = m2b.group(1)
    # temporal tails
    seg = re.sub(r"\s*\((?:as of |in )?[\d,\s\-and]+\)\s*$", "", seg)
    seg = re.sub(r"\s+in the years? [\d,\s\-and]+$", "", seg)
    seg = re.sub(r"\s+in \d{4}((,\s*|\s+(and|to|-)\s*)\d{4})*$", "", seg)
    seg = re.sub(r"\s*\d{4}[-\u2013]present$", "", seg)
    seg = re.sub(r",?\s*\(?\d{4}( (and|to) \d{4})?\)?$", "", seg)
    seg = re.sub(r"\s+is the answer$", "", seg, flags=re.I)
    seg = re.sub(r"\s*\(Answer[^)]*\)$", "", seg, flags=re.I)
    seg = re.sub(r",? depending on.*$", "", seg, flags=re.I)
    if ". " in seg:
        seg = seg.split(". ")[0]
    seg = re.sub(r"^(that|the|a)\s+", "", seg, flags=re.I)
    # multi-employer lists: cut at ", in YYYY" / " and YYYY"
    seg = re.split(r",\s+(?:in\s+)?\d{4}", seg)[0]
    seg = re.split(r"\s+and\s+(?:in\s+)?\d{4}", seg)[0]
    seg = re.sub(r"\s+in (?:the )?(?:years?|a specific year|a certain point in time"
                 r"|the year closest.*|various years.*)$", "", seg, flags=re.I)
    # constrained relative-clause cuts ("as the ...", "that he/she/was ...")
    seg = re.split(r"\s+as\s+the\s+", seg)[0]
    seg = re.split(r"\s+that\s+(?:he|she|they|was|is|were)\b", seg)[0]
    seg = seg.strip().strip(_QUOTE).strip()
    # rstrip trailing "." only when NOT part of an abbreviation (prev char is
    # a single uppercase letter, e.g. "A.F.C.") or when prev char is alnum
    if seg.endswith(".") and len(seg) >= 2 and seg[-2].isalnum():
        prev = seg[-2]
        before = seg[-3] if len(seg) >= 3 else ""
        if not (prev.isupper() and (not before or not before.isalnum() or before == ".")):
            seg = seg[:-1]
    toks = seg.split()
    return " ".join(toks[:12]) if toks else seg


EXTRACTION_RULE = (
    "ex28: answer-is/quote marker capture; abbrev-safe sentence split with "
    "verb-pattern sentence preference; leading-prefix strip (recent/consistent/"
    "correct answer is); date-prefix strip; verb-object capture (plays/works/"
    "worked/played for, member of/part of); year-tail removal ((as of YYYY), "
    "in YYYY[ and YYYY], YYYY-present); '. ' cross-sentence cut; leading "
    "that/the/a strip; ', in YYYY' / ' and YYYY' list cuts; hedge-tail cut "
    "(depending on...); 'in years/a specific year/a certain point in time' "
    "tail cut; 'as the' / 'that he|she|they|was|is|were' relative cut; cap 12 "
    "tokens. Frozen on 481 diag: EM 0.2308 / F1 0.3943 (raw EM 0.0042 / F1 "
    "0.1471)."
)


def score_query(prediction, answers, extracted=None):
    """Returns both raw and extracted views when `extracted` is provided.
    Main-table metrics use the extracted view (post-hoc extraction)."""
    out = {
        "em_raw": float(em_score(prediction, answers)),
        "f1_raw": float(f1_score(prediction, answers)),
    }
    if extracted is not None:
        out["em_extracted"] = float(em_score(extracted, answers))
        out["f1_extracted"] = float(f1_score(extracted, answers))
    return out


def aggregate(rows, extracted=True):
    """rows: per-query dicts. extracted=True aggregates the extracted view
    (Table-1 main numbers); False aggregates the raw view (appendix)."""
    def _mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None
    hist = [r for r in rows if r["type"] == "historical"]
    latest = [r for r in rows if r.get("latest")]
    conflict = [r for r in rows if r.get("conflicting")]
    stale = [r for r in rows if r.get("wrong_version")]
    suf = "_extracted" if extracted else "_raw"
    out = {
        "n_total": len(rows),
        "extraction_rule": _EXTRACTION_RULE_ID if extracted else "raw",
        "hist_n": len(hist),
        "hist_f1": sum(r["f1" + suf] for r in hist) / len(hist) if hist else None,
        "hist_em": sum(r["em" + suf] for r in hist) / len(hist) if hist else None,
        "latest_n": len(latest),
        "latest_em": sum(r["em" + suf] for r in latest) / len(latest) if latest else None,
        "conflict_n": len(conflict),
        "conflict_f1": sum(r["f1" + suf] for r in conflict) / len(conflict) if conflict else None,
        "stale_n": len(stale),
        "stale_rate": sum(r["em" + suf] for r in stale) / len(stale) if stale else None,
        "mean_prompt_tokens": sum(r["prompt_tokens"] for r in rows) / len(rows),
        # rows from the concurrent shard runner have read_seconds=None (fan-out
        # batch timing not attributable per row); average over recorded rows only
        "mean_read_seconds": _mean([r["read_seconds"] for r in rows
                                    if r.get("read_seconds") is not None]),
    }
    out["denominator_note"] = DENOMINATOR_NOTE
    return out


# Contract §5: denominators and conflict definition must be explicit in every
# summary. Written at t1; applies unchanged to the full-test run (t5).
DENOMINATOR_NOTE = (
    "Denominators (contract §5): hist_n = all rows with type=historical. "
    "conflict_n uses the METADATA definition: a query is conflicting iff its "
    "answer set differs from at least one other version of the same state_key "
    "(equivalently on TempLAMA: historical_answers != current_answers; on the "
    "481 diag this = 474/481, 7 queries have identical answers across all "
    "versions). The retrieval-side flags definition (as_of < latest as_of for "
    "the key AND not latest; = 481 on the diag) is appendix-only. latest_n on "
    "the full test = 4,057 (0 on the 481 diag, so latest EM is N/A there). "
    "Main-table metrics use the extracted view (post-hoc rule ex28, frozen on "
    "the 481 diag; not re-tuned on full-test results); raw view is appendix."
)
