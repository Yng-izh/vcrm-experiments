#!/usr/bin/env python3
"""Unit-testable pieces of C4 runner (contract §4/§6), no server/network needed.

Covers: ingestion_view oracle isolation, row schema vs contract §4, AST audit
report regeneration, smoke sample integrity, extraction/score shim parity with C1.
Run: python3 selftest.py
"""
import ast
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


def test_ingestion_view():
    src = (HERE / "write_phase.py").read_text()
    tree = ast.parse(src)
    # write_phase must reference only {id,time,text} keys on event dicts
    string_consts = {n.value for n in ast.walk(tree)
                     if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    forbidden = {"answers", "as_of", "latest", "state_key"}
    leaks = forbidden & string_consts
    check("write_phase zero oracle constants", not leaks, str(leaks or ""))


def test_row_schema():
    src = (HERE / "runner.py").read_text()
    required = ["query_id", "episode_id", "question", "type", "answers",
                "prompt_tokens", "raw_answer", "extracted_answer",
                "em_raw", "f1_raw", "em_extracted", "f1_extracted",
                "read_seconds", "retrieved_facts", "writer_model",
                "writer_tokens", "route", "shard"]
    missing = [k for k in required if f'"{k}"' not in src]
    check("runner row schema = contract §4", not missing, str(missing))


def test_route_pin():
    src = (HERE / "runner.py").read_text()
    check("route.status=mem0_oss", '"status": "mem0_oss"' in src)
    check("pin sha present", "94c3fe9f238f3dbf29c9ce98643bd71eb13077cd" in src)


def test_smoke_sample():
    p = ROOT / "probe/runs/sample100.json"
    if not p.exists():
        # server-only artifact (data/templama_test.json not in workspace);
        # runner.py enforces seed=42 + n at load time. SKIP locally.
        print("[SKIP] smoke sample — server-side artifact, runner validates at load")
        return
    s = json.loads(p.read_text())
    check("sample seed=42", s["seed"] == 42)
    check("sample n=100", len(s["ids"]) == 100, f"n={len(s['ids'])}")
    check("sample ids unique", len(set(s["ids"])) == len(s["ids"]))


def test_writer_config():
    src = (HERE / "write_phase.py").read_text()
    check("writer = gpt-4o-mini (§9 rev-5)", '"gpt-4o-mini"' in src and "deepseek" not in src)
    check("base_url = ccode relay", "https://api.ccode.vip/v1" in src)
    check("embedder CPU", '"device": "cpu"' in src)
    check("key from env only", 'os.environ["DEEPSEEK_API_KEY"]' in src
          and "sk-" not in src.replace("sk-placeholder-not-real", ""))


def test_budget_guards():
    src = (HERE / "write_phase.py").read_text()
    check("USD200 budget const", "BUDGET_USD = 200.0" in src)
    check("50% early alert (§9 rev-7)", "ALERT_FRAC = 0.5" in src and "ALERT" in src)
    check("extrapolation check", "full_run_extrapolation_usd" in src)


def test_smoke20():
    r = (HERE / "runner.py").read_text()
    check("smoke=20 (§9 rev-6)", "hist[:16] + latest_q[:4]" in r)
    check("smoke tag smoke20", '"smoke20"' in r)
    check("no deepseek remnant", "deepseek" not in r)


def test_ast_report():
    p = HERE / "runs/ast_scan_report.json"
    if not p.exists():
        check("ast_scan_report.json exists", False)
        return
    r = json.loads(p.read_text())
    check("ast audit verdict pass", r["verdict"] == "pass")
    check("ast audit scope covers write path",
          "write_phase.py" in r["scope"] and "runner.py" in r["scope"])


def test_reader_scoring_verbatim():
    # scoring must be C1's code via import shim, not a copy
    shim = (HERE / "c1_scoring.py").read_text()
    check("scoring is import shim (no copy)", "importlib.util.spec_from_file_location" in shim
          and "def extract_answer" not in shim)
    r = (HERE / "runner.py").read_text()
    check("reader via C1 read_answer", "c1_reader.read_answer" in r)
    check("prompt via C1 build_prompt", "build_prompt" in r)


if __name__ == "__main__":
    test_ingestion_view()
    test_row_schema()
    test_route_pin()
    test_smoke_sample()
    test_smoke20()
    test_writer_config()
    test_budget_guards()
    test_ast_report()
    test_reader_scoring_verbatim()
    print(f"\n{'ALL PASS' if not FAILURES else f'FAILURES: {FAILURES}'}")
    sys.exit(1 if FAILURES else 0)
