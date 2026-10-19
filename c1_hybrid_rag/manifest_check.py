"""Content-level manifest digest check for C1 data (captain step 2, archived in c1_hybrid_rag/).

Manifest binds digest(episodes) = sha256(json.dumps(episodes, sort_keys=True,
ensure_ascii=False)) per vcrm/vcrm/io.py::digest — NOT raw file bytes.
Verifies digest + episode/query counts per split; exits 1 on any mismatch.
"""
import hashlib, json, sys
from pathlib import Path

ROOT = Path("/share/latent2/vcrm_icae")
m = json.loads((ROOT / "data/prepared/manifest.json").read_text())
ok = True
for split in ("train", "val", "test"):
    p = ROOT / f"data/prepared/templama_{split}.json"
    d = json.loads(p.read_text())
    h = hashlib.sha256(json.dumps(d, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    exp = m["splits"][split]["sha256"]
    nq = sum(len(e["queries"]) for e in d); nev = sum(len(e["events"]) for e in d)
    match = (h == exp) and (len(d) == m["splits"][split]["episodes"]) and (nq == m["splits"][split]["queries"])
    ok &= match
    print(f"{split}: digest={h[:16]} manifest={exp[:16]} match={match} "
          f"episodes={len(d)} queries={nq} events={nev}")
# semantic spot check: every query keeps a non-empty answers list (test)
d = json.loads((ROOT / "data/prepared/templama_test.json").read_text())
bad = [q["id"] for ep in d for q in ep["queries"] if not (isinstance(q["answers"], list) and q["answers"])]
print("queries with empty answers:", len(bad))
ok &= not bad
print("RESULT:", "PASS — manifest digest binds current files (canonical-JSON digest, not file bytes)" if ok else "FAIL")
sys.exit(0 if ok else 1)
