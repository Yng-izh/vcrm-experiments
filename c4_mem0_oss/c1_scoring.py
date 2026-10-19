"""Thin re-export shim: C4 uses C1's frozen scoring verbatim (contract §4:
vcrm/metrics.py EM/F1 + frozen ex28 extraction, "与 C1/C2/C3 相同 EXTRACTION_RULE").

Path note: C1 lives at <ROOT>/c1_hybrid_rag/scoring.py; we import it by file
location so the frozen code is never copied or reimplemented here.
"""
import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_C1_SCORING = _ROOT / "c1_hybrid_rag" / "scoring.py"
_VCRM = str(_ROOT / "vcrm")
if not (_ROOT / "vcrm").exists():
    # workspace copy lacks vcrm/ (server-only); on v100 the real root is /share/latent2/vcrm_icae
    _ROOT = Path("/share/latent2/vcrm_icae")
    _VCRM = str(_ROOT / "vcrm")
if _VCRM not in sys.path:
    sys.path.insert(0, _VCRM)   # C1 scoring.py does `from metrics import ...`
_spec = importlib.util.spec_from_file_location("c1_scoring_impl", _C1_SCORING)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["c1_scoring_impl"] = _mod
_spec.loader.exec_module(_mod)

score_query = _mod.score_query
extract_answer = _mod.extract_answer
EXTRACTION_RULE = _mod.EXTRACTION_RULE
EXTRACTION_RULE_ID = _mod._EXTRACTION_RULE_ID
DENOMINATOR_NOTE = _mod.DENOMINATOR_NOTE
aggregate = _mod.aggregate
