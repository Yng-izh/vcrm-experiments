"""Re-export shim: import C1's frozen inject.py verbatim (zero-copy discipline).

C1 lives at <ROOT>/c1_hybrid_rag/inject.py; on v100 ROOT=/share/latent2/vcrm_icae.
Workspace fallback mirrors c1_scoring.py's pattern.
"""
import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_C1_INJECT = _ROOT / "c1_hybrid_rag" / "inject.py"
if not _C1_INJECT.exists():
    _C1_INJECT = Path("/share/latent2/vcrm_icae/c1_hybrid_rag/inject.py")
_spec = importlib.util.spec_from_file_location("c1_inject_impl", _C1_INJECT)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["c1_inject_impl"] = _mod
_spec.loader.exec_module(_mod)

PROMPT_TEMPLATE = _mod.PROMPT_TEMPLATE
build_prompt = _mod.build_prompt
chat_wrap = _mod.chat_wrap
