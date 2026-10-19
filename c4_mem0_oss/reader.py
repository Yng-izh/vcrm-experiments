"""Re-export shim: import C1's frozen reader.py verbatim (zero-copy discipline).

C1 lives at <ROOT>/c1_hybrid_rag/reader.py; on v100 ROOT=/share/latent2/vcrm_icae.
reader.py calls the vllm OpenAI endpoint (localhost:8001, mistral-7b, greedy/128/seed42).
"""
import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_C1_READER = _ROOT / "c1_hybrid_rag" / "reader.py"
if not _C1_READER.exists():
    _C1_READER = Path("/share/latent2/vcrm_icae/c1_hybrid_rag/reader.py")
_spec = importlib.util.spec_from_file_location("c1_reader_impl", _C1_READER)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["c1_reader_impl"] = _mod
_spec.loader.exec_module(_mod)

read_answer = _mod.read_answer

# Phase B 5-shard read: C1's frozen reader hardcodes VLLM_URL=localhost:8001.
# The shim overrides the MODULE-LEVEL constant on the loaded impl (the frozen
# function body reads the global at call time), so each shard process points
# at its own vllm port via C4_VLLM_PORT. No frozen-file edit; prompt/decoding
# params (greedy/128/seed42) untouched.
import os as _os
_port = _os.environ.get("C4_VLLM_PORT")
if _port:
    _mod.VLLM_URL = f"http://localhost:{_port}/v1/completions"
