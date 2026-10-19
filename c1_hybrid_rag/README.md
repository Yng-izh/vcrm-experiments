# c1_hybrid_rag — C1 frozen protocol dependency files (mirrored from server)

This directory mirrors the C1 frozen protocol implementation from
`v100-2:/share/latent2/vcrm_icae/c1_hybrid_rag/` so the thin-wrapper
experiments (c4_mem0_oss etc.) can be independently audited in this repo
without server access. Files are copied verbatim — never modified or run here.

## Mirrored files and md5 (2026-10-19, scp from v100-2, byte-identical verified)

| File | md5 | Role |
|---|---|---|
| `reader.py` | `a6d91d2ba5b61249101c130d3c5743fa` | Frozen C1 reader (vllm Mistral-7B, greedy/128/seed42) |
| `scoring.py` | `1083dc06b91f7b8d3d6b08e0fe811218` | Frozen C1 scoring (EM/F1 + ex28 extraction) |
| `inject.py` | `85b29420e4823e2182faebb3842fc983` | Frozen C1 PROMPT_TEMPLATE / build_prompt / chat_wrap |

md5 above computed locally (`md5`) and on the server (`md5sum`); both sides
agree for all three files. Re-verify with:

```
md5 reader.py scoring.py inject.py   # macOS (or md5sum on Linux)
```

Any future protocol change on the server must be re-mirrored with a new
Revision entry in the consuming experiment contract.
