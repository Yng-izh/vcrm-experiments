# c4_mem0_oss — Mem0 OSS (v2.2.1) on TempLAMA

External baseline for the v3 main table (row 5): Mem0 open-source memory system evaluated on TempLAMA test (34,963 queries / 4,057 episodes) with a frozen C1 reader protocol.

## Results (extracted main view, EM/F1)

| Split | EM | F1 | n |
|---|---|---|---|
| historical | 0.1241 | 0.3111 | 30,906 |
| latest | 0.1935 | 0.4217 | 4,057 |
| conflict (metadata) | 0.1327 | 0.3209 | 34,407 |
| **overall** | **0.1321** | **0.3240** | **34,963** |

Raw (pre-extraction) appendix: em_raw 0.00246 / f1_raw 0.14825. Retrieval: facts_avg 3.47, nonempty 99.49%. Read latency 2.453 s/query (measured, end-to-end incl. retry). Write cost: estimated / partially observed (tiktoken fallback; mem0 add() usage not surfaced; relay billing pending) — $0.479 extrapolated from tiktoken estimates over 4,038 episodes, gpt-4o-mini via mem0 writer, far below the $200 cap.

> **SUPERSEDED**: the numbers above were produced under the pre-fix protocol
> (no as_of year injected into the read-path query; episode-level resume).
> They are kept for provenance only; the as_of-injected protocol (see
> Revisions in `CONTRACT_C4_MEM0_OSS.md`) requires a full rerun before any
> of these figures can be cited.

## Protocol: Controlled State Routing (csr_v1)

The unified protocol definition is [`PROTOCOL_CSR_V1.md`](../PROTOCOL_CSR_V1.md).
C4's as_of injection and per-episode routing already satisfy csr_v1: **csr_v1
is consistent with the as_of injection already merged into C4 — no further
code changes are required**; on rerun, use a separate output directory
(`runs_csr_v1/`) and do not overwrite the legacy `runs/`.

## Protocol (frozen)

- **Memory system**: mem0ai v2.2.1 @ 94c3fe9f238f3dbf29c9ce98643bd71eb13077cd, native retrieval top-10; embedder BGE-base-en-v1.5 (local, 768d)
- **Writer**: gpt-4o-mini (openai-compatible endpoint), mem0 extraction+consolidation pipeline
- **Reader**: frozen C1 protocol — Mistral-7B-Instruct-v0.2 (vllm), greedy, 128 new tokens, seed 42; frozen reader.py untouched (port override via `C4_VLLM_PORT` shim)
- **Extraction**: frozen ex28 rule (post-hoc deterministic extraction, identical to C1/C2)
- **Metrics**: EM/F1 on raw and extracted views; frozen c1_scoring.py; server-side `vcrm/metrics.py` is authoritative (independently recomputed in audit, agreement to 8 decimal places)
- **Denominator**: 34,963, no exclusions. conflict subset = same state_key with ≥2 distinct answer-set versions (n=34,407, identical to C1/C2 implementation)

## Protocol naming: Controlled State Routing vs End-to-End Retrieval

| | End-to-End Retrieval | Controlled State Routing (this work) |
|---|---|---|
| Write path | raw event text ingested; store owns consolidation | raw event text ingested via mem0 add() (`{id, time, text}` views only) |
| Read path | free-form query against the store; the store is the only state | query carries the episode's as_of year ("As of {as_of}, {question}"); retrieval filtered per-episode via mem0 `user_id` |
| State | implicit in the store | controlled: one store namespace per episode; no hidden state oracle touched |
| Claim | measures the memory system end-to-end | measures the memory system as a controlled per-episode state under a frozen reader protocol |

**Current C4/VCRM TempLAMA results belong to Controlled State Routing.** The
write path never touches oracle fields (contract §6); the read path uses the
query's as_of year as a legitimate query condition and the per-episode
`user_id` filter — the `route.protocol: "controlled_state_routing"` field in
each runner row names this honestly. These results must not be presented as
end-to-end memory-system retrieval numbers.

## Files

- `CONTRACT_C4_MEM0_OSS.md` — frozen experiment contract (FROZEN_MD5 0109281fd4c8be13d4019be7912897bf)
- `write_phase.py` — crash-proof/resumable 4-process memory writer (md5 6c572321…)
- `merge_qdrant_shards.py` — shard merge + verify (13,423 points, VERIFY PASSED)
- `runner.py` — Phase B reader/eval runner, 4-shard aware (per-shard history_db_path; md5 8bdcef28…)
- `reader.py` — frozen C1 reader (md5 97da5e75…, do not modify)
- `c1_scoring.py`, `selftest.py`, `gen_ast_report.py`, `closeout_full.py`, `split_write_shards.py`, `summarize_smoke.py`, `inject.py`
- `runs/` — full_run_summary.json, full_run_t8_aggregate.json, conflict qid mirror, GPU inventories, ast_scan_report (small artifacts only; full outputs on server)

## Integrity

- Concatenated output `c4_test.jsonl` (not committed, 86 MB): md5 61fbcc58aa9f76b0e4f040c0df57fc93, 34,963 rows, unique qids, 0 missing/extra vs prepared set, 0 errors
- AST audit: no oracle/answer leakage in write path (0 hits); platform-reported scores are excluded from all summaries by contract

## Reproduce (sketch)

1. Prepare TempLAMA test set (`data/prepared/templama_test.json` on server)
2. `split_write_shards.py` → `write_phase.py` ×4 (write with retry/resume, done-set) → `merge_qdrant_shards.py --shards 4 --target qdrant_data_full --verify`
3. Launch 4 vllm readers (ports 8011-8014) → `runner.py --split test --num-shards 4 --shard {0..3} --tag full_34963` with per-shard `QDRANT_PATH` copy + `MEM0_DIR` (QdrantLocal takes an exclusive flock per store dir)
4. `closeout_full.py` → concat, strict qid reconciliation, four-metric scoring
