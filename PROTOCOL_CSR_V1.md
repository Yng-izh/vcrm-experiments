# PROTOCOL_CSR_V1 — Controlled State Routing rev1 (frozen definition)

Protocol ID: `csr_v1` (Controlled State Routing rev1).
Status: code changes complete + dryrun; **no experiments run yet** under this
protocol. All pre-`csr_v1` results are SUPERSEDED (see below).

## 1. Definition

**Controlled State Routing (csr_v1)** is a controlled evaluation protocol for
temporal QA over versioned fact groups (TempLAMA):

- **Write path**: the *complete history* of each fact group (episode) is
  written/available to the method under test — every version of every fact,
  oldest to newest. No oracle fields (gold answer, target version marker) are
  ever written, read, or filtered on.
- **Read path**: each query carries its **explicit as_of year** — the query
  text is `"As of {as_of}, {question}"` (helper `build_query_text` in
  `c1_hybrid_rag/retrieval.py`). Retrieval/state lookup is **restricted to the
  query's own episode** (C1/C2: `episode_id` filter before ranking; C4: mem0
  `user_id` filter). The method must resolve which version of the fact is
  current as of the query year from the stored history alone.
- **What it is NOT**: this is not End-to-End Retrieval. Under End-to-End
  Retrieval the store is the only state and the query is free-form with no
  year condition and no per-episode restriction — it measures the memory/
  retrieval system as a black box. Under csr_v1 the as_of year and the
  per-episode scoping are legitimate, declared query conditions; the
  measurement target is *correct temporal state routing under a frozen reader
  protocol*, not raw retrieval quality.
- **conflicting definition (unified across C1/C2/C4)**: a query is
  `conflicting = True` iff the same `state_key` has **≥ 2 distinct answer
  sets** across its queries (metadata definition; computed from query
  metadata only).
- **Reader protocol**: frozen. C1/C2/C4 use the frozen Mistral-7B reader
  (greedy, 128 new tokens, seed 42) and frozen ex28 extraction. C3 uses the
  official `vcrm.reader.FrozenReader` read protocol
  (`FrozenReader.answer(query, bundle=...)`), which builds the As-of-prefixed
  question internally.

## 2. Naming

- C1 (`c1_hybrid_rag`): BM25+BGE hybrid RAG under csr_v1.
- C2 (`c2_llmlingua2_rag`): LLMLingua-2 compression + RAG under csr_v1.
- C3 (`c3_icae_full_rag`): renamed **ICAE-Full (Controlled)**. It must no
  longer be presented as "RAG": it reads via latent ICAE memory slots under
  the official FrozenReader read protocol, not via text retrieval.
- C4 (`c4_mem0_oss`): Mem0 OSS baseline under csr_v1 (as_of injection and
  per-episode routing were already in place; no code change needed).
- Ours (VCRM/ICAE project method): already satisfies the csr_v1 read path
  (as_of-prefixed question via FrozenReader); no change needed.

## 3. Per-method change table

| Method | as_of year injection | Episode-scoped retrieval | conflict definition | Reader protocol | Naming | Output dir |
|---|---|---|---|---|---|---|
| C1 | added: `build_query_text` for retrieval query + Reader prompt | added: `retrieve(episode_id=...)` filter before BM25/BGE+RRF | unified metadata definition | unchanged (frozen Mistral reader) | unchanged | `c1_hybrid_rag/runs_csr_v1/` |
| C2 | same as C1 | same as C1 | same as C1 | unchanged | unchanged | own `runs_csr_v1/` |
| C3 | via official FrozenReader (`question()` builds `As of: {date}` prefix) | n/a (latent memory per episode bundle) | per unified definition | **changed to official `vcrm.reader.FrozenReader.answer()`** | **renamed ICAE-Full (Controlled)**; no longer called RAG | own `runs_csr_v1/` |
| C4 | already merged (pre-existing) | already merged (`user_id` filter) | already metadata-based | unchanged (frozen C1 reader) | unchanged | `c4_mem0_oss/runs_csr_v1/` on rerun |
| Ours | already in FrozenReader | per-episode latent bundles | per unified definition | FrozenReader | unchanged | own `runs_csr_v1/` |

Row schema (C1/C2): each JSONL row carries `protocol: "csr_v1"`,
`query_text` (the exact injected query string), and the recomputed
`conflicting` flag. Resume refuses any output file whose rows do not carry
`protocol == "csr_v1"`.

## 4. SUPERSEDED results

All previously reported C1/C2/C3/C4/Ours TempLAMA numbers were produced under
the **pre-csr protocol** (no as_of year in the retrieval/read query, and/or
non-episode-scoped retrieval, and/or placeholder conflict flags, and/or
non-official C3 read protocol). They are kept for provenance only and are
**SUPERSEDED**; none may be cited as csr_v1 results. Old `runs/` directories
are left untouched; all new runs write to separate `runs_csr_v1/` directories.

## 5. Re-test route (three steps)

1. **Validation batch**: rerun a fixed validation subset of **100–200 fixed
   query ids** (pinned qid list, committed with the run) under csr_v1.
2. **Gate**: check the validation batch (protocol fields present, conflict
   flags consistent, sanity metrics vs. expectations). Only after it passes,
   proceed.
3. **Full run**: execute the full official test runs under csr_v1 in the
   dedicated `runs_csr_v1/` output directories; freeze results.

No experiment has been executed yet (code changes + dryrun only).
