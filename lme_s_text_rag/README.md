# LongMemEval-S text-RAG baseline (t13)

Session-level hybrid retrieval + frozen Mistral reader. **Produces hypotheses
only — scoring is done by the official LongMemEval judge elsewhere (t12 spec,
t14 wiring). This script never computes scores.**

Files: `run_text_rag.py` (the only script; plus this README).

## Protocol differences vs TempLAMA csr_v1 (do NOT "fix" these)

- **No episode partition**: retrieval is over the WHOLE haystack (all sessions
  of a question's memory), not episode-scoped.
- **No as_of / state_key / oracle masking**: there are no such fields; the
  question is used verbatim.
- Sessions are retrieved per question; each session is one retrieval document
  with a `[date]` prefix (helps temporal-reasoning questions).

## Reader choice (and why)

Reuses the **C1 vLLM shim** (`c1_hybrid_rag/reader.read_answer`) rather than
`vcrm/reader.FrozenReader`:

- `FrozenReader` is the ICAE latent-memory reader — its `answer()` requires
  either a latent bundle or `text` framed for ICAE, enforces
  `max_answer_tokens` from the frozen config (64), and is audited for the
  TempLAMA protocol. LongMemEval needs plain long-text prompting with a
  longer generation budget.
- The C1 shim talks to a standard vLLM OpenAI server with
  `temperature=0, seed=42` (frozen semantics) and lets us raise
  `max_tokens` to 256 via `--max-new-tokens` without touching any frozen
  config. Same Mistral-7B-Instruct-v0.2 as the TempLAMA main table.

Start the vLLM server first (single GPU), e.g.:

```
python3 -m vllm.entrypoints.openai.api_server \
  --model /share/latent2/vcrm_icae/models/Mistral-7B-Instruct-v0.2 \
  --served-model-name mistral-7b --dtype half --gpu-memory-utilization 0.85 \
  --max-model-len 32768 --port 8001
```

## Data path — TO BE CONFIRMED

The LongMemEval-S haystack/questions file path is NOT hardcoded. Pass it:

```
--data /path/to/longmemeval_s.json   # exact official filename TBD (t12)
```

Schema adaptation for the official release (session field names) is isolated
in `normalize_item()` — adjust there if the released schema differs.

## Sharding & resume

- `--shard k --shards N`: runs question k, k+N, ... of the sorted id list.
- `--limit N`: truncates to the first N questions (after sorting) before
  sharding; useful for smoke runs.
- **Resume**: done set = all `question_id`s already present as
  `runs/<tag>/<question_id>.json` across all shard dirs of the tag; they are
  skipped on re-run. Re-running is always safe.

## Cost estimate

LongMemEval-S: 500 questions, 10 top-k sessions/question → ~10-30k prompt
tokens per question on a single V100 vLLM instance. At observed C4-throughput
(~2-3 s/query for short prompts, longer here) expect roughly
**0.5-1.5 s/token-1k → ~1-2 min/question**, i.e. ~8-17 GPU-hours for all 500
single-GPU; shard across GPUs to scale linearly.

## Output

`runs/<tag>/<question_id>.json` per question:
`question_id, question, question_type, hypothesis, retrieved_session_ids,
retrieved_session_dates, prompt_tokens, read_seconds`.
No answer extraction/normalization, no scores. Field naming here is
provisional; the official-eval field mapping is owned by t12/t14.
