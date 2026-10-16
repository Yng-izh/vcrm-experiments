# VCRM: ICAE Versioned Memory

The active baseline is official ICAE v2, Mistral-7B-Instruct-v0.2 with the
instruction-finetuned ICAE checkpoint. The TransMem study is retained in
`legacy/transmem` and the separate immutable phase12 results. It is not a main
comparison or a prerequisite. No original results/manifests are overwritten.

The current authorized architecture study is `run_architecture.py` with
`config.architecture.json`: complete the external baseline table, run frozen
read/FP16-code diagnostics, then gated E1-E4 and one matched final C/D pair.
See `ARCHITECTURE_ABLATIONS.zh.md` for exact controls and stop rules. The unfinished
`matched_v4` run is paused and preserved; it is not a completed baseline. The
`--phase all` architecture workflow does not access held-out test; test is an
explicit separate phase after final checkpoints are frozen.

The latest module design and limitations are recorded in `ARCHITECTURE.zh.md`.
The authorized three-step workflow is controlled by `run_paper.sh` and
`config.experiment.json`. See `RUN_PAPER.zh.md` for stages, gate behavior and limitations.

## First Gate

New event -> ICAE encoder -> independent FP16 version entry -> explicit state/time
index -> frozen ICAE decoder + query. Current means the latest observed version;
historical entries remain independent. No raw-history fallback, codec, hidden-state
reconstruction objective, RL or learned eviction is enabled.

First use the official checkpoint zero-shot on independent validation. The old
short-fact adaptation is retired. An opt-in train-only paragraph/multifact protocol
now adapts only `memory_token_embed`; released rank-512 LoRA and decoder weights are
verified unchanged. It remains disabled in the checked-in config until its audit is
reviewed. Official source train and val are filtered for entity overlaps; test is
unchanged. The supplied-fact streaming protocol is explicit, not the official task.
LongMemEval-S preparation and the original custom-judge protocol are retained for
later transfer evaluation; none of its 500 QA may train/adapt/select this reader.

The constructor uses an unchanged official ICAE source, replacing its hardcoded
FlashAttention2/BF16 loader at runtime with FP16/SDPA for V100. Adapter checkpoint
keys must exactly match the expected LoRA+memory embedding keys. The official
Mistral instruction token layout and adapter-disabled decoding are preserved.
Entries over the published 5120-token limit raise, not silently crop. Longer inputs
use an explicit contiguous-token chunking extension only in the audited 24-case
diagnostic. No cropping, one original BOS, all chunks retained. Decoder prefill is
chunked with a complete KV cache; parity checks greedy output on a small fixture.

## Commands

Host v100-2, isolated project `/share/latent2/vcrm_icae`. Credentials stay outside
manifests and Git in `.env`, mode 600. No API is needed for TempLAMA validation.

```bash
# Environment/download preparation, no GPU allocation.
./run_experiments.sh --stage download --run-name download
./run_experiments.sh --stage prepare --run-name preparation
./run_experiments.sh --stage preflight
./run_experiments.sh --stage parity --gpus auto --run-name implementation_parity
./run_experiments.sh --stage unit-tests --run-name code_checks
./run_experiments.sh --stage training-audit --run-name training_inputs

# First confirm loading/decoding on a debug validation episode.
./run_experiments.sh --stage readability --limit 1 --gpus auto --run-name smoke
./run_experiments.sh --stage summarize --limit 1 --run-name smoke

# Zero-shot pilot (100 fact groups, not 100 yearly QA).
./run_experiments.sh --stage readability --limit 100 --gpus all --run-name zero_pilot
./run_experiments.sh --stage summarize --limit 100 --run-name zero_pilot
# Full zero-shot validation, required before a gate decision.
./run_experiments.sh --stage readability --gpus all --run-name zero_validation
./run_experiments.sh --stage summarize --run-name zero_validation
./run_experiments.sh --stage gate --run-name zero_validation
# Fixed post-hoc error cases, same Mistral latent/text/no-memory/oracle controls.
./run_experiments.sh --stage lme-audit --run-name lme24
./run_experiments.sh --stage lme-diagnostic --gpus all --run-name lme24
./run_experiments.sh --stage lme-summary --run-name lme24
```

Thresholds are provisional engineering criteria fixed before validation: single
entry accuracy >=0.8, current accuracy >=0.8, historical F1 >=0.8, latent minus
no-memory accuracy >=0.1. Full validation and manual review are mandatory. Debug
or incomplete runs cannot unlock the gate. Failed reading means repair/replace the
checkpoint, not introduce a codec. Passing only makes the run eligible for manual
review; compression training stays blocked until the recorded error review is approved.

The single-entry and multi-version indexed tensors must match after a durable
save/reload. Generate once for identical tensors, not twice as purported independent
evidence. Report text/no-memory/wrong-version controls, changed-fact historical
coverage, answer EM/F1, group macro accuracy, resident and decoder slots, exact
serialized bytes, read tokens, KV bytes, write/read seconds and GPU peak. All controls share the same decoder and selected
fact version. Text controls have unequal inference cost, reported explicitly.

GPU workers use only cards observed idle, shard whole episodes and log separately.
Run code/config/data fingerprints prevent reuse across changed protocols. Encoder
adaptation currently restarts only in a new run; optimizer resume is not implemented.

## Storage Caveat

128 x 4096 FP16 tokens occupy 1,048,576 payload bytes per segment, often far larger
than a short UTF-8 fact log. Exact budgets include memory, keys, IDs, timestamps,
format and metadata. Model/adapter weights are separate shared costs. Do not assume
latent memory wins persistent bytes; longer passages and inference token/latency
costs are separate hypotheses requiring measurements.

Once this gate succeeds, compare Independent ICAE Memory and Version-Conditioned
ICAE Memory with the same frozen reader, QA supervision, train samples/steps and
resident/active latent-slot budgets. `training-audit` verifies C/D capacity matching;
`codec-train --method C|D` is opt-in and disabled by default. Codes directly decode
at m slots, not expand to 128 and claim active-context savings. The loss combines
gold QA CE, detached uncompressed-reader logit KL, pooled alignment, and explicit
wrong-version unlikelihood. Current anchors, dependencies and rewriting costs count.
MemoryLLM/M+ remain external related work and later competitors.

LongMemEval C/D must use shared query-independent, label-free state-aligned spans,
not unrelated whole sessions. `SpanWriter` creates exact 256-token source spans;
semantic similarity is only an association candidate, and a separate validated
resolver must confirm an actual update before state merging.
The 24-case pilot is nonrepresentative and cannot tune a model or authorize codec
training. >=10 correct is a candidate signal; <=4 prompts reading/routing diagnosis;
5--9 is inconclusive. Full validation and same-reader text controls remain necessary.

## Latest Results

- Architecture v6 ablation matrix (E1/E2/E3/E4) on Templama val: see [results/architecture_v6/RESULTS_LOG.zh.md](results/architecture_v6/RESULTS_LOG.zh.md).
- E1 supported, E2 supported (weak), E3 supported (strong, all_compressed F1 +0.392), E4 vs c_best(E3) weak positive (+0.0025 F1), pending compare/selection decision.
- Next: selection_decision.json -> final pair (CFinal/DFinal) -> held-out test phase.
