# VCRM: ICAE Versioned Memory

The active baseline is official ICAE v2, Mistral-7B-Instruct-v0.2 with the
instruction-finetuned ICAE checkpoint. The TransMem study is retained in
`legacy/transmem` and the separate immutable phase12 results. It is not a main
comparison or a prerequisite. No original results/manifests are overwritten.

## First Gate

New event -> ICAE encoder -> independent FP16 version entry -> explicit state/time
index -> frozen ICAE decoder + query. Current means the latest observed version;
historical entries remain independent. No raw-history fallback, codec, hidden-state
reconstruction objective, RL or learned eviction is enabled.

Use published TempLAMA train for encoder-only answer-loss adaptation and its
independent validation split for current/historical reading. Official source train
and val are filtered for entity overlaps; test is unchanged. This adaptation and
the supplied-fact streaming protocol are explicit, not the official TempLAMA task.
LongMemEval-S preparation and the original custom-judge protocol are retained for
later transfer evaluation; none of its 500 QA may train/adapt/select this reader.

The constructor uses an unchanged official ICAE source, replacing its hardcoded
FlashAttention2/BF16 loader at runtime with FP16/SDPA for V100. Adapter checkpoint
keys must exactly match the expected LoRA+memory embedding keys. The official
Mistral instruction token layout and adapter-disabled decoding are preserved.
Entries over the published 5120-token limit raise, not silently crop. Longer inputs
need a separately audited chunking policy before LongMemEval transfer.

## Commands

Host v100-2, isolated project `/share/latent2/vcrm_icae`. Credentials stay outside
manifests and Git in `.env`, mode 600. No API is needed for TempLAMA validation.

```bash
# Environment/download preparation, no GPU allocation.
./run_experiments.sh --stage download --run-name download
./run_experiments.sh --stage prepare --run-name preparation
./run_experiments.sh --stage preflight
./run_experiments.sh --stage parity --gpus auto --run-name implementation_parity

# First confirm loading/decoding on a debug validation episode.
./run_experiments.sh --stage readability --limit 1 --gpus auto --run-name smoke
./run_experiments.sh --stage summarize --limit 1 --run-name smoke

# Full train-only encoder adaptation, no VCRM codec.
./run_experiments.sh --stage adapt --gpus auto --run-name encoder_adapt
./run_experiments.sh --stage activate --adaptation-run encoder_adapt
./run_experiments.sh --config config.adapted.json --stage readability \
  --gpus all --run-name adapted_validation
./run_experiments.sh --config config.adapted.json --stage summarize --run-name adapted_validation
./run_experiments.sh --config config.adapted.json --stage gate \
  --adaptation-run encoder_adapt --run-name adapted_validation
```

Thresholds are provisional engineering criteria fixed before validation: single
entry accuracy >=0.8, current accuracy >=0.8, historical F1 >=0.8, latent minus
no-memory accuracy >=0.1. Full validation and manual review are mandatory. Debug
or incomplete runs cannot unlock the gate. Failed reading means repair/replace the
checkpoint, not introduce a codec. Passing only makes the run eligible for manual
review; no compression training entry exists yet.

The single-entry and multi-version indexed tensors must match after a durable
save/reload. Generate once for identical tensors, not twice as purported independent
evidence. Report text/no-memory/wrong-version controls, changed-fact historical
coverage, answer EM/F1, group macro accuracy, exact serialized bytes, read tokens,
write/read seconds and GPU peak. All controls share the same decoder and selected
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
persistent byte budget. Future history codes should optimize historical answer
loss, not recover complete original hidden states. Current anchors, dependencies
and rewriting costs must count. MemoryLLM/M+ remain external related work and later
competitors, not prerequisites for this engineering gate.
