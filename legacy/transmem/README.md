# VCRM Experiments

This project replaces the deprecated risk-gate experiment. It never imports its
code, predictions, training data, thresholds, or scores. The unchanged external
TransMem checkpoint/backbone and published datasets are external dependencies.
The new research plan is `EXPERIMENT_PLAN.zh.md`.

## Runtime and Data

- Host: v100-1, project `/share/latent2/vcrm`.
- Runtime: `.venv` with the tested latent2 Python 3.10/Torch 2.5.1/CUDA 12.4 base;
  project-local additional packages do not modify the base environment.
- Bootstrap: `.venv/bin/python bootstrap.py`. URLs, upstream revisions and SHA256
  hashes are recorded in `data_manifest.json`.
- LongMemEval: the complete September 2025 cleaned-S file (500 questions).
- LoCoMo: official locomo10.json, categories 1/2/3/4 (1,540 QA), no evaluation QA training.
- TempLAMA: original Google Research yearly queries/answer sets. Their splits are
  entity-relation-disjoint, not globally entity-disjoint. Train/val overlapping
  entities are excluded, test unchanged; `entity_split_audit.json` records this adaptation.
  Do not call the filtered training split the unchanged official split.
- STALE: pinned official code and dependencies are present. Its published repository
  contains generation code/seeds, not the full claimed 400-scenario evaluation set.
  It is a supplemental experiment blocked on author-released data. This runner
  does not generate replacement evaluation sets or run a purported STALE benchmark.
  `--stage import-stale --stale-data FILE --stale-provenance author-release` ingests a
  supplied official-format file without oracle conflict labels. `--dataset stale`
  runs the shared streaming protocol; `--stage stale-score --conflict-type T1` calls
  the official STALE judge. Synthetic replacements require `generated-extension`.
- API credentials live in `.env` (600), outside manifests/artifacts. The default judge
  is Qwen3.5-9B using the official LongMemEval prompt, not the official GPT-4o judge.

## Concrete Implementation Decisions

These are experimental choices, not claimed results or properties of upstream TransMem.

1. A memory event is one timestamped conversation session, or one published TempLAMA
   fact change. Consecutive years with the same valid-answer set produce one event
   at the first observed year, while *all* original yearly queries/labels are retained.
   This shared deduplication prevents inventing storage pressure from duplicate facts.
   No gold answer/evidence fields enter conversational ingestion.
   TempLAMA facts are reconstructed from the published yearly fact labels and supplied
   to *every* method as stream input; evaluation uses the unchanged yearly labels.
2. Sessions use upstream chat formatting, HM positions and hidden-state layers. They
   include the observation timestamp in a DATE/CONVERSATION envelope, so temporal
   information is part of the latent input, not just an inaccessible index.
   They are encoded independently with frozen Qwen and frozen TransMem. Chunked causal
   encoding retains all tokens and KV, not an overflow-text shortcut.
3. Conversation state association is a shared, label-free hashed lexical cosine rule
   (threshold 0.8). It is deliberately a coarse prototype, NOT a verified semantic
   fact/conflict detector. TempLAMA uses its explicit entity-relation keys. Neither
   method receives LongMemEval `has_answer` or `answer_session_ids`.
4. Query inference sees only question/time and reloaded durable store. No full-history
   prompt, raw conversation cache, extracted train features, or oracle evidence is
   used as a retrieval fallback. Empty stores yield frozen-backbone answers and are logged.
   LongMemEval uses opaque session ordinals, not evidence-bearing official IDs; gold IDs
   are joined only during offline error analysis. Its date prompt is neutral `Current date`
   under the supplied-history protocol, not an added causal cutoff.
5. A = full latent entries with shared lexical retrieval and FIFO byte eviction.
   B = A plus version selection (latest or explicitly requested past year).
   C = full current anchors plus independent bottleneck codes for historical entries.
   D = full current anchors plus bottleneck codes of `(old - current)`, reconstructed
   as `current + decoder(code)`. C/D share the same active encoder/decoder layers,
   initialization, rank, samples, steps, and losses; they differ only in residual
   subtraction/addition. Layer/slot positions are preserved.
6. D re-encodes surviving histories against the new current anchor on each update.
   This uses only decoded memory, not old raw text; accumulated quantization/rebasing
   drift and extra write compute are real limitations. C does not rebase its independent
   histories. All compressed entries depend on an anchor; eviction never leaves a
   dangling dependency. Latest anchors may consume substantial bytes at small budgets.
7. C/D training uses official *filtered train* facts at sequential prefixes, with both
   historical QA and latest self-reconstruction QA plus hidden-state MSE. The latest
   auxiliary reconstructs the current state, although deployment stores current anchors
   uncompressed. This auxiliary is not evidence of better current-state compression.
   QA loss backpropagates through the frozen reader into codecs; no LLM/TransMem weights
   are updated. FP16 code rounding uses a straight-through path during training.
8. E = original text events, with the same state association, time selection and FIFO
   budget rules. Reads use whole entries that fit the text token cap. Text and latent
   caps are different units; report actual read tokens, latent slots, FLOPs proxies,
   latency and peak GPU memory, not claims of equal read cost.
9. Budget is the actual `.memory` file byte length, including format/header, IDs,
   timestamps, keys and payloads. No hidden persistent search index is consulted.
   Model and codec weights are separately reported shared-model costs. Predictions,
   scoring logs and write traces are audit artifacts, never inference memory.
10. Three byte points are frozen from C's validation-set serialized *layout* p90,
    using one current anchor plus 25%, 50%, 100% of the remaining historical footprint.
    This avoids budgets incapable of storing even one current anchor. No test scores select budgets. Placeholders in
    layout measurement are only for byte counting, not model results.

## Unified Entry

Every command below uses `./run_experiments.sh`. No automatic full experiment/expensive
training is started merely by setup. GPU indices are physical; `auto` chooses one idle
card, `all` chooses currently idle cards. The runner refuses busy cards.

```bash
./run_experiments.sh --stage preflight
./run_experiments.sh --stage prepare --run-name setup
./run_experiments.sh --stage budgets --run-name setup

# Phase 0: use the actual upstream LoCoMo evaluator, no modified scoring prompt.
./run_experiments.sh --stage reproduce --gpus auto --run-name upstream_locomo
# Review upstream_locomo/locomo_upstream.json and reproduction_protocol.json.
# FP16 is a documented runtime-only V100 adaptation; no paper-number match is assumed.

# Phase 1: A/B/E streaming. Full reproduction must be completed and manually reviewed.
./run_experiments.sh --stage stream --dataset longmemeval --split test \
  --methods transmem_streaming version_index text_log --gpus all \
  --run-name lme_baselines --accept-reproduction
./run_experiments.sh --stage score --run-name lme_baselines
./run_experiments.sh --stage report --run-name lme_baselines

# Phase 2: test the independent codec first; do not require D for C-only evaluation.
./run_experiments.sh --stage features --split train --gpus auto --run-name train_features
./run_experiments.sh --stage train --methods independent --gpus auto --run-name independent_codec
./run_experiments.sh --stage stream --dataset templama --split val \
  --methods transmem_streaming version_index independent text_log \
  --gpus all --run-name independent_validation
./run_experiments.sh --stage score --run-name independent_validation
./run_experiments.sh --stage report --run-name independent_validation

# Only train D if baseline evidence justifies residual compression.
./run_experiments.sh --stage train --methods vcrm --gpus auto --run-name residual_codec

# Phase 3: fixed-budget comparison, validation before final test.
./run_experiments.sh --stage stream --dataset templama --split val \
  --methods transmem_streaming version_index independent vcrm text_log \
  --gpus all --run-name templama_validation
./run_experiments.sh --stage score --run-name templama_validation
./run_experiments.sh --stage report --run-name templama_validation
```

Use test only once choices are frozen; repeat with `--split test` and a separate run name.
Use `--dataset longmemeval`/`locomo` for transfer evaluation with the same frozen codecs.
The initial default reader caps are 32 latent slots, 2,048 text tokens, 256 answer tokens.
Upstream reproduction alone retains the author's 50-token cap for comparability.

`--limit N` creates a debug run; use its *same limit* for score/report and distinct run
names. Debug codecs are written separately and refused by formal evaluation. No random
codec is substituted for an untrained method. A smoke reproduction does not unlock formal
training. Full reproduction is blocked until reviewed, not automatically certified.

Resume with the same code/config/limit/run-name. Atomic episode/store writes preserve
completed work. GPU sharding is per episode; this avoids duplicate memory ingestion for
questions sharing a stream. Feature/training stages use one GPU; reproduction uses the
released multi-GPU sharding/merger and validates complete original question coverage.
On Transformers 4.51, `vcrm/runtime.py` translates the released decoder cache keyword
without editing upstream sources; cache-equivalence tests must pass before use.

Stage-2 diagnostics can use `--baseline-no-eviction` with only `transmem_streaming` on
LongMemEval. This uses a 1 TiB sentinel ceiling, reports actual file bytes, and is NOT
a fixed-budget comparison. Pass the same flag for score/report to retain the fingerprint.
After a failed baseline diagnosis, `stage2_decision.json` holds formal codec training
and multi-method comparisons; repair/approve a new protocol explicitly before continuing.
Training saves epoch weights and provenance but currently restarts from the fixed seed
when relaunched; optimizer-state resume is not implemented. Do not treat an interrupted
training restart as continuation of its previous update count.

## Results and Risks

Outputs: per-episode stores, selected versions, eviction/write traces, token ids, generation
limit flags, scoring provenance, per-type QA summaries, latest accuracy, historical F1,
retention, stale-version answer rate, paired entity-relation clustered bootstrap, and
measured write/read costs. Reports include CSV and storage-tradeoff PNGs made only from
actual scored predictions, explicitly marked smoke/partial when appropriate.
TempLAMA dates are observation-year ordering, not real event
arrival timestamps. Missing years are not assigned invented validity intervals.

Potential failure modes to test before publication: untrained streaming reader distribution
shift, coarse session-level state association, anchor/codec rebasing drift, tiny-budget
empty stores, text-vs-latent read-cost differences, and V100 FP16 numerical differences.
No claim of reproducing paper metrics or demonstrating VCRM superiority follows from setup.
