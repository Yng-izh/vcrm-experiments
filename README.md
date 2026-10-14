# VCRM Experiments

Reproducible artifacts for the TransMem baseline audit and the first
TransMem-Streaming diagnostic on v100-2.

## Results

- **Official TransMem + LoCoMo reproduction:** 1,540 questions, overall token F1 **53.95%** versus the paper's 54.11%.
- **TransMem-Streaming + LongMemEval-S diagnostic:** 500 questions, overall judged accuracy **11.20%**; Knowledge Update **14.10%**.
- The streaming run used a no-eviction diagnostic ceiling. It is not a fixed-budget comparison and does not establish a VCRM improvement.
- Formal VCRM codec training and comparisons are held until the streaming reader/encoding protocol is repaired and independently validated.

## Repository Layout

- `run.py`, `run_experiments.sh`, `vcrm/`: unified experiment runner and implementation.
- `tests/`: 23 project tests.
- `diagnostics/`: reader parity and streaming error-control tools.
- `results/v1002_phase12/`: reports, provenance, manifests, predictions, scores, and source snapshots from the completed v100-2 runs.
- `IMPLEMENTATION.md`: detailed protocol and implementation notes.
- `EXPERIMENT_PLAN.zh.md`: the planned VCRM experiment design.

## Reproduction Notes

The repository does not include model weights, raw benchmark files, upstream
checkouts, durable stores, or API credentials. Download the pinned external
dependencies separately and provide paths through a local `config.json` and
`.env`. The judge key is read only from `OPENAI_API_KEY`; no key value is
stored in this repository.

`config.json` is the v100-2 path template, not a portable installation config.
Set `project`, data/output/upstream/model paths and `judge_base_url` for your
own environment. Keep `stage2_decision.json` under the configured project so
the failed-baseline hold remains effective. Install Python/CUDA dependencies
before invoking the runner; `IMPLEMENTATION.md` documents the tested runtime.

Pinned artifacts used for the completed experiments:

| Artifact | Revision |
|---|---|
| TransMem, `Haodong-Lei-Ray/Project4` | `6d7a6127967ad44482c958b555d8c83bfb75e9b9` |
| delta-Mem, `Haodong-Lei-Ray/delta-Mem` | `bb6e99bb6b3a94ddae7f9b4743bfd8ae37e98675` |
| `Qwen/Qwen3-4B-Instruct-2507` | `cdbee75f17c01a7cc42f958dc650907174af0554` |
| `Rayleihaodong/TransMem-qwen3-4B-D4` | `85ba1010eb226e312e208940180bb60424a2cb79` |

The full hashes and runtime audit are in `phase12_input_audit.json` and
`provenance.json`. Do not replace pinned upstream sources with an unverified
checkout. `bootstrap.py` uses HEAD for some supplemental dependencies and is
not an exact historical environment restorer.

The LoCoMo result uses the released TransMem evaluator and scoring protocol,
with the documented V100 FP16 + SDPA runtime adapter. The LongMemEval result
uses the official prompt with a Qwen3.5-9B-compatible judge endpoint, not the
official GPT-4o score.

```bash
./run_experiments.sh --stage preflight
python -m pytest -q tests
```

See `results/v1002_phase12/阶段1复现报告.md` and
`results/v1002_phase12/阶段2基线报告.md` for the full conclusions.

The original-protocol TransMem result on these same 500 LongMemEval-S questions
has not yet been run in this phase-1/2 project. The poor streaming result alone
does not establish that the official TransMem checkpoint fails on this dataset.
The paper also describes overflow chunk-level HM aggregation; this is not the
same as this project's session-level retrieval wrapper.
