# Completed Experiment Artifacts

## csr_v1 controlled state routing (latest)

| File | Contents |
|---|---|
| `full_test_report.md` | Five-method frozen table (34,963 test queries): Ours (v6/E3) 0.8512 EM, C1 0.3906, C3 0.3624, C4 0.2504, C2 0.1123 |
| `ours_v6_summary.json` | Machine-readable Ours v6/E3 summary incl. latent-vs-fallback split |
| `ours_protocol_split.md` | Pure-latent vs text-fallback split + protocol-consistency fix record |
| `smoke20_report.md` | Pre-full-run gate report (20 pinned qids, 3 methods) |
| `BACKUP_PATHS_20261020.txt` | Off-site hidden backup locations of the project tree |

The latest ICAE C16/D16 results are under `icae_aligned_v3/`; see
[`icae_aligned_v3/README.zh.md`](icae_aligned_v3/README.zh.md) for the protocol,
scores, limitations, and source snapshot. The fixed 481-query diagnostic gives
C16 55.07% historical F1 and D16 51.56%; D-C's clustered 95% confidence interval
includes zero. These are our custom official-checkpoint evaluations, not
published ICAE benchmark numbers or untouched test results.

The earlier TransMem reproduction and streaming diagnostics remain under
`v1002_phase12/`. The table below describes that archival study.

| File / Directory | Contents |
|---|---|
| `阶段1复现报告.md` | Full LoCoMo reproduction and paper comparison |
| `阶段2基线报告.md` | LongMemEval-S streaming result and limitations |
| `locomo_upstream.json` | 1,540 original evaluator records |
| `lme_predictions_scores.tar.gz` | 500 `episodes/*.json` predictions and 500 `scores/*.json` files |
| `lme_reports/` | Aggregate JSON, CSV, Markdown, and diagnostic figure |
| `error_controls/` | 24 post-hoc selected-error cases, controls, judge cache, and summary |
| `phase12_input_audit.json`, `provenance.json` | Released source/model/data hashes and audit |
| `manifest.json` | Immutable completed LongMemEval run identity |
| `reader_parity_status.json` | Actual GPU implementation parity checks |
| `reproduction_protocol.json`, `reproduction_status.json` | LoCoMo protocol and review status |
| `final_verification.json` | Test counts, result completeness, and released GPUs |
| `stage2_decision.json` | Hold on formal training/comparisons |
| `experiment_source_snapshot.tar.gz` | Actual completed-run core source and config |
| `experiment_source_supplement.tar.gz` | Completed-run bootstrap/GPU entry source |
| `handoff_source.tar.gz`, `source_snapshot.tar.gz` | Local handoff snapshots; not the immutable completed-run source |

All 19 source hashes listed in the completed-run manifest match the two
`experiment_source_*` archives. The top-level delivered runner includes a
post-experiment hold gate, so it is not byte-identical to that archived runner.
Do not overwrite historical run fingerprints to resume with changed code.

The original experiment config contains machine paths and the judge endpoint,
but no credential values. We preserve it for audit rather than rewriting history.
Raw benchmark histories, memory stores, model weights, and upstream code are
not included. External code/data retain their original licenses; no ownership
or relicensing of those dependencies is asserted here.

The 24 controls were selected from failed all-evidence-present cases. Their
scores are diagnostics, not benchmark-wide text-baseline accuracy. No VCRM
superiority, fixed-budget improvement, or definitive cause of the streaming
failure has been established.
