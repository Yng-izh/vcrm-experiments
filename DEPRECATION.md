# Previous Protocol Retired

The previous training-free risk gate, full-history On/Off runs, thresholds, signals,
generated answers, scores and reports have been removed from the active workflow.
They are not VCRM results and are not used to train, tune, initialize or evaluate VCRM.

Recoverable archival location on v100-1:
`/share/latent2/deprecated/risk_gate_20261013/`

Local old code: `work/deprecated/`; old exported report: `outputs/deprecated/`.
Archiving rather than irreversible deletion preserves an audit trail only. New entrypoints
never read that directory. Credential access is migrated to VCRM `.env` with mode 600.

Retained external dependencies are the frozen pretrained backbone, the published
TransMem checkpoint, upstream source and published datasets. These are not outputs of
the discarded risk-gate experiment. LongMemEval raw data has been upgraded to cleaned-S.
