# Previous Protocol Retired

## ICAE Route (2026-10-14)

The active reader is now ICAE v2. TransMem reader/codec/training/runner sources
are research history under `legacy/transmem`; the complete previous local checkout
is preserved separately in `work/vcrm_transmem_archive`. Legacy files are not
executable entrypoints with the new package and must not be imported by ICAE runs.
The immutable v100-2 phase12 project/results remain unchanged. LoCoMo reproduction
is not a prerequisite or main comparator for the new protocol.
The old difference codec and hidden-state MSE are disabled. New history compression
will be defined only after independent ICAE version entries pass validation/manual
review. No archived answers/scores train the new reader.

## Prior Risk-Gate Route

The previous training-free risk gate, full-history On/Off runs, thresholds, signals,
generated answers, scores and reports have been removed from the active workflow.
They are not VCRM results and are not used to train, tune, initialize or evaluate VCRM.

Recoverable archival location on v100-1:
`/share/latent2/deprecated/risk_gate_20261013/`

Local old code: `work/deprecated/`; old exported report: `outputs/deprecated/`.
Archiving rather than irreversible deletion preserves an audit trail only. New entrypoints
never read that directory. Credential access is migrated to VCRM `.env` with mode 600.

For that prior TransMem route, retained external dependencies were the frozen pretrained backbone, the published
TransMem checkpoint, upstream source and published datasets. These are not outputs of
the discarded risk-gate experiment. LongMemEval raw data has been upgraded to cleaned-S.
