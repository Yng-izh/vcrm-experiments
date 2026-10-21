#!/usr/bin/env bash
# t14: one-shot wiring for the OFFICIAL LongMemEval-S evaluation.
#
# DEFAULT IS DRY-RUN: prints the commands, executes nothing.
# Pass --execute to actually run (requires judge API access; costs money).
#
# Required environment variables (NAMES ONLY — never commit values):
#   OPENAI_API_KEY        judge key for official gpt-4o (required by evaluate_qa.py)
#   OPENAI_ORGANIZATION   optional, only for multi-org accounts
#
# Overridable path/config variables (with defaults per t12 spec):
#   LME_TAG               run tag under lme_s_text_rag/runs/          (default: lme_s)
#   LME_DIR               this repo's lme_s_text_rag dir on the server (default: /share/latent2/vcrm_icae/lme_s_text_rag)
#   LME_REPO              official LongMemEval repo                   (default: /share/latent2/LongMemEval)
#   LME_REF               official S data json (ref_file)             (default: /share/latent2/vcrm_icae/data/longmemeval_s_cleaned.json)
#   LME_JUDGE_MODEL       metric_model_short, official supports gpt-4o (default: gpt-4o)
#
set -euo pipefail

EXECUTE=0
[[ "${1:-}" == "--execute" ]] && EXECUTE=1

TAG="${LME_TAG:-lme_s}"
LME_DIR="${LME_DIR:-/share/latent2/vcrm_icae/lme_s_text_rag}"
LME_REPO="${LME_REPO:-/share/latent2/LongMemEval}"
LME_REF="${LME_REF:-/share/latent2/vcrm_icae/data/longmemeval_s_cleaned.json}"
LME_JUDGE_MODEL="${LME_JUDGE_MODEL:-gpt-4o}"

PRED="${LME_DIR}/official_eval_in/${TAG}/predictions.jsonl"
ARCHIVE_DIR="${LME_DIR}/official_eval_out/${TAG}"

run() {
  echo "+ $*"
  if [[ $EXECUTE -eq 1 ]]; then "$@"; fi
}

echo "== step 1: build official eval input from runs/${TAG} =="
run python3 "${LME_DIR}/make_official_eval_input.py" --tag "${TAG}" --ref "${LME_REF}"

echo "== step 2: official judge (gpt-4o) via LongMemEval evaluate_qa.py =="
if [[ $EXECUTE -eq 1 ]]; then
  : "${OPENAI_API_KEY:?must export OPENAI_API_KEY (and optionally OPENAI_ORGANIZATION) for the official judge}"
fi
run bash -c "cd '${LME_REPO}/src/evaluation' && python3 evaluate_qa.py '${LME_JUDGE_MODEL}' '${PRED}' '${LME_REF}'"

echo "== step 3: official aggregation (print_qa_metrics) + archive =="
RESULTS="${PRED}.eval-results-${LME_JUDGE_MODEL}"
run python3 "${LME_REPO}/src/evaluation/print_qa_metrics.py" "${RESULTS}.log" "${LME_REF}"
run mkdir -p "${ARCHIVE_DIR}"
run cp "${PRED}" "${ARCHIVE_DIR}/"
run cp "${RESULTS}" "${RESULTS}.log" "${ARCHIVE_DIR}/" 2>/dev/null || true

if [[ $EXECUTE -eq 0 ]]; then
  echo "== dry-run complete (nothing executed). Re-run with --execute to actually evaluate. =="
fi
