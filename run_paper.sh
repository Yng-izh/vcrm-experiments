#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$ROOT/.env" ]]; then
  set -a; source "$ROOT/.env"; set +a
fi
exec "${VCRM_PYTHON:-$ROOT/.venv/bin/python}" "$ROOT/run_paper.py" "$@"
