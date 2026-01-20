#!/usr/bin/env bash
set -euo pipefail

# Sequential orchestrator runner (live mode)
#
# Usage:
#   ./run_live.sh                    # Default: starts with codex
#   ./run_live.sh --only-claude      # ONLY use Claude for ALL operations
#   ./run_live.sh --only-codex       # ONLY use Codex for ALL operations
#   ./run_live.sh --start-agent claude  # Start with Claude (but allow alternation)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check for venv python interpreter (.venv/bin/python or .venv/bin/python3)
if [ -x "${ROOT}/.venv/bin/python" ]; then
  PY="${ROOT}/.venv/bin/python"
elif [ -x "${ROOT}/.venv/bin/python3" ]; then
  PY="${ROOT}/.venv/bin/python3"
else
  PY="$(command -v python3)"
fi

echo "[run_live.sh] Using interpreter: ${PY}"

# Pass through all CLI arguments to loop.py
exec "${PY}" -u "${ROOT}/bridge/loop.py" \
  --mode live \
  --config "${ROOT}/bridge/config.json" \
  --start-agent codex \
  "$@"
