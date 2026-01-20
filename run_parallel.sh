#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${ROOT}/.venv"

if [ -x "${VENV}/bin/python3" ]; then
  PY="${VENV}/bin/python3"
else
  PY="$(command -v python3)"
fi

echo "[run_parallel.sh] Using interpreter: ${PY}"

# Pass through all CLI arguments to loop.py
exec "${PY}" -u "${ROOT}/bridge/loop.py" \
  --mode live \
  --runner parallel \
  --config "${ROOT}/bridge/config.json" \
  --start-agent codex \
  "$@"
