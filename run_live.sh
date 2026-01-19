#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${ROOT}/.venv"

if [ -x "${VENV}/bin/python3" ]; then
  PY="${VENV}/bin/python3"
else
  PY="$(command -v python3)"
fi

echo "[run_live.sh] Using interpreter: ${PY}"

exec "${PY}" -u "${ROOT}/bridge/loop.py" --mode live --config "${ROOT}/bridge/config.json" --start-agent codex
