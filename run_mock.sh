#!/usr/bin/env bash
set -euo pipefail

# Deterministic run: uses bridge/mock_scenarios/milestone_demo.json, no LLM calls.

# Prefer venv interpreter for reproducibility; fall back to system python3.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -x "$SCRIPT_DIR/.venv/bin/python3" ]; then
  PYTHON="$SCRIPT_DIR/.venv/bin/python3"
elif [ -x "$SCRIPT_DIR/.venv/bin/python" ]; then
  PYTHON="$SCRIPT_DIR/.venv/bin/python"
else
  PYTHON="python3"
fi
echo "[run_mock.sh] Using interpreter: $PYTHON"

"$PYTHON" -u bridge/loop.py --mode mock --start-agent gemini "$@"
