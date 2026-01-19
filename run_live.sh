#!/usr/bin/env bash
set -euo pipefail

# Live run: requires Codex CLI, Gemini CLI, and Claude Code.

# Prefer venv interpreter for reproducibility; fall back to system python3.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -x "$SCRIPT_DIR/.venv/bin/python3" ]; then
  PYTHON="$SCRIPT_DIR/.venv/bin/python3"
elif [ -x "$SCRIPT_DIR/.venv/bin/python" ]; then
  PYTHON="$SCRIPT_DIR/.venv/bin/python"
else
  PYTHON="python3"
fi
echo "[run_live.sh] Using interpreter: $PYTHON"

"$PYTHON" -u bridge/loop.py --mode live --start-agent gemini "$@"
