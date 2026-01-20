#!/usr/bin/env bash
set -euo pipefail

# Parallel orchestrator runner
#
# Usage:
#   ./run_parallel.sh                    # Single run (default: uses both agents)
#   ./run_parallel.sh --auto-continue    # Loop until success or max runs
#   ORCH_AUTO_CONTINUE=1 ./run_parallel.sh  # Same as --auto-continue
#   ./run_parallel.sh --no-auto-continue # Disable auto-continue even if env var set
#
# Agent restriction flags (HARD ENFORCEMENT):
#   ./run_parallel.sh --only-claude       # ONLY use Claude for ALL operations
#   ./run_parallel.sh --only-codex        # ONLY use Codex for ALL operations
#   ./run_parallel.sh --only-claude --auto-continue  # Combine with auto-continue
#
# To run only specific tasks:
#   ./run_parallel.sh --only-task M1-DSL-SCHEMA --only-task M2-OPENEMS-SCHEMA
#
# To allow resource-intensive tasks:
#   ./run_parallel.sh --allow-resource-intensive
#
# Verifying no agent violations (after run):
#   grep -r "agent=codex" runs/*/  # Should find NO matches when --only-claude
#   grep -r "planning tasks via codex" runs/*/  # Should find NO matches when --only-claude

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
