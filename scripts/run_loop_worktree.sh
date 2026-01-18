#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Working tree must be clean before loop testing." >&2
  exit 1
fi

WORKTREE="/tmp/ff_loop_test"
if [[ -d "$WORKTREE" ]]; then
  git worktree remove -f "$WORKTREE" >/dev/null 2>&1 || true
fi

git worktree add -f "$WORKTREE" HEAD

cleanup() {
  git worktree remove -f "$WORKTREE" >/dev/null 2>&1 || true
}
trap cleanup EXIT

cd "$WORKTREE"
python3 -u bridge/loop.py \
  --mode mock \
  --mock-scenario bridge/mock_scenarios/tri_agent_smoke.json \
  --start-agent gemini \
  --no-agent-branch
