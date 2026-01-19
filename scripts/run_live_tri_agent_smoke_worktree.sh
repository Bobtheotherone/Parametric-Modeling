#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Working tree must be clean before live loop testing." >&2
  exit 1
fi

WORKTREE="/tmp/ff_live_tri_agent_smoke"
if [[ -d "$WORKTREE" ]]; then
  git worktree remove -f "$WORKTREE" >/dev/null 2>&1 || true
fi

git worktree add -f "$WORKTREE" HEAD

cleanup() {
  git worktree remove -f "$WORKTREE" >/dev/null 2>&1 || true
}
trap cleanup EXIT

cd "$WORKTREE"

SMOKE_DIR="$WORKTREE/agent_smoke"
mkdir -p "$SMOKE_DIR"
LOG_FILE="$SMOKE_DIR/loop.log"

set +e
FF_AGENT_SMOKE_DIR="$SMOKE_DIR" \
python3 -u bridge/loop.py \
  --mode live \
  --smoke-route gemini,codex,claude \
  --start-agent gemini \
  --no-agent-branch | tee "$LOG_FILE"
rc=${PIPESTATUS[0]}
set -e

if [[ "$rc" -ne 0 ]]; then
  echo "Live tri-agent smoke failed (rc=$rc)." >&2
  exit 1
fi

if grep -q "JSON INVALID" "$LOG_FILE" || grep -q "could not parse/validate JSON" "$LOG_FILE"; then
  echo "Schema validation failed during live run." >&2
  exit 1
fi

for agent in gemini codex claude; do
  if ! grep -q "agent=$agent" "$LOG_FILE"; then
    echo "Agent was not called: $agent" >&2
    exit 1
  fi
  marker="$SMOKE_DIR/$agent.txt"
  if [[ ! -f "$marker" ]]; then
    echo "Missing marker file: $marker" >&2
    exit 1
  fi
  if ! grep -q "$agent" "$marker"; then
    echo "Marker file missing agent name: $marker" >&2
    exit 1
  fi
  if ! grep -Eq '^[0-9]{4}-[0-9]{2}-[0-9]{2}T' "$marker"; then
    echo "Marker file missing timestamp: $marker" >&2
    exit 1
  fi
done

if ! awk '/CALL / && $0 !~ /write_access=1/ {exit 1}' "$LOG_FILE"; then
  echo "One or more calls ran without write_access=1." >&2
  exit 1
fi

echo "Live tri-agent smoke succeeded. Markers at $SMOKE_DIR"
