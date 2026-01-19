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
CONFIG_PATH="$SMOKE_DIR/smoke_config.json"
PREFLIGHT_LOG="$SMOKE_DIR/claude_preflight.log"

python3 - <<'PY' "$WORKTREE/bridge/config.json" "$CONFIG_PATH"
import json
import sys

src, dst = sys.argv[1], sys.argv[2]
config = json.loads(open(src, "r", encoding="utf-8").read())
config["limits"]["max_total_calls"] = 3
config["limits"]["max_calls_per_agent"] = 1
config["limits"]["quota_retry_attempts"] = 1
with open(dst, "w", encoding="utf-8") as handle:
    json.dump(config, handle)
PY

set +e
LOOP_TIMEOUT_S=180
export FF_AGENT_SMOKE_DIR="$SMOKE_DIR"
export FF_SMOKE=1
export FF_STREAM_AGENT_OUTPUT=both
export GEMINI_TIMEOUT_S=60
export CLAUDE_TIMEOUT_S=60
export CLAUDE_HELP_TIMEOUT_S=5
export CODEX_TIMEOUT_S=60
export CODEX_ASK_FOR_APPROVAL=never

if ! "$WORKTREE/scripts/claude_preflight.sh" >"$PREFLIGHT_LOG" 2>&1; then
  echo "Claude preflight failed; see $PREFLIGHT_LOG" >&2
  cat "$PREFLIGHT_LOG" >&2
  exit 1
fi

if command -v timeout >/dev/null 2>&1; then
  timeout "${LOOP_TIMEOUT_S}s" python3 -u bridge/loop.py \
    --mode live \
    --config "$CONFIG_PATH" \
    --smoke-route claude,codex,gemini \
    --start-agent claude \
    --no-agent-branch | tee "$LOG_FILE"
else
  python3 -u bridge/loop.py \
    --mode live \
    --config "$CONFIG_PATH" \
    --smoke-route claude,codex,gemini \
    --start-agent claude \
    --no-agent-branch | tee "$LOG_FILE"
fi
rc=${PIPESTATUS[0]}
set -e

if [[ "$rc" -ne 0 ]]; then
  echo "Live tri-agent smoke failed (rc=$rc)." >&2
  exit 1
fi

if grep -q "\\[orchestrator\\] ERROR" "$LOG_FILE" || grep -q "\\[orchestrator\\] QUOTA" "$LOG_FILE"; then
  echo "Agent invocation failed during live run." >&2
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

python3 - <<'PY' "$WORKTREE" "$LOG_FILE"
import json
import re
import sys
from pathlib import Path

worktree = Path(sys.argv[1])
log_path = Path(sys.argv[2])
text = log_path.read_text(encoding="utf-8")
m = re.search(r"run_id=([0-9A-Za-zTZ]+)", text)
if not m:
    raise SystemExit("Failed to determine run_id from log.")
run_id = m.group(1)
calls_dir = worktree / "runs" / run_id / "calls"
claude_turn = None
for turn_path in sorted(calls_dir.glob("*/turn.json")):
    data = json.loads(turn_path.read_text(encoding="utf-8"))
    if data.get("agent") == "claude":
        claude_turn = data
        break
if claude_turn is None:
    raise SystemExit("Claude turn not found in calls.")
summary = str(claude_turn.get("summary", ""))
summary_l = summary.lower()
if "synthesized" in summary_l or "did not emit" in summary_l:
    raise SystemExit("Claude did not complete successfully.")
if "wrapper_status=ok" not in summary_l:
    raise SystemExit("Claude wrapper_status=ok missing in summary.")
if "auth_mode=subscription" not in summary_l:
    raise SystemExit("Claude auth_mode=subscription missing in summary.")
PY

echo "Live tri-agent smoke succeeded. Markers at $SMOKE_DIR"
