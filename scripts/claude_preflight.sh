#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CLAUDE_BIN="${CLAUDE_BIN:-claude}"

if ! command -v "$CLAUDE_BIN" >/dev/null 2>&1; then
  echo "[claude-preflight] ERROR: claude binary not found: $CLAUDE_BIN" >&2
  exit 1
fi

if [[ -n "${ANTHROPIC_API_KEY:-}" || -n "${CLAUDE_API_KEY:-}" ]]; then
  echo "[claude-preflight] WARNING: ANTHROPIC_API_KEY/CLAUDE_API_KEY is set; Claude Code will use API billing instead of subscription." >&2
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

PROMPT_PATH="$TMP_DIR/prompt.txt"
OUT_PATH="$TMP_DIR/out.json"
SCHEMA_PATH="$ROOT/bridge/turn.schema.json"
WRAP_LOG="$TMP_DIR/claude_wrapper.log"

printf '%s\n' '**Milestone:** M0' 'CL-1' > "$PROMPT_PATH"

export FF_SMOKE=1
export CLAUDE_TIMEOUT_S="${CLAUDE_PREFLIGHT_TIMEOUT_S:-25}"
export CLAUDE_HELP_TIMEOUT_S="${CLAUDE_PREFLIGHT_HELP_TIMEOUT_S:-5}"

if ! "$ROOT/bridge/agents/claude.sh" "$PROMPT_PATH" "$SCHEMA_PATH" "$OUT_PATH" >"$WRAP_LOG" 2>&1; then
  echo "[claude-preflight] ERROR: claude wrapper execution failed" >&2
  cat "$WRAP_LOG" >&2
  exit 3
fi

set +e
python3 - "$OUT_PATH" "$SCHEMA_PATH" <<'PY'
import json
import sys

import jsonschema  # type: ignore[import-untyped]

payload = json.loads(open(sys.argv[1], "r", encoding="utf-8").read())
schema = json.loads(open(sys.argv[2], "r", encoding="utf-8").read())
jsonschema.validate(instance=payload, schema=schema)
summary = str(payload.get("summary", ""))
summary_l = summary.lower()
if "wrapper_status=ok" not in summary_l or "auth_mode=subscription" not in summary_l:
    raise SystemExit(12)
print(json.dumps(payload, indent=2, sort_keys=True))
PY
status=$?
set -e

if [[ "$status" -ne 0 ]]; then
  cat >&2 <<'EOF'
[claude-preflight] ERROR: Claude subscription auth not confirmed.
[claude-preflight] To use Pro/Max subscription:
  - Run /logout inside Claude Code
  - Run: claude update
  - Restart your terminal
  - Run: claude and pick your subscription account
[claude-preflight] NOTE: ANTHROPIC_API_KEY/CLAUDE_API_KEY env vars override subscription and force API billing.
EOF
  exit 4
fi

echo "[claude-preflight] OK"
