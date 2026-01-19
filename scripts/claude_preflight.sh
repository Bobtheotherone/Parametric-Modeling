#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CLAUDE_BIN="${CLAUDE_BIN:-claude}"

if ! command -v "$CLAUDE_BIN" >/dev/null 2>&1; then
  echo "[claude-preflight] ERROR: claude binary not found: $CLAUDE_BIN" >&2
  exit 1
fi

if [[ -z "${ANTHROPIC_API_KEY:-}" && -z "${CLAUDE_API_KEY:-}" ]]; then
  echo "[claude-preflight] ERROR: missing Claude credentials (set ANTHROPIC_API_KEY or CLAUDE_API_KEY)" >&2
  exit 2
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

PROMPT_PATH="$TMP_DIR/prompt.txt"
OUT_PATH="$TMP_DIR/out.json"
SCHEMA_PATH="$ROOT/bridge/turn.schema.json"

printf '%s\n' '**Milestone:** M0' 'CL-1' > "$PROMPT_PATH"

if ! "$ROOT/bridge/agents/claude.sh" "$PROMPT_PATH" "$SCHEMA_PATH" "$OUT_PATH" >/dev/null 2>&1; then
  echo "[claude-preflight] ERROR: claude wrapper execution failed" >&2
  exit 3
fi

python3 - <<'PY' "$OUT_PATH" "$SCHEMA_PATH"
import json
import sys

import jsonschema  # type: ignore[import-untyped]

payload = json.loads(open(sys.argv[1], "r", encoding="utf-8").read())
schema = json.loads(open(sys.argv[2], "r", encoding="utf-8").read())
jsonschema.validate(instance=payload, schema=schema)
print(json.dumps(payload, indent=2, sort_keys=True))
PY

echo "[claude-preflight] OK"
