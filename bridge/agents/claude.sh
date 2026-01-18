#!/usr/bin/env bash
set -euo pipefail

PROMPT_FILE="${1:?prompt_file}"
SCHEMA_FILE="${2:?schema_file}"
OUT_FILE="${3:?out_file}"

MODEL="${CLAUDE_MODEL:-claude-3-7-sonnet-latest}"

prompt="$(cat "$PROMPT_FILE")"

# Claude Code schema enforcement has changed across versions. We try the strict path first,
# then fall back to a plain JSON output mode.
if claude -p "$prompt" --json-schema "$SCHEMA_FILE" --model "$MODEL" > "$OUT_FILE" 2>/dev/null; then
  cat "$OUT_FILE"
  exit 0
fi

# Fallback: request JSON output format and rely on the orchestrator to validate against schema.
claude -p "$prompt" --output-format json --model "$MODEL" > "$OUT_FILE"

# Some Claude Code versions wrap the response in an outer JSON object. Try to extract the
# actual model text if present; otherwise pass through.
python3 - <<'PY' < "$OUT_FILE" > "$OUT_FILE.tmp" || true
import json, sys

raw = sys.stdin.read()
try:
    obj = json.loads(raw)
except Exception:
    print(raw)
    raise SystemExit(0)

for key in ("response", "content", "output", "text"):
    if isinstance(obj, dict) and isinstance(obj.get(key), str) and obj.get(key).strip():
        print(obj[key])
        raise SystemExit(0)

print(raw)
PY

mv "$OUT_FILE.tmp" "$OUT_FILE"
cat "$OUT_FILE"
