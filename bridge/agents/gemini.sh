#!/usr/bin/env bash
set -euo pipefail

PROMPT_FILE="${1:?prompt_file}"
SCHEMA_FILE="${2:?schema_file}" # unused (Gemini CLI doesn't enforce JSON-schema output)
OUT_FILE="${3:?out_file}"

MODEL="${GEMINI_MODEL:-gemini-3-pro-preview}"

prompt="$(cat "$PROMPT_FILE")"

# Gemini CLI JSON output includes a wrapper object; we extract the actual model response.
raw_json="$(gemini --model "$MODEL" --output-format json --prompt "$prompt")"

response="$(
  python3 - <<'PY'
import json, sys
payload = json.loads(sys.stdin.read())
# Gemini CLI's JSON output includes a "response" field.
# If the shape ever changes, adjust here.
print(payload.get("response", ""))
PY
  <<<"$raw_json"
)"

# Write and emit the extracted response (which should itself be JSON matching bridge/turn.schema.json)
printf '%s' "$response" > "$OUT_FILE"
cat "$OUT_FILE"
