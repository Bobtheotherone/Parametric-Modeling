#!/usr/bin/env bash
set -euo pipefail

PROMPT_FILE="${1:?prompt_file}"
SCHEMA_FILE="${2:?schema_file}"
OUT_FILE="${3:?out_file}"

MODEL="${CLAUDE_MODEL:-claude-3-7-sonnet-latest}"
TIMEOUT_S="${CLAUDE_TIMEOUT_S:-180}"

prompt="$(cat "$PROMPT_FILE")"

# Claude Code v2+ expects --json-schema to be an *inline JSON schema string* (not a file path).
# Older versions may have accepted a file path. We support both, with a hard timeout so the
# orchestrator never deadlocks on interactive prompts / mis-invocations.
SCHEMA_JSON="$(
  python3 -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1])), separators=(",",":")))' \
    "$SCHEMA_FILE"
)"

_run() {
  # Usage: _run <outfile> <errfile> <claude args...>
  local outfile="$1"; shift
  local errfile="$1"; shift

  # Clear previous outputs
  : > "$outfile"
  : > "$errfile"

  if command -v timeout >/dev/null 2>&1; then
    timeout "${TIMEOUT_S}s" claude "$@" >"$outfile" 2>"$errfile"
  else
    claude "$@" >"$outfile" 2>"$errfile"
  fi
}

# We always disable tools and permission prompts to avoid headless hangs.
COMMON_ARGS=(
  -p "$prompt"
  --model "$MODEL"
  --no-session-persistence
  --permission-mode dontAsk
  --tools ""
)

ERR_FILE="${OUT_FILE}.stderr"

# 1) Preferred: inline schema JSON (works on Claude Code v2.1.12 per your test).
if _run "$OUT_FILE" "$ERR_FILE" "${COMMON_ARGS[@]}" --json-schema "$SCHEMA_JSON"; then
  # Claude should have printed the model output directly (text mode by default).
  cat "$OUT_FILE"
  exit 0
fi

# 2) Compatibility fallback: some versions may accept a schema *path*.
if _run "$OUT_FILE" "$ERR_FILE" "${COMMON_ARGS[@]}" --json-schema "$SCHEMA_FILE"; then
  cat "$OUT_FILE"
  exit 0
fi

# 3) Last resort: request JSON wrapper and extract the actual model text payload.
# (The orchestrator will validate against bridge/turn.schema.json itself.)
if _run "$OUT_FILE" "$ERR_FILE" "${COMMON_ARGS[@]}" --output-format json; then
  python3 - <<'PY' <"$OUT_FILE" >"$OUT_FILE.tmp" || true
import json, sys

raw = sys.stdin.read().strip()
if not raw:
    raise SystemExit(0)

try:
    obj = json.loads(raw)
except Exception:
    # If it's not valid JSON, pass through as-is.
    print(raw)
    raise SystemExit(0)

# Claude Code --output-format json typically wraps response in a result object:
# {"type":"result", ... , "result":"<MODEL_TEXT>"}.
for key in ("result", "response", "content", "output", "text", "message"):
    if isinstance(obj, dict) and key in obj and obj[key] is not None:
        val = obj[key]
        if isinstance(val, str):
            print(val)
        else:
            print(json.dumps(val))
        raise SystemExit(0)

# Fallback: print raw JSON wrapper
print(raw)
PY
  mv "$OUT_FILE.tmp" "$OUT_FILE"
  cat "$OUT_FILE"
  exit 0
fi

# If we got here, claude failed in all modes; surface stderr to help debugging.
# Keep stdout empty so the orchestrator treats it as a failure.
exit 1
