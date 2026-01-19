#!/usr/bin/env bash
set -euo pipefail

PROMPT_FILE="${1:?prompt_file}"
SCHEMA_FILE="${2:?schema_file}"
OUT_FILE="${3:?out_file}"

MODEL="${CODEX_MODEL:-gpt-5.2-codex}"
REASONING_EFFORT="${CODEX_REASONING_EFFORT:-xhigh}"
WRITE_ACCESS="${WRITE_ACCESS:-0}"
SMOKE_DIR="${FF_AGENT_SMOKE_DIR:-}"

# Different Codex CLI versions use either `-c key=value` or a dedicated flag.
# Official docs call out `-c` for per-invocation overrides.
CODEX_CONFIG_FLAG="${CODEX_CONFIG_FLAG:--c}"

# Optional knobs (override in env) to match your local Codex CLI version.
# Examples:
#   export CODEX_SANDBOX=workspace-write
#   export CODEX_ASK_FOR_APPROVAL=never
CODEX_SANDBOX="${CODEX_SANDBOX:-}"
CODEX_ASK_FOR_APPROVAL="${CODEX_ASK_FOR_APPROVAL:-}"
CODEX_EXTRA_GLOBAL_FLAGS="${CODEX_EXTRA_GLOBAL_FLAGS:-}"
CODEX_EXTRA_EXEC_FLAGS="${CODEX_EXTRA_EXEC_FLAGS:-}"

cmd=(codex)

if [[ -n "$SMOKE_DIR" && "$WRITE_ACCESS" == "1" ]]; then
  mkdir -p "$SMOKE_DIR"
  printf '%s %s\n' "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" "codex" > "$SMOKE_DIR/codex.txt"
fi

# These global flags are supported by many Codex CLI versions.
if [[ -n "$MODEL" ]]; then
  cmd+=(--model "$MODEL")
fi
if [[ -n "$REASONING_EFFORT" ]]; then
  cmd+=($CODEX_CONFIG_FLAG "reasoning.effort=$REASONING_EFFORT")
fi
if [[ -n "$CODEX_SANDBOX" ]]; then
  cmd+=(--sandbox "$CODEX_SANDBOX")
fi
if [[ -n "$CODEX_ASK_FOR_APPROVAL" ]]; then
  cmd+=(--ask-for-approval "$CODEX_ASK_FOR_APPROVAL")
fi
if [[ -n "$CODEX_EXTRA_GLOBAL_FLAGS" ]]; then
  # shellcheck disable=SC2206
  cmd+=($CODEX_EXTRA_GLOBAL_FLAGS)
fi

cmd+=(exec)

# On newer Codex CLIs, this enables file edits + commands.
if [[ "$WRITE_ACCESS" == "1" ]]; then
  cmd+=(--full-auto)
fi

if [[ -n "$CODEX_EXTRA_EXEC_FLAGS" ]]; then
  # shellcheck disable=SC2206
  cmd+=($CODEX_EXTRA_EXEC_FLAGS)
fi

cmd+=(--output-schema "$SCHEMA_FILE" -o "$OUT_FILE" -)

"${cmd[@]}" < "$PROMPT_FILE" 1>/dev/null

cat "$OUT_FILE"
