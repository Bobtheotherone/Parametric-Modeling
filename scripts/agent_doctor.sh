#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

fail=0

check_cmd() {
  local name="$1"
  shift
  if ! command -v "$name" >/dev/null 2>&1; then
    echo "[agent-doctor] MISSING: $name" >&2
    fail=1
    return
  fi
  echo "[agent-doctor] $name: $*"
  if ! "$name" "$@"; then
    echo "[agent-doctor] FAILED: $name $*" >&2
    fail=1
  fi
}

check_cmd codex --version

if command -v gemini >/dev/null 2>&1; then
  if ! gemini --version >/dev/null 2>&1; then
    check_cmd gemini --help
  else
    check_cmd gemini --version
  fi
else
  echo "[agent-doctor] MISSING: gemini" >&2
  fail=1
fi

check_cmd claude --version
check_cmd claude --help

SCHEMA_PATH="$ROOT/bridge/turn.schema.json"

validate_turn() {
  local path="$1"
  python3 - <<'PY' "$path" "$SCHEMA_PATH"
import json
import sys

import jsonschema  # type: ignore[import-untyped]

payload = json.loads(open(sys.argv[1], "r", encoding="utf-8").read())
schema = json.loads(open(sys.argv[2], "r", encoding="utf-8").read())
jsonschema.validate(instance=payload, schema=schema)
PY
}

run_wrapper() {
  local agent="$1"
  local prompt="$2"

  local tmpdir
  tmpdir="$(mktemp -d)"
  trap 'rm -rf "$tmpdir"' RETURN

  local prompt_path="$tmpdir/prompt.txt"
  local out_path="$tmpdir/out.json"

  printf '%s\n' "$prompt" > "$prompt_path"

  local wrapper="$ROOT/bridge/agents/${agent}.sh"
  if ! "$wrapper" "$prompt_path" "$SCHEMA_PATH" "$out_path" >/dev/null 2>&1; then
    echo "[agent-doctor] wrapper failed: $agent" >&2
    fail=1
    return
  fi

  if ! validate_turn "$out_path"; then
    echo "[agent-doctor] schema validation failed: $agent" >&2
    fail=1
  else
    echo "[agent-doctor] wrapper ok: $agent"
  fi
}

if [[ -n "${OPENAI_API_KEY:-}" ]]; then
  run_wrapper codex $'**Milestone:** M0\nCX-1\n'
else
  echo "[agent-doctor] SKIP: codex wrapper (OPENAI_API_KEY missing)"
fi

if [[ -n "${GEMINI_API_KEY:-}" || -n "${GOOGLE_API_KEY:-}" ]]; then
  run_wrapper gemini $'**Milestone:** M0\nGM-1\n'
else
  echo "[agent-doctor] SKIP: gemini wrapper (GEMINI_API_KEY/GOOGLE_API_KEY missing)"
fi

if [[ -n "${ANTHROPIC_API_KEY:-}" || -n "${CLAUDE_API_KEY:-}" ]]; then
  run_wrapper claude $'**Milestone:** M0\nCL-1\n'
else
  echo "[agent-doctor] SKIP: claude wrapper (ANTHROPIC_API_KEY/CLAUDE_API_KEY missing)"
fi

exit "$fail"
