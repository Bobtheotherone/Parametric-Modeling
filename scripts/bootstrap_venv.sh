#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LOCKFILE="$ROOT/uv.lock"
if [ ! -f "$LOCKFILE" ]; then
  echo "uv.lock not found; run 'uv lock' to generate it." >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it with: python -m pip install uv" >&2
  exit 1
fi

VENV_PATH="${VENV_PATH:-$ROOT/.venv}"
export UV_PROJECT_ENVIRONMENT="$VENV_PATH"

uv venv "$VENV_PATH"

SYNC_FLAGS=(--frozen)
if uv sync --help 2>/dev/null | grep -q -- "--dev"; then
  SYNC_FLAGS+=(--dev)
elif uv sync --help 2>/dev/null | grep -q -- "--group"; then
  SYNC_FLAGS+=(--group dev)
fi

uv sync "${SYNC_FLAGS[@]}"

echo "Environment ready at $VENV_PATH"
