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

# Sync locked dependencies + optional dev extras.
# --frozen: use the lockfile exactly (no re-lock)
# --extra dev: install [project.optional-dependencies] dev group
uv sync --frozen --extra dev

# Install the project itself in editable mode (src-layout).
# --no-deps: dependencies already synced above; only install our package.
# This ensures `python -m tools.*` can import formula_foundry without PYTHONPATH hacks.
uv pip install -e . --no-deps

echo "Environment ready at $VENV_PATH"
