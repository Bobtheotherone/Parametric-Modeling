#!/usr/bin/env bash
set -euo pipefail

# Initializes a local git repo so agent CLIs have a sandbox.
# Also sets a local user.name/user.email if missing.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [ ! -d .git ]; then
  git init
fi

if ! git config user.email >/dev/null; then
  git config user.email "agent@example.com"
fi
if ! git config user.name >/dev/null; then
  git config user.name "Agent Runner"
fi

# Initial commit if none exists.
if ! git rev-parse HEAD >/dev/null 2>&1; then
  git add -A
  git commit -m "chore: initial state" >/dev/null
fi

echo "Git repo ready: $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD)"
