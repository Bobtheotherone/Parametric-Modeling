#!/usr/bin/env bash
set -euo pipefail

# Live run: requires Codex CLI, Gemini CLI, and Claude Code.

python3 -u bridge/loop.py --mode live --start-agent gemini "$@"
