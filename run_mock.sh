#!/usr/bin/env bash
set -euo pipefail

# Deterministic run: uses bridge/mock_scenarios/milestone_demo.json, no LLM calls.

python3 -u bridge/loop.py --mode mock --start-agent gemini "$@"
