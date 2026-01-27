# Formula Foundry — Tri-Agent Orchestrator (Codex + Gemini + Claude)

This is an **agentic coding harness** tailored to Formula Foundry milestones (M0–M9).

Core idea:
1) You paste the current milestone's spec into `DESIGN_DOCUMENT.md`.
2) The agents iterate until:
   - every requirement in the doc is mapped to pytest(s),
   - the mapped tests exist and pass,
   - `python -m tools.verify --strict-git` passes,
   - the repo is clean and committed.

Unlike generic agent templates, this one is intentionally **over-engineered** for:
- proof-driven progress (requirements ⇄ tests ⇄ verification),
- reproducible runs with full audit trails under `runs/`,
- minimal “LLM hallucination surface area” (the orchestrator runs the gates).

---

## Quickstart

### 0) Prereqs
- Python 3.10+
- Git
- Node.js (for Codex CLI + Gemini CLI)
- Installed CLIs:
  - Codex CLI (`npm i -g @openai/codex`)
  - Gemini CLI (see https://github.com/google-gemini/gemini-cli)
  - Claude Code CLI (see https://code.claude.com/docs/en/cli-reference)

### 1) Install Python dev deps

```bash
python -m pip install -e ".[dev]"
```

### 2) Init a git repo

```bash
./bootstrap_repo.sh
```

### 3) Run a deterministic mock campaign (no LLM calls)

```bash
./run_mock.sh
```

### 4) Run a live campaign

```bash
./run_live.sh
```

---

## How the workflow enforces “specs proven by pytest”

### Required DESIGN_DOCUMENT contract
`tools/spec_lint.py` enforces:
- milestone ID present
- a list of normative requirement IDs
- a Definition of Done section
- a Test Matrix mapping each requirement ID to pytest node ids

### Required gates
`tools/verify.py` is the single source of truth gate runner:
- spec_lint (including test-node existence)
- pytest
- optional: ruff + mypy
- git_guard (secret scan + status)

The orchestrator refuses to stop unless the completion gates pass.

---

## Where to put each milestone doc

- Work-in-progress doc: `DESIGN_DOCUMENT.md`
- Templates for M0..M9: `docs/milestones/`

---

## Configuration

- `bridge/config.json` controls call caps, fallback order, and agent scripts.
- Model overrides via env vars:
  - `CODEX_MODEL`, `CODEX_REASONING_EFFORT`
  - `GEMINI_MODEL`
  - `CLAUDE_MODEL`

### Planner profiles

The parallel orchestrator supports multiple planning profiles:

- `balanced` (default): conservative planning and retries.
- `throughput`: maximize parallel utilization with backfill tasks.
- `engineering`: fail-fast, no backfill, and per-task reports.

Enable engineering mode:

```bash
./run_parallel.sh --planner-profile engineering
```

Or via environment:

```bash
ORCH_PLANNER_PROFILE=engineering ./run_parallel.sh
```

---

## Safety

- `.env` is gitignored.
- `tools/git_guard.py` performs a cheap local scan for common credential patterns.
