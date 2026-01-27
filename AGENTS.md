\# AGENTS.md — Multi-Agent Orchestrator Protocol (Tri-Agent Reliability)



This repository is an \*\*engineering-first\*\* "formula foundry" substrate. The tri-agent loop (Codex + Gemini + Claude) operates under a parallel orchestrator that assigns tasks to isolated worktrees. Each agent must follow the rules below to avoid conflicts and maintain quality.



You are the \*\*assigned agent for THIS task/worktree\*\*. Other agents may be running concurrently in their own worktrees.



---



\## 0) Non-Negotiable Operating Rules



\### 0.1 Single-writer-per-worktree rule

\- You are the only writer for THIS task and THIS worktree. Do not modify files outside your declared scope.

\- Do \*\*not\*\* offload work to other agents unless you are blocked. If you must delegate, use the Delegation Packet template (§0.5).

\- All loop tests must be run in a way that \*\*cannot modify tracked files\*\* (see §1).



\### 0.2 Two-mode workflow: EDIT mode vs TEST mode

To avoid “modifying code while testing the loop,” you must strictly separate:



\*\*EDIT mode\*\*

\- You may modify code/tests/docs.

\- You must keep commits small and atomic.

\- You must run `python -m tools.verify --strict-git` before and after each commit.



\*\*TEST mode\*\*

\- \*\*No code modifications allowed.\*\*

\- You may only run commands that create ignored artifacts (e.g., `runs/`, `.pytest\_cache/`).

\- You must begin TEST mode with a \*\*clean working tree\*\*:

&nbsp; - `git status --porcelain` must be empty.

\- If you need to run multi-agent loop tests, do it in a \*\*separate worktree\*\* (recommended) or from a clean commit.



\### 0.3 Worktree isolation (preferred)

When you need to test the orchestrator/loop:

\- Create a disposable worktree from the current HEAD commit:

&nbsp; - `git worktree add -f /tmp/ff\_loop\_test HEAD`

\- Run loop tests \*\*only\*\* in that worktree.

\- Remove the worktree afterwards:

&nbsp; - `git worktree remove -f /tmp/ff\_loop\_test`



This guarantees loop testing never collides with uncommitted edits.



\### 0.4 “tools.verify” is the source of truth

Milestone completion is defined by `python -m tools.verify --strict-git` \*\*and nothing else\*\*.

If `tools.verify` does not enforce a requirement, you must fix `tools.verify` and/or tests so it does.



---





\## 0.5) Behavioral Directives (Must Follow)

\- No green-by-any-means: do not skip/xfail/delete tests or disable verification to make CI green.
\- No delegating work you can do yourself; only delegate when blocked and use the Delegation Packet below.
\- No scaffolding/shims/incomplete work; no TODO-only or placeholder changes.
\- No regressions: add targeted tests for behavior changes and record exact commands run.

\### Mandatory Work Report Template (required in every response)

\- Summary: what changed and why
\- Files changed: list of paths
\- Commands run: exact shell commands
\- Tests run: exact test commands
\- Results: pass/fail with key output
\- Follow-ups: remaining risks or next steps

\### Delegation Packet Template (only if unavoidable)

\- Objective:
\- Scope boundary:
\- Current state:
\- Constraints:
\- Reproduction steps (if a bug):
\- Exact commands to run:
\- Acceptance criteria:
\- Deliverables:
\- Risks / rollback plan:
\## 1) Your Mission (What “Done” Means)



You must complete:

1\) \*\*M0 is truly complete\*\*:

&nbsp;  - No “stub” tests remain for requirements in M0’s Test Matrix.

&nbsp;  - `tools.verify` enforces M0’s substrate gates (smoke + repro-check) when the current DESIGN\_DOCUMENT is M0.

&nbsp;  - Deterministic environment claims are real: GPU-first dependency path is locked (CuPy policy is explicit and enforceable).

&nbsp;  - Container and supply-chain pinning requirements are enforced by real tests (not placeholders).

&nbsp;  - Required docs exist and are verified by tests.

&nbsp;  - Hardware config matches the target laptop.



2\) \*\*Tri-agent loop reliability\*\*:

&nbsp;  - Gemini wrapper must always produce JSON conforming to `bridge/turn.schema.json` (even on errors).

&nbsp;  - Claude wrapper must be investigated and fixed so it works in the loop (or fails with actionable, schema-valid output).

&nbsp;  - Add \*\*offline, deterministic pytest coverage\*\* for wrappers using stub executables so CI does not require real API keys.



3\) \*\*No-repo-mutation loop testing\*\*:

&nbsp;  - Provide a reliable way to test the loop without touching tracked files:

&nbsp;    - either by a `--readonly` / env-based read-only mode

&nbsp;    - or by worktree-based isolation + enforced clean-tree policy



---



\## 2) Execution Plan (High-Level)



\### Step A — Baseline audit (TEST mode)

1\) `python -m tools.verify --strict-git`

2\) Identify all M0 requirements mapped to:

&nbsp;  - skipped tests

&nbsp;  - pass-only tests

&nbsp;  - or missing enforcement in `tools.verify`

3\) Identify Gemini/Claude wrapper failures by reproducing them in a \*\*worktree\*\* and capturing stderr/stdout.



\### Step B — Close M0 gaps (EDIT mode)

Implement the missing M0 contract items, in this priority order:



\#### B1) Eliminate M0 stub tests (hard requirement)

\- Replace any `@pytest.mark.skip(reason="M0 stub")` for tests referenced in the M0 Test Matrix with real tests.

\- Ensure `tools.verify` fails if any mapped test is skipped or is a placeholder.



\#### B2) Make `tools.verify` milestone-aware (hard requirement)

\- Parse `DESIGN\_DOCUMENT.md` and detect the milestone ID.

\- When milestone is `M0`, `tools.verify` must run:

&nbsp; - `python -m tools.m0 smoke`

&nbsp; - `python -m tools.m0 repro-check`

\- Add tests proving that verify invokes those gates.



\#### B3) Container contract (hard requirement)

\- Add `docker/Dockerfile` that is base-image \*\*digest pinned\*\* (`FROM …@sha256:…`).

\- Add tests that:

&nbsp; - confirm digest pinning exists

&nbsp; - confirm the image is intended to run the M0 gates (command surface / doc contract)



\#### B4) Supply-chain pinning for GitHub Actions (conditional)

\- If `.github/workflows/\*` exists, enforce:

&nbsp; - `uses:` must be pinned to full commit SHA (no floating tags)

\- If workflows do not exist, test must pass.



\#### B5) GPU-first determinism (policy becomes real)

\- Decide a policy:

&nbsp; - Either CuPy is a required dependency for the default install \*\*or\*\*

&nbsp; - CuPy is required for a `gpu` extra and M0 requires that extra for “target laptop reproducibility”

\- Update `pyproject.toml` and lockfile accordingly.

\- Add a test that inspects the lockfile so the policy can’t silently regress.



\#### B6) Fix hardware config

\- Update `config/hardware.yaml` so it reflects the actual target laptop:

&nbsp; - `vram\_gb: 24`

&nbsp; - CPU cores per your intended scheduler assumptions



\#### B7) Docs exist and are verifiable

\- Add docs for:

&nbsp; - reproduction procedure

&nbsp; - determinism guarantees + limitations

\- Replace doc stub tests with content-asserting tests.



\### Step C — Fix Gemini + Claude wrappers (EDIT mode)

\#### C1) Gemini wrapper

\- Guarantee a schema-valid `Turn` JSON even when:

&nbsp; - the CLI returns malformed JSON

&nbsp; - the process errors

&nbsp; - the model returns free-text

\- Implement a “wrap invalid output into a schema-valid error turn” rule.

\- Add pytest coverage using a fake gemini executable in PATH.



\#### C2) Claude wrapper

\- Reproduce the error deterministically:

&nbsp; - record the exact command invoked

&nbsp; - capture stderr

&nbsp; - capture exit code

\- Fix the wrapper to support the installed CLI’s actual flags.

&nbsp; - Add a capability-discovery path: `claude --help` / version probing

&nbsp; - Add an override env var for the command path and flags (e.g., `CLAUDE\_BIN`, `CLAUDE\_ARGS\_JSON\_MODE`)

\- Ensure failures are reported as schema-valid turns (not raw text).

\- Add pytest coverage using a fake claude executable that simulates both success and failure.



\### Step D — Loop testing without repo mutation (TEST mode)

\- In a disposable worktree:

&nbsp; - run a short orchestrator session configured to do \*\*no writes\*\*

&nbsp; - ensure all three agents produce schema-valid turns and handoffs work

\- Do not modify code during this step; if fixes are required, return to EDIT mode, commit, then retest.



---



\## 3) Commit Discipline



\- Work on a dedicated branch: `codex/m0-hardening` (or similar).

\- Make atomic commits:

&nbsp; 1) verify/m0 gate integration

&nbsp; 2) docker + tests

&nbsp; 3) workflow pinning + tests

&nbsp; 4) dependency lock updates

&nbsp; 5) docs + tests

&nbsp; 6) gemini wrapper + tests

&nbsp; 7) claude wrapper + tests

&nbsp; 8) readonly loop test harness (if needed)

\- After each commit:

&nbsp; - `python -m tools.verify --strict-git`

&nbsp; - `git status --porcelain` must be clean



---



\## 4) Testing Strategy (No Flakes)



\- Prefer deterministic unit tests.

\- For wrapper tests:

&nbsp; - create a temporary executable script in `tmp\_path`

&nbsp; - point the wrapper at it via env vars or PATH manipulation

&nbsp; - validate JSON schema output and error handling

\- Avoid time-based asserts that fail on slower machines.

&nbsp; - Instead assert that `tools.verify` runs the correct subcommands (mock `subprocess.run`).



---



\## 5) Definition of Done (Gate to Start M1)



You may declare “M0 complete / start M1” only when:

\- `python -m tools.verify --strict-git` passes

\- No mapped M0 tests are skipped or placeholder

\- `tools.verify` runs M0 smoke + repro-check automatically under M0

\- Container pinning + supply-chain pinning + docs are real and tested

\- GPU dependency policy is explicit and enforced

\- Hardware config matches target laptop

\- Gemini + Claude + Codex wrappers all:

&nbsp; - produce schema-valid turns

&nbsp; - work in the orchestrator loop (or produce actionable schema-valid failure turns)

\- Loop tests are runnable in TEST mode without touching tracked files (worktree isolation or readonly mode)



---



\## 6) Expected Behavior Summary



\- Be skeptical: if a gate can be bypassed, close the bypass.

\- Prefer correctness and determinism over convenience.

\- Never “paper over” missing functionality by skipping tests.

\- Never modify code while running loop tests; always commit first or use a worktree.



