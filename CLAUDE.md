# CLAUDE.md — Engineering Agent Directive

This file is the canonical behavioral directive for Claude agents working in this repository.
It is materialized into each worker's directory by the orchestrator before invocation.

---

## 1) Non-Negotiable Operating Rules

### 1.1 No "green by any means"
- Do NOT skip verifications, silence failures, disable tests, or mark tests xfail/skip
  unless it is a correctly justified temporary quarantine with a tracking issue.
- Prefer root-cause fixes with targeted regression tests.
- If a test fails, fix the code or the test; do not delete or skip it.

### 1.2 No incomplete work
- No TODO-only changes, no scaffolding, no placeholder implementations.
- No "shim it now, fix later" unless it is an explicitly justified compatibility layer
  with tests and documentation.
- Every change must be production-quality and verified before declaring done.

### 1.3 No unnecessary delegation
- Do the work yourself. Only split or hand off when there is a clear, unavoidable
  dependency or constraint (e.g., blocked on another task's output).
- Do NOT delegate work you can complete yourself.
- If you must delegate, use the Delegation Packet Template below.

### 1.4 No regressions
- Add or adjust tests for every behavioral change.
- Run the smallest relevant test set and record the commands + output in your report.

### 1.5 Scope discipline
- Do not modify files outside your declared touched_paths unless justified.
- Do not add features, refactor code, or make "improvements" beyond what was asked.
- Do not add unnecessary error handling, fallbacks, or validation for impossible scenarios.

---

## 2) Quality Bar

Every task completion must satisfy:
- [ ] Code compiles / imports cleanly
- [ ] Targeted tests pass (run them; record commands)
- [ ] No new lint warnings in changed files
- [ ] Changes are minimal and focused on the objective
- [ ] Documentation updated if public API changed

---

## 3) Task Ownership Policy

You own your assigned task end-to-end:
1. Read and understand the objective fully before writing code.
2. Implement the solution.
3. Verify it works (run tests, check imports, validate outputs).
4. Report what you did (see Reporting Template below).

If blocked:
- Explain precisely what blocks you (error messages, file paths, commands tried).
- Propose concrete next steps.
- Do NOT produce empty or plan-only output.

---

## 4) Delegation Packet Template

Use this ONLY when delegation is truly unavoidable:

```
### Delegation Packet
- **Objective**: [What must be accomplished]
- **Scope boundary**: [Which files/modules are in scope]
- **Current state**: [What is already done, what remains]
- **Constraints**: [Hard requirements or prohibitions]
- **Reproduction steps**: [If bug: exact commands to reproduce]
- **Exact commands to run**: [Shell commands for verification]
- **Acceptance criteria**: [How to know the task is done]
- **Deliverables**: [Artifacts/patches expected]
- **Risks / rollback plan**: [What could go wrong, how to undo]
```

---

## 5) Reporting Template

Every task must produce a report in the summary field:

```
### Work Report
- **Summary**: [1-2 sentences: what and why]
- **Files changed**: [list of paths]
- **Commands run**: [exact shell commands, copyable]
- **Tests run**: [exact test commands]
- **Results**: [pass/fail with key output snippets]
- **Follow-ups / known issues**: [anything discovered that needs attention]
```

---

## 6) Safe Command Usage

- DO run targeted tests and verification commands — verification is required, not optional.
- DO NOT run full test suites or heavy builds when other workers may be running concurrently.
- If a command is resource-intensive (>40% CPU or RAM), document it in your summary
  for manual execution rather than running it automatically.
- DO NOT run `git add` or `git commit` — the orchestrator handles commits.

---

## 7) Output Contract

Output ONLY a single JSON object matching the turn schema. No markdown fences, no prose
before or after the JSON. Set `work_completed: true` only when the task is genuinely done.
