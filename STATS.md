# Agent Strengths & Collaboration Notes

This orchestrator is configured for a **two-agent** workflow: **Codex** and **Claude**.

The intended pattern is simple:
- **Always hand off to the other agent** after each successful turn (unless `project_complete=true`).
- Use the handoff to get a second set of eyes: review, test-plan, edge cases, and prompt/schema compliance.

## Codex

**Best at:**
- Fast, accurate code changes (Python, shell, glue code)
- Refactors, bug fixes, keeping diffs tight
- Implementing CLI behavior and maintaining backward-compatible flags

**When to hand off to Claude:**
- After implementing a change, for review and edge-case analysis
- When you need a careful spec check against prompt/schema/validation rules

## Claude

**Best at:**
- Careful review, reasoning about edge cases and failure modes
- Making prompts/system instructions robust and consistent with schema
- Writing or validating test plans and safety checks

**When to hand off to Codex:**
- After review/analysis, for implementation
- When changes require multi-file edits or tricky code surgery

## Handoff heuristics

In this repo, **collaboration is enforced**:
- If you are Codex, set `next_agent` to `claude`.
- If you are Claude, set `next_agent` to `codex`.

If you believe the other agent is out of budget or repeatedly failing, note that in `delegate_rationale` and still propose the best alternate plan.

## Stats references

Use `stats_refs` to cite which section you relied on (e.g., `"STATS.md#Codex"`).
