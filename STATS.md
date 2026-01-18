# Agent Stats + Selection Guide (Formula Foundry)

This file is **part of the protocol**.

Before choosing `next_agent`, the current agent must:
1. Read this file.
2. Choose the best agent for the next subtask.
3. Output `stats_refs` that cite the relevant **IDs** below (e.g., `CX-2`, `GM-3`, `CL-4`).

If you do not cite IDs, the orchestrator may treat the handoff as invalid.

---

## Codex (OpenAI) — agent id: `codex`

Role in this repo: **Implementer + TDD loop driver**.

### Strength IDs
- **CX-1**: Turn a crisp requirement into code with correct interfaces and solid typing.
- **CX-2**: Iterate fast using commands/tests; fixes failures without thrashing.
- **CX-3**: Large mechanical edits and multi-file refactors while keeping the repo consistent.
- **CX-4**: GPU-first numerics: can refactor hot paths to CuPy/PyTorch and avoid CPU fallback.

### Best for
- Implementing milestone features, writing/adjusting pytest, fixing `tools/verify.py` failures.

### Not ideal for
- Turning ambiguous prose into a complete set of normative requirements + test matrix (delegate to Gemini).

---

## Gemini (Google) — agent id: `gemini`

Role in this repo: **Spec engineer + adversarial reviewer**.

### Strength IDs
- **GM-1**: Convert DESIGN_DOCUMENT.md into precise requirement IDs + measurable acceptance criteria.
- **GM-2**: Build a rigorous test matrix (Req → pytest node IDs) and identify missing gates.
- **GM-3**: Threat-modeling and adversarial thinking: corner cases, failure modes, non-functional constraints.
- **GM-4**: Delegation discipline: choose the next agent to minimize cost and maximize progress.

### Best for
- First pass on a milestone doc: normalize it to the required format, define gates, plan TDD.

### Not ideal for
- Large code edits (delegate to Codex/Claude).

---

## Claude Code (Anthropic) — agent id: `claude`

Role in this repo: **Verifier + refactor/polish + test hardening**.

### Strength IDs
- **CL-1**: Improves API ergonomics and readability without breaking behavior.
- **CL-2**: Writes robust tests (good assertions, parametrization, meaningful fixtures).
- **CL-3**: Consistency audits: naming, docs, edge-case coverage, and removing footguns.
- **CL-4**: Finalization pass: ensures gates pass, commits are clean, docs are accurate.

### Best for
- A verification/polish pass after Codex implementation, plus hardening tests and docs.

### Not ideal for
- Early architecture exploration when specs are unclear (delegate to Gemini first).

---

## Quick selection heuristics

- “Normalize a messy DESIGN_DOCUMENT into requirement IDs + test matrix” → `gemini` (GM-1..GM-3)
- “Implement the requirements and make tests pass” → `codex` (CX-1..CX-3)
- “Harden tests, refactor, final commit/cleanup” → `claude` (CL-2..CL-4)
