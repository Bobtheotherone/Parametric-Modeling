"""Turn normalization - Design-doc-agnostic output normalization.

This module handles normalizing messy agent output to the full Turn schema,
making the orchestrator robust to:
- Prose/markdown around JSON
- Missing or incorrect invariant fields (agent, milestone_id, stats_refs)
- Partial payloads with only required fields
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any

AGENTS: tuple[str, ...] = ("codex", "claude")


@dataclasses.dataclass
class NormalizationResult:
    """Result of turn normalization."""

    success: bool
    turn: dict[str, Any] | None
    warnings: list[str]
    error: str | None = None


class TurnNormalizer:
    """Normalizes agent output to full Turn schema.

    This class extracts payload from messy agent output and fills invariant fields,
    making the orchestrator robust to:
    - Prose/markdown around JSON
    - Missing or incorrect invariant fields (agent, milestone_id, stats_refs)
    - Partial payloads with only required fields
    """

    PAYLOAD_REQUIRED = {"summary", "work_completed", "project_complete"}
    VALID_PHASES = ("plan", "implement", "verify", "finalize")
    VALID_AGENTS = ("codex", "claude")

    def __init__(
        self,
        expected_agent: str,
        expected_milestone_id: str,
        stats_id_set: set[str],
        default_phase: str = "implement",
    ):
        self.expected_agent = expected_agent
        self.expected_milestone_id = expected_milestone_id
        self.stats_id_set = stats_id_set
        self.default_phase = default_phase

    def normalize(self, raw_output: str) -> NormalizationResult:
        """Normalize raw agent output to full Turn.

        Returns NormalizationResult with:
        - success: True if normalization succeeded
        - turn: The normalized turn dict (if success)
        - warnings: List of auto-corrections made
        - error: Error message (if not success)
        """
        warnings: list[str] = []

        # Step 1: Extract JSON payload from raw output
        payload = self._extract_payload(raw_output)
        if payload is None:
            return NormalizationResult(
                success=False,
                turn=None,
                warnings=warnings,
                error="Cannot extract JSON payload from output",
            )

        # Step 2: Check required payload fields
        missing = self.PAYLOAD_REQUIRED - set(payload.keys())
        if missing:
            return NormalizationResult(
                success=False,
                turn=None,
                warnings=warnings,
                error=f"Missing required payload fields: {sorted(missing)}",
            )

        # Step 3: Build normalized turn with invariant overrides
        turn = self._build_normalized_turn(payload, warnings)

        return NormalizationResult(
            success=True,
            turn=turn,
            warnings=warnings,
            error=None,
        )

    def _extract_payload(self, raw: str) -> dict[str, Any] | None:
        """Extract JSON payload from raw output, tolerating prose/UI junk."""
        raw = raw.strip()
        if not raw:
            return None

        # Try direct JSON parse first
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

        # Strip markdown fences
        if raw.startswith("```"):
            lines = raw.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            raw = "\n".join(lines).strip()
            try:
                obj = json.loads(raw)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass

        # Extract first balanced JSON object
        start = raw.find("{")
        if start == -1:
            return None

        depth = 0
        in_str = False
        esc = False

        for i in range(start, len(raw)):
            ch = raw[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = raw[start : i + 1]
                        try:
                            obj = json.loads(candidate)
                            if isinstance(obj, dict):
                                return obj
                        except json.JSONDecodeError:
                            pass
                        break

        return None

    def _build_normalized_turn(self, payload: dict[str, Any], warnings: list[str]) -> dict[str, Any]:
        """Build normalized turn from payload, overriding invariants."""
        turn: dict[str, Any] = {}

        # INVARIANT: agent - always use expected, log if different
        payload_agent = payload.get("agent")
        if payload_agent and payload_agent != self.expected_agent:
            warnings.append(f"agent mismatch: payload={payload_agent}, expected={self.expected_agent} (auto-corrected)")
        turn["agent"] = self.expected_agent

        # INVARIANT: milestone_id - always use expected, log if different
        payload_milestone = payload.get("milestone_id")
        if payload_milestone and str(payload_milestone) != self.expected_milestone_id:
            warnings.append(
                f"milestone_id mismatch: payload={payload_milestone}, expected={self.expected_milestone_id} (auto-corrected)"
            )
        turn["milestone_id"] = self.expected_milestone_id

        # INVARIANT: stats_refs - derive from agent if missing/invalid
        payload_stats = payload.get("stats_refs", [])
        if isinstance(payload_stats, list):
            valid_stats = [s for s in payload_stats if isinstance(s, str) and s in self.stats_id_set]
        else:
            valid_stats = []
        if not valid_stats:
            default_stat = "CL-1" if self.expected_agent == "claude" else "CX-1"
            if default_stat in self.stats_id_set:
                valid_stats = [default_stat]
            elif self.stats_id_set:
                valid_stats = [sorted(self.stats_id_set)[0]]
            else:
                valid_stats = ["CL-1"]  # Fallback
            warnings.append(f"stats_refs defaulted to {valid_stats}")
        turn["stats_refs"] = valid_stats

        # phase - use payload if valid, else default
        payload_phase = payload.get("phase")
        if payload_phase in self.VALID_PHASES:
            turn["phase"] = payload_phase
        else:
            turn["phase"] = self.default_phase
            if payload_phase:
                warnings.append(f"invalid phase '{payload_phase}', using '{self.default_phase}'")

        # Required payload fields (already validated)
        turn["summary"] = str(payload.get("summary", ""))
        turn["work_completed"] = bool(payload.get("work_completed", False))

        # Optional fields with defaults
        turn["project_complete"] = bool(payload.get("project_complete", False))
        turn["gates_passed"] = self._to_str_list(payload.get("gates_passed", []))

        # requirement_progress
        rp = payload.get("requirement_progress", {})
        if not isinstance(rp, dict):
            rp = {}
        turn["requirement_progress"] = {
            "covered_req_ids": self._to_str_list(rp.get("covered_req_ids", [])),
            "tests_added_or_modified": self._to_str_list(rp.get("tests_added_or_modified", [])),
            "commands_run": self._to_str_list(rp.get("commands_run", [])),
        }

        # next_agent - use payload if valid, else same agent
        payload_next = payload.get("next_agent")
        if payload_next in self.VALID_AGENTS:
            turn["next_agent"] = payload_next
        else:
            turn["next_agent"] = self.expected_agent

        turn["next_prompt"] = str(payload.get("next_prompt", ""))
        turn["delegate_rationale"] = str(payload.get("delegate_rationale", ""))

        # needs_write_access - default True for coding agents
        nwa = payload.get("needs_write_access")
        if isinstance(nwa, bool):
            turn["needs_write_access"] = nwa
        else:
            turn["needs_write_access"] = True

        # artifacts
        turn["artifacts"] = self._normalize_artifacts(payload.get("artifacts", []))

        return turn

    def _to_str_list(self, val: Any) -> list[str]:
        """Convert to list of strings."""
        if not isinstance(val, list):
            return []
        return [str(x).strip() for x in val if isinstance(x, str) and x.strip()]

    def _normalize_artifacts(self, val: Any) -> list[dict[str, str]]:
        """Normalize artifacts list."""
        if not isinstance(val, list):
            return []
        result = []
        for item in val:
            if isinstance(item, dict):
                path = item.get("path")
                desc = item.get("description")
                if isinstance(path, str) and path.strip() and isinstance(desc, str):
                    result.append({"path": path.strip(), "description": desc.strip()})
        return result


def normalize_agent_output(
    raw_output: str,
    expected_agent: str,
    expected_milestone_id: str,
    stats_id_set: set[str],
    default_phase: str = "implement",
) -> NormalizationResult:
    """Convenience function to normalize agent output.

    This is the main entry point for the TurnNormalizer.
    """
    normalizer = TurnNormalizer(
        expected_agent=expected_agent,
        expected_milestone_id=expected_milestone_id,
        stats_id_set=stats_id_set,
        default_phase=default_phase,
    )
    return normalizer.normalize(raw_output)


def validate_turn_lenient(
    obj: Any,
    *,
    expected_agent: str,
    expected_milestone_id: str | None = None,
    stats_id_set: set[str],
) -> tuple[bool, str, list[str]]:
    """Lenient turn validation that auto-corrects mismatches.

    Returns: (is_valid, error_or_ok, warnings)

    Unlike strict validation, this:
    - Auto-corrects agent/milestone_id mismatches (with warnings)
    - Does not fail on these mismatches
    """
    warnings: list[str] = []

    if not isinstance(obj, dict):
        return False, "turn is not an object", warnings

    required_keys = [
        "agent",
        "milestone_id",
        "phase",
        "work_completed",
        "project_complete",
        "summary",
        "gates_passed",
        "requirement_progress",
        "next_agent",
        "next_prompt",
        "delegate_rationale",
        "stats_refs",
        "needs_write_access",
        "artifacts",
    ]

    for k in required_keys:
        if k not in obj:
            return False, f"missing key: {k}", warnings

    # Agent mismatch - WARN but don't fail, auto-correct
    if obj.get("agent") != expected_agent:
        warnings.append(f"agent mismatch: expected {expected_agent}, got {obj.get('agent')} (auto-corrected)")
        obj["agent"] = expected_agent

    # Milestone mismatch - WARN but don't fail, auto-correct
    if expected_milestone_id is not None and str(obj.get("milestone_id")) != expected_milestone_id:
        warnings.append(
            f"milestone_id mismatch: expected {expected_milestone_id}, got {obj.get('milestone_id')} (auto-corrected)"
        )
        obj["milestone_id"] = expected_milestone_id

    if obj["agent"] not in AGENTS or obj["next_agent"] not in AGENTS:
        return False, "invalid agent id in agent/next_agent", warnings

    if obj.get("phase") not in ("plan", "implement", "verify", "finalize"):
        return False, "invalid phase", warnings

    if not isinstance(obj["work_completed"], bool) or not isinstance(obj["project_complete"], bool):
        return False, "work_completed/project_complete must be boolean", warnings

    for k in ("summary", "next_prompt", "delegate_rationale"):
        if not isinstance(obj.get(k), str):
            return False, f"{k} must be a string", warnings

    if not isinstance(obj["needs_write_access"], bool):
        return False, "needs_write_access must be boolean", warnings

    if not isinstance(obj["gates_passed"], list) or not all(isinstance(x, str) for x in obj["gates_passed"]):
        return False, "gates_passed must be array of strings", warnings

    if not isinstance(obj["stats_refs"], list) or not all(isinstance(x, str) for x in obj["stats_refs"]):
        return False, "stats_refs must be array of strings", warnings
    if not obj["stats_refs"]:
        return False, "stats_refs is empty", warnings
    unknown = [x for x in obj["stats_refs"] if x not in stats_id_set]
    if unknown:
        # Auto-correct unknown stats_refs
        valid_refs = [x for x in obj["stats_refs"] if x in stats_id_set]
        if not valid_refs:
            default_ref = "CL-1" if expected_agent == "claude" else "CX-1"
            valid_refs = [default_ref] if default_ref in stats_id_set else list(stats_id_set)[:1] or ["CL-1"]
        warnings.append(f"unknown stats_refs {unknown} removed, using {valid_refs}")
        obj["stats_refs"] = valid_refs

    rp = obj.get("requirement_progress")
    if not isinstance(rp, dict):
        return False, "requirement_progress must be object", warnings
    for k in ("covered_req_ids", "tests_added_or_modified", "commands_run"):
        if k not in rp:
            return False, f"requirement_progress missing key: {k}", warnings
        if not isinstance(rp[k], list) or not all(isinstance(x, str) for x in rp[k]):
            return False, f"requirement_progress.{k} must be array of strings", warnings

    if not isinstance(obj["artifacts"], list):
        return False, "artifacts must be an array", warnings
    for i, a in enumerate(obj["artifacts"]):
        if not isinstance(a, dict):
            return False, f"artifact[{i}] must be object", warnings
        if set(a.keys()) != {"path", "description"}:
            return (
                False,
                f"artifact[{i}] must have exactly keys: path, description",
                warnings,
            )
        if not isinstance(a.get("path"), str) or not isinstance(a.get("description"), str):
            return False, f"artifact[{i}] path/description must be strings", warnings

    extra = set(obj.keys()) - set(required_keys)
    if extra:
        return False, f"unexpected keys present: {sorted(extra)}", warnings

    return True, "ok", warnings
