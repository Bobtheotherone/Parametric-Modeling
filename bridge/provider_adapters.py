#!/usr/bin/env python3
"""Provider Output Adapters for unified structured output handling.

This module provides adapters for different AI providers (OpenAI, Claude) that
ensure consistent Turn objects are produced regardless of provider-specific
output formats and quirks.

Key features:
- OpenAI: Uses strict JSON schema outputs when supported
- Claude: Uses tool/function calling for structured output
- Unified internal Turn object via ProviderOutputAdapter + TurnNormalizer
- Bounded repair fallback for occasional malformed outputs
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# Import TurnNormalizer for consistent normalization
# This will be imported from loop.py at runtime to avoid circular imports


@dataclass
class AdapterResult:
    """Result from provider output adapter."""
    success: bool
    turn: dict[str, Any] | None
    raw_output: str
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    needs_retry: bool = False
    needs_repair: bool = False


@dataclass
class RepairResult:
    """Result from repair attempt."""
    success: bool
    repaired_output: str
    error: str | None = None


class ProviderOutputAdapter(ABC):
    """Base class for provider-specific output adapters."""

    @abstractmethod
    def extract_turn(self, raw_output: str, expected_agent: str, expected_milestone_id: str) -> AdapterResult:
        """Extract Turn object from provider-specific output format.

        Args:
            raw_output: Raw output from the provider
            expected_agent: Expected agent name for normalization
            expected_milestone_id: Expected milestone ID for normalization

        Returns:
            AdapterResult with extracted turn or error information
        """
        pass

    @abstractmethod
    def get_schema_config(self, schema: dict) -> dict:
        """Get provider-specific schema configuration.

        Args:
            schema: JSON schema for the Turn object

        Returns:
            Provider-specific configuration for structured output
        """
        pass


class OpenAIOutputAdapter(ProviderOutputAdapter):
    """Adapter for OpenAI (Codex) structured output.

    OpenAI supports strict JSON schema outputs via response_format.
    This adapter handles:
    - Strict JSON schema validation
    - Extra prose before/after JSON extraction
    - Missing key recovery
    - Type coercion for common mismatches
    """

    def extract_turn(self, raw_output: str, expected_agent: str, expected_milestone_id: str) -> AdapterResult:
        warnings = []
        raw_output = raw_output.strip()

        if not raw_output:
            return AdapterResult(
                success=False,
                turn=None,
                raw_output=raw_output,
                error="Empty output from OpenAI",
                needs_retry=True,
            )

        # OpenAI with strict schema should produce clean JSON
        # But sometimes there's extra prose - try to extract JSON
        turn_obj = self._try_parse_json(raw_output)

        if turn_obj is None:
            # Try extracting from code fences
            turn_obj = self._extract_from_fences(raw_output)
            if turn_obj is not None:
                warnings.append("Extracted JSON from code fences")

        if turn_obj is None:
            # Try extracting first balanced JSON object
            turn_obj = self._extract_balanced_json(raw_output)
            if turn_obj is not None:
                warnings.append("Extracted balanced JSON object from prose")

        if turn_obj is None:
            return AdapterResult(
                success=False,
                turn=None,
                raw_output=raw_output,
                error="Cannot extract JSON from OpenAI output",
                needs_repair=True,
            )

        # Apply invariant overrides
        turn_obj = self._apply_invariants(turn_obj, expected_agent, expected_milestone_id, warnings)

        # Validate required fields
        missing = self._check_required_fields(turn_obj)
        if missing:
            return AdapterResult(
                success=False,
                turn=turn_obj,
                raw_output=raw_output,
                warnings=warnings,
                error=f"Missing required fields: {missing}",
                needs_repair=True,
            )

        return AdapterResult(
            success=True,
            turn=turn_obj,
            raw_output=raw_output,
            warnings=warnings,
        )

    def get_schema_config(self, schema: dict) -> dict:
        """Get OpenAI-specific schema configuration.

        Returns configuration for response_format json_schema strict mode.
        """
        return {
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "turn",
                    "strict": True,
                    "schema": schema,
                }
            }
        }

    def _try_parse_json(self, text: str) -> dict | None:
        try:
            obj = json.loads(text)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None

    def _extract_from_fences(self, text: str) -> dict | None:
        if not text.startswith("```"):
            return None
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return self._try_parse_json("\n".join(lines))

    def _extract_balanced_json(self, text: str) -> dict | None:
        start = text.find("{")
        if start == -1:
            return None

        depth = 0
        in_str = False
        esc = False

        for i in range(start, len(text)):
            ch = text[i]
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
                        candidate = text[start:i + 1]
                        return self._try_parse_json(candidate)
        return None

    def _apply_invariants(
        self,
        turn: dict,
        expected_agent: str,
        expected_milestone_id: str,
        warnings: list[str],
    ) -> dict:
        """Apply invariant field overrides."""
        # Agent must match expected
        if turn.get("agent") != expected_agent:
            warnings.append(f"agent corrected: {turn.get('agent')} -> {expected_agent}")
            turn["agent"] = expected_agent

        # Milestone must match expected
        if turn.get("milestone_id") != expected_milestone_id:
            warnings.append(f"milestone_id corrected: {turn.get('milestone_id')} -> {expected_milestone_id}")
            turn["milestone_id"] = expected_milestone_id

        return turn

    def _check_required_fields(self, turn: dict) -> list[str]:
        """Check for missing required fields."""
        required = [
            "summary", "work_completed", "project_complete",
            "phase", "next_agent", "next_prompt", "delegate_rationale",
            "stats_refs", "needs_write_access", "artifacts",
            "gates_passed", "requirement_progress",
        ]
        return [f for f in required if f not in turn]


class ClaudeOutputAdapter(ProviderOutputAdapter):
    """Adapter for Claude structured output.

    Claude uses tool/function calling for structured output.
    This adapter handles:
    - Tool result extraction from Claude CLI JSON stream
    - Assistant message content parsing
    - Wrapper metadata stripping
    - Extra text around JSON
    """

    def extract_turn(self, raw_output: str, expected_agent: str, expected_milestone_id: str) -> AdapterResult:
        warnings = []
        raw_output = raw_output.strip()

        if not raw_output:
            return AdapterResult(
                success=False,
                turn=None,
                raw_output=raw_output,
                error="Empty output from Claude",
                needs_retry=True,
            )

        # Claude CLI outputs JSON stream with multiple events
        turn_obj = self._extract_from_json_stream(raw_output)

        if turn_obj is not None:
            warnings.append("Extracted from Claude JSON stream")
        else:
            # Try direct JSON parse
            turn_obj = self._try_parse_json(raw_output)

        if turn_obj is None:
            # Try extracting from code fences
            turn_obj = self._extract_from_fences(raw_output)
            if turn_obj is not None:
                warnings.append("Extracted JSON from code fences")

        if turn_obj is None:
            # Try balanced JSON extraction
            turn_obj = self._extract_balanced_json(raw_output)
            if turn_obj is not None:
                warnings.append("Extracted balanced JSON object")

        if turn_obj is None:
            return AdapterResult(
                success=False,
                turn=None,
                raw_output=raw_output,
                error="Cannot extract JSON from Claude output",
                needs_repair=True,
            )

        # Apply invariant overrides
        turn_obj = self._apply_invariants(turn_obj, expected_agent, expected_milestone_id, warnings)

        # Validate required fields
        missing = self._check_required_fields(turn_obj)
        if missing:
            # Try to fill defaults for some fields
            turn_obj = self._fill_defaults(turn_obj, warnings)
            missing = self._check_required_fields(turn_obj)
            if missing:
                return AdapterResult(
                    success=False,
                    turn=turn_obj,
                    raw_output=raw_output,
                    warnings=warnings,
                    error=f"Missing required fields: {missing}",
                    needs_repair=True,
                )

        return AdapterResult(
            success=True,
            turn=turn_obj,
            raw_output=raw_output,
            warnings=warnings,
        )

    def get_schema_config(self, schema: dict) -> dict:
        """Get Claude-specific schema configuration.

        Returns configuration for tool/function calling.
        """
        return {
            "tools": [{
                "name": "submit_turn",
                "description": "Submit the turn result with structured output",
                "input_schema": schema,
            }],
            "tool_choice": {"type": "tool", "name": "submit_turn"},
        }

    def _extract_from_json_stream(self, text: str) -> dict | None:
        """Extract turn from Claude CLI JSON stream output."""
        objects = self._parse_json_sequence(text)
        if not objects:
            return None

        # Look for type=="result" events
        for obj in reversed(objects):
            if isinstance(obj, dict) and obj.get("type") == "result":
                result_str = obj.get("result")
                if isinstance(result_str, str):
                    turn = self._try_parse_json(self._strip_fences(result_str))
                    if turn is not None:
                        return turn

        # Look for assistant messages
        for obj in reversed(objects):
            if isinstance(obj, dict) and obj.get("type") == "assistant":
                message = obj.get("message", {})
                content = message.get("content", [])
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_content = block.get("text", "")
                        turn = self._try_parse_json(self._strip_fences(text_content))
                        if turn is not None:
                            return turn

        return None

    def _parse_json_sequence(self, text: str) -> list:
        """Parse multiple JSON objects from a stream."""
        objects = []
        decoder = json.JSONDecoder()
        idx = 0
        n = len(text)

        while idx < n:
            while idx < n and text[idx] in " \t\n\r":
                idx += 1
            if idx >= n:
                break
            try:
                obj, end = decoder.raw_decode(text, idx)
                if isinstance(obj, list):
                    objects.extend(obj)
                else:
                    objects.append(obj)
                idx = end
            except json.JSONDecodeError:
                idx += 1

        return objects

    def _try_parse_json(self, text: str) -> dict | None:
        try:
            obj = json.loads(text)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None

    def _strip_fences(self, text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            return "\n".join(lines).strip()
        return text

    def _extract_from_fences(self, text: str) -> dict | None:
        stripped = self._strip_fences(text)
        if stripped != text:
            return self._try_parse_json(stripped)
        return None

    def _extract_balanced_json(self, text: str) -> dict | None:
        start = text.find("{")
        if start == -1:
            return None

        depth = 0
        in_str = False
        esc = False

        for i in range(start, len(text)):
            ch = text[i]
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
                        candidate = text[start:i + 1]
                        return self._try_parse_json(candidate)
        return None

    def _apply_invariants(
        self,
        turn: dict,
        expected_agent: str,
        expected_milestone_id: str,
        warnings: list[str],
    ) -> dict:
        """Apply invariant field overrides."""
        if turn.get("agent") != expected_agent:
            warnings.append(f"agent corrected: {turn.get('agent')} -> {expected_agent}")
            turn["agent"] = expected_agent

        if turn.get("milestone_id") != expected_milestone_id:
            warnings.append(f"milestone_id corrected: {turn.get('milestone_id')} -> {expected_milestone_id}")
            turn["milestone_id"] = expected_milestone_id

        return turn

    def _fill_defaults(self, turn: dict, warnings: list[str]) -> dict:
        """Fill default values for missing optional-ish fields."""
        defaults = {
            "gates_passed": [],
            "requirement_progress": {
                "covered_req_ids": [],
                "tests_added_or_modified": [],
                "commands_run": [],
            },
            "artifacts": [],
            "phase": "implement",
            "needs_write_access": True,
            "delegate_rationale": "",
            "next_prompt": "",
        }

        for key, value in defaults.items():
            if key not in turn:
                turn[key] = value
                warnings.append(f"Filled default for {key}")

        return turn

    def _check_required_fields(self, turn: dict) -> list[str]:
        """Check for missing required fields."""
        required = [
            "summary", "work_completed", "project_complete",
        ]
        return [f for f in required if f not in turn]


class OutputRepairService:
    """Service for repairing malformed outputs.

    Implements bounded repair: at most 1 repair attempt per output,
    then triggers task retry.
    """

    MAX_REPAIR_ATTEMPTS = 1

    def __init__(self, repair_prompt_builder: Callable[[str, str], str] | None = None):
        """Initialize repair service.

        Args:
            repair_prompt_builder: Optional function to build repair prompts
        """
        self.repair_prompt_builder = repair_prompt_builder or self._default_repair_prompt

    def attempt_repair(
        self,
        raw_output: str,
        error_message: str,
        repair_count: int,
    ) -> RepairResult:
        """Attempt to repair malformed output.

        Args:
            raw_output: The malformed output
            error_message: Error message describing the issue
            repair_count: Number of previous repair attempts

        Returns:
            RepairResult indicating success/failure
        """
        if repair_count >= self.MAX_REPAIR_ATTEMPTS:
            return RepairResult(
                success=False,
                repaired_output="",
                error="Max repair attempts exceeded",
            )

        # Try basic repairs
        repaired = self._try_basic_repairs(raw_output)
        if repaired:
            return RepairResult(success=True, repaired_output=repaired)

        # Can't repair without calling model
        return RepairResult(
            success=False,
            repaired_output="",
            error="Cannot repair without model call - trigger retry",
        )

    def _try_basic_repairs(self, raw_output: str) -> str | None:
        """Try basic text-based repairs without model calls."""
        # Strip common wrapper text
        patterns_to_strip = [
            (r"^Here(?:'s| is) (?:the|my|your) (?:JSON|response|output)[:\s]*", ""),
            (r"```json\s*", ""),
            (r"```\s*$", ""),
            (r"^\s*```\s*", ""),
        ]

        text = raw_output.strip()
        for pattern, replacement in patterns_to_strip:
            text = re.sub(pattern, replacement, text, flags=re.MULTILINE | re.IGNORECASE)
        text = text.strip()

        # Try to parse after stripping
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return json.dumps(obj)
        except json.JSONDecodeError:
            pass

        # Try extracting balanced JSON
        start = text.find("{")
        if start != -1:
            depth = 0
            in_str = False
            esc = False
            for i in range(start, len(text)):
                ch = text[i]
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
                            candidate = text[start:i + 1]
                            try:
                                obj = json.loads(candidate)
                                if isinstance(obj, dict):
                                    return json.dumps(obj)
                            except json.JSONDecodeError:
                                pass
                            break

        return None

    def _default_repair_prompt(self, raw_output: str, error_message: str) -> str:
        """Build default repair prompt."""
        return f"""The previous output was invalid JSON. Error: {error_message}

Original output (first 2000 chars):
{raw_output[:2000]}

Please output ONLY valid JSON matching the schema. No markdown, no code fences, no extra text."""


def get_adapter_for_agent(agent: str) -> ProviderOutputAdapter:
    """Get the appropriate adapter for an agent type.

    Args:
        agent: Agent name ("codex" or "claude")

    Returns:
        Appropriate ProviderOutputAdapter instance
    """
    if agent == "codex":
        return OpenAIOutputAdapter()
    elif agent == "claude":
        return ClaudeOutputAdapter()
    else:
        # Default to Claude adapter for unknown agents
        return ClaudeOutputAdapter()


def normalize_provider_output(
    raw_output: str,
    agent: str,
    expected_agent: str,
    expected_milestone_id: str,
    stats_id_set: set[str] | None = None,
) -> AdapterResult:
    """Normalize provider output using the appropriate adapter.

    This is the main entry point for provider-agnostic output normalization.

    Args:
        raw_output: Raw output from the provider
        agent: Provider/agent name ("codex" or "claude")
        expected_agent: Expected agent name for invariant override
        expected_milestone_id: Expected milestone ID for invariant override
        stats_id_set: Optional set of valid stats IDs

    Returns:
        AdapterResult with normalized turn or error information
    """
    adapter = get_adapter_for_agent(agent)
    result = adapter.extract_turn(raw_output, expected_agent, expected_milestone_id)

    # If successful and stats_id_set provided, validate/fix stats_refs
    if result.success and result.turn and stats_id_set:
        turn = result.turn
        stats_refs = turn.get("stats_refs", [])
        valid_refs = [s for s in stats_refs if s in stats_id_set]
        if not valid_refs:
            # Default stats ref
            default = "CL-1" if expected_agent == "claude" else "CX-1"
            if default in stats_id_set:
                valid_refs = [default]
            elif stats_id_set:
                valid_refs = [sorted(stats_id_set)[0]]
            else:
                valid_refs = ["CL-1"]  # Ultimate fallback
            result.warnings.append(f"stats_refs defaulted to {valid_refs}")
        turn["stats_refs"] = valid_refs

    return result
