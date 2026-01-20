#!/usr/bin/env python3
"""Extract a JSON object matching a given schema from a raw text stream.

This script is designed to robustly extract valid JSON from Claude CLI output,
which may contain multiple JSON objects, streaming events, or embedded JSON strings.

Usage:
    extract_json_by_schema.py <raw_stream_file> <schema_file> <out_file>

Returns:
    Exit 0 if a valid JSON object was found and written.
    Exit 1 if no valid JSON object was found.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

try:
    import jsonschema

    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False


# -----------------------------
# Task Plan Normalization
# -----------------------------

# Valid keys for task items per task_plan.schema.json
TASK_ITEM_VALID_KEYS = {"id", "title", "description", "preferred_agent", "estimated_intensity", "locks", "depends_on", "solo"}

# Valid top-level keys for task plan
TASK_PLAN_VALID_KEYS = {"milestone_id", "max_parallel_tasks", "rationale", "tasks"}


def is_task_plan_schema(schema: dict) -> bool:
    """Detect if schema is for ParallelTaskPlan."""
    # Check by title
    if schema.get("title") == "ParallelTaskPlan":
        return True
    # Check by required keys
    required = set(schema.get("required", []))
    task_plan_required = {"milestone_id", "max_parallel_tasks", "tasks", "rationale"}
    return task_plan_required.issubset(required)


def normalize_task_plan(obj: Any) -> Any:
    """Normalize a task plan object to match expected schema.

    Handles common model drift issues:
    - "intensity" -> "estimated_intensity"
    - "dependencies" -> "depends_on"
    - "agent" -> "preferred_agent"
    - Missing depends_on, locks, solo fields
    - Wrapped in {"task_plan": {...}}
    - Extra unknown keys (which fail additionalProperties=false)

    Returns the normalized object (modified in place).
    """
    if not isinstance(obj, dict):
        return obj

    # Unwrap if nested in {"task_plan": {...}}
    if "task_plan" in obj and isinstance(obj["task_plan"], dict):
        inner = obj["task_plan"]
        # Check if inner looks like a task plan
        if "tasks" in inner or "milestone_id" in inner:
            obj = inner

    # Normalize tasks array
    tasks = obj.get("tasks", [])
    if isinstance(tasks, list):
        for task in tasks:
            if not isinstance(task, dict):
                continue

            # Key renames
            if "intensity" in task and "estimated_intensity" not in task:
                task["estimated_intensity"] = task.pop("intensity")

            if "dependencies" in task and "depends_on" not in task:
                task["depends_on"] = task.pop("dependencies")

            if "agent" in task and "preferred_agent" not in task:
                task["preferred_agent"] = task.pop("agent")

            # Default values for missing required keys
            if "depends_on" not in task:
                task["depends_on"] = []

            if "locks" not in task:
                task["locks"] = []

            if "solo" not in task:
                task["solo"] = False

            # Ensure arrays are actually arrays
            if not isinstance(task.get("depends_on"), list):
                task["depends_on"] = []
            if not isinstance(task.get("locks"), list):
                task["locks"] = []

            # Remove unknown keys (additionalProperties=false)
            unknown_keys = set(task.keys()) - TASK_ITEM_VALID_KEYS
            for key in unknown_keys:
                del task[key]

    # Ensure top-level required fields exist
    if "milestone_id" not in obj:
        obj["milestone_id"] = "M0"
    if "max_parallel_tasks" not in obj:
        obj["max_parallel_tasks"] = 4
    if "rationale" not in obj:
        obj["rationale"] = ""
    if "tasks" not in obj:
        obj["tasks"] = []

    # Remove unknown top-level keys (additionalProperties=false)
    unknown_top_keys = set(obj.keys()) - TASK_PLAN_VALID_KEYS
    for key in unknown_top_keys:
        del obj[key]

    return obj


def strip_fences(s: str) -> str:
    """Remove markdown code fences from a string.

    Handles both:
    - Strings that start with ``` (entire content is fenced)
    - Strings with prose before/after fenced content (extract fenced part)
    """
    s = s.strip()

    # Case 1: Entire string starts with fence
    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()

    # Case 2: Fence is embedded in prose - extract the fenced content
    # Look for ```json or ``` followed by content
    fence_start = s.find("```json")
    if fence_start == -1:
        fence_start = s.find("```")

    if fence_start != -1:
        # Find where the fence opening line ends
        newline_after_fence = s.find("\n", fence_start)
        if newline_after_fence == -1:
            return s

        # Find the closing fence
        fence_close = s.find("```", newline_after_fence)
        if fence_close != -1:
            # Extract content between fences
            return s[newline_after_fence + 1 : fence_close].strip()

    return s


def extract_balanced_json(s: str, start: int = 0) -> tuple[str | None, int]:
    """Extract a balanced JSON object starting from position start.

    Returns (json_string, end_position) or (None, start) if no object found.
    """
    idx = s.find("{", start)
    if idx == -1:
        return None, start

    depth = 0
    in_str = False
    esc = False

    for i in range(idx, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[idx : i + 1], i + 1

    return None, start


def find_all_json_objects(text: str) -> list[str]:
    """Find all top-level JSON objects in a text stream."""
    objects = []
    pos = 0
    while pos < len(text):
        obj_str, end_pos = extract_balanced_json(text, pos)
        if obj_str is None:
            break
        objects.append(obj_str)
        pos = end_pos
    return objects


def try_parse_json(s: str) -> Any | None:
    """Try to parse a string as JSON."""
    try:
        return json.loads(s)
    except Exception:
        return None


def extract_json_from_text(text: str) -> list[Any]:
    """Extract all valid JSON objects from a text string.

    Tries multiple strategies:
    1. Direct JSON parsing of the whole string
    2. Stripping markdown fences and parsing
    3. Extracting balanced JSON objects from the text

    Returns a list of parsed JSON objects (may be empty).
    """
    results = []

    # Strategy 1: Direct parse
    direct = try_parse_json(text)
    if direct is not None:
        results.append(direct)
        return results  # If direct parse works, use it

    # Strategy 2: Strip fences and parse
    stripped = strip_fences(text)
    if stripped != text:  # Only try if stripping changed something
        parsed = try_parse_json(stripped)
        if parsed is not None:
            results.append(parsed)
            return results

    # Strategy 3: Extract balanced JSON objects
    pos = 0
    while pos < len(text):
        obj_str, end_pos = extract_balanced_json(text, pos)
        if obj_str is None:
            break
        parsed = try_parse_json(obj_str)
        if parsed is not None:
            results.append(parsed)
        pos = end_pos

    return results


def validate_against_schema(obj: Any, schema: dict) -> bool:
    """Validate an object against a JSON schema."""
    if not HAS_JSONSCHEMA:
        # If jsonschema is not available, just check it's a dict with some expected keys
        if not isinstance(obj, dict):
            return False
        # Basic check: ensure at least one required key from schema exists
        required = schema.get("required", [])
        if required:
            return any(k in obj for k in required)
        return True

    try:
        jsonschema.validate(instance=obj, schema=schema)
        return True
    except jsonschema.ValidationError:
        return False


def _try_validate_with_normalization(obj: Any, schema: dict, is_task_plan: bool) -> dict | None:
    """Try to validate an object, normalizing task plans if needed.

    Returns the (possibly normalized) object if valid, None otherwise.
    """
    if not isinstance(obj, dict):
        return None

    # First try direct validation
    if validate_against_schema(obj, schema):
        return obj

    # If it's a task plan schema, try normalization
    if is_task_plan:
        # Make a copy to avoid mutating original during exploration
        normalized = copy.deepcopy(obj)
        normalized = normalize_task_plan(normalized)
        if validate_against_schema(normalized, schema):
            return normalized

    return None


def extract_from_claude_stream(text: str, schema: dict) -> dict | None:
    """Extract a valid JSON object from Claude CLI stream output.

    Claude CLI may output:
    - Multiple JSON objects (streaming events)
    - JSON with type="result" containing a "result" string field
    - JSON with type="assistant" containing message.content[*].text
    - Direct JSON objects

    Priority:
    1. Direct objects that validate against schema (prefer last one)
    2. Result events with embedded JSON in "result" field
    3. Assistant messages with JSON in text content

    For task plan schemas, normalization is applied to handle common model drift:
    - "intensity" -> "estimated_intensity"
    - "dependencies" -> "depends_on"
    - Missing fields get defaults
    """
    candidates: list[dict] = []
    is_task_plan = is_task_plan_schema(schema)

    # First, try to parse the entire text as JSON sequence
    decoder = json.JSONDecoder()
    idx = 0
    n = len(text)
    parsed_objects: list[Any] = []

    while idx < n:
        # Skip whitespace
        while idx < n and text[idx] in " \t\n\r":
            idx += 1
        if idx >= n:
            break

        try:
            obj, end = decoder.raw_decode(text, idx)
            parsed_objects.append(obj)
            idx = end
        except json.JSONDecodeError:
            idx += 1

    # Flatten arrays (Claude outputs arrays of events)
    flattened: list[Any] = []
    for obj in parsed_objects:
        if isinstance(obj, list):
            flattened.extend(obj)
        else:
            flattened.append(obj)

    # Check each object
    for obj in flattened:
        if not isinstance(obj, dict):
            continue

        # Check if this object directly validates (with normalization for task plans)
        validated = _try_validate_with_normalization(obj, schema, is_task_plan)
        if validated is not None:
            candidates.append(validated)
            continue

        # Check if it's a Claude CLI event with embedded JSON
        obj_type = obj.get("type", "")

        # type="result" with "result" string field
        if obj_type == "result":
            result_str = obj.get("result")
            if isinstance(result_str, str) and result_str.strip():
                # Try multiple extraction strategies
                extracted = extract_json_from_text(result_str)
                for inner in extracted:
                    if isinstance(inner, dict):
                        validated = _try_validate_with_normalization(inner, schema, is_task_plan)
                        if validated is not None:
                            candidates.append(validated)

        # type="assistant" with message.content[*].text
        elif obj_type == "assistant":
            message = obj.get("message", {})
            content = message.get("content", [])
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_content = block.get("text", "")
                    if text_content:
                        # Try multiple extraction strategies
                        extracted = extract_json_from_text(text_content)
                        for inner in extracted:
                            if isinstance(inner, dict):
                                validated = _try_validate_with_normalization(inner, schema, is_task_plan)
                                if validated is not None:
                                    candidates.append(validated)

    # If no candidates found from parsed objects, try extracting balanced JSON
    if not candidates:
        all_objects = find_all_json_objects(text)
        for obj_str in all_objects:
            obj = try_parse_json(obj_str)
            if isinstance(obj, dict):
                validated = _try_validate_with_normalization(obj, schema, is_task_plan)
                if validated is not None:
                    candidates.append(validated)

    # Return the last valid candidate (most likely to be the final answer)
    return candidates[-1] if candidates else None


def main() -> int:
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <raw_stream_file> <schema_file> <out_file>", file=sys.stderr)
        return 1

    raw_path = Path(sys.argv[1])
    schema_path = Path(sys.argv[2])
    out_path = Path(sys.argv[3])

    if not raw_path.exists():
        print(f"ERROR: Raw stream file not found: {raw_path}", file=sys.stderr)
        return 1

    if not schema_path.exists():
        print(f"ERROR: Schema file not found: {schema_path}", file=sys.stderr)
        return 1

    try:
        text = raw_path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"ERROR: Failed to read raw stream: {e}", file=sys.stderr)
        return 1

    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"ERROR: Failed to load schema: {e}", file=sys.stderr)
        return 1

    result = extract_from_claude_stream(text, schema)

    if result is None:
        print("ERROR: No valid JSON object found matching schema", file=sys.stderr)
        # Save debug info
        debug_path = out_path.parent / f"{out_path.stem}.extract_debug.txt"
        try:
            debug_path.write_text(
                f"Schema: {schema_path}\n"
                f"Required keys: {schema.get('required', [])}\n"
                f"Raw text length: {len(text)}\n"
                f"Raw text preview:\n{text[:2000]}\n",
                encoding="utf-8",
            )
            print(f"Debug info saved to: {debug_path}", file=sys.stderr)
        except Exception:
            pass
        return 1

    try:
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"SUCCESS: Valid JSON extracted to {out_path}", file=sys.stderr)
        return 0
    except Exception as e:
        print(f"ERROR: Failed to write output: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
