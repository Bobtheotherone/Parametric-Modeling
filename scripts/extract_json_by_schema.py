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

import json
import re
import sys
from pathlib import Path
from typing import Any, List, Optional

try:
    import jsonschema
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False


def strip_fences(s: str) -> str:
    """Remove markdown code fences from a string."""
    s = s.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return s


def extract_balanced_json(s: str, start: int = 0) -> tuple[Optional[str], int]:
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
                    return s[idx:i+1], i+1

    return None, start


def find_all_json_objects(text: str) -> List[str]:
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


def try_parse_json(s: str) -> Optional[Any]:
    """Try to parse a string as JSON."""
    try:
        return json.loads(s)
    except Exception:
        return None


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


def extract_from_claude_stream(text: str, schema: dict) -> Optional[dict]:
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
    """
    candidates: List[dict] = []

    # First, try to parse the entire text as JSON sequence
    decoder = json.JSONDecoder()
    idx = 0
    n = len(text)
    parsed_objects: List[Any] = []

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
    flattened: List[Any] = []
    for obj in parsed_objects:
        if isinstance(obj, list):
            flattened.extend(obj)
        else:
            flattened.append(obj)

    # Check each object
    for obj in flattened:
        if not isinstance(obj, dict):
            continue

        # Check if this object directly validates against schema
        if validate_against_schema(obj, schema):
            candidates.append(obj)
            continue

        # Check if it's a Claude CLI event with embedded JSON
        obj_type = obj.get("type", "")

        # type="result" with "result" string field
        if obj_type == "result":
            result_str = obj.get("result")
            if isinstance(result_str, str) and result_str.strip():
                inner = try_parse_json(strip_fences(result_str))
                if isinstance(inner, dict) and validate_against_schema(inner, schema):
                    candidates.append(inner)

        # type="assistant" with message.content[*].text
        elif obj_type == "assistant":
            message = obj.get("message", {})
            content = message.get("content", [])
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_content = block.get("text", "")
                    if text_content:
                        inner = try_parse_json(strip_fences(text_content))
                        if isinstance(inner, dict) and validate_against_schema(inner, schema):
                            candidates.append(inner)

    # If no candidates found from parsed objects, try extracting balanced JSON
    if not candidates:
        all_objects = find_all_json_objects(text)
        for obj_str in all_objects:
            obj = try_parse_json(obj_str)
            if isinstance(obj, dict) and validate_against_schema(obj, schema):
                candidates.append(obj)

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
                encoding="utf-8"
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
