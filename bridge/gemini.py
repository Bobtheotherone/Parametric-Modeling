#!/usr/bin/env python3
"""Gemini wrapper that always emits schema-valid Turn JSON."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

try:
    from bridge.turns import build_error_turn
except Exception:  # pragma: no cover - import shim for direct execution
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT))
    from bridge.turns import build_error_turn  # type: ignore[import-not-found]

try:
    import jsonschema  # type: ignore[import-untyped]

    HAS_JSONSCHEMA = True
except Exception:
    HAS_JSONSCHEMA = False


MILESTONE_PATTERNS = [
    re.compile(r"\*\*Milestone:\*\*\s*(M\d+)\b", re.IGNORECASE),
    re.compile(r"\*\*Milestone\s+ID:\*\*\s*(M\d+)\b", re.IGNORECASE),
    re.compile(r"^Milestone:\s*(M\d+)\b", re.IGNORECASE | re.MULTILINE),
]

REQUIRED_KEYS = [
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


def main() -> int:
    args = _parse_args()
    prompt_text = _read_text(args.prompt_path)
    schema = _load_schema(args.schema_path)
    allowed_agents = _extract_enum(schema, "agent") or ["codex", "claude"]
    allowed_phases = _extract_enum(schema, "phase") or ["plan", "implement", "verify", "finalize"]

    agent_preferred = os.environ.get("GEMINI_AGENT", "gemini")
    agent_id = _select_from_allowed(agent_preferred, allowed_agents)
    milestone_id = _extract_milestone(prompt_text) or os.environ.get("MILESTONE_ID", "M0")

    stats_id_set = _load_stats_ids(_repo_root())
    stats_refs = _extract_stats_refs(prompt_text)
    needs_write_access = _env_truthy("WRITE_ACCESS") or _env_truthy("ORCH_WRITE_ACCESS")

    try:
        result = _run_gemini(prompt_text)
    except Exception as exc:
        payload = _build_error_payload(
            agent_id=agent_id,
            milestone_id=milestone_id,
            allowed_agents=allowed_agents,
            allowed_phases=allowed_phases,
            summary="Gemini wrapper error: CLI invocation failed",
            error_detail=str(exc),
            stats_refs=stats_refs,
            stats_id_set=stats_id_set,
            needs_write_access=needs_write_access,
        )
    else:
        if result.returncode != 0:
            payload = _build_error_payload(
                agent_id=agent_id,
                milestone_id=milestone_id,
                allowed_agents=allowed_agents,
                allowed_phases=allowed_phases,
                summary="Gemini wrapper error: non-zero exit",
                error_detail=_format_process_error(result),
                stats_refs=stats_refs,
                stats_id_set=stats_id_set,
                needs_write_access=needs_write_access,
            )
        else:
            raw_text = (result.stdout or "").strip()
            fallback_text = (result.stderr or "").strip()
            candidate = _extract_json_payload(raw_text) or _extract_json_payload(fallback_text)
            if candidate is None:
                payload = _build_error_payload(
                    agent_id=agent_id,
                    milestone_id=milestone_id,
                    allowed_agents=allowed_agents,
                    allowed_phases=allowed_phases,
                    summary="Gemini wrapper error: malformed output",
                    error_detail=_format_parse_error(raw_text or fallback_text),
                    stats_refs=stats_refs,
                    stats_id_set=stats_id_set,
                    needs_write_access=needs_write_access,
                )
            else:
                valid, err = _validate_turn(candidate, schema, allowed_agents, allowed_phases)
                if valid:
                    payload = candidate
                else:
                    payload = _build_error_payload(
                        agent_id=agent_id,
                        milestone_id=milestone_id,
                        allowed_agents=allowed_agents,
                        allowed_phases=allowed_phases,
                        summary="Gemini wrapper error: output failed schema validation",
                        error_detail=_format_validation_error(err, raw_text or fallback_text),
                        stats_refs=stats_refs,
                        stats_id_set=stats_id_set,
                        needs_write_access=needs_write_access,
                    )

    _write_json(args.out_path, payload)
    sys.stdout.write(json.dumps(payload, indent=2, ensure_ascii=True))
    sys.stdout.write("\n")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gemini wrapper")
    parser.add_argument("prompt_path", type=Path)
    parser.add_argument("schema_path", type=Path)
    parser.add_argument("out_path", type=Path)
    return parser.parse_args()


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _load_schema(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _extract_enum(schema: dict[str, Any] | None, key: str) -> list[str]:
    if not isinstance(schema, dict):
        return []
    props = schema.get("properties", {})
    if not isinstance(props, dict):
        return []
    entry = props.get(key, {})
    if not isinstance(entry, dict):
        return []
    enum = entry.get("enum")
    if not isinstance(enum, list):
        return []
    return [str(item) for item in enum if isinstance(item, str)]


def _select_from_allowed(preferred: str, allowed: list[str]) -> str:
    if preferred in allowed:
        return preferred
    if "gemini" in allowed:
        return "gemini"
    if allowed:
        return allowed[0]
    return "codex"


def _extract_milestone(prompt_text: str) -> str | None:
    for pattern in MILESTONE_PATTERNS:
        match = pattern.search(prompt_text)
        if match:
            return match.group(1)
    return None


def _extract_stats_refs(prompt_text: str) -> list[str]:
    matches = re.findall(r"\b[A-Z]{2,}-\d+\b", prompt_text)
    seen: set[str] = set()
    out: list[str] = []
    for item in matches:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_stats_ids(root: Path) -> set[str] | None:
    path = root / "STATS.md"
    if not path.exists():
        return None
    text = _read_text(path)
    ids = {m for m in re.findall(r"\b[A-Z]{2,}-\d+\b", text)}
    return ids if ids else None


def _env_truthy(name: str) -> bool:
    val = os.environ.get(name, "").strip().lower()
    return val in {"1", "true", "yes", "on"}


def _run_gemini(prompt_text: str) -> subprocess.CompletedProcess[str]:
    gemini_bin = os.environ.get("GEMINI_BIN", "gemini")
    cmd = [gemini_bin]
    model = os.environ.get("GEMINI_MODEL", "").strip()
    model_flag = os.environ.get("GEMINI_MODEL_FLAG", "--model")
    if model:
        cmd.extend([model_flag, model])

    extra_args = os.environ.get("GEMINI_ARGS", "").strip()
    if extra_args:
        cmd.extend(shlex.split(extra_args))

    prompt_flag = os.environ.get("GEMINI_PROMPT_FLAG", "").strip()
    stdin_text: str | None = prompt_text
    if prompt_flag:
        cmd.extend([prompt_flag, prompt_text])
        if not _env_truthy("GEMINI_PROMPT_VIA_STDIN"):
            stdin_text = None

    timeout_s = int(os.environ.get("GEMINI_TIMEOUT_S", "300"))
    return subprocess.run(
        cmd,
        input=stdin_text,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )


def _extract_json_payload(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    for candidate in _candidate_json_texts(text):
        obj = _try_parse_json(candidate)
        if obj is None:
            continue
        unwrapped = _unwrap_payload(obj)
        if isinstance(unwrapped, dict):
            return unwrapped
    return None


def _candidate_json_texts(text: str) -> Iterable[str]:
    stripped = text.strip()
    if stripped:
        yield stripped
    if stripped.startswith("```"):
        yield _strip_code_fence(stripped)
    for line in stripped.splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            yield line
    balanced = _extract_balanced_json(stripped)
    if balanced:
        yield balanced


def _strip_code_fence(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _try_parse_json(text: str) -> dict[str, Any] | None:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, str):
        try:
            obj2 = json.loads(obj)
        except json.JSONDecodeError:
            return None
        return obj2 if isinstance(obj2, dict) else None
    return None


def _extract_balanced_json(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for idx in range(start, len(text)):
        ch = text[idx]
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
                    return text[start : idx + 1]
    return None


def _unwrap_payload(obj: dict[str, Any]) -> dict[str, Any]:
    if "turn" in obj and isinstance(obj["turn"], dict):
        return obj["turn"]
    if "result" in obj:
        inner = obj["result"]
        if isinstance(inner, dict):
            return inner
        if isinstance(inner, str):
            parsed = _try_parse_json(inner)
            if isinstance(parsed, dict):
                return parsed
    if "response" in obj and isinstance(obj["response"], dict):
        return obj["response"]
    return obj


def _validate_turn(
    obj: dict[str, Any],
    schema: dict[str, Any] | None,
    allowed_agents: list[str],
    allowed_phases: list[str],
) -> tuple[bool, str]:
    if HAS_JSONSCHEMA and schema:
        try:
            jsonschema.validate(instance=obj, schema=schema)
            return True, ""
        except Exception as exc:
            return False, str(exc)
    return _validate_turn_minimal(obj, allowed_agents, allowed_phases)


def _validate_turn_minimal(
    obj: dict[str, Any],
    allowed_agents: list[str],
    allowed_phases: list[str],
) -> tuple[bool, str]:
    if not isinstance(obj, dict):
        return False, "turn is not an object"
    for key in REQUIRED_KEYS:
        if key not in obj:
            return False, f"missing key: {key}"
    if allowed_agents:
        if obj.get("agent") not in allowed_agents:
            return False, "invalid agent"
        if obj.get("next_agent") not in allowed_agents:
            return False, "invalid next_agent"
    if allowed_phases and obj.get("phase") not in allowed_phases:
        return False, "invalid phase"
    if not isinstance(obj.get("work_completed"), bool) or not isinstance(obj.get("project_complete"), bool):
        return False, "work_completed/project_complete must be boolean"
    for key in ("summary", "next_prompt", "delegate_rationale"):
        if not isinstance(obj.get(key), str):
            return False, f"{key} must be string"
    if not isinstance(obj.get("needs_write_access"), bool):
        return False, "needs_write_access must be boolean"
    if not isinstance(obj.get("gates_passed"), list) or not all(isinstance(item, str) for item in obj.get("gates_passed", [])):
        return False, "gates_passed must be array of strings"
    stats_refs = obj.get("stats_refs")
    if not isinstance(stats_refs, list) or not stats_refs or not all(isinstance(item, str) for item in stats_refs):
        return False, "stats_refs must be non-empty array of strings"
    rp = obj.get("requirement_progress")
    if not isinstance(rp, dict):
        return False, "requirement_progress must be object"
    for key in ("covered_req_ids", "tests_added_or_modified", "commands_run"):
        items = rp.get(key)
        if not isinstance(items, list) or not all(isinstance(x, str) for x in items):
            return False, f"requirement_progress.{key} must be array of strings"
    artifacts = obj.get("artifacts")
    if not isinstance(artifacts, list):
        return False, "artifacts must be array"
    for idx, item in enumerate(artifacts):
        if not isinstance(item, dict):
            return False, f"artifact[{idx}] must be object"
        if set(item.keys()) != {"path", "description"}:
            return False, f"artifact[{idx}] must have path/description"
        if not isinstance(item.get("path"), str) or not isinstance(item.get("description"), str):
            return False, f"artifact[{idx}] path/description must be strings"
    return True, ""


def _build_error_payload(
    *,
    agent_id: str,
    milestone_id: str,
    allowed_agents: list[str],
    allowed_phases: list[str],
    summary: str,
    error_detail: str,
    stats_refs: list[str],
    stats_id_set: set[str] | None,
    needs_write_access: bool,
) -> dict[str, Any]:
    payload = build_error_turn(
        agent=agent_id,
        milestone_id=milestone_id,
        summary=summary,
        error_detail=error_detail,
        stats_refs=stats_refs,
        stats_id_set=stats_id_set,
        needs_write_access=needs_write_access,
    )
    payload["agent"] = _coerce_enum(payload.get("agent"), allowed_agents, agent_id or "codex")
    payload["next_agent"] = _coerce_enum(payload.get("next_agent"), allowed_agents, payload["agent"])
    payload["phase"] = _coerce_enum(payload.get("phase"), allowed_phases, "plan")
    if not payload.get("milestone_id"):
        payload["milestone_id"] = milestone_id or "M0"
    if not payload.get("stats_refs"):
        payload["stats_refs"] = ["CX-1"]
    return payload


def _coerce_enum(value: Any, allowed: list[str], fallback: str) -> str:
    if isinstance(value, str) and value in allowed:
        return value
    if allowed:
        return allowed[0]
    return fallback


def _format_process_error(result: subprocess.CompletedProcess[str]) -> str:
    lines = [f"exit_code={result.returncode}"]
    stdout = _truncate(result.stdout or "")
    stderr = _truncate(result.stderr or "")
    if stderr:
        lines.append(f"stderr={stderr}")
    if stdout:
        lines.append(f"stdout={stdout}")
    return "\n".join(lines)


def _format_parse_error(raw_text: str) -> str:
    snippet = _truncate(raw_text)
    if snippet:
        return f"raw_output={snippet}"
    return "raw_output_empty"


def _format_validation_error(err: str, raw_text: str) -> str:
    snippet = _truncate(raw_text)
    if snippet:
        return f"validation_error={err}\nraw_output={snippet}"
    return f"validation_error={err}"


def _truncate(text: str, limit: int = 800) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...(truncated)"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
