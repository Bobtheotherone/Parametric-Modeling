#!/usr/bin/env bash
set -euo pipefail

PROMPT_FILE="${1:?prompt_file}"
SCHEMA_FILE="${2:?schema_file}"
OUT_FILE="${3:?out_file}"

MODEL="${CLAUDE_MODEL:-claude-opus-4-5}"
FF_SMOKE="${FF_SMOKE:-0}"
DEFAULT_TIMEOUT_S=86400
SMOKE_TIMEOUT_S=180
if [[ "$FF_SMOKE" == "1" ]]; then
  TIMEOUT_S="${CLAUDE_TIMEOUT_S:-$SMOKE_TIMEOUT_S}"
else
  TIMEOUT_S="${CLAUDE_TIMEOUT_S:-$DEFAULT_TIMEOUT_S}"
fi
CLAUDE_BIN="${CLAUDE_BIN:-claude}"
CLAUDE_ARGS_JSON_MODE="${CLAUDE_ARGS_JSON_MODE:-}"
CLAUDE_TOOLS="${CLAUDE_TOOLS:-}"
CLAUDE_HELP_TIMEOUT_S="${CLAUDE_HELP_TIMEOUT_S:-5}"
SMOKE_DIR="${FF_AGENT_SMOKE_DIR:-}"
WRITE_ACCESS="${WRITE_ACCESS:-0}"

# Schema kind signal from orchestrator: "task_plan" or "turn" (default: "turn")
# When task_plan, we skip turn normalization and just extract valid JSON.
ORCH_SCHEMA_KIND="${ORCH_SCHEMA_KIND:-turn}"

prompt="$(cat "$PROMPT_FILE")"

if [[ -n "$SMOKE_DIR" && "$WRITE_ACCESS" == "1" ]]; then
  mkdir -p "$SMOKE_DIR"
  printf '%s %s\n' "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" "claude" > "$SMOKE_DIR/claude.txt"
fi

# Build schema-aware prompt reminder
# The reminder tells the model which schema to use based on what was requested
SCHEMA_BASENAME="$(basename "$SCHEMA_FILE")"
if [[ "$ORCH_SCHEMA_KIND" == "task_plan" ]]; then
  prompt="${prompt}

REMINDER (NON-NEGOTIABLE):
- Tools are DISABLED. Do NOT output <task>, <read>, <edit>, <bash> blocks.
- Output EXACTLY ONE JSON object matching ${SCHEMA_BASENAME}.
- Required keys: milestone_id, max_parallel_tasks, rationale, tasks (array).
- No markdown. No code fences. No extra text before/after the JSON."
else
  prompt="${prompt}

REMINDER (NON-NEGOTIABLE):
- Tools are DISABLED. Do NOT output <task>, <read>, <edit>, <bash> blocks.
- Output EXACTLY ONE JSON object matching ${SCHEMA_BASENAME}.
- No markdown. No code fences. No extra text before/after the JSON."
fi

# Claude Code expects --json-schema as an *inline JSON string* (not a file path).
SCHEMA_JSON="$(
  python3 -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1])), separators=(",",":")))' \
    "$SCHEMA_FILE"
)"

ERR_SCHEMA="${OUT_FILE}.stderr.schema"
ERR_PLAIN="${OUT_FILE}.stderr.plain"
WRAP_SCHEMA="${OUT_FILE}.wrapper_schema.json"
WRAP_PLAIN="${OUT_FILE}.wrapper_plain.json"
RAW_STREAM="${OUT_FILE}.wrapper_schema_claude_raw_stream.txt"

HELP_TEXT="$(python3 - <<'PY' "$CLAUDE_BIN" "$CLAUDE_HELP_TIMEOUT_S"
import os
import signal
import subprocess
import sys

bin_path = sys.argv[1]
timeout_s = int(sys.argv[2])
out = ""
try:
    proc = subprocess.Popen(
        [bin_path, "--help"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    stdout, stderr = proc.communicate(timeout=timeout_s)
    out = stdout or stderr or ""
except subprocess.TimeoutExpired:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except Exception:
        pass
    out = ""
except Exception:
    out = ""
print(out)
PY
)"

supports_flag() {
  local flag="$1"
  [[ "$HELP_TEXT" == *"$flag"* ]]
}

PROMPT_FLAG="-p"
if supports_flag "--prompt"; then
  PROMPT_FLAG="--prompt"
fi

COMMON_ARGS=("$PROMPT_FLAG" "$prompt")
if [[ -n "$MODEL" ]] && supports_flag "--model"; then
  COMMON_ARGS+=(--model "$MODEL")
fi
if supports_flag "--no-session-persistence"; then
  COMMON_ARGS+=(--no-session-persistence)
fi
if supports_flag "--permission-mode"; then
  COMMON_ARGS+=(--permission-mode dontAsk)
fi
# Prefer newer/stronger flags that skip permission prompts entirely (when supported).
if supports_flag "--dangerously-skip-permissions"; then
  COMMON_ARGS+=(--dangerously-skip-permissions)
elif supports_flag "--skip-permissions"; then
  COMMON_ARGS+=(--skip-permissions)
elif supports_flag "--skip-permission-prompts"; then
  COMMON_ARGS+=(--skip-permission-prompts)
fi
if [[ -n "$CLAUDE_TOOLS" ]] && supports_flag "--tools"; then
  COMMON_ARGS+=(--tools "$CLAUDE_TOOLS")
fi

JSON_ARGS=()
if [[ -n "$CLAUDE_ARGS_JSON_MODE" ]]; then
  read -r -a JSON_ARGS <<< "$CLAUDE_ARGS_JSON_MODE"
else
  if supports_flag "--output-format"; then
    JSON_ARGS+=(--output-format json)
  fi
  if supports_flag "--json-schema"; then
    JSON_ARGS+=(--json-schema "$SCHEMA_JSON")
  elif supports_flag "--json"; then
    JSON_ARGS+=(--json)
  fi
fi

PLAIN_ARGS=()
if supports_flag "--output-format"; then
  PLAIN_ARGS+=(--output-format json)
elif supports_flag "--json"; then
  PLAIN_ARGS+=(--json)
fi

run_claude() {
  # args: <mode:json|plain> <wrap_path> <err_path>
  local mode="$1"
  local wrap_path="$2"
  local err_path="$3"
  local -a mode_args=()

  if [[ "$mode" == "json" ]]; then
    mode_args=("${JSON_ARGS[@]}")
  else
    mode_args=("${PLAIN_ARGS[@]}")
  fi

  local -a cmd=("$CLAUDE_BIN" "${COMMON_ARGS[@]}" "${mode_args[@]}")
  local rc=0

  python3 - <<'PY' "$TIMEOUT_S" "$wrap_path" "$err_path" "${wrap_path}.meta.json" "${cmd[@]}"
import json
import os
import signal
import subprocess
import sys
import threading

timeout_s = int(sys.argv[1])
wrap_path = sys.argv[2]
err_path = sys.argv[3]
meta_path = sys.argv[4]
cmd = sys.argv[5:]

out = ""
err = ""
rc = 0

stdout_lines = []
stderr_lines = []

proc = subprocess.Popen(
    cmd,
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    start_new_session=True,
    bufsize=1,
)

def reader(stream, sink, dest):
    for line in iter(stream.readline, ""):
        sink.append(line)
        dest.write(line)
        dest.flush()
    stream.close()

t_out = threading.Thread(target=reader, args=(proc.stdout, stdout_lines, sys.stdout), daemon=True)
t_err = threading.Thread(target=reader, args=(proc.stderr, stderr_lines, sys.stderr), daemon=True)
t_out.start()
t_err.start()

try:
    proc.wait(timeout=timeout_s)
    rc = proc.returncode
except subprocess.TimeoutExpired:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except Exception:
        pass
    rc = 124
    stderr_lines.append(f"\\nTIMEOUT after {timeout_s}s\\n")

t_out.join()
t_err.join()
out = "".join(stdout_lines)
err = "".join(stderr_lines)

with open(wrap_path, "w", encoding="utf-8") as handle:
    handle.write(out)

with open(err_path, "a", encoding="utf-8") as handle:
    handle.write(err)

with open(meta_path, "w", encoding="utf-8") as handle:
    json.dump({"cmd": cmd, "rc": rc}, handle)
PY
}

: > "$ERR_SCHEMA"
: > "$ERR_PLAIN"
: > "$WRAP_SCHEMA"
: > "$WRAP_PLAIN"

# Try schema-enforced first (best chance of correct structured output).
run_claude "json" "$WRAP_SCHEMA" "$ERR_SCHEMA"

# If that produced an error wrapper or no usable result, we'll fall back to plain.
# Normalization logic below will choose the better wrapper automatically.
run_claude "plain" "$WRAP_PLAIN" "$ERR_PLAIN"

# Save combined raw stream for debugging
cat "$WRAP_SCHEMA" "$WRAP_PLAIN" > "$RAW_STREAM" 2>/dev/null || true

# ============================================================================
# Schema-kind dispatch: task_plan vs turn
# ============================================================================

if [[ "$ORCH_SCHEMA_KIND" == "task_plan" ]]; then
  # TASK_PLAN MODE: No turn normalization, just extract valid JSON matching schema
  # Use the extract_json_by_schema.py helper for robust extraction

  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  EXTRACT_SCRIPT="$SCRIPT_DIR/../../scripts/extract_json_by_schema.py"

  if [[ -f "$EXTRACT_SCRIPT" ]]; then
    if python3 "$EXTRACT_SCRIPT" "$RAW_STREAM" "$SCHEMA_FILE" "$OUT_FILE"; then
      cat "$OUT_FILE"
      exit 0
    else
      echo "ERROR: extract_json_by_schema.py failed to find valid task_plan JSON" >&2
      echo "  raw_stream: $RAW_STREAM" >&2
      echo "  schema: $SCHEMA_FILE" >&2
      exit 1
    fi
  else
    # Fallback: inline extraction for task_plan schema
    python3 - <<'PY' "$WRAP_SCHEMA" "$WRAP_PLAIN" "$SCHEMA_FILE" "$OUT_FILE"
import json
import sys
from pathlib import Path

wrap_schema = sys.argv[1]
wrap_plain = sys.argv[2]
schema_path = sys.argv[3]
out_path = sys.argv[4]

try:
    import jsonschema
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False

def validate(obj, schema):
    if not HAS_JSONSCHEMA:
        # Basic check: must have required keys
        required = schema.get("required", [])
        return isinstance(obj, dict) and all(k in obj for k in required)
    try:
        jsonschema.validate(instance=obj, schema=schema)
        return True
    except jsonschema.ValidationError:
        return False

def strip_fences(s):
    s = s.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return s

def extract_from_stream(text, schema):
    candidates = []
    decoder = json.JSONDecoder()
    idx = 0
    n = len(text)

    parsed = []
    while idx < n:
        while idx < n and text[idx] in " \t\n\r":
            idx += 1
        if idx >= n:
            break
        try:
            obj, end = decoder.raw_decode(text, idx)
            parsed.append(obj)
            idx = end
        except json.JSONDecodeError:
            idx += 1

    # Flatten arrays
    flattened = []
    for obj in parsed:
        if isinstance(obj, list):
            flattened.extend(obj)
        else:
            flattened.append(obj)

    for obj in flattened:
        if not isinstance(obj, dict):
            continue

        # Direct match
        if validate(obj, schema):
            candidates.append(obj)
            continue

        # Check for embedded result
        if obj.get("type") == "result":
            result_str = obj.get("result")
            if isinstance(result_str, str):
                try:
                    inner = json.loads(strip_fences(result_str))
                    if isinstance(inner, dict) and validate(inner, schema):
                        candidates.append(inner)
                except:
                    pass

        # Check assistant message content
        if obj.get("type") == "assistant":
            message = obj.get("message", {})
            content = message.get("content", [])
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_content = block.get("text", "")
                    try:
                        inner = json.loads(strip_fences(text_content))
                        if isinstance(inner, dict) and validate(inner, schema):
                            candidates.append(inner)
                    except:
                        pass

    return candidates[-1] if candidates else None

# Load schema
schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))

# Try both wrappers
result = None
for wrapper_path in [wrap_schema, wrap_plain]:
    try:
        text = Path(wrapper_path).read_text(encoding="utf-8")
        result = extract_from_stream(text, schema)
        if result:
            break
    except Exception:
        pass

if result is None:
    sys.stderr.write("ERROR: Could not extract valid task_plan JSON from Claude output\n")
    sys.exit(1)

Path(out_path).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
PY

    rc=$?
    if [[ $rc -ne 0 ]]; then
      echo "ERROR: Task plan extraction failed" >&2
      exit 1
    fi
    exit 0
  fi
fi

# ============================================================================
# TURN MODE: Full turn normalization (original behavior)
# ============================================================================

python3 - <<'PY' "$WRAP_SCHEMA" "$WRAP_PLAIN" "$SCHEMA_FILE" "$PROMPT_FILE" "$ERR_SCHEMA" "$ERR_PLAIN" > "$OUT_FILE" || true
import json, os, re, sys

wrap_schema, wrap_plain, schema_path, prompt_path, err_schema, err_plain = (
    sys.argv[1],
    sys.argv[2],
    sys.argv[3],
    sys.argv[4],
    sys.argv[5],
    sys.argv[6],
)
schema = json.loads(open(schema_path, "r", encoding="utf-8").read())
prompt_text = open(prompt_path, "r", encoding="utf-8").read()
api_vars = [name for name in ("ANTHROPIC_API_KEY", "CLAUDE_API_KEY") if os.environ.get(name)]
auth_mode = "api_key" if api_vars else "subscription"
auth_warning = ""
if api_vars:
    auth_warning = (
        "WARNING: "
        + ", ".join(api_vars)
        + " set; Claude Code will use API billing instead of Pro/Max subscription."
    )

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

def jload(s: str):
    try:
        return json.loads(s)
    except Exception:
        return None

def read_json(path: str):
    try:
        raw = open(path, "r", encoding="utf-8").read().strip()
    except Exception:
        return None
    return jload(raw) if raw else None


def parse_json_sequence(path: str) -> list:
    """Parse a file containing one or more JSON values (sequence/stream).

    Handles:
    - Multiple concatenated JSON values (with or without whitespace)
    - Pretty-printed multi-line JSON arrays/objects
    - JSON Lines (one per line)
    - Single minified JSON array

    Uses json.JSONDecoder().raw_decode() to robustly parse any valid JSON sequence.
    """
    try:
        raw = open(path, "r", encoding="utf-8").read()
    except Exception:
        return []

    objects = []
    decoder = json.JSONDecoder()
    idx = 0
    n = len(raw)

    while idx < n:
        # Skip whitespace
        while idx < n and raw[idx] in " \t\n\r":
            idx += 1
        if idx >= n:
            break

        try:
            obj, end = decoder.raw_decode(raw, idx)
            # Flatten top-level arrays (Claude outputs arrays of events)
            if isinstance(obj, list):
                objects.extend(obj)
            else:
                objects.append(obj)
            idx = end
        except json.JSONDecodeError:
            # Skip one character and try again (handles trailing garbage)
            idx += 1

    return objects


def extract_turn_from_claude_jsonlines(path: str):
    """
    Extract the turn JSON from Claude CLI JSON stream output.

    Claude CLI outputs JSON events that can be:
    - Multiple concatenated JSON values
    - Pretty-printed multi-line JSON arrays/objects
    - JSON Lines (one per line)

    Priority order for extraction:
    1. Last event with type=="result" AND has a string field "result" → parse that string
    2. Else last event with type=="assistant" and message.content[*].type=="text" → concatenate text
    3. Else fail
    """
    objects = parse_json_sequence(path)

    if not objects:
        return None

    # Priority 1: Look for type=="result" events with a "result" string field
    result_events = [
        obj for obj in objects
        if isinstance(obj, dict) and obj.get("type") == "result"
    ]

    for evt in reversed(result_events):
        result_str = evt.get("result")
        if isinstance(result_str, str) and result_str.strip():
            turn = parse_turn_from_text(result_str)
            if turn is not None:
                return turn

    # Priority 2: Look for type=="assistant" events with message.content[*].text
    assistant_messages = [
        obj for obj in objects
        if isinstance(obj, dict) and obj.get("type") == "assistant" and "message" in obj
    ]

    if not assistant_messages:
        return None

    last_msg = assistant_messages[-1]
    message = last_msg.get("message", {})
    content = message.get("content", [])

    # Extract text from content blocks
    text_content = ""
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            text_content += block.get("text", "")

    if not text_content:
        return None

    # Try to parse the text as a turn JSON
    return parse_turn_from_text(text_content)

def read_text(path: str) -> str:
    try:
        return open(path, "r", encoding="utf-8").read()
    except Exception:
        return ""

def read_meta(path: str):
    meta_path = path + ".meta.json"
    return read_json(meta_path) or {}

def strip_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return s

def extract_balanced_object(s: str) -> str | None:
    start = s.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
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
                    return s[start : i + 1]
    return None

def extract_stats_ids(text: str) -> list[str]:
    return sorted(set(re.findall(r"\b(?:CX|CL)-\d+\b", text)))

def choose_stats_ref(ids: list[str]) -> list[str]:
    if not ids:
        return ["CL-1"]
    cl = [x for x in ids if x.startswith("CL-")]
    return [cl[0] if cl else ids[0]]

def extract_milestone_id(text: str) -> str:
    m = re.search(r'"milestone_id"\s*:\s*"([^"]+)"', text)
    if m:
        return m.group(1)
    m2 = re.search(r"\*\*Milestone:\*\*\s*(M\d+)\b", text)
    return m2.group(1) if m2 else "M0"

def redact_args(cmd: list[str]) -> list[str]:
    redacted = []
    patterns = [
        re.compile(r"sk-[A-Za-z0-9]{20,}"),
        re.compile(r"sk-ant-[A-Za-z0-9\\-]{10,}"),
        re.compile(r"AIza[0-9A-Za-z_\\-]{20,}"),
    ]
    for arg in cmd:
        clean = arg
        for pat in patterns:
            if pat.search(clean):
                clean = pat.sub("***", clean)
        redacted.append(clean)
    return redacted

def diagnostic_block(label: str, meta: dict, stderr_text: str) -> str:
    cmd = meta.get("cmd", [])
    if isinstance(cmd, list):
        cmd_text = " ".join(redact_args([str(c) for c in cmd]))
    else:
        cmd_text = "(unknown)"
    rc = meta.get("rc", "unknown")
    stderr_snip = (stderr_text or "")[:4096]
    return f"{label}: cmd={cmd_text}\\nrc={rc}\\nstderr={stderr_snip}"

def to_bool(x, default=False):
    return x if isinstance(x, bool) else default

def to_str(x, default=""):
    return x if isinstance(x, str) else default

def to_str_list(x):
    if not isinstance(x, list):
        return []
    return [i.strip() for i in x if isinstance(i, str) and i.strip()]

def normalize_requirement_progress(x):
    rp = x if isinstance(x, dict) else {}
    return {
        "covered_req_ids": to_str_list(rp.get("covered_req_ids", [])),
        "tests_added_or_modified": to_str_list(rp.get("tests_added_or_modified", [])),
        "commands_run": to_str_list(rp.get("commands_run", [])),
    }

def normalize_artifacts(x):
    if not isinstance(x, list):
        return []
    out = []
    for a in x:
        if not isinstance(a, dict):
            continue
        p = a.get("path")
        d = a.get("description")
        if isinstance(p, str) and p.strip() and isinstance(d, str) and d.strip():
            out.append({"path": p.strip(), "description": d.strip()})
    return out

def append_wrapper_meta(summary: str, status: str) -> str:
    parts = []
    base = summary.strip()
    if base:
        parts.append(base)
    if auth_warning:
        parts.append(auth_warning)
    parts.append(f"wrapper_status={status} auth_mode={auth_mode}")
    return "\\n".join(parts)

def wrapper_to_model_text(wrapper: dict) -> tuple[bool, str]:
    # Claude Code wrapper format commonly: {"type":"result", "is_error":..., "result":"..."}
    is_error = bool(wrapper.get("is_error", False)) if isinstance(wrapper, dict) else True
    text = ""
    if isinstance(wrapper, dict):
        for k in ("result", "response", "content", "output", "text", "message"):
            v = wrapper.get(k)
            if isinstance(v, str) and v.strip():
                text = v.strip()
                break
    return is_error, text

def parse_turn_from_text(model_text: str):
    model_text = strip_fences(model_text)
    turn = jload(model_text)
    if isinstance(turn, dict):
        return turn
    obj_txt = extract_balanced_object(model_text)
    if obj_txt:
        t2 = jload(obj_txt)
        if isinstance(t2, dict):
            return t2
    return None

# Try to extract turn from Claude CLI JSON Lines output.
# Claude CLI outputs JSON Lines with init events and assistant messages.
# The turn JSON is in message.content[0].text of the last assistant message.

best_turn = None
best_model_text = ""

# Try schema wrapper first (more likely to have structured output)
for wrapper_path in [wrap_schema, wrap_plain]:
    turn = extract_turn_from_claude_jsonlines(wrapper_path)
    if turn is not None:
        best_turn = turn
        # Read raw content for diagnostics
        try:
            best_model_text = open(wrapper_path, "r", encoding="utf-8").read()[:8000]
        except Exception:
            best_model_text = ""
        break

# Fallback: try legacy wrapper format (single JSON object with result/response keys)
if best_turn is None:
    for wrapper_path in [wrap_schema, wrap_plain]:
        ws = read_json(wrapper_path)
        if isinstance(ws, dict):
            is_error, mt = wrapper_to_model_text(ws)
            if mt:
                t = parse_turn_from_text(mt)
                if t is not None:
                    best_turn = t
                    best_model_text = mt
                    break

stats_ids = extract_stats_ids(prompt_text)
milestone_id = extract_milestone_id(prompt_text)

def synthesize(reason: str, model_text: str) -> dict:
    meta_schema = read_meta(wrap_schema)
    meta_plain = read_meta(wrap_plain)
    diag = "\\n\\n".join(
        [
            diagnostic_block("schema_attempt", meta_schema, read_text(err_schema)),
            diagnostic_block("plain_attempt", meta_plain, read_text(err_plain)),
        ]
    )
    summary = append_wrapper_meta(f"{reason} Diagnostics:\\n{diag}", "error")
    return {
        "agent": "claude",
        "milestone_id": milestone_id,
        "phase": "plan",
        "work_completed": False,
        "project_complete": False,
        "summary": summary,
        "gates_passed": [],
        "requirement_progress": {"covered_req_ids": [], "tests_added_or_modified": [], "commands_run": []},
        "next_agent": "codex",
        "next_prompt": "Use tools.verify output; fix remaining gates (ruff/git_guard/etc.), then commit + re-run verify.",
        "delegate_rationale": (strip_fences(model_text)[:4000] if model_text else "(empty model output)"),
        "stats_refs": choose_stats_ref(stats_ids),
        "needs_write_access": True,
        "artifacts": [],
    }

if best_turn is None:
    # Save raw output to debug file for diagnostics
    import pathlib
    out_path = pathlib.Path(sys.argv[1])  # wrap_schema path
    debug_path = out_path.parent / f"{out_path.stem}_claude_raw_stream.txt"
    try:
        raw_content = ""
        for p in [wrap_schema, wrap_plain]:
            try:
                raw_content += f"\\n=== {p} ===\\n"
                raw_content += open(p, "r", encoding="utf-8").read()
            except Exception:
                raw_content += "(could not read)\\n"
        debug_path.write_text(raw_content, encoding="utf-8")
    except Exception:
        pass

    # Write error to stderr
    reason = "Claude did not emit a parseable JSON turn."
    diag = f"Raw output saved to: {debug_path}\\n"
    diag += f"wrap_schema: {wrap_schema}\\n"
    diag += f"wrap_plain: {wrap_plain}\\n"
    sys.stderr.write(f"ERROR: {reason}\\n{diag}\\n")

    # Exit with non-zero to signal failure to orchestrator
    raise SystemExit(1)

# Normalize parsed turn to strict schema
t = best_turn
out = {}
out["agent"] = "claude"
out["milestone_id"] = milestone_id

phase = t.get("phase")
out["phase"] = phase if phase in ("plan", "implement", "verify", "finalize") else "plan"

out["work_completed"] = to_bool(t.get("work_completed"), False)
out["project_complete"] = to_bool(t.get("project_complete"), False)

out["summary"] = append_wrapper_meta(to_str(t.get("summary"), ""), "ok")
out["gates_passed"] = to_str_list(t.get("gates_passed", []))
out["requirement_progress"] = normalize_requirement_progress(t.get("requirement_progress"))

na = t.get("next_agent")
out["next_agent"] = na if na in ("codex", "claude") else "codex"
out["next_prompt"] = to_str(t.get("next_prompt"), "")
out["delegate_rationale"] = to_str(t.get("delegate_rationale"), "")

refs = to_str_list(t.get("stats_refs", []))
refs = [r for r in refs if r in stats_ids]
out["stats_refs"] = refs if refs else choose_stats_ref(stats_ids)

out["artifacts"] = normalize_artifacts(t.get("artifacts"))

# IMPORTANT: keep write access on if delegating to a coding agent.
if out["next_agent"] in ("codex", "claude"):
    out["needs_write_access"] = True
else:
    out["needs_write_access"] = to_bool(t.get("needs_write_access"), True)

# Emit only required keys
out = {k: out.get(k) for k in REQUIRED_KEYS}

# Validate against schema
try:
    import jsonschema
    jsonschema.validate(instance=out, schema=schema)
except ImportError:
    # jsonschema not available; skip validation (tests will catch schema issues)
    pass
except jsonschema.ValidationError as e:
    import pathlib
    out_path = pathlib.Path(sys.argv[1])
    debug_path = out_path.parent / f"{out_path.stem}_claude_raw_stream.txt"
    try:
        raw_content = ""
        for p in [wrap_schema, wrap_plain]:
            try:
                raw_content += f"\\n=== {p} ===\\n"
                raw_content += open(p, "r", encoding="utf-8").read()
            except Exception:
                raw_content += "(could not read)\\n"
        debug_path.write_text(raw_content, encoding="utf-8")
    except Exception:
        pass

    sys.stderr.write(f"ERROR: Normalized turn failed schema validation: {e.message}\\n")
    sys.stderr.write(f"Raw output saved to: {debug_path}\\n")
    raise SystemExit(1)

print(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
PY

# If normalization failed for some unexpected reason, exit with error instead of synthesizing.
if [[ ! -s "$OUT_FILE" ]]; then
  echo "ERROR: Claude wrapper produced empty output. Raw files:" >&2
  echo "  schema: $WRAP_SCHEMA" >&2
  echo "  plain: $WRAP_PLAIN" >&2
  echo "  stderr_schema: $ERR_SCHEMA" >&2
  echo "  stderr_plain: $ERR_PLAIN" >&2
  exit 1
fi

cat "$OUT_FILE"
exit 0
