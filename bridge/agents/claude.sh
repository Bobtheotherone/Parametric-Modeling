#!/usr/bin/env bash
set -euo pipefail

PROMPT_FILE="${1:?prompt_file}"
SCHEMA_FILE="${2:?schema_file}"
OUT_FILE="${3:?out_file}"

MODEL="${CLAUDE_MODEL:-claude-3-7-sonnet-latest}"
TIMEOUT_S="${CLAUDE_TIMEOUT_S:-180}"

prompt="$(cat "$PROMPT_FILE")"

# Final hard reminder at the end helps on long prompts.
prompt="${prompt}

REMINDER (NON-NEGOTIABLE): Output EXACTLY ONE JSON object that matches the provided JSON schema.
No markdown. No code fences. No extra text before/after the JSON."

# Claude Code expects --json-schema as an *inline JSON string*, not a file path.
SCHEMA_JSON="$(
  python3 -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1])), separators=(",",":")))' \
    "$SCHEMA_FILE"
)"

ERR_FILE="${OUT_FILE}.stderr"
WRAP_JSON="${OUT_FILE}.wrapper.json"

# Run Claude in JSON wrapper mode; apply schema validation at the CLI level.
if command -v timeout >/dev/null 2>&1; then
  timeout "${TIMEOUT_S}s" claude \
    -p "$prompt" \
    --output-format json \
    --json-schema "$SCHEMA_JSON" \
    --model "$MODEL" \
    --no-session-persistence \
    --permission-mode dontAsk \
    --tools "" \
    >"$WRAP_JSON" 2>"$ERR_FILE" || true
else
  claude \
    -p "$prompt" \
    --output-format json \
    --json-schema "$SCHEMA_JSON" \
    --model "$MODEL" \
    --no-session-persistence \
    --permission-mode dontAsk \
    --tools "" \
    >"$WRAP_JSON" 2>"$ERR_FILE" || true
fi

# Extract Claude wrapper JSON -> model text -> parse/normalize into strict turn schema.
python3 - <<'PY' "$WRAP_JSON" "$SCHEMA_FILE" "$PROMPT_FILE" > "$OUT_FILE" || true
import json, re, sys

wrap_path, schema_path, prompt_path = sys.argv[1], sys.argv[2], sys.argv[3]
raw = open(wrap_path, "r", encoding="utf-8").read().strip()
schema = json.loads(open(schema_path, "r", encoding="utf-8").read())
prompt_text = open(prompt_path, "r", encoding="utf-8").read()

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

def salvage_json_obj(s: str):
    i = s.find("{")
    if i == -1:
        return None
    return jload(s[i:])

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
    return sorted(set(re.findall(r"\b(?:CX|GM|CL)-\d+\b", text)))

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

def to_bool(x, default=False):
    return x if isinstance(x, bool) else default

def to_str(x, default=""):
    return x if isinstance(x, str) else default

def to_str_list(x):
    if not isinstance(x, list):
        return []
    return [i.strip() for i in x if isinstance(i, str) and i.strip()]

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

def normalize_requirement_progress(x):
    rp = x if isinstance(x, dict) else {}
    return {
        "covered_req_ids": to_str_list(rp.get("covered_req_ids", [])),
        "tests_added_or_modified": to_str_list(rp.get("tests_added_or_modified", [])),
        "commands_run": to_str_list(rp.get("commands_run", [])),
    }

# Parse Claude wrapper JSON
wrapper = jload(raw) or salvage_json_obj(raw)
model_text = ""
is_error = False

if isinstance(wrapper, dict):
    is_error = bool(wrapper.get("is_error", False))
    for k in ("result", "response", "content", "output", "text", "message"):
        v = wrapper.get(k)
        if isinstance(v, str) and v.strip():
            model_text = v.strip()
            break
        if isinstance(v, (dict, list)):
            model_text = json.dumps(v)
            break

model_text = strip_fences(model_text)

# Parse the model output into a dict turn.
turn = jload(model_text)
if turn is None:
    obj_txt = extract_balanced_object(model_text)
    if obj_txt:
        turn = jload(obj_txt)

stats_ids = extract_stats_ids(prompt_text)
milestone_id = extract_milestone_id(prompt_text)

# If Claude failed or didn't produce JSON, synthesize a valid turn.
if not isinstance(turn, dict):
    out = {
        "agent": "claude",
        "milestone_id": milestone_id,
        "phase": "plan",
        "work_completed": False,
        "project_complete": False,
        "summary": "Claude output was not valid JSON under schema enforcement; wrapper synthesized a valid turn.",
        "gates_passed": [],
        "requirement_progress": {"covered_req_ids": [], "tests_added_or_modified": [], "commands_run": []},
        "next_agent": "codex",
        "next_prompt": "Claude did not emit schema-valid JSON. Continue using tools.verify output; fix remaining gates, then re-run.",
        "delegate_rationale": (
            f"(claude is_error={is_error})\n"
            + (model_text[:4000] if model_text else "(empty model output)")
        ),
        "stats_refs": choose_stats_ref(stats_ids),
        # Keep write access flowing to codex by default:
        "needs_write_access": True,
        "artifacts": [],
    }
    print(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    raise SystemExit(0)

# Normalize parsed turn into the strict schema expected by bridge/loop.py
out = {}
out["agent"] = "claude"
out["milestone_id"] = milestone_id

phase = turn.get("phase")
out["phase"] = phase if phase in ("plan", "implement", "verify", "finalize") else "plan"

out["work_completed"] = to_bool(turn.get("work_completed"), False)
out["project_complete"] = to_bool(turn.get("project_complete"), False)

out["summary"] = to_str(turn.get("summary"), "")
out["next_prompt"] = to_str(turn.get("next_prompt"), "")
out["delegate_rationale"] = to_str(turn.get("delegate_rationale"), "")

na = turn.get("next_agent")
out["next_agent"] = na if na in ("codex", "gemini", "claude") else "codex"

out["gates_passed"] = to_str_list(turn.get("gates_passed", []))

refs = to_str_list(turn.get("stats_refs", []))
refs = [r for r in refs if r in stats_ids]
out["stats_refs"] = refs if refs else choose_stats_ref(stats_ids)

out["requirement_progress"] = normalize_requirement_progress(turn.get("requirement_progress"))
out["artifacts"] = normalize_artifacts(turn.get("artifacts"))

# **Key policy change**: If we delegate to a coding agent, keep write access enabled.
# This prevents the orchestrator from dropping WRITE_ACCESS on the next call.
if out["next_agent"] in ("codex", "claude"):
    out["needs_write_access"] = True
else:
    out["needs_write_access"] = to_bool(turn.get("needs_write_access"), True)

out = {k: out[k] for k in REQUIRED_KEYS}
print(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
PY

# Guarantee non-empty output.
if [[ ! -s "$OUT_FILE" ]]; then
  echo "{}" > "$OUT_FILE"
fi

cat "$OUT_FILE"
