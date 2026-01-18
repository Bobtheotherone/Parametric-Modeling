#!/usr/bin/env bash
set -euo pipefail

PROMPT_FILE="${1:?prompt_file}"
SCHEMA_FILE="${2:?schema_file}"
OUT_FILE="${3:?out_file}"

MODEL="${CLAUDE_MODEL:-claude-3-7-sonnet-latest}"
TIMEOUT_S="${CLAUDE_TIMEOUT_S:-180}"

prompt="$(cat "$PROMPT_FILE")"

ERR_FILE="${OUT_FILE}.stderr"
WRAP_JSON="${OUT_FILE}.wrapper.json"

# Run Claude in JSON wrapper mode so stdout is always machine-parseable.
# We keep headless-safe flags to avoid hanging on interactive permissions.
if command -v timeout >/dev/null 2>&1; then
  timeout "${TIMEOUT_S}s" claude \
    -p "$prompt" \
    --output-format json \
    --model "$MODEL" \
    --no-session-persistence \
    --permission-mode dontAsk \
    --tools "" \
    >"$WRAP_JSON" 2>"$ERR_FILE" || true
else
  claude \
    -p "$prompt" \
    --output-format json \
    --model "$MODEL" \
    --no-session-persistence \
    --permission-mode dontAsk \
    --tools "" \
    >"$WRAP_JSON" 2>"$ERR_FILE" || true
fi

# Extract model "result" and normalize to the strict orchestrator schema.
python3 - <<'PY' "$WRAP_JSON" "$SCHEMA_FILE" "$PROMPT_FILE" > "$OUT_FILE" || true
import json, re, sys

wrap_path, schema_path, prompt_path = sys.argv[1], sys.argv[2], sys.argv[3]
raw = open(wrap_path, "r", encoding="utf-8").read().strip()
schema = json.loads(open(schema_path, "r", encoding="utf-8").read())
prompt_text = open(prompt_path, "r", encoding="utf-8").read()

# The orchestrator validator allows ONLY these keys (and requires all of them).
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

def extract_stats_ids(text: str) -> list[str]:
    return sorted(set(re.findall(r"\b(?:CX|GM|CL)-\d+\b", text)))

def choose_stats_ref(ids: list[str]) -> list[str]:
    if not ids:
        return ["CL-1"]  # last-resort; prompt usually contains CL-*
    cl = [x for x in ids if x.startswith("CL-")]
    return [cl[0] if cl else ids[0]]

def extract_milestone_id(text: str) -> str:
    # Prefer the orchestrator state blob: "milestone_id": "M0"
    m = re.search(r'"milestone_id"\s*:\s*"([^"]+)"', text)
    if m:
        return m.group(1)
    # Fallback: DESIGN_DOCUMENT excerpt: **Milestone:** M0
    m2 = re.search(r"\*\*Milestone:\*\*\s*(M\d+)\b", text)
    return m2.group(1) if m2 else "M0"

def to_bool(x, default=False):
    return x if isinstance(x, bool) else default

def to_str(x, default=""):
    return x if isinstance(x, str) else default

def to_str_list(x):
    if not isinstance(x, list):
        return []
    out = []
    for i in x:
        if isinstance(i, str) and i.strip():
            out.append(i.strip())
    return out

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
if isinstance(wrapper, dict):
    for k in ("result", "response", "content", "output", "text", "message"):
        v = wrapper.get(k)
        if isinstance(v, str) and v.strip():
            model_text = v.strip()
            break
        if isinstance(v, (dict, list)):
            model_text = json.dumps(v)
            break

model_text = strip_fences(model_text)

# Try to parse the model text as JSON object
turn = jload(model_text)
if turn is None:
    l, r = model_text.find("{"), model_text.rfind("}")
    if l != -1 and r != -1 and r > l:
        turn = jload(model_text[l : r + 1].strip())

# If still not a dict, synthesize a minimal valid turn so the orchestrator can proceed.
stats_ids = extract_stats_ids(prompt_text)
milestone_id = extract_milestone_id(prompt_text)

if not isinstance(turn, dict):
    out = {
        "agent": "claude",
        "milestone_id": milestone_id,
        "phase": "plan",
        "work_completed": False,
        "project_complete": False,
        "summary": "Claude returned non-JSON; wrapper coerced to a valid turn. See delegate_rationale.",
        "gates_passed": [],
        "requirement_progress": {"covered_req_ids": [], "tests_added_or_modified": [], "commands_run": []},
        "next_agent": "codex",
        "next_prompt": "Claude output was not valid JSON. Continue the task using tools.verify output; fix remaining gates.",
        "delegate_rationale": (model_text[:4000] if model_text else "(empty model output)"),
        "stats_refs": choose_stats_ref(stats_ids),
        "needs_write_access": True,
        "artifacts": [],
    }
    print(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    raise SystemExit(0)

# Normalize a real parsed dict to strict schema
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

out["needs_write_access"] = to_bool(turn.get("needs_write_access"), True)
out["gates_passed"] = to_str_list(turn.get("gates_passed", []))

# stats_refs must be non-empty and must reference known IDs
refs = to_str_list(turn.get("stats_refs", []))
refs = [r for r in refs if r in stats_ids]
out["stats_refs"] = refs if refs else choose_stats_ref(stats_ids)

out["requirement_progress"] = normalize_requirement_progress(turn.get("requirement_progress"))
out["artifacts"] = normalize_artifacts(turn.get("artifacts"))

# Output only required keys, no extras.
out = {k: out[k] for k in REQUIRED_KEYS}
print(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
PY

# Guarantee non-empty output so the orchestrator never sees "empty output".
if [[ ! -s "$OUT_FILE" ]]; then
  echo "{}" > "$OUT_FILE"
fi

cat "$OUT_FILE"
