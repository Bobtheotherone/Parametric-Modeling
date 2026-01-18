#!/usr/bin/env bash
set -euo pipefail

PROMPT_FILE="${1:?prompt_file}"
SCHEMA_FILE="${2:?schema_file}"
OUT_FILE="${3:?out_file}"

MODEL="${CLAUDE_MODEL:-claude-3-7-sonnet-latest}"
TIMEOUT_S="${CLAUDE_TIMEOUT_S:-180}"

prompt="$(cat "$PROMPT_FILE")"

# Strong tail reminder helps on long prompts.
prompt="${prompt}

REMINDER (NON-NEGOTIABLE): Output EXACTLY ONE JSON object matching the turn schema.
No markdown. No code fences. No extra text before/after the JSON."

# Claude Code expects --json-schema as an *inline JSON string*, not a file path.
SCHEMA_JSON="$(
  python3 -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1])), separators=(",",":")))' \
    "$SCHEMA_FILE"
)"

ERR_FILE="${OUT_FILE}.stderr"
WRAP1_JSON="${OUT_FILE}.wrapper1.json"
WRAP2_JSON="${OUT_FILE}.wrapper2.json"
: > "$ERR_FILE"

_run_claude() {
  # args: <prompt_text> <wrap_json_path> <use_schema:0|1>
  local ptxt="$1"
  local wrap="$2"
  local use_schema="$3"

  if command -v timeout >/dev/null 2>&1; then
    if [[ "$use_schema" == "1" ]]; then
      timeout "${TIMEOUT_S}s" claude \
        -p "$ptxt" \
        --output-format json \
        --json-schema "$SCHEMA_JSON" \
        --model "$MODEL" \
        --no-session-persistence \
        --permission-mode dontAsk \
        --tools "" \
        >"$wrap" 2>>"$ERR_FILE" || true
    else
      timeout "${TIMEOUT_S}s" claude \
        -p "$ptxt" \
        --output-format json \
        --model "$MODEL" \
        --no-session-persistence \
        --permission-mode dontAsk \
        --tools "" \
        >"$wrap" 2>>"$ERR_FILE" || true
    fi
  else
    if [[ "$use_schema" == "1" ]]; then
      claude \
        -p "$ptxt" \
        --output-format json \
        --json-schema "$SCHEMA_JSON" \
        --model "$MODEL" \
        --no-session-persistence \
        --permission-mode dontAsk \
        --tools "" \
        >"$wrap" 2>>"$ERR_FILE" || true
    else
      claude \
        -p "$ptxt" \
        --output-format json \
        --model "$MODEL" \
        --no-session-persistence \
        --permission-mode dontAsk \
        --tools "" \
        >"$wrap" 2>>"$ERR_FILE" || true
    fi
  fi
}

_normalize_or_fail() {
  # args: <wrap_json_path>
  local wrap="$1"

  python3 - "$wrap" "$SCHEMA_FILE" "$PROMPT_FILE" <<'PY'
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

wrapper = jload(raw) or salvage_json_obj(raw)
if not isinstance(wrapper, dict):
    raise SystemExit(2)

# Claude Code JSON wrapper commonly includes: {"type":"result", ... , "result":"<text>"}
is_error = bool(wrapper.get("is_error", False))
model_text = ""
for k in ("result", "response", "content", "output", "text", "message"):
    v = wrapper.get(k)
    if isinstance(v, str) and v.strip():
        model_text = v.strip()
        break
    if isinstance(v, (dict, list)):
        model_text = json.dumps(v)
        break

model_text = strip_fences(model_text)

if is_error or not model_text:
    raise SystemExit(3)

turn = jload(model_text)
if turn is None:
    obj_txt = extract_balanced_object(model_text)
    if obj_txt:
        turn = jload(obj_txt)

if not isinstance(turn, dict):
    raise SystemExit(4)

stats_ids = extract_stats_ids(prompt_text)
milestone_id = extract_milestone_id(prompt_text)

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

# Keep write access flowing to coding agents.
if out["next_agent"] in ("codex", "claude"):
    out["needs_write_access"] = True
else:
    out["needs_write_access"] = to_bool(turn.get("needs_write_access"), True)

out = {k: out.get(k) for k in REQUIRED_KEYS}
print(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
PY
}

# Attempt 1: NO schema enforcement (avoids schema-rejection causing empty/errored result)
_run_claude "$prompt" "$WRAP1_JSON" "0"
if _normalize_or_fail "$WRAP1_JSON" > "$OUT_FILE"; then
  : # ok
else
  # Attempt 2: minimal correction prompt + schema enforcement
  # Extract milestone + a known stats id from the original orchestrator prompt
  MID="$(python3 - <<'PY' < "$PROMPT_FILE"
import re,sys
t=sys.stdin.read()
m=re.search(r'"milestone_id"\s*:\s*"([^"]+)"', t)
print(m.group(1) if m else "M0")
PY
)"
  STAT="$(python3 - <<'PY' < "$PROMPT_FILE"
import re,sys
t=sys.stdin.read()
ids=sorted(set(re.findall(r"\b(?:CX|GM|CL)-\d+\b", t)))
cl=[x for x in ids if x.startswith("CL-")]
print((cl[0] if cl else (ids[0] if ids else "CL-1")))
PY
)"
  CORR_PROMPT="You MUST output exactly one JSON object matching the schema. No markdown.
Use:
agent='claude'
milestone_id='${MID}'
phase='plan'
work_completed=false
project_complete=false
summary='(correction) emit schema-valid JSON'
gates_passed=[]
requirement_progress={covered_req_ids:[],tests_added_or_modified:[],commands_run:[]}
next_agent='codex'
next_prompt='Continue with the next smallest implementable step per tools.verify.'
delegate_rationale='Correction pass.'
stats_refs=['${STAT}']
needs_write_access=true
artifacts=[]
Return ONLY the JSON object."

  _run_claude "$CORR_PROMPT" "$WRAP2_JSON" "1"
  if _normalize_or_fail "$WRAP2_JSON" > "$OUT_FILE"; then
    : # ok
  else
    # Last resort: synthesize a minimal valid turn.
    MID_F="$MID"
    STAT_F="$STAT"
    printf '%s' "{\"agent\":\"claude\",\"milestone_id\":\"${MID_F}\",\"phase\":\"plan\",\"work_completed\":false,\"project_complete\":false,\"summary\":\"Claude failed to emit valid JSON; synthesized.\",\"gates_passed\":[],\"requirement_progress\":{\"covered_req_ids\":[],\"tests_added_or_modified\":[],\"commands_run\":[]},\"next_agent\":\"codex\",\"next_prompt\":\"Continue based on tools.verify output; fix remaining gates.\",\"delegate_rationale\":\"See ${OUT_FILE}.stderr and wrapper json files.\",\"stats_refs\":[\"${STAT_F}\"],\"needs_write_access\":true,\"artifacts\":[]}" > "$OUT_FILE"
  fi
fi

# Guarantee non-empty output
if [[ ! -s "$OUT_FILE" ]]; then
  echo "{}" > "$OUT_FILE"
fi

cat "$OUT_FILE"
