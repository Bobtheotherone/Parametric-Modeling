#!/usr/bin/env bash
set -euo pipefail

PROMPT_FILE="${1:?prompt_file}"
SCHEMA_FILE="${2:?schema_file}"
OUT_FILE="${3:?out_file}"

MODEL="${CLAUDE_MODEL:-claude-3-7-sonnet-latest}"
TIMEOUT_S="${CLAUDE_TIMEOUT_S:-180}"
CLAUDE_BIN="${CLAUDE_BIN:-claude}"
CLAUDE_ARGS_JSON_MODE="${CLAUDE_ARGS_JSON_MODE:-}"
CLAUDE_TOOLS="${CLAUDE_TOOLS:-}"
CLAUDE_HELP_TIMEOUT_S="${CLAUDE_HELP_TIMEOUT_S:-5}"
SMOKE_DIR="${FF_AGENT_SMOKE_DIR:-}"
WRITE_ACCESS="${WRITE_ACCESS:-0}"

CREDS_MISSING="true"
for var in ANTHROPIC_API_KEY CLAUDE_API_KEY; do
  if [[ -n "${!var:-}" ]]; then
    CREDS_MISSING="false"
    break
  fi
done

prompt="$(cat "$PROMPT_FILE")"

if [[ -n "$SMOKE_DIR" && "$WRITE_ACCESS" == "1" ]]; then
  mkdir -p "$SMOKE_DIR"
  printf '%s %s\n' "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" "claude" > "$SMOKE_DIR/claude.txt"
fi

# Headless/orchestrator safety: tools are disabled in this wrapper invocation.
# Prevent Claude Code from emitting tool-markup (<task>, <read>, <edit>, <bash>).
prompt="${prompt}

REMINDER (NON-NEGOTIABLE):
- Tools are DISABLED. Do NOT output <task>, <read>, <edit>, <bash> blocks.
- Output EXACTLY ONE JSON object matching bridge/turn.schema.json.
- No markdown. No code fences. No extra text before/after the JSON."

# Claude Code expects --json-schema as an *inline JSON string* (not a file path).
SCHEMA_JSON="$(
  python3 -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1])), separators=(",",":")))' \
    "$SCHEMA_FILE"
)"

ERR_SCHEMA="${OUT_FILE}.stderr.schema"
ERR_PLAIN="${OUT_FILE}.stderr.plain"
WRAP_SCHEMA="${OUT_FILE}.wrapper_schema.json"
WRAP_PLAIN="${OUT_FILE}.wrapper_plain.json"

if [[ "$CREDS_MISSING" == "true" ]]; then
  python3 - <<'PY' "$SCHEMA_FILE" "$PROMPT_FILE" "$OUT_FILE"
import json
import re
import sys

schema_path, prompt_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
schema = json.loads(open(schema_path, "r", encoding="utf-8").read())
prompt_text = open(prompt_path, "r", encoding="utf-8").read()

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

stats_ids = extract_stats_ids(prompt_text)
milestone_id = extract_milestone_id(prompt_text)

summary = "Claude credentials missing (creds_missing=true). Diagnostics: cmd=(skipped) rc=missing_creds stderr="

turn = {
    "agent": "claude",
    "milestone_id": milestone_id,
    "phase": "plan",
    "work_completed": False,
    "project_complete": False,
    "summary": summary,
    "gates_passed": [],
    "requirement_progress": {"covered_req_ids": [], "tests_added_or_modified": [], "commands_run": []},
    "next_agent": "codex",
    "next_prompt": "Set ANTHROPIC_API_KEY or CLAUDE_API_KEY, then re-run claude preflight.",
    "delegate_rationale": "creds_missing=true",
    "stats_refs": choose_stats_ref(stats_ids),
    "needs_write_access": True,
    "artifacts": [],
}

required = schema.get("required", [])
for k in required:
    if k not in turn:
        if k == "artifacts":
            turn[k] = []
        elif k == "gates_passed":
            turn[k] = []
        elif k == "requirement_progress":
            turn[k] = {"covered_req_ids": [], "tests_added_or_modified": [], "commands_run": []}
        elif k == "stats_refs":
            turn[k] = choose_stats_ref(stats_ids)
        elif k in ("work_completed", "project_complete", "needs_write_access"):
            turn[k] = False
        else:
            turn[k] = ""

with open(out_path, "w", encoding="utf-8") as handle:
    json.dump(turn, handle, ensure_ascii=False, separators=(",", ":"))
PY
  cat "$OUT_FILE"
  exit 0
fi

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

timeout_s = int(sys.argv[1])
wrap_path = sys.argv[2]
err_path = sys.argv[3]
meta_path = sys.argv[4]
cmd = sys.argv[5:]

out = ""
err = ""
rc = 0
try:
    proc = subprocess.Popen(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    stdout, stderr = proc.communicate(timeout=timeout_s)
    out = stdout or ""
    err = stderr or ""
    rc = proc.returncode
except subprocess.TimeoutExpired as exc:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except Exception:
        pass
    out = (exc.stdout or "")
    err = (exc.stderr or "")
    rc = 124
    err = (err or "") + f"\\nTIMEOUT after {timeout_s}s\\n"

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

# If that produced an error wrapper or no usable result, we’ll fall back to plain.
# Normalization logic below will choose the better wrapper automatically.
run_claude "plain" "$WRAP_PLAIN" "$ERR_PLAIN"

python3 - <<'PY' "$WRAP_SCHEMA" "$WRAP_PLAIN" "$SCHEMA_FILE" "$PROMPT_FILE" "$ERR_SCHEMA" "$ERR_PLAIN" > "$OUT_FILE" || true
import json, re, sys

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

# Choose best wrapper: prefer schema wrapper if not error and parseable.
ws = read_json(wrap_schema)
wp = read_json(wrap_plain)

candidates = []
if isinstance(ws, dict):
    candidates.append(ws)
if isinstance(wp, dict):
    candidates.append(wp)

best_turn = None
best_model_text = ""
best_is_error = True

for w in candidates:
    is_error, mt = wrapper_to_model_text(w)
    if not mt:
        continue
    t = parse_turn_from_text(mt)
    if t is not None:
        # Prefer non-error; otherwise first parseable.
        if best_turn is None or (best_is_error and not is_error):
            best_turn = t
            best_model_text = mt
            best_is_error = is_error

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
    return {
        "agent": "claude",
        "milestone_id": milestone_id,
        "phase": "plan",
        "work_completed": False,
        "project_complete": False,
        "summary": f"{reason} (creds_missing=false) Diagnostics:\\n{diag}",
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
    out = synthesize("Claude did not emit a parseable JSON turn; synthesized.", best_model_text)
    print(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    raise SystemExit(0)

# Normalize parsed turn to strict schema
t = best_turn
out = {}
out["agent"] = "claude"
out["milestone_id"] = milestone_id

phase = t.get("phase")
out["phase"] = phase if phase in ("plan", "implement", "verify", "finalize") else "plan"

out["work_completed"] = to_bool(t.get("work_completed"), False)
out["project_complete"] = to_bool(t.get("project_complete"), False)

out["summary"] = to_str(t.get("summary"), "")
out["gates_passed"] = to_str_list(t.get("gates_passed", []))
out["requirement_progress"] = normalize_requirement_progress(t.get("requirement_progress"))

na = t.get("next_agent")
out["next_agent"] = na if na in ("codex", "gemini", "claude") else "codex"
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
print(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
PY

# If normalization failed for some unexpected reason, emit a minimal valid JSON turn.
if [[ ! -s "$OUT_FILE" ]]; then
  echo '{"agent":"claude","milestone_id":"M0","phase":"plan","work_completed":false,"project_complete":false,"summary":"Claude wrapper produced empty output; synthesized.","gates_passed":[],"requirement_progress":{"covered_req_ids":[],"tests_added_or_modified":[],"commands_run":[]},"next_agent":"codex","next_prompt":"Continue using tools.verify output; fix remaining gates and commit.","delegate_rationale":"(empty)","stats_refs":["CL-1"],"needs_write_access":true,"artifacts":[]}' > "$OUT_FILE"
fi

cat "$OUT_FILE"
exit 0
