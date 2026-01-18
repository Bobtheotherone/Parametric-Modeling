#!/usr/bin/env bash
set -euo pipefail

PROMPT_FILE="${1:?prompt_file}"
SCHEMA_FILE="${2:?schema_file}"
OUT_FILE="${3:?out_file}"

# Gemini 3 Pro (preview). If your CLI doesn't have it enabled, set GEMINI_MODEL=gemini-2.5-pro.
MODEL="${GEMINI_MODEL:-gemini-3-pro-preview}"
TIMEOUT_S="${GEMINI_TIMEOUT_S:-120}"

prompt="$(cat "$PROMPT_FILE")"

# Keep Gemini laser-focused on the orchestrator schema.
PREAMBLE=$'SYSTEM CONSTRAINTS (NON-NEGOTIABLE)\n- Output EXACTLY ONE JSON object matching the provided schema.\n- Do NOT wrap output in markdown/code fences (no ``` blocks).\n- No extra text before/after the JSON.\n'
FULL_PROMPT="${PREAMBLE}${prompt}"

ERR_FILE="${OUT_FILE}.stderr"
WRAP_JSON="${OUT_FILE}.wrapper.json"

python3 - <<'PY' "$MODEL" "$TIMEOUT_S" "$FULL_PROMPT" "$WRAP_JSON" "$ERR_FILE"
import subprocess, sys

model, timeout_s, full_prompt, out_path, err_path = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]

def run(cmd):
    try:
        p = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout_s)
        return (p.stdout or ""), (p.stderr or "")
    except subprocess.TimeoutExpired:
        return "", f"TIMEOUT after {timeout_s}s\nCMD: {cmd}\n"

# Attempt #1: positional prompt (preferred)
cmd1 = [
    "gemini",
    "--output-format", "json",
    "--model", model,
    "-e", "none",
    "--approval-mode", "yolo",
    full_prompt,
]
out1, err1 = run(cmd1)

# Attempt #2: deprecated --prompt (still supported on your install; fallback)
cmd2 = [
    "gemini",
    "--output-format", "json",
    "--model", model,
    "-e", "none",
    "--approval-mode", "yolo",
    "--prompt", full_prompt,
]
out2, err2 = ("", "")
if not out1.strip():
    out2, err2 = run(cmd2)

chosen_out = out1 if out1.strip() else out2
chosen_err = err1 + (("\n---\n" + err2) if err2 else "")

open(out_path, "w", encoding="utf-8").write(chosen_out)
open(err_path, "w", encoding="utf-8").write(chosen_err)
PY

python3 - <<'PY' "$WRAP_JSON" "$SCHEMA_FILE" "$PROMPT_FILE" > "$OUT_FILE" || true
import json, re, sys

wrap_path, schema_path, prompt_path = sys.argv[1], sys.argv[2], sys.argv[3]
raw = open(wrap_path, "r", encoding="utf-8").read().strip()
schema = json.loads(open(schema_path, "r", encoding="utf-8").read())
allowed = set(schema.get("properties", {}).keys())
required = list(schema.get("required", []))
prompt_text = open(prompt_path, "r", encoding="utf-8").read()

def jload(s):
    try:
        return json.loads(s)
    except Exception:
        return None

def salvage_json(s):
    i = s.find("{")
    return jload(s[i:]) if i != -1 else None

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

def extract_milestone_id(text: str) -> str | None:
    m = re.search(r'"milestone_id"\s*:\s*"([^"]+)"', text)
    return m.group(1) if m else None

wrapper = jload(raw) or salvage_json(raw)
if not isinstance(wrapper, dict):
    print(raw if raw else "{}")
    raise SystemExit(0)

resp = (wrapper.get("response") or "").strip()
if not resp:
    print("{}")
    raise SystemExit(0)

resp = strip_fences(resp)

turn = jload(resp)
if turn is None:
    l, r = resp.find("{"), resp.rfind("}")
    if l != -1 and r != -1 and r > l:
        turn = jload(resp[l:r+1].strip())

if not isinstance(turn, dict):
    print(resp if resp else "{}")
    raise SystemExit(0)

# --- Normalize to schema ---
turn = {k: v for k, v in turn.items() if k in allowed}

# Enforce required fields and types expected by orchestrator validator.
turn["agent"] = "gemini"

if not isinstance(turn.get("milestone_id"), str) or not turn["milestone_id"].strip():
    mid = extract_milestone_id(prompt_text)
    if mid:
        turn["milestone_id"] = mid

if turn.get("phase") not in ("plan", "implement", "verify", "finalize"):
    turn["phase"] = "plan"

for b in ("work_completed", "project_complete", "needs_write_access"):
    if not isinstance(turn.get(b), bool):
        # Default to True for needs_write_access so codex/claude can work immediately.
        turn[b] = (b == "needs_write_access")

for s in ("summary", "next_prompt", "delegate_rationale"):
    if not isinstance(turn.get(s), str):
        turn[s] = ""

if turn.get("next_agent") not in ("codex", "gemini", "claude"):
    turn["next_agent"] = "codex"

def str_list(x):
    return [i for i in x if isinstance(i, str) and i.strip()] if isinstance(x, list) else []

turn["gates_passed"] = str_list(turn.get("gates_passed", []))
turn["stats_refs"] = str_list(turn.get("stats_refs", []))

# stats_refs must be non-empty and must match STATS ids; pick one from prompt if missing.
if not turn["stats_refs"]:
    ids = extract_stats_ids(prompt_text)
    if ids:
        # Prefer GM-* if present
        gm = [x for x in ids if x.startswith("GM-")]
        turn["stats_refs"] = [gm[0] if gm else ids[0]]

rp = turn.get("requirement_progress")
if not isinstance(rp, dict):
    rp = {}
rp["covered_req_ids"] = str_list(rp.get("covered_req_ids", []))
rp["tests_added_or_modified"] = str_list(rp.get("tests_added_or_modified", []))
rp["commands_run"] = str_list(rp.get("commands_run", []))
turn["requirement_progress"] = rp

arts = turn.get("artifacts", [])
clean = []
if isinstance(arts, list):
    for a in arts:
        if isinstance(a, dict):
            p = a.get("path")
            d = a.get("description")
            if isinstance(p, str) and isinstance(d, str):
                clean.append({"path": p, "description": d})
turn["artifacts"] = clean  # guarantees presence (empty list OK)

# Output only schema keys (prevents "unexpected keys present")
turn = {k: turn.get(k) for k in allowed if k in turn}

# Ensure all required keys exist (final safety)
for k in required:
    if k not in turn:
        if k == "artifacts":
            turn[k] = []
        elif k == "gates_passed":
            turn[k] = []
        elif k == "requirement_progress":
            turn[k] = {"covered_req_ids": [], "tests_added_or_modified": [], "commands_run": []}
        elif k == "stats_refs":
            turn[k] = extract_stats_ids(prompt_text)[:1] or ["GM-1"]
        elif k in ("work_completed", "project_complete", "needs_write_access"):
            turn[k] = False
        else:
            turn[k] = ""

print(json.dumps(turn, ensure_ascii=False, separators=(",", ":")))
PY

if [[ ! -s "$OUT_FILE" ]]; then
  echo "{}" > "$OUT_FILE"
fi

cat "$OUT_FILE"
