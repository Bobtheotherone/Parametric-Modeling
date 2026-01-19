from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path
from typing import Any

import jsonschema  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "bridge" / "turn.schema.json"
GEMINI_WRAPPER = ROOT / "bridge" / "agents" / "gemini.sh"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR)


def _validate_turn(payload: dict[str, Any]) -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.validate(instance=payload, schema=schema)


def test_gemini_wrapper_emits_schema_valid_json_on_error(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    gemini_stub = bin_dir / "gemini"
    _write_executable(gemini_stub, "#!/usr/bin/env bash\necho 'not-json'\n")

    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("**Milestone:** M0\nGM-1\n", encoding="utf-8")
    out_path = tmp_path / "out.json"

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"

    subprocess.run(
        [str(GEMINI_WRAPPER), str(prompt_path), str(SCHEMA_PATH), str(out_path)],
        check=True,
        env=env,
        text=True,
    )

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    _validate_turn(payload)
    assert payload["agent"] == "gemini"
