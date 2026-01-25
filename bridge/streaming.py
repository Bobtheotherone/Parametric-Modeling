from __future__ import annotations

import contextlib
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any


def run_cmd_with_streaming(
    *,
    cmd: list[str],
    cwd: Path,
    env: dict[str, str],
    agent: str,
    stream_mode: str,
    call_dir: Path,
) -> tuple[int, str, str]:
    """Run subprocess with prefixed streaming and log file output."""
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
    )

    out_chunks: list[str] = []
    err_chunks: list[str] = []

    call_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = call_dir / "agent_stdout.log"
    stderr_log = call_dir / "agent_stderr.log"

    stream_mode = stream_mode.lower()
    stream_stdout = stream_mode in ("stdout", "both")
    stream_stderr = stream_mode in ("stderr", "both")

    def _pump(src: Any, chunks: list[str], log_path: Path, *, is_err: bool) -> None:
        try:
            assert src is not None
            with log_path.open("w", encoding="utf-8") as log_file:
                for line in iter(src.readline, ""):
                    chunks.append(line)
                    log_file.write(line)
                    log_file.flush()
                    if is_err:
                        if stream_stderr:
                            sys.stderr.write(f"[{agent}][stderr] {line}")
                            sys.stderr.flush()
                    else:
                        if stream_stdout:
                            sys.stdout.write(f"[{agent}][stdout] {line}")
                            sys.stdout.flush()
        finally:
            with contextlib.suppress(Exception):
                src.close()

    t_out = threading.Thread(
        target=_pump,
        args=(proc.stdout, out_chunks, stdout_log),
        kwargs={"is_err": False},
        daemon=True,
    )
    t_err = threading.Thread(
        target=_pump,
        args=(proc.stderr, err_chunks, stderr_log),
        kwargs={"is_err": True},
        daemon=True,
    )
    t_out.start()
    t_err.start()

    rc = proc.wait()
    t_out.join(timeout=2)
    t_err.join(timeout=2)
    return rc, "".join(out_chunks), "".join(err_chunks)
