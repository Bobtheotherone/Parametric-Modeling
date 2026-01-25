"""Environment bootstrap and dependency management.

Provides controlled environment setup to prevent "missing install" failures
during verify. The bootstrap step:
1. Runs deterministic install commands (pip install -e .[dev] or uv sync)
2. Logs all operations
3. Does NOT perform arbitrary package installs

This module is called:
- Once at orchestrator run start
- During repair when MISSING_DEPENDENCY category is detected
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO


@dataclass
class BootstrapResult:
    """Result of a bootstrap operation."""

    success: bool
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    elapsed_s: float
    skipped: bool = False
    skip_reason: str = ""


def _get_install_command(project_root: Path) -> list[str] | None:
    """Determine the appropriate install command for this project.

    Returns None if no recognizable package manager setup is found.
    """
    # Check for uv.lock (uv project)
    if (project_root / "uv.lock").exists():
        return ["uv", "sync", "--dev"]

    # Check for pyproject.toml
    pyproject = project_root / "pyproject.toml"
    if pyproject.exists():
        content = pyproject.read_text(encoding="utf-8")
        # Check if it has [project] section (PEP 621)
        if "[project]" in content:
            # Use pip install -e .[dev] - most common pattern
            return [sys.executable, "-m", "pip", "install", "-e", ".[dev]", "--quiet"]

    # Check for setup.py (legacy)
    if (project_root / "setup.py").exists():
        return [sys.executable, "-m", "pip", "install", "-e", ".[dev]", "--quiet"]

    # Check for requirements.txt as fallback
    if (project_root / "requirements.txt").exists():
        return [sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "--quiet"]

    return None


def run_bootstrap(
    project_root: Path,
    log_path: Path | None = None,
    *,
    force: bool = False,
    verbose: bool = True,
    log_file: TextIO | None = None,
) -> BootstrapResult:
    """Run environment bootstrap.

    Args:
        project_root: Project root directory
        log_path: Path to write bootstrap log (optional)
        force: Force reinstall even if already bootstrapped
        verbose: Print progress messages
        log_file: Optional file handle to write logs to

    Returns:
        BootstrapResult with operation details
    """
    start_time = datetime.now(timezone.utc)

    def _log(msg: str) -> None:
        if verbose:
            print(f"[bootstrap] {msg}")
        if log_file:
            log_file.write(f"[{datetime.now(timezone.utc).isoformat()}] {msg}\n")
            log_file.flush()

    # Determine install command
    cmd = _get_install_command(project_root)
    if cmd is None:
        _log("No package manager detected, skipping bootstrap")
        return BootstrapResult(
            success=True,
            command=[],
            returncode=0,
            stdout="",
            stderr="",
            elapsed_s=0.0,
            skipped=True,
            skip_reason="No package manager detected (no pyproject.toml, setup.py, or requirements.txt)",
        )

    # Check for marker file to skip redundant bootstraps
    marker_path = project_root / ".bootstrap_done"
    if not force and marker_path.exists():
        # Check if marker is recent (within last hour)
        marker_age = datetime.now(timezone.utc).timestamp() - marker_path.stat().st_mtime
        if marker_age < 3600:  # 1 hour
            _log(f"Bootstrap marker exists and is recent ({marker_age:.0f}s old), skipping")
            return BootstrapResult(
                success=True,
                command=cmd,
                returncode=0,
                stdout="",
                stderr="",
                elapsed_s=0.0,
                skipped=True,
                skip_reason="Recent bootstrap marker exists",
            )

    _log(f"Running: {' '.join(cmd)}")

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
            env=os.environ.copy(),
        )

        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()

        if proc.returncode == 0:
            _log(f"Bootstrap completed successfully in {elapsed:.1f}s")
            # Write marker file
            marker_path.write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")
        else:
            _log(f"Bootstrap failed with rc={proc.returncode}")
            if proc.stderr:
                _log(f"stderr: {proc.stderr[:500]}")

        # Write to log file if provided
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write(f"Return code: {proc.returncode}\n")
                f.write(f"Elapsed: {elapsed:.1f}s\n")
                f.write("\n=== STDOUT ===\n")
                f.write(proc.stdout)
                f.write("\n=== STDERR ===\n")
                f.write(proc.stderr)

        return BootstrapResult(
            success=(proc.returncode == 0),
            command=cmd,
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            elapsed_s=elapsed,
        )

    except subprocess.TimeoutExpired:
        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
        _log("Bootstrap timed out after 600s")
        return BootstrapResult(
            success=False,
            command=cmd,
            returncode=-1,
            stdout="",
            stderr="Bootstrap timed out after 600s",
            elapsed_s=elapsed,
        )
    except Exception as e:
        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
        _log(f"Bootstrap error: {e}")
        return BootstrapResult(
            success=False,
            command=cmd,
            returncode=-1,
            stdout="",
            stderr=str(e),
            elapsed_s=elapsed,
        )


def clear_bootstrap_marker(project_root: Path) -> None:
    """Clear the bootstrap marker to force re-bootstrap on next run."""
    marker_path = project_root / ".bootstrap_done"
    if marker_path.exists():
        marker_path.unlink()
