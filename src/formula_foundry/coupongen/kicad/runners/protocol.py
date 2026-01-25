"""IKicadRunner protocol definition.

This module defines the runner protocol that all KiCad CLI runner
implementations must follow.

Satisfies CP-1.2 and Section 13.1.2 requirements.

Note: ZonePolicy and DEFAULT_ZONE_POLICY are re-exported from
kicad.policy for backward compatibility. The authoritative source
is kicad/policy.py (REQ-M1-006).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

# Re-export from the authoritative policy module for backward compatibility
from ..policy import DEFAULT_ZONE_POLICY, ZonePolicy


@dataclass(frozen=True)
class KicadRunResult:
    """Result of a kicad-cli command execution.

    Attributes:
        returncode: Exit code from the process (0 = success, 5 = DRC violations).
        stdout: Standard output from the command.
        stderr: Standard error from the command.
        command: The full command that was executed.
    """

    returncode: int
    stdout: str
    stderr: str
    command: list[str]

    @property
    def success(self) -> bool:
        """Return True if the command succeeded (exit code 0)."""
        return self.returncode == 0

    @property
    def has_drc_violations(self) -> bool:
        """Return True if DRC returned violations (exit code 5)."""
        return self.returncode == 5


@runtime_checkable
class IKicadRunner(Protocol):
    """Protocol for KiCad CLI runner implementations.

    Runners are responsible for executing kicad-cli commands, either
    locally or via Docker. They must handle:
    - DRC execution with --severity-all --exit-code-violations --format json
    - Gerber and drill exports
    - Version string retrieval

    All runners must be stateless and thread-safe.
    """

    def run(
        self,
        args: Sequence[str],
        cwd: Path,
        env: Mapping[str, str] | None = None,
    ) -> KicadRunResult:
        """Execute a kicad-cli command.

        Args:
            args: Command arguments to pass to kicad-cli (e.g., ["pcb", "drc", ...]).
            cwd: Working directory for the command. For Docker runners, this
                directory is mounted into the container.
            env: Optional environment variables to set for the command.

        Returns:
            KicadRunResult containing exit code, stdout, stderr, and the full command.
        """
        ...

    def kicad_cli_version(self, cwd: Path) -> str:
        """Get the kicad-cli version string.

        Args:
            cwd: Working directory (used for Docker volume mounting).

        Returns:
            Version string from kicad-cli --version output.

        Raises:
            RuntimeError: If version cannot be determined.
        """
        ...


__all__ = [
    "IKicadRunner",
    "KicadRunResult",
    "ZonePolicy",
    "DEFAULT_ZONE_POLICY",
]
