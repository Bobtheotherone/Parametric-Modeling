"""KiCad CLI runner implementations.

This package provides pluggable runner implementations for kicad-cli:
- DockerKicadRunner: Runs kicad-cli inside a pinned Docker container
- (Future) LocalKicadRunner: Runs system-installed kicad-cli binary

All runners implement the IKicadRunner protocol.

Satisfies CP-1.2: Authoritative CI integration with pinned Docker toolchain.
"""

from __future__ import annotations

from .docker import DockerKicadRunner
from .protocol import IKicadRunner, KicadRunResult

__all__ = [
    "DockerKicadRunner",
    "IKicadRunner",
    "KicadRunResult",
]
