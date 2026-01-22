"""KiCad CLI runner implementations.

This package provides pluggable runner implementations for kicad-cli:
- DockerKicadRunner: Runs kicad-cli inside a pinned Docker container
- (Future) LocalKicadRunner: Runs system-installed kicad-cli binary

All runners implement the IKicadRunner protocol.

Satisfies CP-1.2: Authoritative CI integration with pinned Docker toolchain.
Satisfies REQ-M1-015: Timeout handling and --define-var variable injection.
"""

from __future__ import annotations

from .docker import (
    DEFAULT_DOCKER_TIMEOUT_SEC,
    DockerKicadRunner,
    DockerKicadTimeoutError,
    DockerMountError,
    build_define_var_args,
    load_docker_image_ref,
    parse_kicad_version,
)
from .protocol import IKicadRunner, KicadRunResult

__all__ = [
    # Constants
    "DEFAULT_DOCKER_TIMEOUT_SEC",
    # Exceptions
    "DockerKicadTimeoutError",
    "DockerMountError",
    # Protocol and result types
    "IKicadRunner",
    "KicadRunResult",
    # Implementations
    "DockerKicadRunner",
    # Functions
    "build_define_var_args",
    "load_docker_image_ref",
    "parse_kicad_version",
]
