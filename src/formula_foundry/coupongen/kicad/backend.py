"""KiCad backend interface and implementations.

This module defines the IKiCadBackend interface and the primary BackendA
implementation that uses S-expression generation for headless .kicad_pcb
file creation.

Satisfies REQ-M1-012 and REQ-M1-013.
"""

from __future__ import annotations

import abc
from pathlib import Path

from ..resolve import ResolvedDesign
from ..spec import CouponSpec
from .board_writer import (
    BoardWriter,
    build_board_text,
    deterministic_uuid,
    deterministic_uuid_indexed,
    write_board,
)


class IKiCadBackend(abc.ABC):
    """Abstract interface for KiCad backend implementations.

    Backends are responsible for generating KiCad project files from
    CouponSpec and ResolvedDesign.
    """

    name: str

    @abc.abstractmethod
    def write_board(
        self, spec: CouponSpec, resolved: ResolvedDesign, out_dir: Path
    ) -> Path:
        """Write a KiCad board file.

        Args:
            spec: Coupon specification.
            resolved: Resolved design with concrete parameters.
            out_dir: Output directory for the board file.

        Returns:
            Path to the generated .kicad_pcb file.
        """
        raise NotImplementedError


class BackendA(IKiCadBackend):
    """Primary backend using S-expression generation.

    This backend generates .kicad_pcb files using the sexpr module,
    with deterministic UUIDv5-based tstamp generation. It is headless
    and does not require KiCad to be installed for file generation.
    """

    name = "sexpr"

    def write_board(
        self, spec: CouponSpec, resolved: ResolvedDesign, out_dir: Path
    ) -> Path:
        """Write a KiCad board file using S-expression generation.

        Args:
            spec: Coupon specification.
            resolved: Resolved design with concrete parameters.
            out_dir: Output directory for the board file.

        Returns:
            Path to the generated .kicad_pcb file.
        """
        return write_board(spec, resolved, out_dir)


__all__ = [
    "BackendA",
    "BoardWriter",
    "IKiCadBackend",
    "build_board_text",
    "deterministic_uuid",
    "deterministic_uuid_indexed",
    "write_board",
]
