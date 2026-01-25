"""KiCad integration module for coupongen.

This module provides:
- S-expression parsing and generation (sexpr)
- Board file writing with deterministic UUIDs (board_writer, backend)
- Silkscreen annotations for provenance (annotations)
- Artifact canonicalization for hashing (canonicalize)
- KiCad CLI runner (cli)
- Zone policy for DRC/export (policy) - REQ-M1-006
- CLI flag builders (cli_flags) - REQ-M1-006

Satisfies REQ-M1-006, REQ-M1-010, REQ-M1-012 and REQ-M1-013.
"""

from __future__ import annotations

from .annotations import (
    build_annotations_from_spec,
    build_coupon_annotation,
    build_silkscreen_text,
)
from .backend import (
    BackendA,
    BoardWriter,
    IKiCadBackend,
    build_board_text,
    deterministic_uuid,
    deterministic_uuid_indexed,
    write_board,
)
from .canonicalize import (
    canonical_hash_board,
    canonical_hash_drc_json,
    canonical_hash_drill,
    canonical_hash_export,
    canonical_hash_file,
    canonical_hash_files,
    canonical_hash_gerber,
    canonical_hash_kicad_pcb,
    canonicalize_board,
    canonicalize_drc_json,
    canonicalize_drill,
    canonicalize_export,
    canonicalize_gerber,
    canonicalize_kicad_pcb,
    normalize_line_endings,
)
from .cli import (
    DEFAULT_TIMEOUT_SEC,
    KicadCliError,
    KicadCliMode,
    KicadCliRunner,
    KicadCliTimeoutError,
    KicadErrorCode,
    ParsedKicadError,
    build_define_var_args,
    build_drc_args,
    get_kicad_cli_version,
    parse_kicad_error,
    zone_policy_record,
)
from .cli_flags import (
    SeverityLevel,
    build_drc_flags,
    build_export_drill_flags,
    build_export_gerber_flags,
    get_drc_refill_flag,
    get_export_check_flag,
)
from .policy import (
    DEFAULT_ZONE_POLICY,
    ZonePolicy,
    get_zone_policy_record,
)
from .runners import (
    DEFAULT_DOCKER_TIMEOUT_SEC,
    DockerKicadRunner,
    DockerKicadTimeoutError,
    DockerMountError,
    IKicadRunner,
    KicadRunResult,
    load_docker_image_ref,
    parse_kicad_version,
)
from .sexpr import (
    SExprAtom,
    SExprList,
    SExprNode,
    SExprParseError,
    SExprWriter,
    dump,
    dump_compact,
    format_atom,
    format_decimal,
    format_string,
    mm_point,
    nm_to_mm,
    parse,
    parse_all,
)

__all__ = [
    # Annotations
    "build_annotations_from_spec",
    "build_coupon_annotation",
    "build_silkscreen_text",
    # Backend and board writer
    "BackendA",
    "BoardWriter",
    "IKiCadBackend",
    "build_board_text",
    "deterministic_uuid",
    "deterministic_uuid_indexed",
    "write_board",
    # CLI - Constants
    "DEFAULT_TIMEOUT_SEC",
    # CLI - Types and Enums
    "KicadCliMode",
    "KicadErrorCode",
    "SeverityLevel",
    # CLI - Exceptions
    "KicadCliError",
    "KicadCliTimeoutError",
    # CLI - Data classes
    "ParsedKicadError",
    "KicadCliRunner",
    # CLI - Functions
    "build_define_var_args",
    "build_drc_args",
    "get_kicad_cli_version",
    "parse_kicad_error",
    "zone_policy_record",
    # CLI Flags - Functions (REQ-M1-006)
    "build_drc_flags",
    "build_export_drill_flags",
    "build_export_gerber_flags",
    "get_drc_refill_flag",
    "get_export_check_flag",
    # Policy - Classes and Constants (REQ-M1-006)
    "DEFAULT_ZONE_POLICY",
    "ZonePolicy",
    "get_zone_policy_record",
    # Runners - Constants
    "DEFAULT_DOCKER_TIMEOUT_SEC",
    # Runners - Exceptions
    "DockerKicadTimeoutError",
    "DockerMountError",
    # Runners - Classes
    "DockerKicadRunner",
    "IKicadRunner",
    "KicadRunResult",
    # Runners - Functions
    "load_docker_image_ref",
    "parse_kicad_version",
    # Canonicalization
    "canonical_hash_board",
    "canonical_hash_drill",
    "canonical_hash_drc_json",
    "canonical_hash_export",
    "canonical_hash_file",
    "canonical_hash_files",
    "canonical_hash_gerber",
    "canonical_hash_kicad_pcb",
    "canonicalize_board",
    "canonicalize_drill",
    "canonicalize_drc_json",
    "canonicalize_export",
    "canonicalize_gerber",
    "canonicalize_kicad_pcb",
    "normalize_line_endings",
    # S-expression
    "SExprAtom",
    "SExprList",
    "SExprNode",
    "SExprParseError",
    "SExprWriter",
    "dump",
    "dump_compact",
    "format_atom",
    "format_decimal",
    "format_string",
    "mm_point",
    "nm_to_mm",
    "parse",
    "parse_all",
]
