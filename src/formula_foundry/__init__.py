"""Formula Foundry: Parametric coupon generator for high-speed interconnect coupons.

This repository generates manufacturable, DRC-clean high-speed interconnect coupons
from a parametric geometry DSL, producing KiCad source + fabrication Gerbers/drills
+ a cryptographically hashed manifest suitable for simulation + equation discovery.

Public API
----------
The following functions provide the core workflow for coupon generation:

- :func:`load_spec` - Load a CouponSpec from a YAML/JSON file
- :func:`resolve` - Resolve a CouponSpec to concrete integer-nm geometry
- :func:`generate_kicad` - Generate KiCad board files from resolved design
- :func:`run_drc` - Run KiCad DRC check on a board file
- :func:`export_fab` - Export Gerber and drill files for fabrication

Example
-------
>>> from formula_foundry import load_spec, resolve, generate_kicad, run_drc, export_fab
>>> spec = load_spec(Path("coupon.yaml"))
>>> resolved = resolve(spec)
>>> project = generate_kicad(resolved, spec, output_dir)
>>> drc_report = run_drc(project.board_path, spec.toolchain.kicad)
>>> fab_hashes = export_fab(project.board_path, fab_dir, spec.toolchain.kicad)
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Core API functions (REQ-M1-022)
from formula_foundry.api import (
    BuildResult,
    DrcReport,
    KiCadProjectPaths,
    build_coupon,
    export_fab,
    generate_kicad,
    load_spec,
    run_drc,
    validate_spec,
)

# Constraint types
from formula_foundry.coupongen.constraints import (
    ConstraintEvaluation,
    ConstraintProof,
    ConstraintResult,
    ConstraintViolation,
    RepairInfo,
)

# Alias resolve_spec as resolve for cleaner top-level API
from formula_foundry.coupongen.resolve import (
    ResolvedDesign,
    resolve,
)

# Core spec types
from formula_foundry.coupongen.spec import CouponSpec

if TYPE_CHECKING:
    from formula_foundry import commands as commands
    from formula_foundry.coupongen.kicad.cli import KicadCliMode
    from formula_foundry.coupongen.spec import KicadToolchain

def __getattr__(name: str) -> Any:
    if name == "commands":
        return import_module("formula_foundry.commands")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Core API functions (REQ-M1-022 - orchestration by M6-M8)
    "load_spec",
    "resolve",
    "generate_kicad",
    "run_drc",
    "export_fab",
    # Additional convenience functions
    "validate_spec",
    "build_coupon",
    # Core types
    "CouponSpec",
    "ResolvedDesign",
    "BuildResult",
    "DrcReport",
    "KiCadProjectPaths",
    # Constraint types
    "ConstraintEvaluation",
    "ConstraintProof",
    "ConstraintResult",
    "ConstraintViolation",
    "RepairInfo",
    # Command helpers (REQ-M1-018)
    "commands",
]
