"""Public API helpers for the canonical coupon build pipeline."""

from __future__ import annotations

from pathlib import Path

from formula_foundry.coupongen.api import (
    BuildResult,
    ConstraintMode,
    DrcReport,
    KiCadProjectPaths,
    KicadRunnerProtocol,
    export_fab,
    generate_kicad,
    load_spec,
    run_drc,
    validate_spec,
)
from formula_foundry.coupongen.kicad import BackendA
from formula_foundry.coupongen.kicad.cli import KicadCliMode
from formula_foundry.coupongen.spec import CouponSpec
from formula_foundry.pipeline import run_build_pipeline


def build_coupon(
    spec: CouponSpec,
    *,
    out_root: Path,
    mode: KicadCliMode = "local",
    constraint_mode: ConstraintMode | None = None,
    runner: KicadRunnerProtocol | None = None,
    backend: BackendA | None = None,
    kicad_cli_version: str | None = None,
    lock_file: Path | None = None,
) -> BuildResult:
    """Build a coupon from a specification using the canonical pipeline."""
    return run_build_pipeline(
        spec,
        out_root=out_root,
        kicad_mode=mode,
        validation_mode="engine",
        constraint_mode=constraint_mode,
        runner=runner,
        backend=backend,
        kicad_cli_version=kicad_cli_version,
        lock_file=lock_file,
    )


__all__ = [
    "BuildResult",
    "DrcReport",
    "KiCadProjectPaths",
    "build_coupon",
    "export_fab",
    "generate_kicad",
    "load_spec",
    "run_drc",
    "validate_spec",
]
