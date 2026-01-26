"""Canonical build pipeline entrypoint for coupon generation."""

from __future__ import annotations

from pathlib import Path

from formula_foundry.coupongen.api import (
    BuildResult,
    ConstraintMode,
    KicadRunnerProtocol,
    ValidationMode,
)
from formula_foundry.coupongen.api import (
    run_build_pipeline as _run_build_pipeline,
)
from formula_foundry.coupongen.kicad import BackendA
from formula_foundry.coupongen.kicad.cli import KicadCliMode
from formula_foundry.coupongen.spec import CouponSpec


def run_build_pipeline(
    spec: CouponSpec,
    *,
    out_root: Path,
    kicad_mode: KicadCliMode = "local",
    validation_mode: ValidationMode = "engine",
    constraint_mode: ConstraintMode | None = None,
    runner: KicadRunnerProtocol | None = None,
    backend: BackendA | None = None,
    kicad_cli_version: str | None = None,
    lock_file: Path | None = None,
) -> BuildResult:
    """Run the canonical pipeline: validate -> resolve -> generate -> drc -> export -> manifest."""
    return _run_build_pipeline(
        spec,
        out_root=out_root,
        kicad_mode=kicad_mode,
        validation_mode=validation_mode,
        constraint_mode=constraint_mode,
        runner=runner,
        backend=backend,
        kicad_cli_version=kicad_cli_version,
        lock_file=lock_file,
    )


__all__ = ["run_build_pipeline"]
