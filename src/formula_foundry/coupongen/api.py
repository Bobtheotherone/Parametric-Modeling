from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import yaml  # type: ignore[import-untyped]

from formula_foundry.substrate import canonical_json_dumps

from .constraints import ConstraintEvaluation, constraint_proof_payload, enforce_constraints
from .families import validate_family
from .hashing import canonical_hash_export_text, coupon_id_from_design_hash
from .kicad import BackendA, KicadCliRunner
from .kicad.cli import KicadCliMode
from .manifest import build_manifest, load_manifest, toolchain_hash, write_manifest
from .resolve import ResolvedDesign, design_hash, resolve
from .spec import CouponSpec, KicadToolchain


@dataclass(frozen=True)
class KiCadProjectPaths:
    board_path: Path
    project_dir: Path


@dataclass(frozen=True)
class DrcReport:
    report_path: Path
    returncode: int


@dataclass(frozen=True)
class BuildResult:
    output_dir: Path
    design_hash: str
    coupon_id: str
    manifest_path: Path
    cache_hit: bool
    toolchain_hash: str


class KicadRunnerProtocol(Protocol):
    def run_drc(self, board_path: Path, report_path: Path) -> subprocess.CompletedProcess[str]:
        ...

    def export_gerbers(self, board_path: Path, out_dir: Path) -> subprocess.CompletedProcess[str]:
        ...

    def export_drill(self, board_path: Path, out_dir: Path) -> subprocess.CompletedProcess[str]:
        ...


def load_spec(path: Path) -> CouponSpec:
    payload = _load_spec_payload(path)
    return CouponSpec.model_validate(payload)


def resolve_spec(spec: CouponSpec) -> ResolvedDesign:
    return resolve(spec)


def validate_spec(spec: CouponSpec, *, out_dir: Path) -> ConstraintEvaluation:
    validate_family(spec)
    evaluation = enforce_constraints(spec)
    _write_validation_outputs(evaluation, out_dir)
    return evaluation


def generate_kicad(
    resolved: ResolvedDesign,
    spec: CouponSpec,
    out_dir: Path,
    *,
    backend: BackendA | None = None,
) -> KiCadProjectPaths:
    backend = backend or BackendA()
    board_path = backend.write_board(spec, resolved, out_dir)
    return KiCadProjectPaths(board_path=board_path, project_dir=out_dir)


def run_drc(
    board_path: Path,
    toolchain: KicadToolchain,
    *,
    mode: KicadCliMode = "local",
    report_path: Path | None = None,
    runner: KicadRunnerProtocol | None = None,
) -> DrcReport:
    resolved_report = report_path or board_path.parent / "drc.json"
    runner = runner or KicadCliRunner(mode=mode, docker_image=toolchain.docker_image)
    proc = runner.run_drc(board_path, resolved_report)
    return DrcReport(report_path=resolved_report, returncode=proc.returncode)


def export_fab(
    board_path: Path,
    out_dir: Path,
    toolchain: KicadToolchain,
    *,
    mode: KicadCliMode = "local",
    runner: KicadRunnerProtocol | None = None,
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    runner = runner or KicadCliRunner(mode=mode, docker_image=toolchain.docker_image)
    gerber_dir = out_dir / "gerbers"
    drill_dir = out_dir / "drill"
    gerber_dir.mkdir(parents=True, exist_ok=True)
    drill_dir.mkdir(parents=True, exist_ok=True)
    runner.export_gerbers(board_path, gerber_dir)
    runner.export_drill(board_path, drill_dir)
    return _hash_export_tree(out_dir)


def _hash_export_tree(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            text = path.read_text(encoding="utf-8", errors="ignore")
            hashes[path.relative_to(root).as_posix()] = canonical_hash_export_text(text)
    return hashes


def _load_spec_payload(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("CouponSpec file must contain a mapping")
    return payload


def build_coupon(
    spec: CouponSpec,
    *,
    out_root: Path,
    mode: KicadCliMode = "local",
    runner: KicadRunnerProtocol | None = None,
    backend: BackendA | None = None,
    kicad_cli_version: str | None = None,
) -> BuildResult:
    validate_family(spec)
    evaluation = enforce_constraints(spec)
    resolved = evaluation.resolved
    design_hash_value = design_hash(resolved)
    coupon_id = coupon_id_from_design_hash(design_hash_value)
    output_dir = out_root / f"{coupon_id}-{design_hash_value}"
    manifest_path = output_dir / "manifest.json"

    toolchain_meta = {
        "kicad_version": evaluation.spec.toolchain.kicad.version,
        "docker_image": evaluation.spec.toolchain.kicad.docker_image,
        "mode": mode,
        "kicad_cli_version": kicad_cli_version or "unknown",
    }
    toolchain_hash_value = toolchain_hash(toolchain_meta)

    if manifest_path.exists():
        manifest = load_manifest(manifest_path)
        if (
            manifest.get("design_hash") == design_hash_value
            and manifest.get("toolchain_hash") == toolchain_hash_value
        ):
            return BuildResult(
                output_dir=output_dir,
                design_hash=design_hash_value,
                coupon_id=coupon_id,
                manifest_path=manifest_path,
                cache_hit=True,
                toolchain_hash=toolchain_hash_value,
            )

    _write_validation_outputs(evaluation, output_dir)
    project = generate_kicad(resolved, evaluation.spec, output_dir, backend=backend)
    report = run_drc(
        project.board_path,
        evaluation.spec.toolchain.kicad,
        mode=mode,
        runner=runner,
    )
    if evaluation.spec.constraints.drc.must_pass and report.returncode != 0:
        raise RuntimeError(f"KiCad DRC failed with returncode {report.returncode}")
    export_hashes = export_fab(
        project.board_path,
        output_dir / "fab",
        evaluation.spec.toolchain.kicad,
        mode=mode,
        runner=runner,
    )
    manifest = build_manifest(
        spec=evaluation.spec,
        resolved=resolved,
        proof=evaluation.proof,
        design_hash=design_hash_value,
        coupon_id=coupon_id,
        toolchain=toolchain_meta,
        toolchain_hash_value=toolchain_hash_value,
        export_hashes=export_hashes,
        drc_report_path=report.report_path,
        drc_returncode=report.returncode,
    )
    write_manifest(manifest_path, manifest)
    return BuildResult(
        output_dir=output_dir,
        design_hash=design_hash_value,
        coupon_id=coupon_id,
        manifest_path=manifest_path,
        cache_hit=False,
        toolchain_hash=toolchain_hash_value,
    )


def _write_validation_outputs(evaluation: ConstraintEvaluation, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    resolved_payload = canonical_json_dumps(evaluation.resolved.model_dump(mode="json"))
    (out_dir / "resolved_design.json").write_text(resolved_payload, encoding="utf-8")
    proof_payload = canonical_json_dumps(constraint_proof_payload(evaluation.proof))
    (out_dir / "constraint_proof.json").write_text(proof_payload, encoding="utf-8")
