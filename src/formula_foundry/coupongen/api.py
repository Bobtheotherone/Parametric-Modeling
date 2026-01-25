"""Build pipeline API for coupon generation (CP-3.5).

This module provides the public API for coupon generation, including:
- load_spec: Load a CouponSpec from YAML/JSON file
- validate_spec: Validate spec using ConstraintEngine (Tier 0-3)
- generate_kicad: Generate KiCad project files
- run_drc: Run KiCad DRC on a board file
- export_fab: Export gerbers and drill files
- build_coupon: Full build pipeline (validate/repair -> generate -> DRC -> export)

CP-3.5: Pipeline Integration
- All validation uses ConstraintEngine as the single unified path
- Build flow: validate/repair (Tier0-3) -> generate -> DRC -> export
- Removes/deprecates minimal core constraints path in favor of full tiered system
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

import yaml  # type: ignore[import-untyped]

from formula_foundry.substrate import canonical_json_dumps, get_git_sha

from .constraints import (
    ConstraintEvaluation,
    TieredConstraintResult,
    constraint_proof_payload,
    enforce_constraints,
)
from .constraints.engine import ConstraintEngineResult, create_constraint_engine
from .fab_profiles import get_fab_limits, load_fab_profile
from .families import validate_family
from .hashing import canonical_hash_export_text, coupon_id_from_design_hash
from .kicad import BackendA, KicadCliRunner
from .kicad.cli import KicadCliMode
from .manifest import build_manifest, load_manifest, toolchain_hash, write_manifest
from .resolve import ResolvedDesign, design_hash, resolve
from .spec import CouponSpec, KicadToolchain
from .export import KicadExportError
from .toolchain import ToolchainLoadError, resolve_docker_image_ref
from .toolchain_capture import capture_toolchain_provenance

if TYPE_CHECKING:
    from .constraints.tiers import TieredConstraintProof

# Constraint mode type alias
ConstraintMode = Literal["REJECT", "REPAIR"]


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


@dataclass(frozen=True)
class ValidationResult:
    """Result from validate_spec_with_engine().

    Attributes:
        spec: The (possibly repaired) CouponSpec
        resolved: The ResolvedDesign from the (possibly repaired) spec
        proof: The TieredConstraintProof with per-constraint evaluations
        engine_result: The full ConstraintEngineResult
        was_repaired: True if REPAIR mode was used and changes were made
    """

    spec: CouponSpec
    resolved: ResolvedDesign
    proof: TieredConstraintProof
    engine_result: ConstraintEngineResult
    was_repaired: bool


def _ensure_docker_image_meta(toolchain_meta: dict[str, Any], docker_image: str | None) -> None:
    if not docker_image:
        return
    docker_meta = toolchain_meta.get("docker")
    if not isinstance(docker_meta, dict):
        toolchain_meta["docker"] = {"image_ref": docker_image}
        return
    docker_meta.setdefault("image_ref", docker_image)


class KicadRunnerProtocol(Protocol):
    def run_drc(self, board_path: Path, report_path: Path) -> subprocess.CompletedProcess[str]: ...

    def export_gerbers(self, board_path: Path, out_dir: Path) -> subprocess.CompletedProcess[str]: ...

    def export_drill(self, board_path: Path, out_dir: Path) -> subprocess.CompletedProcess[str]: ...


def load_spec(path: Path) -> CouponSpec:
    payload = _load_spec_payload(path)
    return CouponSpec.model_validate(payload)


def resolve_spec(spec: CouponSpec) -> ResolvedDesign:
    return resolve(spec)


def validate_spec(spec: CouponSpec, *, out_dir: Path) -> ConstraintEvaluation:
    """Validate spec using the legacy constraint system (backward compatibility).

    For new code, prefer validate_spec_with_engine() which uses the unified
    ConstraintEngine with full tiered validation (Tier 0-3).

    Args:
        spec: CouponSpec to validate
        out_dir: Directory to write validation outputs

    Returns:
        ConstraintEvaluation with spec, resolved, proof, and repair_info
    """
    validate_family(spec)
    evaluation = enforce_constraints(spec)
    _write_validation_outputs(evaluation, out_dir)
    return evaluation


def validate_spec_with_engine(
    spec: CouponSpec,
    *,
    out_dir: Path,
    mode: ConstraintMode | None = None,
) -> ValidationResult:
    """Validate spec using ConstraintEngine (CP-3.5 unified path).

    This is the preferred validation method that uses the unified ConstraintEngine
    with full tiered validation (Tier 0-3) and connectivity oracle integration.

    The build flow is: validate/repair (Tier0-3) -> generate -> DRC -> export

    Args:
        spec: CouponSpec to validate
        out_dir: Directory to write validation outputs
        mode: Override constraint mode ("REJECT" or "REPAIR").
              If None, uses spec.constraints.mode.

    Returns:
        ValidationResult with spec, resolved, proof, and engine result

    Raises:
        ConstraintViolationError: If mode is REJECT and constraints fail
    """
    validate_family(spec)

    # Determine constraint mode
    constraint_mode: ConstraintMode = mode or spec.constraints.mode  # type: ignore[assignment]

    # Get fab limits from profile
    try:
        profile = load_fab_profile(spec.fab_profile.id)
        fab_limits = get_fab_limits(profile)
    except FileNotFoundError:
        # Fall back to conservative defaults
        fab_limits = {
            "min_trace_width_nm": 100_000,
            "min_gap_nm": 100_000,
            "min_drill_nm": 200_000,
            "min_annular_ring_nm": 100_000,
            "min_via_diameter_nm": 300_000,
            "min_edge_clearance_nm": 200_000,
            "min_via_to_via_nm": 200_000,
            "min_board_width_nm": 5_000_000,
        }

    # Apply any overrides from the spec
    if spec.fab_profile.overrides:
        for key, value in spec.fab_profile.overrides.items():
            if key in fab_limits:
                fab_limits[key] = int(value)

    # Create engine and validate
    engine = create_constraint_engine(fab_limits=fab_limits)
    result = engine.validate_or_repair(spec, mode=constraint_mode)

    # Write validation outputs
    _write_engine_validation_outputs(result, out_dir)

    # Use the spec from the engine result (repaired if REPAIR mode changed anything)
    validated_spec = result.spec

    return ValidationResult(
        spec=validated_spec,
        resolved=result.resolved,
        proof=result.proof,
        engine_result=result,
        was_repaired=result.was_repaired,
    )


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
    lock_file: Path | None = None,
) -> DrcReport:
    resolved_report = report_path or board_path.parent / "drc.json"
    if runner is None:
        # Resolve docker image from lock file if spec has placeholder
        docker_image = resolve_docker_image_ref(
            toolchain.docker_image,
            lock_path=lock_file,
            mode=mode,
        )
        runner = KicadCliRunner(mode=mode, docker_image=docker_image)
    proc = runner.run_drc(board_path, resolved_report)
    return DrcReport(report_path=resolved_report, returncode=proc.returncode)


def export_fab(
    board_path: Path,
    out_dir: Path,
    toolchain: KicadToolchain,
    *,
    mode: KicadCliMode = "local",
    runner: KicadRunnerProtocol | None = None,
    lock_file: Path | None = None,
) -> dict[str, str]:
    """Export fabrication files (Gerbers and drill) from a board.

    Raises:
        KicadExportError: If Gerber or drill export fails (non-zero exit code
            or no files produced).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if runner is None:
        # Resolve docker image from lock file if spec has placeholder
        docker_image = resolve_docker_image_ref(
            toolchain.docker_image,
            lock_path=lock_file,
            mode=mode,
        )
        runner = KicadCliRunner(mode=mode, docker_image=docker_image)
    gerber_dir = out_dir / "gerbers"
    drill_dir = out_dir / "drill"
    gerber_dir.mkdir(parents=True, exist_ok=True)
    drill_dir.mkdir(parents=True, exist_ok=True)

    # Export Gerbers with fail-fast behavior
    gerber_result = runner.export_gerbers(board_path, gerber_dir)
    if gerber_result.returncode != 0:
        raise KicadExportError(
            f"KiCad Gerber export failed for {board_path.name}",
            command=getattr(gerber_result, "args", None),
            returncode=gerber_result.returncode,
            stdout=gerber_result.stdout,
            stderr=gerber_result.stderr,
        )

    # Verify Gerbers were actually produced
    gerber_files = list(gerber_dir.glob("*.gbr")) + list(gerber_dir.glob("*.g*"))
    if not gerber_files:
        raise KicadExportError(
            f"KiCad Gerber export completed but no Gerber files were created in {gerber_dir}. "
            "This typically indicates a permissions issue (container cannot write to bind-mounted directory).",
            command=getattr(gerber_result, "args", None),
            returncode=gerber_result.returncode,
            stdout=gerber_result.stdout,
            stderr=gerber_result.stderr,
        )

    # Export drill files with fail-fast behavior
    drill_result = runner.export_drill(board_path, drill_dir)
    if drill_result.returncode != 0:
        raise KicadExportError(
            f"KiCad drill export failed for {board_path.name}",
            command=getattr(drill_result, "args", None),
            returncode=drill_result.returncode,
            stdout=drill_result.stdout,
            stderr=drill_result.stderr,
        )

    # Verify drill files were actually produced
    drill_files = list(drill_dir.glob("*.drl")) + list(drill_dir.glob("*.exc"))
    if not drill_files:
        raise KicadExportError(
            f"KiCad drill export completed but no drill files were created in {drill_dir}. "
            "This typically indicates a permissions issue (container cannot write to bind-mounted directory).",
            command=getattr(drill_result, "args", None),
            returncode=drill_result.returncode,
            stdout=drill_result.stdout,
            stderr=drill_result.stderr,
        )

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
    lock_file: Path | None = None,
) -> BuildResult:
    """Build a coupon from a specification.

    For new code, prefer build_coupon_with_engine() which uses the unified
    ConstraintEngine with full tiered validation (Tier 0-3).

    For docker mode, this function captures complete toolchain provenance
    by running kicad-cli --version inside the container (per CP-5.3).

    Args:
        spec: The coupon specification.
        out_root: Root output directory.
        mode: KiCad CLI mode ("local" or "docker").
        runner: Custom KiCad runner (for testing).
        backend: Custom KiCad backend (for testing).
        kicad_cli_version: Optional pre-captured kicad-cli version (for testing).
        lock_file: Path to toolchain lock file (for docker mode).

    Returns:
        BuildResult with output paths and cache status.

    Raises:
        ToolchainProvenanceError: If docker mode and provenance cannot be captured.
    """
    validate_family(spec)
    evaluation = enforce_constraints(spec)
    resolved = evaluation.resolved
    design_hash_value = design_hash(resolved)
    coupon_id = coupon_id_from_design_hash(design_hash_value)
    output_dir = out_root / f"{coupon_id}-{design_hash_value}"
    manifest_path = output_dir / "manifest.json"

    # Capture toolchain provenance (CP-5.1/5.3: always capture complete provenance including lock file hash)
    if kicad_cli_version is not None:
        # Pre-captured version provided (for testing)
        # Still try to load lock_file_toolchain_hash from lock file (CP-5.1)
        lock_file_hash: str | None = None
        if lock_file is not None:
            try:
                from .toolchain import load_toolchain_lock

                tc_config = load_toolchain_lock(lock_path=lock_file)
                lock_file_hash = tc_config.toolchain_hash
            except Exception:
                pass  # lock file hash is optional for testing paths
        toolchain_meta: dict[str, Any] = {
            "kicad": {
                "version": evaluation.spec.toolchain.kicad.version,
                "cli_version_output": kicad_cli_version,
            },
            "docker": {
                "image_ref": evaluation.spec.toolchain.kicad.docker_image,
            },
            "mode": mode,
            "generator_git_sha": get_git_sha(Path.cwd()) if Path.cwd().exists() else "0" * 40,
        }
        if lock_file_hash:
            toolchain_meta["lock_file_toolchain_hash"] = lock_file_hash
    else:
        # Capture provenance dynamically
        provenance = capture_toolchain_provenance(
            mode=mode,
            kicad_version=evaluation.spec.toolchain.kicad.version,
            docker_image=evaluation.spec.toolchain.kicad.docker_image,
            workdir=out_root,
            lock_file=lock_file,
        )
        toolchain_meta = provenance.to_metadata()
        # Ensure docker image_ref is set for compatibility
        if mode == "docker" and "docker" not in toolchain_meta:
            toolchain_meta["docker"] = {"image_ref": provenance.docker_image_ref}

    _ensure_docker_image_meta(toolchain_meta, evaluation.spec.toolchain.kicad.docker_image)
    toolchain_hash_value = toolchain_hash(toolchain_meta)

    if manifest_path.exists():
        manifest = load_manifest(manifest_path)
        if manifest.get("design_hash") == design_hash_value and manifest.get("toolchain_hash") == toolchain_hash_value:
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
        lock_file=lock_file,
    )
    if evaluation.spec.constraints.drc.must_pass and report.returncode != 0:
        raise RuntimeError(f"KiCad DRC failed with returncode {report.returncode}")
    export_hashes_raw = export_fab(
        project.board_path,
        output_dir / "fab",
        evaluation.spec.toolchain.kicad,
        mode=mode,
        runner=runner,
        lock_file=lock_file,
    )
    # Add fab/ prefix to export paths so manifest paths match actual file locations
    export_hashes = {f"fab/{path}": h for path, h in export_hashes_raw.items()}
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


def build_coupon_with_engine(
    spec: CouponSpec,
    *,
    out_root: Path,
    kicad_mode: KicadCliMode = "local",
    constraint_mode: ConstraintMode | None = None,
    runner: KicadRunnerProtocol | None = None,
    backend: BackendA | None = None,
    kicad_cli_version: str | None = None,
    lock_file: Path | None = None,
) -> BuildResult:
    """Build coupon using ConstraintEngine (CP-3.5 unified path).

    This is the preferred build method that uses the unified ConstraintEngine
    with full tiered validation (Tier 0-3) and connectivity oracle integration.

    Build flow: validate/repair (Tier0-3) -> generate -> DRC -> export

    For docker mode, this function captures complete toolchain provenance
    by running kicad-cli --version inside the container (per CP-5.3).
    Docker builds must never have 'unknown' values for toolchain fields.

    Args:
        spec: CouponSpec to build
        out_root: Root directory for output artifacts
        kicad_mode: KiCad CLI mode ("local" or "docker")
        constraint_mode: Override constraint mode ("REJECT" or "REPAIR").
                        If None, uses spec.constraints.mode.
        runner: Optional KiCad runner protocol for testing
        backend: Optional board writer backend
        kicad_cli_version: Optional KiCad CLI version string (for testing)
        lock_file: Path to toolchain lock file (for docker mode)

    Returns:
        BuildResult with output paths and hashes

    Raises:
        ConstraintViolationError: If constraint_mode is REJECT and constraints fail
        RuntimeError: If DRC fails and spec.constraints.drc.must_pass is True
        ToolchainProvenanceError: If docker mode and provenance cannot be captured
    """
    validate_family(spec)

    # Determine constraint mode
    mode: ConstraintMode = constraint_mode or spec.constraints.mode  # type: ignore[assignment]

    # Get fab limits from profile
    try:
        profile = load_fab_profile(spec.fab_profile.id)
        fab_limits = get_fab_limits(profile)
    except FileNotFoundError:
        # Fall back to conservative defaults
        fab_limits = {
            "min_trace_width_nm": 100_000,
            "min_gap_nm": 100_000,
            "min_drill_nm": 200_000,
            "min_annular_ring_nm": 100_000,
            "min_via_diameter_nm": 300_000,
            "min_edge_clearance_nm": 200_000,
            "min_via_to_via_nm": 200_000,
            "min_board_width_nm": 5_000_000,
        }

    # Apply any overrides from the spec
    if spec.fab_profile.overrides:
        for key, value in spec.fab_profile.overrides.items():
            if key in fab_limits:
                fab_limits[key] = int(value)

    # Create engine and validate/repair
    engine = create_constraint_engine(fab_limits=fab_limits)
    engine_result = engine.validate_or_repair(spec, mode=mode)
    spec = engine_result.spec

    resolved = engine_result.resolved
    design_hash_value = design_hash(resolved)
    coupon_id = coupon_id_from_design_hash(design_hash_value)
    output_dir = out_root / f"{coupon_id}-{design_hash_value}"
    manifest_path = output_dir / "manifest.json"

    # Capture toolchain provenance (CP-5.1/5.3: always capture complete provenance including lock file hash)
    if kicad_cli_version is not None:
        # Pre-captured version provided (for testing)
        # Still try to load lock_file_toolchain_hash from lock file (CP-5.1)
        lock_file_hash: str | None = None
        if lock_file is not None:
            try:
                from .toolchain import load_toolchain_lock

                tc_config = load_toolchain_lock(lock_path=lock_file)
                lock_file_hash = tc_config.toolchain_hash
            except Exception:
                pass  # lock file hash is optional for testing paths
        toolchain_meta: dict[str, Any] = {
            "kicad": {
                "version": spec.toolchain.kicad.version,
                "cli_version_output": kicad_cli_version,
            },
            "docker": {
                "image_ref": spec.toolchain.kicad.docker_image,
            },
            "mode": kicad_mode,
            "generator_git_sha": get_git_sha(Path.cwd()) if Path.cwd().exists() else "0" * 40,
        }
        if lock_file_hash:
            toolchain_meta["lock_file_toolchain_hash"] = lock_file_hash
    else:
        # Capture provenance dynamically (CP-5.1/5.3: no 'unknown' values for docker builds)
        provenance = capture_toolchain_provenance(
            mode=kicad_mode,
            kicad_version=spec.toolchain.kicad.version,
            docker_image=spec.toolchain.kicad.docker_image,
            workdir=out_root,
            lock_file=lock_file,
        )
        toolchain_meta = provenance.to_metadata()
        # Ensure docker image_ref is set for compatibility
        if kicad_mode == "docker" and "docker" not in toolchain_meta:
            toolchain_meta["docker"] = {"image_ref": provenance.docker_image_ref}

    _ensure_docker_image_meta(toolchain_meta, spec.toolchain.kicad.docker_image)
    toolchain_hash_value = toolchain_hash(toolchain_meta)

    # Check cache
    if manifest_path.exists():
        manifest = load_manifest(manifest_path)
        if manifest.get("design_hash") == design_hash_value and manifest.get("toolchain_hash") == toolchain_hash_value:
            return BuildResult(
                output_dir=output_dir,
                design_hash=design_hash_value,
                coupon_id=coupon_id,
                manifest_path=manifest_path,
                cache_hit=True,
                toolchain_hash=toolchain_hash_value,
            )

    # Write validation outputs
    _write_engine_validation_outputs(engine_result, output_dir)

    # Generate KiCad project
    project = generate_kicad(resolved, spec, output_dir, backend=backend)

    # Run DRC
    report = run_drc(
        project.board_path,
        spec.toolchain.kicad,
        mode=kicad_mode,
        runner=runner,
        lock_file=lock_file,
    )
    if spec.constraints.drc.must_pass and report.returncode != 0:
        raise RuntimeError(f"KiCad DRC failed with returncode {report.returncode}")

    # Export fabrication files
    export_hashes_raw = export_fab(
        project.board_path,
        output_dir / "fab",
        spec.toolchain.kicad,
        mode=kicad_mode,
        runner=runner,
        lock_file=lock_file,
    )
    # Add fab/ prefix to export paths so manifest paths match actual file locations
    export_hashes = {f"fab/{path}": h for path, h in export_hashes_raw.items()}

    # Build and write manifest
    # Note: We need to convert the TieredConstraintProof to the legacy ConstraintProof format
    # for the manifest builder. This is a temporary bridge until the manifest is updated.
    from .constraints.core import ConstraintProof
    from .constraints.core import ConstraintResult as LegacyConstraintResult

    # Convert tiered proof to legacy proof format
    legacy_constraints = tuple(
        LegacyConstraintResult(
            constraint_id=c.constraint_id,
            description=c.description,
            tier=c.tier if c.tier in ("T0", "T1", "T2", "T3", "T4") else "T0",  # type: ignore[arg-type]
            value=c.value,
            limit=c.limit,
            margin=c.margin,
            passed=c.passed,
        )
        for c in engine_result.proof.constraints
    )
    empty_tier: tuple[TieredConstraintResult, ...] = ()
    engine_tiers: tuple[Literal["T0", "T1", "T2", "T3"], ...] = ("T0", "T1", "T2", "T3")
    legacy_tiers: dict[str, tuple[LegacyConstraintResult, ...]] = {
        tier: tuple(
            LegacyConstraintResult(
                constraint_id=c.constraint_id,
                description=c.description,
                tier=c.tier if c.tier in ("T0", "T1", "T2", "T3", "T4") else "T0",  # type: ignore[arg-type]
                value=c.value,
                limit=c.limit,
                margin=c.margin,
                passed=c.passed,
            )
            for c in engine_result.proof.tiers.get(tier, empty_tier)
        )
        for tier in engine_tiers
    }
    legacy_tiers["T4"] = ()
    legacy_proof = ConstraintProof(
        constraints=legacy_constraints,
        tiers=legacy_tiers,  # type: ignore[arg-type]
        passed=engine_result.proof.passed,
    )

    manifest = build_manifest(
        spec=spec,
        resolved=resolved,
        proof=legacy_proof,
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
    """Write legacy validation outputs (resolved_design.json, constraint_proof.json)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    resolved_payload = canonical_json_dumps(evaluation.resolved.model_dump(mode="json"))
    (out_dir / "resolved_design.json").write_text(resolved_payload, encoding="utf-8")
    proof_payload = canonical_json_dumps(constraint_proof_payload(evaluation.proof))
    (out_dir / "constraint_proof.json").write_text(proof_payload, encoding="utf-8")


def _write_engine_validation_outputs(result: ConstraintEngineResult, out_dir: Path) -> None:
    """Write validation outputs from ConstraintEngine (CP-3.5).

    Outputs:
    - resolved_design.json: Resolved design parameters
    - constraint_proof.json: Per-constraint evaluations with signed margins
    - repair_map.json: Repair details if REPAIR mode was used (optional)
    - repaired_spec.json: Canonical repaired spec if REPAIR mode changed parameters

    Args:
        result: The ConstraintEngineResult from validation
        out_dir: Directory to write outputs
    """
    from .constraints.repair import write_repair_map

    out_dir.mkdir(parents=True, exist_ok=True)

    # Write resolved design
    resolved_payload = canonical_json_dumps(result.resolved.model_dump(mode="json"))
    (out_dir / "resolved_design.json").write_text(resolved_payload, encoding="utf-8")

    # Write constraint proof
    proof_doc = result.to_proof_document()
    proof_path = out_dir / "constraint_proof.json"
    proof_doc.write_to_file(proof_path)

    # Write repair map if repairs were made
    if result.was_repaired and result.repair_result is not None:
        repair_map_path = out_dir / "repair_map.json"
        write_repair_map(repair_map_path, result.repair_result)
        repaired_spec_path = out_dir / "repaired_spec.json"
        repaired_payload = canonical_json_dumps(result.spec.model_dump(mode="json"))
        repaired_spec_path.write_text(repaired_payload, encoding="utf-8")
