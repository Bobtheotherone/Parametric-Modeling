"""Export pipeline with caching for coupon generation.

This module implements the full build pipeline:
    resolve -> validate -> generate -> drc -> export

Satisfies:
    - REQ-M1-019: All output directories must be keyed by design_hash and coupon_id;
                  re-running build must not create divergent outputs.
    - REQ-M1-020: The build pipeline must implement caching keyed by
                  design_hash + toolchain_hash and must be deterministic
                  when cache hits occur.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from formula_foundry.substrate import canonical_json_dumps

from .constraints import ConstraintEvaluation, constraint_proof_payload, enforce_constraints


class KicadExportError(RuntimeError):
    """Raised when KiCad export commands fail.

    This exception provides detailed diagnostic information including the
    command that failed, exit code, stdout, and stderr to help diagnose
    export failures (e.g., permission issues, missing files).
    """

    def __init__(
        self,
        message: str,
        command: list[str] | None = None,
        returncode: int | None = None,
        stdout: str = "",
        stderr: str = "",
    ) -> None:
        self.command = command or []
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        details = [message]
        if command:
            details.append(f"Command: {' '.join(command)}")
        if returncode is not None:
            details.append(f"Exit code: {returncode}")
        if stdout:
            details.append(f"stdout: {stdout}")
        if stderr:
            details.append(f"stderr: {stderr}")
        super().__init__("\n".join(details))
from .families import validate_family
from .hashing import canonical_hash_export_text, coupon_id_from_design_hash
from .kicad import BackendA, KicadCliRunner
from .kicad.cli import KicadCliMode
from .layer_validation import (
    LayerValidationResult,
    validate_family_layer_requirements,
    validate_layer_set,
)
from .manifest import build_manifest, load_manifest, toolchain_hash, write_manifest
from .resolve import ResolvedDesign, design_hash
from .spec import CouponSpec, KicadToolchain
from .toolchain_capture import ToolchainProvenance, capture_toolchain_provenance


class KicadRunnerProtocol(Protocol):
    """Protocol for KiCad CLI runner implementations."""

    def run_drc(self, board_path: Path, report_path: Path) -> Any:  # subprocess.CompletedProcess[str]
        """Run DRC check on the board."""
        ...

    def export_gerbers(self, board_path: Path, out_dir: Path) -> Any:  # subprocess.CompletedProcess[str]
        """Export Gerber files from the board."""
        ...

    def export_drill(self, board_path: Path, out_dir: Path) -> Any:  # subprocess.CompletedProcess[str]
        """Export drill files from the board."""
        ...


@dataclass(frozen=True)
class CacheKey:
    """Cache key combining design_hash and toolchain_hash.

    Satisfies REQ-M1-020: Cache keyed by design_hash + toolchain_hash.
    """

    design_hash: str
    toolchain_hash: str

    @property
    def combined_hash(self) -> str:
        """Compute combined hash for cache lookup."""
        combined = f"{self.design_hash}:{self.toolchain_hash}"
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    def matches(self, manifest: dict[str, Any]) -> bool:
        """Check if this cache key matches a manifest's hashes."""
        return manifest.get("design_hash") == self.design_hash and manifest.get("toolchain_hash") == self.toolchain_hash


@dataclass(frozen=True)
class ExportResult:
    """Result of the export pipeline.

    Attributes:
        output_dir: Path to output directory (keyed by coupon_id-design_hash).
        design_hash: SHA256 hash of the resolved design.
        coupon_id: Human-readable identifier derived from design_hash.
        manifest_path: Path to the manifest.json file.
        cache_hit: True if the build was served from cache.
        toolchain_hash: SHA256 hash of the toolchain configuration.
        cache_key: The cache key used for this build.
    """

    output_dir: Path
    design_hash: str
    coupon_id: str
    manifest_path: Path
    cache_hit: bool
    toolchain_hash: str
    cache_key: CacheKey


@dataclass(frozen=True)
class PipelineStage:
    """Represents a stage in the export pipeline."""

    name: str
    completed: bool
    output_path: Path | None = None


@dataclass(frozen=True)
class PipelineProgress:
    """Tracks progress through the export pipeline stages."""

    stages: tuple[PipelineStage, ...]

    @property
    def all_completed(self) -> bool:
        """Check if all stages are completed."""
        return all(stage.completed for stage in self.stages)


class ExportPipeline:
    """Full build pipeline: resolve -> validate -> generate -> drc -> export.

    This class implements the complete export pipeline with caching support.
    It satisfies both REQ-M1-019 and REQ-M1-020.

    Per Section 13.5.3, the pipeline validates that all expected layers are
    present in the export output based on the copper layer count and family.

    Example usage:
        >>> pipeline = ExportPipeline(out_root=Path("output"))
        >>> result = pipeline.run(spec)
        >>> print(f"Output at: {result.output_dir}")
        >>> print(f"Cache hit: {result.cache_hit}")
    """

    def __init__(
        self,
        out_root: Path,
        *,
        mode: KicadCliMode = "local",
        runner: KicadRunnerProtocol | None = None,
        backend: BackendA | None = None,
        kicad_cli_version: str | None = None,
        lock_file: Path | None = None,
        validate_layers: bool = True,
    ) -> None:
        """Initialize the export pipeline.

        Args:
            out_root: Root directory for all outputs.
            mode: KiCad CLI mode ("local" or "docker").
            runner: Custom KiCad runner (for testing).
            backend: Custom KiCad backend (for testing).
            kicad_cli_version: Version of kicad-cli being used (for testing).
            lock_file: Path to toolchain lock file (for docker mode).
            validate_layers: Whether to validate exported layers (default True).
        """
        self.out_root = out_root
        self.mode = mode
        self._runner = runner
        self._backend = backend or BackendA()
        self._kicad_cli_version = kicad_cli_version
        self._lock_file = lock_file
        self._provenance: ToolchainProvenance | None = None
        self._validate_layers = validate_layers

    def _get_runner(self, toolchain: KicadToolchain) -> KicadRunnerProtocol:
        """Get or create the KiCad runner."""
        if self._runner is not None:
            return self._runner
        return KicadCliRunner(mode=self.mode, docker_image=toolchain.docker_image)

    def _build_cache_key(self, resolved: ResolvedDesign, spec: CouponSpec) -> CacheKey:
        """Build a cache key from the resolved design and toolchain.

        Satisfies REQ-M1-020: Cache keyed by design_hash + toolchain_hash.
        """
        design_hash_value = design_hash(resolved)
        toolchain_meta = self._build_toolchain_meta(spec)
        toolchain_hash_value = toolchain_hash(toolchain_meta)
        return CacheKey(design_hash=design_hash_value, toolchain_hash=toolchain_hash_value)

    def _build_toolchain_meta(self, spec: CouponSpec) -> dict[str, Any]:
        """Build toolchain metadata dictionary.

        For docker mode, this captures complete toolchain provenance (CP-5.3).
        """
        if self._kicad_cli_version is not None:
            # Pre-captured version provided (for testing)
            return {
                "kicad_version": spec.toolchain.kicad.version,
                "docker_image": spec.toolchain.kicad.docker_image,
                "mode": self.mode,
                "kicad_cli_version": self._kicad_cli_version,
            }

        # Capture provenance dynamically (CP-5.3)
        if self._provenance is None:
            self._provenance = capture_toolchain_provenance(
                mode=self.mode,
                kicad_version=spec.toolchain.kicad.version,
                docker_image=spec.toolchain.kicad.docker_image,
                workdir=self.out_root,
                lock_file=self._lock_file,
            )

        return self._provenance.to_metadata()

    def _compute_output_dir(self, design_hash_value: str) -> Path:
        """Compute the output directory path.

        Satisfies REQ-M1-019: Outputs keyed by design_hash and coupon_id.
        """
        coupon_id = coupon_id_from_design_hash(design_hash_value)
        return self.out_root / f"{coupon_id}-{design_hash_value}"

    def _check_cache(self, cache_key: CacheKey, output_dir: Path) -> bool:
        """Check if we have a valid cache hit.

        Satisfies REQ-M1-020: Deterministic cache behavior.
        """
        manifest_path = output_dir / "manifest.json"
        if not manifest_path.exists():
            return False

        manifest = load_manifest(manifest_path)
        return cache_key.matches(manifest)

    def run(self, spec: CouponSpec) -> ExportResult:
        """Run the full export pipeline.

        Stages:
            1. Resolve: Convert spec to integer-nm resolved design
            2. Validate: Check constraints (REJECT or REPAIR mode)
            3. Generate: Create KiCad board file
            4. DRC: Run KiCad Design Rule Check
            5. Export: Generate Gerbers and drill files

        Args:
            spec: The coupon specification to build.

        Returns:
            ExportResult with paths and cache status.

        Raises:
            RuntimeError: If DRC fails and must_pass is True.
            ConstraintViolation: If constraints fail in REJECT mode.
        """
        # Stage 1: Validate family
        validate_family(spec)

        # Stage 1b: Validate family layer requirements (per Section 13.5.3)
        validate_family_layer_requirements(
            copper_layers=spec.stackup.copper_layers,
            family=spec.coupon_family,
        )

        # Stage 2: Validate constraints and resolve
        evaluation = enforce_constraints(spec)
        resolved = evaluation.resolved

        # Build cache key
        cache_key = self._build_cache_key(resolved, evaluation.spec)
        output_dir = self._compute_output_dir(cache_key.design_hash)
        manifest_path = output_dir / "manifest.json"
        coupon_id = coupon_id_from_design_hash(cache_key.design_hash)

        # Check cache (REQ-M1-020)
        if self._check_cache(cache_key, output_dir):
            return ExportResult(
                output_dir=output_dir,
                design_hash=cache_key.design_hash,
                coupon_id=coupon_id,
                manifest_path=manifest_path,
                cache_hit=True,
                toolchain_hash=cache_key.toolchain_hash,
                cache_key=cache_key,
            )

        # Stage 3: Generate KiCad board
        # Pass design_hash to enable silkscreen annotations (REQ-M1-010)
        self._write_validation_outputs(evaluation, output_dir)
        board_path = self._backend.write_board(
            evaluation.spec, resolved, output_dir, design_hash=cache_key.design_hash
        )

        # Stage 4: Run DRC
        runner = self._get_runner(evaluation.spec.toolchain.kicad)
        report_path = output_dir / "drc.json"
        proc = runner.run_drc(board_path, report_path)
        if evaluation.spec.constraints.drc.must_pass and proc.returncode != 0:
            raise RuntimeError(f"KiCad DRC failed with returncode {proc.returncode}")

        # Stage 5: Export Gerbers and drill
        fab_dir = output_dir / "fab"
        export_hashes_raw = self._export_fab(board_path, fab_dir, evaluation.spec.toolchain.kicad)

        # Stage 6: Validate layer set (per Section 13.5.3)
        # Use raw paths (gerbers/...) for layer validation since it expects that format
        layer_validation_result: LayerValidationResult | None = None
        if self._validate_layers:
            layer_validation_result = validate_layer_set(
                export_paths=list(export_hashes_raw.keys()),
                copper_layers=evaluation.spec.stackup.copper_layers,
                family=evaluation.spec.coupon_family,
                gerber_dir="gerbers/",
                strict=True,
            )

        # Add fab/ prefix to export paths so manifest paths match actual file locations
        export_hashes = {f"fab/{path}": h for path, h in export_hashes_raw.items()}

        # Write manifest
        toolchain_meta = self._build_toolchain_meta(evaluation.spec)
        manifest = build_manifest(
            spec=evaluation.spec,
            resolved=resolved,
            proof=evaluation.proof,
            design_hash=cache_key.design_hash,
            coupon_id=coupon_id,
            toolchain=toolchain_meta,
            toolchain_hash_value=cache_key.toolchain_hash,
            export_hashes=export_hashes,
            drc_report_path=report_path,
            drc_returncode=proc.returncode,
            layer_validation=layer_validation_result,
        )
        write_manifest(manifest_path, manifest)

        return ExportResult(
            output_dir=output_dir,
            design_hash=cache_key.design_hash,
            coupon_id=coupon_id,
            manifest_path=manifest_path,
            cache_hit=False,
            toolchain_hash=cache_key.toolchain_hash,
            cache_key=cache_key,
        )

    def _write_validation_outputs(self, evaluation: ConstraintEvaluation, out_dir: Path) -> None:
        """Write resolved design and constraint proof to output directory."""
        out_dir.mkdir(parents=True, exist_ok=True)
        resolved_payload = canonical_json_dumps(evaluation.resolved.model_dump(mode="json"))
        (out_dir / "resolved_design.json").write_text(resolved_payload, encoding="utf-8")
        proof_payload = canonical_json_dumps(constraint_proof_payload(evaluation.proof))
        (out_dir / "constraint_proof.json").write_text(proof_payload, encoding="utf-8")

    def _export_fab(
        self,
        board_path: Path,
        out_dir: Path,
        toolchain: KicadToolchain,
    ) -> dict[str, str]:
        """Export fabrication files and compute canonical hashes.

        Raises:
            KicadExportError: If Gerber or drill export fails (non-zero exit code).
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        runner = self._get_runner(toolchain)

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
    """Compute canonical hashes for all files in the export tree.

    Returns:
        Dictionary mapping relative paths to their canonical SHA256 hashes.
    """
    hashes: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            text = path.read_text(encoding="utf-8", errors="ignore")
            hashes[path.relative_to(root).as_posix()] = canonical_hash_export_text(text)
    return hashes


def run_export_pipeline(
    spec: CouponSpec,
    *,
    out_root: Path,
    mode: KicadCliMode = "local",
    runner: KicadRunnerProtocol | None = None,
    backend: BackendA | None = None,
    kicad_cli_version: str | None = None,
    lock_file: Path | None = None,
    validate_layers: bool = True,
) -> ExportResult:
    """Convenience function to run the export pipeline.

    This is a functional wrapper around ExportPipeline for simpler usage.

    Satisfies:
        - REQ-M1-019: Outputs keyed by design_hash and coupon_id.
        - REQ-M1-020: Caching keyed by design_hash + toolchain_hash.
        - CP-5.3: Toolchain provenance always captured for docker builds.

    Args:
        spec: The coupon specification to build.
        out_root: Root directory for all outputs.
        mode: KiCad CLI mode ("local" or "docker").
        runner: Custom KiCad runner (for testing).
        backend: Custom KiCad backend (for testing).
        kicad_cli_version: Version of kicad-cli being used (for testing).
        lock_file: Path to toolchain lock file (for docker mode).
        validate_layers: Whether to validate exported layers (default True).

    Returns:
        ExportResult with paths and cache status.

    Example:
        >>> result = run_export_pipeline(spec, out_root=Path("output"))
        >>> if result.cache_hit:
        ...     print("Build served from cache")
    """
    pipeline = ExportPipeline(
        out_root=out_root,
        mode=mode,
        runner=runner,
        backend=backend,
        kicad_cli_version=kicad_cli_version,
        lock_file=lock_file,
        validate_layers=validate_layers,
    )
    return pipeline.run(spec)


def compute_cache_key(
    spec: CouponSpec,
    *,
    mode: KicadCliMode = "local",
    lock_file: Path | None = None,
    workdir: Path | None = None,
    kicad_cli_version: str | None = None,
) -> CacheKey:
    """Compute the cache key for a spec without running the pipeline.

    This is useful for cache invalidation checks or batch processing.
    For docker mode, this will capture toolchain provenance by running
    kicad-cli --version inside the container (per CP-5.3).

    Args:
        spec: The coupon specification.
        mode: KiCad CLI mode for toolchain hash.
        lock_file: Path to toolchain lock file (for docker mode).
        workdir: Working directory for running kicad-cli.
        kicad_cli_version: Optional pre-captured kicad-cli version (for testing).

    Returns:
        CacheKey with design_hash and toolchain_hash.

    Raises:
        ToolchainProvenanceError: If docker mode and provenance cannot be captured.
    """
    validate_family(spec)
    evaluation = enforce_constraints(spec)
    resolved = evaluation.resolved

    design_hash_value = design_hash(resolved)

    if kicad_cli_version is not None:
        # Pre-captured version provided (for testing)
        toolchain_meta = {
            "kicad_version": spec.toolchain.kicad.version,
            "docker_image": spec.toolchain.kicad.docker_image,
            "mode": mode,
            "kicad_cli_version": kicad_cli_version,
        }
    else:
        # Capture provenance dynamically (CP-5.3)
        provenance = capture_toolchain_provenance(
            mode=mode,
            kicad_version=spec.toolchain.kicad.version,
            docker_image=spec.toolchain.kicad.docker_image,
            workdir=workdir or Path.cwd(),
            lock_file=lock_file,
        )
        toolchain_meta = provenance.to_metadata()
    toolchain_hash_value = toolchain_hash(toolchain_meta)

    return CacheKey(design_hash=design_hash_value, toolchain_hash=toolchain_hash_value)


def is_cache_valid(
    spec: CouponSpec,
    out_root: Path,
    *,
    mode: KicadCliMode = "local",
    lock_file: Path | None = None,
    kicad_cli_version: str | None = None,
) -> bool:
    """Check if a valid cache exists for the given spec.

    For docker mode, this will capture toolchain provenance by running
    kicad-cli --version inside the container (per CP-5.3).

    Args:
        spec: The coupon specification.
        out_root: Root directory for outputs.
        mode: KiCad CLI mode for toolchain hash.
        lock_file: Path to toolchain lock file (for docker mode).
        kicad_cli_version: Optional pre-captured kicad-cli version (for testing).

    Returns:
        True if a valid cache hit would occur.

    Raises:
        ToolchainProvenanceError: If docker mode and provenance cannot be captured.
    """
    cache_key = compute_cache_key(
        spec, mode=mode, lock_file=lock_file, workdir=out_root, kicad_cli_version=kicad_cli_version
    )
    coupon_id = coupon_id_from_design_hash(cache_key.design_hash)
    output_dir = out_root / f"{coupon_id}-{cache_key.design_hash}"
    manifest_path = output_dir / "manifest.json"

    if not manifest_path.exists():
        return False

    manifest = load_manifest(manifest_path)
    return cache_key.matches(manifest)
