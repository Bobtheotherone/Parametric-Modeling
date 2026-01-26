"""CLI for coupon generation (CP-4.3, REQ-M1-018).

This module provides the command-line interface for coupon generation:
- validate: Validate spec using ConstraintEngine (Tier 0-3)
- generate: Generate KiCad project from spec
- drc: Run KiCad DRC on a board file
- export: Export gerbers and drill files
- build: Full build pipeline (validate/repair -> resolve -> generate -> DRC -> export -> manifest)
- batch-filter: Filter batch of normalized design vectors using GPU prefilter
- build-batch: Build multiple coupons from spec template and u vectors
- lint-spec-coverage: Check spec coverage (REQ-M1-018)
- explain: Human-readable resolved design + tightest-constraint summary (REQ-M1-018)

CP-3.5: Pipeline Integration
- The validate command now uses ConstraintEngine by default for full tiered validation
- Build command supports --use-engine flag to use ConstraintEngine

CP-4.3: GPU Pipeline Integration
- build-batch integrates GPU Tier0-2 filter on candidates by default
- Falls back to NumPy if CuPy is unavailable
- Records CuPy/CUDA versions in manifest when GPU is used

REQ-M1-018: CLI Commands
- lint-spec-coverage: Non-zero exit on coverage failures (unused provided or unconsumed expected paths)
- explain: Human-readable resolved + tightest-constraint summary using canonical pipeline outputs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np

from formula_foundry.substrate import canonical_json_dumps

from .api import (
    build_coupon,
    build_coupon_with_engine,
    export_fab,
    generate_kicad,
    load_spec,
    run_drc,
    validate_spec,
    validate_spec_with_engine,
)
from .constraints.gpu_filter import batch_filter as gpu_batch_filter
from .constraints.gpu_filter import is_gpu_available
from .constraints.tiers import ConstraintViolationError
from .fab_profiles import get_fab_limits, list_available_profiles, load_fab_profile
from .param_mapping import u_to_spec_f1
from .spec import CouponSpec, KicadToolchain


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="coupongen", description="Parametric coupon generator CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="Validate spec and emit resolved design + constraints")
    validate.add_argument("spec", type=Path)
    validate.add_argument("--out", type=Path, default=Path("."))
    validate.add_argument(
        "--use-engine",
        action="store_true",
        default=True,
        help="Use ConstraintEngine for full tiered validation (CP-3.5, default: True)",
    )
    validate.add_argument(
        "--legacy",
        action="store_true",
        default=False,
        help="Use legacy constraint system instead of ConstraintEngine",
    )
    validate.add_argument(
        "--constraint-mode",
        choices=["REJECT", "REPAIR"],
        default=None,
        help="Override constraint mode (default: use spec.constraints.mode)",
    )

    generate = subparsers.add_parser("generate", help="Generate KiCad project from spec")
    generate.add_argument("spec", type=Path)
    generate.add_argument("--out", type=Path, required=True)

    drc = subparsers.add_parser("drc", help="Run KiCad DRC on a board file")
    drc.add_argument("board", type=Path)
    drc.add_argument("--mode", choices=["local", "docker"], default="local")
    drc.add_argument("--report", type=Path)
    drc.add_argument("--toolchain-image", type=str, default="")

    export = subparsers.add_parser("export", help="Export gerbers + drill for a board file")
    export.add_argument("board", type=Path)
    export.add_argument("--out", type=Path, required=True)
    export.add_argument("--mode", choices=["local", "docker"], default="local")
    export.add_argument("--toolchain-image", type=str, default="")

    build = subparsers.add_parser("build", help="Generate + DRC + export")
    build.add_argument("spec", type=Path)
    build.add_argument("--out", type=Path, required=True)
    build.add_argument("--mode", choices=["local", "docker"], default="local")
    build.add_argument("--toolchain-image", type=str, default="")
    build.add_argument(
        "--use-engine",
        action="store_true",
        default=True,
        help="Use ConstraintEngine for full tiered validation (CP-3.5, default: True)",
    )
    build.add_argument(
        "--legacy",
        action="store_true",
        default=False,
        help="Deprecated alias for the canonical build pipeline (kept for compatibility)",
    )
    build.add_argument(
        "--constraint-mode",
        choices=["REJECT", "REPAIR"],
        default=None,
        help="Override constraint mode (default: use spec.constraints.mode)",
    )

    # CP-4.2: Batch filter command - filter normalized design vectors using GPU filter API
    batch_filter = subparsers.add_parser(
        "batch-filter",
        help="Filter batch of normalized design vectors using GPU prefilter (CP-4.1 API)",
    )
    batch_filter.add_argument(
        "u_npy",
        type=Path,
        help="Input file with normalized vectors: .npy (N, d) or .jsonl (one JSON array per line)",
    )
    batch_filter.add_argument("--out", type=Path, required=True, help="Output directory for results")
    batch_filter.add_argument(
        "--repair",
        action="store_true",
        default=False,
        help="Enable REPAIR mode (default: REJECT mode)",
    )
    batch_filter.add_argument(
        "--profile",
        type=str,
        default="generic",
        help=f"Fab profile ID for constraints (available: {', '.join(list_available_profiles())})",
    )
    batch_filter.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for deterministic repair (default: 0)",
    )
    batch_filter.add_argument(
        "--no-gpu",
        action="store_true",
        default=False,
        help="Disable GPU acceleration (use NumPy backend)",
    )

    # CP-4.3: Build batch command - build multiple coupons from spec template and u vectors
    # with integrated GPU Tier0-2 filtering
    build_batch = subparsers.add_parser(
        "build-batch",
        help="Build multiple coupons from spec template and filtered u vectors (CP-4.3)",
    )
    build_batch.add_argument("spec_template", type=Path, help="Spec template YAML file")
    build_batch.add_argument("--u", type=Path, required=True, help="Input .npy file with u vectors (N, d)")
    build_batch.add_argument("--out", type=Path, required=True, help="Output directory root for builds")
    build_batch.add_argument("--mode", choices=["local", "docker"], default="local")
    build_batch.add_argument("--toolchain-image", type=str, default="")
    build_batch.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of coupons to build (default: all)",
    )
    build_batch.add_argument(
        "--skip-filter",
        action="store_true",
        default=False,
        help="Skip GPU feasibility filtering (assume all vectors are valid)",
    )
    build_batch.add_argument(
        "--no-gpu",
        action="store_true",
        default=False,
        help="Disable GPU acceleration for filtering (use NumPy backend)",
    )
    build_batch.add_argument(
        "--profile",
        type=str,
        default="generic",
        help=f"Fab profile ID for GPU filter constraints (available: {', '.join(list_available_profiles())})",
    )
    build_batch.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for deterministic repair in GPU filter (default: 0)",
    )
    build_batch.add_argument(
        "--constraint-mode",
        choices=["REJECT", "REPAIR"],
        default="REPAIR",
        help="GPU filter constraint mode (default: REPAIR)",
    )

    # REQ-M1-018: lint-spec-coverage command
    lint_spec = subparsers.add_parser(
        "lint-spec-coverage",
        help="Check spec coverage: non-zero exit on unused provided or unconsumed expected paths",
    )
    lint_spec.add_argument("spec", type=Path, help="Spec file to lint (YAML or JSON)")
    lint_spec.add_argument(
        "--strict",
        action="store_true",
        default=True,
        help="Strict mode: fail on any unused provided or unconsumed expected paths (default: True)",
    )
    lint_spec.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output in JSON format instead of human-readable",
    )

    # REQ-M1-018: explain command
    explain = subparsers.add_parser(
        "explain",
        help="Human-readable resolved design + tightest-constraint summary",
    )
    explain.add_argument("spec", type=Path, help="Spec file to explain (YAML or JSON)")
    explain.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output file for explain report (default: stdout)",
    )
    explain.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output in JSON format instead of human-readable",
    )
    explain.add_argument(
        "--constraint-mode",
        choices=["REJECT", "REPAIR"],
        default=None,
        help="Override constraint mode (default: use spec.constraints.mode)",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "validate":
        spec = load_spec(args.spec)
        # CP-3.5: Use ConstraintEngine by default unless --legacy is specified
        if args.legacy:
            validate_spec(spec, out_dir=args.out)
            payload = canonical_json_dumps({"status": "valid", "engine": False})
        else:
            try:
                result = validate_spec_with_engine(
                    spec,
                    out_dir=args.out,
                    mode=args.constraint_mode,  # type: ignore[arg-type]
                )
                payload = canonical_json_dumps(
                    {
                        "status": "valid",
                        "engine": True,
                        "was_repaired": result.was_repaired,
                        "constraints_passed": result.proof.passed,
                        "total_constraints": len(result.proof.constraints),
                    }
                )
            except ConstraintViolationError as e:
                error_payload = canonical_json_dumps(
                    {
                        "status": "invalid",
                        "engine": True,
                        "violation_tier": e.tier,
                        "constraint_ids": e.constraint_ids,
                    }
                )
                sys.stderr.write(error_payload + "\n")
                return 1
        sys.stdout.write(payload + "\n")
        return 0

    if args.command == "generate":
        spec = load_spec(args.spec)
        # Use legacy validate_spec for generate since we need the evaluation result
        evaluation = validate_spec(spec, out_dir=args.out)
        generate_kicad(evaluation.resolved, evaluation.spec, args.out)
        return 0

    if args.command == "drc":
        toolchain = _toolchain_from_image(args.toolchain_image)
        report = run_drc(
            args.board,
            toolchain,
            mode=args.mode,
            report_path=args.report,
        )
        payload = canonical_json_dumps({"report_path": str(report.report_path), "returncode": report.returncode})
        sys.stdout.write(payload + "\n")
        return 0 if report.returncode == 0 else 2

    if args.command == "export":
        toolchain = _toolchain_from_image(args.toolchain_image)
        hashes = export_fab(args.board, args.out, toolchain, mode=args.mode)
        payload = canonical_json_dumps({"outputs": hashes})
        sys.stdout.write(payload + "\n")
        return 0

    if args.command == "build":
        spec = load_spec(args.spec)
        if args.toolchain_image:
            spec_payload = spec.model_dump(mode="json")
            spec_payload["toolchain"]["kicad"]["docker_image"] = args.toolchain_image
            spec = CouponSpec.model_validate(spec_payload)

        # CP-3.5: Canonical build pipeline uses ConstraintEngine; --legacy is a compatibility alias.
        if args.legacy:
            result = build_coupon(
                spec,
                out_root=args.out,
                mode=args.mode,
                constraint_mode=args.constraint_mode,  # type: ignore[arg-type]
            )
        else:
            try:
                result = build_coupon_with_engine(
                    spec,
                    out_root=args.out,
                    kicad_mode=args.mode,
                    constraint_mode=args.constraint_mode,  # type: ignore[arg-type]
                )
            except ConstraintViolationError as e:
                error_payload = canonical_json_dumps(
                    {
                        "status": "constraint_violation",
                        "engine": True,
                        "violation_tier": e.tier,
                        "constraint_ids": e.constraint_ids,
                    }
                )
                sys.stderr.write(error_payload + "\n")
                return 1

        build_payload = canonical_json_dumps(
            {
                "output_dir": str(result.output_dir),
                "design_hash": result.design_hash,
                "coupon_id": result.coupon_id,
                "cache_hit": result.cache_hit,
            }
        )
        sys.stdout.write(build_payload + "\n")
        return 0

    if args.command == "batch-filter":
        return _run_batch_filter(args)

    if args.command == "build-batch":
        return _run_build_batch(args)

    if args.command == "lint-spec-coverage":
        return _run_lint_spec_coverage(args)

    if args.command == "explain":
        return _run_explain(args)

    parser.error(f"Unknown command: {args.command}")
    return 2


def _toolchain_from_image(image: str) -> KicadToolchain:
    resolved = image or "kicad/kicad:9.0.7@sha256:unknown"
    return KicadToolchain(version="9.0.7", docker_image=resolved)


def _load_u_vectors(input_path: Path) -> np.ndarray | None:
    """Load normalized design vectors from .npy or .jsonl file.

    Args:
        input_path: Path to input file (.npy or .jsonl)

    Returns:
        2D numpy array of shape (N, d) or None on error
    """
    suffix = input_path.suffix.lower()

    if suffix == ".npy":
        return np.load(input_path)
    elif suffix == ".jsonl":
        import json

        vectors = []
        with open(input_path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    vec = json.loads(line)
                    if not isinstance(vec, list):
                        raise ValueError(f"Line {line_num}: expected JSON array, got {type(vec).__name__}")
                    vectors.append(vec)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Line {line_num}: invalid JSON: {e}") from e
        if not vectors:
            raise ValueError("JSONL file contains no valid vectors")
        return np.array(vectors, dtype=np.float64)
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Use .npy or .jsonl")


def _run_batch_filter(args: argparse.Namespace) -> int:
    """Run batch-filter command: filter normalized design vectors using GPU prefilter.

    Per CP-4.2, this command wires to the GPU filter API (CP-4.1) to prefilter
    candidate design vectors before expensive KiCad operations.

    Input: u.npy or u.jsonl with normalized vectors (N, d) where d is the parameter dimension (19 for F1)
    Output: Directory with:
        - mask.npy: Boolean feasibility mask (N,)
        - u_repaired.npy: Repaired normalized vectors (N, d)
        - metadata.json: Filtering statistics and parameters
    """
    # Load input u vectors
    input_path: Path = args.u_npy
    if not input_path.exists():
        sys.stderr.write(f"Error: Input file not found: {input_path}\n")
        return 1

    try:
        u_batch = _load_u_vectors(input_path)
        if u_batch is None:
            sys.stderr.write("Error: Failed to load input file\n")
            return 1
    except ValueError as e:
        sys.stderr.write(f"Error: {e}\n")
        return 1
    except Exception as e:
        sys.stderr.write(f"Error: Failed to load input file: {e}\n")
        return 1

    if u_batch.ndim != 2:
        sys.stderr.write(f"Error: Expected 2D array (N, d), got shape {u_batch.shape}\n")
        return 1

    # Load fab profile
    try:
        profile = load_fab_profile(args.profile)
        fab_limits = get_fab_limits(profile)
    except FileNotFoundError:
        available = list_available_profiles()
        sys.stderr.write(f"Error: Unknown fab profile '{args.profile}'. Available: {', '.join(available)}\n")
        return 1

    # Determine mode
    mode: Literal["REJECT", "REPAIR"] = "REPAIR" if args.repair else "REJECT"
    use_gpu = not args.no_gpu

    # Run batch filter
    result = gpu_batch_filter(
        u_batch,
        profiles=fab_limits,
        mode=mode,
        seed=args.seed,
        use_gpu=use_gpu,
    )

    # Create output directory
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save outputs
    np.save(out_dir / "mask.npy", result.mask)
    np.save(out_dir / "u_repaired.npy", result.u_repaired)

    # Build metadata
    metadata = {
        "input_file": str(input_path),
        "n_candidates": result.n_candidates,
        "n_feasible": result.n_feasible,
        "feasibility_rate": result.feasibility_rate,
        "mode": mode,
        "seed": args.seed,
        "profile": args.profile,
        "use_gpu": use_gpu,
        "tier_violations_summary": {tier: int(counts.sum()) for tier, counts in result.tier_violations.items()},
    }
    metadata_path = out_dir / "metadata.json"
    metadata_path.write_text(canonical_json_dumps(metadata), encoding="utf-8")

    # Write summary to stdout
    summary_payload = canonical_json_dumps(
        {
            "output_dir": str(out_dir),
            "n_candidates": result.n_candidates,
            "n_feasible": result.n_feasible,
            "feasibility_rate": result.feasibility_rate,
            "mode": mode,
        }
    )
    sys.stdout.write(summary_payload + "\n")

    return 0


def _run_build_batch(args: argparse.Namespace) -> int:
    """Run build-batch command: build multiple coupons from spec template and u vectors.

    Per CP-4.3, this command integrates the GPU batch filter with the build pipeline
    to efficiently process large candidate batches. The pipeline is:

    1. GPU Tier0-2 filter on candidates (unless --skip-filter)
    2. For survivors: map u -> spec params (resolve)
    3. Tier3 CPU constraint check via ConstraintEngine
    4. Generate KiCad board
    5. Run DRC
    6. Export fabrication files
    7. Build manifest with CuPy/CUDA versions when GPU used

    GPU filter uses CuPy by default when available, falls back to NumPy otherwise.
    """
    spec_template_path: Path = args.spec_template
    u_npy_path: Path = args.u

    if not spec_template_path.exists():
        sys.stderr.write(f"Error: Spec template not found: {spec_template_path}\n")
        return 1

    if not u_npy_path.exists():
        sys.stderr.write(f"Error: Input u vectors not found: {u_npy_path}\n")
        return 1

    # Load spec template
    try:
        spec_template = load_spec(spec_template_path)
    except Exception as e:
        sys.stderr.write(f"Error: Failed to load spec template: {e}\n")
        return 1

    # Load u vectors
    try:
        u_batch = np.load(u_npy_path)
    except Exception as e:
        sys.stderr.write(f"Error: Failed to load u vectors: {e}\n")
        return 1

    if u_batch.ndim != 2:
        sys.stderr.write(f"Error: Expected 2D array (N, d), got shape {u_batch.shape}\n")
        return 1

    n_total = len(u_batch)
    use_gpu = not args.no_gpu
    gpu_used = False
    cupy_version = None
    cuda_version = None

    # Track GPU availability for manifest
    if use_gpu and is_gpu_available():
        gpu_used = True
        try:
            import cupy as cp

            cupy_version = cp.__version__
            cuda_version = str(cp.cuda.runtime.runtimeGetVersion())
        except Exception:
            pass

    # Create output directory
    out_root: Path = args.out
    out_root.mkdir(parents=True, exist_ok=True)

    # Step 1: GPU Tier0-2 filter (unless --skip-filter)
    filter_result = None
    u_feasible = u_batch
    feasible_indices = np.arange(n_total)

    if not args.skip_filter:
        # Load fab profile for GPU filter
        try:
            profile = load_fab_profile(args.profile)
            fab_limits = get_fab_limits(profile)
        except FileNotFoundError:
            available = list_available_profiles()
            sys.stderr.write(f"Error: Unknown fab profile '{args.profile}'. Available: {', '.join(available)}\n")
            return 1

        # Determine filter mode
        filter_mode: Literal["REJECT", "REPAIR"] = args.constraint_mode

        # Run GPU filter
        filter_result = gpu_batch_filter(
            u_batch,
            profiles=fab_limits,
            mode=filter_mode,
            seed=args.seed,
            use_gpu=use_gpu,
        )

        # Get feasible vectors (either original or repaired based on mode)
        if filter_mode == "REPAIR":
            u_feasible = filter_result.u_repaired[filter_result.mask]
        else:
            u_feasible = u_batch[filter_result.mask]

        feasible_indices = np.where(filter_result.mask)[0]

        # Write filter metadata
        filter_meta = {
            "n_candidates": filter_result.n_candidates,
            "n_feasible": filter_result.n_feasible,
            "feasibility_rate": filter_result.feasibility_rate,
            "mode": filter_mode,
            "seed": args.seed,
            "profile": args.profile,
            "use_gpu": use_gpu,
            "gpu_used": gpu_used,
            "tier_violations_summary": {tier: int(counts.sum()) for tier, counts in filter_result.tier_violations.items()},
        }
        if gpu_used:
            filter_meta["cupy_version"] = cupy_version
            filter_meta["cuda_version"] = cuda_version

        filter_meta_path = out_root / "filter_metadata.json"
        filter_meta_path.write_text(canonical_json_dumps(filter_meta), encoding="utf-8")

        # Save filter outputs
        np.save(out_root / "mask.npy", filter_result.mask)
        np.save(out_root / "u_repaired.npy", filter_result.u_repaired)

    # Apply limit after filtering
    limit = args.limit if args.limit is not None else len(u_feasible)
    limit = min(limit, len(u_feasible))

    # Track build results
    builds_completed = 0
    builds_failed = 0
    build_results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    # Step 2-6: For each feasible u vector, build the coupon
    for i in range(limit):
        u_vec = u_feasible[i]
        original_idx = int(feasible_indices[i]) if i < len(feasible_indices) else i

        try:
            # Map u vector to CouponSpec
            spec = u_to_spec_f1(u_vec, spec_template)

            # Override toolchain image if specified
            if args.toolchain_image:
                spec_payload = spec.model_dump(mode="json")
                spec_payload["toolchain"]["kicad"]["docker_image"] = args.toolchain_image
                spec = CouponSpec.model_validate(spec_payload)

            # Build coupon using ConstraintEngine (includes Tier3 CPU check)
            try:
                result = build_coupon_with_engine(
                    spec,
                    out_root=out_root,
                    kicad_mode=args.mode,
                )

                build_results.append(
                    {
                        "index": original_idx,
                        "design_hash": result.design_hash,
                        "coupon_id": result.coupon_id,
                        "output_dir": str(result.output_dir),
                        "cache_hit": result.cache_hit,
                        "status": "success",
                    }
                )
                builds_completed += 1

                # Add GPU metadata to manifest if GPU was used
                if gpu_used and result.manifest_path.exists():
                    _add_gpu_metadata_to_manifest(
                        result.manifest_path,
                        cupy_version=cupy_version,
                        cuda_version=cuda_version,
                    )

            except ConstraintViolationError as e:
                builds_failed += 1
                errors.append(
                    {
                        "index": original_idx,
                        "error": "constraint_violation",
                        "tier": e.tier,
                        "constraint_ids": e.constraint_ids,
                    }
                )
            except RuntimeError as e:
                builds_failed += 1
                errors.append(
                    {
                        "index": original_idx,
                        "error": "build_error",
                        "message": str(e),
                    }
                )

        except Exception as e:
            builds_failed += 1
            errors.append(
                {
                    "index": original_idx,
                    "error": "spec_mapping_error",
                    "message": str(e),
                }
            )

    # Write batch summary
    batch_summary = {
        "output_dir": str(out_root),
        "n_input_vectors": n_total,
        "n_feasible_after_filter": len(u_feasible),
        "limit_applied": limit,
        "skip_filter": args.skip_filter,
        "builds_completed": builds_completed,
        "builds_failed": builds_failed,
        "gpu_filter_used": not args.skip_filter,
        "gpu_backend_used": gpu_used,
        "status": "completed",
    }

    if gpu_used:
        batch_summary["cupy_version"] = cupy_version
        batch_summary["cuda_version"] = cuda_version

    if filter_result is not None:
        batch_summary["filter_feasibility_rate"] = filter_result.feasibility_rate

    batch_summary_path = out_root / "batch_summary.json"
    batch_summary_path.write_text(canonical_json_dumps(batch_summary), encoding="utf-8")

    # Write detailed build results
    results_payload = {
        "builds": build_results,
        "errors": errors,
    }
    results_path = out_root / "build_results.json"
    results_path.write_text(canonical_json_dumps(results_payload), encoding="utf-8")

    # Write summary to stdout
    summary_payload = canonical_json_dumps(batch_summary)
    sys.stdout.write(summary_payload + "\n")

    return 0 if builds_failed == 0 else 1


def _add_gpu_metadata_to_manifest(
    manifest_path: Path,
    cupy_version: str | None,
    cuda_version: str | None,
) -> None:
    """Add GPU metadata to an existing manifest file.

    Per CP-4.3, when GPU is used for batch filtering, record CuPy/CUDA versions
    in the manifest under toolchain.gpu section.

    Args:
        manifest_path: Path to the manifest.json file
        cupy_version: CuPy version string (or None if unavailable)
        cuda_version: CUDA runtime version string (or None if unavailable)
    """
    import json

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        # Add GPU metadata to toolchain
        if "toolchain" not in manifest:
            manifest["toolchain"] = {}

        manifest["toolchain"]["gpu"] = {
            "used": True,
            "cupy_version": cupy_version,
            "cuda_runtime_version": cuda_version,
        }

        # Re-write manifest
        manifest_path.write_text(canonical_json_dumps(manifest) + "\n", encoding="utf-8")
    except Exception:
        # Don't fail the build if we can't update the manifest
        pass


def _run_lint_spec_coverage(args: argparse.Namespace) -> int:
    """Run lint-spec-coverage command: check spec coverage for unused/unconsumed paths.

    Per REQ-M1-018, this command provides a lint check for spec coverage.
    Delegates to the command module implementation.

    Args:
        args: Parsed command-line arguments

    Returns:
        0 if coverage is complete, 1 if coverage failures
    """
    from formula_foundry.cli import run_lint_spec_coverage

    return run_lint_spec_coverage(args)


def _run_explain(args: argparse.Namespace) -> int:
    """Run explain command: human-readable resolved design + tightest-constraint summary.

    Per REQ-M1-018, this command provides a human-readable summary.
    Delegates to the command module implementation.

    Args:
        args: Parsed command-line arguments

    Returns:
        0 on success, 1 on validation errors
    """
    from formula_foundry.cli import run_explain

    return run_explain(args)


if __name__ == "__main__":
    raise SystemExit(main())
