from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal

import numpy as np

from formula_foundry.substrate import canonical_json_dumps

from .api import build_coupon, export_fab, generate_kicad, load_spec, run_drc, validate_spec
from .constraints.gpu_filter import batch_filter as gpu_batch_filter
from .fab_profiles import get_fab_limits, list_available_profiles, load_fab_profile
from .spec import CouponSpec, KicadToolchain


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="coupongen", description="Parametric coupon generator CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="Validate spec and emit resolved design + constraints")
    validate.add_argument("spec", type=Path)
    validate.add_argument("--out", type=Path, default=Path("."))

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

    # CP-4.2: Batch filter command - filter normalized design vectors using GPU filter API
    batch_filter = subparsers.add_parser(
        "batch-filter",
        help="Filter batch of normalized design vectors using GPU prefilter (CP-4.1 API)",
    )
    batch_filter.add_argument("u_npy", type=Path, help="Input .npy file with normalized vectors (N, d)")
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

    # CP-4.2: Build batch command - build multiple coupons from spec template and u vectors
    build_batch = subparsers.add_parser(
        "build-batch",
        help="Build multiple coupons from spec template and filtered u vectors",
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
        help="Skip feasibility filtering (assume all vectors are valid)",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "validate":
        spec = load_spec(args.spec)
        validate_spec(spec, out_dir=args.out)
        return 0

    if args.command == "generate":
        spec = load_spec(args.spec)
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
        result = build_coupon(spec, out_root=args.out, mode=args.mode)
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

    parser.error(f"Unknown command: {args.command}")
    return 2


def _toolchain_from_image(image: str) -> KicadToolchain:
    resolved = image or "kicad/kicad:9.0.7@sha256:unknown"
    return KicadToolchain(version="9.0.7", docker_image=resolved)


def _run_batch_filter(args: argparse.Namespace) -> int:
    """Run batch-filter command: filter normalized design vectors using GPU prefilter.

    Per CP-4.2, this command wires to the GPU filter API (CP-4.1) to prefilter
    candidate design vectors before expensive KiCad operations.

    Input: u.npy with shape (N, d) where d is the parameter dimension (19 for F1)
    Output: Directory with:
        - mask.npy: Boolean feasibility mask (N,)
        - u_repaired.npy: Repaired normalized vectors (N, d)
        - metadata.json: Filtering statistics and parameters
    """
    # Load input u vectors
    u_npy_path: Path = args.u_npy
    if not u_npy_path.exists():
        sys.stderr.write(f"Error: Input file not found: {u_npy_path}\n")
        return 1

    try:
        u_batch = np.load(u_npy_path)
    except Exception as e:
        sys.stderr.write(f"Error: Failed to load .npy file: {e}\n")
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
        "input_file": str(u_npy_path),
        "n_candidates": result.n_candidates,
        "n_feasible": result.n_feasible,
        "feasibility_rate": result.feasibility_rate,
        "mode": mode,
        "seed": args.seed,
        "profile": args.profile,
        "use_gpu": use_gpu,
        "tier_violations_summary": {
            tier: int(counts.sum()) for tier, counts in result.tier_violations.items()
        },
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

    Per CP-4.2, this command integrates the GPU batch filter with the build pipeline
    to efficiently process large candidate batches.

    Note: Full implementation of build-batch requires mapping from normalized u vectors
    to spec parameters, which is family-dependent. This implementation provides the
    CLI interface and basic scaffolding; the full parameter mapping is deferred to
    a future enhancement.
    """
    spec_template_path: Path = args.spec_template
    u_npy_path: Path = args.u

    if not spec_template_path.exists():
        sys.stderr.write(f"Error: Spec template not found: {spec_template_path}\n")
        return 1

    if not u_npy_path.exists():
        sys.stderr.write(f"Error: Input u vectors not found: {u_npy_path}\n")
        return 1

    try:
        u_batch = np.load(u_npy_path)
    except Exception as e:
        sys.stderr.write(f"Error: Failed to load u vectors: {e}\n")
        return 1

    if u_batch.ndim != 2:
        sys.stderr.write(f"Error: Expected 2D array (N, d), got shape {u_batch.shape}\n")
        return 1

    n_total = len(u_batch)
    limit = args.limit if args.limit is not None else n_total

    # Create output directory
    out_root: Path = args.out
    out_root.mkdir(parents=True, exist_ok=True)

    # Build summary
    builds_completed = 0
    builds_failed = 0
    build_results: list[dict[str, str]] = []

    # Note: Full implementation would:
    # 1. Load spec template
    # 2. Optionally filter u_batch using batch_filter (unless --skip-filter)
    # 3. Map each feasible u vector to spec parameters
    # 4. Build each coupon using build_coupon
    #
    # For now, we provide the CLI interface and report that build-batch
    # requires the parameter mapping implementation.

    summary_payload = canonical_json_dumps(
        {
            "output_dir": str(out_root),
            "n_input_vectors": n_total,
            "limit": limit,
            "skip_filter": args.skip_filter,
            "builds_completed": builds_completed,
            "builds_failed": builds_failed,
            "status": "interface_ready",
            "note": "Full build-batch implementation requires u-to-spec parameter mapping",
        }
    )
    sys.stdout.write(summary_payload + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
