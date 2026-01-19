from __future__ import annotations

import argparse
import sys
from pathlib import Path

from formula_foundry.substrate import canonical_json_dumps

from .api import build_coupon, export_fab, generate_kicad, load_spec, run_drc, validate_spec
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

    parser.error(f"Unknown command: {args.command}")
    return 2


def _toolchain_from_image(image: str) -> KicadToolchain:
    resolved = image or "kicad/kicad:9.0.7@sha256:unknown"
    return KicadToolchain(version="9.0.7", docker_image=resolved)


if __name__ == "__main__":
    raise SystemExit(main())
