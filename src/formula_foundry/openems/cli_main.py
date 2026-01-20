from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, cast

from formula_foundry.substrate import canonical_json_dumps

from .runner import OpenEMSMode, OpenEMSRunner
from .toolchain import OpenEMSToolchain, load_openems_toolchain


def build_parser() -> argparse.ArgumentParser:
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--mode", choices=("local", "docker"), default="local")
    shared.add_argument("--docker-image", default="")
    shared.add_argument("--openems-bin", default="openEMS")
    shared.add_argument("--workdir", default=".")
    shared.add_argument("--toolchain-path", default="")

    parser = argparse.ArgumentParser(prog="openems", description="openEMS runner (local or docker)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    version = subparsers.add_parser("version", help="Capture openEMS version metadata", parents=[shared])
    version.add_argument("--json", nargs="?", const="-", default="-")

    run = subparsers.add_parser("run", help="Run openEMS with provided args", parents=[shared])
    run.add_argument("--json", nargs="?", const="-", default=None)
    run.add_argument("openems_args", nargs=argparse.REMAINDER)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    workdir = Path(args.workdir).resolve()

    toolchain = _resolve_toolchain(args.mode, args.docker_image, args.toolchain_path)
    docker_image = args.docker_image or (toolchain.docker_image if toolchain else None)

    runner = OpenEMSRunner(
        mode=cast_mode(args.mode),
        docker_image=docker_image,
        openems_bin=args.openems_bin,
    )

    if args.command == "version":
        payload = runner.version_metadata(workdir=workdir)
        if toolchain is not None:
            payload["toolchain_version"] = toolchain.version
        _emit_json(payload, args.json)
        return 0 if payload.get("returncode", 1) == 0 else 2

    if args.command == "run":
        if not args.openems_args:
            parser.error("run requires openEMS arguments after '--'")
        proc = runner.run(args.openems_args, workdir=workdir)
        if args.json:
            run_payload: dict[str, Any] = {
                "command": runner.build_command(args.openems_args, workdir=workdir),
                "returncode": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            }
            _emit_json(run_payload, args.json)
        return proc.returncode

    parser.error(f"Unknown command: {args.command}")
    return 2


def cast_mode(value: str) -> OpenEMSMode:
    if value not in ("local", "docker"):
        raise ValueError(f"Unsupported mode: {value}")
    return cast(OpenEMSMode, value)


def _resolve_toolchain(mode: str, docker_image: str, toolchain_path: str) -> OpenEMSToolchain | None:
    if mode != "docker":
        return None
    if docker_image:
        return None
    path = Path(toolchain_path).resolve() if toolchain_path else None
    return load_openems_toolchain(path)


def _emit_json(payload: dict[str, Any], target: str) -> None:
    text = canonical_json_dumps(payload)
    if target == "-":
        sys.stdout.write(f"{text}\n")
        return
    out_path = Path(target)
    out_path.write_text(f"{text}\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
