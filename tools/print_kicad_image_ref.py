#!/usr/bin/env python3
"""
Print the pinned KiCad Docker image reference for CI usage.

Usage:
    python tools/print_kicad_image_ref.py [--lock-file PATH] [--format FORMAT]

Output formats:
    full:   kicad/kicad:9.0.7@sha256:...  (default)
    image:  kicad/kicad:9.0.7
    digest: sha256:...
    tag:    9.0.7

Exit codes:
    0: Success
    1: Lock file not found or invalid
    2: Digest missing or placeholder (not yet pinned)
"""

import argparse
import json
import re
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Print the pinned KiCad Docker image reference"
    )
    parser.add_argument(
        "--lock-file",
        type=Path,
        default=Path(__file__).parent.parent / "toolchain" / "kicad.lock.json",
        help="Path to the lock file (default: toolchain/kicad.lock.json)"
    )
    parser.add_argument(
        "--format",
        choices=["full", "image", "digest", "tag"],
        default="full",
        help="Output format (default: full)"
    )
    parser.add_argument(
        "--allow-placeholder",
        action="store_true",
        help="Allow placeholder digest (exit 0 even if not pinned)"
    )
    args = parser.parse_args()

    lock_path: Path = args.lock_file

    if not lock_path.exists():
        print(f"Error: Lock file not found: {lock_path}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(lock_path) as f:
            lock_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in lock file: {e}", file=sys.stderr)
        sys.exit(1)

    docker_image = lock_data.get("docker_image", "")
    docker_digest = lock_data.get("docker_digest", "")
    kicad_version = lock_data.get("kicad_version", "")

    if not docker_image:
        print("Error: Docker image missing in lock file.", file=sys.stderr)
        sys.exit(1)

    image_base = docker_image
    embedded_digest = None
    if "@sha256:" in docker_image:
        image_base, embedded_digest = docker_image.split("@", 1)

    if embedded_digest and docker_digest and embedded_digest != docker_digest:
        print("Error: docker_image digest does not match docker_digest", file=sys.stderr)
        sys.exit(1)

    if not docker_digest:
        print(
            "Error: Docker digest missing in lock file. "
            "Run 'python tools/pin_kicad_image.py' to resolve the actual digest.",
            file=sys.stderr,
        )
        sys.exit(2)

    is_placeholder = "PLACEHOLDER" in docker_digest.upper()
    if is_placeholder and not args.allow_placeholder:
        print(
            "Error: Docker digest is a placeholder. "
            "Run 'python tools/pin_kicad_image.py' to resolve the actual digest.",
            file=sys.stderr,
        )
        sys.exit(2)

    if not is_placeholder and not re.match(r"^sha256:[0-9a-f]{64}$", docker_digest):
        print("Error: Docker digest must be sha256: followed by 64 hex chars.", file=sys.stderr)
        sys.exit(1)

    # Output based on format
    if args.format == "full":
        print(f"{image_base}@{docker_digest}")
    elif args.format == "image":
        print(image_base)
    elif args.format == "digest":
        print(docker_digest)
    elif args.format == "tag":
        print(kicad_version)

    return 0


if __name__ == "__main__":
    sys.exit(main())
