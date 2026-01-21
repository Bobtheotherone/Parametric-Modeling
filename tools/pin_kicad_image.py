#!/usr/bin/env python3
"""
Resolve the Docker digest for the pinned KiCad image and update the lock file.

Usage:
    python tools/pin_kicad_image.py [--lock-file PATH]

This script:
1. Queries DockerHub for the digest of the kicad/kicad:9.0.7 image
2. Updates toolchain/kicad.lock.json with the resolved digest
3. Computes and stores the toolchain_hash (SHA256 of canonical lock JSON)
"""

import argparse
import hashlib
import json
import re
import sys
import urllib.request
from pathlib import Path

_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_PLACEHOLDER_HEX = {
    "0" * 64,
    "0" * 63 + "1",
}


def is_placeholder_digest(digest: str) -> bool:
    """Return True if digest is an explicit placeholder value."""
    digest_text = str(digest)
    upper = digest_text.upper()
    if "PLACEHOLDER" in upper or "UNKNOWN" in upper:
        return True
    if digest_text.startswith("sha256:"):
        hex_part = digest_text.split("sha256:", 1)[1]
        if hex_part in _PLACEHOLDER_HEX:
            return True
    return False


def get_dockerhub_token(repository: str) -> str:
    """Get an anonymous auth token from DockerHub."""
    url = f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{repository}:pull"
    with urllib.request.urlopen(url) as resp:
        data = json.loads(resp.read().decode())
        return data["token"]


def get_image_digest(repository: str, tag: str, token: str) -> str | None:
    """Get the manifest digest for a Docker image tag.

    Returns the sha256 digest or None if not found.
    """
    url = f"https://registry-1.docker.io/v2/{repository}/manifests/{tag}"
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            # Request manifest list or single manifest
            "Accept": "application/vnd.docker.distribution.manifest.v2+json, "
                     "application/vnd.docker.distribution.manifest.list.v2+json, "
                     "application/vnd.oci.image.index.v1+json, "
                     "application/vnd.oci.image.manifest.v1+json"
        }
    )
    try:
        with urllib.request.urlopen(req) as resp:
            # The Docker-Content-Digest header contains the canonical digest
            digest = resp.headers.get("Docker-Content-Digest")
            return digest
    except urllib.error.HTTPError as e:
        print(f"Error fetching manifest: {e}", file=sys.stderr)
        return None


def compute_toolchain_hash(lock_data: dict) -> str:
    """Compute SHA256 hash of the lock data (excluding toolchain_hash itself)."""
    # Create a copy without toolchain_hash for hashing
    data_for_hash = {k: v for k, v in lock_data.items() if k != "toolchain_hash"}
    # Canonical JSON: sorted keys, no extra whitespace
    canonical = json.dumps(data_for_hash, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description="Pin KiCad Docker image digest in the toolchain lock file"
    )
    parser.add_argument(
        "--lock-file",
        type=Path,
        default=Path(__file__).parent.parent / "toolchain" / "kicad.lock.json",
        help="Path to the lock file (default: toolchain/kicad.lock.json)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved reference without updating the file"
    )
    args = parser.parse_args()

    lock_path: Path = args.lock_file

    # Load existing lock file
    if not lock_path.exists():
        print(f"Error: Lock file not found: {lock_path}", file=sys.stderr)
        sys.exit(1)

    with open(lock_path) as f:
        lock_data = json.load(f)

    repository = "kicad/kicad"
    tag = lock_data.get("kicad_version", "9.0.7")

    print(f"Resolving digest for {repository}:{tag}...")

    # Get auth token
    token = get_dockerhub_token(repository)

    # Get the digest
    digest = get_image_digest(repository, tag, token)

    if not digest:
        print("Error: Could not resolve image digest", file=sys.stderr)
        sys.exit(1)

    if is_placeholder_digest(digest):
        print("Error: Digest resolved to placeholder value", file=sys.stderr)
        sys.exit(1)

    if not _DIGEST_PATTERN.match(digest):
        print(f"Error: Digest has unexpected format: {digest}", file=sys.stderr)
        sys.exit(1)

    # Update lock data
    lock_data["docker_digest"] = digest
    lock_data["docker_image"] = f"{repository}:{tag}"
    lock_data["docker_ref"] = f"{repository}:{tag}@{digest}"

    # Compute toolchain hash
    toolchain_hash = compute_toolchain_hash(lock_data)
    lock_data["toolchain_hash"] = toolchain_hash

    pinned_ref = f"{repository}:{tag}@{digest}"

    if args.dry_run:
        print(f"Would pin: {pinned_ref}")
        print(f"Toolchain hash: {toolchain_hash}")
    else:
        # Write updated lock file
        with open(lock_path, "w") as f:
            json.dump(lock_data, f, indent=2)
            f.write("\n")
        print(f"Updated {lock_path}")
        print(f"Pinned: {pinned_ref}")
        print(f"Toolchain hash: {toolchain_hash}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
