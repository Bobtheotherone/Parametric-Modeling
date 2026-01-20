"""Manifest generation for coupon builds.

This module generates manifest.json files for every coupon build with complete
provenance information and export hashes.

Satisfies:
    - REQ-M1-018: The repo must emit a manifest.json for every build containing
                  required provenance fields and export hashes.

Required manifest fields (per DESIGN_DOCUMENT.md Section 9.3):
    - schema_version, coupon_family
    - design_hash, coupon_id
    - resolved_design
    - derived_features + dimensionless_groups
    - fab_profile_id + resolved limits
    - stackup
    - toolchain (KiCad version, docker image tag/digest, kicad-cli --version output)
    - exports list with canonical hashes
    - verification (DRC summary + constraint_proof summary)
    - lineage (git commit hash, UTC timestamp)
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from formula_foundry.substrate import canonical_json_dumps, get_git_sha, sha256_bytes

from .constraints import ConstraintProof, resolve_fab_limits
from .resolve import ResolvedDesign
from .spec import CouponSpec


@dataclass(frozen=True)
class ManifestPaths:
    manifest_path: Path
    output_dir: Path


def toolchain_hash(toolchain: Mapping[str, Any]) -> str:
    canonical = canonical_json_dumps(dict(toolchain))
    return sha256_bytes(canonical.encode("utf-8"))


def build_manifest(
    *,
    spec: CouponSpec,
    resolved: ResolvedDesign,
    proof: ConstraintProof,
    design_hash: str,
    coupon_id: str,
    toolchain: Mapping[str, Any],
    toolchain_hash_value: str,
    export_hashes: Mapping[str, str],
    drc_report_path: Path,
    drc_returncode: int,
    git_sha: str | None = None,
    timestamp_utc: str | None = None,
) -> dict[str, Any]:
    """Build a manifest dictionary with all required provenance fields.

    Satisfies REQ-M1-018: The repo must emit a manifest.json for every build
    containing required provenance fields and export hashes.

    Args:
        spec: The original coupon specification.
        resolved: The resolved design with concrete integer-nm values.
        proof: The constraint proof from validation.
        design_hash: SHA256 hash of the canonical resolved design.
        coupon_id: Human-readable identifier derived from design_hash.
        toolchain: Toolchain metadata (kicad_version, docker_image, mode, kicad_cli_version).
        toolchain_hash_value: SHA256 hash of the toolchain metadata.
        export_hashes: Mapping of relative export paths to their canonical hashes.
        drc_report_path: Path to the DRC JSON report.
        drc_returncode: Return code from the DRC check.
        git_sha: Optional explicit git SHA (defaults to HEAD of cwd).
        timestamp_utc: Optional explicit UTC timestamp (defaults to now).

    Returns:
        Dictionary with all required manifest fields ready for JSON serialization.
    """
    resolved_git_sha = _resolve_git_sha(git_sha)
    timestamp = timestamp_utc or _utc_timestamp()
    exports = [
        {"path": path, "hash": export_hashes[path]}
        for path in sorted(export_hashes.keys())
    ]
    failed_constraints = [result.constraint_id for result in proof.constraints if not result.passed]
    return {
        "schema_version": spec.schema_version,
        "coupon_family": spec.coupon_family,
        "design_hash": design_hash,
        "coupon_id": coupon_id,
        "resolved_design": resolved.model_dump(mode="json"),
        "derived_features": dict(resolved.derived_features),
        "dimensionless_groups": dict(resolved.dimensionless_groups),
        "fab_profile": {
            "id": spec.fab_profile.id,
            "limits": resolve_fab_limits(spec),
        },
        "stackup": spec.stackup.model_dump(mode="json"),
        "toolchain": dict(toolchain),
        "toolchain_hash": toolchain_hash_value,
        "exports": exports,
        "verification": {
            "constraints": {
                "passed": proof.passed,
                "failed_ids": failed_constraints,
            },
            "drc": {
                "returncode": drc_returncode,
                "report_path": str(drc_report_path),
            },
        },
        "lineage": {
            "git_sha": resolved_git_sha,
            "timestamp_utc": timestamp,
        },
    }


def write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    text = canonical_json_dumps(dict(manifest))
    path.write_text(f"{text}\n", encoding="utf-8")


def load_manifest(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _resolve_git_sha(explicit: str | None) -> str:
    if explicit and len(explicit) == 40:
        return explicit
    try:
        return get_git_sha(Path.cwd())
    except Exception:
        return "0" * 40
