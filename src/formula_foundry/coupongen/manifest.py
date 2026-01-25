"""Manifest generation for coupon builds.

This module generates manifest.json files for every coupon build with complete
provenance information and export hashes.

Satisfies:
    - REQ-M1-013: The manifest MUST include a spec-consumption summary, footprint
                  provenance (paths + hashes of source footprint content), and an
                  explicit zone policy record (refill/check behavior and toolchain
                  versioning).
    - REQ-M1-018: The repo must emit a manifest.json for every build containing
                  required provenance fields and export hashes.
    - CP-5.1: Ensure toolchain provenance always captured (lock_file_toolchain_hash)

Required manifest fields (per DESIGN_DOCUMENT.md Section 13.5.1):
    - schema_version, coupon_family
    - design_hash, coupon_id
    - resolved_design (integer nm params)
    - derived_features + dimensionless_groups
    - fab_profile (id + resolved limits)
    - stackup (nm thicknesses + material props)
    - toolchain:
        - kicad.version
        - kicad.cli_version_output
        - docker.image_ref (tag+digest)
        - mode
        - generator_git_sha
        - lock_file_toolchain_hash (CP-5.1: from toolchain lock file)
    - toolchain_hash (computed from runtime toolchain metadata)
    - exports list with canonical hashes
    - verification:
        - constraints (passed + failed_ids)
        - drc (returncode, report_path, summary, canonical_hash)
        - layer_set (per Section 13.5.3) - validation of expected layers
    - lineage (git sha, UTC timestamp - explicitly excluded from design_hash)
    - spec_consumption: consumed/expected/unused paths (REQ-M1-013)
    - footprint_provenance: paths and hashes of source footprints (REQ-M1-013)
    - zone_policy: refill/check behavior and toolchain versioning (REQ-M1-013)

DRC canonicalization is delegated to kicad/canonicalize.py per Section 13.5.2,
which is the authoritative source for all artifact canonicalization algorithms.
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
from .geom.footprint_meta import FootprintMeta, load_footprint_meta
from .kicad.canonicalize import canonical_hash_drc_json
from .kicad.runners.protocol import ZonePolicy
from .layer_validation import LayerValidationResult, layer_validation_payload
from .resolve import ResolvedDesign
from .spec import CouponSpec


@dataclass(frozen=True)
class ManifestPaths:
    manifest_path: Path
    output_dir: Path


def toolchain_hash(toolchain: Mapping[str, Any]) -> str:
    canonical = canonical_json_dumps(dict(toolchain))
    return sha256_bytes(canonical.encode("utf-8"))


def _build_footprint_provenance(spec: CouponSpec) -> dict[str, dict[str, str]]:
    """Build footprint provenance from spec connectors.

    Extracts footprint IDs from spec, loads metadata, and returns provenance
    with paths and hashes for each unique footprint.

    REQ-M1-013: Footprint provenance must include paths and hashes of source
    footprint content.

    Args:
        spec: The coupon specification containing connector footprint IDs.

    Returns:
        Dictionary mapping footprint IDs to provenance info:
        {
            "footprint_id": {
                "path": "/path/to/footprint.kicad_mod",
                "footprint_hash": "sha256...",
                "metadata_hash": "sha256..."
            }
        }
        Keys are sorted for stable ordering.
    """
    provenance: dict[str, dict[str, str]] = {}
    # Collect unique footprint IDs from left and right connectors
    footprint_ids = {spec.connectors.left.footprint, spec.connectors.right.footprint}
    for fp_id in sorted(footprint_ids):
        try:
            meta: FootprintMeta = load_footprint_meta(fp_id)
            provenance[fp_id] = {
                "path": str(meta.footprint_file),
                "footprint_hash": meta.footprint_hash,
                "metadata_hash": meta.metadata_hash,
            }
        except FileNotFoundError:
            # If metadata not found, record with empty hashes for traceability
            provenance[fp_id] = {
                "path": "",
                "footprint_hash": "",
                "metadata_hash": "",
            }
    return provenance


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
    layer_validation: LayerValidationResult | None = None,
    git_sha: str | None = None,
    timestamp_utc: str | None = None,
    footprint_provenance: Mapping[str, dict[str, str]] | None = None,
    zone_policy: ZonePolicy | None = None,
) -> dict[str, Any]:
    """Build a manifest dictionary with all required provenance fields.

    Satisfies REQ-M1-013 and REQ-M1-018: The repo must emit a manifest.json for
    every build containing required provenance fields, export hashes,
    spec-consumption summary, footprint provenance, and zone policy record.

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
        layer_validation: Optional layer set validation result (per Section 13.5.3).
        git_sha: Optional explicit git SHA (defaults to HEAD of cwd).
        timestamp_utc: Optional explicit UTC timestamp (defaults to now).
        footprint_provenance: Optional mapping of footprint IDs to provenance info
            (paths + hashes). If None, computed from spec connectors (REQ-M1-013).
        zone_policy: Optional ZonePolicy record for zone refill/check behavior
            and toolchain versioning (REQ-M1-013). If None, uses default policy.

    Returns:
        Dictionary with all required manifest fields ready for JSON serialization.
    """
    resolved_git_sha = _resolve_git_sha(git_sha)
    timestamp = timestamp_utc or _utc_timestamp()
    exports = [{"path": path, "hash": export_hashes[path]} for path in sorted(export_hashes.keys())]
    failed_constraints = [result.constraint_id for result in proof.constraints if not result.passed]
    drc_summary = parse_drc_summary(drc_report_path)
    drc_canonical_hash = canonicalize_drc_report(drc_report_path)

    # REQ-M1-013: Build footprint provenance if not explicitly provided
    fp_provenance = (
        dict(footprint_provenance) if footprint_provenance is not None
        else _build_footprint_provenance(spec)
    )

    # REQ-M1-013: Build zone policy record if not explicitly provided
    from .kicad.runners.protocol import DEFAULT_ZONE_POLICY
    zp_record = zone_policy.to_dict() if zone_policy is not None else DEFAULT_ZONE_POLICY.to_dict()

    manifest = {
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
                "summary": drc_summary,
                "canonical_hash": drc_canonical_hash,
            },
            "layer_set": layer_validation_payload(layer_validation) if layer_validation else None,
        },
        "lineage": {
            "git_sha": resolved_git_sha,
            "timestamp_utc": timestamp,
        },
        # REQ-M1-013: Footprint provenance with paths and hashes (stable ordering)
        "footprint_provenance": dict(sorted(fp_provenance.items())),
        # REQ-M1-013: Explicit zone policy details
        "zone_policy": zp_record,
    }
    # REQ-M1-013: Include spec-consumption summary in manifest
    consumption_summary = resolved.get_spec_consumption_summary()
    if consumption_summary is not None:
        manifest["spec_consumption"] = consumption_summary
    return manifest


def parse_drc_summary(drc_report_path: Path) -> dict[str, int]:
    """Parse a KiCad DRC JSON report and return a summary.

    Args:
        drc_report_path: Path to the DRC JSON report file.

    Returns:
        Dictionary with violation, warning, and exclusion counts.
    """
    if not drc_report_path.exists():
        return {"violations": 0, "warnings": 0, "exclusions": 0}

    try:
        report = json.loads(drc_report_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"violations": 0, "warnings": 0, "exclusions": 0}

    violations = len(report.get("violations", []))
    warnings = len(report.get("warnings", []))
    exclusions = len(report.get("exclusions", []))

    return {"violations": violations, "warnings": warnings, "exclusions": exclusions}


def canonicalize_drc_report(drc_report_path: Path) -> str:
    """Canonicalize a DRC report and return its hash.

    This function delegates to the authoritative canonicalization algorithm
    in kicad/canonicalize.py (per Section 13.5.2), which handles:
    - Removing timestamps (date, time, timestamp, generated_at)
    - Removing environment keys (kicad_version, host, source, schema_version)
    - Normalizing paths to filenames only
    - Sorting all object keys alphabetically
    - Preserving list ordering (semantically significant)

    Args:
        drc_report_path: Path to the DRC JSON report file.

    Returns:
        SHA-256 hash of the canonicalized DRC report.
    """
    if not drc_report_path.exists():
        return sha256_bytes(b"")

    try:
        report = json.loads(drc_report_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return sha256_bytes(b"")

    # Delegate to the authoritative canonicalization implementation
    return canonical_hash_drc_json(report)


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
