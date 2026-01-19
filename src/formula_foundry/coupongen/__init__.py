"""Coupon generator (coupongen) package."""

from .api import (
    BuildResult,
    DrcReport,
    KiCadProjectPaths,
    build_coupon,
    export_fab,
    generate_kicad,
    load_spec,
    resolve_spec,
    run_drc,
    validate_spec,
)
from .constraints import (
    ConstraintEvaluation,
    ConstraintProof,
    ConstraintResult,
    ConstraintViolation,
    RepairInfo,
    constraint_proof_payload,
    enforce_constraints,
    evaluate_constraints,
    resolve_fab_limits,
)
from .families import FAMILY_F0, FAMILY_F1, SUPPORTED_FAMILIES, validate_family
from .hashing import (
    canonical_hash_export_text,
    canonical_hash_kicad_pcb_text,
    canonicalize_export_text,
    canonicalize_kicad_pcb_text,
    coupon_id_from_design_hash,
)
from .kicad import BackendA, IKiCadBackend, KicadCliMode, KicadCliRunner, build_drc_args, deterministic_uuid
from .manifest import ManifestPaths, build_manifest, load_manifest, toolchain_hash, write_manifest
from .paths import FOOTPRINT_LIB_DIR, REPO_ROOT
from .resolve import ResolvedDesign, design_hash, resolve, resolved_design_canonical_json
from .spec import COUPONSPEC_SCHEMA, CouponSpec, load_couponspec
from .units import LengthNM, parse_length_nm

__all__ = [
    "COUPONSPEC_SCHEMA",
    "BuildResult",
    "CouponSpec",
    "ConstraintEvaluation",
    "ConstraintProof",
    "ConstraintResult",
    "ConstraintViolation",
    "DrcReport",
    "FAMILY_F0",
    "FAMILY_F1",
    "FOOTPRINT_LIB_DIR",
    "ManifestPaths",
    "BackendA",
    "IKiCadBackend",
    "KicadCliMode",
    "KicadCliRunner",
    "LengthNM",
    "KiCadProjectPaths",
    "REPO_ROOT",
    "RepairInfo",
    "ResolvedDesign",
    "SUPPORTED_FAMILIES",
    "build_coupon",
    "build_manifest",
    "build_drc_args",
    "canonical_hash_export_text",
    "canonical_hash_kicad_pcb_text",
    "canonicalize_export_text",
    "canonicalize_kicad_pcb_text",
    "coupon_id_from_design_hash",
    "constraint_proof_payload",
    "design_hash",
    "deterministic_uuid",
    "enforce_constraints",
    "evaluate_constraints",
    "export_fab",
    "generate_kicad",
    "load_manifest",
    "load_spec",
    "load_couponspec",
    "parse_length_nm",
    "resolve_fab_limits",
    "resolve_spec",
    "resolve",
    "resolved_design_canonical_json",
    "run_drc",
    "toolchain_hash",
    "validate_spec",
    "validate_family",
    "write_manifest",
]
