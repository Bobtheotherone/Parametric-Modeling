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
    resolve_fab_limits_from_profile,
)
from .export import (
    CacheKey,
    ExportPipeline,
    ExportResult,
    PipelineProgress,
    PipelineStage,
    compute_cache_key,
    is_cache_valid,
    run_export_pipeline,
)
from .fab_profiles import (
    FAB_PROFILE_SCHEMA,
    FAB_PROFILES_DIR,
    BoardConstraints,
    DrillConstraints,
    FabCapabilityProfile,
    SilkscreenConstraints,
    SoldermaskConstraints,
    TraceConstraints,
    ViaConstraints,
    clear_profile_cache,
    get_fab_limits,
    list_available_profiles,
    load_fab_profile,
    load_fab_profile_from_dict,
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
from .stackups import (
    STACKUP_SCHEMA,
    STACKUPS_DIR,
    DielectricProperties,
    StackupLayer,
    StackupProfile,
    clear_stackup_cache,
    compute_total_thickness,
    get_copper_layer_names,
    get_dielectric_between_layers,
    get_effective_er,
    get_thickness_between_layers,
    list_available_stackups,
    load_stackup,
    load_stackup_from_dict,
)
from .toolchain import (
    DEFAULT_LOCK_PATH,
    ToolchainConfig,
    ToolchainLoadError,
    compute_toolchain_hash,
    load_toolchain_lock,
)
from .units import LengthNM, parse_length_nm

__all__ = [
    # Schemas
    "COUPONSPEC_SCHEMA",
    "FAB_PROFILE_SCHEMA",
    "STACKUP_SCHEMA",
    # Paths
    "FAB_PROFILES_DIR",
    "FOOTPRINT_LIB_DIR",
    "REPO_ROOT",
    "STACKUPS_DIR",
    # Spec types
    "BuildResult",
    "CacheKey",
    "CouponSpec",
    "DrcReport",
    "ExportPipeline",
    "ExportResult",
    "KiCadProjectPaths",
    "ManifestPaths",
    "PipelineProgress",
    "PipelineStage",
    "ResolvedDesign",
    # Constraint types
    "ConstraintEvaluation",
    "ConstraintProof",
    "ConstraintResult",
    "ConstraintViolation",
    "RepairInfo",
    # Fab profile types
    "BoardConstraints",
    "DrillConstraints",
    "FabCapabilityProfile",
    "SilkscreenConstraints",
    "SoldermaskConstraints",
    "TraceConstraints",
    "ViaConstraints",
    # Stackup types
    "DielectricProperties",
    "StackupLayer",
    "StackupProfile",
    # KiCad types
    "BackendA",
    "IKiCadBackend",
    "KicadCliMode",
    "KicadCliRunner",
    # Constants
    "FAMILY_F0",
    "FAMILY_F1",
    "LengthNM",
    "SUPPORTED_FAMILIES",
    # API functions
    "build_coupon",
    "build_drc_args",
    "build_manifest",
    "canonical_hash_export_text",
    "canonical_hash_kicad_pcb_text",
    "canonicalize_export_text",
    "canonicalize_kicad_pcb_text",
    "constraint_proof_payload",
    "coupon_id_from_design_hash",
    "design_hash",
    "deterministic_uuid",
    "enforce_constraints",
    "evaluate_constraints",
    "export_fab",
    "generate_kicad",
    "load_couponspec",
    "load_manifest",
    "load_spec",
    "parse_length_nm",
    "resolve",
    "resolve_fab_limits",
    "resolve_fab_limits_from_profile",
    "resolve_spec",
    "resolved_design_canonical_json",
    "run_drc",
    "toolchain_hash",
    "validate_family",
    "validate_spec",
    "write_manifest",
    # Fab profile functions
    "clear_profile_cache",
    "get_fab_limits",
    "list_available_profiles",
    "load_fab_profile",
    "load_fab_profile_from_dict",
    # Stackup functions
    "clear_stackup_cache",
    "compute_total_thickness",
    "get_copper_layer_names",
    "get_dielectric_between_layers",
    "get_effective_er",
    "get_thickness_between_layers",
    "list_available_stackups",
    "load_stackup",
    "load_stackup_from_dict",
    # Export pipeline functions
    "compute_cache_key",
    "is_cache_valid",
    "run_export_pipeline",
    # Toolchain types
    "ToolchainConfig",
    "ToolchainLoadError",
    # Toolchain constants
    "DEFAULT_LOCK_PATH",
    # Toolchain functions
    "compute_toolchain_hash",
    "load_toolchain_lock",
]
