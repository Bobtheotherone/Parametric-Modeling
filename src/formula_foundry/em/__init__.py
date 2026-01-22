"""EM (Electromagnetic) simulation support modules.

This package provides solver-agnostic EM simulation configuration and utilities,
including mesh generation, frequency analysis, adaptive refinement,
Touchstone file I/O for S-parameter data, and S-parameter validation.

REQ-M2-007: Includes validation for passivity, reciprocity, and causality.
"""

from .mesh import (
    AdaptiveMeshDensity,
    FrequencyRange,
    MeshConfig,
    compute_adaptive_mesh_density,
    compute_min_wavelength_nm,
)
from .touchstone import (
    FrequencyUnit,
    NetworkType,
    SParameterData,
    SParameterFormat,
    TouchstoneOptions,
    create_empty_sparam_data,
    create_thru_sparam_data,
    from_skrf_network,
    merge_sparam_data,
    read_touchstone,
    read_touchstone_from_string,
    to_skrf_network,
    validate_with_skrf,
    write_touchstone,
    write_touchstone_to_string,
)
from .validation import (
    CausalityCheckResult,
    PassivityCheckResult,
    ReciprocityCheckResult,
    SParameterValidationResult,
    ValidationStatus,
    build_validation_manifest_entry,
    check_causality,
    check_passivity,
    check_reciprocity,
    check_stability_2port,
    compute_stability_k_factor,
    validate_sparam_data,
    validate_touchstone_file,
)

__all__ = [
    # Mesh configuration
    "AdaptiveMeshDensity",
    "FrequencyRange",
    "MeshConfig",
    "compute_adaptive_mesh_density",
    "compute_min_wavelength_nm",
    # Touchstone I/O
    "FrequencyUnit",
    "NetworkType",
    "SParameterData",
    "SParameterFormat",
    "TouchstoneOptions",
    "create_empty_sparam_data",
    "create_thru_sparam_data",
    "from_skrf_network",
    "merge_sparam_data",
    "read_touchstone",
    "read_touchstone_from_string",
    "to_skrf_network",
    "validate_with_skrf",
    "write_touchstone",
    "write_touchstone_to_string",
    # S-parameter Validation (REQ-M2-007)
    "CausalityCheckResult",
    "PassivityCheckResult",
    "ReciprocityCheckResult",
    "SParameterValidationResult",
    "ValidationStatus",
    "build_validation_manifest_entry",
    "check_causality",
    "check_passivity",
    "check_reciprocity",
    "check_stability_2port",
    "compute_stability_k_factor",
    "validate_sparam_data",
    "validate_touchstone_file",
]
