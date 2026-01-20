"""EM (Electromagnetic) simulation support modules.

This package provides solver-agnostic EM simulation configuration and utilities,
including mesh generation, frequency analysis, and adaptive refinement.
"""
from .mesh import (
    AdaptiveMeshDensity,
    FrequencyRange,
    MeshConfig,
    compute_adaptive_mesh_density,
    compute_min_wavelength_nm,
)

__all__ = [
    "AdaptiveMeshDensity",
    "FrequencyRange",
    "MeshConfig",
    "compute_adaptive_mesh_density",
    "compute_min_wavelength_nm",
]
