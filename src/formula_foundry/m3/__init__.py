"""Formula Foundry M3: Artifact storage backbone.

This module provides content-addressed storage with atomic writes,
lineage tracking, and integration with DVC/MLflow tooling.
"""

from formula_foundry.m3.artifact_store import ArtifactStore

__all__ = ["ArtifactStore"]
