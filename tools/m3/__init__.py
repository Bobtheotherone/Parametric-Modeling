"""
M3 - Artifact and Dataset Management Module

This module provides schemas and tools for managing artifacts, datasets,
and pipeline runs in the Formula Foundry system.

Schemas:
    - artifact.v1.schema.json: Schema for individual artifacts
    - dataset.v1.schema.json: Schema for versioned collections of artifacts
    - run.v1.schema.json: Schema for pipeline execution records

Scripts:
    - validate_schemas.py: Validate JSON files against M3 schemas
"""

__version__ = "0.1.0"

from pathlib import Path

SCHEMAS_DIR = Path(__file__).parent / "schemas"
SCRIPTS_DIR = Path(__file__).parent / "scripts"


def get_schema_path(schema_type: str, version: int = 1) -> Path:
    """
    Get the path to a schema file.

    Args:
        schema_type: One of 'artifact', 'dataset', or 'run'
        version: Schema version (default: 1)

    Returns:
        Path to the schema file
    """
    return SCHEMAS_DIR / f"{schema_type}.v{version}.schema.json"
