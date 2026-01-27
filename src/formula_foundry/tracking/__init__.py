"""M3 Experiment tracking module using MLflow.

This module provides MLflow integration for Formula Foundry, with:
- SQLite backend store for local experiment tracking
- Local artifact root for artifact storage
- Run wrapper for logging M3 artifact references
- Integration with substrate manifest and artifact store
"""

from . import logger
from .config import (
    DEFAULT_CONFIG_PATH,
    MLflowConfig,
    TrackingConfig,
    load_mlflow_config,
)
from .logger import (
    FormulaFoundryTracker,
    TrackedRun,
    get_tracker,
    log_artifact_reference,
    log_manifest,
    log_run_artifacts,
)

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "FormulaFoundryTracker",
    "MLflowConfig",
    "TrackedRun",
    "TrackingConfig",
    "get_tracker",
    "load_mlflow_config",
    "log_artifact_reference",
    "log_manifest",
    "log_run_artifacts",
    "logger",
]
