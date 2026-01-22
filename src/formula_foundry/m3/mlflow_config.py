"""MLflow configuration for M3 artifact storage backbone.

This module provides MLflow integration with SQLite backend store and file-based
artifact root for the Formula Foundry M3 tracking infrastructure. It re-exports
the tracking module components and adds M3-specific functionality for:

- Experiment tracking with SQLite backend
- Run logging with artifact store integration
- Metric recording with lineage tracking

Design doc reference: Section 16 - MLflow configuration with SQLite backend
"""

from __future__ import annotations

import json
import os
import socket
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Re-export tracking module components for convenient access
from formula_foundry.tracking import (
    DEFAULT_CONFIG_PATH,
    FormulaFoundryTracker,
    MLflowConfig,
    TrackedRun,
    TrackingConfig,
    get_tracker,
    load_mlflow_config,
    log_artifact_reference,
    log_manifest,
    log_run_artifacts,
)

if TYPE_CHECKING:
    from formula_foundry.m3.artifact_store import ArtifactManifest, ArtifactStore
    from formula_foundry.m3.registry import ArtifactRegistry

__all__ = [
    # Re-exported from tracking module
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
    # M3-specific exports
    "M3RunContext",
    "M3Tracker",
    "create_m3_run",
    "get_m3_tracker",
    "log_m3_artifact",
    "log_m3_metric",
    "log_m3_metrics",
    "setup_mlflow_environment",
]


# Default paths for M3 tracking
DEFAULT_MLFLOW_DB_PATH = "data/mlflow/mlruns.db"
DEFAULT_MLFLOW_ARTIFACT_ROOT = "data/mlflow/artifacts"


class M3TrackingError(RuntimeError):
    """Raised when M3 tracking operations fail."""


@dataclass
class M3RunMetadata:
    """Metadata for an M3 tracking run.

    This captures provenance information for runs, integrating with
    both MLflow tracking and the M3 artifact store.
    """

    run_id: str
    experiment_id: str
    mlflow_run_id: str
    started_utc: str
    hostname: str
    username: str | None = None
    stage_name: str | None = None
    artifact_store_run_id: str | None = None
    tags: dict[str, str] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, list[tuple[float, int | None]]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "mlflow_run_id": self.mlflow_run_id,
            "started_utc": self.started_utc,
            "hostname": self.hostname,
            "username": self.username,
            "stage_name": self.stage_name,
            "artifact_store_run_id": self.artifact_store_run_id,
            "tags": self.tags,
            "params": self.params,
            "metrics": self.metrics,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


@dataclass
class M3RunContext:
    """Context for an active M3 tracking run.

    Combines MLflow tracking with M3 artifact store integration,
    providing unified logging of parameters, metrics, and artifacts.
    """

    tracked_run: TrackedRun
    metadata: M3RunMetadata
    artifact_store: ArtifactStore | None = None
    registry: ArtifactRegistry | None = None

    def log_param(self, key: str, value: Any) -> None:
        """Log a parameter to both MLflow and M3 metadata."""
        self.tracked_run.log_param(key, value)
        self.metadata.params[key] = value

    def log_params(self, params: Mapping[str, Any]) -> None:
        """Log multiple parameters."""
        self.tracked_run.log_params(params)
        self.metadata.params.update(params)

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        """Log a metric to both MLflow and M3 metadata."""
        self.tracked_run.log_metric(key, value, step=step)
        if key not in self.metadata.metrics:
            self.metadata.metrics[key] = []
        self.metadata.metrics[key].append((value, step))

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        """Log multiple metrics."""
        self.tracked_run.log_metrics(metrics, step=step)
        for key, value in metrics.items():
            if key not in self.metadata.metrics:
                self.metadata.metrics[key] = []
            self.metadata.metrics[key].append((value, step))

    def set_tag(self, key: str, value: str) -> None:
        """Set a tag on the run."""
        self.tracked_run.set_tag(key, value)
        self.metadata.tags[key] = value

    def set_tags(self, tags: Mapping[str, str]) -> None:
        """Set multiple tags."""
        self.tracked_run.set_tags(tags)
        self.metadata.tags.update(tags)

    def log_artifact(
        self,
        logical_path: str,
        digest: str,
        size_bytes: int | None = None,
    ) -> None:
        """Log an artifact reference to MLflow.

        This creates a reference file linking to the M3 artifact store
        instead of duplicating the artifact content.
        """
        self.tracked_run.log_artifact_reference(
            logical_path=logical_path,
            digest=digest,
            size_bytes=size_bytes,
        )

    def log_manifest(self, manifest: ArtifactManifest) -> None:
        """Log an artifact manifest, creating a reference in MLflow."""
        self.log_artifact(
            logical_path=manifest.storage_path or manifest.artifact_id,
            digest=manifest.content_hash.digest,
            size_bytes=manifest.byte_size,
        )


class M3Tracker:
    """MLflow tracker configured for M3 artifact storage backbone.

    This class extends FormulaFoundryTracker with M3-specific functionality
    for integrating experiment tracking with the artifact store and registry.

    Example usage:
        tracker = M3Tracker(project_root=Path("."))
        with tracker.start_run(
            run_name="my-experiment",
            stage_name="simulation",
        ) as ctx:
            ctx.log_param("mesh_size", 1000)
            ctx.log_metric("loss", 0.05)
            ctx.log_artifact("result.touchstone", digest, size)
    """

    def __init__(
        self,
        config: MLflowConfig | None = None,
        project_root: Path | None = None,
        artifact_store: ArtifactStore | None = None,
        registry: ArtifactRegistry | None = None,
    ) -> None:
        """Initialize the M3 tracker.

        Args:
            config: MLflow configuration. Loaded from config/mlflow.yaml if None.
            project_root: Project root directory. Auto-detected if None.
            artifact_store: Optional M3 artifact store for integration.
            registry: Optional M3 registry for run indexing.
        """
        self._tracker = FormulaFoundryTracker(config=config, project_root=project_root)
        self.artifact_store = artifact_store
        self.registry = registry
        self._hostname = socket.gethostname()
        self._username = os.environ.get("USER") or os.environ.get("USERNAME")

    @property
    def config(self) -> MLflowConfig:
        """Get the MLflow configuration."""
        return self._tracker.config

    @property
    def project_root(self) -> Path:
        """Get the project root directory."""
        return self._tracker.project_root

    def initialize(self) -> None:
        """Initialize MLflow with the configured tracking URI and artifact root."""
        self._tracker.initialize()

    def get_or_create_experiment(self, name: str) -> str:
        """Get or create an experiment by name."""
        return self._tracker.get_or_create_experiment(name)

    @staticmethod
    def _generate_run_id() -> str:
        """Generate a unique run ID."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        unique = uuid.uuid4().hex[:8]
        return f"m3run-{timestamp}-{unique}"

    @staticmethod
    def _now_utc_iso() -> str:
        """Get current UTC time in ISO 8601 format."""
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    @contextmanager
    def start_run(
        self,
        run_name: str | None = None,
        experiment_name: str | None = None,
        stage_name: str | None = None,
        nested: bool = False,
        tags: Mapping[str, str] | None = None,
    ) -> Iterator[M3RunContext]:
        """Start an M3 tracked run.

        Args:
            run_name: Optional name for the run.
            experiment_name: Experiment to log to. Uses default if None.
            stage_name: Pipeline stage name for provenance.
            nested: Whether this is a nested run.
            tags: Additional tags to set on the run.

        Yields:
            M3RunContext for logging parameters, metrics, and artifacts.
        """
        self.initialize()

        run_id = self._generate_run_id()
        started_utc = self._now_utc_iso()

        with self._tracker.start_run(
            run_name=run_name or run_id,
            experiment_name=experiment_name,
            nested=nested,
            tags=tags,
        ) as tracked_run:
            metadata = M3RunMetadata(
                run_id=run_id,
                experiment_id=tracked_run.experiment_id,
                mlflow_run_id=tracked_run.mlflow_run_id,
                started_utc=started_utc,
                hostname=self._hostname,
                username=self._username,
                stage_name=stage_name,
                tags=dict(tags) if tags else {},
            )

            ctx = M3RunContext(
                tracked_run=tracked_run,
                metadata=metadata,
                artifact_store=self.artifact_store,
                registry=self.registry,
            )

            # Set M3-specific tags
            ctx.set_tag("m3_run_id", run_id)
            if stage_name:
                ctx.set_tag("m3_stage", stage_name)

            try:
                yield ctx
            finally:
                # Index the run in the registry if available
                if self.registry is not None:
                    ended_utc = self._now_utc_iso()
                    self.registry.index_run(
                        run_id=run_id,
                        started_utc=started_utc,
                        status="completed",
                        stage_name=stage_name,
                        ended_utc=ended_utc,
                        hostname=self._hostname,
                        generator="m3_tracker",
                        generator_version="0.1.0",
                        config={"mlflow_run_id": tracked_run.mlflow_run_id},
                    )


# Global M3 tracker instance (lazy initialized)
_m3_tracker: M3Tracker | None = None


def get_m3_tracker(
    config: MLflowConfig | None = None,
    project_root: Path | None = None,
    artifact_store: ArtifactStore | None = None,
    registry: ArtifactRegistry | None = None,
) -> M3Tracker:
    """Get or create the global M3 tracker instance.

    Args:
        config: MLflow configuration. Uses loaded config if None.
        project_root: Project root directory. Auto-detected if None.
        artifact_store: Optional M3 artifact store for integration.
        registry: Optional M3 registry for run indexing.

    Returns:
        Global M3Tracker instance.
    """
    global _m3_tracker
    if _m3_tracker is None:
        _m3_tracker = M3Tracker(
            config=config,
            project_root=project_root,
            artifact_store=artifact_store,
            registry=registry,
        )
    return _m3_tracker


def setup_mlflow_environment(project_root: Path | None = None) -> MLflowConfig:
    """Set up MLflow environment with SQLite backend and local artifact root.

    This function:
    1. Loads the MLflow configuration from config/mlflow.yaml
    2. Ensures required directories exist (data/mlflow/, data/mlflow/artifacts)
    3. Sets environment variables for MLflow

    Args:
        project_root: Project root directory. Auto-detected if None.

    Returns:
        The loaded MLflow configuration.

    Example:
        config = setup_mlflow_environment()
        print(f"Tracking URI: {config.get_tracking_uri(project_root)}")
    """
    from formula_foundry.tracking.config import get_project_root

    root = project_root or get_project_root()
    config = load_mlflow_config(project_root=root)

    # Ensure directories exist
    config.tracking.ensure_directories(root)

    # Set environment variables
    tracking_uri = config.get_tracking_uri(root)
    artifact_uri = config.get_artifact_uri(root)

    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    os.environ["MLFLOW_ARTIFACT_ROOT"] = artifact_uri

    return config


def create_m3_run(
    run_name: str | None = None,
    experiment_name: str | None = None,
    stage_name: str | None = None,
    project_root: Path | None = None,
    tags: Mapping[str, str] | None = None,
) -> Iterator[M3RunContext]:
    """Convenience function to create an M3 tracking run.

    This is a shorthand for getting the global tracker and starting a run.

    Args:
        run_name: Optional name for the run.
        experiment_name: Experiment to log to. Uses default if None.
        stage_name: Pipeline stage name for provenance.
        project_root: Project root directory. Auto-detected if None.
        tags: Additional tags to set on the run.

    Yields:
        M3RunContext for logging.

    Example:
        with create_m3_run(run_name="my-sim", stage_name="em_simulation") as ctx:
            ctx.log_param("frequency", 10e9)
            ctx.log_metric("s21_db", -0.5)
    """
    tracker = get_m3_tracker(project_root=project_root)
    return tracker.start_run(
        run_name=run_name,
        experiment_name=experiment_name,
        stage_name=stage_name,
        tags=tags,
    )


def log_m3_metric(key: str, value: float, step: int | None = None) -> None:
    """Log a metric directly using MLflow (convenience function).

    This logs directly to the active MLflow run, if any.
    For full M3 tracking, use M3RunContext.log_metric instead.

    Args:
        key: Metric name.
        value: Metric value.
        step: Optional step number.
    """
    try:
        import mlflow

        mlflow.log_metric(key, value, step=step)
    except ImportError:
        pass
    except Exception:
        pass  # Silently fail if no active run


def log_m3_metrics(metrics: Mapping[str, float], step: int | None = None) -> None:
    """Log multiple metrics directly using MLflow (convenience function).

    This logs directly to the active MLflow run, if any.
    For full M3 tracking, use M3RunContext.log_metrics instead.

    Args:
        metrics: Dictionary of metric name -> value.
        step: Optional step number.
    """
    try:
        import mlflow

        mlflow.log_metrics(dict(metrics), step=step)
    except ImportError:
        pass
    except Exception:
        pass  # Silently fail if no active run


def log_m3_artifact(
    logical_path: str,
    digest: str,
    size_bytes: int | None = None,
    project_root: Path | None = None,
) -> None:
    """Log an artifact reference directly to MLflow (convenience function).

    This creates a reference file linking to the M3 artifact store.
    For full M3 tracking, use M3RunContext.log_artifact instead.

    Args:
        logical_path: Logical path of the artifact.
        digest: Content hash digest.
        size_bytes: Optional size in bytes.
        project_root: Project root directory.
    """
    log_artifact_reference(
        logical_path=logical_path,
        digest=digest,
        size_bytes=size_bytes,
        project_root=project_root,
    )
