"""MLflow logging wrapper for Formula Foundry M3 artifact tracking.

This module provides a run wrapper that integrates MLflow with the existing
substrate manifest and artifact store infrastructure.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..substrate import (
    ArtifactManifest,
    Manifest,
    RunArtifacts,
)
from .config import (
    MLflowConfig,
    get_project_root,
    load_mlflow_config,
)

# Global tracker instance (lazy initialized)
_tracker: FormulaFoundryTracker | None = None


class TrackingError(RuntimeError):
    """Raised when tracking operations fail."""


@dataclass
class TrackedRun:
    """Represents an active tracked run.

    Combines MLflow run context with substrate RunArtifacts for
    unified logging of both MLflow metrics and artifact references.
    """

    mlflow_run_id: str
    experiment_id: str
    run_artifacts: RunArtifacts | None
    config: MLflowConfig
    project_root: Path

    def log_param(self, key: str, value: Any) -> None:
        """Log a parameter to MLflow."""
        try:
            import mlflow

            mlflow.log_param(key, value)
        except ImportError:
            pass

    def log_params(self, params: Mapping[str, Any]) -> None:
        """Log multiple parameters to MLflow."""
        try:
            import mlflow

            mlflow.log_params(dict(params))
        except ImportError:
            pass

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        """Log a metric to MLflow."""
        try:
            import mlflow

            mlflow.log_metric(key, value, step=step)
        except ImportError:
            pass

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        """Log multiple metrics to MLflow."""
        try:
            import mlflow

            mlflow.log_metrics(dict(metrics), step=step)
        except ImportError:
            pass

    def set_tag(self, key: str, value: str) -> None:
        """Set a tag on the run."""
        try:
            import mlflow

            mlflow.set_tag(key, value)
        except ImportError:
            pass

    def set_tags(self, tags: Mapping[str, str]) -> None:
        """Set multiple tags on the run."""
        try:
            import mlflow

            mlflow.set_tags(dict(tags))
        except ImportError:
            pass

    def log_artifact_reference(
        self,
        logical_path: str,
        digest: str,
        size_bytes: int | None = None,
    ) -> None:
        """Log a reference to an artifact in the substrate artifact store.

        Instead of duplicating artifacts, we log a JSON file with the
        artifact reference (digest + logical path) that can be used to
        retrieve the artifact from the content-addressed store.
        """
        log_artifact_reference(
            logical_path=logical_path,
            digest=digest,
            size_bytes=size_bytes,
            config=self.config,
            project_root=self.project_root,
        )


class FormulaFoundryTracker:
    """MLflow tracker configured for Formula Foundry.

    This class manages MLflow experiment and run lifecycle, integrating
    with the substrate infrastructure for artifact tracking.
    """

    def __init__(
        self,
        config: MLflowConfig | None = None,
        project_root: Path | None = None,
    ) -> None:
        """Initialize the tracker.

        Args:
            config: MLflow configuration. Loaded from config/mlflow.yaml if None.
            project_root: Project root directory. Auto-detected if None.
        """
        self.project_root = project_root or get_project_root()
        self.config = config or load_mlflow_config(project_root=self.project_root)
        self._initialized = False

    def initialize(self) -> None:
        """Initialize MLflow with the configured tracking URI and artifact root.

        This should be called once at application startup. It:
        - Ensures directories exist
        - Sets the tracking URI
        - Creates the default experiment if it doesn't exist
        """
        if self._initialized:
            return

        # Ensure directories exist
        self.config.tracking.ensure_directories(self.project_root)

        try:
            import mlflow

            # Set tracking URI (SQLite database)
            tracking_uri = self.config.get_tracking_uri(self.project_root)
            mlflow.set_tracking_uri(tracking_uri)

            # Set artifact root via environment variable
            artifact_uri = self.config.get_artifact_uri(self.project_root)
            os.environ["MLFLOW_ARTIFACT_ROOT"] = artifact_uri

            # Create or get default experiment
            experiment = mlflow.get_experiment_by_name(self.config.tracking.default_experiment)
            if experiment is None:
                mlflow.create_experiment(
                    self.config.tracking.default_experiment,
                    artifact_location=artifact_uri,
                )

            self._initialized = True
        except ImportError:
            # MLflow not installed - tracker will operate in no-op mode
            self._initialized = True

    def get_or_create_experiment(self, name: str) -> str:
        """Get or create an experiment by name.

        Args:
            name: Experiment name.

        Returns:
            Experiment ID.
        """
        self.initialize()

        try:
            import mlflow

            experiment = mlflow.get_experiment_by_name(name)
            if experiment is not None:
                return experiment.experiment_id

            artifact_uri = self.config.get_artifact_uri(self.project_root)
            return mlflow.create_experiment(name, artifact_location=artifact_uri)
        except ImportError:
            return "0"

    @contextmanager
    def start_run(
        self,
        run_name: str | None = None,
        experiment_name: str | None = None,
        run_artifacts: RunArtifacts | None = None,
        manifest: Manifest | None = None,
        nested: bool = False,
        tags: Mapping[str, str] | None = None,
    ) -> Iterator[TrackedRun]:
        """Start a tracked MLflow run.

        Args:
            run_name: Optional name for the run.
            experiment_name: Experiment to log to. Uses default if None.
            run_artifacts: Optional substrate RunArtifacts to link.
            manifest: Optional substrate Manifest to log.
            nested: Whether this is a nested run.
            tags: Additional tags to set on the run.

        Yields:
            TrackedRun context for logging.
        """
        self.initialize()

        exp_name = experiment_name or self.config.tracking.default_experiment
        experiment_id = self.get_or_create_experiment(exp_name)

        try:
            import mlflow

            with mlflow.start_run(
                run_name=run_name,
                experiment_id=experiment_id,
                nested=nested,
            ) as run:
                tracked = TrackedRun(
                    mlflow_run_id=run.info.run_id,
                    experiment_id=experiment_id,
                    run_artifacts=run_artifacts,
                    config=self.config,
                    project_root=self.project_root,
                )

                # Log manifest data as tags/params if provided
                if manifest is not None:
                    log_manifest(manifest, config=self.config)

                # Log run_id linkage if run_artifacts provided
                if run_artifacts is not None:
                    mlflow.set_tag("substrate_run_id", run_artifacts.run_id)
                    mlflow.set_tag("substrate_run_dir", str(run_artifacts.run_dir))

                # Log additional tags
                if tags:
                    mlflow.set_tags(dict(tags))

                yield tracked
        except ImportError:
            # No-op mode when MLflow not installed
            yield TrackedRun(
                mlflow_run_id="",
                experiment_id=experiment_id,
                run_artifacts=run_artifacts,
                config=self.config,
                project_root=self.project_root,
            )


def get_tracker(
    config: MLflowConfig | None = None,
    project_root: Path | None = None,
) -> FormulaFoundryTracker:
    """Get or create the global tracker instance.

    Args:
        config: MLflow configuration. Uses loaded config if None.
        project_root: Project root directory. Auto-detected if None.

    Returns:
        Global FormulaFoundryTracker instance.
    """
    global _tracker
    if _tracker is None:
        _tracker = FormulaFoundryTracker(config=config, project_root=project_root)
    return _tracker


def log_manifest(
    manifest: Manifest,
    *,
    config: MLflowConfig | None = None,
) -> None:
    """Log substrate manifest data to MLflow.

    Logs manifest fields as tags according to configuration.
    Required tags from manifest are always logged.
    """
    try:
        import mlflow

        # Log required tags from manifest
        mlflow.set_tag("git_sha", manifest.git_sha)
        mlflow.set_tag("design_doc_sha256", manifest.design_doc_sha256)
        mlflow.set_tag("environment_fingerprint", manifest.environment_fingerprint)

        # Log determinism as tags
        det = manifest.determinism
        if "mode" in det:
            mlflow.set_tag("determinism_mode", det["mode"])
        if "seeds" in det and isinstance(det["seeds"], dict):
            for key, value in det["seeds"].items():
                mlflow.set_tag(f"seed_{key}", str(value))
        if "cublas_workspace_config" in det and det["cublas_workspace_config"]:
            mlflow.set_tag("cublas_workspace_config", det["cublas_workspace_config"])

        # Log command line as param
        if manifest.command_line:
            mlflow.log_param("command_line", " ".join(manifest.command_line[:5]))

    except ImportError:
        pass


def log_artifact_reference(
    logical_path: str,
    digest: str,
    size_bytes: int | None = None,
    *,
    config: MLflowConfig | None = None,
    project_root: Path | None = None,
) -> None:
    """Log a reference to an artifact in the substrate artifact store.

    Creates a JSON file with the artifact reference that can be used
    to retrieve the artifact from the content-addressed store.
    """
    try:
        import tempfile

        import mlflow

        ref_data = {
            "logical_path": logical_path,
            "digest": digest,
            "size_bytes": size_bytes,
            "store_type": "substrate",
        }

        # Create a temporary file with the reference
        f"{logical_path.replace('/', '_')}.ref.json"
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            delete=False,
            encoding="utf-8",
        ) as f:
            json.dump(ref_data, f, indent=2)
            temp_path = f.name

        try:
            mlflow.log_artifact(temp_path, artifact_path="refs")
        finally:
            os.unlink(temp_path)

    except ImportError:
        pass


def log_run_artifacts(
    run_artifacts: RunArtifacts,
    artifact_manifest: ArtifactManifest | None = None,
    *,
    config: MLflowConfig | None = None,
    project_root: Path | None = None,
) -> None:
    """Log substrate RunArtifacts to MLflow.

    Logs the run_id linkage and optionally logs artifact references
    from the artifact manifest.
    """
    try:
        import mlflow

        mlflow.set_tag("substrate_run_id", run_artifacts.run_id)
        mlflow.set_tag("substrate_run_dir", str(run_artifacts.run_dir))

        if artifact_manifest is not None:
            for entry in artifact_manifest.artifacts:
                log_artifact_reference(
                    logical_path=entry.path,
                    digest=entry.digest,
                    size_bytes=entry.size_bytes,
                    config=config,
                    project_root=project_root,
                )

    except ImportError:
        pass
