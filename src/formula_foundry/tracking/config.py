"""MLflow configuration loader for Formula Foundry M3 tracking.

This module loads and validates the MLflow configuration from config/mlflow.yaml
and provides dataclasses for type-safe access to configuration values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path("config/mlflow.yaml")

# Default values if config is missing
DEFAULT_BACKEND_STORE_URI = "sqlite:///data/mlflow/mlruns.db"
DEFAULT_ARTIFACT_ROOT = "data/mlflow/artifacts"
DEFAULT_EXPERIMENT = "formula-foundry"


class MLflowConfigError(ValueError):
    """Raised when MLflow configuration is invalid."""


@dataclass(frozen=True)
class TrackingConfig:
    """Tracking server configuration."""

    backend_store_uri: str
    artifact_root: str
    default_experiment: str

    def ensure_directories(self, project_root: Path) -> None:
        """Ensure artifact directories exist."""
        artifact_path = project_root / self.artifact_root
        artifact_path.mkdir(parents=True, exist_ok=True)

        # Also ensure the SQLite database directory exists
        if self.backend_store_uri.startswith("sqlite:///"):
            db_path = self.backend_store_uri.replace("sqlite:///", "")
            db_dir = project_root / Path(db_path).parent
            db_dir.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ExperimentConfig:
    """Experiment naming configuration."""

    prefix: str
    names: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class TagConfig:
    """Tag configuration for runs."""

    required: tuple[str, ...]
    optional: tuple[str, ...]


@dataclass(frozen=True)
class ArtifactConfig:
    """Artifact linking configuration."""

    link_mode: str
    references: tuple[str, ...]
    direct: tuple[str, ...]


@dataclass(frozen=True)
class RetentionConfig:
    """Retention and cleanup configuration."""

    min_retention_days: int
    min_runs_per_experiment: int
    archive_after_days: int


@dataclass(frozen=True)
class MLflowConfig:
    """Complete MLflow configuration."""

    tracking: TrackingConfig
    experiments: ExperimentConfig
    tags: TagConfig
    parameters: dict[str, tuple[str, ...]]
    metrics: dict[str, tuple[str, ...]]
    artifacts: ArtifactConfig
    retention: RetentionConfig

    def get_tracking_uri(self, project_root: Path) -> str:
        """Get the absolute tracking URI for MLflow.

        Converts relative SQLite paths to absolute paths based on project root.
        """
        uri = self.tracking.backend_store_uri
        if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):
            # Relative path - make absolute
            db_path = uri.replace("sqlite:///", "")
            absolute_path = project_root / db_path
            return f"sqlite:///{absolute_path}"
        return uri

    def get_artifact_uri(self, project_root: Path) -> str:
        """Get the absolute artifact root URI for MLflow."""
        artifact_path = project_root / self.tracking.artifact_root
        return str(artifact_path.absolute())


def load_mlflow_config(
    config_path: Path | None = None,
    project_root: Path | None = None,
) -> MLflowConfig:
    """Load MLflow configuration from YAML file.

    Args:
        config_path: Path to mlflow.yaml config file. Defaults to config/mlflow.yaml.
        project_root: Project root directory. Defaults to current working directory.

    Returns:
        MLflowConfig with all configuration values.

    Raises:
        MLflowConfigError: If configuration is invalid or missing required fields.
    """
    root = project_root or Path.cwd()
    path = config_path or (root / DEFAULT_CONFIG_PATH)

    if not path.exists():
        # Return defaults if config doesn't exist
        return _create_default_config()

    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise MLflowConfigError(f"Invalid YAML in {path}: {e}") from e

    if not isinstance(data, dict):
        raise MLflowConfigError(f"Config must be a mapping, got {type(data).__name__}")

    return _parse_config(data)


def _create_default_config() -> MLflowConfig:
    """Create a default MLflow configuration."""
    return MLflowConfig(
        tracking=TrackingConfig(
            backend_store_uri=DEFAULT_BACKEND_STORE_URI,
            artifact_root=DEFAULT_ARTIFACT_ROOT,
            default_experiment=DEFAULT_EXPERIMENT,
        ),
        experiments=ExperimentConfig(prefix="ff", names={}),
        tags=TagConfig(
            required=("git_sha", "design_doc_sha256", "environment_fingerprint", "determinism_mode"),
            optional=("coupon_id", "design_hash", "toolchain_hash", "backend"),
        ),
        parameters={},
        metrics={},
        artifacts=ArtifactConfig(
            link_mode="reference",
            references=("manifest.json", "logs.jsonl"),
            direct=("metrics.json", "params.json"),
        ),
        retention=RetentionConfig(
            min_retention_days=30,
            min_runs_per_experiment=100,
            archive_after_days=90,
        ),
    )


def _parse_config(data: dict[str, Any]) -> MLflowConfig:
    """Parse configuration from YAML data."""
    tracking_data = data.get("tracking", {})
    tracking = TrackingConfig(
        backend_store_uri=tracking_data.get("backend_store_uri", DEFAULT_BACKEND_STORE_URI),
        artifact_root=tracking_data.get("artifact_root", DEFAULT_ARTIFACT_ROOT),
        default_experiment=tracking_data.get("default_experiment", DEFAULT_EXPERIMENT),
    )

    exp_data = data.get("experiments", {})
    experiments = ExperimentConfig(
        prefix=exp_data.get("prefix", "ff"),
        names=exp_data.get("names", {}),
    )

    tags_data = data.get("tags", {})
    tags = TagConfig(
        required=tuple(tags_data.get("required", [])),
        optional=tuple(tags_data.get("optional", [])),
    )

    params_data = data.get("parameters", {})
    parameters = {k: tuple(v) for k, v in params_data.items()}

    metrics_data = data.get("metrics", {})
    metrics = {k: tuple(v) for k, v in metrics_data.items()}

    artifacts_data = data.get("artifacts", {})
    artifacts = ArtifactConfig(
        link_mode=artifacts_data.get("link_mode", "reference"),
        references=tuple(artifacts_data.get("references", [])),
        direct=tuple(artifacts_data.get("direct", [])),
    )

    retention_data = data.get("retention", {})
    retention = RetentionConfig(
        min_retention_days=retention_data.get("min_retention_days", 30),
        min_runs_per_experiment=retention_data.get("min_runs_per_experiment", 100),
        archive_after_days=retention_data.get("archive_after_days", 90),
    )

    return MLflowConfig(
        tracking=tracking,
        experiments=experiments,
        tags=tags,
        parameters=parameters,
        metrics=metrics,
        artifacts=artifacts,
        retention=retention,
    )


def get_project_root() -> Path:
    """Get the project root directory.

    Walks up from current directory looking for pyproject.toml or .git.
    Falls back to current directory if not found.
    """
    current = Path.cwd()
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    return current
