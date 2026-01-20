"""Tests for MLflow configuration loading."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from formula_foundry.tracking.config import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_BACKEND_STORE_URI,
    DEFAULT_CONFIG_PATH,
    DEFAULT_EXPERIMENT,
    MLflowConfig,
    MLflowConfigError,
    TrackingConfig,
    load_mlflow_config,
)


class TestTrackingConfig:
    """Tests for TrackingConfig."""

    def test_backend_store_uri(self) -> None:
        """REQ-M3-001: Backend store URI must use SQLite."""
        config = TrackingConfig(
            backend_store_uri="sqlite:///data/mlflow/mlruns.db",
            artifact_root="data/mlflow/artifacts",
            default_experiment="test",
        )
        assert config.backend_store_uri == "sqlite:///data/mlflow/mlruns.db"
        assert config.backend_store_uri.startswith("sqlite:///")

    def test_artifact_root(self) -> None:
        """REQ-M3-002: Artifact root must be configured."""
        config = TrackingConfig(
            backend_store_uri="sqlite:///test.db",
            artifact_root="data/mlflow/artifacts",
            default_experiment="test",
        )
        assert config.artifact_root == "data/mlflow/artifacts"

    def test_ensure_directories(self, tmp_path: Path) -> None:
        """TrackingConfig.ensure_directories creates required directories."""
        config = TrackingConfig(
            backend_store_uri="sqlite:///data/mlflow/mlruns.db",
            artifact_root="data/mlflow/artifacts",
            default_experiment="test",
        )
        config.ensure_directories(tmp_path)

        assert (tmp_path / "data/mlflow/artifacts").exists()
        assert (tmp_path / "data/mlflow").exists()


class TestLoadConfig:
    """Tests for load_mlflow_config."""

    def test_load_config(self, tmp_path: Path) -> None:
        """REQ-M3-006: Configuration must be loadable from YAML."""
        config_data = {
            "tracking": {
                "backend_store_uri": "sqlite:///custom/path.db",
                "artifact_root": "custom/artifacts",
                "default_experiment": "custom-exp",
            },
            "experiments": {
                "prefix": "test",
                "names": {"m1": "test-m1"},
            },
            "tags": {
                "required": ["git_sha"],
                "optional": ["custom_tag"],
            },
            "parameters": {
                "coupongen": ["param1", "param2"],
            },
            "metrics": {
                "simulation": ["metric1"],
            },
            "artifacts": {
                "link_mode": "reference",
                "references": ["manifest.json"],
                "direct": ["metrics.json"],
            },
            "retention": {
                "min_retention_days": 60,
                "min_runs_per_experiment": 50,
                "archive_after_days": 180,
            },
        }

        config_path = tmp_path / "mlflow.yaml"
        with config_path.open("w") as f:
            yaml.dump(config_data, f)

        config = load_mlflow_config(config_path=config_path, project_root=tmp_path)

        assert config.tracking.backend_store_uri == "sqlite:///custom/path.db"
        assert config.tracking.artifact_root == "custom/artifacts"
        assert config.tracking.default_experiment == "custom-exp"
        assert config.experiments.prefix == "test"
        assert config.experiments.names == {"m1": "test-m1"}
        assert config.tags.required == ("git_sha",)
        assert config.tags.optional == ("custom_tag",)
        assert config.parameters == {"coupongen": ("param1", "param2")}
        assert config.metrics == {"simulation": ("metric1",)}
        assert config.artifacts.link_mode == "reference"
        assert config.retention.min_retention_days == 60

    def test_load_config_defaults(self, tmp_path: Path) -> None:
        """Configuration uses defaults when file doesn't exist."""
        config = load_mlflow_config(
            config_path=tmp_path / "nonexistent.yaml",
            project_root=tmp_path,
        )

        assert config.tracking.backend_store_uri == DEFAULT_BACKEND_STORE_URI
        assert config.tracking.artifact_root == DEFAULT_ARTIFACT_ROOT
        assert config.tracking.default_experiment == DEFAULT_EXPERIMENT

    def test_load_config_partial(self, tmp_path: Path) -> None:
        """Configuration uses defaults for missing fields."""
        config_data = {
            "tracking": {
                "default_experiment": "partial-exp",
            },
        }

        config_path = tmp_path / "mlflow.yaml"
        with config_path.open("w") as f:
            yaml.dump(config_data, f)

        config = load_mlflow_config(config_path=config_path, project_root=tmp_path)

        assert config.tracking.backend_store_uri == DEFAULT_BACKEND_STORE_URI
        assert config.tracking.artifact_root == DEFAULT_ARTIFACT_ROOT
        assert config.tracking.default_experiment == "partial-exp"

    def test_load_config_invalid_yaml(self, tmp_path: Path) -> None:
        """Invalid YAML raises MLflowConfigError."""
        config_path = tmp_path / "mlflow.yaml"
        config_path.write_text("{ invalid yaml: [")

        with pytest.raises(MLflowConfigError, match="Invalid YAML"):
            load_mlflow_config(config_path=config_path, project_root=tmp_path)

    def test_load_config_non_mapping(self, tmp_path: Path) -> None:
        """Non-mapping YAML raises MLflowConfigError."""
        config_path = tmp_path / "mlflow.yaml"
        config_path.write_text("- item1\n- item2\n")

        with pytest.raises(MLflowConfigError, match="must be a mapping"):
            load_mlflow_config(config_path=config_path, project_root=tmp_path)


class TestMLflowConfig:
    """Tests for MLflowConfig."""

    def test_get_tracking_uri_relative(self, tmp_path: Path) -> None:
        """get_tracking_uri converts relative paths to absolute."""
        config = MLflowConfig(
            tracking=TrackingConfig(
                backend_store_uri="sqlite:///data/mlflow/mlruns.db",
                artifact_root="data/mlflow/artifacts",
                default_experiment="test",
            ),
            experiments=config._create_default_experiment_config(),
            tags=config._create_default_tag_config(),
            parameters={},
            metrics={},
            artifacts=config._create_default_artifact_config(),
            retention=config._create_default_retention_config(),
        )

        uri = config.get_tracking_uri(tmp_path)

        assert uri.startswith("sqlite:///")
        assert str(tmp_path) in uri
        assert uri.endswith("mlruns.db")

    def test_get_tracking_uri_absolute(self, tmp_path: Path) -> None:
        """get_tracking_uri preserves absolute paths."""
        abs_path = tmp_path / "absolute.db"
        config = MLflowConfig(
            tracking=TrackingConfig(
                backend_store_uri=f"sqlite:///{abs_path}",
                artifact_root="data/mlflow/artifacts",
                default_experiment="test",
            ),
            experiments=_create_default_experiment_config(),
            tags=_create_default_tag_config(),
            parameters={},
            metrics={},
            artifacts=_create_default_artifact_config(),
            retention=_create_default_retention_config(),
        )

        uri = config.get_tracking_uri(tmp_path)

        assert uri == f"sqlite:///{abs_path}"

    def test_get_artifact_uri(self, tmp_path: Path) -> None:
        """get_artifact_uri returns absolute path."""
        config = MLflowConfig(
            tracking=TrackingConfig(
                backend_store_uri="sqlite:///test.db",
                artifact_root="data/mlflow/artifacts",
                default_experiment="test",
            ),
            experiments=_create_default_experiment_config(),
            tags=_create_default_tag_config(),
            parameters={},
            metrics={},
            artifacts=_create_default_artifact_config(),
            retention=_create_default_retention_config(),
        )

        uri = config.get_artifact_uri(tmp_path)

        assert str(tmp_path) in uri
        assert uri.endswith("artifacts")


def _create_default_experiment_config():
    from formula_foundry.tracking.config import ExperimentConfig
    return ExperimentConfig(prefix="ff", names={})


def _create_default_tag_config():
    from formula_foundry.tracking.config import TagConfig
    return TagConfig(
        required=("git_sha", "design_doc_sha256", "environment_fingerprint", "determinism_mode"),
        optional=("coupon_id", "design_hash", "toolchain_hash", "backend"),
    )


def _create_default_artifact_config():
    from formula_foundry.tracking.config import ArtifactConfig
    return ArtifactConfig(
        link_mode="reference",
        references=("manifest.json", "logs.jsonl"),
        direct=("metrics.json", "params.json"),
    )


def _create_default_retention_config():
    from formula_foundry.tracking.config import RetentionConfig
    return RetentionConfig(
        min_retention_days=30,
        min_runs_per_experiment=100,
        archive_after_days=90,
    )
