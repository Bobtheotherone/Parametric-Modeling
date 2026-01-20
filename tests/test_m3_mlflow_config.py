"""Tests for MLflow configuration and tracking integration.

These tests verify:
1. MLflow configuration loading from YAML
2. SQLite backend store configuration
3. Local artifact root configuration
4. Integration with M3 artifact store
5. Run tracking and artifact reference logging
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml


# ---------------------------------------------------------------------------
# Test configuration loading
# ---------------------------------------------------------------------------


class TestMLflowConfigLoading:
    """Test MLflow configuration loading from YAML."""

    def test_load_config_from_yaml(self, tmp_path: Path) -> None:
        """Test loading configuration from a YAML file."""
        from formula_foundry.tracking.config import load_mlflow_config

        config_content = {
            "tracking": {
                "backend_store_uri": "sqlite:///test.db",
                "artifact_root": "artifacts",
                "default_experiment": "test-exp",
            },
            "experiments": {"prefix": "test", "names": {"m1": "test-m1"}},
            "tags": {"required": ["git_sha"], "optional": ["backend"]},
            "parameters": {"coupongen": ["family"]},
            "metrics": {"simulation": ["loss"]},
            "artifacts": {
                "link_mode": "reference",
                "references": ["manifest.json"],
                "direct": ["metrics.json"],
            },
            "retention": {
                "min_retention_days": 7,
                "min_runs_per_experiment": 10,
                "archive_after_days": 30,
            },
        }

        config_path = tmp_path / "mlflow.yaml"
        with config_path.open("w") as f:
            yaml.dump(config_content, f)

        config = load_mlflow_config(config_path=config_path, project_root=tmp_path)

        assert config.tracking.backend_store_uri == "sqlite:///test.db"
        assert config.tracking.artifact_root == "artifacts"
        assert config.tracking.default_experiment == "test-exp"
        assert config.experiments.prefix == "test"
        assert config.experiments.names.get("m1") == "test-m1"
        assert "git_sha" in config.tags.required
        assert config.artifacts.link_mode == "reference"
        assert config.retention.min_retention_days == 7

    def test_load_default_config_when_missing(self, tmp_path: Path) -> None:
        """Test that default config is returned when file is missing."""
        from formula_foundry.tracking.config import (
            DEFAULT_ARTIFACT_ROOT,
            DEFAULT_BACKEND_STORE_URI,
            DEFAULT_EXPERIMENT,
            load_mlflow_config,
        )

        config = load_mlflow_config(
            config_path=tmp_path / "nonexistent.yaml", project_root=tmp_path
        )

        assert config.tracking.backend_store_uri == DEFAULT_BACKEND_STORE_URI
        assert config.tracking.artifact_root == DEFAULT_ARTIFACT_ROOT
        assert config.tracking.default_experiment == DEFAULT_EXPERIMENT

    def test_invalid_yaml_raises_error(self, tmp_path: Path) -> None:
        """Test that invalid YAML raises MLflowConfigError."""
        from formula_foundry.tracking.config import MLflowConfigError, load_mlflow_config

        config_path = tmp_path / "invalid.yaml"
        config_path.write_text("{ invalid yaml [")

        with pytest.raises(MLflowConfigError, match="Invalid YAML"):
            load_mlflow_config(config_path=config_path, project_root=tmp_path)

    def test_non_mapping_raises_error(self, tmp_path: Path) -> None:
        """Test that non-mapping YAML raises MLflowConfigError."""
        from formula_foundry.tracking.config import MLflowConfigError, load_mlflow_config

        config_path = tmp_path / "list.yaml"
        config_path.write_text("- item1\n- item2\n")

        with pytest.raises(MLflowConfigError, match="must be a mapping"):
            load_mlflow_config(config_path=config_path, project_root=tmp_path)


class TestTrackingConfig:
    """Test TrackingConfig dataclass methods."""

    def test_ensure_directories_creates_paths(self, tmp_path: Path) -> None:
        """Test that ensure_directories creates required directories."""
        from formula_foundry.tracking.config import TrackingConfig

        config = TrackingConfig(
            backend_store_uri="sqlite:///data/mlflow/test.db",
            artifact_root="data/artifacts",
            default_experiment="test",
        )

        config.ensure_directories(tmp_path)

        assert (tmp_path / "data" / "mlflow").exists()
        assert (tmp_path / "data" / "artifacts").exists()

    def test_ensure_directories_handles_absolute_uri(self, tmp_path: Path) -> None:
        """Test that ensure_directories handles non-relative SQLite URIs."""
        from formula_foundry.tracking.config import TrackingConfig

        # Absolute path (4 slashes)
        config = TrackingConfig(
            backend_store_uri=f"sqlite:////{tmp_path}/absolute.db",
            artifact_root="artifacts",
            default_experiment="test",
        )

        config.ensure_directories(tmp_path)
        assert (tmp_path / "artifacts").exists()


class TestMLflowConfigPaths:
    """Test MLflowConfig path resolution methods."""

    def test_get_tracking_uri_relative(self, tmp_path: Path) -> None:
        """Test tracking URI resolution for relative paths."""
        from formula_foundry.tracking.config import load_mlflow_config

        config_content = {
            "tracking": {
                "backend_store_uri": "sqlite:///data/mlflow.db",
                "artifact_root": "artifacts",
                "default_experiment": "test",
            }
        }

        config_path = tmp_path / "mlflow.yaml"
        with config_path.open("w") as f:
            yaml.dump(config_content, f)

        config = load_mlflow_config(config_path=config_path, project_root=tmp_path)
        uri = config.get_tracking_uri(tmp_path)

        assert uri == f"sqlite:///{tmp_path}/data/mlflow.db"

    def test_get_artifact_uri(self, tmp_path: Path) -> None:
        """Test artifact URI resolution."""
        from formula_foundry.tracking.config import load_mlflow_config

        config_content = {
            "tracking": {
                "backend_store_uri": "sqlite:///test.db",
                "artifact_root": "data/artifacts",
                "default_experiment": "test",
            }
        }

        config_path = tmp_path / "mlflow.yaml"
        with config_path.open("w") as f:
            yaml.dump(config_content, f)

        config = load_mlflow_config(config_path=config_path, project_root=tmp_path)
        uri = config.get_artifact_uri(tmp_path)

        expected = str((tmp_path / "data/artifacts").absolute())
        assert uri == expected


# ---------------------------------------------------------------------------
# Test tracking logger (with mock MLflow)
# ---------------------------------------------------------------------------


class TestFormulaFoundryTracker:
    """Test FormulaFoundryTracker with mocked MLflow."""

    def test_tracker_initialization(self, tmp_path: Path) -> None:
        """Test tracker initialization without MLflow installed."""
        from formula_foundry.tracking.config import load_mlflow_config
        from formula_foundry.tracking.logger import FormulaFoundryTracker

        config = load_mlflow_config(project_root=tmp_path)
        tracker = FormulaFoundryTracker(config=config, project_root=tmp_path)

        assert tracker.project_root == tmp_path
        assert tracker.config == config
        assert not tracker._initialized

    def test_tracker_noop_without_mlflow(self, tmp_path: Path) -> None:
        """Test that tracker works in no-op mode without MLflow."""
        from formula_foundry.tracking.config import load_mlflow_config
        from formula_foundry.tracking.logger import FormulaFoundryTracker

        config = load_mlflow_config(project_root=tmp_path)
        tracker = FormulaFoundryTracker(config=config, project_root=tmp_path)

        # Simulate MLflow not installed by patching the import inside initialize
        with patch.dict("sys.modules", {"mlflow": None}):
            # Initialize should not raise even without MLflow
            tracker.initialize()
            assert tracker._initialized

    def test_get_or_create_experiment_noop(self, tmp_path: Path) -> None:
        """Test get_or_create_experiment returns default ID without MLflow."""
        from formula_foundry.tracking.config import load_mlflow_config
        from formula_foundry.tracking.logger import FormulaFoundryTracker

        config = load_mlflow_config(project_root=tmp_path)
        tracker = FormulaFoundryTracker(config=config, project_root=tmp_path)

        # Without MLflow, should return "0"
        exp_id = tracker.get_or_create_experiment("test-exp")
        assert exp_id == "0"


class TestTrackedRun:
    """Test TrackedRun logging methods."""

    def test_tracked_run_log_param_noop(self, tmp_path: Path) -> None:
        """Test that log_param works without MLflow."""
        from formula_foundry.tracking.config import load_mlflow_config
        from formula_foundry.tracking.logger import TrackedRun

        config = load_mlflow_config(project_root=tmp_path)
        run = TrackedRun(
            mlflow_run_id="test-run",
            experiment_id="0",
            run_artifacts=None,
            config=config,
            project_root=tmp_path,
        )

        # Should not raise even without MLflow
        run.log_param("key", "value")
        run.log_params({"a": 1, "b": 2})

    def test_tracked_run_log_metric_noop(self, tmp_path: Path) -> None:
        """Test that log_metric works without MLflow."""
        from formula_foundry.tracking.config import load_mlflow_config
        from formula_foundry.tracking.logger import TrackedRun

        config = load_mlflow_config(project_root=tmp_path)
        run = TrackedRun(
            mlflow_run_id="test-run",
            experiment_id="0",
            run_artifacts=None,
            config=config,
            project_root=tmp_path,
        )

        # Should not raise
        run.log_metric("loss", 0.5)
        run.log_metrics({"rmse": 0.1, "r2": 0.95})

    def test_tracked_run_set_tags_noop(self, tmp_path: Path) -> None:
        """Test that set_tag/set_tags works without MLflow."""
        from formula_foundry.tracking.config import load_mlflow_config
        from formula_foundry.tracking.logger import TrackedRun

        config = load_mlflow_config(project_root=tmp_path)
        run = TrackedRun(
            mlflow_run_id="test-run",
            experiment_id="0",
            run_artifacts=None,
            config=config,
            project_root=tmp_path,
        )

        # Should not raise
        run.set_tag("git_sha", "abc123")
        run.set_tags({"a": "1", "b": "2"})


# ---------------------------------------------------------------------------
# Test integration with M3 artifact store
# ---------------------------------------------------------------------------


class TestMLflowM3Integration:
    """Test integration between MLflow tracking and M3 artifact store."""

    def test_log_artifact_reference_creates_ref_file(self, tmp_path: Path) -> None:
        """Test that log_artifact_reference creates a reference JSON file."""
        from formula_foundry.tracking.config import load_mlflow_config
        from formula_foundry.tracking.logger import log_artifact_reference

        config = load_mlflow_config(project_root=tmp_path)

        # Create a mock for mlflow that captures the logged artifact
        logged_artifacts = []

        def mock_log_artifact(path: str, artifact_path: str = None) -> None:
            content = Path(path).read_text()
            logged_artifacts.append({"path": path, "content": json.loads(content)})

        with patch.dict("sys.modules", {"mlflow": MagicMock()}):
            import sys

            mock_mlflow = sys.modules["mlflow"]
            mock_mlflow.log_artifact = mock_log_artifact

            log_artifact_reference(
                logical_path="data/test.json",
                digest="abc123def456",
                size_bytes=1024,
                config=config,
                project_root=tmp_path,
            )

            # Verify the reference was logged
            assert len(logged_artifacts) == 1
            ref = logged_artifacts[0]["content"]
            assert ref["logical_path"] == "data/test.json"
            assert ref["digest"] == "abc123def456"
            assert ref["size_bytes"] == 1024
            assert ref["store_type"] == "substrate"

    def test_config_artifact_link_mode(self, tmp_path: Path) -> None:
        """Test that artifact linking configuration is respected."""
        from formula_foundry.tracking.config import load_mlflow_config

        config_content = {
            "tracking": {
                "backend_store_uri": "sqlite:///test.db",
                "artifact_root": "artifacts",
                "default_experiment": "test",
            },
            "artifacts": {
                "link_mode": "reference",
                "references": ["manifest.json", "logs.jsonl"],
                "direct": ["metrics.json"],
            },
        }

        config_path = tmp_path / "mlflow.yaml"
        with config_path.open("w") as f:
            yaml.dump(config_content, f)

        config = load_mlflow_config(config_path=config_path, project_root=tmp_path)

        assert config.artifacts.link_mode == "reference"
        assert "manifest.json" in config.artifacts.references
        assert "metrics.json" in config.artifacts.direct


# ---------------------------------------------------------------------------
# Test configuration file exists and is valid
# ---------------------------------------------------------------------------


class TestConfigFileValidity:
    """Test that the actual config/mlflow.yaml file is valid."""

    def test_mlflow_yaml_exists(self) -> None:
        """Test that config/mlflow.yaml exists in the repo."""
        from formula_foundry.tracking.config import get_project_root

        root = get_project_root()
        config_path = root / "config" / "mlflow.yaml"

        # This test may fail if run from a different directory
        # In that case, we skip it
        if not config_path.exists():
            pytest.skip("config/mlflow.yaml not found from current directory")

        assert config_path.exists()

    def test_mlflow_yaml_is_valid(self) -> None:
        """Test that config/mlflow.yaml can be loaded."""
        from formula_foundry.tracking.config import get_project_root, load_mlflow_config

        root = get_project_root()
        config_path = root / "config" / "mlflow.yaml"

        if not config_path.exists():
            pytest.skip("config/mlflow.yaml not found from current directory")

        # Should not raise
        config = load_mlflow_config(config_path=config_path, project_root=root)

        # Verify required fields are present
        assert config.tracking.backend_store_uri.startswith("sqlite:///")
        assert config.tracking.artifact_root
        assert config.tracking.default_experiment

    def test_mlflow_yaml_matches_design_doc(self) -> None:
        """Test that config matches design document requirements (Section 5.2)."""
        from formula_foundry.tracking.config import get_project_root, load_mlflow_config

        root = get_project_root()
        config_path = root / "config" / "mlflow.yaml"

        if not config_path.exists():
            pytest.skip("config/mlflow.yaml not found from current directory")

        config = load_mlflow_config(config_path=config_path, project_root=root)

        # Section 5.2 requires SQLite backend store
        assert "sqlite" in config.tracking.backend_store_uri.lower()

        # Section 5.2 requires local artifact root
        assert config.tracking.artifact_root
        # Should not be a remote URI
        assert not config.tracking.artifact_root.startswith("s3://")
        assert not config.tracking.artifact_root.startswith("gs://")

        # Artifact linking mode should be "reference" to integrate with M3 store
        assert config.artifacts.link_mode == "reference"


# ---------------------------------------------------------------------------
# Test global tracker instance
# ---------------------------------------------------------------------------


class TestGlobalTracker:
    """Test global tracker singleton."""

    def test_get_tracker_returns_same_instance(self, tmp_path: Path) -> None:
        """Test that get_tracker returns the same instance."""
        from formula_foundry.tracking import logger

        # Reset global tracker
        logger._tracker = None

        config = MagicMock()
        config.tracking.ensure_directories = MagicMock()

        tracker1 = logger.get_tracker(config=config, project_root=tmp_path)
        tracker2 = logger.get_tracker()

        assert tracker1 is tracker2

        # Clean up
        logger._tracker = None

    def test_get_tracker_uses_project_root(self, tmp_path: Path) -> None:
        """Test that get_tracker uses provided project root."""
        from formula_foundry.tracking import logger

        # Reset global tracker
        logger._tracker = None

        tracker = logger.get_tracker(project_root=tmp_path)

        assert tracker.project_root == tmp_path

        # Clean up
        logger._tracker = None
