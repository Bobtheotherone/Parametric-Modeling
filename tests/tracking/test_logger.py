"""Tests for MLflow logging wrapper."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from formula_foundry.substrate import (
    ArtifactEntry,
    ArtifactManifest,
    Manifest,
    RunArtifacts,
)
from formula_foundry.tracking.config import (
    ArtifactConfig,
    ExperimentConfig,
    MLflowConfig,
    RetentionConfig,
    TagConfig,
    TrackingConfig,
)
from formula_foundry.tracking.logger import (
    FormulaFoundryTracker,
    TrackedRun,
    log_artifact_reference,
    log_manifest,
    log_run_artifacts,
)


def _create_test_config() -> MLflowConfig:
    """Create a test MLflow configuration."""
    return MLflowConfig(
        tracking=TrackingConfig(
            backend_store_uri="sqlite:///data/mlflow/mlruns.db",
            artifact_root="data/mlflow/artifacts",
            default_experiment="test-experiment",
        ),
        experiments=ExperimentConfig(prefix="test", names={}),
        tags=TagConfig(
            required=("git_sha", "design_doc_sha256", "environment_fingerprint", "determinism_mode"),
            optional=("coupon_id",),
        ),
        parameters={},
        metrics={},
        artifacts=ArtifactConfig(
            link_mode="reference",
            references=("manifest.json",),
            direct=("metrics.json",),
        ),
        retention=RetentionConfig(
            min_retention_days=30,
            min_runs_per_experiment=100,
            archive_after_days=90,
        ),
    )


def _create_test_manifest() -> Manifest:
    """Create a test manifest."""
    return Manifest(
        git_sha="a" * 40,
        design_doc_sha256="b" * 64,
        environment_fingerprint="c" * 64,
        determinism={
            "mode": "strict",
            "seeds": {"python": 42, "numpy": 42, "cupy": 42, "torch": 42},
            "cublas_workspace_config": ":4096:8",
        },
        command_line=["python", "run.py", "--mode", "test"],
        artifacts={"output.json": "d" * 64},
    )


def _create_test_run_artifacts(tmp_path: Path) -> RunArtifacts:
    """Create test run artifacts."""
    run_dir = tmp_path / "runs" / "test-run-001"
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    logs_path = run_dir / "logs.jsonl"
    logs_path.touch()
    manifest_path = run_dir / "manifest.json"

    return RunArtifacts(
        run_id="test-run-001",
        run_dir=run_dir,
        manifest_path=manifest_path,
        logs_path=logs_path,
        artifacts_dir=artifacts_dir,
    )


class TestTrackedRun:
    """Tests for TrackedRun."""

    def test_log_param_without_mlflow(self) -> None:
        """TrackedRun.log_param works without mlflow installed."""
        config = _create_test_config()
        run = TrackedRun(
            mlflow_run_id="",
            experiment_id="0",
            run_artifacts=None,
            config=config,
            project_root=Path.cwd(),
        )
        # Should not raise
        run.log_param("key", "value")

    def test_log_metric_without_mlflow(self) -> None:
        """TrackedRun.log_metric works without mlflow installed."""
        config = _create_test_config()
        run = TrackedRun(
            mlflow_run_id="",
            experiment_id="0",
            run_artifacts=None,
            config=config,
            project_root=Path.cwd(),
        )
        # Should not raise
        run.log_metric("metric", 1.0)

    def test_set_tag_without_mlflow(self) -> None:
        """TrackedRun.set_tag works without mlflow installed."""
        config = _create_test_config()
        run = TrackedRun(
            mlflow_run_id="",
            experiment_id="0",
            run_artifacts=None,
            config=config,
            project_root=Path.cwd(),
        )
        # Should not raise
        run.set_tag("tag", "value")


class TestFormulaFoundryTracker:
    """Tests for FormulaFoundryTracker."""

    def test_tracker_initialization(self, tmp_path: Path) -> None:
        """Tracker initializes without mlflow installed."""
        config = _create_test_config()
        tracker = FormulaFoundryTracker(config=config, project_root=tmp_path)

        assert tracker.project_root == tmp_path
        assert tracker.config == config
        assert tracker._initialized is False

    def test_tracker_initialize_creates_directories(self, tmp_path: Path) -> None:
        """Tracker.initialize creates required directories."""
        config = _create_test_config()
        tracker = FormulaFoundryTracker(config=config, project_root=tmp_path)

        tracker.initialize()

        assert (tmp_path / "data/mlflow/artifacts").exists()
        assert (tmp_path / "data/mlflow").exists()
        assert tracker._initialized is True

    def test_start_run_context_manager(self, tmp_path: Path) -> None:
        """Tracker.start_run works as context manager without mlflow."""
        config = _create_test_config()
        tracker = FormulaFoundryTracker(config=config, project_root=tmp_path)

        with tracker.start_run(run_name="test-run") as run:
            assert isinstance(run, TrackedRun)
            assert run.config == config
            assert run.project_root == tmp_path

    def test_run_artifacts_linkage(self, tmp_path: Path) -> None:
        """REQ-M3-004: Run wrapper supports linking to substrate RunArtifacts."""
        config = _create_test_config()
        tracker = FormulaFoundryTracker(config=config, project_root=tmp_path)
        run_artifacts = _create_test_run_artifacts(tmp_path)

        with tracker.start_run(
            run_name="linked-run",
            run_artifacts=run_artifacts,
        ) as run:
            assert run.run_artifacts == run_artifacts
            assert run.run_artifacts.run_id == "test-run-001"


class TestLogManifest:
    """Tests for log_manifest."""

    def test_log_manifest(self) -> None:
        """REQ-M3-003: Tracker logs substrate manifest data."""
        manifest = _create_test_manifest()

        # Mock mlflow
        with mock.patch.dict(sys.modules, {"mlflow": mock.MagicMock()}):
            import mlflow

            log_manifest(manifest)

            # Verify tags were set
            calls = mlflow.set_tag.call_args_list
            tag_keys = [call[0][0] for call in calls]

            assert "git_sha" in tag_keys
            assert "design_doc_sha256" in tag_keys
            assert "environment_fingerprint" in tag_keys
            assert "determinism_mode" in tag_keys


class TestLogArtifactReference:
    """Tests for log_artifact_reference."""

    def test_artifact_reference(self, tmp_path: Path) -> None:
        """REQ-M3-005: Artifact references are logged as JSON files."""
        config = _create_test_config()

        # Mock mlflow and tempfile
        with mock.patch.dict(sys.modules, {"mlflow": mock.MagicMock()}):
            import mlflow

            logged_artifact_path = None

            def capture_artifact(path: str, artifact_path: str | None = None) -> None:
                nonlocal logged_artifact_path
                logged_artifact_path = path

            mlflow.log_artifact = capture_artifact

            log_artifact_reference(
                logical_path="coupon.kicad_pcb",
                digest="e" * 64,
                size_bytes=12345,
                config=config,
                project_root=tmp_path,
            )

            # Verify a JSON file was created (even if temp file is deleted)
            # The function should have been called with a .json file
            assert logged_artifact_path is not None
            assert logged_artifact_path.endswith(".json")


class TestLogRunArtifacts:
    """Tests for log_run_artifacts."""

    def test_log_run_artifacts_basic(self, tmp_path: Path) -> None:
        """log_run_artifacts logs substrate_run_id tag."""
        run_artifacts = _create_test_run_artifacts(tmp_path)

        with mock.patch.dict(sys.modules, {"mlflow": mock.MagicMock()}):
            import mlflow

            log_run_artifacts(run_artifacts)

            # Verify tag was set
            calls = mlflow.set_tag.call_args_list
            tag_dict = {call[0][0]: call[0][1] for call in calls}

            assert "substrate_run_id" in tag_dict
            assert tag_dict["substrate_run_id"] == "test-run-001"

    def test_log_run_artifacts_with_manifest(self, tmp_path: Path) -> None:
        """log_run_artifacts logs artifact references from manifest."""
        run_artifacts = _create_test_run_artifacts(tmp_path)
        artifact_manifest = ArtifactManifest.from_entries([
            ArtifactEntry(path="file1.txt", digest="f" * 64, size_bytes=100),
            ArtifactEntry(path="file2.txt", digest="0" * 64, size_bytes=200),
        ])

        logged_paths: list[str] = []

        with mock.patch.dict(sys.modules, {"mlflow": mock.MagicMock()}):
            import mlflow

            def capture_artifact(path: str, artifact_path: str | None = None) -> None:
                logged_paths.append(path)

            mlflow.log_artifact = capture_artifact

            log_run_artifacts(
                run_artifacts,
                artifact_manifest=artifact_manifest,
                project_root=tmp_path,
            )

            # Two artifacts should have been logged
            assert len(logged_paths) == 2


class TestNoopWithoutMlflow:
    """Tests for no-op behavior when mlflow is not installed."""

    def test_noop_without_mlflow(self, tmp_path: Path) -> None:
        """REQ-M3-007: Tracker operates in no-op mode when MLflow is not installed."""
        config = _create_test_config()
        tracker = FormulaFoundryTracker(config=config, project_root=tmp_path)

        # Simulate mlflow not installed by removing it from sys.modules
        original_modules = dict(sys.modules)
        if "mlflow" in sys.modules:
            del sys.modules["mlflow"]

        try:
            # Initialize should work
            tracker._initialized = False
            tracker.initialize()
            assert tracker._initialized is True

            # start_run should work and return a TrackedRun
            with tracker.start_run(run_name="noop-run") as run:
                assert isinstance(run, TrackedRun)

                # All logging methods should work without error
                run.log_param("key", "value")
                run.log_params({"k1": "v1", "k2": "v2"})
                run.log_metric("metric", 1.0)
                run.log_metrics({"m1": 1.0, "m2": 2.0})
                run.set_tag("tag", "value")
                run.set_tags({"t1": "v1", "t2": "v2"})
        finally:
            sys.modules.update(original_modules)
