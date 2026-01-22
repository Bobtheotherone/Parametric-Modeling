"""M3 CLI: Artifact storage backbone commands.

This module provides the `m3` command-line interface for initializing and
managing the Formula Foundry data provenance system.

Commands:
    init: Initialize data directory structure, DVC, MLflow config, and registry.
    run: Execute a DVC stage with metadata stamping and artifact tracking.
    gc: Garbage collect old artifacts with configurable retention policies.
    audit: Generate deterministic provenance reports for artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

# Version for provenance tracking
__version__ = "0.1.0"


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the m3 CLI."""
    parser = argparse.ArgumentParser(
        prog="m3",
        description="Formula Foundry M3: Artifact storage backbone CLI",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # init subcommand
    init_parser = subparsers.add_parser(
        "init",
        help="Initialize data directory structure, DVC, MLflow, and registry",
    )
    init_parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Project root directory (defaults to current directory)",
    )
    init_parser.add_argument(
        "--force",
        action="store_true",
        help="Reinitialize even if already initialized",
    )
    init_parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress non-error output",
    )

    # dataset subcommand
    dataset_parser = subparsers.add_parser(
        "dataset",
        help="Inspect and compare dataset snapshots",
    )
    dataset_subparsers = dataset_parser.add_subparsers(
        dest="dataset_command",
        required=True,
    )

    # dataset show subcommand
    dataset_show_parser = dataset_subparsers.add_parser(
        "show",
        help="Display detailed information about a dataset snapshot",
    )
    dataset_show_parser.add_argument(
        "dataset_id",
        help="Dataset ID or path to dataset JSON manifest",
    )
    dataset_show_parser.add_argument(
        "--version",
        "-v",
        type=str,
        default=None,
        help="Dataset version (default: latest if multiple exist)",
    )
    dataset_show_parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Project root directory (defaults to auto-detect)",
    )
    dataset_show_parser.add_argument(
        "--json",
        action="store_true",
        dest="output_json",
        help="Output raw JSON instead of formatted text",
    )
    dataset_show_parser.add_argument(
        "--members",
        action="store_true",
        help="Show detailed member list (can be verbose for large datasets)",
    )
    dataset_show_parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress non-error output",
    )

    # dataset diff subcommand
    dataset_diff_parser = dataset_subparsers.add_parser(
        "diff",
        help="Compare two dataset versions and show differences",
    )
    dataset_diff_parser.add_argument(
        "dataset_a",
        help="First dataset ID or path (the 'from' version)",
    )
    dataset_diff_parser.add_argument(
        "dataset_b",
        help="Second dataset ID or path (the 'to' version)",
    )
    dataset_diff_parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Project root directory (defaults to auto-detect)",
    )
    dataset_diff_parser.add_argument(
        "--json",
        action="store_true",
        dest="output_json",
        help="Output diff as JSON instead of formatted text",
    )
    dataset_diff_parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress non-error output",
    )

    # artifact subcommand
    artifact_parser = subparsers.add_parser(
        "artifact",
        help="Inspect and query artifacts in the store",
    )
    artifact_subparsers = artifact_parser.add_subparsers(
        dest="artifact_command",
        required=True,
    )

    # artifact show subcommand
    artifact_show_parser = artifact_subparsers.add_parser(
        "show",
        help="Display detailed information about an artifact",
    )
    artifact_show_parser.add_argument(
        "artifact_id",
        help="Artifact ID to display",
    )
    artifact_show_parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Project root directory (defaults to auto-detect)",
    )
    artifact_show_parser.add_argument(
        "--json",
        action="store_true",
        dest="output_json",
        help="Output raw JSON manifest instead of formatted text",
    )
    artifact_show_parser.add_argument(
        "--content",
        action="store_true",
        help="Also display artifact content (if text-based)",
    )
    artifact_show_parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify artifact integrity by checking content hash",
    )
    artifact_show_parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress non-error output",
    )

    # artifact list subcommand
    artifact_list_parser = artifact_subparsers.add_parser(
        "list",
        help="List artifacts in the store with optional filtering",
    )
    artifact_list_parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Project root directory (defaults to auto-detect)",
    )
    artifact_list_parser.add_argument(
        "--type",
        "-t",
        type=str,
        dest="artifact_type",
        help="Filter by artifact type (e.g., touchstone, coupon_spec)",
    )
    artifact_list_parser.add_argument(
        "--run",
        "-r",
        type=str,
        dest="run_id",
        help="Filter by run ID",
    )
    artifact_list_parser.add_argument(
        "--role",
        type=str,
        action="append",
        dest="roles",
        default=[],
        help="Filter by role (can be specified multiple times)",
    )
    artifact_list_parser.add_argument(
        "--after",
        type=str,
        dest="created_after",
        help="Filter to artifacts created after this timestamp (ISO 8601)",
    )
    artifact_list_parser.add_argument(
        "--before",
        type=str,
        dest="created_before",
        help="Filter to artifacts created before this timestamp (ISO 8601)",
    )
    artifact_list_parser.add_argument(
        "--limit",
        "-n",
        type=int,
        default=None,
        help="Maximum number of artifacts to show",
    )
    artifact_list_parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Number of artifacts to skip (for pagination)",
    )
    artifact_list_parser.add_argument(
        "--order-by",
        type=str,
        choices=["created_utc", "artifact_id", "byte_size"],
        default="created_utc",
        help="Field to order results by (default: created_utc)",
    )
    artifact_list_parser.add_argument(
        "--asc",
        action="store_true",
        dest="order_asc",
        help="Order ascending instead of descending",
    )
    artifact_list_parser.add_argument(
        "--json",
        action="store_true",
        dest="output_json",
        help="Output as JSON instead of formatted table",
    )
    artifact_list_parser.add_argument(
        "--long",
        "-l",
        action="store_true",
        dest="long_format",
        help="Show detailed information for each artifact",
    )
    artifact_list_parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress non-error output (only show artifact IDs)",
    )

    # run subcommand
    run_parser = subparsers.add_parser(
        "run",
        help="Execute a DVC stage with metadata stamping and artifact tracking",
    )
    run_parser.add_argument(
        "stage",
        help="DVC stage name to run (e.g., 'generate_coupon', 'run_drc')",
    )
    run_parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Project root directory (defaults to auto-detect)",
    )
    run_parser.add_argument(
        "--run-type",
        type=str,
        default=None,
        help="Run type classification (auto-detected if not provided)",
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be run without executing",
    )
    run_parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force re-run even if stage is cached",
    )
    run_parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress non-error output",
    )
    run_parser.add_argument(
        "--tag",
        "-t",
        action="append",
        default=[],
        dest="tags",
        metavar="KEY=VALUE",
        help="Add a tag (can be specified multiple times)",
    )

    # gc subcommand
    gc_parser = subparsers.add_parser(
        "gc",
        help="Garbage collect old artifacts with configurable retention policies",
    )
    gc_parser.add_argument(
        "--policy",
        "-p",
        type=str,
        default="laptop_default",
        help="Retention policy to use (default: laptop_default)",
    )
    gc_parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Project root directory (defaults to auto-detect)",
    )
    gc_parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Show what would be deleted without deleting (default)",
    )
    gc_parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete artifacts (disables dry-run)",
    )
    gc_parser.add_argument(
        "--no-dvc",
        action="store_true",
        help="Skip running dvc gc",
    )
    gc_parser.add_argument(
        "--list-policies",
        action="store_true",
        help="List available retention policies and exit",
    )
    gc_parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress non-error output",
    )
    gc_parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )

    # gc pin subcommand
    gc_pin_parser = subparsers.add_parser(
        "gc-pin",
        help="Pin an artifact to protect it from garbage collection",
    )
    gc_pin_parser.add_argument(
        "--artifact-id",
        type=str,
        help="Artifact ID to pin",
    )
    gc_pin_parser.add_argument(
        "--run-id",
        type=str,
        help="Run ID to pin (protects all artifacts from this run)",
    )
    gc_pin_parser.add_argument(
        "--dataset-id",
        type=str,
        help="Dataset ID to pin (protects all artifacts in this dataset)",
    )
    gc_pin_parser.add_argument(
        "--reason",
        type=str,
        help="Reason for pinning (for documentation)",
    )
    gc_pin_parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Project root directory",
    )

    # gc unpin subcommand
    gc_unpin_parser = subparsers.add_parser(
        "gc-unpin",
        help="Unpin an artifact to allow garbage collection",
    )
    gc_unpin_parser.add_argument(
        "--artifact-id",
        type=str,
        help="Artifact ID to unpin",
    )
    gc_unpin_parser.add_argument(
        "--run-id",
        type=str,
        help="Run ID to unpin",
    )
    gc_unpin_parser.add_argument(
        "--dataset-id",
        type=str,
        help="Dataset ID to unpin",
    )
    gc_unpin_parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Project root directory",
    )

    # gc estimate subcommand
    gc_estimate_parser = subparsers.add_parser(
        "gc-estimate",
        help="Estimate space savings from garbage collection",
    )
    gc_estimate_parser.add_argument(
        "--policy",
        "-p",
        type=str,
        default="laptop_default",
        help="Retention policy to use (default: laptop_default)",
    )
    gc_estimate_parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Project root directory",
    )
    gc_estimate_parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )

    # audit subcommand
    audit_parser = subparsers.add_parser(
        "audit",
        help="Generate deterministic provenance report for artifacts",
    )
    audit_parser.add_argument(
        "artifact_id",
        nargs="?",
        default=None,
        help="Artifact ID to audit (if not provided, audits all artifacts)",
    )
    audit_parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Project root directory (defaults to auto-detect)",
    )
    audit_parser.add_argument(
        "--format",
        type=str,
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    audit_parser.add_argument(
        "--trace-roots",
        action="store_true",
        help="Trace lineage to root artifacts",
    )
    audit_parser.add_argument(
        "--verify-hashes",
        action="store_true",
        help="Verify content hashes for artifacts",
    )
    audit_parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Maximum depth for ancestor traversal",
    )
    audit_parser.add_argument(
        "--required-roles",
        type=str,
        default=None,
        help="Comma-separated list of roles that must exist in roots",
    )
    audit_parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress non-error output",
    )

    # verify subcommand
    verify_parser = subparsers.add_parser(
        "verify",
        help="Verify artifact integrity: hashes, lineage, and metadata",
    )
    verify_parser.add_argument(
        "artifact_id",
        nargs="?",
        default=None,
        help="Artifact ID to verify (if not provided, verifies all artifacts)",
    )
    verify_parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Project root directory (defaults to auto-detect)",
    )
    verify_parser.add_argument(
        "--format",
        type=str,
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    verify_parser.add_argument(
        "--hash",
        action="store_true",
        dest="check_hash",
        help="Verify content hashes (default: on)",
    )
    verify_parser.add_argument(
        "--no-hash",
        action="store_true",
        dest="skip_hash",
        help="Skip content hash verification",
    )
    verify_parser.add_argument(
        "--lineage",
        action="store_true",
        dest="check_lineage",
        help="Verify lineage consistency (check all referenced inputs exist)",
    )
    verify_parser.add_argument(
        "--manifest",
        action="store_true",
        dest="check_manifest",
        help="Validate manifest against artifact schema",
    )
    verify_parser.add_argument(
        "--registry",
        action="store_true",
        dest="check_registry",
        help="Verify registry consistency (ensure registry matches manifests)",
    )
    verify_parser.add_argument(
        "--full",
        action="store_true",
        help="Run all verification checks (hash, lineage, manifest, registry)",
    )
    verify_parser.add_argument(
        "--repair",
        action="store_true",
        help="Attempt to repair issues (re-index missing registry entries)",
    )
    verify_parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress non-error output",
    )

    return parser


def _log(msg: str, quiet: bool = False) -> None:
    """Print a message unless quiet mode is enabled."""
    if not quiet:
        sys.stdout.write(msg + "\n")


def _now_utc_iso() -> str:
    """Get current UTC time in ISO 8601 format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# =============================================================================
# Run metadata dataclasses
# =============================================================================


@dataclass
class GitInfo:
    """Git repository information for provenance tracking."""

    commit: str
    branch: str | None
    dirty: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "git_commit": self.commit,
            "git_branch": self.branch,
            "git_dirty": self.dirty,
        }


@dataclass
class ToolVersion:
    """Tool version information."""

    name: str
    version: str
    commit_sha: str | None = None
    docker_image: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        result: dict[str, Any] = {"name": self.name, "version": self.version}
        if self.commit_sha:
            result["commit_sha"] = self.commit_sha
        if self.docker_image:
            result["docker_image"] = self.docker_image
        return result


@dataclass
class RunProvenance:
    """Provenance information for a run."""

    hostname: str
    git_commit: str
    git_branch: str | None = None
    git_dirty: bool = False
    username: str | None = None
    toolchain: list[ToolVersion] = field(default_factory=list)
    python_version: str | None = None
    command: str | None = None
    working_directory: str | None = None
    ci_run_id: str | None = None
    ci_job_url: str | None = None
    container_digest: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict conforming to run.v1 schema."""
        result: dict[str, Any] = {
            "hostname": self.hostname,
            "git_commit": self.git_commit,
        }
        if self.git_branch:
            result["git_branch"] = self.git_branch
        result["git_dirty"] = self.git_dirty
        if self.username:
            result["username"] = self.username
        if self.toolchain:
            result["toolchain"] = [t.to_dict() for t in self.toolchain]
        if self.python_version:
            result["python_version"] = self.python_version
        if self.command:
            result["command"] = self.command
        if self.working_directory:
            result["working_directory"] = self.working_directory
        if self.ci_run_id:
            result["ci_run_id"] = self.ci_run_id
        if self.ci_job_url:
            result["ci_job_url"] = self.ci_job_url
        if self.container_digest:
            result["container_digest"] = self.container_digest
        return result


@dataclass
class RunInputs:
    """Input artifacts for a run."""

    artifacts: list[dict[str, Any]] = field(default_factory=list)
    dataset_id: str | None = None
    dataset_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        result: dict[str, Any] = {"artifacts": self.artifacts}
        if self.dataset_id:
            result["dataset_id"] = self.dataset_id
        if self.dataset_version:
            result["dataset_version"] = self.dataset_version
        return result


@dataclass
class RunOutputs:
    """Output artifacts from a run."""

    artifacts: list[dict[str, Any]] = field(default_factory=list)
    total_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "artifacts": self.artifacts,
            "total_bytes": self.total_bytes,
        }


@dataclass
class StageExecution:
    """Information about a single stage execution."""

    stage_name: str
    status: str
    started_utc: str
    finished_utc: str | None = None
    duration_seconds: float | None = None
    cached: bool = False
    cache_hit_reason: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        result: dict[str, Any] = {
            "stage_name": self.stage_name,
            "status": self.status,
            "started_utc": self.started_utc,
        }
        if self.finished_utc:
            result["finished_utc"] = self.finished_utc
        if self.duration_seconds is not None:
            result["duration_seconds"] = self.duration_seconds
        result["cached"] = self.cached
        if self.cache_hit_reason:
            result["cache_hit_reason"] = self.cache_hit_reason
        if self.error_message:
            result["error_message"] = self.error_message
        return result


@dataclass
class RunMetadata:
    """Full run metadata conforming to run.v1 schema."""

    run_id: str
    run_type: str
    status: str
    started_utc: str
    provenance: RunProvenance
    inputs: RunInputs
    outputs: RunOutputs
    name: str | None = None
    description: str | None = None
    finished_utc: str | None = None
    duration_seconds: float | None = None
    stages: list[StageExecution] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    dvc_stage_hash: str | None = None
    tags: dict[str, str] = field(default_factory=dict)
    annotations: dict[str, Any] = field(default_factory=dict)
    log_paths: list[str] = field(default_factory=list)
    error: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict conforming to run.v1 schema."""
        result: dict[str, Any] = {
            "schema_version": 1,
            "run_id": self.run_id,
            "run_type": self.run_type,
            "status": self.status,
            "started_utc": self.started_utc,
            "provenance": self.provenance.to_dict(),
            "inputs": self.inputs.to_dict(),
            "outputs": self.outputs.to_dict(),
        }
        if self.name:
            result["name"] = self.name
        if self.description:
            result["description"] = self.description
        if self.finished_utc:
            result["finished_utc"] = self.finished_utc
        if self.duration_seconds is not None:
            result["duration_seconds"] = self.duration_seconds
        if self.stages:
            result["stages"] = [s.to_dict() for s in self.stages]
        if self.parameters:
            result["parameters"] = self.parameters
        if self.dvc_stage_hash:
            result["dvc_stage_hash"] = self.dvc_stage_hash
        if self.tags:
            result["tags"] = self.tags
        if self.annotations:
            result["annotations"] = self.annotations
        if self.log_paths:
            result["log_paths"] = self.log_paths
        if self.error:
            result["error"] = self.error
        return result

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


# =============================================================================
# Git and environment helpers
# =============================================================================


def _get_git_info(cwd: Path) -> GitInfo | None:
    """Get git repository information.

    Args:
        cwd: Working directory to run git commands in.

    Returns:
        GitInfo if in a git repo, None otherwise.
    """
    git_marker = cwd / ".git"
    if not git_marker.exists():
        return None
    try:
        # Get current commit SHA
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        )
        commit = result.stdout.strip()

        # Get current branch
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
        branch = result.stdout.strip() if result.returncode == 0 else None

        # Check if dirty
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
        dirty = bool(result.stdout.strip()) if result.returncode == 0 else False

        return GitInfo(commit=commit, branch=branch, dirty=dirty)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _get_python_version() -> str:
    """Get the current Python version string."""
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def _get_dvc_version() -> str | None:
    """Get DVC version if available."""
    if not shutil.which("dvc"):
        return None
    try:
        result = subprocess.run(
            ["dvc", "version", "--json"],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(result.stdout)
        return data.get("dvc_version", data.get("DVC version"))
    except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError):
        # Fallback: try plain version
        try:
            result = subprocess.run(
                ["dvc", "--version"],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip().split()[-1]
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None


def _get_container_digest() -> str | None:
    """Get container image digest if running inside a container.

    Checks for /.dockerenv and reads /proc/self/cgroup to detect container.
    Returns the container ID if found.
    """
    # Check if running in Docker
    if Path("/.dockerenv").exists():
        # Try to read container ID from cgroup
        try:
            cgroup_path = Path("/proc/self/cgroup")
            if cgroup_path.exists():
                content = cgroup_path.read_text()
                for line in content.splitlines():
                    # Docker container IDs are typically 64 hex chars
                    parts = line.split("/")
                    for part in reversed(parts):
                        if len(part) == 64 and all(c in "0123456789abcdef" for c in part):
                            return part[:12]  # Return short form
        except (OSError, PermissionError):
            pass
        return "unknown-container"

    # Check for Kubernetes pod
    if os.environ.get("KUBERNETES_SERVICE_HOST"):
        return os.environ.get("HOSTNAME", "k8s-pod")

    return None


def _get_dvc_lock_hash(cwd: Path) -> str | None:
    """Compute hash of dvc.lock file for provenance tracking.

    The dvc.lock file captures the complete pipeline state including
    all dependencies, parameters, and output hashes. Hashing it provides
    a single value that represents the entire pipeline reproducibility.

    Args:
        cwd: Working directory containing dvc.lock.

    Returns:
        SHA256 hash of dvc.lock content, or None if file doesn't exist.
    """
    import hashlib

    lock_path = cwd / "dvc.lock"
    if not lock_path.exists():
        return None

    try:
        content = lock_path.read_bytes()
        return hashlib.sha256(content).hexdigest()
    except (OSError, PermissionError):
        return None


def _get_environment_stamp() -> dict[str, Any]:
    """Capture environment variables relevant for reproducibility.

    Returns a dictionary of environment variables that affect pipeline
    execution, excluding sensitive values like API keys.
    """
    # Variables relevant for reproducibility
    relevant_vars = [
        "PYTHONHASHSEED",
        "PYTHONDONTWRITEBYTECODE",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "CUDA_VISIBLE_DEVICES",
        "TF_CPP_MIN_LOG_LEVEL",
        "TORCH_CUDA_ARCH_LIST",
        "MLFLOW_TRACKING_URI",
        "DVC_NO_ANALYTICS",
    ]

    env_stamp: dict[str, Any] = {}
    for var in relevant_vars:
        value = os.environ.get(var)
        if value is not None:
            env_stamp[var] = value

    return env_stamp


def _write_run_json(
    run_dir: Path,
    metadata: RunMetadata,
    quiet: bool = False,
) -> Path:
    """Write run metadata to runs/<run_id>/run.json.

    This function creates the run directory and writes the metadata file.
    It's called both at the start of a run (for resumability) and at the
    end (with final status).

    Args:
        run_dir: Directory for this run (typically data/runs/<run_id>).
        metadata: RunMetadata to serialize.
        quiet: If True, suppress logging.

    Returns:
        Path to the written run.json file.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    run_json_path = run_dir / "run.json"

    # Write atomically using temp file + rename
    import tempfile

    fd, tmp_path_str = tempfile.mkstemp(
        suffix=".tmp",
        prefix="run.json.",
        dir=run_dir,
    )
    tmp_path = Path(tmp_path_str)

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(metadata.to_json())
            f.flush()
            os.fsync(f.fileno())

        # Atomic rename
        tmp_path.rename(run_json_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    _log(f"Wrote run metadata to {run_json_path}", quiet)
    return run_json_path


def _infer_run_type(stage: str) -> str:
    """Infer run type from stage name.

    Args:
        stage: DVC stage name.

    Returns:
        A run_type enum value.
    """
    stage_lower = stage.lower()

    if "coupon" in stage_lower or "generate" in stage_lower:
        return "coupon_generation"
    if "drc" in stage_lower:
        return "drc_validation"
    if "em" in stage_lower or "simulation" in stage_lower or "sim" in stage_lower:
        return "em_simulation"
    if "dataset" in stage_lower:
        return "dataset_build"
    if "train" in stage_lower:
        return "model_training"
    if "eval" in stage_lower:
        return "model_evaluation"
    # Check validation before formula_discovery to correctly match "validate_formula"
    if "valid" in stage_lower:
        return "formula_validation"
    if "formula" in stage_lower or "discovery" in stage_lower:
        return "formula_discovery"
    if "export" in stage_lower or "gerber" in stage_lower:
        return "export"
    if "gc" in stage_lower:
        return "gc_sweep"
    if "verify" in stage_lower or "integrity" in stage_lower:
        return "integrity_check"

    return "other"


def _parse_tags(tag_args: list[str]) -> dict[str, str]:
    """Parse tag arguments in KEY=VALUE format.

    Args:
        tag_args: List of strings in KEY=VALUE format.

    Returns:
        Dictionary of tag key-value pairs.
    """
    tags: dict[str, str] = {}
    for tag in tag_args:
        if "=" in tag:
            key, value = tag.split("=", 1)
            tags[key.strip()] = value.strip()
    return tags


def _run_dvc_stage(
    stage: str,
    cwd: Path,
    force: bool = False,
) -> tuple[int, str, str, bool]:
    """Execute a DVC stage via dvc repro.

    Args:
        stage: DVC stage name to run.
        cwd: Working directory for DVC command.
        force: If True, force re-run even if cached.

    Returns:
        Tuple of (return_code, stdout, stderr, was_cached).
    """
    cmd = ["dvc", "repro", stage]
    if force:
        cmd.append("--force")

    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )

    # Detect if stage was cached (DVC outputs specific messages)
    was_cached = (
        "didn't change" in result.stdout.lower()
        or "Stage" in result.stdout
        and "cached" in result.stdout.lower()
        or result.returncode == 0
        and "Running" not in result.stdout
    )

    return result.returncode, result.stdout, result.stderr, was_cached


def cmd_init(root: Path | None, force: bool, quiet: bool) -> int:
    """Initialize the M3 data directory structure.

    Creates:
    - data/objects/: Content-addressed artifact storage
    - data/manifests/: Artifact manifests (JSON)
    - data/mlflow/: MLflow tracking database and artifacts
    - data/registry.db: SQLite artifact registry

    Also initializes the registry schema and validates configuration.

    Args:
        root: Project root directory (defaults to cwd).
        force: If True, reinitialize even if already initialized.
        quiet: If True, suppress non-error output.

    Returns:
        0 on success, 2 on error.
    """
    project_root = root or Path.cwd()

    # Find project root by looking for markers
    if not root:
        project_root = _find_project_root(project_root)

    _log(f"Initializing M3 in {project_root}", quiet)

    # Check if already initialized
    data_dir = project_root / "data"
    registry_db = data_dir / "registry.db"
    if registry_db.exists() and not force:
        _log("Already initialized. Use --force to reinitialize.", quiet)
        return 0

    # Step 1: Create data directory structure
    _log("Creating data directory structure...", quiet)
    try:
        _create_data_directories(data_dir)
    except OSError as e:
        sys.stderr.write(f"Error creating directories: {e}\n")
        return 2

    # Step 2: Initialize artifact store (ensures directories exist)
    _log("Initializing artifact store...", quiet)
    try:
        _init_artifact_store(data_dir)
    except Exception as e:
        sys.stderr.write(f"Error initializing artifact store: {e}\n")
        return 2

    # Step 3: Load and validate MLflow config, ensure directories
    _log("Configuring MLflow tracking...", quiet)
    try:
        _init_mlflow_config(project_root)
    except Exception as e:
        sys.stderr.write(f"Error configuring MLflow: {e}\n")
        return 2

    # Step 4: Initialize registry
    _log("Initializing artifact registry...", quiet)
    try:
        _init_registry(registry_db)
    except Exception as e:
        sys.stderr.write(f"Error initializing registry: {e}\n")
        return 2

    # Step 5: Verify DVC configuration exists
    _log("Verifying DVC configuration...", quiet)
    dvc_dir = project_root / ".dvc"
    if not dvc_dir.exists():
        _log("Warning: DVC not initialized. Run 'dvc init' to set up DVC.", quiet)
    else:
        _log("DVC configuration found.", quiet)

    # Step 6: Record initialization
    _log("Recording initialization...", quiet)
    try:
        _record_init_run(data_dir, project_root)
    except Exception as e:
        sys.stderr.write(f"Error recording initialization: {e}\n")
        return 2

    _log("M3 initialization complete.", quiet)
    _log("", quiet)
    _log("Next steps:", quiet)
    _log("  - Run 'm3 verify' to check data integrity", quiet)
    _log("  - Run 'm3 run <stage>' to execute pipeline stages", quiet)

    return 0


def _find_project_root(start: Path) -> Path:
    """Find project root by looking for markers (pyproject.toml or .git)."""
    current = start.resolve()
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    return start


def _create_data_directories(data_dir: Path) -> None:
    """Create the data directory structure."""
    # Core directories
    (data_dir / "objects").mkdir(parents=True, exist_ok=True)
    (data_dir / "manifests").mkdir(parents=True, exist_ok=True)
    (data_dir / "mlflow" / "artifacts").mkdir(parents=True, exist_ok=True)
    (data_dir / "datasets").mkdir(parents=True, exist_ok=True)
    (data_dir / "runs").mkdir(parents=True, exist_ok=True)


def _init_artifact_store(data_dir: Path) -> None:
    """Initialize the artifact store."""
    from formula_foundry.m3.artifact_store import ArtifactStore

    store = ArtifactStore(
        root=data_dir,
        generator="m3_init",
        generator_version=__version__,
    )
    # This ensures the directories exist
    store._ensure_dirs()


def _init_mlflow_config(project_root: Path) -> None:
    """Initialize MLflow configuration and directories."""
    from formula_foundry.tracking.config import load_mlflow_config

    # Load configuration (uses defaults if config file doesn't exist)
    config = load_mlflow_config(project_root=project_root)

    # Ensure directories exist for SQLite and artifacts
    config.tracking.ensure_directories(project_root)


def _init_registry(registry_db: Path) -> None:
    """Initialize the artifact registry database."""
    from formula_foundry.m3.registry import ArtifactRegistry

    registry = ArtifactRegistry(registry_db)
    registry.initialize()
    registry.close()


def _record_init_run(data_dir: Path, project_root: Path) -> None:
    """Record the initialization as a run in the registry."""
    import socket

    from formula_foundry.m3.registry import ArtifactRegistry

    registry_db = data_dir / "registry.db"
    registry = ArtifactRegistry(registry_db)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    unique_suffix = uuid.uuid4().hex[:8]
    run_id = f"init-{timestamp}-{unique_suffix}"
    hostname = socket.gethostname()

    registry.index_run(
        run_id=run_id,
        started_utc=_now_utc_iso(),
        status="completed",
        stage_name="m3_init",
        ended_utc=_now_utc_iso(),
        hostname=hostname,
        generator="m3_init",
        generator_version=__version__,
        config={"project_root": str(project_root)},
    )
    registry.close()


def cmd_run(
    stage: str,
    root: Path | None,
    run_type: str | None,
    dry_run: bool,
    force: bool,
    quiet: bool,
    tags: list[str],
) -> int:
    """Execute a DVC stage with metadata stamping and artifact tracking.

    This command wraps `dvc repro <stage>` and:
    1. Records run start time and provenance (git commit, branch, container digest)
    2. Executes the DVC stage
    3. Captures stage output and timing
    4. Stores run metadata as an artifact in the M3 store
    5. Indexes the run in the registry

    Args:
        stage: DVC stage name to run.
        root: Project root directory (auto-detected if None).
        run_type: Run type classification (auto-detected if None).
        dry_run: If True, show what would be done without executing.
        force: If True, force re-run even if stage is cached.
        quiet: If True, suppress non-error output.
        tags: List of tags in KEY=VALUE format.

    Returns:
        0 on success, non-zero on error.
    """
    import socket
    import time

    from formula_foundry.m3.artifact_store import ArtifactStore
    from formula_foundry.m3.registry import ArtifactRegistry

    # Find project root
    project_root = root or Path.cwd()
    if not root:
        project_root = _find_project_root(project_root)

    _log(f"M3 Run: stage={stage}, root={project_root}", quiet)

    # Verify M3 is initialized
    data_dir = project_root / "data"
    registry_db = data_dir / "registry.db"
    if not registry_db.exists():
        sys.stderr.write("Error: M3 not initialized. Run 'm3 init' first.\n")
        return 2

    # Verify dvc.yaml exists
    dvc_yaml = project_root / "dvc.yaml"
    if not dvc_yaml.exists():
        sys.stderr.write(f"Error: dvc.yaml not found at {dvc_yaml}\n")
        return 2

    # Gather provenance information
    _log("Gathering provenance information...", quiet)

    git_info = _get_git_info(project_root)
    if git_info is None:
        sys.stderr.write("Error: Not in a git repository. Git commit info is required.\n")
        return 2

    hostname = socket.gethostname()
    username = os.environ.get("USER") or os.environ.get("USERNAME")
    python_version = _get_python_version()
    dvc_version = _get_dvc_version()
    container_digest = _get_container_digest()
    dvc_lock_hash = _get_dvc_lock_hash(project_root)
    environment_stamp = _get_environment_stamp()

    # Build toolchain list
    toolchain: list[ToolVersion] = [ToolVersion(name="python", version=python_version)]
    if dvc_version:
        toolchain.append(ToolVersion(name="dvc", version=dvc_version))
    toolchain.append(ToolVersion(name="m3", version=__version__))

    # Determine run type
    effective_run_type = run_type or _infer_run_type(stage)
    _log(f"Run type: {effective_run_type}", quiet)

    # Parse tags
    parsed_tags = _parse_tags(tags)

    # Generate run ID
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    unique_suffix = uuid.uuid4().hex[:8]
    run_id = f"run-{timestamp}-{unique_suffix}"

    # Build command string for provenance
    command = f"m3 run {stage}"
    if force:
        command += " --force"
    if tags:
        for tag in tags:
            command += f" --tag {tag}"

    # Dry run mode
    if dry_run:
        _log("", quiet)
        _log("Dry run - would execute:", quiet)
        _log(f"  Run ID: {run_id}", quiet)
        _log(f"  Stage: {stage}", quiet)
        _log(f"  Run type: {effective_run_type}", quiet)
        _log(f"  Git commit: {git_info.commit[:12]}", quiet)
        _log(f"  Git branch: {git_info.branch}", quiet)
        _log(f"  Git dirty: {git_info.dirty}", quiet)
        _log(f"  DVC lock hash: {dvc_lock_hash[:12] if dvc_lock_hash else 'none'}...", quiet)
        _log(f"  Container: {container_digest or 'none'}", quiet)
        _log(f"  Command: dvc repro {stage}{' --force' if force else ''}", quiet)
        if parsed_tags:
            _log(f"  Tags: {parsed_tags}", quiet)
        return 0

    # Verify DVC is available (only needed for actual execution)
    if not shutil.which("dvc"):
        sys.stderr.write("Error: DVC not found in PATH.\n")
        return 2

    # Build provenance
    provenance = RunProvenance(
        hostname=hostname,
        git_commit=git_info.commit,
        git_branch=git_info.branch,
        git_dirty=git_info.dirty,
        username=username,
        toolchain=toolchain,
        python_version=python_version,
        command=command,
        working_directory=str(project_root),
        ci_run_id=os.environ.get("CI_JOB_ID") or os.environ.get("GITHUB_RUN_ID"),
        ci_job_url=os.environ.get("CI_JOB_URL") or os.environ.get("GITHUB_SERVER_URL"),
        container_digest=container_digest,
    )

    # Create run metadata
    started_utc = _now_utc_iso()
    start_time = time.time()

    run_metadata = RunMetadata(
        run_id=run_id,
        run_type=effective_run_type,
        status="running",
        started_utc=started_utc,
        provenance=provenance,
        inputs=RunInputs(),
        outputs=RunOutputs(),
        name=f"DVC stage: {stage}",
        tags=parsed_tags,
        dvc_stage_hash=dvc_lock_hash,
        annotations={"environment": environment_stamp} if environment_stamp else {},
    )

    # Create run directory and write initial run.json (for resumability)
    # Per design doc section 15.3, we write run.json early so that on restart
    # we can detect incomplete runs and either resume or recompute
    runs_dir = data_dir / "runs"
    run_dir = runs_dir / run_id
    run_json_path = _write_run_json(run_dir, run_metadata, quiet)

    # Initialize registry and record run start
    registry = ArtifactRegistry(registry_db)
    registry.index_run(
        run_id=run_id,
        started_utc=started_utc,
        status="in_progress",
        stage_name=stage,
        hostname=hostname,
        generator="m3_run",
        generator_version=__version__,
        config={
            "stage": stage,
            "force": force,
            "run_type": effective_run_type,
            "dvc_lock_hash": dvc_lock_hash,
        },
    )

    _log(f"Run ID: {run_id}", quiet)
    _log(f"Git commit: {git_info.commit[:12]} ({git_info.branch})", quiet)
    if dvc_lock_hash:
        _log(f"DVC lock hash: {dvc_lock_hash[:12]}...", quiet)
    if container_digest:
        _log(f"Container: {container_digest}", quiet)
    _log("", quiet)
    _log(f"Running: dvc repro {stage}{'--force' if force else ''}", quiet)

    # Execute DVC stage
    stage_started_utc = _now_utc_iso()
    return_code, stdout, stderr, was_cached = _run_dvc_stage(stage, project_root, force)
    stage_finished_utc = _now_utc_iso()

    # Calculate duration
    end_time = time.time()
    duration_seconds = end_time - start_time
    finished_utc = _now_utc_iso()

    # Determine status
    if return_code == 0:
        status = "completed"
        stage_status = "cached" if was_cached else "completed"
    else:
        status = "failed"
        stage_status = "failed"

    # Log output
    if stdout:
        _log("", quiet)
        _log("DVC output:", quiet)
        for line in stdout.strip().split("\n"):
            _log(f"  {line}", quiet)

    if return_code != 0 and stderr:
        sys.stderr.write("\nDVC errors:\n")
        for line in stderr.strip().split("\n"):
            sys.stderr.write(f"  {line}\n")

    # Create stage execution record
    stage_execution = StageExecution(
        stage_name=stage,
        status=stage_status,
        started_utc=stage_started_utc,
        finished_utc=stage_finished_utc,
        duration_seconds=duration_seconds,
        cached=was_cached,
        cache_hit_reason="DVC cache hit" if was_cached else None,
        error_message=stderr if return_code != 0 else None,
    )

    # Update run metadata
    run_metadata.status = status
    run_metadata.finished_utc = finished_utc
    run_metadata.duration_seconds = duration_seconds
    run_metadata.stages = [stage_execution]

    # Capture DVC lock hash after run (may have changed if stage modified outputs)
    dvc_lock_hash_after = _get_dvc_lock_hash(project_root)
    if dvc_lock_hash_after and dvc_lock_hash_after != dvc_lock_hash:
        run_metadata.dvc_stage_hash = dvc_lock_hash_after
        _log(f"DVC lock updated: {dvc_lock_hash_after[:12]}...", quiet)

    if return_code != 0:
        run_metadata.error = {
            "error_type": "DVCError",
            "message": stderr.strip() if stderr else "DVC stage failed",
            "stage_name": stage,
            "recoverable": True,
        }

    # Record log paths
    run_metadata.log_paths = [str(run_json_path.relative_to(project_root))]

    # Update run.json with final status (for resumability/auditing)
    _write_run_json(run_dir, run_metadata, quiet)

    # Update run status in registry
    registry.update_run_status(
        run_id=run_id,
        status=status,
        ended_utc=finished_utc,
    )

    # Store run metadata as artifact
    _log("", quiet)
    _log("Storing run metadata...", quiet)

    store = ArtifactStore(
        root=data_dir,
        generator="m3_run",
        generator_version=__version__,
    )

    metadata_json = run_metadata.to_json()
    manifest = store.put(
        content=metadata_json.encode("utf-8"),
        artifact_type="log",
        roles=["metadata"],
        run_id=run_id,
        artifact_id=f"{run_id}-metadata",
        stage_name=stage,
        media_type="application/json",
        tags={"run_id": run_id, "stage": stage, "status": status},
    )

    # Index the metadata artifact
    registry.index_artifact(manifest)
    registry.close()

    _log(f"Metadata artifact: {manifest.artifact_id}", quiet)
    _log("", quiet)

    if return_code == 0:
        _log(f"Run completed in {duration_seconds:.2f}s", quiet)
        if was_cached:
            _log("(Stage outputs were cached)", quiet)
    else:
        _log(f"Run failed after {duration_seconds:.2f}s", quiet)

    return return_code


# =============================================================================
# Dataset commands
# =============================================================================


def _format_bytes(byte_size: int) -> str:
    """Format byte size in human-readable form."""
    if byte_size < 1024:
        return f"{byte_size} B"
    if byte_size < 1024 * 1024:
        return f"{byte_size / 1024:.1f} KB"
    if byte_size < 1024 * 1024 * 1024:
        return f"{byte_size / (1024 * 1024):.1f} MB"
    return f"{byte_size / (1024 * 1024 * 1024):.2f} GB"


def _resolve_dataset_path(
    dataset_id: str,
    version: str | None,
    project_root: Path,
) -> Path | None:
    """Resolve a dataset ID or path to a manifest file path.

    Args:
        dataset_id: Dataset ID or direct path to manifest.
        version: Optional version string.
        project_root: Project root directory.

    Returns:
        Path to the dataset manifest, or None if not found.
    """
    # Check if dataset_id is a direct path
    path = Path(dataset_id)
    if path.exists() and path.suffix == ".json":
        return path

    # Look in data/datasets/
    datasets_dir = project_root / "data" / "datasets"
    if not datasets_dir.exists():
        return None

    if version:
        # Look for exact version match
        manifest_path = datasets_dir / f"{dataset_id}_{version}.json"
        if manifest_path.exists():
            return manifest_path
    else:
        # Find all versions and return the most recent
        pattern = f"{dataset_id}_*.json"
        matches = sorted(datasets_dir.glob(pattern))
        # Filter out parquet index files
        matches = [m for m in matches if not m.stem.endswith("_index")]
        if matches:
            return matches[-1]  # Most recent by sort order

    return None


def cmd_dataset_show(
    dataset_id: str,
    version: str | None,
    root: Path | None,
    output_json: bool,
    show_members: bool,
    quiet: bool,
) -> int:
    """Display detailed information about a dataset snapshot.

    Args:
        dataset_id: Dataset ID or path to manifest.
        version: Optional version string.
        root: Project root directory.
        output_json: If True, output raw JSON.
        show_members: If True, show detailed member list.
        quiet: If True, suppress non-error output.

    Returns:
        0 on success, non-zero on error.
    """
    from formula_foundry.m3.dataset_snapshot import (
        DatasetNotFoundError,
        DatasetSnapshotReader,
    )

    # Find project root
    project_root = root or Path.cwd()
    if not root:
        project_root = _find_project_root(project_root)

    # Resolve dataset path
    manifest_path = _resolve_dataset_path(dataset_id, version, project_root)
    if manifest_path is None:
        sys.stderr.write(f"Error: Dataset not found: {dataset_id}")
        if version:
            sys.stderr.write(f" (version {version})")
        sys.stderr.write("\n")
        sys.stderr.write("Hint: Check data/datasets/ or provide a path to a JSON manifest.\n")
        return 2

    # Load the snapshot
    try:
        reader = DatasetSnapshotReader(snapshot_path=manifest_path)
        snapshot = reader.load()
    except DatasetNotFoundError as e:
        sys.stderr.write(f"Error: {e}\n")
        return 2
    except Exception as e:
        sys.stderr.write(f"Error loading dataset: {e}\n")
        return 2

    # Output JSON if requested
    if output_json:
        sys.stdout.write(snapshot.to_json(indent=2) + "\n")
        return 0

    # Formatted output
    lines: list[str] = []
    lines.append(f"Dataset: {snapshot.dataset_id}")
    lines.append(f"Version: {snapshot.version}")
    if snapshot.name:
        lines.append(f"Name: {snapshot.name}")
    if snapshot.description:
        lines.append(f"Description: {snapshot.description}")
    lines.append(f"Created: {snapshot.created_utc}")
    lines.append("")

    # Summary
    lines.append("Summary:")
    lines.append(f"  Members: {snapshot.member_count}")
    lines.append(f"  Total size: {_format_bytes(snapshot.total_bytes)}")
    lines.append(f"  Content hash: {snapshot.content_hash.digest[:16]}...")
    if snapshot.parent_version:
        lines.append(f"  Parent version: {snapshot.parent_version}")
    if snapshot.index_path:
        lines.append(f"  Parquet index: {snapshot.index_path}")
    lines.append("")

    # Provenance
    lines.append("Provenance:")
    lines.append(f"  Generator: {snapshot.provenance.generator}")
    lines.append(f"  Version: {snapshot.provenance.generator_version}")
    if snapshot.provenance.git_commit:
        lines.append(f"  Git commit: {snapshot.provenance.git_commit[:12]}")
    if snapshot.provenance.pipeline_stage:
        lines.append(f"  Pipeline stage: {snapshot.provenance.pipeline_stage}")
    if snapshot.provenance.source_runs:
        lines.append(f"  Source runs: {len(snapshot.provenance.source_runs)}")
    lines.append("")

    # Statistics
    if snapshot.statistics:
        stats = snapshot.statistics
        lines.append("Statistics:")
        if stats.by_artifact_type:
            lines.append("  By artifact type:")
            for atype, info in stats.by_artifact_type.items():
                lines.append(f"    {atype}: {info.get('count', 0)} ({_format_bytes(info.get('total_bytes', 0))})")
        if stats.unique_coupons:
            lines.append(f"  Unique coupons: {stats.unique_coupons}")
        if stats.frequency_range_hz:
            fmin = stats.frequency_range_hz.get("min", 0)
            fmax = stats.frequency_range_hz.get("max", 0)
            lines.append(f"  Frequency range: {fmin / 1e6:.1f} MHz - {fmax / 1e9:.1f} GHz")
        if stats.parameter_ranges:
            lines.append("  Parameter ranges:")
            for param, ranges in stats.parameter_ranges.items():
                pmin = ranges.get("min", 0)
                pmax = ranges.get("max", 0)
                lines.append(f"    {param}: {pmin} - {pmax}")
        lines.append("")

    # Splits
    if snapshot.splits:
        lines.append("Splits:")
        for split_name, split_def in snapshot.splits.items():
            lines.append(f"  {split_name}: {split_def.count} members ({split_def.fraction * 100:.1f}%)")
        lines.append("")

    # Tags
    if snapshot.tags:
        lines.append("Tags:")
        for key, value in snapshot.tags.items():
            lines.append(f"  {key}: {value}")
        lines.append("")

    # Members (if requested)
    if show_members and snapshot.members:
        lines.append("Members:")
        for i, member in enumerate(snapshot.members, 1):
            lines.append(f"  [{i}] {member.artifact_id}")
            lines.append(f"      Type: {member.artifact_type}, Role: {member.role}")
            lines.append(f"      Size: {_format_bytes(member.byte_size)}, Hash: {member.content_hash.digest[:12]}...")
            if member.storage_path:
                lines.append(f"      Path: {member.storage_path}")
        lines.append("")

    # Output
    for line in lines:
        _log(line, quiet)

    return 0


def cmd_dataset_diff(
    dataset_a: str,
    dataset_b: str,
    root: Path | None,
    output_json: bool,
    quiet: bool,
) -> int:
    """Compare two dataset versions and show differences.

    Args:
        dataset_a: First dataset ID or path (the 'from' version).
        dataset_b: Second dataset ID or path (the 'to' version).
        root: Project root directory.
        output_json: If True, output diff as JSON.
        quiet: If True, suppress non-error output.

    Returns:
        0 on success (including no differences), non-zero on error.
    """
    from formula_foundry.m3.dataset_snapshot import (
        DatasetNotFoundError,
        DatasetSnapshotReader,
    )

    # Find project root
    project_root = root or Path.cwd()
    if not root:
        project_root = _find_project_root(project_root)

    # Resolve dataset paths
    path_a = _resolve_dataset_path(dataset_a, None, project_root)
    if path_a is None:
        sys.stderr.write(f"Error: Dataset not found: {dataset_a}\n")
        return 2

    path_b = _resolve_dataset_path(dataset_b, None, project_root)
    if path_b is None:
        sys.stderr.write(f"Error: Dataset not found: {dataset_b}\n")
        return 2

    # Load snapshots
    try:
        reader_a = DatasetSnapshotReader(snapshot_path=path_a)
        snapshot_a = reader_a.load()
        reader_b = DatasetSnapshotReader(snapshot_path=path_b)
        snapshot_b = reader_b.load()
    except DatasetNotFoundError as e:
        sys.stderr.write(f"Error: {e}\n")
        return 2
    except Exception as e:
        sys.stderr.write(f"Error loading datasets: {e}\n")
        return 2

    # Build member sets by artifact_id
    members_a = {m.artifact_id: m for m in snapshot_a.members}
    members_b = {m.artifact_id: m for m in snapshot_b.members}

    ids_a = set(members_a.keys())
    ids_b = set(members_b.keys())

    # Compute differences
    added_ids = ids_b - ids_a
    removed_ids = ids_a - ids_b
    common_ids = ids_a & ids_b

    # Check for modified members (same ID but different hash)
    modified_ids: list[str] = []
    for artifact_id in common_ids:
        if members_a[artifact_id].content_hash.digest != members_b[artifact_id].content_hash.digest:
            modified_ids.append(artifact_id)

    unchanged_count = len(common_ids) - len(modified_ids)

    # Build diff result
    diff_result: dict[str, Any] = {
        "from": {
            "dataset_id": snapshot_a.dataset_id,
            "version": snapshot_a.version,
            "member_count": snapshot_a.member_count,
            "total_bytes": snapshot_a.total_bytes,
        },
        "to": {
            "dataset_id": snapshot_b.dataset_id,
            "version": snapshot_b.version,
            "member_count": snapshot_b.member_count,
            "total_bytes": snapshot_b.total_bytes,
        },
        "summary": {
            "added": len(added_ids),
            "removed": len(removed_ids),
            "modified": len(modified_ids),
            "unchanged": unchanged_count,
        },
        "added": sorted(added_ids),
        "removed": sorted(removed_ids),
        "modified": sorted(modified_ids),
    }

    # Output JSON if requested
    if output_json:
        sys.stdout.write(json.dumps(diff_result, indent=2) + "\n")
        return 0

    # Formatted output
    lines: list[str] = []
    lines.append("Dataset Comparison")
    lines.append("=" * 50)
    lines.append("")
    lines.append(
        f"From: {snapshot_a.dataset_id} {snapshot_a.version} "
        f"({snapshot_a.member_count} members, {_format_bytes(snapshot_a.total_bytes)})"
    )
    lines.append(
        f"To:   {snapshot_b.dataset_id} {snapshot_b.version} "
        f"({snapshot_b.member_count} members, {_format_bytes(snapshot_b.total_bytes)})"
    )
    lines.append("")

    # Summary
    lines.append("Summary:")
    lines.append(f"  Added:     {len(added_ids)}")
    lines.append(f"  Removed:   {len(removed_ids)}")
    lines.append(f"  Modified:  {len(modified_ids)}")
    lines.append(f"  Unchanged: {unchanged_count}")
    lines.append("")

    # Check for hash match
    if snapshot_a.content_hash.digest == snapshot_b.content_hash.digest:
        lines.append("Datasets are identical (same content hash).")
        lines.append("")
    elif not added_ids and not removed_ids and not modified_ids:
        lines.append("No member differences detected.")
        lines.append("")
    else:
        # Show details
        if added_ids:
            lines.append(f"Added ({len(added_ids)}):")
            for artifact_id in sorted(added_ids):
                member = members_b[artifact_id]
                lines.append(f"  + {artifact_id}")
                lines.append(f"    Type: {member.artifact_type}, Size: {_format_bytes(member.byte_size)}")
            lines.append("")

        if removed_ids:
            lines.append(f"Removed ({len(removed_ids)}):")
            for artifact_id in sorted(removed_ids):
                member = members_a[artifact_id]
                lines.append(f"  - {artifact_id}")
                lines.append(f"    Type: {member.artifact_type}, Size: {_format_bytes(member.byte_size)}")
            lines.append("")

        if modified_ids:
            lines.append(f"Modified ({len(modified_ids)}):")
            for artifact_id in sorted(modified_ids):
                old_member = members_a[artifact_id]
                new_member = members_b[artifact_id]
                lines.append(f"  ~ {artifact_id}")
                lines.append(f"    Old hash: {old_member.content_hash.digest[:12]}... ({_format_bytes(old_member.byte_size)})")
                lines.append(f"    New hash: {new_member.content_hash.digest[:12]}... ({_format_bytes(new_member.byte_size)})")
            lines.append("")

    # Size delta
    size_delta = snapshot_b.total_bytes - snapshot_a.total_bytes
    if size_delta > 0:
        lines.append(f"Size change: +{_format_bytes(size_delta)}")
    elif size_delta < 0:
        lines.append(f"Size change: -{_format_bytes(abs(size_delta))}")
    else:
        lines.append("Size change: none")

    # Output
    for line in lines:
        _log(line, quiet)

    return 0


# =============================================================================
# Artifact commands
# =============================================================================


def cmd_artifact_show(
    artifact_id: str,
    root: Path | None,
    output_json: bool,
    show_content: bool,
    verify: bool,
    quiet: bool,
) -> int:
    """Display detailed information about an artifact.

    Args:
        artifact_id: The artifact ID to display.
        root: Project root directory.
        output_json: If True, output raw JSON manifest.
        show_content: If True, also display artifact content.
        verify: If True, verify artifact integrity.
        quiet: If True, suppress non-error output.

    Returns:
        0 on success, non-zero on error.
    """
    from formula_foundry.m3.artifact_store import (
        ArtifactNotFoundError,
        ArtifactStore,
    )

    # Find project root
    project_root = root or Path.cwd()
    if not root:
        project_root = _find_project_root(project_root)

    # Verify M3 is initialized
    data_dir = project_root / "data"
    if not data_dir.exists():
        sys.stderr.write("Error: M3 not initialized. Run 'm3 init' first.\n")
        return 2

    # Get artifact from store
    store = ArtifactStore(
        root=data_dir,
        generator="m3_cli",
        generator_version=__version__,
    )

    try:
        manifest = store.get_manifest(artifact_id)
    except ArtifactNotFoundError:
        sys.stderr.write(f"Error: Artifact not found: {artifact_id}\n")
        return 2

    # Output JSON if requested
    if output_json:
        sys.stdout.write(manifest.to_json(indent=2) + "\n")
        return 0

    # Verify integrity if requested
    integrity_ok: bool | None = None
    if verify:
        try:
            integrity_ok = store.verify(artifact_id)
        except Exception as e:
            sys.stderr.write(f"Error verifying artifact: {e}\n")
            integrity_ok = False

    # Formatted output
    lines: list[str] = []
    lines.append(f"Artifact: {manifest.artifact_id}")
    lines.append(f"Type: {manifest.artifact_type}")
    lines.append(f"Created: {manifest.created_utc}")
    lines.append("")

    # Content hash and size
    lines.append("Content:")
    lines.append(f"  Algorithm: {manifest.content_hash.algorithm}")
    lines.append(f"  Digest: {manifest.content_hash.digest}")
    lines.append(f"  Spec ID: {manifest.spec_id}")
    lines.append(f"  Size: {_format_bytes(manifest.byte_size)}")
    if manifest.media_type:
        lines.append(f"  Media type: {manifest.media_type}")
    if manifest.storage_path:
        lines.append(f"  Storage path: {manifest.storage_path}")
    lines.append("")

    # Integrity status
    if verify:
        if integrity_ok:
            lines.append("Integrity: VERIFIED (content hash matches)")
        else:
            lines.append("Integrity: FAILED (content hash mismatch)")
        lines.append("")

    # Provenance
    lines.append("Provenance:")
    lines.append(f"  Generator: {manifest.provenance.generator}")
    lines.append(f"  Version: {manifest.provenance.generator_version}")
    lines.append(f"  Hostname: {manifest.provenance.hostname}")
    if manifest.provenance.username:
        lines.append(f"  Username: {manifest.provenance.username}")
    if manifest.provenance.command:
        lines.append(f"  Command: {manifest.provenance.command}")
    if manifest.provenance.working_directory:
        lines.append(f"  Working dir: {manifest.provenance.working_directory}")
    if manifest.provenance.ci_run_id:
        lines.append(f"  CI run ID: {manifest.provenance.ci_run_id}")
    lines.append("")

    # Lineage
    lines.append("Lineage:")
    lines.append(f"  Run ID: {manifest.lineage.run_id}")
    if manifest.lineage.stage_name:
        lines.append(f"  Stage: {manifest.lineage.stage_name}")
    if manifest.lineage.dataset_id:
        lines.append(f"  Dataset ID: {manifest.lineage.dataset_id}")
    if manifest.lineage.dataset_version:
        lines.append(f"  Dataset version: {manifest.lineage.dataset_version}")
    if manifest.lineage.inputs:
        lines.append(f"  Inputs: {len(manifest.lineage.inputs)}")
        for inp in manifest.lineage.inputs[:5]:  # Show first 5
            lines.append(f"    - {inp.artifact_id} ({inp.relation})")
        if len(manifest.lineage.inputs) > 5:
            lines.append(f"    ... and {len(manifest.lineage.inputs) - 5} more")
    if manifest.lineage.outputs:
        lines.append(f"  Outputs: {len(manifest.lineage.outputs)}")
        for out in manifest.lineage.outputs[:5]:  # Show first 5
            lines.append(f"    - {out.artifact_id} ({out.relation})")
        if len(manifest.lineage.outputs) > 5:
            lines.append(f"    ... and {len(manifest.lineage.outputs) - 5} more")
    lines.append("")

    # Roles
    lines.append(f"Roles: {', '.join(manifest.roles)}")

    # Tags
    if manifest.tags:
        lines.append("")
        lines.append("Tags:")
        for key, value in manifest.tags.items():
            lines.append(f"  {key}: {value}")

    # Annotations
    if manifest.annotations:
        lines.append("")
        lines.append("Annotations:")
        for key, value in list(manifest.annotations.items())[:10]:
            value_str = str(value)
            if len(value_str) > 60:
                value_str = value_str[:57] + "..."
            lines.append(f"  {key}: {value_str}")
        if len(manifest.annotations) > 10:
            lines.append(f"  ... and {len(manifest.annotations) - 10} more")

    # Output formatted text
    for line in lines:
        _log(line, quiet)

    # Show content if requested
    if show_content:
        _log("", quiet)
        _log("Content:", quiet)
        _log("-" * 50, quiet)
        try:
            content = store.get(manifest.content_hash.digest)
            # Try to decode as text
            try:
                text = content.decode("utf-8")
                # Limit output for large content
                max_lines = 100
                content_lines = text.split("\n")
                if len(content_lines) > max_lines:
                    for line in content_lines[:max_lines]:
                        _log(line, quiet)
                    _log(f"... ({len(content_lines) - max_lines} more lines)", quiet)
                else:
                    _log(text, quiet)
            except UnicodeDecodeError:
                _log(f"(Binary content, {_format_bytes(len(content))})", quiet)
        except Exception as e:
            sys.stderr.write(f"Error reading content: {e}\n")

    return 0 if integrity_ok is not False else 1


def cmd_artifact_list(
    root: Path | None,
    artifact_type: str | None,
    run_id: str | None,
    roles: list[str],
    created_after: str | None,
    created_before: str | None,
    limit: int | None,
    offset: int,
    order_by: str,
    order_asc: bool,
    output_json: bool,
    long_format: bool,
    quiet: bool,
) -> int:
    """List artifacts in the store with optional filtering.

    Args:
        root: Project root directory.
        artifact_type: Filter by artifact type.
        run_id: Filter by run ID.
        roles: Filter by roles.
        created_after: Filter to artifacts created after this timestamp.
        created_before: Filter to artifacts created before this timestamp.
        limit: Maximum number of artifacts to show.
        offset: Number of artifacts to skip.
        order_by: Field to order by.
        order_asc: If True, order ascending.
        output_json: If True, output as JSON.
        long_format: If True, show detailed info for each artifact.
        quiet: If True, only show artifact IDs.

    Returns:
        0 on success, non-zero on error.
    """
    from formula_foundry.m3.registry import ArtifactRegistry

    # Find project root
    project_root = root or Path.cwd()
    if not root:
        project_root = _find_project_root(project_root)

    # Verify M3 is initialized
    data_dir = project_root / "data"
    registry_db = data_dir / "registry.db"
    if not registry_db.exists():
        sys.stderr.write("Error: M3 not initialized. Run 'm3 init' first.\n")
        return 2

    # Query registry
    registry = ArtifactRegistry(registry_db)

    try:
        records = registry.query_artifacts(
            artifact_type=artifact_type,
            run_id=run_id,
            roles=roles if roles else None,
            created_after=created_after,
            created_before=created_before,
            limit=limit,
            offset=offset,
            order_by=order_by,  # type: ignore[arg-type]
            order_desc=not order_asc,
        )
    finally:
        registry.close()

    # Get total count for summary
    total_count = len(records)
    if limit is not None and total_count >= limit:
        # Reopen to get full count
        registry = ArtifactRegistry(registry_db)
        full_count = registry.count_artifacts(
            artifact_type=artifact_type,
            run_id=run_id,
        )
        registry.close()
    else:
        full_count = total_count + offset

    # Output JSON if requested
    if output_json:
        output_data = {
            "count": len(records),
            "total": full_count,
            "offset": offset,
            "artifacts": [
                {
                    "artifact_id": r.artifact_id,
                    "artifact_type": r.artifact_type,
                    "content_hash": {
                        "algorithm": r.content_hash_algorithm,
                        "digest": r.content_hash_digest,
                    },
                    "byte_size": r.byte_size,
                    "created_utc": r.created_utc,
                    "run_id": r.run_id,
                    "stage_name": r.stage_name,
                    "roles": r.roles,
                    "tags": r.tags,
                }
                for r in records
            ],
        }
        sys.stdout.write(json.dumps(output_data, indent=2) + "\n")
        return 0

    # Quiet mode - just IDs
    if quiet:
        for record in records:
            sys.stdout.write(record.artifact_id + "\n")
        return 0

    # No results
    if not records:
        _log("No artifacts found.", False)
        return 0

    # Formatted output
    if long_format:
        # Detailed view for each artifact
        for i, record in enumerate(records):
            if i > 0:
                _log("", False)
                _log("-" * 50, False)
            _log(f"Artifact: {record.artifact_id}", False)
            _log(f"  Type: {record.artifact_type}", False)
            _log(f"  Created: {record.created_utc}", False)
            _log(f"  Size: {_format_bytes(record.byte_size)}", False)
            _log(f"  Hash: {record.content_hash_digest[:16]}...", False)
            if record.run_id:
                _log(f"  Run: {record.run_id}", False)
            if record.stage_name:
                _log(f"  Stage: {record.stage_name}", False)
            _log(f"  Roles: {', '.join(record.roles)}", False)
            if record.tags:
                tag_str = ", ".join(f"{k}={v}" for k, v in record.tags.items())
                if len(tag_str) > 60:
                    tag_str = tag_str[:57] + "..."
                _log(f"  Tags: {tag_str}", False)
    else:
        # Table view
        _log(f"Found {total_count} artifacts (showing {len(records)}, offset {offset})", False)
        if full_count > total_count + offset:
            _log(f"Total matching: {full_count}", False)
        _log("", False)

        # Header
        _log(f"{'ARTIFACT_ID':<40} {'TYPE':<20} {'SIZE':>10} {'CREATED':<20}", False)
        _log("-" * 94, False)

        for record in records:
            artifact_id = record.artifact_id
            if len(artifact_id) > 38:
                artifact_id = artifact_id[:35] + "..."

            artifact_type = record.artifact_type
            if len(artifact_type) > 18:
                artifact_type = artifact_type[:15] + "..."

            size_str = _format_bytes(record.byte_size)

            # Truncate timestamp to date + time
            created = record.created_utc[:19]

            _log(f"{artifact_id:<40} {artifact_type:<20} {size_str:>10} {created:<20}", False)

    _log("", False)
    _log(f"Total: {full_count} artifacts", False)

    return 0


# =============================================================================
# GC commands
# =============================================================================


def cmd_gc(
    policy: str,
    root: Path | None,
    dry_run: bool,
    execute: bool,
    no_dvc: bool,
    list_policies: bool,
    quiet: bool,
    output_json: bool,
) -> int:
    """Run garbage collection with a retention policy.

    Args:
        policy: Name of the retention policy to use.
        root: Project root directory (auto-detected if None).
        dry_run: If True, only show what would be deleted.
        execute: If True, actually delete (overrides dry_run).
        no_dvc: If True, skip running dvc gc.
        list_policies: If True, list available policies and exit.
        quiet: If True, suppress non-error output.
        output_json: If True, output results as JSON.

    Returns:
        0 on success, non-zero on error.
    """
    from formula_foundry.m3.artifact_store import ArtifactStore
    from formula_foundry.m3.gc import (
        BUILTIN_POLICIES,
        GarbageCollector,
        PolicyNotFoundError,
        format_bytes,
        load_policies_from_file,
    )
    from formula_foundry.m3.lineage_graph import LineageGraph
    from formula_foundry.m3.registry import ArtifactRegistry

    # Find project root
    project_root = root or Path.cwd()
    if not root:
        project_root = _find_project_root(project_root)

    # List policies mode
    if list_policies:
        all_policies = dict(BUILTIN_POLICIES)
        config_file = project_root / "config" / "gc_policies.yaml"
        if config_file.exists():
            all_policies.update(load_policies_from_file(config_file))

        if output_json:
            data = {name: pol.to_dict() for name, pol in all_policies.items()}
            sys.stdout.write(json.dumps(data, indent=2) + "\n")
        else:
            _log("Available retention policies:", quiet)
            _log("", quiet)
            for name, pol in sorted(all_policies.items()):
                desc = pol.description or "No description"
                budget = format_bytes(pol.space_budget_bytes) if pol.space_budget_bytes else "unlimited"
                _log(f"  {name}:", quiet)
                _log(f"    Description: {desc}", quiet)
                _log(f"    Keep min age: {pol.keep_min_age_days} days", quiet)
                _log(f"    Keep min count: {pol.keep_min_count}", quiet)
                _log(f"    Space budget: {budget}", quiet)
                _log("", quiet)
        return 0

    # Verify M3 is initialized
    data_dir = project_root / "data"
    registry_db = data_dir / "registry.db"
    if not registry_db.exists():
        sys.stderr.write("Error: M3 not initialized. Run 'm3 init' first.\n")
        return 2

    # Determine actual dry_run mode (execute overrides dry_run)
    actual_dry_run = not execute

    _log("M3 Garbage Collection", quiet)
    _log(f"  Policy: {policy}", quiet)
    _log(f"  Mode: {'dry-run' if actual_dry_run else 'EXECUTE'}", quiet)
    _log("", quiet)

    # Initialize components
    store = ArtifactStore(
        root=data_dir,
        generator="m3_gc",
        generator_version=__version__,
    )
    registry = ArtifactRegistry(registry_db)

    # Try to load lineage graph
    lineage_db = data_dir / "lineage.sqlite"
    lineage = None
    if lineage_db.exists():
        lineage = LineageGraph(lineage_db)

    # Create GC instance
    gc = GarbageCollector(
        data_dir=data_dir,
        store=store,
        registry=registry,
        lineage=lineage,
    )

    # Run GC
    try:
        result = gc.run(
            policy=policy,
            dry_run=actual_dry_run,
            run_dvc_gc=not no_dvc,
            enforce_space_budget=True,
        )
    except PolicyNotFoundError as e:
        sys.stderr.write(f"Error: {e}\n")
        sys.stderr.write("Use --list-policies to see available policies.\n")
        registry.close()
        if lineage:
            lineage.close()
        return 2
    except Exception as e:
        sys.stderr.write(f"Error during GC: {e}\n")
        registry.close()
        if lineage:
            lineage.close()
        return 2

    # Output results
    if output_json:
        sys.stdout.write(result.to_json() + "\n")
    else:
        _log(f"GC {'would delete' if actual_dry_run else 'deleted'}:", quiet)
        _log(f"  Artifacts scanned: {result.artifacts_scanned}", quiet)
        _log(f"  Artifacts {'to delete' if actual_dry_run else 'deleted'}: {result.artifacts_deleted}", quiet)
        _log(f"  Space {'to free' if actual_dry_run else 'freed'}: {format_bytes(result.bytes_freed)}", quiet)
        _log("", quiet)
        _log("Protection summary:", quiet)
        _log(f"  Pinned artifacts: {result.pinned_protected}", quiet)
        _log(f"  With descendants: {result.descendant_protected}", quiet)
        _log("", quiet)
        _log("Storage:", quiet)
        _log(f"  Before: {format_bytes(result.bytes_total_before)}", quiet)
        _log(f"  After: {format_bytes(result.bytes_total_after)}", quiet)

        if result.dvc_gc_ran:
            _log("", quiet)
            _log(f"DVC GC: {'ran' if result.dvc_gc_ran else 'skipped'}", quiet)
            if result.dvc_gc_output and not quiet:
                for line in result.dvc_gc_output.strip().split("\n")[:5]:
                    _log(f"  {line}", quiet)

        if result.errors:
            sys.stderr.write("\nErrors:\n")
            for error in result.errors:
                sys.stderr.write(f"  {error}\n")

        if actual_dry_run:
            _log("", quiet)
            _log("This was a dry run. Use --execute to actually delete.", quiet)

    registry.close()
    if lineage:
        lineage.close()

    return 0 if not result.errors else 1


def cmd_gc_pin(
    artifact_id: str | None,
    run_id: str | None,
    dataset_id: str | None,
    reason: str | None,
    root: Path | None,
) -> int:
    """Pin an artifact to protect it from garbage collection.

    Args:
        artifact_id: Specific artifact ID to pin.
        run_id: Run ID to pin all artifacts from.
        dataset_id: Dataset ID to pin all artifacts from.
        reason: Reason for pinning.
        root: Project root directory.

    Returns:
        0 on success, non-zero on error.
    """
    from formula_foundry.m3.artifact_store import ArtifactStore
    from formula_foundry.m3.gc import GarbageCollector
    from formula_foundry.m3.registry import ArtifactRegistry

    if not any([artifact_id, run_id, dataset_id]):
        sys.stderr.write("Error: Must specify --artifact-id, --run-id, or --dataset-id\n")
        return 2

    project_root = root or Path.cwd()
    if not root:
        project_root = _find_project_root(project_root)

    data_dir = project_root / "data"
    registry_db = data_dir / "registry.db"

    if not registry_db.exists():
        sys.stderr.write("Error: M3 not initialized. Run 'm3 init' first.\n")
        return 2

    store = ArtifactStore(root=data_dir, generator="m3_gc_pin", generator_version=__version__)
    registry = ArtifactRegistry(registry_db)

    gc = GarbageCollector(data_dir=data_dir, store=store, registry=registry)

    gc.pin_artifact(
        artifact_id=artifact_id,
        run_id=run_id,
        dataset_id=dataset_id,
        reason=reason,
    )

    registry.close()

    target = artifact_id or run_id or dataset_id
    sys.stdout.write(f"Pinned: {target}\n")
    if reason:
        sys.stdout.write(f"Reason: {reason}\n")

    return 0


def cmd_gc_unpin(
    artifact_id: str | None,
    run_id: str | None,
    dataset_id: str | None,
    root: Path | None,
) -> int:
    """Unpin an artifact to allow garbage collection.

    Args:
        artifact_id: Specific artifact ID to unpin.
        run_id: Run ID to unpin.
        dataset_id: Dataset ID to unpin.
        root: Project root directory.

    Returns:
        0 on success, non-zero on error.
    """
    from formula_foundry.m3.artifact_store import ArtifactStore
    from formula_foundry.m3.gc import GarbageCollector
    from formula_foundry.m3.registry import ArtifactRegistry

    if not any([artifact_id, run_id, dataset_id]):
        sys.stderr.write("Error: Must specify --artifact-id, --run-id, or --dataset-id\n")
        return 2

    project_root = root or Path.cwd()
    if not root:
        project_root = _find_project_root(project_root)

    data_dir = project_root / "data"
    registry_db = data_dir / "registry.db"

    if not registry_db.exists():
        sys.stderr.write("Error: M3 not initialized. Run 'm3 init' first.\n")
        return 2

    store = ArtifactStore(root=data_dir, generator="m3_gc_unpin", generator_version=__version__)
    registry = ArtifactRegistry(registry_db)

    gc = GarbageCollector(data_dir=data_dir, store=store, registry=registry)

    removed = gc.unpin_artifact(
        artifact_id=artifact_id,
        run_id=run_id,
        dataset_id=dataset_id,
    )

    registry.close()

    target = artifact_id or run_id or dataset_id
    if removed:
        sys.stdout.write(f"Unpinned: {target}\n")
    else:
        sys.stdout.write(f"Not pinned: {target}\n")

    return 0


def cmd_gc_estimate(
    policy: str,
    root: Path | None,
    output_json: bool,
) -> int:
    """Estimate space savings from garbage collection.

    Args:
        policy: Name of the retention policy to use.
        root: Project root directory.
        output_json: If True, output results as JSON.

    Returns:
        0 on success, non-zero on error.
    """
    from formula_foundry.m3.artifact_store import ArtifactStore
    from formula_foundry.m3.gc import GarbageCollector, PolicyNotFoundError, format_bytes
    from formula_foundry.m3.lineage_graph import LineageGraph
    from formula_foundry.m3.registry import ArtifactRegistry

    project_root = root or Path.cwd()
    if not root:
        project_root = _find_project_root(project_root)

    data_dir = project_root / "data"
    registry_db = data_dir / "registry.db"

    if not registry_db.exists():
        sys.stderr.write("Error: M3 not initialized. Run 'm3 init' first.\n")
        return 2

    store = ArtifactStore(root=data_dir, generator="m3_gc_estimate", generator_version=__version__)
    registry = ArtifactRegistry(registry_db)

    lineage_db = data_dir / "lineage.sqlite"
    lineage = LineageGraph(lineage_db) if lineage_db.exists() else None

    gc = GarbageCollector(data_dir=data_dir, store=store, registry=registry, lineage=lineage)

    try:
        estimate = gc.estimate_savings(policy=policy)
    except PolicyNotFoundError as e:
        sys.stderr.write(f"Error: {e}\n")
        registry.close()
        if lineage:
            lineage.close()
        return 2

    if output_json:
        sys.stdout.write(json.dumps(estimate, indent=2) + "\n")
    else:
        sys.stdout.write(f"GC Estimate for policy: {policy}\n")
        sys.stdout.write("\n")
        sys.stdout.write(f"  Total artifacts: {estimate['total_artifacts']}\n")
        sys.stdout.write(f"  To delete: {estimate['artifacts_to_delete']}\n")
        sys.stdout.write(f"  To keep: {estimate['artifacts_to_keep']}\n")
        sys.stdout.write("\n")
        sys.stdout.write(f"  Current size: {format_bytes(estimate['current_total_bytes'])}\n")
        sys.stdout.write(f"  Space to free: {format_bytes(estimate['bytes_to_delete'])}\n")
        sys.stdout.write(f"  Size after GC: {format_bytes(estimate['estimated_after_bytes'])}\n")
        if estimate["space_budget_bytes"]:
            sys.stdout.write("\n")
            sys.stdout.write(f"  Space budget: {format_bytes(estimate['space_budget_bytes'])}\n")
            status = "within budget" if estimate["within_budget"] else "OVER BUDGET"
            sys.stdout.write(f"  Status: {status}\n")

    registry.close()
    if lineage:
        lineage.close()

    return 0


def cmd_audit(
    artifact_id: str | None,
    root: Path | None,
    output_format: str,
    trace_roots: bool,
    verify_hashes: bool,
    max_depth: int | None,
    required_roles: str | None,
    quiet: bool,
) -> int:
    """Generate deterministic provenance report for artifacts.

    This command audits artifacts by:
    1. Listing all ancestors (inputs that contributed to the artifact)
    2. Computing key metrics (node count, edge count, types, etc.)
    3. Optionally verifying content hashes
    4. Optionally checking that required roles exist in root artifacts

    Args:
        artifact_id: Artifact ID to audit (if None, audits all).
        root: Project root directory (auto-detected if None).
        output_format: Output format ("text" or "json").
        trace_roots: If True, trace lineage to root artifacts.
        verify_hashes: If True, verify content hashes.
        max_depth: Maximum depth for ancestor traversal.
        required_roles: Comma-separated list of required roles in roots.
        quiet: If True, suppress non-error output.

    Returns:
        0 on success, 2 on error.
    """
    from formula_foundry.m3.artifact_store import ArtifactStore
    from formula_foundry.m3.lineage_graph import (
        LineageGraph,
        NodeNotFoundError,
    )
    from formula_foundry.m3.registry import ArtifactRegistry

    # Find project root
    project_root = root or Path.cwd()
    if not root:
        project_root = _find_project_root(project_root)

    # Verify M3 is initialized
    data_dir = project_root / "data"
    registry_db = data_dir / "registry.db"
    if not registry_db.exists():
        sys.stderr.write("Error: M3 not initialized. Run 'm3 init' first.\n")
        return 2

    lineage_db = data_dir / "lineage.sqlite"

    # Initialize components
    store = ArtifactStore(
        root=data_dir,
        generator="m3_audit",
        generator_version=__version__,
    )
    registry = ArtifactRegistry(registry_db)
    graph = LineageGraph(lineage_db)
    graph.initialize()

    # Parse required roles
    roles_list: list[str] | None = None
    if required_roles:
        roles_list = [r.strip() for r in required_roles.split(",")]

    # Build graph from store if not already populated
    if graph.count_nodes() == 0:
        _log("Building lineage graph from artifact store...", quiet)
        count = graph.build_from_store(store, clear_first=False)
        _log(f"Indexed {count} artifacts into lineage graph.", quiet)

    # Determine artifacts to audit
    if artifact_id:
        artifact_ids = [artifact_id]
    else:
        artifact_ids = store.list_manifests()

    if not artifact_ids:
        if output_format == "json":
            report = {
                "schema_version": 1,
                "generated_utc": _now_utc_iso(),
                "total_artifacts": 0,
                "verification_failures": 0,
                "missing_roles": 0,
                "artifacts": [],
                "graph_stats": graph.get_stats(),
            }
            sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
        else:
            _log("No artifacts found to audit.", quiet)
        graph.close()
        registry.close()
        return 0

    # Prepare audit results
    audit_results: list[dict[str, Any]] = []
    verification_failures: list[str] = []
    missing_roles: list[str] = []

    _log(f"Auditing {len(artifact_ids)} artifact(s)...", quiet)
    _log("", quiet)

    for art_id in artifact_ids:
        result: dict[str, Any] = {
            "artifact_id": art_id,
            "status": "ok",
            "issues": [],
        }

        try:
            # Get manifest
            manifest = store.get_manifest(art_id)
            result["artifact_type"] = manifest.artifact_type
            result["content_hash"] = manifest.content_hash.digest
            result["byte_size"] = manifest.byte_size
            result["created_utc"] = manifest.created_utc
            result["roles"] = manifest.roles
            result["run_id"] = manifest.lineage.run_id
            result["stage_name"] = manifest.lineage.stage_name

            # Verify hash if requested
            if verify_hashes:
                try:
                    is_valid = store.verify(art_id)
                    result["hash_verified"] = is_valid
                    if not is_valid:
                        result["status"] = "hash_mismatch"
                        result["issues"].append("Content hash mismatch")
                        verification_failures.append(art_id)
                except Exception as e:
                    result["hash_verified"] = False
                    result["status"] = "verification_error"
                    result["issues"].append(f"Verification error: {e}")
                    verification_failures.append(art_id)

            # Get lineage information
            if graph.has_node(art_id):
                try:
                    if trace_roots:
                        subgraph = graph.trace_to_roots(art_id)
                    else:
                        subgraph = graph.get_ancestors(art_id, max_depth=max_depth)

                    result["ancestor_count"] = subgraph.node_count - 1  # Exclude self
                    result["edge_count"] = subgraph.edge_count
                    result["root_ids"] = subgraph.get_roots()

                    # Compute type distribution in ancestors
                    type_counts: dict[str, int] = {}
                    for node in subgraph.nodes.values():
                        t = node.artifact_type
                        type_counts[t] = type_counts.get(t, 0) + 1
                    result["ancestor_types"] = type_counts

                    # Check required roles in roots
                    if roles_list:
                        root_roles: set[str] = set()
                        for root_id in subgraph.get_roots():
                            if root_id != art_id:
                                try:
                                    root_manifest = store.get_manifest(root_id)
                                    root_roles.update(root_manifest.roles)
                                except Exception:
                                    pass

                        missing = set(roles_list) - root_roles
                        if missing:
                            result["status"] = "missing_roles"
                            result["issues"].append(f"Required roles not found in roots: {sorted(missing)}")
                            missing_roles.append(art_id)
                        result["root_roles"] = sorted(root_roles)

                except NodeNotFoundError:
                    result["ancestor_count"] = 0
                    result["edge_count"] = 0
                    result["root_ids"] = [art_id]
            else:
                # Node not in graph, add it
                graph.add_manifest(manifest)
                result["ancestor_count"] = 0
                result["edge_count"] = 0
                result["root_ids"] = [art_id]
                result["issues"].append("Node was not in lineage graph (now indexed)")

        except Exception as e:
            result["status"] = "error"
            result["issues"].append(f"Error: {e}")

        audit_results.append(result)

    # Output results
    if output_format == "json":
        report = {
            "schema_version": 1,
            "generated_utc": _now_utc_iso(),
            "total_artifacts": len(artifact_ids),
            "verification_failures": len(verification_failures),
            "missing_roles": len(missing_roles),
            "artifacts": audit_results,
            "graph_stats": graph.get_stats(),
        }
        sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    else:
        # Text format
        for result in audit_results:
            _log(f"Artifact: {result['artifact_id']}", quiet)
            _log(f"  Status: {result['status']}", quiet)
            if "artifact_type" in result:
                _log(f"  Type: {result['artifact_type']}", quiet)
            if "byte_size" in result:
                _log(f"  Size: {result['byte_size']} bytes", quiet)
            if "content_hash" in result:
                _log(f"  Hash: {result['content_hash'][:16]}...", quiet)
            if "created_utc" in result:
                _log(f"  Created: {result['created_utc']}", quiet)
            if "roles" in result:
                _log(f"  Roles: {', '.join(result['roles'])}", quiet)
            if "run_id" in result:
                _log(f"  Run ID: {result['run_id']}", quiet)
            if "ancestor_count" in result:
                _log(f"  Ancestors: {result['ancestor_count']}", quiet)
            if "root_ids" in result:
                if len(result["root_ids"]) <= 5:
                    _log(f"  Roots: {', '.join(result['root_ids'])}", quiet)
                else:
                    _log(f"  Roots: {len(result['root_ids'])} root artifacts", quiet)
            if result.get("hash_verified") is not None:
                status = "PASS" if result["hash_verified"] else "FAIL"
                _log(f"  Hash verification: {status}", quiet)
            if result.get("issues"):
                for issue in result["issues"]:
                    _log(f"  Issue: {issue}", quiet)
            _log("", quiet)

        # Summary
        _log("=" * 60, quiet)
        _log(f"Total artifacts audited: {len(artifact_ids)}", quiet)
        if verify_hashes:
            passed = len(artifact_ids) - len(verification_failures)
            _log(f"Hash verification: {passed}/{len(artifact_ids)} passed", quiet)
        if roles_list:
            passed = len(artifact_ids) - len(missing_roles)
            _log(f"Required roles check: {passed}/{len(artifact_ids)} passed", quiet)

        # Graph stats
        stats = graph.get_stats()
        _log("", quiet)
        _log("Lineage Graph Stats:", quiet)
        _log(f"  Total nodes: {stats['node_count']}", quiet)
        _log(f"  Total edges: {stats['edge_count']}", quiet)
        _log(f"  Root nodes: {stats['root_count']}", quiet)
        _log(f"  Leaf nodes: {stats['leaf_count']}", quiet)

    graph.close()
    registry.close()

    # Return error code if there were failures
    if verification_failures or missing_roles:
        return 2

    return 0


def cmd_verify(
    artifact_id: str | None,
    root: Path | None,
    output_format: str,
    check_hash: bool,
    skip_hash: bool,
    check_lineage: bool,
    check_manifest: bool,
    check_registry: bool,
    full: bool,
    repair: bool,
    quiet: bool,
) -> int:
    """Verify artifact integrity: hashes, lineage consistency, and metadata validation.

    This command performs comprehensive integrity verification:
    1. Hash verification: Recompute content hashes and compare with stored digests
    2. Lineage consistency: Verify all referenced input artifacts exist
    3. Manifest validation: Check manifest structure against artifact.v1 schema
    4. Registry consistency: Ensure registry entries match actual manifests

    Args:
        artifact_id: Artifact ID to verify (if None, verifies all).
        root: Project root directory (auto-detected if None).
        output_format: Output format ("text" or "json").
        check_hash: If True, verify content hashes.
        skip_hash: If True, skip hash verification (overrides check_hash).
        check_lineage: If True, check lineage consistency.
        check_manifest: If True, validate manifest schema.
        check_registry: If True, check registry consistency.
        full: If True, run all verification checks.
        repair: If True, attempt to repair issues.
        quiet: If True, suppress non-error output.

    Returns:
        0 on success (all checks passed),
        1 on warnings (issues found but repaired),
        2 on errors (integrity problems found).
    """
    from formula_foundry.m3.artifact_store import (
        ArtifactNotFoundError,
        ArtifactStore,
    )
    from formula_foundry.m3.lineage_graph import (
        LineageGraph,
    )
    from formula_foundry.m3.registry import ArtifactNotIndexedError, ArtifactRegistry

    # Determine which checks to run
    if full:
        do_hash = True
        do_lineage = True
        do_manifest = True
        do_registry = True
    else:
        # Default: just hash verification unless something else specified
        any_explicit = check_hash or check_lineage or check_manifest or check_registry
        do_hash = check_hash or (not any_explicit and not skip_hash)
        do_lineage = check_lineage
        do_manifest = check_manifest
        do_registry = check_registry

    if skip_hash:
        do_hash = False

    # Find project root
    project_root = root or Path.cwd()
    if not root:
        project_root = _find_project_root(project_root)

    # Verify M3 is initialized
    data_dir = project_root / "data"
    registry_db = data_dir / "registry.db"
    if not registry_db.exists():
        sys.stderr.write("Error: M3 not initialized. Run 'm3 init' first.\n")
        return 2

    lineage_db = data_dir / "lineage.sqlite"

    # Initialize components
    store = ArtifactStore(
        root=data_dir,
        generator="m3_verify",
        generator_version=__version__,
    )
    registry = ArtifactRegistry(registry_db)
    graph = LineageGraph(lineage_db)
    graph.initialize()

    # Build graph from store if needed
    # Suppress logging for JSON output
    suppress_log = quiet or (output_format == "json")

    if do_lineage and graph.count_nodes() == 0:
        _log("Building lineage graph from artifact store...", suppress_log)
        count = graph.build_from_store(store, clear_first=False)
        _log(f"Indexed {count} artifacts into lineage graph.", suppress_log)

    # Determine artifacts to verify
    if artifact_id:
        artifact_ids = [artifact_id]
    else:
        artifact_ids = store.list_manifests()

    if not artifact_ids:
        if output_format == "json":
            report = {
                "schema_version": 1,
                "generated_utc": _now_utc_iso(),
                "verification_type": "m3_verify",
                "checks_performed": {
                    "hash": do_hash,
                    "lineage": do_lineage,
                    "manifest": do_manifest,
                    "registry": do_registry,
                },
                "total_artifacts": 0,
                "passed": 0,
                "warnings": 0,
                "errors": 0,
                "artifacts": [],
            }
            sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
        else:
            _log("No artifacts found to verify.", suppress_log)
        graph.close()
        registry.close()
        return 0

    # Define valid types and roles for schema validation
    valid_types: set[str] = {
        "coupon_spec",
        "resolved_design",
        "kicad_board",
        "kicad_project",
        "gerber",
        "drill_file",
        "fab_package",
        "em_simulation_config",
        "em_simulation_result",
        "touchstone",
        "sparam_matrix",
        "dataset_index",
        "dataset_snapshot",
        "model_checkpoint",
        "formula_candidate",
        "validation_report",
        "drc_report",
        "manifest",
        "log",
        "other",
    }
    valid_roles: set[str] = {
        "geometry",
        "config",
        "oracle_output",
        "intermediate",
        "final_output",
        "validation",
        "metadata",
        "cache",
        "checkpoint",
        "dataset_member",
        "root_input",
    }
    valid_relations: set[str] = {
        "derived_from",
        "generated_by",
        "validated_by",
        "config_from",
        "sibling_of",
        "supersedes",
    }

    # Prepare verification results
    verify_results: list[dict[str, Any]] = []
    total_passed = 0
    total_warnings = 0
    total_errors = 0
    repaired_count = 0

    _log(f"Verifying {len(artifact_ids)} artifact(s)...", suppress_log)
    checks_desc = []
    if do_hash:
        checks_desc.append("hash")
    if do_lineage:
        checks_desc.append("lineage")
    if do_manifest:
        checks_desc.append("manifest")
    if do_registry:
        checks_desc.append("registry")
    _log(f"Checks: {', '.join(checks_desc)}", suppress_log)
    _log("", suppress_log)

    for art_id in artifact_ids:
        result: dict[str, Any] = {
            "artifact_id": art_id,
            "status": "pass",
            "checks": {},
            "issues": [],
            "repaired": [],
        }
        has_error = False
        has_warning = False

        try:
            # Get manifest
            manifest = store.get_manifest(art_id)
            result["artifact_type"] = manifest.artifact_type
            result["content_hash"] = manifest.content_hash.digest
            result["byte_size"] = manifest.byte_size

            # Check 1: Hash verification
            if do_hash:
                try:
                    is_valid = store.verify(art_id)
                    result["checks"]["hash"] = {
                        "passed": is_valid,
                        "expected": manifest.content_hash.digest,
                    }
                    if not is_valid:
                        result["issues"].append("Content hash mismatch - artifact may be corrupted")
                        has_error = True
                except ArtifactNotFoundError:
                    result["checks"]["hash"] = {"passed": False, "error": "Content file not found"}
                    result["issues"].append("Content file missing from object store")
                    has_error = True
                except Exception as e:
                    result["checks"]["hash"] = {"passed": False, "error": str(e)}
                    result["issues"].append(f"Hash verification error: {e}")
                    has_error = True

            # Check 2: Lineage consistency
            if do_lineage:
                lineage_check: dict[str, Any] = {"passed": True, "missing_inputs": []}
                for inp in manifest.lineage.inputs:
                    if not store.exists_by_id(inp.artifact_id):
                        lineage_check["missing_inputs"].append(inp.artifact_id)
                        lineage_check["passed"] = False

                result["checks"]["lineage"] = lineage_check
                if not lineage_check["passed"]:
                    result["issues"].append(f"Missing input artifacts: {lineage_check['missing_inputs']}")
                    has_error = True

            # Check 3: Manifest schema validation
            if do_manifest:
                manifest_check: dict[str, Any] = {"passed": True, "issues": []}

                # Validate required fields
                if not manifest.artifact_id:
                    manifest_check["issues"].append("Missing artifact_id")
                    manifest_check["passed"] = False

                # Validate artifact_type
                if manifest.artifact_type not in valid_types:
                    manifest_check["issues"].append(f"Invalid artifact_type: {manifest.artifact_type}")
                    manifest_check["passed"] = False

                # Validate roles
                for role in manifest.roles:
                    if role not in valid_roles:
                        manifest_check["issues"].append(f"Invalid role: {role}")
                        manifest_check["passed"] = False

                if not manifest.roles:
                    manifest_check["issues"].append("Empty roles list")
                    manifest_check["passed"] = False

                # Validate lineage references
                for inp in manifest.lineage.inputs:
                    if inp.relation not in valid_relations:
                        manifest_check["issues"].append(f"Invalid relation '{inp.relation}' for input {inp.artifact_id}")
                        manifest_check["passed"] = False

                # Validate content hash format
                if manifest.content_hash.algorithm not in ("sha256", "sha384", "sha512", "blake3"):
                    manifest_check["issues"].append(f"Invalid hash algorithm: {manifest.content_hash.algorithm}")
                    manifest_check["passed"] = False

                if len(manifest.content_hash.digest) < 64:
                    manifest_check["issues"].append("Hash digest too short")
                    manifest_check["passed"] = False

                # Validate provenance
                if not manifest.provenance.generator:
                    manifest_check["issues"].append("Missing provenance.generator")
                    manifest_check["passed"] = False

                if not manifest.provenance.hostname:
                    manifest_check["issues"].append("Missing provenance.hostname")
                    manifest_check["passed"] = False

                result["checks"]["manifest"] = manifest_check
                if not manifest_check["passed"]:
                    result["issues"].extend(manifest_check["issues"])
                    has_error = True

            # Check 4: Registry consistency
            if do_registry:
                registry_check: dict[str, Any] = {"passed": True, "issues": []}

                # Check if artifact is in registry
                try:
                    registry_record = registry.get_artifact(art_id)

                    # Verify registry data matches manifest
                    if registry_record.content_hash_digest != manifest.content_hash.digest:
                        registry_check["passed"] = False
                        registry_check["issues"].append(
                            f"Registry hash mismatch: registry={registry_record.content_hash_digest[:16]}..., "
                            f"manifest={manifest.content_hash.digest[:16]}..."
                        )
                        has_error = True

                    if registry_record.artifact_type != manifest.artifact_type:
                        registry_check["passed"] = False
                        registry_check["issues"].append(
                            f"Registry type mismatch: registry={registry_record.artifact_type}, manifest={manifest.artifact_type}"
                        )
                        has_error = True

                except ArtifactNotIndexedError:
                    # Artifact not found in registry
                    registry_check["passed"] = False
                    registry_check["issues"].append("Artifact not found in registry")

                    if repair:
                        # Attempt to repair by re-indexing
                        try:
                            registry.index_artifact(manifest)
                            result["repaired"].append("Re-indexed artifact into registry")
                            repaired_count += 1
                            registry_check["repaired"] = True
                            has_warning = True
                        except Exception as e:
                            registry_check["repair_error"] = str(e)
                            has_error = True
                    else:
                        has_error = True

                result["checks"]["registry"] = registry_check
                # Only add issues if repair didn't fix them
                if registry_check["issues"] and not registry_check.get("repaired"):
                    result["issues"].extend(registry_check["issues"])

        except ArtifactNotFoundError:
            result["status"] = "error"
            result["issues"].append(f"Manifest not found for artifact: {art_id}")
            has_error = True
        except Exception as e:
            result["status"] = "error"
            result["issues"].append(f"Verification error: {e}")
            has_error = True

        # Determine final status
        if has_error:
            result["status"] = "error"
            total_errors += 1
        elif has_warning:
            result["status"] = "warning"
            total_warnings += 1
        else:
            result["status"] = "pass"
            total_passed += 1

        verify_results.append(result)

    # Output results
    if output_format == "json":
        report = {
            "schema_version": 1,
            "generated_utc": _now_utc_iso(),
            "verification_type": "m3_verify",
            "checks_performed": {
                "hash": do_hash,
                "lineage": do_lineage,
                "manifest": do_manifest,
                "registry": do_registry,
            },
            "total_artifacts": len(artifact_ids),
            "passed": total_passed,
            "warnings": total_warnings,
            "errors": total_errors,
            "repaired_count": repaired_count,
            "artifacts": verify_results,
        }
        sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    else:
        # Text format
        for result in verify_results:
            status_marker = {
                "pass": "[PASS]",
                "warning": "[WARN]",
                "error": "[FAIL]",
            }.get(result["status"], "[????]")

            _log(f"{status_marker} {result['artifact_id']}", quiet)

            if "artifact_type" in result:
                _log(f"    Type: {result['artifact_type']}", quiet)
            if "byte_size" in result:
                _log(f"    Size: {result['byte_size']} bytes", quiet)

            # Show check results
            checks = result.get("checks", {})
            if "hash" in checks:
                h_status = "PASS" if checks["hash"].get("passed") else "FAIL"
                _log(f"    Hash verification: {h_status}", quiet)
            if "lineage" in checks:
                l_status = "PASS" if checks["lineage"].get("passed") else "FAIL"
                _log(f"    Lineage consistency: {l_status}", quiet)
                if checks["lineage"].get("missing_inputs"):
                    _log(f"      Missing: {checks['lineage']['missing_inputs']}", quiet)
            if "manifest" in checks:
                m_status = "PASS" if checks["manifest"].get("passed") else "FAIL"
                _log(f"    Manifest validation: {m_status}", quiet)
            if "registry" in checks:
                r_status = "PASS" if checks["registry"].get("passed") else "FAIL"
                _log(f"    Registry consistency: {r_status}", quiet)
                if checks["registry"].get("repaired"):
                    _log("      (Repaired)", quiet)

            # Show issues
            if result.get("issues"):
                for issue in result["issues"]:
                    _log(f"    Issue: {issue}", quiet)

            # Show repairs
            if result.get("repaired"):
                for repair_action in result["repaired"]:
                    _log(f"    Repaired: {repair_action}", quiet)

            _log("", quiet)

        # Summary
        _log("=" * 60, quiet)
        _log(f"Verification complete: {len(artifact_ids)} artifacts", quiet)
        _log(f"  Passed:   {total_passed}", quiet)
        _log(f"  Warnings: {total_warnings}", quiet)
        _log(f"  Errors:   {total_errors}", quiet)
        if repair and repaired_count > 0:
            _log(f"  Repaired: {repaired_count}", quiet)

    graph.close()
    registry.close()

    # Return appropriate exit code
    if total_errors > 0:
        return 2
    if total_warnings > 0:
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Main entry point for the m3 CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "init":
        return cmd_init(
            root=args.root,
            force=args.force,
            quiet=args.quiet,
        )

    if args.command == "run":
        return cmd_run(
            stage=args.stage,
            root=args.root,
            run_type=args.run_type,
            dry_run=args.dry_run,
            force=args.force,
            quiet=args.quiet,
            tags=args.tags,
        )

    if args.command == "dataset":
        if args.dataset_command == "show":
            return cmd_dataset_show(
                dataset_id=args.dataset_id,
                version=args.version,
                root=args.root,
                output_json=args.output_json,
                show_members=args.members,
                quiet=args.quiet,
            )
        if args.dataset_command == "diff":
            return cmd_dataset_diff(
                dataset_a=args.dataset_a,
                dataset_b=args.dataset_b,
                root=args.root,
                output_json=args.output_json,
                quiet=args.quiet,
            )
        parser.error(f"Unknown dataset command: {args.dataset_command}")

    if args.command == "artifact":
        if args.artifact_command == "show":
            return cmd_artifact_show(
                artifact_id=args.artifact_id,
                root=args.root,
                output_json=args.output_json,
                show_content=args.content,
                verify=args.verify,
                quiet=args.quiet,
            )
        if args.artifact_command == "list":
            return cmd_artifact_list(
                root=args.root,
                artifact_type=args.artifact_type,
                run_id=args.run_id,
                roles=args.roles,
                created_after=args.created_after,
                created_before=args.created_before,
                limit=args.limit,
                offset=args.offset,
                order_by=args.order_by,
                order_asc=args.order_asc,
                output_json=args.output_json,
                long_format=args.long_format,
                quiet=args.quiet,
            )
        parser.error(f"Unknown artifact command: {args.artifact_command}")

    if args.command == "gc":
        return cmd_gc(
            policy=args.policy,
            root=args.root,
            dry_run=args.dry_run,
            execute=args.execute,
            no_dvc=args.no_dvc,
            list_policies=args.list_policies,
            quiet=args.quiet,
            output_json=args.json,
        )

    if args.command == "gc-pin":
        return cmd_gc_pin(
            artifact_id=args.artifact_id,
            run_id=args.run_id,
            dataset_id=args.dataset_id,
            reason=args.reason,
            root=args.root,
        )

    if args.command == "gc-unpin":
        return cmd_gc_unpin(
            artifact_id=args.artifact_id,
            run_id=args.run_id,
            dataset_id=args.dataset_id,
            root=args.root,
        )

    if args.command == "gc-estimate":
        return cmd_gc_estimate(
            policy=args.policy,
            root=args.root,
            output_json=args.json,
        )

    if args.command == "audit":
        return cmd_audit(
            artifact_id=args.artifact_id,
            root=args.root,
            output_format=args.format,
            trace_roots=args.trace_roots,
            verify_hashes=args.verify_hashes,
            max_depth=args.max_depth,
            required_roles=args.required_roles,
            quiet=args.quiet,
        )

    if args.command == "verify":
        return cmd_verify(
            artifact_id=args.artifact_id,
            root=args.root,
            output_format=args.format,
            check_hash=args.check_hash,
            skip_hash=args.skip_hash,
            check_lineage=args.check_lineage,
            check_manifest=args.check_manifest,
            check_registry=args.check_registry,
            full=args.full,
            repair=args.repair,
            quiet=args.quiet,
        )

    # Should not reach here due to required=True on subparsers
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
