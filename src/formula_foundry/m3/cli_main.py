"""M3 CLI: Artifact storage backbone commands.

This module provides the `m3` command-line interface for initializing and
managing the Formula Foundry data provenance system.

Commands:
    init: Initialize data directory structure, DVC, MLflow config, and registry.
    run: Execute a DVC stage with metadata stamping and artifact tracking.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from typing import Literal, Sequence

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
        or "Stage" in result.stdout and "cached" in result.stdout.lower()
        or result.returncode == 0 and "Running" not in result.stdout
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

    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
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
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
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
    )

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
        config={"stage": stage, "force": force, "run_type": effective_run_type},
    )

    _log(f"Run ID: {run_id}", quiet)
    _log(f"Git commit: {git_info.commit[:12]} ({git_info.branch})", quiet)
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

    if return_code != 0:
        run_metadata.error = {
            "error_type": "DVCError",
            "message": stderr.strip() if stderr else "DVC stage failed",
            "stage_name": stage,
            "recoverable": True,
        }

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
        LineageSubgraph,
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
                            result["issues"].append(
                                f"Required roles not found in roots: {sorted(missing)}"
                            )
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

    # Should not reach here due to required=True on subparsers
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
