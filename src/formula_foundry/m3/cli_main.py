"""M3 CLI: Artifact storage backbone commands.

This module provides the `m3` command-line interface for initializing and
managing the Formula Foundry data provenance system.

Commands:
    init: Initialize data directory structure, DVC, MLflow config, and registry.
"""

from __future__ import annotations

import argparse
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Sequence

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

    return parser


def _log(msg: str, quiet: bool = False) -> None:
    """Print a message unless quiet mode is enabled."""
    if not quiet:
        sys.stdout.write(msg + "\n")


def _now_utc_iso() -> str:
    """Get current UTC time in ISO 8601 format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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

    # Should not reach here due to required=True on subparsers
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
