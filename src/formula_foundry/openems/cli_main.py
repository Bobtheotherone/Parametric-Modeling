"""M2 CLI: openEMS simulation subsystem commands.

This module provides the `m2` command-line interface for running openEMS FDTD
simulations and extracting S-parameters.

Commands:
    sim run: Run a single simulation from a config file.
    sim batch: Run batch simulations from a directory of configs.
    sim status: Check status of a simulation run.
    sparam extract: Extract S-parameters from a completed simulation.
    validate: Validate a simulation manifest.

REQ-M2-010: CLI interface for M2 openEMS simulation subsystem.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, cast

from formula_foundry.substrate import canonical_json_dumps

from .batch_runner import (
    BatchConfig,
    BatchSimulationRunner,
    SimulationJob,
    load_batch_result_summary,
    write_batch_result,
)
from .gpu_batch_runner import (
    GPUBatchConfig,
    GPUBatchMode,
    GPUBatchSimulationRunner,
    check_cuda_available,
    detect_nvidia_gpus,
    write_gpu_batch_result,
)
from .convergence import validate_simulation_convergence
from .geometry import (
    BoardOutlineSpec,
    DiscontinuitySpec,
    GeometrySpec,
    LayerSpec,
    StackupMaterialsSpec,
    StackupSpec,
    TransmissionLineSpec,
)
from .manifest import load_m2_manifest, validate_m2_manifest
from .runner import OpenEMSMode, OpenEMSRunner
from .sim_runner import SimulationRunner, SimulationSolverMode
from .sparam_extract import (
    ExtractionConfig,
    extract_sparams,
    write_extraction_result,
)
from .spec import SimulationSpec, load_simulationspec
from .toolchain import OpenEMSToolchain, load_openems_toolchain

logger = logging.getLogger(__name__)

# Version for provenance tracking
__version__ = "0.1.0"


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the M2 CLI.

    Returns:
        ArgumentParser with all M2 subcommands configured.
    """
    # Shared arguments
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--mode", choices=("local", "docker"), default="local")
    shared.add_argument("--docker-image", default="")
    shared.add_argument("--openems-bin", default="openEMS")
    shared.add_argument("--workdir", default=".")
    shared.add_argument("--toolchain-path", default="")
    shared.add_argument("-v", "--verbose", action="count", default=0)

    parser = argparse.ArgumentParser(
        prog="m2",
        description="Formula Foundry M2: openEMS simulation subsystem CLI",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (can be repeated)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Legacy commands: version and run (for backward compatibility)
    version_cmd = subparsers.add_parser(
        "version",
        help="Capture openEMS version metadata",
        parents=[shared],
    )
    version_cmd.add_argument("--json", nargs="?", const="-", default="-")

    run_legacy = subparsers.add_parser(
        "run",
        help="Run openEMS with provided args (legacy)",
        parents=[shared],
    )
    run_legacy.add_argument("--json", nargs="?", const="-", default=None)
    run_legacy.add_argument("openems_args", nargs=argparse.REMAINDER)

    # sim subcommand
    sim_parser = subparsers.add_parser(
        "sim",
        help="Simulation commands",
    )
    sim_subparsers = sim_parser.add_subparsers(dest="sim_command", required=True)

    # sim run
    sim_run = sim_subparsers.add_parser(
        "run",
        help="Run a single simulation from a config file",
        parents=[shared],
    )
    sim_run.add_argument(
        "config",
        type=Path,
        help="Path to simulation config file (JSON/YAML)",
    )
    sim_run.add_argument(
        "--geometry",
        type=Path,
        help="Path to geometry config file (optional, can be embedded in config)",
    )
    sim_run.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory for simulation results",
    )
    sim_run.add_argument(
        "--timeout",
        type=float,
        default=3600.0,
        help="Timeout in seconds (default: 3600)",
    )
    sim_run.add_argument(
        "--solver-mode",
        choices=("stub", "cli"),
        default="stub",
        help="Solver mode: stub for testing, cli for real openEMS",
    )
    sim_run.add_argument(
        "--no-convergence",
        action="store_true",
        help="Skip convergence validation after simulation",
    )
    sim_run.add_argument(
        "--json",
        nargs="?",
        const="-",
        default=None,
        help="Output result as JSON (to file or stdout if '-')",
    )

    # sim batch
    sim_batch = sim_subparsers.add_parser(
        "batch",
        help="Run batch simulations from a directory of configs",
        parents=[shared],
    )
    sim_batch.add_argument(
        "config_dir",
        type=Path,
        help="Directory containing simulation config files",
    )
    sim_batch.add_argument(
        "--geometry-dir",
        type=Path,
        help="Directory containing geometry files (optional)",
    )
    sim_batch.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output base directory for simulation results",
    )
    sim_batch.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum parallel simulations (default: 4)",
    )
    sim_batch.add_argument(
        "--timeout",
        type=float,
        default=3600.0,
        help="Timeout per simulation in seconds (default: 3600)",
    )
    sim_batch.add_argument(
        "--solver-mode",
        choices=("stub", "cli"),
        default="stub",
        help="Solver mode: stub for testing, cli for real openEMS",
    )
    sim_batch.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop batch on first failure",
    )
    sim_batch.add_argument(
        "--no-convergence",
        action="store_true",
        help="Skip convergence validation",
    )
    sim_batch.add_argument(
        "--json",
        nargs="?",
        const="-",
        default=None,
        help="Output batch result as JSON",
    )

    # GPU batch arguments (REQ-M2-010)
    sim_batch.add_argument(
        "--use-gpu",
        action="store_true",
        help="Enable GPU acceleration for batch simulations",
    )
    sim_batch.add_argument(
        "--gpu-mode",
        choices=("auto", "force_gpu", "force_cpu", "hybrid"),
        default="auto",
        help="GPU batching mode: auto (use GPU if available), force_gpu (require GPU), "
        "force_cpu (CPU only), hybrid (use both)",
    )
    sim_batch.add_argument(
        "--gpu-devices",
        type=str,
        default=None,
        help="Comma-separated list of GPU device IDs to use (e.g., '0,1,2')",
    )
    sim_batch.add_argument(
        "--vram-per-sim",
        type=int,
        default=2048,
        help="Estimated VRAM per simulation in MB (default: 2048)",
    )
    sim_batch.add_argument(
        "--gpu-memory-fraction",
        type=float,
        default=0.8,
        help="Fraction of GPU memory to use (0.1-1.0, default: 0.8)",
    )
    sim_batch.add_argument(
        "--max-sims-per-gpu",
        type=int,
        default=4,
        help="Maximum concurrent simulations per GPU (default: 4)",
    )
    sim_batch.add_argument(
        "--no-gpu-fallback",
        action="store_true",
        help="Disable CPU fallback on GPU failure",
    )
    sim_batch.add_argument(
        "--oom-retries",
        type=int,
        default=2,
        help="Number of OOM retries before fallback (default: 2)",
    )

    # sim status
    sim_status = sim_subparsers.add_parser(
        "status",
        help="Check status of a simulation run",
    )
    sim_status.add_argument(
        "run_id",
        help="Run ID or path to simulation output directory",
    )
    sim_status.add_argument(
        "--json",
        action="store_true",
        help="Output status as JSON",
    )

    # sparam subcommand
    sparam_parser = subparsers.add_parser(
        "sparam",
        help="S-parameter commands",
    )
    sparam_subparsers = sparam_parser.add_subparsers(
        dest="sparam_command",
        required=True,
    )

    # sparam extract
    sparam_extract = sparam_subparsers.add_parser(
        "extract",
        help="Extract S-parameters from a completed simulation",
    )
    sparam_extract.add_argument(
        "sim_dir",
        type=Path,
        help="Path to simulation output directory",
    )
    sparam_extract.add_argument(
        "--config",
        type=Path,
        help="Path to simulation config (for frequency/port info)",
    )
    sparam_extract.add_argument(
        "--out",
        type=Path,
        help="Output directory for extracted S-parameters (default: sim_dir)",
    )
    sparam_extract.add_argument(
        "--format",
        choices=("touchstone", "csv", "both"),
        default="touchstone",
        help="Output format (default: touchstone)",
    )
    sparam_extract.add_argument(
        "--json",
        action="store_true",
        help="Output extraction result as JSON",
    )

    # validate subcommand
    validate_cmd = subparsers.add_parser(
        "validate",
        help="Validate a simulation manifest",
    )
    validate_cmd.add_argument(
        "manifest",
        type=Path,
        help="Path to manifest file to validate",
    )
    validate_cmd.add_argument(
        "--strict",
        action="store_true",
        help="Fail on warnings",
    )
    validate_cmd.add_argument(
        "--json",
        action="store_true",
        help="Output validation result as JSON",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Main entry point for M2 CLI.

    Args:
        argv: Command line arguments (defaults to sys.argv).

    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # Set up logging
    log_level = logging.WARNING
    if args.verbose >= 1:
        log_level = logging.INFO
    if args.verbose >= 2:
        log_level = logging.DEBUG
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    try:
        if args.command == "version":
            return _cmd_version(args)
        elif args.command == "run":
            return _cmd_run_legacy(args)
        elif args.command == "sim":
            return _cmd_sim(args)
        elif args.command == "sparam":
            return _cmd_sparam(args)
        elif args.command == "validate":
            return _cmd_validate(args)
        else:
            parser.error(f"Unknown command: {args.command}")
            return 2
    except Exception as e:
        logger.error("Error: %s", e)
        if args.verbose >= 2:
            import traceback

            traceback.print_exc()
        return 1


# =============================================================================
# Command Handlers
# =============================================================================


def _cmd_version(args: argparse.Namespace) -> int:
    """Handle version command."""
    workdir = Path(args.workdir).resolve()
    toolchain = _resolve_toolchain(args.mode, args.docker_image, args.toolchain_path)
    docker_image = args.docker_image or (toolchain.docker_image if toolchain else None)

    runner = OpenEMSRunner(
        mode=_cast_mode(args.mode),
        docker_image=docker_image,
        openems_bin=args.openems_bin,
    )

    payload = runner.version_metadata(workdir=workdir)
    if toolchain is not None:
        payload["toolchain_version"] = toolchain.version
    _emit_json(payload, args.json)
    return 0 if payload.get("returncode", 1) == 0 else 2


def _cmd_run_legacy(args: argparse.Namespace) -> int:
    """Handle legacy run command."""
    workdir = Path(args.workdir).resolve()
    toolchain = _resolve_toolchain(args.mode, args.docker_image, args.toolchain_path)
    docker_image = args.docker_image or (toolchain.docker_image if toolchain else None)

    runner = OpenEMSRunner(
        mode=_cast_mode(args.mode),
        docker_image=docker_image,
        openems_bin=args.openems_bin,
    )

    if not args.openems_args:
        sys.stderr.write("Error: run requires openEMS arguments after '--'\n")
        return 2

    proc = runner.run(args.openems_args, workdir=workdir)
    if args.json:
        run_payload: dict[str, Any] = {
            "command": runner.build_command(args.openems_args, workdir=workdir),
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
        _emit_json(run_payload, args.json)
    return proc.returncode


def _cmd_sim(args: argparse.Namespace) -> int:
    """Handle sim subcommand."""
    if args.sim_command == "run":
        return _cmd_sim_run(args)
    elif args.sim_command == "batch":
        return _cmd_sim_batch(args)
    elif args.sim_command == "status":
        return _cmd_sim_status(args)
    else:
        sys.stderr.write(f"Unknown sim command: {args.sim_command}\n")
        return 2


def _cmd_sim_run(args: argparse.Namespace) -> int:
    """Handle sim run command - run single simulation."""
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        sys.stderr.write(f"Error: Config file not found: {config_path}\n")
        return 1

    # Load simulation spec
    spec_data = _load_json_or_yaml(config_path)
    spec = load_simulationspec(spec_data)

    # Load or build geometry spec
    geometry = _load_geometry(args, spec)

    # Set up simulation runner
    sim_runner = _create_sim_runner(args)

    # Run simulation
    output_dir = Path(args.out).resolve()
    logger.info("Running simulation to %s", output_dir)

    result = sim_runner.run(
        spec,
        geometry,
        output_dir=output_dir,
        timeout_sec=args.timeout,
    )

    # Run convergence validation unless disabled
    convergence_report = None
    if not args.no_convergence:
        try:
            convergence_report = validate_simulation_convergence(
                result.outputs_dir,
                spec,
                simulation_hash=result.simulation_hash,
            )
            logger.info(
                "Convergence: %s (%d/%d passed)",
                convergence_report.overall_status.value,
                convergence_report.n_passed,
                len(convergence_report.checks),
            )
        except Exception as e:
            logger.warning("Convergence validation failed: %s", e)

    # Build result payload
    payload: dict[str, Any] = {
        "simulation_hash": result.simulation_hash,
        "manifest_hash": result.manifest_hash,
        "output_dir": str(result.output_dir),
        "outputs_dir": str(result.outputs_dir),
        "cache_hit": result.cache_hit,
    }
    if result.execution_time_sec is not None:
        payload["execution_time_sec"] = result.execution_time_sec
    if result.sparam_path is not None:
        payload["sparam_path"] = str(result.sparam_path)
    if convergence_report is not None:
        payload["convergence"] = {
            "status": convergence_report.overall_status.value,
            "n_passed": convergence_report.n_passed,
            "n_failed": convergence_report.n_failed,
        }

    if args.json:
        _emit_json(payload, args.json)
    else:
        _print_sim_result(payload)

    return 0


def _cmd_sim_batch(args: argparse.Namespace) -> int:
    """Handle sim batch command - run batch simulations."""
    config_dir = Path(args.config_dir).resolve()
    if not config_dir.is_dir():
        sys.stderr.write(f"Error: Config directory not found: {config_dir}\n")
        return 1

    output_base = Path(args.out).resolve()
    output_base.mkdir(parents=True, exist_ok=True)

    # Find all config files
    config_files = list(config_dir.glob("*.json")) + list(config_dir.glob("*.yaml"))
    if not config_files:
        sys.stderr.write(f"Error: No config files found in {config_dir}\n")
        return 1

    logger.info("Found %d config files in %s", len(config_files), config_dir)

    # Create batch jobs
    jobs: list[SimulationJob] = []
    for config_path in sorted(config_files):
        try:
            spec_data = _load_json_or_yaml(config_path)
            spec = load_simulationspec(spec_data)
            geometry = _load_geometry_from_dir(args.geometry_dir, spec)

            job_id = config_path.stem
            job_output = output_base / job_id
            jobs.append(
                SimulationJob(
                    job_id=job_id,
                    spec=spec,
                    geometry=geometry,
                    output_dir=job_output,
                )
            )
        except Exception as e:
            logger.warning("Failed to load config %s: %s", config_path, e)
            if args.fail_fast:
                sys.stderr.write(f"Error loading {config_path}: {e}\n")
                return 1

    if not jobs:
        sys.stderr.write("Error: No valid simulation jobs created\n")
        return 1

    # Configure batch runner
    batch_config = BatchConfig(
        max_workers=args.max_workers,
        timeout_per_sim_sec=args.timeout,
        fail_fast=args.fail_fast,
        validate_convergence=not args.no_convergence,
    )

    sim_runner = _create_sim_runner(args)

    # Progress callback
    def progress_callback(progress: Any) -> None:
        logger.info(
            "Progress: %d/%d complete (%.1f%%)",
            progress.finished,
            progress.total,
            progress.percent_complete,
        )

    # Check if GPU mode is requested (REQ-M2-010)
    use_gpu_batch = getattr(args, "use_gpu", False)
    gpu_mode_str = getattr(args, "gpu_mode", "auto")

    if use_gpu_batch or gpu_mode_str != "auto":
        # GPU batch execution (REQ-M2-010)
        return _run_gpu_batch(
            jobs=jobs,
            sim_runner=sim_runner,
            batch_config=batch_config,
            output_base=output_base,
            args=args,
            progress_callback=progress_callback,
        )
    else:
        # Standard CPU batch execution
        batch_runner = BatchSimulationRunner(sim_runner, batch_config)

        # Run batch
        logger.info("Starting batch of %d simulations", len(jobs))
        batch_result = batch_runner.run(jobs, progress_callback=progress_callback)

        # Write batch result
        result_path = output_base / "batch_result.json"
        write_batch_result(batch_result, result_path)
        logger.info("Batch result written to %s", result_path)

        # Build output payload
        payload = batch_result.to_dict()
        payload["result_path"] = str(result_path)

        if args.json:
            _emit_json(payload, args.json)
        else:
            _print_batch_result(batch_result)

        return 0 if batch_result.all_passed else 1


def _run_gpu_batch(
    jobs: list[SimulationJob],
    sim_runner: SimulationRunner,
    batch_config: BatchConfig,
    output_base: Path,
    args: argparse.Namespace,
    progress_callback: Any,
) -> int:
    """Run GPU-accelerated batch simulations (REQ-M2-010).

    Args:
        jobs: List of simulation jobs.
        sim_runner: SimulationRunner instance.
        batch_config: Batch configuration.
        output_base: Output directory.
        args: Command line arguments.
        progress_callback: Progress callback function.

    Returns:
        Exit code (0 for success).
    """
    # Parse GPU device IDs if specified
    gpu_device_ids: tuple[int, ...] | None = None
    if getattr(args, "gpu_devices", None):
        try:
            gpu_device_ids = tuple(int(d.strip()) for d in args.gpu_devices.split(","))
        except ValueError:
            sys.stderr.write(f"Error: Invalid GPU device IDs: {args.gpu_devices}\n")
            return 1

    # Map GPU mode string to enum
    gpu_mode_map = {
        "auto": GPUBatchMode.AUTO,
        "force_gpu": GPUBatchMode.FORCE_GPU,
        "force_cpu": GPUBatchMode.FORCE_CPU,
        "hybrid": GPUBatchMode.HYBRID,
    }
    gpu_mode = gpu_mode_map.get(getattr(args, "gpu_mode", "auto"), GPUBatchMode.AUTO)

    # Configure GPU batch runner
    gpu_config = GPUBatchConfig(
        mode=gpu_mode,
        device_ids=gpu_device_ids,
        vram_per_sim_mb=getattr(args, "vram_per_sim", 2048),
        gpu_memory_fraction=getattr(args, "gpu_memory_fraction", 0.8),
        max_sims_per_gpu=getattr(args, "max_sims_per_gpu", 4),
        fallback_to_cpu=not getattr(args, "no_gpu_fallback", False),
        oom_retry_count=getattr(args, "oom_retries", 2),
        track_utilization=True,
    )

    # Check GPU availability
    if gpu_config.mode != GPUBatchMode.FORCE_CPU:
        detected_gpus = detect_nvidia_gpus()
        if detected_gpus:
            logger.info(
                "GPU batch mode enabled with %d GPU(s): %s",
                len(detected_gpus),
                ", ".join(f"{g.device_name} ({g.device_id})" for g in detected_gpus),
            )
        else:
            if gpu_config.mode == GPUBatchMode.FORCE_GPU:
                sys.stderr.write("Error: No GPUs detected but --gpu-mode=force_gpu\n")
                return 1
            logger.warning("No GPUs detected, falling back to CPU execution")
    else:
        logger.info("GPU batch mode disabled (--gpu-mode=force_cpu)")

    # Create GPU batch runner
    gpu_batch_runner = GPUBatchSimulationRunner(
        sim_runner,
        batch_config,
        gpu_config,
    )

    # Run GPU batch
    logger.info("Starting GPU batch of %d simulations", len(jobs))
    gpu_result = gpu_batch_runner.run(jobs, progress_callback=progress_callback)

    # Write GPU batch result
    result_path = output_base / "gpu_batch_result.json"
    write_gpu_batch_result(gpu_result, result_path)
    logger.info("GPU batch result written to %s", result_path)

    # Build output payload
    payload = gpu_result.to_dict()
    payload["result_path"] = str(result_path)

    if args.json:
        _emit_json(payload, args.json)
    else:
        _print_gpu_batch_result(gpu_result)

    return 0 if gpu_result.batch_result.all_passed else 1


def _print_gpu_batch_result(result: Any) -> None:
    """Print GPU batch result in human-readable format."""
    batch = result.batch_result
    print("GPU Batch simulation completed:")
    print(f"  Total jobs: {len(batch.jobs)}")
    print(f"  Completed: {batch.n_completed}")
    print(f"  Failed: {batch.n_failed}")
    print(f"  Skipped: {batch.n_skipped}")
    print(f"  Success rate: {batch.success_rate:.1f}%")
    print(f"  Total time: {batch.total_time_sec:.2f}s")

    # GPU-specific stats
    print(f"\nGPU Execution Stats:")
    print(f"  GPU jobs: {result.n_gpu_jobs}")
    print(f"  CPU fallback jobs: {result.n_cpu_fallback_jobs}")
    print(f"  OOM retries: {result.n_oom_retries}")
    print(f"  OOM failures: {result.n_oom_failures}")

    # GPU utilization metrics
    if result.gpu_metrics:
        print(f"\nGPU Utilization:")
        for device_id, metrics in result.gpu_metrics.items():
            print(f"  GPU {device_id} ({metrics.device_name}):")
            print(f"    Jobs run: {metrics.total_jobs_run}")
            print(f"    Avg utilization: {metrics.avg_utilization_percent:.1f}%")
            print(f"    Peak utilization: {metrics.max_utilization_percent:.1f}%")
            print(f"    Avg memory used: {metrics.avg_memory_used_mb:.0f} MB")
            print(f"    Peak memory used: {metrics.max_memory_used_mb} MB")

    if batch.config.validate_convergence:
        print(f"\nConvergence:")
        print(f"  Passed: {batch.n_convergence_passed}")
        print(f"  Failed: {batch.n_convergence_failed}")


def _cmd_sim_status(args: argparse.Namespace) -> int:
    """Handle sim status command - check simulation status."""
    run_id = args.run_id

    # Try to interpret run_id as a path
    run_path = Path(run_id)
    if not run_path.exists():
        # Try as a relative path in current directory
        run_path = Path.cwd() / run_id
    if not run_path.exists():
        sys.stderr.write(f"Error: Simulation directory not found: {run_id}\n")
        return 1

    run_path = run_path.resolve()

    # Check for manifest
    manifest_path = run_path / "simulation_manifest.json"
    batch_result_path = run_path / "batch_result.json"

    status: dict[str, Any] = {
        "run_id": run_id,
        "path": str(run_path),
        "exists": True,
    }

    if manifest_path.exists():
        # Single simulation
        try:
            manifest = load_m2_manifest(manifest_path)
            status["type"] = "single"
            status["simulation_hash"] = manifest.get("simulation_hash")
            status["status"] = "completed"
            if "execution_time_sec" in manifest:
                status["execution_time_sec"] = manifest["execution_time_sec"]
            if "convergence" in manifest:
                status["convergence"] = manifest["convergence"]
        except Exception as e:
            status["status"] = "error"
            status["error"] = str(e)
    elif batch_result_path.exists():
        # Batch simulation
        try:
            batch_summary = load_batch_result_summary(batch_result_path)
            status["type"] = "batch"
            status["status"] = "completed"
            status["total_jobs"] = batch_summary.get("total_jobs")
            status["n_completed"] = batch_summary.get("n_completed")
            status["n_failed"] = batch_summary.get("n_failed")
            status["success_rate"] = batch_summary.get("success_rate")
        except Exception as e:
            status["status"] = "error"
            status["error"] = str(e)
    else:
        # Check if simulation is in progress
        outputs_dir = run_path / "sim_outputs"
        if outputs_dir.exists():
            status["type"] = "single"
            status["status"] = "in_progress"
        else:
            status["type"] = "unknown"
            status["status"] = "not_found"

    if args.json:
        _emit_json(status, "-")
    else:
        _print_status(status)

    return 0 if status.get("status") == "completed" else 1


def _cmd_sparam(args: argparse.Namespace) -> int:
    """Handle sparam subcommand."""
    if args.sparam_command == "extract":
        return _cmd_sparam_extract(args)
    else:
        sys.stderr.write(f"Unknown sparam command: {args.sparam_command}\n")
        return 2


def _cmd_sparam_extract(args: argparse.Namespace) -> int:
    """Handle sparam extract command - extract S-parameters."""
    sim_dir = Path(args.sim_dir).resolve()
    outputs_dir = sim_dir / "sim_outputs"
    if not outputs_dir.exists():
        outputs_dir = sim_dir  # Maybe they pointed directly at outputs

    if not outputs_dir.exists():
        sys.stderr.write(f"Error: Simulation output directory not found: {sim_dir}\n")
        return 1

    # Load config if provided
    if args.config:
        config_path = Path(args.config).resolve()
        spec_data = _load_json_or_yaml(config_path)
        spec = load_simulationspec(spec_data)
        extraction_config = ExtractionConfig(
            frequency_spec=spec.frequency,
            port_specs=spec.ports,
            output_format=args.format,
        )
    else:
        # Create minimal config from defaults
        from .spec import FrequencySpec, PortSpec

        # Try to infer from available files
        extraction_config = ExtractionConfig(
            frequency_spec=FrequencySpec(
                f_start_hz=1_000_000,
                f_stop_hz=10_000_000_000,
                n_points=201,
            ),
            port_specs=[
                PortSpec(
                    id="P1",
                    type="lumped",
                    impedance_ohm=50.0,
                    excite=True,
                    position_nm=(0, 0, 0),
                    direction="x",
                ),
                PortSpec(
                    id="P2",
                    type="lumped",
                    impedance_ohm=50.0,
                    excite=False,
                    position_nm=(0, 0, 0),
                    direction="x",
                ),
            ],
            output_format=args.format,
        )

    # Extract S-parameters
    try:
        result = extract_sparams(outputs_dir, extraction_config)
    except FileNotFoundError as e:
        sys.stderr.write(f"Error: {e}\n")
        return 1

    # Write results
    out_dir = Path(args.out).resolve() if args.out else sim_dir
    output_paths = write_extraction_result(result, out_dir)

    # Build output payload
    payload: dict[str, Any] = {
        "canonical_hash": result.canonical_hash,
        "n_ports": result.s_parameters.n_ports,
        "n_frequencies": result.s_parameters.n_frequencies,
        "f_min_hz": result.s_parameters.f_min_hz,
        "f_max_hz": result.s_parameters.f_max_hz,
        "output_paths": {k: str(v) for k, v in output_paths.items()},
        "metrics": result.metrics,
    }

    if args.json:
        _emit_json(payload, "-")
    else:
        _print_extraction_result(payload)

    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    """Handle validate command - validate simulation manifest."""
    manifest_path = Path(args.manifest).resolve()
    if not manifest_path.exists():
        sys.stderr.write(f"Error: Manifest file not found: {manifest_path}\n")
        return 1

    try:
        manifest = load_m2_manifest(manifest_path)
    except json.JSONDecodeError as e:
        sys.stderr.write(f"Error: Invalid JSON in manifest: {e}\n")
        return 1

    errors = validate_m2_manifest(manifest)

    result: dict[str, Any] = {
        "path": str(manifest_path),
        "valid": len(errors) == 0,
        "errors": errors,
    }

    if manifest:
        result["simulation_hash"] = manifest.get("simulation_hash")
        result["schema_version"] = manifest.get("schema_version")

    if args.json:
        _emit_json(result, "-")
    else:
        if result["valid"]:
            print(f"Manifest is valid: {manifest_path}")
            if result.get("simulation_hash"):
                print(f"  Simulation hash: {result['simulation_hash']}")
        else:
            print(f"Manifest has {len(errors)} error(s): {manifest_path}")
            for error in errors:
                print(f"  - {error}")

    return 0 if result["valid"] else 1


# =============================================================================
# Helper Functions
# =============================================================================


def _cast_mode(value: str) -> OpenEMSMode:
    """Cast string to OpenEMSMode."""
    if value not in ("local", "docker"):
        raise ValueError(f"Unsupported mode: {value}")
    return cast(OpenEMSMode, value)


def _resolve_toolchain(
    mode: str,
    docker_image: str,
    toolchain_path: str,
) -> OpenEMSToolchain | None:
    """Resolve toolchain from arguments."""
    if mode != "docker":
        return None
    if docker_image:
        return None
    path = Path(toolchain_path).resolve() if toolchain_path else None
    return load_openems_toolchain(path)


def _create_sim_runner(args: argparse.Namespace) -> SimulationRunner:
    """Create a SimulationRunner from command arguments."""
    solver_mode: SimulationSolverMode = getattr(args, "solver_mode", "stub")

    if solver_mode == "cli":
        mode = getattr(args, "mode", "local")
        docker_image = getattr(args, "docker_image", "")
        openems_bin = getattr(args, "openems_bin", "openEMS")

        toolchain = _resolve_toolchain(
            mode,
            docker_image,
            getattr(args, "toolchain_path", ""),
        )
        resolved_image = docker_image or (toolchain.docker_image if toolchain else None)

        openems_runner = OpenEMSRunner(
            mode=_cast_mode(mode),
            docker_image=resolved_image,
            openems_bin=openems_bin,
        )
        return SimulationRunner(mode="cli", openems_runner=openems_runner)

    return SimulationRunner(mode="stub")


def _load_json_or_yaml(path: Path) -> dict[str, Any]:
    """Load JSON or YAML file."""
    content = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml

            return yaml.safe_load(content)
        except ImportError:
            raise ImportError("PyYAML required for YAML config files")
    return json.loads(content)


def _load_geometry(
    args: argparse.Namespace,
    spec: SimulationSpec,
) -> GeometrySpec:
    """Load geometry spec from args or create stub."""
    if hasattr(args, "geometry") and args.geometry:
        geom_path = Path(args.geometry).resolve()
        geom_data = _load_json_or_yaml(geom_path)
        return GeometrySpec.model_validate(geom_data)

    # Create stub geometry from spec
    return _create_stub_geometry(spec.geometry_ref.design_hash)


def _load_geometry_from_dir(
    geometry_dir: Path | None,
    spec: SimulationSpec,
) -> GeometrySpec:
    """Load geometry from directory or create stub."""
    if geometry_dir:
        geom_path = geometry_dir / f"{spec.geometry_ref.design_hash}.json"
        if geom_path.exists():
            geom_data = _load_json_or_yaml(geom_path)
            return GeometrySpec.model_validate(geom_data)

    return _create_stub_geometry(spec.geometry_ref.design_hash)


def _create_stub_geometry(design_hash: str) -> GeometrySpec:
    """Create a stub geometry spec for testing.

    This creates a minimal valid geometry specification that can be
    used with stub simulation mode for testing CLI workflows.
    """
    return GeometrySpec(
        schema_version=1,
        design_hash=design_hash,
        coupon_family="stub",
        units="nm",
        origin="EDGE_L_CENTER",
        board=BoardOutlineSpec(
            width_nm=20_000_000,
            length_nm=80_000_000,
            corner_radius_nm=2_000_000,
        ),
        stackup=StackupSpec(
            copper_layers=4,
            thicknesses_nm={
                "L1_to_L2": 180_000,
                "L2_to_L3": 800_000,
                "L3_to_L4": 180_000,
            },
            materials=StackupMaterialsSpec(er=4.1, loss_tangent=0.02),
        ),
        layers=[
            LayerSpec(id="L1", z_nm=0, role="signal"),
            LayerSpec(id="L2", z_nm=180_000, role="ground"),
            LayerSpec(id="L3", z_nm=980_000, role="ground"),
            LayerSpec(id="L4", z_nm=1_160_000, role="signal"),
        ],
        transmission_line=TransmissionLineSpec(
            type="CPWG",
            layer="F.Cu",
            w_nm=300_000,
            gap_nm=180_000,
            length_left_nm=25_000_000,
            length_right_nm=25_000_000,
        ),
        discontinuity=DiscontinuitySpec(
            type="VIA_TRANSITION",
            parameters_nm={"via_drill_nm": 300_000, "via_diameter_nm": 650_000},
        ),
        parameters_nm={},
        derived_features={},
        dimensionless_groups={},
    )


def _emit_json(payload: dict[str, Any], target: str) -> None:
    """Emit JSON to stdout or file."""
    text = canonical_json_dumps(payload)
    if target == "-":
        sys.stdout.write(f"{text}\n")
        return
    out_path = Path(target)
    out_path.write_text(f"{text}\n", encoding="utf-8")


def _print_sim_result(payload: dict[str, Any]) -> None:
    """Print simulation result in human-readable format."""
    print("Simulation completed:")
    print(f"  Simulation hash: {payload['simulation_hash'][:12]}...")
    print(f"  Output dir: {payload['output_dir']}")
    print(f"  Cache hit: {payload['cache_hit']}")
    if "execution_time_sec" in payload:
        print(f"  Execution time: {payload['execution_time_sec']:.2f}s")
    if "convergence" in payload:
        conv = payload["convergence"]
        print(f"  Convergence: {conv['status']} ({conv['n_passed']} passed, {conv['n_failed']} failed)")
    if "sparam_path" in payload:
        print(f"  S-parameters: {payload['sparam_path']}")


def _print_batch_result(result: Any) -> None:
    """Print batch result in human-readable format."""
    print("Batch simulation completed:")
    print(f"  Total jobs: {len(result.jobs)}")
    print(f"  Completed: {result.n_completed}")
    print(f"  Failed: {result.n_failed}")
    print(f"  Skipped: {result.n_skipped}")
    print(f"  Success rate: {result.success_rate:.1f}%")
    print(f"  Total time: {result.total_time_sec:.2f}s")
    if result.config.validate_convergence:
        print(f"  Convergence passed: {result.n_convergence_passed}")
        print(f"  Convergence failed: {result.n_convergence_failed}")


def _print_status(status: dict[str, Any]) -> None:
    """Print status in human-readable format."""
    print(f"Simulation status: {status.get('status', 'unknown')}")
    print(f"  Run ID: {status.get('run_id')}")
    print(f"  Type: {status.get('type', 'unknown')}")
    print(f"  Path: {status.get('path')}")

    if status.get("type") == "single":
        if "simulation_hash" in status:
            print(f"  Simulation hash: {status['simulation_hash'][:12]}...")
        if "execution_time_sec" in status:
            print(f"  Execution time: {status['execution_time_sec']:.2f}s")
    elif status.get("type") == "batch":
        print(f"  Total jobs: {status.get('total_jobs')}")
        print(f"  Completed: {status.get('n_completed')}")
        print(f"  Failed: {status.get('n_failed')}")
        print(f"  Success rate: {status.get('success_rate', 0):.1f}%")

    if "error" in status:
        print(f"  Error: {status['error']}")


def _print_extraction_result(payload: dict[str, Any]) -> None:
    """Print extraction result in human-readable format."""
    print("S-parameter extraction completed:")
    print(f"  Hash: {payload['canonical_hash'][:12]}...")
    print(f"  Ports: {payload['n_ports']}")
    print(f"  Frequencies: {payload['n_frequencies']}")
    print(f"  Frequency range: {payload['f_min_hz'] / 1e9:.3f} - {payload['f_max_hz'] / 1e9:.3f} GHz")

    print("  Output files:")
    for fmt, path in payload["output_paths"].items():
        print(f"    {fmt}: {path}")

    metrics = payload.get("metrics", {})
    if "s11_mean_db" in metrics:
        print(f"  S11 mean: {metrics['s11_mean_db']:.2f} dB")
    if "s21_mean_db" in metrics:
        print(f"  S21 mean: {metrics['s21_mean_db']:.2f} dB")


if __name__ == "__main__":
    raise SystemExit(main())
