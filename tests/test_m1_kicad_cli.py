from __future__ import annotations

from pathlib import Path

import pytest

import formula_foundry.coupongen.kicad.cli as cli_module
from formula_foundry.coupongen.kicad import (
    DEFAULT_TIMEOUT_SEC,
    KicadCliRunner,
    KicadCliTimeoutError,
    KicadErrorCode,
    ParsedKicadError,
    build_define_var_args,
    build_drc_args,
    parse_kicad_error,
)
from formula_foundry.coupongen.kicad.runners import (
    DEFAULT_DOCKER_TIMEOUT_SEC,
    DockerKicadRunner,
    DockerKicadTimeoutError,
)


def test_module_imported_from_workspace() -> None:
    """Verify cli module is imported from workspace, not a stale installed package.

    This hardening test catches CI regressions where a stale version of the
    package might be installed in site-packages, causing tests to run against
    the wrong code.
    """
    module_path = Path(cli_module.__file__).resolve()
    # The module should be under src/formula_foundry/ in the workspace
    # or under site-packages if installed with pip install -e .
    # Either way, it should contain our expected default severity
    # Check that the function has severity="all" as default
    import inspect

    from formula_foundry.coupongen.kicad.cli import build_drc_args as fn

    sig = inspect.signature(fn)
    default_severity = sig.parameters["severity"].default
    assert default_severity == "all", (
        f"build_drc_args has wrong default severity: {default_severity!r}. "
        f"Expected 'all'. Module loaded from: {module_path}. "
        "This may indicate a stale installed package in site-packages."
    )


def test_kicad_cli_runner_modes(tmp_path: Path) -> None:
    runner_local = KicadCliRunner(mode="local")
    local_cmd = runner_local.build_command(["pcb", "drc"], workdir=tmp_path)

    assert local_cmd[0] == "kicad-cli"

    runner_docker = KicadCliRunner(mode="docker", docker_image="kicad/kicad:9.0.7@sha256:deadbeef")
    docker_cmd = runner_docker.build_command(["pcb", "drc"], workdir=tmp_path)

    assert docker_cmd[0] == "docker"
    assert "kicad/kicad:9.0.7@sha256:deadbeef" in docker_cmd


def test_kicad_cli_runner_docker_includes_user_flag(tmp_path: Path) -> None:
    """Docker command must include --user flag for bind-mount permissions.

    This is critical for CI environments where the host user ID (e.g., 1001
    on GitHub Actions) differs from the container user (uid 1000 in kicad/kicad).
    Without --user, the container cannot write to bind-mounted directories.
    """
    import os

    runner = KicadCliRunner(mode="docker", docker_image="kicad/kicad:9.0.7")
    cmd = runner.build_command(["--version"], workdir=tmp_path)

    # Verify --user flag is present with host UID:GID
    assert "--user" in cmd, "Docker command must include --user flag"
    user_idx = cmd.index("--user")
    user_value = cmd[user_idx + 1]

    # Verify format is uid:gid
    assert ":" in user_value, "--user value must be uid:gid format"
    uid, gid = user_value.split(":")
    assert uid.isdigit(), "UID must be numeric"
    assert gid.isdigit(), "GID must be numeric"

    # Verify it matches host user
    assert int(uid) == os.getuid(), f"UID should be {os.getuid()}, got {uid}"
    assert int(gid) == os.getgid(), f"GID should be {os.getgid()}, got {gid}"


def test_kicad_cli_runner_docker_sets_home_env(tmp_path: Path) -> None:
    """Docker command must set HOME=/tmp for numeric UID without passwd entry.

    When running as --user uid:gid, the numeric UID may not have a passwd
    entry in the container. KiCad needs a writable HOME for config files.
    """
    runner = KicadCliRunner(mode="docker", docker_image="kicad/kicad:9.0.7")
    cmd = runner.build_command(["--version"], workdir=tmp_path)

    # Find all -e flags and their values
    env_values = []
    for i, arg in enumerate(cmd):
        if arg == "-e" and i + 1 < len(cmd):
            env_values.append(cmd[i + 1])

    assert "HOME=/tmp" in env_values, "Docker command must set HOME=/tmp"


def test_kicad_cli_runner_local_mode_no_user_flag(tmp_path: Path) -> None:
    """Local mode should NOT include Docker-specific flags."""
    runner = KicadCliRunner(mode="local")
    cmd = runner.build_command(["--version"], workdir=tmp_path)

    assert "--user" not in cmd, "Local mode should not have --user flag"
    assert "-e" not in cmd, "Local mode should not have -e flag"


def test_drc_invocation_flags(tmp_path: Path) -> None:
    board = tmp_path / "coupon.kicad_pcb"
    report = tmp_path / "drc.json"
    args = build_drc_args(board, report)

    assert "--severity-all" in args
    assert "--exit-code-violations" in args
    assert "--format" in args
    assert "json" in args
    assert "--output" in args


# ============================================================================
# Tests for REQ-M1-015: Timeout handling
# ============================================================================


def test_default_timeout_constant() -> None:
    """Verify default timeout constants are defined."""
    assert DEFAULT_TIMEOUT_SEC == 300.0
    assert DEFAULT_DOCKER_TIMEOUT_SEC == 300.0


def test_kicad_cli_runner_default_timeout() -> None:
    """Verify KicadCliRunner has default timeout set."""
    runner = KicadCliRunner(mode="local")
    assert runner.default_timeout == DEFAULT_TIMEOUT_SEC


def test_kicad_cli_runner_custom_timeout() -> None:
    """Verify KicadCliRunner accepts custom timeout."""
    runner = KicadCliRunner(mode="local", default_timeout=60.0)
    assert runner.default_timeout == 60.0


def test_kicad_cli_runner_no_timeout() -> None:
    """Verify KicadCliRunner can disable timeout."""
    runner = KicadCliRunner(mode="local", default_timeout=None)
    assert runner.default_timeout is None


def test_docker_runner_default_timeout() -> None:
    """Verify DockerKicadRunner has default timeout set."""
    runner = DockerKicadRunner(docker_image="kicad/kicad:9.0.7")
    assert runner.default_timeout == DEFAULT_DOCKER_TIMEOUT_SEC


def test_kicad_cli_timeout_error_attributes() -> None:
    """Verify KicadCliTimeoutError has expected attributes."""
    error = KicadCliTimeoutError(timeout_sec=60.0, command=["kicad-cli", "--version"])
    assert error.timeout_sec == 60.0
    assert error.command == ["kicad-cli", "--version"]
    assert "60" in str(error)
    assert "timed out" in str(error).lower()


def test_docker_kicad_timeout_error_attributes() -> None:
    """Verify DockerKicadTimeoutError has expected attributes."""
    error = DockerKicadTimeoutError(timeout_sec=120.0, command=["docker", "run"])
    assert error.timeout_sec == 120.0
    assert error.command == ["docker", "run"]
    assert "120" in str(error)


# ============================================================================
# Tests for REQ-M1-015: Variable injection via --define-var
# ============================================================================


def test_build_define_var_args_empty() -> None:
    """Verify empty variables returns empty list."""
    assert build_define_var_args(None) == []
    assert build_define_var_args({}) == []


def test_build_define_var_args_single() -> None:
    """Verify single variable is formatted correctly."""
    args = build_define_var_args({"COUPON_ID": "test-001"})
    assert args == ["--define-var", "COUPON_ID=test-001"]


def test_build_define_var_args_multiple() -> None:
    """Verify multiple variables are formatted correctly."""
    args = build_define_var_args({"COUPON_ID": "test-001", "VERSION": "1.0"})
    # Check both variables are present (order may vary)
    assert len(args) == 4
    assert "--define-var" in args
    assert "COUPON_ID=test-001" in args
    assert "VERSION=1.0" in args


def test_build_define_var_args_with_special_values() -> None:
    """Verify variables with special characters in values work."""
    args = build_define_var_args({"TITLE": "Test Board (v2)"})
    assert args == ["--define-var", "TITLE=Test Board (v2)"]


def test_build_define_var_args_with_underscore_name() -> None:
    """Verify variable names with underscores work."""
    args = build_define_var_args({"MY_VAR_NAME": "value"})
    assert args == ["--define-var", "MY_VAR_NAME=value"]


def test_build_define_var_args_invalid_name_starts_with_number() -> None:
    """Verify variable names starting with numbers are rejected."""
    with pytest.raises(ValueError, match="Invalid variable name"):
        build_define_var_args({"1INVALID": "value"})


def test_build_define_var_args_invalid_name_special_chars() -> None:
    """Verify variable names with special characters are rejected."""
    with pytest.raises(ValueError, match="Invalid variable name"):
        build_define_var_args({"INVALID-NAME": "value"})


def test_build_define_var_args_invalid_name_spaces() -> None:
    """Verify variable names with spaces are rejected."""
    with pytest.raises(ValueError, match="Invalid variable name"):
        build_define_var_args({"INVALID NAME": "value"})


# ============================================================================
# Tests for REQ-M1-015: Error parsing
# ============================================================================


def test_kicad_error_code_values() -> None:
    """Verify KicadErrorCode enum has expected values."""
    assert KicadErrorCode.SUCCESS == 0
    assert KicadErrorCode.GENERAL_ERROR == 1
    assert KicadErrorCode.INVALID_ARGUMENTS == 2
    assert KicadErrorCode.FILE_LOAD_ERROR == 3
    assert KicadErrorCode.FILE_WRITE_ERROR == 4
    assert KicadErrorCode.DRC_VIOLATIONS == 5


def test_parse_kicad_error_success() -> None:
    """Verify parsing of successful exit."""
    result = parse_kicad_error(0, "Done", "")
    assert result.error_code == 0
    assert result.error_type == "success"
    assert not result.is_file_error
    assert not result.is_drc_error


def test_parse_kicad_error_file_load() -> None:
    """Verify parsing of file load error."""
    result = parse_kicad_error(3, "", "Failed to load 'board.kicad_pcb'")
    assert result.error_code == 3
    assert result.error_type == "file_load_error"
    assert result.is_file_error
    assert result.file_path == "board.kicad_pcb"


def test_parse_kicad_error_drc_violations() -> None:
    """Verify parsing of DRC violations error."""
    result = parse_kicad_error(5, "Found 3 violations", "")
    assert result.error_code == 5
    assert result.error_type == "drc_violations"
    assert result.is_drc_error
    assert not result.is_file_error


def test_parse_kicad_error_unknown_code() -> None:
    """Verify parsing of unknown error code."""
    result = parse_kicad_error(99, "", "Unknown error")
    assert result.error_code == 99
    assert result.error_type == "unknown_error_99"


def test_parsed_kicad_error_dataclass() -> None:
    """Verify ParsedKicadError dataclass attributes."""
    error = ParsedKicadError(
        error_code=3,
        error_type="file_load_error",
        message="Failed to load board",
        file_path="/path/to/board.kicad_pcb",
        details=["Additional info"],
    )
    assert error.error_code == 3
    assert error.error_type == "file_load_error"
    assert error.message == "Failed to load board"
    assert error.file_path == "/path/to/board.kicad_pcb"
    assert error.details == ["Additional info"]
    assert error.is_file_error
    assert not error.is_drc_error


# ============================================================================
# Tests for command building with variables
# ============================================================================


def test_kicad_cli_runner_build_command_includes_vars(tmp_path: Path) -> None:
    """Verify build_command can include define-var args."""
    runner = KicadCliRunner(mode="local")
    # Variables are passed through run(), not build_command() directly
    # This test verifies the command structure
    cmd = runner.build_command(["--define-var", "TEST=value", "pcb", "export", "gerbers"], workdir=tmp_path)
    assert "--define-var" in cmd
    assert "TEST=value" in cmd
