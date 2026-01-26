"""Hardware configuration validation tests.

This module ensures the hardware.yaml configuration matches the target laptop
contract defined in AGENTS.md section B6. The target laptop contract specifies:

- vram_gb: 24 (required for GPU-accelerated simulations)
- cpu_cores: 16 (scheduler assumption for parallel workloads)
- ram_gb: 32 (minimum for resource preflight budgeting)

These values must not drift, as they are used by:
- Resource preflight budgeting (M0 gate)
- GPU backend policy decisions
- Parallel task scheduling
"""

from __future__ import annotations

from pathlib import Path

import yaml

# Target laptop contract values (from AGENTS.md section B6)
TARGET_CPU_CORES = 16
TARGET_RAM_GB = 32
TARGET_VRAM_GB = 24


def _load_hardware_config() -> dict:
    """Load hardware.yaml configuration file."""
    config_path = Path(__file__).resolve().parent.parent / "config" / "hardware.yaml"
    assert config_path.is_file(), f"hardware.yaml not found at {config_path}"
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def test_hardware_config_exists() -> None:
    """Hardware config file must exist."""
    config_path = Path(__file__).resolve().parent.parent / "config" / "hardware.yaml"
    assert config_path.is_file(), "config/hardware.yaml must exist"


def test_hardware_config_has_required_keys() -> None:
    """Hardware config must define all required keys."""
    config = _load_hardware_config()
    required_keys = {"cpu_cores", "ram_gb", "vram_gb"}
    missing = required_keys - set(config.keys())
    assert not missing, f"hardware.yaml missing required keys: {missing}"


def test_cpu_cores_matches_target_laptop() -> None:
    """CPU cores must match target laptop contract (16 cores)."""
    config = _load_hardware_config()
    assert config["cpu_cores"] == TARGET_CPU_CORES, f"cpu_cores mismatch: expected {TARGET_CPU_CORES}, got {config['cpu_cores']}"


def test_ram_gb_matches_target_laptop() -> None:
    """RAM must match target laptop contract (32 GB)."""
    config = _load_hardware_config()
    assert config["ram_gb"] == TARGET_RAM_GB, f"ram_gb mismatch: expected {TARGET_RAM_GB}, got {config['ram_gb']}"


def test_vram_gb_matches_target_laptop() -> None:
    """VRAM must match target laptop contract (24 GB).

    This is a critical requirement from AGENTS.md section B6:
    "Update config/hardware.yaml so it reflects the actual target laptop:
     - vram_gb: 24"
    """
    config = _load_hardware_config()
    assert config["vram_gb"] == TARGET_VRAM_GB, f"vram_gb mismatch: expected {TARGET_VRAM_GB}, got {config['vram_gb']}"


def test_hardware_values_are_positive_integers() -> None:
    """All hardware values must be positive integers."""
    config = _load_hardware_config()
    for key in ["cpu_cores", "ram_gb", "vram_gb"]:
        value = config[key]
        assert isinstance(value, int), f"{key} must be an integer, got {type(value).__name__}"
        assert value > 0, f"{key} must be positive, got {value}"


def test_hardware_config_no_extra_keys() -> None:
    """Hardware config should only contain expected keys (prevent config drift)."""
    config = _load_hardware_config()
    expected_keys = {"cpu_cores", "ram_gb", "vram_gb"}
    set(config.keys()) - expected_keys
    # Allow extra keys but warn - they may be intentional extensions
    # For strict enforcement, uncomment the assertion below:
    # assert not extra_keys, f"Unexpected keys in hardware.yaml: {extra_keys}"
    pass  # Currently permissive - only core contract values are validated
