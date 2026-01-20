"""Test asserting Gemini has been removed from the codebase.

We previously had a Gemini wrapper but have since removed it.
This test ensures Gemini artifacts do not re-appear.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_gemini_wrapper_does_not_exist() -> None:
    """Gemini wrapper script should not exist."""
    gemini_path = ROOT / "bridge" / "agents" / "gemini.sh"
    assert not gemini_path.exists(), f"Gemini wrapper should be removed but found at {gemini_path}"


def test_config_has_no_gemini_agent() -> None:
    """Config should not contain gemini agent."""
    import json

    config_path = ROOT / "bridge" / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert "gemini" not in config.get("agents", {}), "Config should not contain gemini agent"
    assert "gemini" not in config.get("enable_agents", []), "enable_agents should not include gemini"
    assert "gemini" not in config.get("fallback_order", []), "fallback_order should not include gemini"
