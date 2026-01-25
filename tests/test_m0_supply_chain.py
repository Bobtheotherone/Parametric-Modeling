from __future__ import annotations

import re
from pathlib import Path


USES_LINE_RE = re.compile(r"^\s*-?\s*uses:\s*(.+)$")


def _strip_inline_comment(value: str) -> str:
    if "#" not in value:
        return value
    return value.split("#", 1)[0].rstrip()


def _strip_wrapping_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1].strip()
    return value


def test_actions_pinned_by_sha_if_present() -> None:
    workflows_dir = Path(".github") / "workflows"
    if not workflows_dir.exists():
        return

    issues: list[str] = []
    for wf in sorted(workflows_dir.glob("*.yml")) + sorted(workflows_dir.glob("*.yaml")):
        content = wf.read_text(encoding="utf-8")
        for idx, line in enumerate(content.splitlines(), start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            match = USES_LINE_RE.match(line)
            if not match:
                continue
            value = _strip_wrapping_quotes(_strip_inline_comment(match.group(1).strip()))
            if not value:
                issues.append(f"{wf}:{idx}: uses missing @<sha>: {value}")
                continue
            if value.startswith("./") or value.startswith("docker://"):
                continue
            if "@" not in value:
                issues.append(f"{wf}:{idx}: uses missing @<sha>: {value}")
                continue
            ref = value.split("@", 1)[1]
            if not re.fullmatch(r"[0-9a-fA-F]{40}", ref):
                issues.append(f"{wf}:{idx}: uses not pinned to full SHA: {value}")

    assert not issues, "Unpinned GitHub Actions detected:\n" + "\n".join(issues)
