from __future__ import annotations

from tools import git_guard


def test_find_tracked_egg_info() -> None:
    paths = [
        "src/formula_foundry.egg-info/PKG-INFO",
        "src/formula_foundry.egg-info/SOURCES.txt",
        "src/formula_foundry/__init__.py",
        "docs/egg-info.txt",
    ]
    hits = git_guard._find_tracked_egg_info(paths)

    assert hits == [
        "src/formula_foundry.egg-info/PKG-INFO",
        "src/formula_foundry.egg-info/SOURCES.txt",
    ]
