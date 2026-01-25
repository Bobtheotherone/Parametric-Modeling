from __future__ import annotations

import argparse
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path


class LoopTestError(RuntimeError):
    pass


class DirtyTreeError(LoopTestError):
    def __init__(self, message: str, status: str) -> None:
        super().__init__(message)
        self.status = status


def _run(cmd: list[str], cwd: Path) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True)
    return proc.returncode, proc.stdout, proc.stderr


def _resolve_project_root(path: str) -> Path:
    candidate = Path(path).resolve()
    rc, out, err = _run(["git", "rev-parse", "--show-toplevel"], candidate)
    if rc != 0:
        raise LoopTestError(f"Failed to resolve git root from {candidate}: {err.strip()}")
    return Path(out.strip())


def _ensure_clean(root: Path, context: str) -> None:
    rc, out, err = _run(["git", "status", "--porcelain=v1"], root)
    if rc != 0:
        raise LoopTestError(f"git status failed ({context}): {err.strip()}")
    if out.strip():
        raise DirtyTreeError(f"Working tree must be clean ({context}).", out.strip())


def _list_worktrees(root: Path) -> set[Path]:
    rc, out, err = _run(["git", "worktree", "list", "--porcelain"], root)
    if rc != 0:
        raise LoopTestError(f"git worktree list failed: {err.strip()}")
    worktrees: set[Path] = set()
    for line in out.splitlines():
        if line.startswith("worktree "):
            worktrees.add(Path(line.split(" ", 1)[1]).resolve())
    return worktrees


def _remove_worktree(root: Path, path: Path) -> None:
    rc, _, err = _run(["git", "worktree", "remove", "-f", str(path)], root)
    if rc != 0:
        raise LoopTestError(f"git worktree remove failed for {path}: {err.strip()}")


@contextmanager
def _worktree_context(root: Path, path: Path, keep: bool) -> Path:
    worktree_path = path.resolve()
    known_worktrees = _list_worktrees(root)
    if worktree_path.exists():
        if worktree_path in known_worktrees:
            _remove_worktree(root, worktree_path)
        else:
            raise LoopTestError(f"Worktree path exists but is not registered with git: {worktree_path}")

    rc, _, err = _run(["git", "worktree", "add", "-f", str(worktree_path), "HEAD"], root)
    if rc != 0:
        raise LoopTestError(f"git worktree add failed: {err.strip()}")

    try:
        yield worktree_path
    finally:
        if keep:
            return
        try:
            _remove_worktree(root, worktree_path)
        except LoopTestError as exc:
            print(f"[loop_test] WARNING: {exc}", file=sys.stderr)


def _has_flag(args: list[str], flag: str) -> bool:
    for item in args:
        if item == flag or item.startswith(f"{flag}="):
            return True
    return False


def _flag_value(args: list[str], flag: str) -> str | None:
    for idx, item in enumerate(args):
        if item == flag and idx + 1 < len(args):
            return args[idx + 1]
        if item.startswith(f"{flag}="):
            return item.split("=", 1)[1]
    return None


def _build_loop_cmd(loop_args: list[str], project_root: Path, loop_script: Path) -> list[str]:
    args = list(loop_args)
    existing_root = _flag_value(args, "--project-root")
    if existing_root is None:
        args = ["--project-root", str(project_root)] + args
    else:
        resolved = Path(existing_root).resolve()
        if resolved != project_root.resolve():
            raise LoopTestError(
                f"--project-root in loop args ({resolved}) does not match isolation root ({project_root.resolve()})."
            )
    if not _has_flag(args, "--no-agent-branch"):
        args.append("--no-agent-branch")
    return [sys.executable, "-u", str(loop_script)] + args


def _run_loop(loop_cmd: list[str], cwd: Path) -> int:
    proc = subprocess.run(loop_cmd, cwd=str(cwd))
    return proc.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run loop tests with worktree or read-only isolation.",
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--isolation", choices=["worktree", "readonly"], default="worktree")
    parser.add_argument("--worktree-path", default="/tmp/ff_loop_test")
    parser.add_argument("--keep-worktree", action="store_true")
    parser.add_argument("--loop-script", default="bridge/loop.py")

    args, loop_args = parser.parse_known_args(argv)
    if loop_args[:1] == ["--"]:
        loop_args = loop_args[1:]

    try:
        project_root = _resolve_project_root(args.project_root)
        _ensure_clean(project_root, "before loop test")

        if args.isolation == "worktree":
            with _worktree_context(project_root, Path(args.worktree_path), args.keep_worktree) as worktree:
                loop_script = (worktree / args.loop_script).resolve()
                if not loop_script.exists():
                    raise LoopTestError(f"Loop script not found: {loop_script}")
                loop_cmd = _build_loop_cmd(loop_args, worktree, loop_script)
                rc = _run_loop(loop_cmd, cwd=worktree)
            _ensure_clean(project_root, "after worktree loop test")
            return rc

        loop_script = (project_root / args.loop_script).resolve()
        if not loop_script.exists():
            raise LoopTestError(f"Loop script not found: {loop_script}")
        loop_cmd = _build_loop_cmd(loop_args, project_root, loop_script)
        rc = _run_loop(loop_cmd, cwd=project_root)
        _ensure_clean(project_root, "after readonly loop test")
        return rc
    except DirtyTreeError as exc:
        print(exc, file=sys.stderr)
        print(exc.status, file=sys.stderr)
        return 2
    except LoopTestError as exc:
        print(f"[loop_test] ERROR: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
