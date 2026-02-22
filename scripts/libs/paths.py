"""Utilities for resolving workspace and repository paths."""

from __future__ import annotations

from pathlib import Path

from .config import MAKEFILE_FILENAME, PYPROJECT_FILENAME


def workspace_root(path: str | Path = ".") -> Path:
    """Resolve and return the absolute path to the workspace root.

    Args:
        path: A starting path, defaults to the current directory.

    Returns:
        The absolute Path to the workspace root.

    """
    return Path(path).resolve()


def repo_root_from_script(script_file: str | Path) -> Path:
    """Resolve the repository root based on the location of a script file.

    Args:
        script_file: The path to the script file (usually __file__).

    Returns:
        The absolute Path to the repository root.

    """
    return Path(script_file).resolve().parents[1]


def workspace_root_from_file(file: str | Path) -> Path:
    """Resolve workspace root by walking up from file location.

    Finds the first directory containing .git, Makefile, and pyproject.toml.

    Args:
        file: Path to a file (usually __file__).

    Returns:
        Absolute Path to the workspace root.

    Raises:
        RuntimeError: If workspace root cannot be found.

    """
    current = Path(file).resolve()
    if current.is_file():
        current = current.parent

    # Walk up the directory tree
    for parent in [current] + list(current.parents):
        markers = {".git", MAKEFILE_FILENAME, PYPROJECT_FILENAME}
        if all((parent / marker).exists() for marker in markers):
            return parent

    msg = (
        "Could not find workspace root (looking for .git, Makefile, "
        f"{PYPROJECT_FILENAME}) starting from {file}"
    )
    raise RuntimeError(msg)
