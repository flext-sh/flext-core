"""Utilities for resolving workspace and repository paths."""

from __future__ import annotations

from pathlib import Path


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
