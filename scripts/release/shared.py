#!/usr/bin/env python3
# Owner-Skill: .claude/skills/scripts-maintenance/SKILL.md
"""Shared utilities for release scripts."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

# pylint: disable=wrong-import-position
from libs.discovery import ProjectInfo  # noqa: E402
from libs.paths import workspace_root as _workspace_root  # noqa: E402
from libs.selection import resolve_projects as _resolve_projects  # noqa: E402
from libs.subprocess import (  # noqa: E402
    run_capture as _run_capture,
    run_checked as _run_checked,
)
from libs.versioning import (  # noqa: E402
    bump_version as _bump_version,
    parse_semver as _parse_semver,
)

Project = ProjectInfo


def workspace_root(path: str | Path = ".") -> Path:
    """Resolve and return the absolute path to the workspace root.

    Args:
        path: A starting path, defaults to the current directory.

    Returns:
        The absolute Path to the workspace root.

    """
    return _workspace_root(path)


def resolve_projects(root: Path, names: list[str]) -> list[Project]:
    """Resolve release project names into ProjectInfo structures.

    This function wraps the standard project resolution to provide more
    specific error messages for the release process.

    Args:
        root: The root directory of the workspace.
        names: A list of project names to resolve.

    Returns:
        A list of resolved ProjectInfo structures.

    Raises:
        RuntimeError: If any of the requested project names are unknown.

    """
    try:
        return _resolve_projects(root, names)
    except RuntimeError as exc:
        raise RuntimeError(
            str(exc).replace("unknown projects", "unknown release projects")
        ) from exc


def parse_semver(version: str) -> tuple[int, int, int]:
    """Parse a semantic version string into a tuple of (major, minor, patch).

    Args:
        version: The semantic version string to parse.

    Returns:
        A tuple containing (major, minor, patch) integers.

    """
    return _parse_semver(version)


def bump_version(current_version: str, bump: str) -> str:
    """Bump a version string by major, minor, or patch level.

    Args:
        current_version: The current version string.
        bump: The level to bump ("major", "minor", or "patch").

    Returns:
        The bumped version string.

    """
    return _bump_version(current_version, bump)


def run_checked(command: list[str], cwd: Path | None = None) -> None:
    """Run a command and raise RuntimeError if it fails.

    Args:
        command: The command line arguments as a list.
        cwd: Optional working directory for the command.

    """
    _run_checked(command, cwd=cwd)


def run_capture(command: list[str], cwd: Path | None = None) -> str:
    """Run a command, capture its output, and raise RuntimeError if it fails.

    Args:
        command: The command line arguments as a list.
        cwd: Optional working directory for the command.

    Returns:
        The stripped standard output of the command.

    """
    return _run_capture(command, cwd=cwd)
