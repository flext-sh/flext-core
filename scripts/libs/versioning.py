"""Utilities for managing project and workspace versioning."""

from __future__ import annotations

import re
from pathlib import Path


def parse_semver(version: str) -> tuple[int, int, int]:
    """Parse a semantic version string into a tuple of (major, minor, patch).

    Args:
        version: The version string to parse.

    Returns:
        A tuple of three integers.

    Raises:
        RuntimeError: If the version string is not a valid semantic version.

    """
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)(?:-dev)?$", version)
    if not match:
        msg = f"invalid version: {version}"
        raise RuntimeError(msg)
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def bump_version(version: str, bump_type: str) -> str:
    """Bump a semantic version string according to the specified bump type.

    Args:
        version: The current version string.
        bump_type: The type of bump ("major", "minor", or "patch").

    Returns:
        The bumped version string.

    Raises:
        RuntimeError: If the bump type is invalid.

    """
    major, minor, patch = parse_semver(version)
    if bump_type == "major":
        major += 1
        minor = 0
        patch = 0
    elif bump_type == "minor":
        minor += 1
        patch = 0
    elif bump_type == "patch":
        patch += 1
    else:
        msg = f"invalid bump type: {bump_type}"
        raise RuntimeError(msg)
    return f"{major}.{minor}.{patch}"


def release_tag_from_branch(branch: str) -> str:
    """Extract a release tag name from a release branch name.

    Args:
        branch: The release branch name (e.g., "release/1.2.3").

    Returns:
        The corresponding tag name (e.g., "v1.2.3").

    """
    if branch.startswith("release/"):
        return f"v{branch.removeprefix('release/')}"
    return ""


def current_workspace_version(workspace_root: Path) -> str:
    """Read the current version from the main pyproject.toml file.

    Args:
        workspace_root: The root directory of the workspace.

    Returns:
        The version string.

    """
    pyproject = workspace_root / "pyproject.toml"
    content = pyproject.read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"(.+?)"', content, re.MULTILINE)
    if not match:
        msg = "version not found in pyproject.toml"
        raise RuntimeError(msg)
    return match.group(1)


def replace_project_version(project_path: Path, version: str) -> None:
    """Update the version field in a project's pyproject.toml file.

    Args:
        project_path: The directory containing the pyproject.toml file.
        version: The new version string to set.

    """
    pyproject = project_path / "pyproject.toml"
    content = pyproject.read_text(encoding="utf-8")
    updated = re.sub(
        r'^version\s*=\s*".+?"',
        f'version = "{version}"',
        content,
        count=1,
        flags=re.MULTILINE,
    )
    _ = pyproject.write_text(updated, encoding="utf-8")
