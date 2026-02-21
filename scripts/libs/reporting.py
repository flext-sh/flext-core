"""Reporting utilities for repository operations."""

from __future__ import annotations

from pathlib import Path


def reports_root(workspace_root: Path) -> Path:
    """Return the root directory for repository reports.

    Args:
        workspace_root: The root directory of the workspace.

    Returns:
        The Path to the .reports directory.

    """
    return workspace_root / ".reports"


def ensure_report_dir(workspace_root: Path, *parts: str) -> Path:
    """Ensure a specific report directory exists and return its path.

    Args:
        workspace_root: The root directory of the workspace.
        parts: Subdirectory parts to join with the reports root.

    Returns:
        The Path to the ensured report directory.

    """
    path = reports_root(workspace_root).joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path
