"""Utilities for interacting with Git repositories."""

from __future__ import annotations

from pathlib import Path

from libs.subprocess import run_capture


def current_branch(repo_root: Path) -> str:
    """Return the name of the current active branch in a Git repository.

    Args:
        repo_root: The root directory of the Git repository.

    Returns:
        The name of the current branch.

    """
    return run_capture(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root)


def tag_exists(repo_root: Path, tag: str) -> bool:
    """Check if a specific tag exists in a Git repository.

    Args:
        repo_root: The root directory of the Git repository.
        tag: The tag name to check.

    Returns:
        True if the tag exists, False otherwise.

    """
    value = run_capture(["git", "tag", "-l", tag], cwd=repo_root)
    return value.strip() == tag
