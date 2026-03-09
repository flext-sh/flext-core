"""Git-specific fixtures for FLEXT infra tests.

Provides fixtures for creating real git repositories with commits.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def real_git_repo(tmp_path: Path) -> Path:
    """Create a real git repository with initial commit.

    Args:
        tmp_path: Pytest temporary directory

    Returns:
        Path to git repository root

    """
    repo_root = tmp_path / "git_repo"
    repo_root.mkdir()

    # Initialize git repo
    subprocess.run(
        ["git", "init"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )

    # Configure git user for commits
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )

    # Create initial file and commit
    (repo_root / "README.md").write_text("# Test Repository\n")
    subprocess.run(
        ["git", "add", "README.md"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )

    return repo_root


__all__ = ["real_git_repo"]
