"""Git operations service for repository interaction.

Wraps Git commands with FlextResult error handling,
replacing bare functions with a service class.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

from flext_core.result import FlextResult, r
from flext_infra.subprocess import CommandRunner


class GitService:
    """Infrastructure service for Git operations.

    Delegates to ``CommandRunner`` for subprocess execution.
    """

    def __init__(self, runner: CommandRunner | None = None) -> None:
        """Initialize the Git service."""
        self._runner = runner or CommandRunner()

    def current_branch(self, repo_root: Path) -> FlextResult[str]:
        """Return the name of the current active branch.

        Args:
            repo_root: The root directory of the Git repository.

        Returns:
            FlextResult[str] with the branch name.

        """
        return self._runner.capture(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo_root,
        )

    def tag_exists(self, repo_root: Path, tag: str) -> FlextResult[bool]:
        """Check if a specific tag exists in the repository.

        Args:
            repo_root: The root directory of the Git repository.
            tag: The tag name to check.

        Returns:
            FlextResult[bool] with True if the tag exists.

        """
        result = self._runner.capture(
            ["git", "tag", "-l", tag],
            cwd=repo_root,
        )
        if result.is_success:
            return r[bool].ok(result.value.strip() == tag)
        return r[bool].fail(result.error or "tag check failed")

    def run(
        self,
        cmd: list[str],
        cwd: Path | None = None,
    ) -> FlextResult[str]:
        """Run an arbitrary git command and capture output.

        Args:
            cmd: Git command arguments (without 'git' prefix).
            cwd: Working directory.

        Returns:
            FlextResult[str] with command output.

        """
        return self._runner.capture(["git", *cmd], cwd=cwd)


__all__ = ["GitService"]
