"""Git operations service for repository interaction.

Wraps Git commands with r error handling,
replacing bare functions with a service class.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path
from typing import override

from flext_core import r, s
from flext_infra import FlextInfraCommandRunner, c, p


class FlextInfraGitService(s[str]):
    """Infrastructure service for Git operations.

    Delegates to ``FlextInfraCommandRunner`` for subprocess execution.
    """

    def __init__(self, runner: p.Infra.SafetyRunner | None = None) -> None:
        """Initialize the Git service."""
        super().__init__()
        selected_runner = runner or FlextInfraCommandRunner()
        self._runner: p.Infra.SafetyRunner = selected_runner

    def current_branch(self, repo_root: Path) -> r[str]:
        """Return the name of the current active branch.

        Args:
            repo_root: The root directory of the Git repository.

        Returns:
            r[str] with the branch name.

        """
        return self._runner.capture(
            [c.Infra.Cli.GIT, "rev-parse", "--abbrev-ref", c.Infra.Git.HEAD],
            cwd=repo_root,
        )

    @override
    def execute(self) -> r[str]:
        """Execute the service (required by s base class)."""
        return r[str].ok("")

    def run(
        self,
        cmd: list[str],
        cwd: Path | None = None,
    ) -> r[str]:
        """Run an arbitrary git command and capture output.

        Args:
            cmd: Git command arguments (without 'git' prefix).
            cwd: Working directory.

        Returns:
            r[str] with command output.

        """
        return self._runner.capture([c.Infra.Cli.GIT, *cmd], cwd=cwd)

    def tag_exists(self, repo_root: Path, tag: str) -> r[bool]:
        """Check if a specific tag exists in the repository.

        Args:
            repo_root: The root directory of the Git repository.
            tag: The tag name to check.

        Returns:
            r[bool] with True if the tag exists.

        """
        result = self._runner.capture(
            [c.Infra.Cli.GIT, "tag", "-l", tag],
            cwd=repo_root,
        )
        if result.is_success:
            return r[bool].ok(result.value.strip() == tag)
        return r[bool].fail(result.error or "tag check failed")


__all__ = ["FlextInfraGitService"]
