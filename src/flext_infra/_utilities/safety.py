"""Domain-agnostic safety primitives for workspace checkpointing.

Provides static helpers for validating workspace cleanliness, creating a
checkpoint stash, and rolling back to a checkpoint stash reference.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from flext_core import r
from flext_infra._utilities.git import FlextInfraUtilitiesGit
from flext_infra._utilities.subprocess import FlextInfraUtilitiesSubprocess
from flext_infra.constants import FlextInfraConstants as c


class FlextInfraUtilitiesSafety:
    """Static safety helpers for git-backed workspace protection."""

    @staticmethod
    def validate_workspace_clean(workspace_root: Path) -> r[bool]:
        """Check if workspace has no uncommitted changes.

        Returns success with True when workspace is clean, False when dirty.
        """
        has_changes = FlextInfraUtilitiesGit.git_has_changes(workspace_root)
        return has_changes.fold(
            on_failure=lambda e: r[bool].fail(e or "git status failed"),
            on_success=lambda v: r[bool].ok(not v),
        )

    @staticmethod
    def create_checkpoint(
        workspace_root: Path,
        *,
        label: str = "flext-safety-checkpoint",
    ) -> r[str]:
        """Create a timestamped git stash checkpoint and return its reference."""
        if not FlextInfraUtilitiesGit.git_is_repo(workspace_root):
            return r[str].ok("")

        has_changes = FlextInfraUtilitiesGit.git_has_changes(workspace_root)
        if has_changes.is_failure:
            return r[str].fail(has_changes.error or "git status failed")
        if not has_changes.value:
            return r[str].ok("")

        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        push_result = FlextInfraUtilitiesGit.git_run_checked(
            [
                "stash",
                "push",
                "-m",
                f"{label}:{timestamp}",
                "--include-untracked",
            ],
            cwd=workspace_root,
        )
        if push_result.is_failure:
            return r[str].fail(push_result.error or "git stash push failed")

        stash_list = FlextInfraUtilitiesGit.git_run(
            ["stash", "list"], cwd=workspace_root
        )
        return stash_list.fold(
            on_failure=lambda e: r[str].fail(e or "git stash list failed"),
            on_success=lambda v: r[str].ok(v.strip()),
        )

    @staticmethod
    def rollback_to_checkpoint(workspace_root: Path, stash_ref: str = "") -> r[bool]:
        """Restore workspace from stash checkpoint using `git stash pop`."""
        if not FlextInfraUtilitiesGit.git_is_repo(workspace_root):
            return r[bool].ok(True)

        command = [c.Infra.Cli.GIT, "stash", "pop"]
        if stash_ref:
            command.append(stash_ref)
        return FlextInfraUtilitiesSubprocess.run_checked(command, cwd=workspace_root)


__all__ = ["FlextInfraUtilitiesSafety"]
