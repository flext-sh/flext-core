"""Canonical Git worktree composition for ``u.Infra``."""

from __future__ import annotations

from flext_infra._utilities._git.worktree_removal import (
    FlextInfraUtilitiesGitWorktreeRemovalMixin,
)


class FlextInfraUtilitiesGitWorktreeMixin(FlextInfraUtilitiesGitWorktreeRemovalMixin):
    """Compose worktree discovery, materialization, checkpoint, patch, and removal."""


__all__: list[str] = ["FlextInfraUtilitiesGitWorktreeMixin"]
