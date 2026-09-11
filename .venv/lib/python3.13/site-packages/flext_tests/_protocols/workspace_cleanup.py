"""Read-only workspace cleanup protocols for flext-tests."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol


class FlextTestsWorkspaceCleanupProtocolsMixin:
    """Structural contracts for cleanup models crossing public interfaces."""

    class WorkspaceCleanupPolicy(Protocol):
        """Config-owned cleanup policy surface."""

        @property
        def residues(self) -> tuple[Path, ...]: ...

    class WorkspaceCleanupRequest(Protocol):
        """Runtime cleanup request surface."""

        @property
        def workspace_root(self) -> Path: ...

        @property
        def policy(
            self,
        ) -> FlextTestsWorkspaceCleanupProtocolsMixin.WorkspaceCleanupPolicy: ...

    class WorkspaceCleanupCandidate(Protocol):
        """Validated cleanup candidate surface."""

        @property
        def relative_path(self) -> Path: ...

        @property
        def path(self) -> Path: ...

        @property
        def kind(self) -> Literal["file", "directory", "symlink"]: ...

        # NOTE (multi-agent): expose immutable dry-run state for stale-plan checks.
        @property
        def fingerprint(self) -> str: ...

    class WorkspaceCleanupPlan(Protocol):
        """Deterministic dry-run plan surface."""

        @property
        def request(
            self,
        ) -> FlextTestsWorkspaceCleanupProtocolsMixin.WorkspaceCleanupRequest: ...

        @property
        def candidates(
            self,
        ) -> tuple[
            FlextTestsWorkspaceCleanupProtocolsMixin.WorkspaceCleanupCandidate, ...
        ]: ...

    class WorkspaceCleanupReport(Protocol):
        """Applied cleanup report surface."""

        @property
        def plan(
            self,
        ) -> FlextTestsWorkspaceCleanupProtocolsMixin.WorkspaceCleanupPlan: ...

        @property
        def removed(self) -> tuple[Path, ...]: ...


__all__: tuple[str, ...] = ("FlextTestsWorkspaceCleanupProtocolsMixin",)
