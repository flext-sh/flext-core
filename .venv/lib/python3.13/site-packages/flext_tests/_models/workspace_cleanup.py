"""Typed workspace cleanup contracts for flext-tests consumers."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from flext_infra import m, u


class FlextTestsWorkspaceCleanupModelsMixin:
    """Immutable cleanup policy, plan, and execution report models."""

    class WorkspaceCleanupPolicy(m.Value):
        """Config-owned exact residue paths eligible for cleanup."""

        residues: Annotated[
            tuple[Path, ...],
            u.Field(
                strict=False,
                description="Exact workspace-relative development residue paths.",
            ),
        ]

    class WorkspaceCleanupRequest(m.Value):
        """Runtime root composed with the original cleanup policy object."""

        workspace_root: Annotated[
            Path, u.Field(strict=False, description="Exact Git workspace root.")
        ]
        policy: Annotated[
            FlextTestsWorkspaceCleanupModelsMixin.WorkspaceCleanupPolicy,
            u.Field(description="Original config-owned cleanup policy."),
        ]

    class WorkspaceCleanupCandidate(m.Value):
        """One validated ignored residue in a cleanup plan."""

        relative_path: Annotated[
            Path, u.Field(strict=False, description="Workspace-relative residue path.")
        ]
        path: Annotated[
            Path, u.Field(strict=False, description="Absolute lexical residue path.")
        ]
        kind: Annotated[
            Literal["file", "directory", "symlink"],
            u.Field(description="Observed filesystem kind during planning."),
        ]
        # NOTE (multi-agent): bind apply to the exact dry-run filesystem state.
        fingerprint: Annotated[
            str, u.Field(description="SHA-256 fingerprint of the planned residue tree.")
        ]

    class WorkspaceCleanupPlan(m.Value):
        """Deterministic dry-run plan retaining its source request."""

        request: Annotated[
            FlextTestsWorkspaceCleanupModelsMixin.WorkspaceCleanupRequest,
            u.Field(description="Original cleanup request."),
        ]
        candidates: Annotated[
            tuple[FlextTestsWorkspaceCleanupModelsMixin.WorkspaceCleanupCandidate, ...],
            u.Field(description="Sorted validated cleanup candidates."),
        ]

    class WorkspaceCleanupReport(m.Value):
        """Applied cleanup report retaining the exact validated plan."""

        plan: Annotated[
            FlextTestsWorkspaceCleanupModelsMixin.WorkspaceCleanupPlan,
            u.Field(description="Exact dry-run plan applied by the operation."),
        ]
        removed: Annotated[
            tuple[Path, ...],
            u.Field(strict=False, description="Sorted paths removed successfully."),
        ]


__all__: tuple[str, ...] = ("FlextTestsWorkspaceCleanupModelsMixin",)
