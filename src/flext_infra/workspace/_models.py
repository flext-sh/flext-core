"""Domain models for the workspace subpackage."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from pydantic import Field

from flext_core import FlextModels


class FlextInfraWorkspaceModels:
    """Models for workspace discovery, sync, and migration.

    Canonical base policy:
    - ``ArbitraryTypesModel`` for mutable discovery and migration payloads.
    - ``FrozenStrictModel`` reserved for immutable workspace config contracts.
    """

    class ProjectInfo(FlextModels.ArbitraryTypesModel):
        """Discovered project metadata for workspace operations."""

        name: str = Field(min_length=1, description="Project name")
        path: Path = Field(description="Absolute or relative project path")
        stack: str = Field(min_length=1, description="Primary technology stack")
        has_tests: bool = Field(default=False, description="Project has test suite")
        has_src: bool = Field(default=True, description="Project has source directory")

    class SyncResult(FlextModels.ArbitraryTypesModel):
        """Result payload for sync operations."""

        files_changed: int = Field(default=0, ge=0, description="Total changed files")
        source: Path = Field(description="Sync source path")
        target: Path = Field(description="Sync target path")
        timestamp: datetime = Field(
            default_factory=lambda: datetime.now(UTC),
            description="Execution timestamp in UTC",
        )

    class MigrationResult(FlextModels.ArbitraryTypesModel):
        """Migration operation outcome with applied changes and errors."""

        project: str = Field(min_length=1, description="Project identifier")
        changes: list[str] = Field(default_factory=list, description="Applied changes")
        errors: list[str] = Field(default_factory=list, description="Migration errors")


__all__ = ["FlextInfraWorkspaceModels"]
