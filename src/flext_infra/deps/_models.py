"""Domain models for the deps subpackage."""

from __future__ import annotations

from pydantic import Field

from flext_core import FlextModels


class FlextInfraDepsModels:
    """Models for dependency detection and management."""

    class DependencyReport(FlextModels.ArbitraryTypesModel):
        """Report of dependency detection for a single project."""

        project: str = Field(min_length=1, description="Project name")
        missing: list[str] = Field(
            default_factory=list, description="Missing dependencies"
        )
        unused: list[str] = Field(
            default_factory=list, description="Unused dependencies"
        )
        outdated: list[str] = Field(
            default_factory=list, description="Outdated dependencies"
        )


__all__ = ["FlextInfraDepsModels"]
