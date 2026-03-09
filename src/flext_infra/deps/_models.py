"""Domain models for the deps subpackage."""

from __future__ import annotations

from pydantic import Field

from flext_core import FlextModels
from flext_infra import t


class FlextInfraDepsModels:
    """Models for dependency detection and modernization reporting."""

    class DependencyReport(FlextModels.ArbitraryTypesModel):
        """Report of dependency detection for a single project."""

        project: str = Field(min_length=1, description="Project name")
        missing: list[str] = Field(
            default_factory=list,
            description="Missing dependencies",
        )
        unused: list[str] = Field(
            default_factory=list,
            description="Unused dependencies",
        )
        outdated: list[str] = Field(
            default_factory=list,
            description="Outdated dependencies",
        )

    class ModernizerFileChanges(FlextModels.ArbitraryTypesModel):
        """Modernizer changes for one pyproject file."""

        file: str = Field(min_length=1, description="Relative pyproject path")
        changes: list[str] = Field(default_factory=list, description="Applied changes")


class FlextInfraDependencyDetectionModels(FlextModels):
    """Models for dependency detection reports and analysis results."""

    class DeptryIssueGroups(FlextModels.ArbitraryTypesModel):
        """Deptry issue grouping model by error code (DEP001-DEP004)."""

        dep001: list[t.Infra.IssueMap] = Field(default_factory=list)
        dep002: list[t.Infra.IssueMap] = Field(default_factory=list)
        dep003: list[t.Infra.IssueMap] = Field(default_factory=list)
        dep004: list[t.Infra.IssueMap] = Field(default_factory=list)

    class DeptryReport(FlextModels.ArbitraryTypesModel):
        """Deptry analysis report with categorized issue modules."""

        missing: list[str | None] = Field(default_factory=list)
        unused: list[str | None] = Field(default_factory=list)
        transitive: list[str | None] = Field(default_factory=list)
        dev_in_runtime: list[str | None] = Field(default_factory=list)
        raw_count: int = Field(default=0, ge=0)

    class ProjectDependencyReport(FlextModels.ArbitraryTypesModel):
        """Project-level dependency report combining deptry results."""

        project: str = Field(min_length=1)
        deptry: FlextInfraDependencyDetectionModels.DeptryReport

    class TypingsReport(FlextModels.ArbitraryTypesModel):
        """Typing stubs analysis report with required/current/delta packages."""

        required_packages: list[str] = Field(default_factory=list)
        hinted: list[str] = Field(default_factory=list)
        missing_modules: list[str] = Field(default_factory=list)
        current: list[str] = Field(default_factory=list)
        to_add: list[str] = Field(default_factory=list)
        to_remove: list[str] = Field(default_factory=list)
        limits_applied: bool = False
        python_version: str | None = None


dm = FlextInfraDependencyDetectionModels


__all__ = ["FlextInfraDependencyDetectionModels", "FlextInfraDepsModels", "dm"]
