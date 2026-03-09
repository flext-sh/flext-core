"""Domain models for the codegen subpackage."""

from __future__ import annotations

from pydantic import Field

from flext_core import FlextModels


def _new_violations() -> list[FlextInfraCodegenModels.CensusViolation]:
    return []


class FlextInfraCodegenModels:
    """Models for codegen census, scaffold, and auto-fix pipelines."""

    class CensusViolation(FlextModels.ArbitraryTypesModel):
        """A single namespace violation detected by the census service."""

        module: str = Field(min_length=1, description="Module file path")
        rule: str = Field(
            min_length=1,
            description="Violated rule identifier (e.g. NS-001)",
        )
        line: int = Field(ge=0, description="Line number of violation")
        message: str = Field(
            min_length=1,
            description="Human-readable violation message",
        )
        fixable: bool = Field(description="Whether this violation can be auto-fixed")

    class CensusReport(FlextModels.ArbitraryTypesModel):
        """Aggregated census report for a single project."""

        project: str = Field(min_length=1, description="Project name")
        violations: list[FlextInfraCodegenModels.CensusViolation] = Field(
            default_factory=_new_violations,
            description="Detected violations",
        )
        total: int = Field(ge=0, description="Total violation count")
        fixable: int = Field(ge=0, description="Count of auto-fixable violations")

    class ScaffoldResult(FlextModels.ArbitraryTypesModel):
        """Result of scaffolding base modules for a project."""

        project: str = Field(min_length=1, description="Project name")
        files_created: list[str] = Field(
            default_factory=list,
            description="Newly created file paths",
        )
        files_skipped: list[str] = Field(
            default_factory=list,
            description="Skipped (already existing) file paths",
        )

    class AutoFixResult(FlextModels.ArbitraryTypesModel):
        """Result of auto-fixing namespace violations for a project."""

        project: str = Field(min_length=1, description="Project name")
        violations_fixed: list[FlextInfraCodegenModels.CensusViolation] = Field(
            default_factory=_new_violations,
            description="Fixed violations",
        )
        violations_skipped: list[FlextInfraCodegenModels.CensusViolation] = Field(
            default_factory=_new_violations,
            description="Skipped violations (not auto-fixable)",
        )
        files_modified: list[str] = Field(
            default_factory=list,
            description="Modified file paths",
        )

    class CodegenPipelineResult(FlextModels.ArbitraryTypesModel):
        """Full pipeline result combining census, scaffold, auto-fix phases."""

        census_before: FlextInfraCodegenModels.CensusReport = Field(
            description="Census report before transformations",
        )
        scaffold: FlextInfraCodegenModels.ScaffoldResult = Field(
            description="Scaffold phase result",
        )
        auto_fix: FlextInfraCodegenModels.AutoFixResult = Field(
            description="Auto-fix phase result",
        )
        census_after: FlextInfraCodegenModels.CensusReport = Field(
            description="Census report after transformations",
        )

    class QualityGateCheck(FlextModels.ArbitraryTypesModel):
        """A single quality gate check result entry."""

        name: str = Field(min_length=1, description="Check identifier")
        passed: bool = Field(description="Whether check passed")
        detail: str = Field(default="", description="Human-readable check detail")
        critical: bool = Field(description="Whether failure is critical")

    class QualityGateProjectFinding(FlextModels.ArbitraryTypesModel):
        """Per-project quality gate findings."""

        project: str = Field(min_length=1, description="Project name")
        violations_total: int = Field(ge=0, description="Total violations")
        fixable_violations: int = Field(ge=0, description="Auto-fixable violations")
        validator_passed: bool = Field(description="Whether validator passed")
        mro_failures: int = Field(ge=0, description="MRO failure count")
        layer_violations: int = Field(ge=0, description="Layer violation count")
        cross_project_reference_violations: int = Field(
            ge=0,
            description="Cross-project reference violation count",
        )


__all__ = ["FlextInfraCodegenModels"]
