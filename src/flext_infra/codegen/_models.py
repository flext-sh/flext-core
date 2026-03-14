"""Domain models for the codegen subpackage."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from flext_core import FlextModels


class FlextInfraCodegenModels:
    """Models for codegen census, scaffold, and auto-fix pipelines."""

    class CensusViolation(FlextModels.ArbitraryTypesModel):
        """A single namespace violation detected by the census service."""

        module: Annotated[str, Field(min_length=1, description="Module file path")]
        rule: Annotated[
            str,
            Field(
                min_length=1,
                description="Violated rule identifier (e.g. NS-001)",
            ),
        ]
        line: Annotated[int, Field(ge=0, description="Line number of violation")]
        message: Annotated[
            str,
            Field(
                min_length=1,
                description="Human-readable violation message",
            ),
        ]
        fixable: Annotated[
            bool, Field(description="Whether this violation can be auto-fixed")
        ]

    class CensusReport(FlextModels.ArbitraryTypesModel):
        """Aggregated census report for a single project."""

        project: Annotated[str, Field(min_length=1, description="Project name")]
        violations: Annotated[
            list[FlextInfraCodegenModels.CensusViolation],
            Field(
                default_factory=list,
                description="Detected violations",
            ),
        ]
        total: Annotated[int, Field(ge=0, description="Total violation count")]
        fixable: Annotated[
            int, Field(ge=0, description="Count of auto-fixable violations")
        ]

    class ScaffoldResult(FlextModels.ArbitraryTypesModel):
        """Result of scaffolding base modules for a project."""

        project: Annotated[str, Field(min_length=1, description="Project name")]
        files_created: Annotated[
            list[str],
            Field(
                default_factory=list,
                description="Newly created file paths",
            ),
        ]
        files_skipped: Annotated[
            list[str],
            Field(
                default_factory=list,
                description="Skipped (already existing) file paths",
            ),
        ]

    class AutoFixResult(FlextModels.ArbitraryTypesModel):
        """Result of auto-fixing namespace violations for a project."""

        project: Annotated[str, Field(min_length=1, description="Project name")]
        violations_fixed: Annotated[
            list[FlextInfraCodegenModels.CensusViolation],
            Field(
                default_factory=list,
                description="Fixed violations",
            ),
        ]
        violations_skipped: Annotated[
            list[FlextInfraCodegenModels.CensusViolation],
            Field(
                default_factory=list,
                description="Skipped violations (not auto-fixable)",
            ),
        ]
        files_modified: Annotated[
            list[str],
            Field(
                default_factory=list,
                description="Modified file paths",
            ),
        ]

    class CodegenPipelineResult(FlextModels.ArbitraryTypesModel):
        """Full pipeline result combining census, scaffold, auto-fix phases."""

        census_before: Annotated[
            FlextInfraCodegenModels.CensusReport,
            Field(
                description="Census report before transformations",
            ),
        ]
        scaffold: Annotated[
            FlextInfraCodegenModels.ScaffoldResult,
            Field(
                description="Scaffold phase result",
            ),
        ]
        auto_fix: Annotated[
            FlextInfraCodegenModels.AutoFixResult,
            Field(
                description="Auto-fix phase result",
            ),
        ]
        census_after: Annotated[
            FlextInfraCodegenModels.CensusReport,
            Field(
                description="Census report after transformations",
            ),
        ]

    class QualityGateCheck(FlextModels.ArbitraryTypesModel):
        """A single quality gate check result entry."""

        name: Annotated[str, Field(min_length=1, description="Check identifier")]
        passed: Annotated[bool, Field(description="Whether check passed")]
        detail: Annotated[
            str, Field(default="", description="Human-readable check detail")
        ]
        critical: Annotated[bool, Field(description="Whether failure is critical")]

    class QualityGateProjectFinding(FlextModels.ArbitraryTypesModel):
        """Per-project quality gate findings."""

        project: Annotated[str, Field(min_length=1, description="Project name")]
        violations_total: Annotated[int, Field(ge=0, description="Total violations")]
        fixable_violations: Annotated[
            int, Field(ge=0, description="Auto-fixable violations")
        ]
        validator_passed: Annotated[bool, Field(description="Whether validator passed")]
        mro_failures: Annotated[int, Field(ge=0, description="MRO failure count")]
        layer_violations: Annotated[
            int, Field(ge=0, description="Layer violation count")
        ]
        cross_project_reference_violations: Annotated[
            int,
            Field(
                ge=0,
                description="Cross-project reference violation count",
            ),
        ]


__all__ = ["FlextInfraCodegenModels"]
