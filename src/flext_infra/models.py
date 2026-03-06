"""Domain models for flext-infra.

Defines data models and domain entities for infrastructure services including
configuration, validation results, and workspace state.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import libcst as cst
from pydantic import ConfigDict, Field

from flext_core import FlextModels

_PROJECT_NAME_DESC = "Project name"


class FlextInfraModels(FlextModels):
    """Pydantic model namespace for infrastructure services."""

    class Infra:
        """Infrastructure-domain models."""

        class ProjectInfo(FlextModels.ArbitraryTypesModel):
            """Discovered project metadata for workspace operations."""

            name: str = Field(min_length=1, description=_PROJECT_NAME_DESC)
            path: Path = Field(description="Absolute or relative project path")
            stack: str = Field(min_length=1, description="Primary technology stack")
            has_tests: bool = Field(default=False, description="Project has test suite")
            has_src: bool = Field(
                default=True, description="Project has source directory"
            )

        class GateResult(FlextModels.ArbitraryTypesModel):
            """Result summary for a single quality gate execution."""

            gate: str = Field(min_length=1, description="Gate name")
            project: str = Field(min_length=1, description=_PROJECT_NAME_DESC)
            passed: bool = Field(description="Gate execution status")
            errors: list[str] = Field(
                default_factory=list,
                description="Gate error messages",
            )
            duration: float = Field(
                default=0.0, ge=0.0, description="Duration in seconds"
            )

        class SyncResult(FlextModels.ArbitraryTypesModel):
            """Result payload for sync operations."""

            files_changed: int = Field(
                default=0, ge=0, description="Total changed files"
            )
            source: Path = Field(description="Sync source path")
            target: Path = Field(description="Sync target path")
            timestamp: datetime = Field(
                default_factory=lambda: datetime.now(UTC),
                description="Execution timestamp in UTC",
            )

        class CommandOutput(FlextModels.ArbitraryTypesModel):
            """Standardized subprocess output payload."""

            stdout: str = Field(default="", description="Captured standard output")
            stderr: str = Field(default="", description="Captured standard error")
            exit_code: int = Field(description="Command exit code")
            duration: float = Field(
                default=0.0, ge=0.0, description="Duration in seconds"
            )

        class BaseMkConfig(FlextModels.ArbitraryTypesModel):
            """Configuration model used to render base.mk templates."""

            project_name: str = Field(min_length=1, description="Project identifier")
            python_version: str = Field(
                min_length=1, description="Target Python version"
            )
            core_stack: str = Field(
                min_length=1, description="Core stack classification"
            )
            package_manager: str = Field(
                default="poetry", description="Dependency manager"
            )
            source_dir: str = Field(default="src", description="Source directory path")
            tests_dir: str = Field(default="tests", description="Tests directory path")
            lint_gates: list[str] = Field(
                default_factory=list,
                description="Enabled quality gates",
            )
            test_command: str = Field(
                default="pytest", description="Default test command"
            )

        class MigrationResult(FlextModels.ArbitraryTypesModel):
            """Migration operation outcome with applied changes and errors."""

            project: str = Field(min_length=1, description="Project identifier")
            changes: list[str] = Field(
                default_factory=list, description="Applied changes"
            )
            errors: list[str] = Field(
                default_factory=list, description="Migration errors"
            )

        class ValidationReport(FlextModels.ArbitraryTypesModel):
            """Validation report model with violations and summary."""

            passed: bool = Field(description="Validation status")
            violations: list[str] = Field(
                default_factory=list,
                description="Collected validation violations",
            )
            summary: str = Field(
                default="",
                description="Human-readable validation summary",
            )

        class ReleaseSpec(FlextModels.ArbitraryTypesModel):
            """Release descriptor with version, tag, and bump metadata."""

            version: str = Field(min_length=1, description="Semantic version string")
            tag: str = Field(min_length=1, description="Git tag for release")
            bump_type: str = Field(min_length=1, description="Release bump type")

        class PrExecutionResult(FlextModels.ArbitraryTypesModel):
            """Result of a single PR operation on a repository."""

            display: str = Field(min_length=1, description="Repository display name")
            status: str = Field(min_length=1, description="Execution status")
            elapsed: int = Field(ge=0, description="Elapsed time in seconds")
            exit_code: int = Field(description="Process exit code")
            log_path: str | None = Field(default=None, description="Log file path")

        class PrOrchestrationResult(FlextModels.ArbitraryTypesModel):
            """Aggregated result of workspace-wide PR orchestration."""

            total: int = Field(ge=0, description="Total repositories processed")
            success: int = Field(ge=0, description="Successful executions")
            fail: int = Field(ge=0, description="Failed executions")
            results: list[FlextInfraModels.Infra.PrExecutionResult] = Field(
                default_factory=list,
                description="Per-repository results",
            )

        class RepoUrls(FlextModels.ArbitraryTypesModel):
            """Repository URL pair with SSH and HTTPS variants."""

            ssh_url: str = Field(default="", description="SSH clone URL")
            https_url: str = Field(default="", description="HTTPS clone URL")

        # -- Census domain models --------------------------------------------------

        class CensusViolation(FlextModels.ArbitraryTypesModel):
            """A single namespace violation detected by the census service."""

            module: str = Field(min_length=1, description="Module file path")
            rule: str = Field(
                min_length=1, description="Violated rule identifier (e.g. NS-001)"
            )
            line: int = Field(ge=0, description="Line number of violation")
            message: str = Field(
                min_length=1, description="Human-readable violation message"
            )
            fixable: bool = Field(
                description="Whether this violation can be auto-fixed"
            )

        class CensusReport(FlextModels.ArbitraryTypesModel):
            """Aggregated census report for a single project."""

            project: str = Field(min_length=1, description="Project name")
            violations: list[FlextInfraModels.Infra.CensusViolation] = Field(
                default_factory=list, description="Detected violations"
            )
            total: int = Field(ge=0, description="Total violation count")
            fixable: int = Field(ge=0, description="Count of auto-fixable violations")

        class ScaffoldResult(FlextModels.ArbitraryTypesModel):
            """Result of scaffolding base modules for a project."""

            project: str = Field(min_length=1, description="Project name")
            files_created: list[str] = Field(
                default_factory=list, description="Newly created file paths"
            )
            files_skipped: list[str] = Field(
                default_factory=list,
                description="Skipped (already existing) file paths",
            )

        class AutoFixResult(FlextModels.ArbitraryTypesModel):
            """Result of auto-fixing namespace violations for a project."""

            project: str = Field(min_length=1, description="Project name")
            violations_fixed: list[FlextInfraModels.Infra.CensusViolation] = Field(
                default_factory=list, description="Fixed violations"
            )
            violations_skipped: list[FlextInfraModels.Infra.CensusViolation] = Field(
                default_factory=list,
                description="Skipped violations (not auto-fixable)",
            )
            files_modified: list[str] = Field(
                default_factory=list, description="Modified file paths"
            )

        class CodegenPipelineResult(FlextModels.ArbitraryTypesModel):
            """Full pipeline result combining census, scaffold, auto-fix phases."""

            census_before: FlextInfraModels.Infra.CensusReport = Field(
                description="Census report before transformations"
            )
            scaffold: FlextInfraModels.Infra.ScaffoldResult = Field(
                description="Scaffold phase result"
            )
            auto_fix: FlextInfraModels.Infra.AutoFixResult = Field(
                description="Auto-fix phase result"
            )
            census_after: FlextInfraModels.Infra.CensusReport = Field(
                description="Census report after transformations"
            )

        # -- Dependency domain models ----------------------------------------------

        class DependencyReport(FlextModels.ArbitraryTypesModel):
            """Report of dependency detection for a single project."""

            project: str = Field(min_length=1, description=_PROJECT_NAME_DESC)
            missing: list[str] = Field(
                default_factory=list, description="Missing dependencies"
            )
            unused: list[str] = Field(
                default_factory=list, description="Unused dependencies"
            )
            outdated: list[str] = Field(
                default_factory=list, description="Outdated dependencies"
            )

        # -- Refactor domain models -----------------------------------------------

        class Refactor:
            """Models for the refactor engine and related tools."""

            class Result(FlextModels.ArbitraryTypesModel):
                """Result of applying refactor rules to a single file."""

                file_path: Path = Field(description="Target file path")
                success: bool = Field(description="Whether the operation succeeded")
                modified: bool = Field(
                    description="Whether the file was actually modified"
                )
                error: str | None = Field(
                    default=None, description="Error message on failure"
                )
                changes: list[str] = Field(
                    default_factory=list,
                    description="Human-readable change descriptions",
                )
                refactored_code: str | None = Field(
                    default=None,
                    description="Resulting source code after transformation",
                )

            class MethodInfo(FlextModels.ArbitraryTypesModel):
                """Metadata about a method used for ordering inside classes."""

                name: str = Field(min_length=1, description="Method name")
                category: str = Field(description="Method category classification")
                node: cst.FunctionDef = Field(
                    description="LibCST FunctionDef node",
                    exclude=True,
                )
                decorators: list[str] = Field(
                    default_factory=list,
                    description="Decorator names applied to this method",
                )

            class Checkpoint(FlextModels.ArbitraryTypesModel):
                """Serialisable checkpoint state for refactor safety recovery."""

                workspace_root: str = Field(
                    min_length=1, description="Workspace root path"
                )
                status: str = Field(default="running", description="Checkpoint status")
                stash_ref: str = Field(default="", description="Git stash reference")
                processed_targets: list[str] = Field(
                    default_factory=list,
                    description="Already-processed file targets",
                )
                updated_at: str = Field(
                    default_factory=lambda: datetime.now(UTC).isoformat(),
                    description="ISO 8601 timestamp of last update",
                )

            class ClassOccurrence(FlextModels.ArbitraryTypesModel):
                """A single class definition occurrence within a source file."""

                model_config = ConfigDict(frozen=True)

                name: str = Field(min_length=1, description="Class name")
                line: int = Field(ge=0, description="Line number (0 = unknown)")
                is_top_level: bool = Field(
                    description="Whether class is at module top level"
                )

            class LooseClassViolation(FlextModels.ArbitraryTypesModel):
                """A detected loose-class naming violation with confidence."""

                model_config = ConfigDict(frozen=True)

                file: str = Field(min_length=1, description="Source file path")
                line: int = Field(ge=1, description="Line number")
                class_name: str = Field(
                    min_length=1, description="Violating class name"
                )
                expected_prefix: str = Field(description="Expected namespace prefix")
                rule: str = Field(min_length=1, description="Violated rule id")
                reason: str = Field(description="Human-readable reason")
                confidence: str = Field(description="Confidence level")
                score: float = Field(ge=0.0, le=1.0, description="Confidence score")

            class FamilyMROResolution(FlextModels.ArbitraryTypesModel):
                """Resolution payload for one facade family MRO."""

                model_config = ConfigDict(frozen=True)

                family: str = Field(min_length=1, description="Facade family letter")
                expected_bases: tuple[str, ...] = Field(
                    description="Expected base class names in order"
                )
                resolved_mro: tuple[str, ...] = Field(
                    description="Resolved MRO class names"
                )
                accessible_namespaces: tuple[str, ...] = Field(
                    description="Namespaces accessible through the MRO"
                )

            class ProjectClassification(FlextModels.ArbitraryTypesModel):
                """Result of classifying a project by kind and family chains."""

                model_config = ConfigDict(frozen=True)

                project_kind: str = Field(
                    min_length=1,
                    description="Project kind (core, domain, platform, integration, app)",
                )
                family_chains: dict[str, list[str]] = Field(
                    description="Family letter to MRO chain mapping"
                )

            class Suggestion(FlextModels.ArbitraryTypesModel):
                """A single refactoring suggestion from analysis."""

                model_config = ConfigDict(frozen=True)

                file: str = Field(min_length=1, description="Source file path")
                name: str = Field(min_length=1, description="Symbol name")
                kind: str = Field(description="Symbol kind (function, class, etc.)")
                target: str = Field(description="Suggested target location")
                reason: str = Field(default="", description="Reason for suggestion")

            class ManualReviewItem(FlextModels.ArbitraryTypesModel):
                """An item flagged for manual review during analysis."""

                model_config = ConfigDict(frozen=True)

                file: str = Field(min_length=1, description="Source file path")
                name: str = Field(min_length=1, description="Symbol name")
                kind: str = Field(description="Symbol kind (function, class, etc.)")
                reason: str = Field(description="Reason for manual review")


m = FlextInfraModels

__all__ = ["FlextInfraModels", "m"]
