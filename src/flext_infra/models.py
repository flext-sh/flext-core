"""Domain models for flext-infra.

Defines data models and domain entities for infrastructure services including
configuration, validation results, and workspace state.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from flext_core import FlextModels as _FlextModels
from pydantic import Field


class FlextInfraModels(_FlextModels):
    """Pydantic model namespace for infrastructure services."""

    class ProjectInfo(_FlextModels.ArbitraryTypesModel):
        """Discovered project metadata for workspace operations."""

        name: str = Field(min_length=1, description="Project name")
        path: Path = Field(description="Absolute or relative project path")
        stack: str = Field(min_length=1, description="Primary technology stack")
        has_tests: bool = Field(default=False, description="Project has test suite")
        has_src: bool = Field(default=True, description="Project has source directory")

    class GateResult(_FlextModels.ArbitraryTypesModel):
        """Result summary for a single quality gate execution."""

        gate: str = Field(min_length=1, description="Gate name")
        project: str = Field(min_length=1, description="Project name")
        passed: bool = Field(description="Gate execution status")
        errors: list[str] = Field(
            default_factory=list,
            description="Gate error messages",
        )
        duration: float = Field(default=0.0, ge=0.0, description="Duration in seconds")

    class SyncResult(_FlextModels.ArbitraryTypesModel):
        """Result payload for sync operations."""

        files_changed: int = Field(default=0, ge=0, description="Total changed files")
        source: Path = Field(description="Sync source path")
        target: Path = Field(description="Sync target path")
        timestamp: datetime = Field(
            default_factory=lambda: datetime.now(UTC),
            description="Execution timestamp in UTC",
        )

    class CommandOutput(_FlextModels.ArbitraryTypesModel):
        """Standardized subprocess output payload."""

        stdout: str = Field(default="", description="Captured standard output")
        stderr: str = Field(default="", description="Captured standard error")
        exit_code: int = Field(description="Command exit code")
        duration: float = Field(default=0.0, ge=0.0, description="Duration in seconds")

    class BaseMkConfig(_FlextModels.ArbitraryTypesModel):
        """Configuration model used to render base.mk templates."""

        project_name: str = Field(min_length=1, description="Project identifier")
        python_version: str = Field(min_length=1, description="Target Python version")
        core_stack: str = Field(min_length=1, description="Core stack classification")
        package_manager: str = Field(default="poetry", description="Dependency manager")
        source_dir: str = Field(default="src", description="Source directory path")
        tests_dir: str = Field(default="tests", description="Tests directory path")
        lint_gates: list[str] = Field(
            default_factory=list,
            description="Enabled quality gates",
        )
        test_command: str = Field(default="pytest", description="Default test command")

    class MigrationResult(_FlextModels.ArbitraryTypesModel):
        """Migration operation outcome with applied changes and errors."""

        project: str = Field(min_length=1, description="Project identifier")
        changes: list[str] = Field(default_factory=list, description="Applied changes")
        errors: list[str] = Field(default_factory=list, description="Migration errors")

    class ValidationReport(_FlextModels.ArbitraryTypesModel):
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

    class ReleaseSpec(_FlextModels.ArbitraryTypesModel):
        """Release descriptor with version, tag, and bump metadata."""

        version: str = Field(min_length=1, description="Semantic version string")
        tag: str = Field(min_length=1, description="Git tag for release")
        bump_type: str = Field(min_length=1, description="Release bump type")

    class PrExecutionResult(_FlextModels.ArbitraryTypesModel):
        """Result of a single PR operation on a repository."""

        display: str = Field(min_length=1, description="Repository display name")
        status: str = Field(min_length=1, description="Execution status")
        elapsed: int = Field(ge=0, description="Elapsed time in seconds")
        exit_code: int = Field(description="Process exit code")
        log_path: str | None = Field(default=None, description="Log file path")

    class PrOrchestrationResult(_FlextModels.ArbitraryTypesModel):
        """Aggregated result of workspace-wide PR orchestration."""

        total: int = Field(ge=0, description="Total repositories processed")
        success: int = Field(ge=0, description="Successful executions")
        fail: int = Field(ge=0, description="Failed executions")
        results: list[FlextInfraModels.PrExecutionResult] = Field(
            default_factory=list,
            description="Per-repository results",
        )

    class RepoUrls(_FlextModels.ArbitraryTypesModel):
        """Repository URL pair with SSH and HTTPS variants."""

        ssh_url: str = Field(default="", description="SSH clone URL")
        https_url: str = Field(default="", description="HTTPS clone URL")


m = FlextInfraModels

__all__ = ["FlextInfraModels", "m"]
