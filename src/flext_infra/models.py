"""Domain models for flext-infra.

Defines data models and domain entities for infrastructure services including
configuration, validation results, and workspace state.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from pydantic import Field

from flext_core import FlextModels


class InfraModels(FlextModels):
    class ProjectInfo(FlextModels.ArbitraryTypesModel):
        name: str = Field(min_length=1, description="Project name")
        path: Path = Field(description="Absolute or relative project path")
        stack: str = Field(min_length=1, description="Primary technology stack")
        has_tests: bool = Field(default=False, description="Project has test suite")
        has_src: bool = Field(default=True, description="Project has source directory")

    class GateResult(FlextModels.ArbitraryTypesModel):
        gate: str = Field(min_length=1, description="Gate name")
        project: str = Field(min_length=1, description="Project name")
        passed: bool = Field(description="Gate execution status")
        errors: list[str] = Field(
            default_factory=list, description="Gate error messages"
        )
        duration: float = Field(default=0.0, ge=0.0, description="Duration in seconds")

    class SyncResult(FlextModels.ArbitraryTypesModel):
        files_changed: int = Field(default=0, ge=0, description="Total changed files")
        source: Path = Field(description="Sync source path")
        target: Path = Field(description="Sync target path")
        timestamp: datetime = Field(
            default_factory=lambda: datetime.now(UTC),
            description="Execution timestamp in UTC",
        )

    class CommandOutput(FlextModels.ArbitraryTypesModel):
        stdout: str = Field(default="", description="Captured standard output")
        stderr: str = Field(default="", description="Captured standard error")
        exit_code: int = Field(description="Command exit code")
        duration: float = Field(default=0.0, ge=0.0, description="Duration in seconds")

    class BaseMkConfig(FlextModels.ArbitraryTypesModel):
        project_name: str = Field(min_length=1, description="Project identifier")
        python_version: str = Field(min_length=1, description="Target Python version")
        core_stack: str = Field(min_length=1, description="Core stack classification")
        package_manager: str = Field(default="poetry", description="Dependency manager")
        source_dir: str = Field(default="src", description="Source directory path")
        tests_dir: str = Field(default="tests", description="Tests directory path")
        lint_gates: list[str] = Field(
            default_factory=list, description="Enabled quality gates"
        )
        test_command: str = Field(default="pytest", description="Default test command")

    class MigrationResult(FlextModels.ArbitraryTypesModel):
        project: str = Field(min_length=1, description="Project identifier")
        changes: list[str] = Field(default_factory=list, description="Applied changes")
        errors: list[str] = Field(default_factory=list, description="Migration errors")

    class ValidationReport(FlextModels.ArbitraryTypesModel):
        passed: bool = Field(description="Validation status")
        violations: list[str] = Field(
            default_factory=list,
            description="Collected validation violations",
        )
        summary: str = Field(
            default="", description="Human-readable validation summary"
        )

    class ReleaseSpec(FlextModels.ArbitraryTypesModel):
        version: str = Field(min_length=1, description="Semantic version string")
        tag: str = Field(min_length=1, description="Git tag for release")
        bump_type: str = Field(min_length=1, description="Release bump type")


FlextInfraModels = InfraModels
im = InfraModels

m = InfraModels

__all__ = ["InfraModels", "FlextInfraModels", "im", "m"]
