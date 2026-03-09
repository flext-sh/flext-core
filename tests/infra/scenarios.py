"""Scenario dataclasses for parametrized testing of infrastructure services.

Provides scenario definitions for subprocess operations, git operations,
workspace states, and dependency detection. Eliminates duplication and serves
as single source of truth for infra test cases.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

from flext_core import t


class SubprocessScenario(BaseModel):
    """Single scenario for subprocess operation testing."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="Unique scenario name")
    cmd: list[str] = Field(description="Command to execute as list of strings")
    expected_output: str = Field(
        default="", description="Expected stdout output from command"
    )
    should_succeed: bool = Field(
        default=True, description="Whether scenario expects zero exit code"
    )
    description: str | None = Field(
        default=None, description="Human-readable scenario description"
    )


class GitScenario(BaseModel):
    """Single scenario for git operation testing."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="Unique scenario name")
    operations: list[str] = Field(
        description="List of git operations to perform (e.g., init, add, commit)"
    )
    expected_state: str = Field(
        description="Expected repository state after operations"
    )
    should_succeed: bool = Field(
        default=True, description="Whether scenario expects all operations to succeed"
    )
    description: str | None = Field(
        default=None, description="Human-readable scenario description"
    )


class WorkspaceScenario(BaseModel):
    """Single scenario for workspace state testing."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="Unique scenario name")
    structure: t.ConfigurationMapping = Field(
        description="Directory structure and file layout for workspace"
    )
    should_be_valid: bool = Field(
        default=True, description="Whether scenario represents valid workspace"
    )
    description: str | None = Field(
        default=None, description="Human-readable scenario description"
    )


class DependencyScenario(BaseModel):
    """Single scenario for dependency detection testing."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="Unique scenario name")
    pyproject_content: str = Field(
        description="Content of pyproject.toml for dependency parsing"
    )
    expected_deps: list[str] = Field(
        description="Expected list of detected dependencies"
    )
    should_succeed: bool = Field(
        default=True, description="Whether scenario expects successful parsing"
    )
    description: str | None = Field(
        default=None, description="Human-readable scenario description"
    )


class SubprocessScenarios:
    """Centralized subprocess scenarios - single source of truth."""

    BASIC_SCENARIOS: ClassVar[list[SubprocessScenario]] = [
        SubprocessScenario(
            name="echo_simple",
            cmd=["echo", "hello"],
            expected_output="hello",
            should_succeed=True,
        ),
        SubprocessScenario(
            name="false_command",
            cmd=["false"],
            expected_output="",
            should_succeed=False,
        ),
    ]


class GitScenarios:
    """Centralized git operation scenarios - single source of truth."""

    BASIC_SCENARIOS: ClassVar[list[GitScenario]] = [
        GitScenario(
            name="git_init",
            operations=["init"],
            expected_state="initialized",
            should_succeed=True,
        ),
        GitScenario(
            name="git_init_add_commit",
            operations=["init", "add", "commit"],
            expected_state="committed",
            should_succeed=True,
        ),
    ]


class WorkspaceScenarios:
    """Centralized workspace state scenarios - single source of truth."""

    VALID_SCENARIOS: ClassVar[list[WorkspaceScenario]] = [
        WorkspaceScenario(
            name="workspace_minimal",
            structure={"pyproject.toml": "", "src": {}, "tests": {}},
            should_be_valid=True,
        ),
        WorkspaceScenario(
            name="workspace_with_git",
            structure={".git": {}, "pyproject.toml": "", "src": {}, "tests": {}},
            should_be_valid=True,
        ),
    ]

    INVALID_SCENARIOS: ClassVar[list[WorkspaceScenario]] = [
        WorkspaceScenario(
            name="workspace_no_pyproject",
            structure={"src": {}, "tests": {}},
            should_be_valid=False,
        ),
    ]


class DependencyScenarios:
    """Centralized dependency detection scenarios - single source of truth."""

    BASIC_SCENARIOS: ClassVar[list[DependencyScenario]] = [
        DependencyScenario(
            name="deps_single",
            pyproject_content=(
                '[tool.poetry.dependencies]\npython = "^3.13"\nrequests = "^2.31.0"'
            ),
            expected_deps=["requests"],
            should_succeed=True,
        ),
        DependencyScenario(
            name="deps_empty",
            pyproject_content='[tool.poetry.dependencies]\npython = "^3.13"',
            expected_deps=[],
            should_succeed=True,
        ),
        DependencyScenario(
            name="deps_invalid_toml",
            pyproject_content="[invalid toml content",
            expected_deps=[],
            should_succeed=False,
        ),
    ]


__all__ = [
    "SubprocessScenario",
    "GitScenario",
    "WorkspaceScenario",
    "DependencyScenario",
    "SubprocessScenarios",
    "GitScenarios",
    "WorkspaceScenarios",
    "DependencyScenarios",
]
