"""Scenario dataclasses for parametrized testing of infrastructure services.

Provides scenario definitions using flext_tests base classes (m, t).
Eliminates duplication and serves as single source of truth for infra test cases.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import ClassVar

from flext_tests import m


class SubprocessScenario(m.Value):
    """Single scenario for subprocess operation testing using m.Value."""

    name: str
    cmd: list[str]
    expected_output: str = ""
    should_succeed: bool = True
    description: str | None = None


class GitScenario(m.Value):
    """Single scenario for git operation testing using m.Value."""

    name: str
    operations: list[str]
    expected_state: str
    should_succeed: bool = True
    description: str | None = None


class WorkspaceScenario(m.Value):
    """Single scenario for workspace state testing using m.Value."""

    name: str
    structure: dict[str, dict[str, str]]
    should_be_valid: bool = True
    description: str | None = None


class DependencyScenario(m.Value):
    """Single scenario for dependency detection testing using m.Value."""

    name: str
    pyproject_content: str
    expected_deps: list[str]
    should_succeed: bool = True
    description: str | None = None


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
            structure={
                "pyproject.toml": {},
                "src": {},
                "tests": {},
            },
            should_be_valid=True,
        ),
        WorkspaceScenario(
            name="workspace_with_git",
            structure={
                ".git": {},
                "pyproject.toml": {},
                "src": {},
                "tests": {},
            },
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
            pyproject_content=('[tool.poetry.dependencies]\npython = "^3.13"'),
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
    "DependencyScenario",
    "DependencyScenarios",
    "GitScenario",
    "GitScenarios",
    "SubprocessScenario",
    "SubprocessScenarios",
    "WorkspaceScenario",
    "WorkspaceScenarios",
]
