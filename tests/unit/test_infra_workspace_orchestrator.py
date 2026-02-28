"""Tests for FlextInfraOrchestratorService.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from flext_core import FlextResult as r
from flext_infra import m as im
from flext_infra.workspace.orchestrator import FlextInfraOrchestratorService


@pytest.fixture
def orchestrator() -> FlextInfraOrchestratorService:
    """Create an orchestrator instance."""
    return FlextInfraOrchestratorService()


def test_orchestrator_executes_verb_across_projects(
    orchestrator: FlextInfraOrchestratorService,
) -> None:
    """Test orchestration of make verb across multiple projects."""
    projects = ["project-a", "project-b"]
    verb = "check"

    mock_output = MagicMock(spec=im.CommandOutput)
    mock_output.exit_code = 0
    mock_output.stdout = "success"

    with patch.object(
        orchestrator._runner, "run", return_value=r[im.CommandOutput].ok(mock_output)
    ):
        result = orchestrator.orchestrate(projects, verb)

        assert result.is_success
        assert len(result.value) == 2


def test_orchestrator_stops_on_first_failure_with_fail_fast(
    orchestrator: FlextInfraOrchestratorService,
) -> None:
    """Test that orchestrator stops on first failure when fail_fast=True."""
    projects = ["project-a", "project-b", "project-c"]
    verb = "test"

    mock_output = MagicMock(spec=im.CommandOutput)
    mock_output.exit_code = 1
    mock_output.stdout = "failed"

    with patch.object(
        orchestrator._runner, "run", return_value=r[im.CommandOutput].ok(mock_output)
    ):
        result = orchestrator.orchestrate(projects, verb, fail_fast=True)

        assert result.is_success
        assert len(result.value) == 1


def test_orchestrator_continues_on_failure_without_fail_fast(
    orchestrator: FlextInfraOrchestratorService,
) -> None:
    """Test that orchestrator continues on failure when fail_fast=False."""
    projects = ["project-a", "project-b"]
    verb = "test"

    mock_output = MagicMock(spec=im.CommandOutput)
    mock_output.exit_code = 1
    mock_output.stdout = "failed"

    with patch.object(
        orchestrator._runner, "run", return_value=r[im.CommandOutput].ok(mock_output)
    ):
        result = orchestrator.orchestrate(projects, verb, fail_fast=False)

        assert result.is_success
        assert len(result.value) == 2


def test_orchestrator_execute_returns_failure() -> None:
    """Test that execute() method returns failure as expected."""
    orchestrator = FlextInfraOrchestratorService()
    result = orchestrator.execute()
    assert result.is_failure


def test_orchestrator_handles_empty_project_list(
    orchestrator: FlextInfraOrchestratorService,
) -> None:
    """Test orchestration with empty project list."""
    result = orchestrator.orchestrate([], "check")
    assert result.is_success
    assert len(result.value) == 0


def test_orchestrator_captures_per_project_output(
    orchestrator: FlextInfraOrchestratorService,
) -> None:
    """Test that orchestrator captures output for each project."""
    projects = ["project-a"]
    verb = "check"

    mock_output = MagicMock(spec=im.CommandOutput)
    mock_output.exit_code = 0
    mock_output.stdout = "success"

    with patch.object(
        orchestrator._runner, "run", return_value=r[im.CommandOutput].ok(mock_output)
    ):
        result = orchestrator.orchestrate(projects, verb)

        assert result.is_success
        assert len(result.value) == 1
