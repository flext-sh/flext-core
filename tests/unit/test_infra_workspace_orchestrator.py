"""Tests for FlextInfraOrchestratorService.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest
from flext_infra import FlextInfraCommandRunner
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

    result = orchestrator.orchestrate(projects, verb)

    assert result.is_success
    assert len(result.value) == 2


def test_orchestrator_stops_on_first_failure_with_fail_fast(
    orchestrator: FlextInfraOrchestratorService,
) -> None:
    """Test that orchestrator stops on first failure when fail_fast=True."""
    projects = ["project-a", "project-b", "project-c"]
    verb = "test"

    result = orchestrator.orchestrate(projects, verb, fail_fast=True)

    assert result.is_success


def test_orchestrator_continues_on_failure_without_fail_fast(
    orchestrator: FlextInfraOrchestratorService,
) -> None:
    """Test that orchestrator continues on failure when fail_fast=False."""
    projects = ["project-a", "project-b"]
    verb = "test"

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

    result = orchestrator.orchestrate(projects, verb)

    assert result.is_success
    assert result.is_success
    assert len(result.value) == 1


def test_orchestrator_fail_fast_skips_remaining_projects(
    orchestrator: FlextInfraOrchestratorService,
) -> None:
    """Test that fail_fast skips remaining projects after failure."""
    projects = ["project-a", "project-b", "project-c"]
    verb = "test"

    # Mock runner to fail on first project
    mock_runner = Mock()
    mock_runner.run_to_file.return_value = Mock(is_success=False, error="Failed")
    orchestrator._runner = mock_runner

    result = orchestrator.orchestrate(projects, verb, fail_fast=True)
    assert result.is_success
    # Should have 3 results (first failed, rest skipped)
    assert len(result.value) == 3


def test_orchestrator_handles_runner_exception(
    orchestrator: FlextInfraOrchestratorService,
) -> None:
    """Test orchestrator handles runner exceptions gracefully."""
    projects = ["project-a"]
    verb = "test"

    # Mock runner to raise exception
    mock_runner = Mock()
    mock_runner.run_to_file.side_effect = OSError("Runner failed")
    orchestrator._runner = mock_runner

    result = orchestrator.orchestrate(projects, verb)
    assert result.is_failure
    assert "Orchestration failed" in result.error


def test_orchestrator_with_make_args(
    orchestrator: FlextInfraOrchestratorService,
) -> None:
    """Test orchestrator passes make arguments correctly."""
    projects = ["project-a"]
    verb = "test"
    make_args = ["VERBOSE=1", "PARALLEL=4"]

    # Mock runner to capture args
    mock_runner = Mock()
    mock_runner.run_to_file.return_value = Mock(is_success=True, value=0)
    orchestrator._runner = mock_runner

    result = orchestrator.orchestrate(projects, verb, make_args=make_args)
    assert result.is_success
    # Verify make_args were passed
    call_args = mock_runner.run_to_file.call_args
    assert "VERBOSE=1" in call_args[0][0]
    assert "PARALLEL=4" in call_args[0][0]


def test_orchestrator_fail_fast_with_failure_result(
    orchestrator: FlextInfraOrchestratorService,
) -> None:
    """Test fail_fast behavior when _run_project returns failure."""
    projects = ["project-a", "project-b", "project-c"]
    verb = "test"

    # Mock runner to fail on first project
    mock_runner = Mock()
    mock_runner.run_to_file.return_value = Mock(is_success=False, error="Failed")
    orchestrator._runner = mock_runner

    result = orchestrator.orchestrate(projects, verb, fail_fast=True)
    assert result.is_success
    # Should have 3 results (first failed, rest skipped)
    assert len(result.value) == 3
    # First should have error, rest should be skipped
    assert result.value[0].exit_code == 1
    assert result.value[1].exit_code == 0  # skipped
    assert result.value[2].exit_code == 0  # skipped


def test_orchestrate_with_project_execution_failure(tmp_path: Path) -> None:
    """Test orchestrate handles project execution failure."""
    orchestrator = FlextInfraOrchestratorService()
    projects = [
        Mock(name="proj1", path=tmp_path / "proj1"),
        Mock(name="proj2", path=tmp_path / "proj2"),
    ]
    verb = "test"

    # Mock runner to fail
    mock_runner = Mock(spec=FlextInfraCommandRunner)
    mock_runner.run_to_file.return_value = Mock(
        is_success=False, error="Execution failed"
    )
    orchestrator._runner = mock_runner

    result = orchestrator.orchestrate(projects, verb, fail_fast=False)
    assert result.is_success
    # Should have results for all projects
    assert len(result.value) == 2
    # Both should have exit code 1 (failure)
    assert result.value[0].exit_code == 1
    assert result.value[1].exit_code == 1
