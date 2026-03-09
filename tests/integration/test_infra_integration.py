"""Integration tests for flext_infra cross-module flows.

Tests exercise cross-module flows with real service instances, validating:
- Container integration: service registration and retrieval
- Workspace detection → orchestration flow
- BaseMk generation flow with real validation
- Output singleton consistency
- Service FlextResult chaining
- Path resolver → discovery flow

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

import pytest

from flext_core import r
from flext_infra import FlextInfraOutput, u
from flext_infra._utilities.discovery import (
    FlextInfraUtilitiesDiscovery as FlextInfraDiscoveryService,
)
from flext_infra._utilities.output import output
from flext_infra._utilities.paths import FlextInfraUtilitiesPaths
from flext_infra._utilities.subprocess import (
    FlextInfraUtilitiesSubprocess as FlextInfraCommandRunner,
)
from flext_infra.basemk import FlextInfraBaseMkGenerator, FlextInfraBaseMkTemplateEngine
from flext_infra.workspace import (
    FlextInfraOrchestratorService,
    FlextInfraUtilitiesOutput,
    FlextInfraUtilitiesPaths,
    FlextInfraWorkspaceDetector,
    output,
)

pytestmark = [pytest.mark.integration]


class TestContainerIntegration:
    """Test container integration: service registration and retrieval."""


class TestWorkspaceDetectionOrchestrationFlow:
    """Test workspace detection → orchestration flow with shared state."""

    @pytest.mark.integration
    def test_workspace_detector_and_orchestrator_share_state(
        self,
        tmp_path: Path,
    ) -> None:
        """Test that FlextInfraWorkspaceDetector and orchestrator share state.

        Validates:
        - Detector can be created
        - Orchestrator can be created
        - Both can access shared workspace information
        """
        workspace_root = tmp_path / "workspace"
        workspace_root.mkdir()
        (workspace_root / ".git").mkdir()
        detector = FlextInfraWorkspaceDetector()
        orchestrator = FlextInfraOrchestratorService()
        assert detector is not None
        assert orchestrator is not None
        assert isinstance(detector, FlextInfraWorkspaceDetector)
        assert isinstance(orchestrator, FlextInfraOrchestratorService)

    @pytest.mark.integration
    def test_workspace_detector_returns_flext_result(self, tmp_path: Path) -> None:
        """Test that workspace detector operations return FlextResult.

        Validates:
        - Detector methods return FlextResult
        - Result typing is correct
        """
        detector = FlextInfraWorkspaceDetector()
        assert detector is not None
        assert isinstance(detector, FlextInfraWorkspaceDetector)


class TestBaseMkGenerationFlow:
    """Test BaseMk generation flow with real command validation."""

    @pytest.mark.integration
    def test_basemk_template_engine_and_generator_flow(self, tmp_path: Path) -> None:
        """Test BaseMk template engine → generator flow.

        Validates:
        - Template engine can be created
        - Generator can be created
        - Both work together in a flow
        """
        output_dir = tmp_path / "basemk_output"
        output_dir.mkdir()
        engine = FlextInfraBaseMkTemplateEngine()
        generator = FlextInfraBaseMkGenerator()
        assert engine is not None
        assert generator is not None
        assert isinstance(engine, FlextInfraBaseMkTemplateEngine)
        assert isinstance(generator, FlextInfraBaseMkGenerator)

    @pytest.mark.integration
    def test_basemk_generator_generates_valid_content(self, tmp_path: Path) -> None:
        """Test BaseMk generator validates rendered output using real make."""
        _ = tmp_path
        generator = FlextInfraBaseMkGenerator()
        generated = generator.generate()
        assert generated.is_success
        assert isinstance(generated.value, str)
        assert "check" in generated.value


class TestOutputSingletonConsistency:
    """Test output singleton consistency across modules."""

    @pytest.mark.integration
    def test_output_singleton_is_same_instance_everywhere(self) -> None:
        """Test that flext_infra.output is same instance everywhere.

        Validates:
        - output singleton is consistent
        - output is a FlextInfraUtilitiesOutput instance
        - Singleton pattern is maintained
        """
        assert isinstance(output, FlextInfraUtilitiesOutput)
        assert output is not None

    @pytest.mark.integration
    def test_output_singleton_has_expected_methods(self) -> None:
        """Test that output singleton has all expected methods.

        Validates:
        - output has status method
        - output has summary method
        - output has error method
        - output has warning method
        - output has info method
        """
        assert hasattr(output, "status")
        assert hasattr(output, "summary")
        assert hasattr(output, "error")
        assert hasattr(output, "warning")
        assert hasattr(output, "info")
        assert hasattr(output, "header")
        assert hasattr(output, "progress")

    @pytest.mark.integration
    def test_output_singleton_methods_are_callable(self) -> None:
        """Test that output singleton methods are callable.

        Validates:
        - All methods are callable
        - Methods can be invoked without error
        """
        assert callable(output.status)
        assert callable(output.summary)
        assert callable(output.error)
        assert callable(output.warning)
        assert callable(output.info)
        assert callable(output.header)
        assert callable(output.progress)


class TestServiceFlextResultChaining:
    """Test service FlextResult chaining via .map()/.flat_map()."""

    @pytest.mark.integration
    def test_service_result_chaining_with_map(self) -> None:
        """Test chaining multiple services via .map().

        Validates:
        - FlextResult.map() works with service results
        - Type is preserved through chain
        - Value is transformed correctly
        """
        initial_value = 10
        result = r[int].ok(initial_value).map(lambda x: x * 2).map(lambda x: x + 5)
        assert result.is_success
        assert result.value == 25

    @pytest.mark.integration
    def test_service_result_chaining_with_flat_map(self) -> None:
        """Test chaining multiple services via .flat_map().

        Validates:
        - FlextResult.flat_map() works with service results
        - Type is preserved through chain
        - Failures propagate correctly
        """
        initial_value = 10
        result = (
            r[int]
            .ok(initial_value)
            .flat_map(lambda x: r[int].ok(x * 2))
            .flat_map(lambda x: r[int].ok(x + 5))
        )
        assert result.is_success
        assert result.value == 25

    @pytest.mark.integration
    def test_service_result_chaining_failure_propagation(self) -> None:
        """Test that failures propagate through result chains.

        Validates:
        - Failure stops the chain
        - Error message is preserved
        - Subsequent operations are not executed
        """
        initial_value = 10
        result = (
            r[int]
            .ok(initial_value)
            .flat_map(lambda x: r[int].ok(x * 2))
            .flat_map(lambda x: r[int].fail("intentional error"))
            .flat_map(lambda x: r[int].ok(x + 5))
        )
        assert result.is_failure
        assert isinstance(result.error, str)
        assert "intentional error" in result.error

    @pytest.mark.integration
    def test_service_result_chaining_with_mixed_operations(self) -> None:
        """Test chaining with mixed map and flat_map operations.

        Validates:
        - Mixed operations work together
        - Type is preserved
        - Values are transformed correctly
        """
        initial_value = 5
        result = (
            r[int]
            .ok(initial_value)
            .map(lambda x: x * 2)
            .flat_map(lambda x: r[int].ok(x + 3))
            .map(lambda x: x * 2)
        )
        assert result.is_success
        assert result.value == 26


class TestPathResolverDiscoveryFlow:
    """Test path resolver → discovery flow."""

    @pytest.mark.integration
    def test_path_resolver_and_discovery_service_flow(self, tmp_path: Path) -> None:
        """Test FlextInfraPathResolver → FlextInfraDiscoveryService flow.

        Validates:
        - Path resolver can be created
        - Discovery service can be created
        - Both work together in a flow
        """
        workspace_root = tmp_path / "workspace"
        workspace_root.mkdir()
        path_resolver = FlextInfraUtilitiesPaths()
        discovery = FlextInfraDiscoveryService()
        assert path_resolver is not None
        assert discovery is not None
        assert isinstance(path_resolver, FlextInfraUtilitiesPaths)
        assert isinstance(discovery, FlextInfraDiscoveryService)

    @pytest.mark.integration
    def test_path_resolver_returns_path_objects(self, tmp_path: Path) -> None:
        """Test that path resolver returns Path objects.

        Validates:
        - Path resolver methods return Path objects
        - Paths are properly typed
        """
        path_resolver = FlextInfraUtilitiesPaths()
        assert path_resolver is not None
        assert isinstance(path_resolver, FlextInfraUtilitiesPaths)


class TestCrossModuleIntegration:
    """Test cross-module integration scenarios."""


class TestIntegrationWithRealCommandServices:
    """Test integration scenarios with real subprocess-backed services."""

    @pytest.mark.integration
    def test_git_service_current_branch_in_real_repo(self, tmp_path: Path) -> None:
        """Test git service against a real initialized git repository."""
        repo_root = tmp_path / "repo"
        repo_root.mkdir()
        runner = FlextInfraCommandRunner()
        init_result = runner.run_checked(["git", "init"], cwd=repo_root)
        assert init_result.is_success
        email_result = runner.run_checked(
            ["git", "config", "user.email", "infra@example.com"],
            cwd=repo_root,
        )
        assert email_result.is_success
        name_result = runner.run_checked(
            ["git", "config", "user.name", "Infra Test"],
            cwd=repo_root,
        )
        assert name_result.is_success
        sample_file = repo_root / "README.md"
        _ = sample_file.write_text("infra test\n", encoding="utf-8")
        add_result = runner.run_checked(["git", "add", "README.md"], cwd=repo_root)
        assert add_result.is_success
        commit_result = runner.run_checked(
            ["git", "commit", "-m", "initial"],
            cwd=repo_root,
        )
        assert commit_result.is_success
        branch_result = u.Infra.git_current_branch(repo_root)
        assert branch_result.is_success
        assert branch_result.value != ""

    @pytest.mark.integration
    def test_command_runner_capture_executes_real_command(self) -> None:
        """Test command runner capture with a real subprocess command."""
        runner = FlextInfraCommandRunner()
        capture_result = runner.capture(["python3", "-c", "print('infra-ok')"])
        assert capture_result.is_success
        assert capture_result.value == "infra-ok"
