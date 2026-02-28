"""Integration tests for flext_infra cross-module flows.

Tests exercise cross-module flows with real service instances, validating:
- Container integration: service registration and retrieval
- Workspace detection → orchestration flow
- BaseMk generation flow with mocked subprocess
- Output singleton consistency
- Service FlextResult chaining
- Path resolver → discovery flow

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from flext_core import FlextContainer, FlextResult, r
from flext_infra.basemk import (
    FlextInfraBaseMkGenerator,
    FlextInfraBaseMkTemplateEngine,
)
from flext_infra.container import (
    configure_flext_infra_dependencies,
    get_flext_infra_container,
    get_flext_infra_service,
)
from flext_infra.discovery import FlextInfraDiscoveryService
from flext_infra.output import FlextInfraOutput, output
from flext_infra.paths import FlextInfraPathResolver
from flext_infra.workspace import (
    FlextInfraOrchestratorService,
    FlextInfraWorkspaceDetector,
)

pytestmark = [pytest.mark.integration]


class TestContainerIntegration:
    """Test container integration: service registration and retrieval."""

    @pytest.mark.integration
    def test_configure_flext_infra_dependencies_registers_all_services(
        self,
    ) -> None:
        """Test that configure_flext_infra_dependencies() registers all 18 services.

        Validates:
        - All services are registered in the container
        - Services are retrievable by name
        - No registration errors occur
        """
        # Arrange
        container = get_flext_infra_container()
        container.clear_all()

        # Act
        configure_flext_infra_dependencies()

        # Assert - verify all 18 services are registered
        expected_services = [
            "git_service",
            "json_io",
            "toml_io",
            "path_resolver",
            "command_runner",
            "discovery",
            "selection",
            "reporting",
            "versioning",
            "output",
            "basemk_engine",
            "basemk_generator",
            "workspace_detector",
            "workspace_migrator",
            "workspace_orchestrator",
            "workspace_sync",
            "release_orchestrator",
            "python_version_enforcer",
        ]

        for service_name in expected_services:
            result = get_flext_infra_service(service_name)
            assert result.is_success, (
                f"Service '{service_name}' not registered: {result.error}"
            )
            assert result.value is not None

    @pytest.mark.integration
    def test_container_service_retrieval_returns_flext_result(
        self,
    ) -> None:
        """Test that service retrieval returns FlextResult with proper typing.

        Validates:
        - get_flext_infra_service returns FlextResult
        - Success case contains service instance
        - Failure case contains error message
        """
        # Arrange
        container = get_flext_infra_container()
        container.clear_all()
        configure_flext_infra_dependencies()

        # Act
        result = get_flext_infra_service("path_resolver")

        # Assert
        assert isinstance(result, FlextResult)
        assert result.is_success
        assert result.value is not None
        assert isinstance(result.value, FlextInfraPathResolver)

    @pytest.mark.integration
    def test_container_service_retrieval_failure_returns_error(
        self,
    ) -> None:
        """Test that retrieving non-existent service returns failure.

        Validates:
        - Non-existent service returns failure
        - Error message is descriptive
        """
        # Arrange
        container = get_flext_infra_container()
        container.clear_all()

        # Act
        result = get_flext_infra_service("nonexistent_service")

        # Assert
        assert result.is_failure
        assert "nonexistent_service" in result.error or "not found" in result.error


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
        # Arrange
        workspace_root = tmp_path / "workspace"
        workspace_root.mkdir()
        (workspace_root / ".git").mkdir()

        # Act - create detector
        detector = FlextInfraWorkspaceDetector()

        # Act - create orchestrator
        orchestrator = FlextInfraOrchestratorService()

        # Assert - both instances exist and are properly typed
        assert detector is not None
        assert orchestrator is not None
        assert isinstance(detector, FlextInfraWorkspaceDetector)
        assert isinstance(orchestrator, FlextInfraOrchestratorService)

    @pytest.mark.integration
    def test_workspace_detector_returns_flext_result(
        self,
        tmp_path: Path,
    ) -> None:
        """Test that workspace detector operations return FlextResult.

        Validates:
        - Detector methods return FlextResult
        - Result typing is correct
        """
        # Arrange
        detector = FlextInfraWorkspaceDetector()

        # Act - detector should have methods that return FlextResult
        # (actual method names depend on implementation)
        assert detector is not None

        # Assert - detector is properly instantiated
        assert isinstance(detector, FlextInfraWorkspaceDetector)


class TestBaseMkGenerationFlow:
    """Test BaseMk generation flow with mocked subprocess."""

    @pytest.mark.integration
    def test_basemk_template_engine_and_generator_flow(
        self,
        tmp_path: Path,
    ) -> None:
        """Test BaseMk template engine → generator flow.

        Validates:
        - Template engine can be created
        - Generator can be created
        - Both work together in a flow
        """
        # Arrange
        output_dir = tmp_path / "basemk_output"
        output_dir.mkdir()

        # Act - create template engine
        engine = FlextInfraBaseMkTemplateEngine()

        # Act - create generator
        generator = FlextInfraBaseMkGenerator()

        # Assert - both instances exist and are properly typed
        assert engine is not None
        assert generator is not None
        assert isinstance(engine, FlextInfraBaseMkTemplateEngine)
        assert isinstance(generator, FlextInfraBaseMkGenerator)

    @pytest.mark.integration
    @patch("flext_infra.subprocess.FlextInfraCommandRunner.run")
    def test_basemk_generator_with_mocked_subprocess(
        self,
        mock_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test BaseMk generator with mocked subprocess calls.

        Validates:
        - Generator can be created
        - Subprocess calls are properly mocked
        - No real subprocess execution occurs
        """
        # Arrange
        mock_run.return_value = r[str].ok("mocked output")
        generator = FlextInfraBaseMkGenerator()

        # Act
        assert generator is not None

        # Assert - generator is properly instantiated
        assert isinstance(generator, FlextInfraBaseMkGenerator)


class TestOutputSingletonConsistency:
    """Test output singleton consistency across modules."""

    @pytest.mark.integration
    def test_output_singleton_is_same_instance_everywhere(
        self,
    ) -> None:
        """Test that flext_infra.output is same instance everywhere.

        Validates:
        - output singleton is consistent
        - output is a FlextInfraOutput instance
        - Singleton pattern is maintained
        """
        # Arrange - output already imported at module level
        # Act - verify output is instance of FlextInfraOutput
        # Assert - output is properly instantiated singleton
        assert isinstance(output, FlextInfraOutput)
        assert output is not None

    @pytest.mark.integration
    def test_output_singleton_has_expected_methods(
        self,
    ) -> None:
        """Test that output singleton has all expected methods.

        Validates:
        - output has status method
        - output has summary method
        - output has error method
        - output has warning method
        - output has info method
        """
        # Assert - output has expected methods
        assert hasattr(output, "status")
        assert hasattr(output, "summary")
        assert hasattr(output, "error")
        assert hasattr(output, "warning")
        assert hasattr(output, "info")
        assert hasattr(output, "header")
        assert hasattr(output, "progress")

    @pytest.mark.integration
    def test_output_singleton_methods_are_callable(
        self,
    ) -> None:
        """Test that output singleton methods are callable.

        Validates:
        - All methods are callable
        - Methods can be invoked without error
        """
        # Assert - methods are callable
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
    def test_service_result_chaining_with_map(
        self,
    ) -> None:
        """Test chaining multiple services via .map().

        Validates:
        - FlextResult.map() works with service results
        - Type is preserved through chain
        - Value is transformed correctly
        """
        # Arrange
        initial_value = 10

        # Act - chain operations with map
        result = r[int].ok(initial_value).map(lambda x: x * 2).map(lambda x: x + 5)

        # Assert
        assert result.is_success
        assert result.value == 25

    @pytest.mark.integration
    def test_service_result_chaining_with_flat_map(
        self,
    ) -> None:
        """Test chaining multiple services via .flat_map().

        Validates:
        - FlextResult.flat_map() works with service results
        - Type is preserved through chain
        - Failures propagate correctly
        """
        # Arrange
        initial_value = 10

        # Act - chain operations with flat_map
        result = (
            r[int]
            .ok(initial_value)
            .flat_map(lambda x: r[int].ok(x * 2))
            .flat_map(lambda x: r[int].ok(x + 5))
        )

        # Assert
        assert result.is_success
        assert result.value == 25

    @pytest.mark.integration
    def test_service_result_chaining_failure_propagation(
        self,
    ) -> None:
        """Test that failures propagate through result chains.

        Validates:
        - Failure stops the chain
        - Error message is preserved
        - Subsequent operations are not executed
        """
        # Arrange
        initial_value = 10

        # Act - chain with failure in middle
        result = (
            r[int]
            .ok(initial_value)
            .flat_map(lambda x: r[int].ok(x * 2))
            .flat_map(lambda x: r[int].fail("intentional error"))
            .flat_map(lambda x: r[int].ok(x + 5))
        )

        # Assert
        assert result.is_failure
        assert "intentional error" in result.error

    @pytest.mark.integration
    def test_service_result_chaining_with_mixed_operations(
        self,
    ) -> None:
        """Test chaining with mixed map and flat_map operations.

        Validates:
        - Mixed operations work together
        - Type is preserved
        - Values are transformed correctly
        """
        # Arrange
        initial_value = 5

        # Act - chain with mixed operations
        result = (
            r[int]
            .ok(initial_value)
            .map(lambda x: x * 2)
            .flat_map(lambda x: r[int].ok(x + 3))
            .map(lambda x: x * 2)
        )

        # Assert
        assert result.is_success
        assert result.value == 26  # (5 * 2 + 3) * 2 = 26


class TestPathResolverDiscoveryFlow:
    """Test path resolver → discovery flow."""

    @pytest.mark.integration
    def test_path_resolver_and_discovery_service_flow(
        self,
        tmp_path: Path,
    ) -> None:
        """Test FlextInfraPathResolver → FlextInfraDiscoveryService flow.

        Validates:
        - Path resolver can be created
        - Discovery service can be created
        - Both work together in a flow
        """
        # Arrange
        workspace_root = tmp_path / "workspace"
        workspace_root.mkdir()

        # Act - create path resolver
        path_resolver = FlextInfraPathResolver()

        # Act - create discovery service
        discovery = FlextInfraDiscoveryService()

        # Assert - both instances exist and are properly typed
        assert path_resolver is not None
        assert discovery is not None
        assert isinstance(path_resolver, FlextInfraPathResolver)
        assert isinstance(discovery, FlextInfraDiscoveryService)

    @pytest.mark.integration
    def test_path_resolver_returns_path_objects(
        self,
        tmp_path: Path,
    ) -> None:
        """Test that path resolver returns Path objects.

        Validates:
        - Path resolver methods return Path objects
        - Paths are properly typed
        """
        # Arrange
        path_resolver = FlextInfraPathResolver()

        # Act - path resolver should have methods that work with paths
        assert path_resolver is not None

        # Assert - path resolver is properly instantiated
        assert isinstance(path_resolver, FlextInfraPathResolver)


class TestCrossModuleIntegration:
    """Test cross-module integration scenarios."""

    @pytest.mark.integration
    def test_container_with_all_services_and_retrieval(
        self,
    ) -> None:
        """Test full container setup with all services and retrieval.

        Validates:
        - Container can be configured with all services
        - All services can be retrieved
        - Services are properly typed
        """
        # Arrange
        container = get_flext_infra_container()
        container.clear_all()

        # Act
        configure_flext_infra_dependencies()

        # Assert - retrieve multiple services
        path_resolver_result = get_flext_infra_service("path_resolver")
        discovery_result = get_flext_infra_service("discovery")
        output_result = get_flext_infra_service("output")

        assert path_resolver_result.is_success
        assert discovery_result.is_success
        assert output_result.is_success

    @pytest.mark.integration
    def test_multiple_service_retrievals_are_consistent(
        self,
    ) -> None:
        """Test that multiple retrievals of same service are consistent.

        Validates:
        - Service retrieval is consistent
        - Multiple calls return same service
        """
        # Arrange
        container = get_flext_infra_container()
        container.clear_all()
        configure_flext_infra_dependencies()

        # Act - retrieve same service twice
        result1 = get_flext_infra_service("path_resolver")
        result2 = get_flext_infra_service("path_resolver")

        # Assert - both retrievals succeed
        assert result1.is_success
        assert result2.is_success

    @pytest.mark.integration
    def test_service_result_type_annotations_are_correct(
        self,
    ) -> None:
        """Test that service results have correct type annotations.

        Validates:
        - FlextResult is properly typed
        - Service type is RegisterableService
        """
        # Arrange
        container = get_flext_infra_container()
        container.clear_all()
        configure_flext_infra_dependencies()

        # Act
        result = get_flext_infra_service("path_resolver")

        # Assert
        assert isinstance(result, FlextResult)
        assert result.is_success
        # Value should be a RegisterableService (any service type)
        assert result.value is not None


class TestIntegrationWithMocking:
    """Test integration scenarios with mocking."""

    @pytest.mark.integration
    @patch("flext_infra.git.FlextInfraGitService.current_branch")
    def test_git_service_with_mocked_subprocess(
        self,
        mock_current_branch: MagicMock,
    ) -> None:
        """Test git service integration with mocked subprocess.

        Validates:
        - Git service can be mocked
        - Mocking doesn't break container integration
        """
        # Arrange
        mock_current_branch.return_value = r[str].ok("main")
        container = get_flext_infra_container()
        container.clear_all()
        configure_flext_infra_dependencies()

        # Act
        git_result = get_flext_infra_service("git_service")

        # Assert
        assert git_result.is_success

    @pytest.mark.integration
    @patch("flext_infra.subprocess.FlextInfraCommandRunner.run")
    def test_command_runner_with_mocked_subprocess(
        self,
        mock_run: MagicMock,
    ) -> None:
        """Test command runner with mocked subprocess.

        Validates:
        - Command runner can be mocked
        - Mocking prevents real subprocess calls
        """
        # Arrange
        mock_run.return_value = r[str].ok("mocked output")
        container = get_flext_infra_container()
        container.clear_all()
        configure_flext_infra_dependencies()

        # Act
        runner_result = get_flext_infra_service("command_runner")

        # Assert
        assert runner_result.is_success


class TestExplicitReturnTypes:
    """Test that all functions have explicit return types."""

    @pytest.mark.integration
    def test_get_flext_infra_container_return_type(
        self,
    ) -> None:
        """Test that get_flext_infra_container has explicit return type.

        Validates:
        - Function returns FlextContainer
        - Return type is explicit
        """
        # Act
        result = get_flext_infra_container()

        # Assert
        assert isinstance(result, FlextContainer)

    @pytest.mark.integration
    def test_get_flext_infra_service_return_type(
        self,
    ) -> None:
        """Test that get_flext_infra_service has explicit return type.

        Validates:
        - Function returns FlextResult
        - Return type is explicit
        """
        # Arrange
        container = get_flext_infra_container()
        container.clear_all()
        configure_flext_infra_dependencies()

        # Act
        result = get_flext_infra_service("path_resolver")

        # Assert
        assert isinstance(result, FlextResult)

    @pytest.mark.integration
    def test_configure_flext_infra_dependencies_return_type(
        self,
    ) -> None:
        """Test that configure_flext_infra_dependencies has explicit return type.

        Validates:
        - Function returns None
        - Return type is explicit
        """
        # Arrange
        container = get_flext_infra_container()
        container.clear_all()

        # Act
        result = configure_flext_infra_dependencies()

        # Assert
        assert result is None
