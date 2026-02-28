"""Tests for flext_infra DI container registration and retrieval.

Tests verify that all FlextInfra services are properly registered in the
DI container and can be retrieved with correct types.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import pytest
from flext_core import FlextContainer, r
from flext_infra import (
    FlextInfraBaseMkGenerator,
    FlextInfraBaseMkTemplateEngine,
    FlextInfraCommandRunner,
    FlextInfraDiscoveryService,
    FlextInfraGitService,
    FlextInfraJsonService,
    FlextInfraOutput,
    FlextInfraPathResolver,
    FlextInfraProjectSelector,
    FlextInfraPythonVersionEnforcer,
    FlextInfraReleaseOrchestrator,
    FlextInfraReportingService,
    FlextInfraTomlService,
    FlextInfraVersioningService,
    configure_flext_infra_dependencies,
    get_flext_infra_container,
    get_flext_infra_service,
    output,
)
from flext_infra.workspace import (
    FlextInfraOrchestratorService,
    FlextInfraProjectMigrator,
    FlextInfraSyncService,
    FlextInfraWorkspaceDetector,
)


class TestInfraContainerFunctions:
    """Test container accessor functions."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Ensure container is configured before each test."""
        configure_flext_infra_dependencies()

    def test_get_flext_infra_container_returns_singleton(self) -> None:
        """Verify get_flext_infra_container returns the global singleton."""
        container1 = get_flext_infra_container()
        container2 = get_flext_infra_container()
        assert container1 is container2
        assert isinstance(container1, FlextContainer)

    def test_get_flext_infra_service_returns_result(self) -> None:
        """Verify get_flext_infra_service returns FlextResult."""
        result = get_flext_infra_service("git_service")
        assert isinstance(result, r)
        assert result.is_success


class TestInfraServiceRegistration:
    """Test that all services are registered in the container."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Ensure container is configured before each test."""
        configure_flext_infra_dependencies()

    def test_git_service_registered(self) -> None:
        """Verify git_service is registered and retrievable."""
        result = get_flext_infra_service("git_service")
        assert result.is_success
        service = result.unwrap()
        assert isinstance(service, FlextInfraGitService)

    def test_json_io_registered(self) -> None:
        """Verify json_io is registered and retrievable."""
        result = get_flext_infra_service("json_io")
        assert result.is_success
        service = result.unwrap()
        assert isinstance(service, FlextInfraJsonService)

    def test_toml_io_registered(self) -> None:
        """Verify toml_io is registered and retrievable."""
        result = get_flext_infra_service("toml_io")
        assert result.is_success
        service = result.unwrap()
        assert isinstance(service, FlextInfraTomlService)

    def test_path_resolver_registered(self) -> None:
        """Verify path_resolver is registered and retrievable."""
        result = get_flext_infra_service("path_resolver")
        assert result.is_success
        service = result.unwrap()
        assert isinstance(service, FlextInfraPathResolver)

    def test_command_runner_registered(self) -> None:
        """Verify command_runner is registered and retrievable."""
        result = get_flext_infra_service("command_runner")
        assert result.is_success
        service = result.unwrap()
        assert isinstance(service, FlextInfraCommandRunner)

    def test_discovery_registered(self) -> None:
        """Verify discovery is registered and retrievable."""
        result = get_flext_infra_service("discovery")
        assert result.is_success
        service = result.unwrap()
        assert isinstance(service, FlextInfraDiscoveryService)

    def test_selection_registered(self) -> None:
        """Verify selection is registered and retrievable."""
        result = get_flext_infra_service("selection")
        assert result.is_success
        service = result.unwrap()
        assert isinstance(service, FlextInfraProjectSelector)

    def test_reporting_registered(self) -> None:
        """Verify reporting is registered and retrievable."""
        result = get_flext_infra_service("reporting")
        assert result.is_success
        service = result.unwrap()
        assert isinstance(service, FlextInfraReportingService)

    def test_versioning_registered(self) -> None:
        """Verify versioning is registered and retrievable."""
        result = get_flext_infra_service("versioning")
        assert result.is_success
        service = result.unwrap()
        assert isinstance(service, FlextInfraVersioningService)

    def test_output_registered(self) -> None:
        """Verify output singleton is registered and retrievable."""
        result = get_flext_infra_service("output")
        assert result.is_success
        service = result.unwrap()
        assert isinstance(service, FlextInfraOutput)
        assert service is output

    def test_basemk_engine_registered(self) -> None:
        """Verify basemk_engine is registered and retrievable."""
        result = get_flext_infra_service("basemk_engine")
        assert result.is_success
        service = result.unwrap()
        assert isinstance(service, FlextInfraBaseMkTemplateEngine)

    def test_basemk_generator_registered(self) -> None:
        """Verify basemk_generator is registered and retrievable."""
        result = get_flext_infra_service("basemk_generator")
        assert result.is_success
        service = result.unwrap()
        assert isinstance(service, FlextInfraBaseMkGenerator)

    def test_workspace_detector_registered(self) -> None:
        """Verify workspace_detector is registered and retrievable."""
        result = get_flext_infra_service("workspace_detector")
        assert result.is_success
        service = result.unwrap()
        assert isinstance(service, FlextInfraWorkspaceDetector)

    def test_workspace_migrator_registered(self) -> None:
        """Verify workspace_migrator is registered and retrievable."""
        result = get_flext_infra_service("workspace_migrator")
        assert result.is_success
        service = result.unwrap()
        assert isinstance(service, FlextInfraProjectMigrator)

    def test_workspace_orchestrator_registered(self) -> None:
        """Verify workspace_orchestrator is registered and retrievable."""
        result = get_flext_infra_service("workspace_orchestrator")
        assert result.is_success
        service = result.unwrap()
        assert isinstance(service, FlextInfraOrchestratorService)

    def test_workspace_sync_registered(self) -> None:
        """Verify workspace_sync is registered and retrievable."""
        result = get_flext_infra_service("workspace_sync")
        assert result.is_success
        service = result.unwrap()
        assert isinstance(service, FlextInfraSyncService)

    def test_release_orchestrator_registered(self) -> None:
        """Verify release_orchestrator is registered and retrievable."""
        result = get_flext_infra_service("release_orchestrator")
        assert result.is_success
        service = result.unwrap()
        assert isinstance(service, FlextInfraReleaseOrchestrator)

    def test_python_version_enforcer_registered(self) -> None:
        """Verify python_version_enforcer is registered and retrievable."""
        result = get_flext_infra_service("python_version_enforcer")
        assert result.is_success
        service = result.unwrap()
        assert isinstance(service, FlextInfraPythonVersionEnforcer)


class TestInfraServiceRetrieval:
    """Test service retrieval behavior."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Ensure container is configured before each test."""
        configure_flext_infra_dependencies()

    def test_nonexistent_service_returns_failure(self) -> None:
        """Verify retrieving nonexistent service returns failure."""
        result = get_flext_infra_service("nonexistent_service")
        assert not result.is_success

    def test_service_retrieval_returns_result_type(self) -> None:
        """Verify service retrieval always returns FlextResult."""
        result = get_flext_infra_service("git_service")
        assert isinstance(result, r)

    def test_multiple_retrievals_return_same_instance(self) -> None:
        """Verify factory services return new instances on each retrieval."""
        result1 = get_flext_infra_service("git_service")
        result2 = get_flext_infra_service("git_service")
        assert result1.is_success
        assert result2.is_success
        # Factory services may return different instances
        service1 = result1.unwrap()
        service2 = result2.unwrap()
        assert isinstance(service1, FlextInfraGitService)
        assert isinstance(service2, FlextInfraGitService)

    def test_output_singleton_returns_same_instance(self) -> None:
        """Verify output singleton returns same instance on each retrieval."""
        result1 = get_flext_infra_service("output")
        result2 = get_flext_infra_service("output")
        assert result1.is_success
        assert result2.is_success
        service1 = result1.unwrap()
        service2 = result2.unwrap()
        assert service1 is service2
        assert service1 is output
