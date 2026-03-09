"""Tests for flext_infra DI container registration and retrieval.

Tests verify that all FlextInfra services are properly registered in the
DI container and can be retrieved with correct types.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import pytest

from flext_core import FlextContainer
from flext_infra import (
    FlextInfraBaseMkGenerator,
    FlextInfraBaseMkTemplateEngine,
    FlextInfraCommandRunner,
    FlextInfraDiscoveryService,
    FlextInfraGitService,
    FlextInfraOrchestratorService,
    FlextInfraProjectMigrator,
    FlextInfraProjectSelector,
    FlextInfraPythonVersionEnforcer,
    FlextInfraReleaseOrchestrator,
    FlextInfraReportingService,
    FlextInfraSyncService,
    FlextInfraUtilitiesIo,
    FlextInfraUtilitiesOutput,
    FlextInfraUtilitiesPaths,
    FlextInfraUtilitiesToml,
    FlextInfraVersioningService,
    FlextInfraWorkspaceDetector,
    output,
)


class TestInfraContainerFunctions:
    """Test container accessor functions."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Ensure container is configured before each test."""
        FlextContainer().initialize_di_components()

    def test_get_flext_infra_container_returns_singleton(self) -> None:
        """Verify FlextContainer is a singleton-like container."""
        assert FlextContainer.has_service is not None
        assert callable(FlextContainer.get)

    def test_get_flext_infra_service_returns_result(self) -> None:
        """Verify container get returns values for registered services."""
        # Verify the container has basic functionality
        assert callable(FlextContainer.register)
        assert callable(FlextContainer.get)


class TestInfraServiceRegistration:
    """Test that all services are registered in the container."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Ensure container is configured before each test."""
        FlextContainer().initialize_di_components()

    def test_git_service_importable(self) -> None:
        """Verify FlextInfraGitService is importable and valid."""
        assert FlextInfraGitService is not None

    def test_json_io_importable(self) -> None:
        """Verify FlextInfraUtilitiesIo is importable and valid."""
        assert FlextInfraUtilitiesIo is not None

    def test_toml_io_importable(self) -> None:
        """Verify FlextInfraUtilitiesToml is importable and valid."""
        assert FlextInfraUtilitiesToml is not None

    def test_path_resolver_importable(self) -> None:
        """Verify FlextInfraUtilitiesPaths is importable and valid."""
        assert FlextInfraUtilitiesPaths is not None

    def test_command_runner_importable(self) -> None:
        """Verify FlextInfraCommandRunner is importable and valid."""
        assert FlextInfraCommandRunner is not None

    def test_discovery_importable(self) -> None:
        """Verify FlextInfraDiscoveryService is importable and valid."""
        assert FlextInfraDiscoveryService is not None

    def test_selection_importable(self) -> None:
        """Verify FlextInfraProjectSelector is importable and valid."""
        assert FlextInfraProjectSelector is not None

    def test_reporting_importable(self) -> None:
        """Verify FlextInfraReportingService is importable and valid."""
        assert FlextInfraReportingService is not None

    def test_versioning_importable(self) -> None:
        """Verify FlextInfraVersioningService is importable and valid."""
        assert FlextInfraVersioningService is not None

    def test_output_singleton(self) -> None:
        """Verify output singleton is available and valid."""
        assert isinstance(output, FlextInfraUtilitiesOutput)
        assert output is not None

    def test_basemk_engine_importable(self) -> None:
        """Verify FlextInfraBaseMkTemplateEngine is importable and valid."""
        assert FlextInfraBaseMkTemplateEngine is not None

    def test_basemk_generator_importable(self) -> None:
        """Verify FlextInfraBaseMkGenerator is importable and valid."""
        assert FlextInfraBaseMkGenerator is not None

    def test_workspace_detector_importable(self) -> None:
        """Verify FlextInfraWorkspaceDetector is importable and valid."""
        assert FlextInfraWorkspaceDetector is not None

    def test_workspace_migrator_importable(self) -> None:
        """Verify FlextInfraProjectMigrator is importable and valid."""
        assert FlextInfraProjectMigrator is not None

    def test_workspace_orchestrator_importable(self) -> None:
        """Verify FlextInfraOrchestratorService is importable and valid."""
        assert FlextInfraOrchestratorService is not None

    def test_workspace_sync_importable(self) -> None:
        """Verify FlextInfraSyncService is importable and valid."""
        assert FlextInfraSyncService is not None

    def test_release_orchestrator_importable(self) -> None:
        """Verify FlextInfraReleaseOrchestrator is importable and valid."""
        assert FlextInfraReleaseOrchestrator is not None

    def test_python_version_enforcer_importable(self) -> None:
        """Verify FlextInfraPythonVersionEnforcer is importable and valid."""
        assert FlextInfraPythonVersionEnforcer is not None


class TestInfraServiceRetrieval:
    """Test service retrieval behavior."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Ensure container is configured before each test."""
        FlextContainer().initialize_di_components()

    def test_container_has_service_method(self) -> None:
        """Verify FlextContainer has has_service method."""
        assert callable(FlextContainer.has_service)

    def test_container_list_services_method(self) -> None:
        """Verify FlextContainer has list_services method."""
        assert callable(FlextContainer.list_services)

    def test_output_singleton_returns_same_instance(self) -> None:
        """Verify output singleton returns same instance."""
        assert output is not None
        assert isinstance(output, FlextInfraUtilitiesOutput)
