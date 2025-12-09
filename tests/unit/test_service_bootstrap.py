"""Tests for FlextService runtime bootstrap patterns.

Module: flext_core.service
Scope: Runtime bootstrap options, service runtime creation, auto-initialization

Tests service bootstrap functionality with real implementations:
- _runtime_bootstrap_options() method returns correct options
- _create_runtime() creates ServiceRuntime with all components
- _create_initial_runtime() uses bootstrap options
- Config, context, container, dispatcher, registry creation
- Auto-execute pattern with runtime bootstrap

Uses real implementations (no mocks) and flext_tests helpers.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import FlextConfig, FlextContainer, FlextContext, FlextService, r, t
from flext_tests import u


class ConcreteTestService(FlextService[r[bool]]):
    """Concrete service for testing bootstrap patterns."""

    @classmethod
    def _runtime_bootstrap_options(cls) -> t.Types.RuntimeBootstrapOptions:
        """Return bootstrap options for this service."""
        return {
            "config_overrides": {"debug": True},
            "subproject": "test",
        }

    def execute(self) -> r[bool]:
        """Execute service logic."""
        return r[bool].ok(True)


class TestServiceBootstrap:
    """Test service runtime bootstrap patterns."""

    def test_runtime_bootstrap_options_exists(self) -> None:
        """Test _runtime_bootstrap_options method exists and returns options."""
        # Act
        options = ConcreteTestService._runtime_bootstrap_options()

        # Assert - options is a dict with expected structure
        assert isinstance(options, dict)
        assert "config_overrides" in options
        assert "subproject" in options

    def test_create_runtime_with_options(self) -> None:
        """Test _create_runtime creates ServiceRuntime with bootstrap options."""
        # Act
        runtime = ConcreteTestService._create_runtime(
            config_overrides={"debug": True},
            subproject="test",
        )

        # Assert - runtime has all components
        assert runtime.config is not None
        assert runtime.context is not None
        assert runtime.container is not None
        assert runtime.dispatcher is not None
        assert runtime.registry is not None

        # Verify types
        assert isinstance(runtime.config, FlextConfig)
        assert isinstance(runtime.context, FlextContext)
        assert isinstance(runtime.container, FlextContainer)

        # Verify config overrides applied
        assert runtime.config.debug is True

    def test_create_initial_runtime_uses_bootstrap_options(self) -> None:
        """Test _create_initial_runtime uses _runtime_bootstrap_options."""
        # Act
        runtime = ConcreteTestService._create_initial_runtime()

        # Assert - runtime created with bootstrap options
        assert runtime.config is not None
        assert runtime.config.debug is True  # From bootstrap options
        assert runtime.container is not None

    def test_create_runtime_with_services(self) -> None:
        """Test _create_runtime accepts services parameter."""
        # Arrange
        test_service = {"test_key": "test_value"}

        # Act
        runtime = ConcreteTestService._create_runtime(services=test_service)

        # Assert - service registered in container
        service_result = runtime.container.get("test_key")
        u.Tests.Result.assert_result_success(service_result)
        assert service_result.value == "test_value"

    def test_create_runtime_with_factories(self) -> None:
        """Test _create_runtime accepts factories parameter."""

        # Arrange
        def factory() -> str:
            return "factory_value"

        test_factories = {"test_factory": factory}

        # Act
        runtime = ConcreteTestService._create_runtime(factories=test_factories)

        # Assert - factory registered in container
        factory_result = runtime.container.get("test_factory")
        u.Tests.Result.assert_result_success(factory_result)
        assert factory_result.value == "factory_value"

    def test_create_runtime_with_resources(self) -> None:
        """Test _create_runtime accepts resources parameter."""

        # Arrange
        def resource() -> str:
            return "resource_value"

        test_resources = {"test_resource": resource}

        # Act
        runtime = ConcreteTestService._create_runtime(resources=test_resources)

        # Assert - resource registered in container
        resource_result = runtime.container.get("test_resource")
        u.Tests.Result.assert_result_success(resource_result)
        assert resource_result.value == "resource_value"

    def test_create_runtime_with_context(self) -> None:
        """Test _create_runtime accepts context parameter."""
        # Arrange
        custom_context = FlextContext.create()

        # Act
        runtime = ConcreteTestService._create_runtime(context=custom_context)

        # Assert - custom context used
        assert runtime.context is custom_context

    def test_create_runtime_with_config_overrides(self) -> None:
        """Test _create_runtime applies config overrides."""
        # Arrange
        config_overrides = {"debug": True, "trace": True}

        # Act
        runtime = ConcreteTestService._create_runtime(config_overrides=config_overrides)

        # Assert - config overrides applied
        assert runtime.config.debug is True
        assert runtime.config.trace is True
