"""Tests for s runtime bootstrap patterns.

Module: flext_core.service
Scope: Runtime bootstrap options, service runtime creation, auto-initialization

Tests service bootstrap functionality with real implementations:
- _runtime_bootstrap_options() method returns correct options
- _create_runtime() creates ServiceRuntime with all components
- _create_initial_runtime() uses bootstrap options
- Config, context, container creation

Uses real implementations (no mocks) and flext_tests helpers.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import override

from flext_tests import u

from flext_core import FlextContext, m, p, r, s


class ConcreteTestService(s[bool]):
    """Concrete service for testing bootstrap patterns."""

    @classmethod
    @override
    def _runtime_bootstrap_options(cls) -> m.RuntimeBootstrapOptions:
        """Return bootstrap options for this service."""
        return m.RuntimeBootstrapOptions(
            config_overrides={"app_name": "test_app"},
            subproject="test",
        )

    @override
    def execute(self) -> r[bool]:
        """Execute service logic."""
        return r[bool].ok(True)


class TestServiceBootstrap:
    """Test service runtime bootstrap patterns."""

    def test_runtime_bootstrap_options_exists(self) -> None:
        """Test _runtime_bootstrap_options method exists and returns options."""
        options = ConcreteTestService._runtime_bootstrap_options()
        assert options is not None
        assert options.config_overrides is not None

    def test_create_runtime_with_options(self) -> None:
        """Test _create_runtime creates ServiceRuntime with bootstrap options."""
        runtime = ConcreteTestService._create_runtime(
            config_overrides={"app_name": "runtime_app"},
            subproject="test",
        )
        assert runtime.config is not None
        assert runtime.context is not None
        assert runtime.container is not None
        assert isinstance(runtime.config, p.Settings)
        assert isinstance(runtime.context, p.Context)
        assert isinstance(runtime.container, p.Container)
        assert runtime.config.app_name == "runtime_app"

    def test_create_initial_runtime_uses_bootstrap_options(self) -> None:
        """Test _create_initial_runtime uses _runtime_bootstrap_options."""
        runtime = ConcreteTestService()._create_initial_runtime()
        assert runtime.config is not None
        assert runtime.config.app_name == "test_app"
        assert runtime.container is not None

    def test_create_runtime_with_services(self) -> None:
        """Test _create_runtime accepts services parameter."""
        test_service = {"test_key": "test_value"}
        runtime = ConcreteTestService._create_runtime(services=test_service)
        service_result = runtime.container.get("test_key")
        _ = u.Tests.Result.assert_success(service_result)
        assert hasattr(runtime, "container")

    def test_create_runtime_with_factories(self) -> None:
        """Test _create_runtime accepts factories parameter."""

        def factory() -> str:
            return "factory_value"

        test_factories = {"test_factory": factory}
        runtime = ConcreteTestService._create_runtime(factories=test_factories)
        factory_result = runtime.container.get("test_factory")
        _ = u.Tests.Result.assert_success(factory_result)
        assert hasattr(runtime, "container")

    def test_create_runtime_with_resources(self) -> None:
        """Test _create_runtime accepts resources parameter."""

        def resource() -> str:
            return "resource_value"

        test_resources = {"test_resource": resource}
        runtime = ConcreteTestService._create_runtime(resources=test_resources)
        resource_result = runtime.container.get("test_resource")
        _ = u.Tests.Result.assert_success(resource_result)
        assert hasattr(runtime, "container")

    def test_create_runtime_with_context(self) -> None:
        """Test _create_runtime accepts context parameter."""
        custom_context = FlextContext.create()
        runtime = ConcreteTestService._create_runtime(context=custom_context)
        assert runtime.context is custom_context

    def test_create_runtime_with_config_overrides(self) -> None:
        """Test _create_runtime applies config overrides."""
        config_overrides = {"app_name": "override_app"}
        runtime = ConcreteTestService._create_runtime(config_overrides=config_overrides)
        assert runtime.config.app_name == "override_app"
