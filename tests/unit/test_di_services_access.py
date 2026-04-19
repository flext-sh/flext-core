"""Incremental tests for service access via Dependency Injection.

Module: flext_core DI services validation
Scope: Real execution validating that core services
       (settings, logger contract, context) are easily accessible via DI
       following Clear Architecture principles

Tests service accessibility via DI:
- FlextSettings via container and service runtime
- Logger contract via container and service runtime
- FlextContext via container and service runtime
- Service registration and injection patterns
- Integration with create_service_runtime

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import ModuleType
from typing import override

from flext_core import (
    FlextContainer,
    FlextContext,
    FlextModelsService,
    FlextSettings,
)
from flext_tests import r, s, tm
from tests import p, t, u


class TestDiServicesAccess:
    """Test FlextSettings accessibility via DI."""

    def test_config_via_container_fetch_global(self) -> None:
        """Test accessing FlextSettings via fetch_global."""
        config1 = FlextSettings.fetch_global()
        config2 = FlextSettings.fetch_global()
        assert config1 is config2
        assert isinstance(config1, p.Settings)

    def test_config_via_service_runtime(self) -> None:
        """Test FlextSettings accessible via s._create_runtime."""
        runtime = s.create_runtime(settings_overrides={"app_name": "test_app"})
        assert runtime.settings is not None
        assert isinstance(runtime.settings, p.Settings)
        assert runtime.settings.app_name == "test_app"

    def test_config_via_container_scoped(self) -> None:
        """Test FlextSettings accessible via scoped container."""
        container = FlextContainer.shared(context=FlextContext())
        scoped = container.scope(settings=FlextSettings(app_name="scoped_config"))
        assert scoped.settings is not None
        assert isinstance(scoped.settings, p.Settings)
        assert scoped.settings.app_name == "scoped_config"

    def test_config_injection_via_wiring(self) -> None:
        """Test injecting FlextSettings via @inject decorator."""
        di_container = u.DependencyIntegration.create_container(
            settings={"app_name": "injected_config"},
        )
        module = ModuleType("config_injection_module")

        @u.DependencyIntegration.inject
        def resolve_config_value(
            app_name: str = u.DependencyIntegration.Provide["settings.app_name"],
        ) -> str:
            return app_name

        setattr(module, "resolve_config_value", resolve_config_value)
        u.DependencyIntegration.wire(di_container, modules=[module])
        try:
            func = getattr(module, "resolve_config_value")
            result = func()
            assert result == "injected_config"
        finally:
            di_container.unwire()

    def test_logger_via_runtime_structlog(self) -> None:
        """Test accessing logger via u.structlog()."""
        structlog_module = u.structlog()
        assert structlog_module is not None

    def test_logger_factory_method(self) -> None:
        """Test module logger creation through the public logging DSL."""
        logger = u.create_module_logger("test_service")
        assert logger is not None
        assert callable(getattr(logger, "bind", None))

    def test_logger_registration_in_container(self) -> None:
        """Test registering a public logger in the container for DI."""
        container = FlextContainer()
        logger_result = container.resolve("logger")
        assert logger_result.success
        assert callable(getattr(logger_result.value, "bind", None))

        custom_logger = u.create_module_logger("service_logger")

        returned_container = container.bind("custom_logger", custom_logger)
        assert returned_container is container
        assert container.has("custom_logger")
        custom_logger_result = container.resolve("custom_logger")
        assert custom_logger_result.success
        assert callable(getattr(custom_logger_result.value, "bind", None))

    def test_context_via_runtime_create(self) -> None:
        """Test creating context via FlextContext.create()."""
        context = FlextContext.create()
        assert context is not None
        assert isinstance(context, p.Context)

    def test_context_via_service_runtime(self) -> None:
        """Test FlextContext accessible via s._create_runtime."""
        custom_context = FlextContext.create()
        runtime = s.create_runtime(context=custom_context)
        assert runtime.context is not None
        assert isinstance(runtime.context, p.Context)
        assert runtime.context is custom_context

    def test_context_registration_in_container(self) -> None:
        """Test registering FlextContext in container for DI."""
        container = FlextContainer.shared(context=FlextContext())
        context_result = container.resolve("context")
        assert context_result.success
        assert isinstance(context_result.value, p.Context)
        custom_context = FlextContext()
        returned_container = container.bind("custom_context", custom_context)
        assert returned_container is container
        assert container.has("custom_context")
        custom_context_result = container.resolve("custom_context")
        assert custom_context_result.success
        assert isinstance(custom_context_result.value, p.Context)
        assert custom_context_result.value is custom_context

    def test_all_services_via_service_runtime(self) -> None:
        """Test all services accessible via single service runtime."""
        custom_context = FlextContext.create()
        runtime = s.create_runtime(
            settings_overrides={"app_name": "integrated_app"},
            context=custom_context,
        )
        assert runtime.settings is not None
        assert isinstance(runtime.settings, p.Settings)
        assert runtime.settings.app_name == "integrated_app"
        assert runtime.context is not None
        assert isinstance(runtime.context, p.Context)
        assert runtime.container is not None
        assert isinstance(runtime.container, p.Container)

    def test_services_in_service_class(self) -> None:
        """Test services accessible in s subclass."""
        FlextContainer.reset_for_testing()

        class ServiceWithDI(s[str]):
            @override
            @classmethod
            def _runtime_bootstrap_options(
                cls,
            ) -> FlextModelsService.RuntimeBootstrapOptions:
                return FlextModelsService.RuntimeBootstrapOptions(
                    settings_overrides={"app_name": "service_app"},
                    container_overrides={"logger": "service"},
                )

            @override
            def execute(self) -> p.Result[str]:
                app_name = self.settings.app_name
                tm.that(app_name, eq="service_app", msg="Settings must be accessible")
                logger_result = self.container.resolve("logger")
                _ = u.Core.Tests.assert_success(logger_result)
                logger = logger_result.value
                assert logger is not None, "Logger must be accessible via DI"
                return r[str].ok(f"app: {app_name}")

        service = ServiceWithDI()
        result = service.execute()
        _ = u.Core.Tests.assert_success(result)
        assert "app: service_app" in result.value

    def test_services_injection_combined(self) -> None:
        """Test injecting multiple services via @inject."""
        services = {"logger_name": "test"}
        di_container = u.DependencyIntegration.create_container(
            settings={"app_name": "injected"},
            services=services,
        )
        module = ModuleType("services_injection_module")

        @u.DependencyIntegration.inject
        def process(
            app_name: str = u.DependencyIntegration.Provide["settings.app_name"],
        ) -> t.StrMapping:
            return {"app": app_name}

        setattr(module, "process", process)
        u.DependencyIntegration.wire(di_container, modules=[module])
        try:
            process_func = getattr(module, "process")
            result = process_func()
            assert result["app"] == "injected"
        finally:
            di_container.unwire()


__all__: list[str] = ["TestDiServicesAccess"]
