"""Incremental tests for service access via Dependency Injection.

Module: flext_core DI services validation
Scope: Real execution validating that core services
       (FlextSettings, FlextLogger, FlextContext) are easily accessible via DI
       following Clear Architecture principles

Tests service accessibility via DI:
- FlextSettings via container and service runtime
- FlextLogger via container and service runtime
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
    FlextLogger,
    FlextModelsService,
    FlextSettings,
)
from flext_tests import tm
from tests import p, r, s, t, u


class TestDiServicesAccess:
    """Test FlextSettings accessibility via DI."""

    def test_config_via_container_get_global(self) -> None:
        """Test accessing FlextSettings via get_global."""
        config1 = FlextSettings.get_global()
        config2 = FlextSettings.get_global()
        assert config1 is config2
        assert isinstance(config1, p.Settings)

    def test_config_via_service_runtime(self) -> None:
        """Test FlextSettings accessible via s._create_runtime."""
        runtime = s._create_runtime(config_overrides={"app_name": "test_app"})
        assert runtime.config is not None
        assert isinstance(runtime.config, p.Settings)
        assert runtime.config.app_name == "test_app"

    def test_config_via_container_scoped(self) -> None:
        """Test FlextSettings accessible via scoped container."""
        container = FlextContainer(_context=FlextContext())
        scoped = container.scoped(config=FlextSettings(app_name="scoped_config"))
        assert scoped.config is not None
        assert isinstance(scoped.config, p.Settings)
        assert scoped.config.app_name == "scoped_config"

    def test_config_injection_via_wiring(self) -> None:
        """Test injecting FlextSettings via @inject decorator."""
        di_container = u.DependencyIntegration.create_container(
            config=t.ConfigMap(root={"app_name": "injected_config"}),
        )
        module = ModuleType("config_injection_module")

        @u.DependencyIntegration.inject
        def get_config(
            app_name: str = u.DependencyIntegration.Provide["config.app_name"],
        ) -> str:
            return app_name

        setattr(module, "get_config", get_config)
        u.DependencyIntegration.wire(di_container, modules=[module])
        try:
            func = getattr(module, "get_config")
            result = func()
            assert result == "injected_config"
        finally:
            di_container.unwire()

    def test_logger_via_runtime_structlog(self) -> None:
        """Test accessing logger via u.structlog()."""
        structlog_module = u.structlog()
        assert structlog_module is not None

    def test_logger_factory_method(self) -> None:
        """Test FlextLogger.create_module_logger."""
        logger = FlextLogger.create_module_logger("test_service")
        assert logger is not None
        assert isinstance(logger, p.Logger)

    def test_logger_registration_in_container(self) -> None:
        """Test registering FlextLogger in container for DI."""
        container = FlextContainer()
        logger_result = container.get("logger")
        assert logger_result.is_success
        assert isinstance(logger_result.value, p.Logger)

        def create_custom_logger() -> FlextLogger:
            return FlextLogger.create_module_logger("service_logger")

        returned_container = container.register(
            "custom_logger",
            create_custom_logger,
            kind="factory",
        )
        assert returned_container is container
        assert container.has_service("custom_logger")
        custom_logger_result = container.get("custom_logger")
        assert custom_logger_result.is_success
        assert isinstance(custom_logger_result.value, p.Logger)

    def test_context_via_runtime_create(self) -> None:
        """Test creating context via FlextContext.create()."""
        context = FlextContext.create()
        assert context is not None
        assert isinstance(context, p.Context)

    def test_context_via_service_runtime(self) -> None:
        """Test FlextContext accessible via s._create_runtime."""
        custom_context = FlextContext.create()
        runtime = s._create_runtime(context=custom_context)
        assert runtime.context is not None
        assert isinstance(runtime.context, p.Context)
        assert runtime.context is custom_context

    def test_context_registration_in_container(self) -> None:
        """Test registering FlextContext in container for DI."""
        container = FlextContainer(_context=FlextContext())
        context_result = container.get("context")
        assert context_result.is_success
        assert isinstance(context_result.value, p.Context)
        custom_context = FlextContext()
        returned_container = container.register(
            "custom_context",
            custom_context,
        )
        assert returned_container is container
        assert container.has_service("custom_context")
        custom_context_result = container.get("custom_context")
        assert custom_context_result.is_success
        assert isinstance(custom_context_result.value, p.Context)
        assert custom_context_result.value is custom_context

    def test_all_services_via_service_runtime(self) -> None:
        """Test all services accessible via single service runtime."""
        custom_context = FlextContext.create()
        runtime = s._create_runtime(
            config_overrides={"app_name": "integrated_app"},
            context=custom_context,
        )
        assert runtime.config is not None
        assert isinstance(runtime.config, p.Settings)
        assert runtime.config.app_name == "integrated_app"
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
                    config_overrides={"app_name": "service_app"},
                    container_overrides={"logger": "service"},
                )

            @override
            def execute(self) -> r[str]:
                app_name = self.config.app_name
                tm.that(app_name, eq="service_app", msg="Config must be accessible")
                logger_result = self.container.get("logger")
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
            config=t.ConfigMap(root={"app_name": "injected"}),
            services=services,
        )
        module = ModuleType("services_injection_module")

        @u.DependencyIntegration.inject
        def process(
            app_name: str = u.DependencyIntegration.Provide["config.app_name"],
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


__all__ = ["TestDiServicesAccess"]
