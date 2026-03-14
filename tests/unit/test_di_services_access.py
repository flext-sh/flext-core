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
    FlextRuntime,
    FlextSettings,
    m,
    r,
    s,
)
from flext_core._models.service import FlextModelsService
from flext_tests import tm, u
from tests.test_utils import assertion_helpers


class TestConfigServiceViaDI:
    """Test FlextSettings accessibility via DI."""

    def test_config_via_container_get_global(self) -> None:
        """Test accessing FlextSettings via get_global."""
        config1 = FlextSettings.get_global()
        config2 = FlextSettings.get_global()
        assert config1 is config2
        assert isinstance(config1, FlextSettings)

    def test_config_via_service_runtime(self) -> None:
        """Test FlextSettings accessible via FlextService._create_runtime."""
        runtime = s._create_runtime(config_overrides={"app_name": "test_app"})
        assert runtime.config is not None
        assert isinstance(runtime.config, FlextSettings)
        assert runtime.config.app_name == "test_app"

    def test_config_via_container_scoped(self) -> None:
        """Test FlextSettings accessible via scoped container."""
        container = FlextContainer(_context=FlextContext())
        scoped = container.scoped(config=FlextSettings(app_name="scoped_config"))
        assert scoped.config is not None
        assert isinstance(scoped.config, FlextSettings)
        assert scoped.config.app_name == "scoped_config"

    def test_config_injection_via_wiring(self) -> None:
        """Test injecting FlextSettings via @inject decorator."""
        di_container = FlextRuntime.DependencyIntegration.create_container(
            config=m.ConfigMap(root={"app_name": "injected_config"}),
        )
        module = ModuleType("config_injection_module")

        @FlextRuntime.DependencyIntegration.inject
        def get_config(
            app_name: str = FlextRuntime.DependencyIntegration.Provide[
                "config.app_name"
            ],
        ) -> str:
            return app_name

        setattr(module, "get_config", get_config)
        FlextRuntime.DependencyIntegration.wire(di_container, modules=[module])
        try:
            func = getattr(module, "get_config")
            result = func()
            assert result == "injected_config"
        finally:
            di_container.unwire()


class TestLoggerServiceViaDI:
    """Test FlextLogger accessibility via DI."""

    def test_logger_via_runtime_structlog(self) -> None:
        """Test accessing logger via FlextRuntime.structlog()."""
        structlog_module = FlextRuntime.structlog()
        assert structlog_module is not None
        assert hasattr(structlog_module, "get_logger")
        assert hasattr(structlog_module, "configure")

    def test_logger_factory_method(self) -> None:
        """Test FlextLogger.create_module_logger."""
        logger = FlextLogger.create_module_logger("test_service")
        assert logger is not None
        assert isinstance(logger, FlextLogger)
        assert hasattr(logger, "info")
        assert hasattr(logger, "error")

    def test_logger_registration_in_container(self) -> None:
        """Test registering FlextLogger in container for DI."""
        container = FlextContainer()
        logger_result = container.get("logger")
        assert logger_result.is_success
        assert isinstance(logger_result.value, FlextLogger)

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
        assert isinstance(custom_logger_result.value, FlextLogger)


class TestContextServiceViaDI:
    """Test FlextContext accessibility via DI."""

    def test_context_via_runtime_create(self) -> None:
        """Test creating context via FlextContext.create()."""
        context = FlextContext.create()
        assert context is not None
        assert isinstance(context, FlextContext)

    def test_context_via_service_runtime(self) -> None:
        """Test FlextContext accessible via FlextService._create_runtime."""
        custom_context = FlextContext.create()
        runtime = s._create_runtime(context=custom_context)
        assert runtime.context is not None
        assert isinstance(runtime.context, FlextContext)
        assert runtime.context is custom_context

    def test_context_registration_in_container(self) -> None:
        """Test registering FlextContext in container for DI."""
        container = FlextContainer(_context=FlextContext())
        context_result = container.get("context")
        assert context_result.is_success
        assert isinstance(context_result.value, FlextContext)
        custom_context = FlextContext()
        returned_container = container.register(
            "custom_context",
            custom_context,
        )
        assert returned_container is container
        assert container.has_service("custom_context")
        custom_context_result = container.get("custom_context")
        assert custom_context_result.is_success
        assert isinstance(custom_context_result.value, FlextContext)
        assert custom_context_result.value is custom_context


class TestServicesIntegrationViaDI:
    """Test integration of all services via DI."""

    def test_all_services_via_service_runtime(self) -> None:
        """Test all services accessible via single service runtime."""
        custom_context = FlextContext.create()
        runtime = s._create_runtime(
            config_overrides={"app_name": "integrated_app"},
            context=custom_context,
        )
        assert runtime.config is not None
        assert isinstance(runtime.config, FlextSettings)
        assert runtime.config.app_name == "integrated_app"
        assert runtime.context is not None
        assert isinstance(runtime.context, FlextContext)
        assert runtime.container is not None
        assert isinstance(runtime.container, FlextContainer)

    def test_services_in_service_class(self) -> None:
        """Test services accessible in FlextService subclass."""
        FlextContainer.reset_for_testing()

        class ServiceWithDI(s[str]):
            @classmethod
            def _runtime_bootstrap_options(cls):
                return FlextModelsService.RuntimeBootstrapOptions(
                    config_overrides={"app_name": "service_app"},
                    services={"logger": FlextLogger.create_module_logger("service")},
                )

            @override
            def execute(self) -> r[str]:
                app_name = self.config.app_name
                tm.that(app_name, eq="service_app", msg="Config must be accessible")
                logger_result = self.container.get("logger")
                _ = u.Tests.Result.assert_success(logger_result)
                logger = logger_result.value
                tm.that(logger, none=False, msg="Logger must be accessible via DI")
                assert hasattr(logger, "info") and callable(getattr(logger, "info")), (
                    "Logger must be accessible via DI"
                )
                return r[str].ok(f"app: {app_name}")

        service = ServiceWithDI()
        result = service.execute()
        _ = assertion_helpers.assert_flext_result_success(result)
        assert "app: service_app" in result.value

    def test_services_injection_combined(self) -> None:
        """Test injecting multiple services via @inject."""
        services = {"logger_name": "test"}
        di_container = FlextRuntime.DependencyIntegration.create_container(
            config=m.ConfigMap(root={"app_name": "injected"}),
            services=services,
        )
        module = ModuleType("services_injection_module")

        @FlextRuntime.DependencyIntegration.inject
        def process(
            app_name: str = FlextRuntime.DependencyIntegration.Provide[
                "config.app_name"
            ],
        ) -> dict[str, str]:
            return {"app": app_name}

        setattr(module, "process", process)
        FlextRuntime.DependencyIntegration.wire(di_container, modules=[module])
        try:
            process_func = getattr(module, "process")
            result = process_func()
            assert result["app"] == "injected"
        finally:
            di_container.unwire()


__all__ = [
    "TestConfigServiceViaDI",
    "TestContextServiceViaDI",
    "TestLoggerServiceViaDI",
    "TestServicesIntegrationViaDI",
]
