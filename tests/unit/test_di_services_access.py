"""Incremental tests for service access via Dependency Injection.

Module: flext_core DI services validation
Scope: Real execution validating that core services
       (FlextConfig, FlextLogger, FlextContext) are easily accessible via DI
       following Clear Architecture principles

Tests service accessibility via DI:
- FlextConfig via container and service runtime
- FlextLogger via container and service runtime
- FlextContext via container and service runtime
- Service registration and injection patterns
- Integration with create_service_runtime

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import ModuleType
from typing import cast

from flext_core import (
    FlextConfig,
    FlextContainer,
    FlextContext,
    FlextLogger,
    FlextRuntime,
    r,
    s,
    t,
)
from flext_tests.matchers import tm
from flext_tests.utilities import u


class TestConfigServiceViaDI:
    """Test FlextConfig accessibility via DI."""

    def test_config_via_container_get_global(self) -> None:
        """Test accessing FlextConfig via container.get_global_instance."""
        # Config is accessible via singleton pattern
        config1 = FlextConfig.get_global_instance()
        config2 = FlextConfig.get_global_instance()
        assert config1 is config2
        assert isinstance(config1, FlextConfig)

    def test_config_via_service_runtime(self) -> None:
        """Test FlextConfig accessible via create_service_runtime."""
        runtime = FlextRuntime.create_service_runtime(
            config_overrides={"app_name": "test_app"}
        )
        assert runtime.config is not None
        assert isinstance(runtime.config, FlextConfig)
        assert runtime.config.app_name == "test_app"

    def test_config_via_container_scoped(self) -> None:
        """Test FlextConfig accessible via scoped container."""
        container = FlextContainer()
        scoped = container.scoped(config=FlextConfig(app_name="scoped_config"))
        assert scoped.config is not None
        assert isinstance(scoped.config, FlextConfig)
        assert scoped.config.app_name == "scoped_config"

    def test_config_injection_via_wiring(self) -> None:
        """Test injecting FlextConfig via @inject decorator."""
        # Use DependencyIntegration to create container with config
        di_container = FlextRuntime.DependencyIntegration.create_container(
            config={"app_name": "injected_config"}
        )

        module = ModuleType("config_injection_module")

        @FlextRuntime.DependencyIntegration.inject
        def get_config(
            app_name: str = FlextRuntime.DependencyIntegration.Provide[
                "config.app_name"
            ],
        ) -> str:
            return app_name

        # Type narrowing: ModuleType can have dynamic attributes
        setattr(module, "get_config", get_config)

        # Wire module
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

        # Logger is auto-registered by default, so we can retrieve it directly
        # or register a custom one with a different name
        logger_result: r[FlextLogger] = container.get("logger")
        assert logger_result.is_success
        assert isinstance(logger_result.value, FlextLogger)

        # Test registering a custom logger with a different name
        def create_custom_logger() -> FlextLogger:
            return FlextLogger.create_module_logger("service_logger")

        # Register custom logger factory with different name
        result = container.register_factory("custom_logger", create_custom_logger)
        assert result.is_success

        # Retrieve custom logger
        custom_logger_result: r[FlextLogger] = container.get("custom_logger")
        assert custom_logger_result.is_success
        assert isinstance(custom_logger_result.value, FlextLogger)


class TestContextServiceViaDI:
    """Test FlextContext accessibility via DI."""

    def test_context_via_runtime_create(self) -> None:
        """Test creating context via FlextRuntime.create_context()."""
        context = FlextRuntime.create_context()
        assert context is not None
        assert isinstance(context, FlextContext)

    def test_context_via_service_runtime(self) -> None:
        """Test FlextContext accessible via create_service_runtime."""
        custom_context = FlextContext()
        runtime = FlextRuntime.create_service_runtime(context=custom_context)
        assert runtime.context is not None
        assert isinstance(runtime.context, FlextContext)
        assert runtime.context is custom_context

    def test_context_registration_in_container(self) -> None:
        """Test registering FlextContext in container for DI."""
        container = FlextContainer()

        # Context is auto-registered by default, so we can retrieve it directly
        context_result: r[FlextContext] = container.get("context")
        assert context_result.is_success
        assert isinstance(context_result.value, FlextContext)

        # Test registering a custom context with a different name
        custom_context = FlextContext()
        result = container.register("custom_context", custom_context)
        assert result.is_success

        # Retrieve custom context
        custom_context_result: r[FlextContext] = container.get("custom_context")
        assert custom_context_result.is_success
        assert isinstance(custom_context_result.value, FlextContext)
        assert custom_context_result.value is custom_context


class TestServicesIntegrationViaDI:
    """Test integration of all services via DI."""

    def test_all_services_via_service_runtime(self) -> None:
        """Test all services accessible via single service runtime."""
        custom_context = FlextContext()

        runtime = FlextRuntime.create_service_runtime(
            config_overrides={"app_name": "integrated_app"},
            context=custom_context,
        )

        # Verify all services accessible
        assert runtime.config is not None
        assert isinstance(runtime.config, FlextConfig)
        assert runtime.config.app_name == "integrated_app"

        assert runtime.context is not None
        assert isinstance(runtime.context, FlextContext)

        assert runtime.container is not None
        assert isinstance(runtime.container, FlextContainer)

    def test_services_in_service_class(self) -> None:
        """Test services accessible in FlextService subclass."""

        class ServiceWithDI(s[str]):
            @classmethod
            def _runtime_bootstrap_options(cls) -> t.Types.RuntimeBootstrapOptions:
                # Create a factory function that returns a context instance
                def create_context() -> FlextContext:
                    """Factory function for creating context instances."""
                    return FlextContext()

                return {
                    "config_overrides": {"app_name": "service_app"},
                    "services": {
                        "logger": FlextLogger.create_module_logger("service"),
                        "custom_context": create_context(),
                    },
                }

            def execute(self) -> r[str]:
                # Access config
                app_name = self.config.app_name
                tm.that(app_name, eq="service_app", msg="Config must be accessible")

                # Access container services
                # Type narrowing: container.get returns r[T], cast to help mypy
                container_get_result: object = self.container.get("logger")
                logger_result = cast("r[t.GeneralValueType]", container_get_result)
                u.Tests.Result.assert_success(logger_result)
                logger = cast("FlextLogger", logger_result.value)
                tm.that(
                    logger,
                    is_=FlextLogger,
                    none=False,
                    msg="Logger must be accessible via DI",
                )

                # Type narrowing: container.get returns r[T], cast to help mypy
                container_get_result2: object = self.container.get("custom_context")
                context_result = cast("r[t.GeneralValueType]", container_get_result2)
                u.Tests.Result.assert_success(context_result)
                context = cast("FlextContext", context_result.value)
                tm.that(
                    context,
                    is_=FlextContext,
                    none=False,
                    msg="Context must be accessible via DI",
                )

                return r[str].ok(f"app: {app_name}")

        service = ServiceWithDI()
        result = service.execute()
        assert result.is_success
        assert "app: service_app" in result.value

    def test_services_injection_combined(self) -> None:
        """Test injecting multiple services via @inject."""
        # Use DependencyIntegration to create container with config and services
        # This ensures config provider is properly configured
        di_container = FlextRuntime.DependencyIntegration.create_container(
            config={"app_name": "injected"},
            services={
                "logger": FlextLogger.create_module_logger("test"),
                "context": FlextContext(),
            },
        )

        # Create module with injected function
        module = ModuleType("services_injection_module")

        @FlextRuntime.DependencyIntegration.inject
        def process(
            app_name: str = FlextRuntime.DependencyIntegration.Provide[
                "config.app_name"
            ],
        ) -> dict[str, str]:
            return {"app": app_name}

        # Type narrowing: ModuleType can have dynamic attributes
        setattr(module, "process", process)

        # Wire module
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
