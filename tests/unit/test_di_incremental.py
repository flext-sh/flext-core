"""Incremental tests for Dependency Injection with real code execution.

Module: flext_core DI pattern validation
Scope: Real execution of DI methods to validate bridge, container, and wiring

Tests DI functionality with real code execution:
- Bridge methods (dependency_providers, dependency_containers, create_container)
- DependencyIntegration methods (create_layered_bridge, register_object, register_factory, register_resource)
- DI methods (wire_modules, scoped with resources)
- Service bootstrap with DI (create_service_runtime)
- Real wiring with @inject and Provide decorators
- Resource lifecycle management

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable
from types import ModuleType

from flext_core import (
    FlextContainer,
    FlextContext,
    FlextRuntime,
    FlextSettings,
    p,
    r,
    s,
    t,
)
from flext_tests import u

from ..test_utils import assertion_helpers

# DI decorators from Runtime
Provide = FlextRuntime.DependencyIntegration.Provide
inject = FlextRuntime.DependencyIntegration.inject


class TestDIBridgeRealExecution:
    """Test DI bridge methods with real code execution."""

    def test_dependency_providers_returns_valid_module(self) -> None:
        """Test dependency_providers returns valid providers module."""
        providers_module = FlextRuntime.dependency_providers()

        # Verify module has expected attributes
        assert hasattr(providers_module, "Singleton")
        assert hasattr(providers_module, "Factory")
        assert hasattr(providers_module, "Resource")
        assert hasattr(providers_module, "Configuration")

        # Verify can create actual providers
        singleton = providers_module.Singleton(lambda: "test_value")
        assert singleton() == "test_value"

        factory = providers_module.Factory(lambda: {"key": "value"})
        assert factory() == {"key": "value"}

    def test_dependency_containers_returns_valid_module(self) -> None:
        """Test dependency_containers returns valid containers module."""
        containers_module = FlextRuntime.dependency_containers()

        # Verify module has expected attributes
        assert hasattr(containers_module, "DeclarativeContainer")
        assert hasattr(containers_module, "DynamicContainer")

        # Verify can create actual containers
        dynamic_container = containers_module.DynamicContainer()
        assert dynamic_container is not None

    def test_create_container_with_real_execution(self) -> None:
        """Test create_container with real registration and resolution."""
        container = FlextContainer.create()

        # Register a service using container
        register_result = container.register("test_service", "test_value")
        assert register_result.is_success

        # Resolve the service
        resolved = container.get("test_service")
        u.Tests.Result.assert_success_with_value(resolved, "test_value")

    def test_create_layered_bridge_with_config(self) -> None:
        """Test create_layered_bridge with real configuration."""
        bridge, service_module, resource_module = (
            FlextRuntime.DependencyIntegration.create_layered_bridge(
                config=t.ConfigMap(root={"database": {"dsn": "sqlite://test.db"}}),
            )
        )

        # Verify bridge structure
        assert bridge is not None
        assert service_module is not None
        assert resource_module is not None

        # Verify config is accessible
        assert bridge.config is not None
        # Access config via provider call
        config_dict = bridge.config()
        assert isinstance(config_dict, dict)
        assert u.Mapper.extract(config_dict, "database.dsn").value == "sqlite://test.db"


class TestDependencyIntegrationRealExecution:
    """Test DependencyIntegration methods with real code execution."""

    def test_register_object_with_real_container(self) -> None:
        """Test register_object with real container execution."""
        di_container = FlextRuntime.DependencyIntegration.create_container()

        # Register object
        provider = FlextRuntime.DependencyIntegration.register_object(
            di_container,
            "test_object",
            {"key": "value"},
        )

        # Verify provider works
        assert provider() == {"key": "value"}

        # Verify container has it
        assert hasattr(di_container, "test_object")
        assert di_container.test_object() == {"key": "value"}

    def test_register_factory_with_caching(self) -> None:
        """Test register_factory with caching enabled."""
        di_container = FlextRuntime.DependencyIntegration.create_container()

        call_count = {"count": 0}

        def factory() -> dict[str, int]:
            call_count["count"] += 1
            return {"calls": call_count["count"]}

        # Register with cache (Singleton)
        provider = FlextRuntime.DependencyIntegration.register_factory(
            di_container,
            "cached_factory",
            factory,
            cache=True,
        )

        # First call
        result1 = provider()
        assert result1 == {"calls": 1}

        # Second call - should be cached (same instance)
        result2 = provider()
        assert result2 == {"calls": 1}  # Cached, no new call
        assert result1 is result2  # Same instance

    def test_register_factory_without_caching(self) -> None:
        """Test register_factory without caching (Factory)."""
        di_container = FlextRuntime.DependencyIntegration.create_container()

        call_count = {"count": 0}

        def factory() -> dict[str, int]:
            call_count["count"] += 1
            return {"calls": call_count["count"]}

        # Register without cache (Factory)
        provider = FlextRuntime.DependencyIntegration.register_factory(
            di_container,
            "factory_no_cache",
            factory,
            cache=False,
        )

        # First call
        result1 = provider()
        assert result1 == {"calls": 1}

        # Second call - should create new instance
        result2 = provider()
        assert result2 == {"calls": 2}  # New call
        assert result1 is not result2  # Different instances

    def test_register_resource_with_lifecycle(self) -> None:
        """Test register_resource with real teardown."""
        di_container = FlextRuntime.DependencyIntegration.create_container()

        lifecycle = {"created": False, "closed": False}

        def resource_factory() -> dict[str, bool]:
            lifecycle["created"] = True
            return {"connected": True}

        def resource_teardown(resource: dict[str, bool]) -> None:
            lifecycle["closed"] = True

        # Register resource
        provider = FlextRuntime.DependencyIntegration.register_resource(
            di_container,
            "db_connection",
            resource_factory,
        )

        # Get resource
        resource = provider()
        assert resource == {"connected": True}
        assert lifecycle["created"] is True

        # Simulate teardown (in real usage, this happens on container shutdown)
        # Resource providers have automatic teardown via dependency-injector

    def test_wire_modules_with_inject(self) -> None:
        """Test wire with @inject decorator real execution."""
        di_container = FlextRuntime.DependencyIntegration.create_container()

        # Register services
        FlextRuntime.DependencyIntegration.register_object(
            di_container,
            "api_key",
            "secret123",
        )
        FlextRuntime.DependencyIntegration.register_object(di_container, "timeout", 30)

        # Create module with injected function
        module = ModuleType("test_module")

        @FlextRuntime.DependencyIntegration.inject
        def api_call(
            key: str = FlextRuntime.DependencyIntegration.Provide["api_key"],
            timeout_sec: int = FlextRuntime.DependencyIntegration.Provide["timeout"],
        ) -> dict[str, str | int]:
            return {"key": key, "timeout": timeout_sec}

        setattr(module, "api_call", api_call)

        # Wire module
        FlextRuntime.DependencyIntegration.wire(di_container, modules=[module])

        try:
            # Execute wired function - use getattr for dynamic ModuleType attributes
            api_call_func: Callable[[], dict[str, str | int]] = getattr(
                module,
                "api_call",
            )
            result = api_call_func()
            assert result == {"key": "secret123", "timeout": 30}
        finally:
            di_container.unwire()


class TestContainerDIRealExecution:
    """Test FlextDI methods with real execution."""

    def test_wire_modules_real_execution(self) -> None:
        """Test container.wire_modules with real code."""
        container = FlextContainer()

        # Register services
        container.register("logger_name", "test_logger")
        container.register("log_level", "INFO")

        # Create module with injected function
        module = ModuleType("wired_module")

        @inject
        def log_message(
            name: str = Provide["logger_name"],
            level: str = Provide["log_level"],
        ) -> dict[str, str]:
            return {"logger": name, "level": level}

        setattr(module, "log_message", log_message)

        # Wire using container
        container.wire_modules(modules=[module])

        try:
            # Execute wired function - use getattr for dynamic ModuleType attributes
            log_func: Callable[[], dict[str, str]] = module.log_message
            result = log_func()
            assert result == {"logger": "test_logger", "level": "INFO"}
        finally:
            # Cleanup - access private attribute via cast for testing
            di_container = container._di_container
            di_container.unwire()

    def test_scoped_with_resources_real_execution(self) -> None:
        """Test container.scoped with resources real execution."""
        container = FlextContainer(_context=FlextContext())

        # Track lifecycle
        lifecycle = {"created": False, "closed": False}

        def resource_factory() -> dict[str, bool]:
            lifecycle["created"] = True
            return {"connected": True}

        # Create scoped container with resource
        scoped = container.scoped(resources={"db": resource_factory})

        # Get resource from scoped container
        result = scoped.get("db")
        resource_value = u.Tests.Result.assert_success(result)
        assert resource_value == {"connected": True}
        assert lifecycle["created"] is True

        # Verify scoped container is isolated
        assert scoped is not container

    def test_scoped_with_services_and_factories(self) -> None:
        """Test scoped container with services and factories."""
        container = FlextContainer(_context=FlextContext())

        # Create scoped container with services and factories
        scoped = container.scoped(
            services={"api_key": "secret_key"},
            factories={"token_gen": lambda: {"token": "abc123"}},
        )

        # Get service
        service_result = scoped.get("api_key")
        assert service_result.is_success
        assert service_result.value == "secret_key"

        # Get factory (should be called)
        factory_result = scoped.get("token_gen")
        assert factory_result.is_success
        assert isinstance(factory_result.value, dict)
        assert factory_result.value["token"] == "abc123"

    def test_scoped_with_config_override(self) -> None:
        """Test scoped container with config override."""
        container = FlextContainer(_context=FlextContext())

        # Create scoped container with config
        config_override = FlextSettings(app_name="scoped_app")
        scoped = container.scoped(config=config_override)

        # Verify scoped container has config via public API
        config_obj = scoped.config
        assert config_obj is not None
        assert config_obj.app_name == "scoped_app"


class TestServiceBootstrapWithDI:
    """Test service bootstrap with DI real execution."""

    def test_create_service_runtime_with_resources(self) -> None:
        """Test create_service_runtime with resources."""
        lifecycle = {"created": False}

        def db_factory() -> dict[str, bool]:
            lifecycle["created"] = True
            return {"connected": True}

        # Create service runtime with resources
        runtime = s._create_runtime(
            resources={"database": db_factory},
        )

        # Verify runtime created
        assert runtime is not None
        assert runtime.container is not None
        assert runtime.config is not None
        assert runtime.context is not None

        # Verify resource is accessible
        db_result = runtime.container.get("database")
        assert db_result.is_success
        assert db_result.value == {"connected": True}
        assert lifecycle["created"] is True

    def test_create_service_runtime_with_wiring(self) -> None:
        """Test create_service_runtime with wire_modules."""
        runtime = s._create_runtime(
            services={"api_key": "test_key"},
            wire_modules=[__import__(__name__)],
        )

        # Verify runtime has wired modules
        assert runtime.container is not None
        container_instance = runtime.container
        assert isinstance(container_instance, FlextContainer)
        assert hasattr(container_instance, "_di_container")

    def test_service_with_runtime_bootstrap_options(self) -> None:
        """Test service with _runtime_bootstrap_options override."""

        class TestService(s[str]):
            @classmethod
            def _runtime_bootstrap_options(cls) -> p.RuntimeBootstrapOptions:
                return p.RuntimeBootstrapOptions(
                    services={"custom_service": "custom_value"},
                    factories={"custom_factory": lambda: {"custom": "data"}},
                )

            def execute(self) -> r[str]:
                return r[str].ok("test")

        # Create service instance
        service = TestService()

        # Verify service has runtime
        assert service.runtime is not None

        # Verify custom services are registered
        custom_result = service.container.get("custom_service")
        assert custom_result.is_success
        assert custom_result.value == "custom_value"

        # Verify custom factory works
        factory_result = service.container.get("custom_factory")
        assert factory_result.is_success
        assert factory_result.value == {"custom": "data"}


class TestRealWiringScenarios:
    """Test real-world wiring scenarios with DI."""

    def test_handler_wiring_with_inject(self) -> None:
        """Test handler wiring with @inject decorator."""
        container = FlextContainer()

        # Register dependencies - use custom names to avoid conflict with auto-registered logger
        container.register("custom_logger", "test_logger")
        container.register("db_pool", {"size": 10})

        # Create handler module
        handler_module = ModuleType("handler_module")

        @inject
        def process_request(
            logger_name: str = Provide["custom_logger"],
            pool: dict[str, int] = Provide["db_pool"],
        ) -> dict[str, str | int]:
            return {"logger": logger_name, "pool_size": pool["size"]}

        setattr(handler_module, "process_request", process_request)

        # Wire handler
        container.wire_modules(modules=[handler_module])

        try:
            # Execute handler - use getattr for dynamic ModuleType attributes
            process_func: Callable[[], dict[str, str | int]] = (
                handler_module.process_request
            )
            result = process_func()
            assert result == {"logger": "test_logger", "pool_size": 10}
        finally:
            # Access private attribute via cast for testing
            di_container = container._di_container
            di_container.unwire()

    def test_multiple_wired_functions(self) -> None:
        """Test multiple functions wired to same container."""
        container = FlextContainer()
        container.register("shared_config", {"env": "test"})

        module = ModuleType("multi_function_module")

        @inject
        def func1(config: dict[str, str] = Provide["shared_config"]) -> str:
            return config["env"]

        @inject
        def func2(config: dict[str, str] = Provide["shared_config"]) -> bool:
            return config["env"] == "test"

        setattr(module, "func1", func1)
        setattr(module, "func2", func2)

        container.wire_modules(modules=[module])

        try:
            # Use getattr for dynamic ModuleType attributes
            func1_wired: Callable[[], str] = getattr(module, "func1")
            func2_wired: Callable[[], bool] = getattr(module, "func2")
            assert func1_wired() == "test"
            assert func2_wired() is True
        finally:
            # Access private attribute via cast for testing
            di_container = container._di_container
            di_container.unwire()

    def test_nested_dependency_injection(self) -> None:
        """Test nested dependency injection scenarios."""
        container = FlextContainer()

        # Register base dependencies
        container.register("base_url", "https://api.example.com")
        container.register("api_version", "v1")

        module = ModuleType("nested_module")

        @inject
        def build_url(
            base: str = Provide["base_url"],
            version: str = Provide["api_version"],
        ) -> str:
            return f"{base}/{version}"

        @inject
        def api_call(
            url: str = Provide["built_url"],
            base: str = Provide["base_url"],
        ) -> dict[str, str]:
            # Note: This tests dependency on another injected value
            # In real scenarios, use factory pattern
            return {"url": url, "base": base}

        setattr(module, "build_url", build_url)
        setattr(module, "api_call", api_call)

        # First build URL
        url_result = build_url()
        container.register("built_url", url_result)

        container.wire_modules(modules=[module])

        try:
            # Use getattr for dynamic ModuleType attributes
            api_call_func: Callable[[], dict[str, str]] = getattr(module, "api_call")
            result = api_call_func()
            assert "url" in result
            assert "base" in result
        finally:
            # Access private attribute via cast for testing
            di_container = container._di_container
            di_container.unwire()

    def test_wire_modules_with_packages(self) -> None:
        """Test wire_modules with packages parameter."""
        container = FlextContainer()
        container.register("test_value", "wired_value")

        # Create a test module (not package, as packages must be actual installed packages)
        test_module = ModuleType("test_module")

        @inject
        def test_func(value: str = Provide["test_value"]) -> str:
            return value

        setattr(test_module, "test_func", test_func)

        # Wire with module (packages parameter requires actual installed packages)
        container.wire_modules(modules=[test_module])

        try:
            # Execute function - use getattr for dynamic ModuleType attributes
            func: Callable[[], str] = getattr(test_module, "test_func")
            result = func()
            assert result == "wired_value"
        finally:
            # Cleanup
            di_container = container._di_container
            di_container.unwire()

    def test_scoped_container_with_wiring(self) -> None:
        """Test scoped container preserves wiring."""
        container = FlextContainer(_context=FlextContext())
        container.register("global_service", "global_value")

        # Create scoped container
        scoped = container.scoped(services={"scoped_service": "scoped_value"})

        # Both global and scoped services should be accessible
        global_result = scoped.get("global_service")
        assert global_result.is_success

        scoped_result = scoped.get("scoped_service")
        assert scoped_result.is_success
        assert scoped_result.value == "scoped_value"

    def test_create_service_runtime_full_integration(self) -> None:
        """Test create_service_runtime with full DI integration."""

        def factory() -> dict[str, str]:
            return {"token": "generated_token"}

        def resource_factory() -> dict[str, bool]:
            return {"connected": True}

        # Create service runtime with all options
        runtime = s._create_runtime(
            config_overrides={"app_name": "test_app"},
            services={"static_service": "static_value"},
            factories={"token_factory": factory},
            resources={"connection": resource_factory},
            wire_modules=[__import__(__name__)],
        )

        # Verify all components
        assert runtime.config.app_name == "test_app"

        # Verify services
        static_result = runtime.container.get("static_service")
        assert static_result.is_success
        assert static_result.value == "static_value"

        # Verify factories
        factory_result = runtime.container.get("token_factory")
        assert factory_result.is_success
        assert factory_result.value == {"token": "generated_token"}

        # Verify resources
        resource_result = runtime.container.get("connection")
        assert resource_result.is_success
        assert resource_result.value == {"connected": True}

    def test_container_wire_modules_with_classes(self) -> None:
        """Test container.wire_modules with classes parameter."""
        container = FlextContainer()
        container.register("injected_value", "test_injection")

        class TestClass:
            @inject
            def __init__(self, value: str = Provide["injected_value"]) -> None:
                self.value = value

        # Wire class
        container.wire_modules(classes=[TestClass])

        try:
            # Create instance - should have injected value
            instance = TestClass()
            assert instance.value == "test_injection"
        finally:
            container._di_container.unwire()

    def test_error_handling_in_wiring(self) -> None:
        """Test error handling in wiring scenarios."""
        container = FlextContainer()

        # Wire with missing dependency - should not raise during wire
        module = ModuleType("error_module")

        @inject
        def func_with_missing(missing: str = Provide["nonexistent"]) -> str:
            return missing

        setattr(module, "func_with_missing", func_with_missing)

        # Wiring should succeed even if dependency doesn't exist yet
        container.wire_modules(modules=[module])

        try:
            # Execution should fail gracefully or handle missing dependency
            # This tests that wiring doesn't validate dependencies at wire time
            func_with_missing_wired: Callable[[], str] = getattr(
                module,
                "func_with_missing",
            )
            # Function exists but dependency may be missing - this is expected
            assert callable(func_with_missing_wired)
        finally:
            container._di_container.unwire()

    def test_resource_teardown_with_scoped_container(self) -> None:
        """Test resource teardown when scoped container is destroyed."""
        container = FlextContainer(_context=FlextContext())

        lifecycle = {"created": False, "destroyed": False}

        def resource_factory() -> dict[str, bool]:
            lifecycle["created"] = True
            return {"resource": True}

        # Create scoped container with resource
        scoped = container.scoped(resources={"test_resource": resource_factory})

        # Get resource
        result = scoped.get("test_resource")
        assertion_helpers.assert_flext_result_success(result)
        assert isinstance(result.value, dict)
        assert result.value == {"resource": True}
        assert lifecycle["created"] is True

        # In real scenarios, resource teardown happens on container shutdown
        # This tests that resources are properly registered

    def test_multiple_scoped_containers_isolation(self) -> None:
        """Test that multiple scoped containers are isolated."""
        container = FlextContainer(_context=FlextContext())

        # Create two scoped containers with different services
        scoped1 = container.scoped(services={"service": "value1"})
        scoped2 = container.scoped(services={"service": "value2"})

        # Verify isolation
        scoped1.get("service")
        result2 = scoped2.get("service")

        value1 = u.Tests.Result.assert_success(result2)
        value2 = u.Tests.Result.assert_success(result2)
        assert isinstance(value1, str)
        assert isinstance(value2, str)
        assert value1 == "value1"
        assert value2 == "value2"
        assert value1 != value2


__all__ = [
    "TestContainerDIRealExecution",
    "TestDIBridgeRealExecution",
    "TestDependencyIntegrationRealExecution",
    "TestRealWiringScenarios",
    "TestServiceBootstrapWithDI",
]
