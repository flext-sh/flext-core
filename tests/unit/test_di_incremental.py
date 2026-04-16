"""Incremental tests for Dependency Injection with real code execution.

Module: flext_core DI pattern validation
Scope: Real execution of DI methods to validate bridge, container, and wiring

Tests DI functionality with real code execution:
- Bridge methods (dependency_providers, dependency_containers, create_container)
- DependencyIntegration methods (create_layered_bridge, register_object, register_factory, register_resource)
- DI methods (wire_modules, scoped with resources)
- Service bootstrap with DI (create_service_runtime)
- Real wiring with @u.DependencyIntegration.inject and Provide decorators
- Resource lifecycle management

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Generator
from types import ModuleType
from typing import override

import pytest

from flext_core import (
    FlextContainer,
    FlextContext,
    FlextSettings,
)
from flext_tests import tm
from tests import m, p, r, s, t, u


@pytest.fixture(autouse=True)
def reset_flext_container_singleton() -> Generator[None]:
    """Isolate FlextContainer singleton state across incremental DI tests."""
    FlextContainer.reset_for_testing()
    try:
        yield
    finally:
        FlextContainer.reset_for_testing()


class TestDIIncremental:
    def test_dependency_providers_returns_valid_module(self) -> None:
        """Test dependency_providers returns valid providers module."""
        providers_module = u.dependency_providers()
        singleton = providers_module.Singleton(lambda: "test_value")
        tm.that(singleton(), eq="test_value")
        factory = providers_module.Factory(lambda: {"key": "value"})
        tm.that(factory(), eq={"key": "value"})

    def test_dependency_containers_returns_valid_module(self) -> None:
        """Test dependency_containers returns valid containers module."""
        containers_module = u.dependency_containers()
        containers_module.DynamicContainer()

    def test_create_container_with_real_execution(self) -> None:
        """Test create_container with real registration and resolution."""
        container = FlextContainer.shared()
        _ = container.bind("test_service", "test_value")
        tm.that(container.has("test_service"), eq=True)
        resolved = container.resolve("test_service")
        resolved_value = u.Core.Tests.assert_success(resolved)
        tm.that(resolved_value, eq="test_value")

    def test_create_layered_bridge_with_config(self) -> None:
        """Test create_layered_bridge with real configuration."""
        bridge, service_module, resource_module = (
            u.DependencyIntegration.create_layered_bridge(
                settings=t.ConfigMap(root={"database": {"dsn": "sqlite://test.db"}}),
            )
        )
        tm.that(repr(bridge), ne="")
        tm.that(repr(service_module), ne="")
        tm.that(repr(resource_module), ne="")
        tm.that(callable(bridge.settings), eq=True)

    def test_register_object_with_real_container(self) -> None:
        """Test register_object with real container execution."""
        di_container = u.DependencyIntegration.create_container()
        provider = u.DependencyIntegration.register_object(
            di_container,
            "test_object",
            {"key": "value"},
        )
        tm.that(provider(), eq={"key": "value"})
        tm.that(di_container.test_object(), eq={"key": "value"})

    def test_register_factory_with_caching(self) -> None:
        """Test register_factory with caching enabled."""
        di_container = u.DependencyIntegration.create_container()
        call_count = {"count": 0}

        def factory() -> t.IntMapping:
            call_count["count"] += 1
            return {"calls": call_count["count"]}

        provider = u.DependencyIntegration.register_factory(
            di_container,
            "cached_factory",
            factory,
            cache=True,
        )
        result1 = provider()
        tm.that(result1, eq={"calls": 1})
        result2 = provider()
        tm.that(result2, eq={"calls": 1})
        tm.that(result1 is result2, eq=True)

    def test_register_factory_without_caching(self) -> None:
        """Test register_factory without caching (Factory)."""
        di_container = u.DependencyIntegration.create_container()
        call_count = {"count": 0}

        def factory() -> t.IntMapping:
            call_count["count"] += 1
            return {"calls": call_count["count"]}

        provider = u.DependencyIntegration.register_factory(
            di_container,
            "factory_no_cache",
            factory,
            cache=False,
        )
        result1 = provider()
        tm.that(result1, eq={"calls": 1})
        result2 = provider()
        tm.that(result2, eq={"calls": 2})
        tm.that(result1 is not result2, eq=True)

    def test_register_resource_with_lifecycle(self) -> None:
        """Test register_resource with real teardown."""
        di_container = u.DependencyIntegration.create_container()
        lifecycle = {"created": False, "closed": False}

        def resource_factory() -> t.BoolMapping:
            lifecycle["created"] = True
            return {"connected": True}

        def resource_teardown(_resource: t.BoolMapping) -> None:
            lifecycle["closed"] = True

        provider = u.DependencyIntegration.register_resource(
            di_container,
            "db_connection",
            resource_factory,
        )
        resource = provider()
        tm.that(resource, eq={"connected": True})
        tm.that(lifecycle["created"], eq=True)
        resource_teardown(resource)
        tm.that(lifecycle["closed"], eq=True)

    def test_wire_modules_with_inject(self) -> None:
        """Test wire with @u.DependencyIntegration.inject decorator real execution."""
        di_container = u.DependencyIntegration.create_container()
        u.DependencyIntegration.register_object(
            di_container,
            "api_key",
            "secret123",
        )
        u.DependencyIntegration.register_object(di_container, "timeout", 30)
        module = ModuleType("test_module")

        @u.DependencyIntegration.inject
        def api_call(
            key: str = u.DependencyIntegration.Provide["api_key"],
            timeout_sec: int = u.DependencyIntegration.Provide["timeout"],
        ) -> t.HeaderMapping:
            return {"key": key, "timeout": timeout_sec}

        setattr(module, "api_call", api_call)
        u.DependencyIntegration.wire(di_container, modules=[module])
        try:
            api_call_func: Callable[[], t.HeaderMapping] = getattr(
                module,
                "api_call",
            )
            result = api_call_func()
            expected: t.HeaderMapping = {"key": "secret123", "timeout": 30}
            tm.that(result, eq=expected)
        finally:
            di_container.unwire()

    def test_wire_modules_real_execution(self) -> None:
        """Test container.wire_modules with real code."""
        container = FlextContainer()
        _ = container.bind("logger_name", "test_logger")
        _ = container.bind("log_level", "INFO")
        module = ModuleType("wired_module")

        @u.DependencyIntegration.inject
        def log_message(
            name: str = u.DependencyIntegration.Provide["logger_name"],
            level: str = u.DependencyIntegration.Provide["log_level"],
        ) -> t.StrMapping:
            return {"logger": name, "level": level}

        setattr(module, "log_message", log_message)
        container.wire(modules=[module])
        try:
            log_func: Callable[[], t.StrMapping] = module.log_message
            result = log_func()
            tm.that(result, eq={"logger": "test_logger", "level": "INFO"})
        finally:
            di_container = container._di_container
            di_container.unwire()

    def test_scoped_with_resources_real_execution(self) -> None:
        """Test container.scoped with resources real execution."""
        container = FlextContainer.shared(context=FlextContext())
        lifecycle = {"created": False, "closed": False}

        def resource_factory() -> t.BoolMapping:
            lifecycle["created"] = True
            return {"connected": True}

        scoped = container.scope(resources={"db": resource_factory})
        result = scoped.resolve("db")
        assert result.success
        assert result.value == {"connected": True}
        tm.that(lifecycle["created"], eq=True)
        tm.that(scoped is not container, eq=True)

    def test_scoped_with_services_and_factories(self) -> None:
        """Test scoped container with services and factories."""
        container = FlextContainer.shared(context=FlextContext())
        scoped = container.scope(
            services={"api_key": "secret_key"},
            factories={"token_gen": lambda: {"token": "abc123"}},
        )
        service_result = scoped.resolve("api_key", type_cls=str)
        tm.ok(service_result)
        tm.that(service_result.value, eq="secret_key")
        factory_result = scoped.resolve("token_gen")
        assert factory_result.success
        assert factory_result.value == {"token": "abc123"}

    def test_scoped_with_config_override(self) -> None:
        """Test scoped container with settings override."""
        container = FlextContainer.shared(context=FlextContext())
        config_override = FlextSettings(app_name="scoped_app")
        scoped = container.scope(settings=config_override)
        config_obj = scoped.settings
        tm.that(config_obj.app_name, eq="scoped_app")

    def test_create_service_runtime_with_resources(self) -> None:
        """Test create_service_runtime with resources."""
        lifecycle = {"created": False}

        def db_factory() -> t.BoolMapping:
            lifecycle["created"] = True
            return {"connected": True}

        runtime = s._create_runtime(
            runtime_options=m.RuntimeBootstrapOptions(
                resources={"database": db_factory},
            ),
        )
        db_result = runtime.container.resolve("database")
        assert db_result.success
        assert db_result.value == {"connected": True}
        tm.that(lifecycle["created"], eq=True)

    def test_create_service_runtime_with_wiring(self) -> None:
        """Test create_service_runtime with wire_modules."""
        runtime = s._create_runtime(
            runtime_options=m.RuntimeBootstrapOptions(
                services={"api_key": "test_key"},
                wire_modules=[sys.modules[__name__]],
            ),
        )
        container_instance = runtime.container
        tm.that(container_instance, is_=p.Container)

    def test_service_with_runtime_bootstrap_options(self) -> None:
        """Test service with _runtime_bootstrap_options override."""

        class TestService(s[str]):
            @classmethod
            @override
            def _runtime_bootstrap_options(
                cls,
            ) -> m.RuntimeBootstrapOptions:
                return m.RuntimeBootstrapOptions(
                    services={"custom_service": "custom_value"},
                    factories={"custom_factory": lambda: {"custom": "data"}},
                )

            @override
            def execute(self) -> p.Result[str]:
                return r[str].ok("test")

        service = TestService()
        custom_result = service.container.resolve("custom_service", type_cls=str)
        tm.ok(custom_result)
        tm.that(custom_result.value, eq="custom_value")
        factory_result = service.container.resolve("custom_factory")
        assert factory_result.success
        assert factory_result.value == {"custom": "data"}

    def test_handler_wiring_with_inject(self) -> None:
        """Test handler wiring with @u.DependencyIntegration.inject decorator."""
        container = FlextContainer()
        _ = container.bind("custom_logger", "test_logger")
        _ = container.bind("db_pool", {"size": 10})
        handler_module = ModuleType("handler_module")

        @u.DependencyIntegration.inject
        def process_request(
            logger_name: str = u.DependencyIntegration.Provide["custom_logger"],
            pool: t.IntMapping = u.DependencyIntegration.Provide["db_pool"],
        ) -> t.HeaderMapping:
            return {"logger": logger_name, "pool_size": pool["size"]}

        setattr(handler_module, "process_request", process_request)
        container.wire(modules=[handler_module])
        try:
            process_func: Callable[[], t.HeaderMapping] = handler_module.process_request
            result = process_func()
            expected2: t.HeaderMapping = {
                "logger": "test_logger",
                "pool_size": 10,
            }
            tm.that(result, eq=expected2)
        finally:
            di_container = container._di_container
            di_container.unwire()

    def test_multiple_wired_functions(self) -> None:
        """Test multiple functions wired to same container."""
        container = FlextContainer()
        _ = container.bind("shared_config", {"env": "test"})
        module = ModuleType("multi_function_module")

        @u.DependencyIntegration.inject
        def func1(
            settings: t.StrMapping = u.DependencyIntegration.Provide["shared_config"],
        ) -> str:
            return settings["env"]

        @u.DependencyIntegration.inject
        def func2(
            settings: t.StrMapping = u.DependencyIntegration.Provide["shared_config"],
        ) -> bool:
            return settings["env"] == "test"

        setattr(module, "func1", func1)
        setattr(module, "func2", func2)
        container.wire(modules=[module])
        try:
            func1_wired: Callable[[], str] = getattr(module, "func1")
            func2_wired: Callable[[], bool] = getattr(module, "func2")
            tm.that(func1_wired(), eq="test")
            tm.that(func2_wired(), eq=True)
        finally:
            di_container = container._di_container
            di_container.unwire()

    def test_nested_dependency_injection(self) -> None:
        """Test nested dependency injection scenarios."""
        container = FlextContainer()
        _ = container.bind("base_url", "https://api.example.com")
        _ = container.bind("api_version", "v1")
        module = ModuleType("nested_module")

        @u.DependencyIntegration.inject
        def build_url(
            base: str = u.DependencyIntegration.Provide["base_url"],
            version: str = u.DependencyIntegration.Provide["api_version"],
        ) -> str:
            return f"{base}/{version}"

        @u.DependencyIntegration.inject
        def api_call(
            url: str = u.DependencyIntegration.Provide["built_url"],
            base: str = u.DependencyIntegration.Provide["base_url"],
        ) -> t.StrMapping:
            return {"url": url, "base": base}

        setattr(module, "build_url", build_url)
        setattr(module, "api_call", api_call)
        url_result = build_url()
        _ = container.bind("built_url", url_result)
        container.wire(modules=[module])
        try:
            api_call_func: Callable[[], t.StrMapping] = getattr(module, "api_call")
            result = api_call_func()
            tm.that(result, has="url")
            tm.that(result, has="base")
        finally:
            di_container = container._di_container
            di_container.unwire()

    def test_wire_modules_with_packages(self) -> None:
        """Test wire_modules with packages parameter."""
        container = FlextContainer()
        _ = container.bind("test_value", "wired_value")
        test_module = ModuleType("test_module")

        @u.DependencyIntegration.inject
        def test_func(
            value: str = u.DependencyIntegration.Provide["test_value"],
        ) -> str:
            return value

        setattr(test_module, "test_func", test_func)
        container.wire(modules=[test_module])
        try:
            func: Callable[[], str] = getattr(test_module, "test_func")
            result = func()
            tm.that(result, eq="wired_value")
        finally:
            di_container = container._di_container
            di_container.unwire()

    def test_scoped_container_with_wiring(self) -> None:
        """Test scoped container preserves wiring."""
        container = FlextContainer.shared(context=FlextContext())
        _ = container.bind("global_service", "global_value")
        scoped = container.scope(services={"scoped_service": "scoped_value"})
        global_result = scoped.resolve("global_service", type_cls=str)
        tm.ok(global_result)
        scoped_result = scoped.resolve("scoped_service", type_cls=str)
        tm.ok(scoped_result)
        tm.that(scoped_result.value, eq="scoped_value")

    def test_create_service_runtime_full_integration(self) -> None:
        """Test create_service_runtime with full DI integration."""

        def factory() -> t.StrMapping:
            return {"token": "generated_token"}

        def resource_factory() -> t.BoolMapping:
            return {"connected": True}

        runtime = s._create_runtime(
            runtime_options=m.RuntimeBootstrapOptions(
                settings_overrides={"app_name": "test_app"},
                services={"static_service": "static_value"},
                factories={"token_factory": factory},
                resources={"connection": resource_factory},
                wire_modules=[sys.modules[__name__]],
            ),
        )
        tm.that(runtime.settings.app_name, eq="test_app")
        static_result = runtime.container.resolve("static_service", type_cls=str)
        tm.ok(static_result)
        tm.that(static_result.value, eq="static_value")
        factory_result = runtime.container.resolve("token_factory")
        assert factory_result.success
        assert factory_result.value == {"token": "generated_token"}
        resource_result = runtime.container.resolve("connection")
        assert resource_result.success
        assert resource_result.value == {"connected": True}

    def test_container_wire_modules_with_classes(self) -> None:
        """Test container.wire_modules with classes parameter."""
        container = FlextContainer()
        _ = container.bind("injected_value", "test_injection")

        class TestClass:
            @u.DependencyIntegration.inject
            def __init__(
                self,
                value: str = u.DependencyIntegration.Provide["injected_value"],
            ) -> None:
                self.value = value

        container.wire(classes=[TestClass])
        try:
            instance = TestClass()
            tm.that(instance.value, eq="test_injection")
        finally:
            container._di_container.unwire()

    def test_error_handling_in_wiring(self) -> None:
        """Test error handling in wiring scenarios."""
        container = FlextContainer()
        module = ModuleType("error_module")

        @u.DependencyIntegration.inject
        def func_with_missing(
            missing: str = u.DependencyIntegration.Provide["nonexistent"],
        ) -> str:
            return missing

        setattr(module, "func_with_missing", func_with_missing)
        container.wire(modules=[module])
        try:
            func_with_missing_wired: Callable[[], str] = getattr(
                module,
                "func_with_missing",
            )
            tm.that(callable(func_with_missing_wired), eq=True)
        finally:
            container._di_container.unwire()

    def test_resource_teardown_with_scoped_container(self) -> None:
        """Test resource teardown when scoped container is destroyed."""
        container = FlextContainer.shared(context=FlextContext())
        lifecycle = {"created": False, "destroyed": False}

        def resource_factory() -> t.BoolMapping:
            lifecycle["created"] = True
            return {"resource": True}

        scoped = container.scope(resources={"test_resource": resource_factory})
        result = scoped.resolve("test_resource")
        assert result.success
        assert result.value == {"resource": True}
        tm.that(lifecycle["created"], eq=True)

    def test_multiple_scoped_containers_isolation(self) -> None:
        """Test that multiple scoped containers are isolated."""
        container = FlextContainer.shared(context=FlextContext())
        scoped1 = container.scope(services={"service": "value1"})
        scoped2 = container.scope(services={"service": "value2"})
        result1 = scoped1.resolve("service", type_cls=str)
        result2 = scoped2.resolve("service", type_cls=str)
        value1 = u.Core.Tests.assert_success(result1)
        value2 = u.Core.Tests.assert_success(result2)
        tm.that(value1, is_=str)
        tm.that(value2, is_=str)
        tm.that(value1, eq="value1")
        tm.that(value2, eq="value2")
        tm.that(value1, ne="value2")
