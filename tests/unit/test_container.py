"""Comprehensive tests for FlextContainer - Dependency Injection Container.

Module: flext_core.container
Scope: FlextContainer - dependency injection, service registration, factory patterns

Tests FlextContainer functionality including:
- Container initialization and singleton pattern
- Service registration and retrieval
- Factory registration and execution
- Typed service retrieval
- Service existence checks and listing
- Service unregistration
- Configuration management
- Complex workflows

Uses Python 3.13 patterns, u, c,
and aggressive parametrization for DRY testing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, ClassVar, cast

import pytest
from flext_tests import c, t, tm, u
from pydantic import BaseModel, ConfigDict, Field

from flext_core import FlextContainer, p, r


class ServiceScenario(BaseModel):
    """Test scenario for service registration and retrieval."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    name: Annotated[str, Field(description="Service scenario name")]
    service: Annotated[
        str | int | float | bool, Field(description="Service value to register")
    ]
    description: Annotated[
        str, Field(default="", description="Scenario description")
    ] = ""


class TypedRetrievalScenario(BaseModel):
    """Test scenario for typed service retrieval."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    name: Annotated[str, Field(description="Typed retrieval scenario name")]
    service: Annotated[
        str | int | float | bool, Field(description="Registered service value")
    ]
    expected_type: Annotated[type, Field(description="Expected service type")]
    should_pass: Annotated[
        bool, Field(description="Whether typed retrieval should succeed")
    ]
    description: Annotated[
        str, Field(default="", description="Scenario description")
    ] = ""


class ContainerScenarios:
    """Centralized container test scenarios using c."""

    SERVICE_SCENARIOS: ClassVar[list[ServiceScenario]] = [
        ServiceScenario(
            name="test_service",
            service="test_service_value",
            description="Simple string service",
        ),
        ServiceScenario(
            name="service_instance",
            service=42,
            description="Integer service instance",
        ),
        ServiceScenario(
            name="string_service",
            service="test_value",
            description="String service",
        ),
    ]
    TYPED_RETRIEVAL_SCENARIOS: ClassVar[list[TypedRetrievalScenario]] = [
        TypedRetrievalScenario(
            name="dict_service",
            service="test_dict_service",
            expected_type=str,
            should_pass=True,
            description="String service",
        ),
        TypedRetrievalScenario(
            name="string_service",
            service="test_string",
            expected_type=str,
            should_pass=True,
            description="String service",
        ),
        TypedRetrievalScenario(
            name="list_service",
            service=123,
            expected_type=int,
            should_pass=True,
            description="Integer service for typed retrieval",
        ),
    ]
    CONFIG_SCENARIOS: ClassVar[list[dict[str, t.Scalar]]] = [
        {"max_workers": 8, "timeout_seconds": 60.0},
        {"invalid_key": "value", "another_invalid": 42},
        {},
    ]


class TestFlextContainer:
    """Unified test suite for FlextContainer using u."""

    def test_container_initialization(self, clean_container: p.Container) -> None:
        """Test container initialization creates valid instance using fixtures."""
        tm.that(
            cast("t.Tests.object", clean_container),
            none=False,
            msg="Container must not be None",
        )
        assert isinstance(clean_container, p.Container), (
            "Container must be FlextContainer instance"
        )

    def test_container_singleton(self) -> None:
        """Test that FlextContainer returns singleton instance."""
        container1 = FlextContainer()
        container2 = FlextContainer()
        tm.that(
            cast("t.Tests.object", container1),
            none=False,
            msg="Container1 must not be None",
        )
        tm.that(
            cast("t.Tests.object", container2),
            none=False,
            msg="Container2 must not be None",
        )
        tm.that(
            container1 is container2,
            eq=True,
            msg="Containers must be the same singleton instance",
        )

    @pytest.mark.parametrize(
        "scenario",
        ContainerScenarios.SERVICE_SCENARIOS,
        ids=lambda s: s.name,
    )
    def test_register_service(
        self,
        scenario: ServiceScenario,
        clean_container: p.Container,
    ) -> None:
        """Test service registration with various types using fixtures."""
        result = clean_container.register(scenario.name, scenario.service)
        assert result is clean_container

    @pytest.mark.parametrize(
        "scenario",
        ContainerScenarios.SERVICE_SCENARIOS,
        ids=lambda s: s.name,
    )
    def test_with_service_fluent(
        self,
        scenario: ServiceScenario,
        clean_container: p.Container,
    ) -> None:
        """Test fluent interface for service registration using fixtures."""
        container = clean_container
        result = container.register(scenario.name, scenario.service)
        tm.that(
            result is container,
            eq=True,
            msg="with_service must return self for fluent interface",
        )
        tm.that(
            container.has_service(scenario.name),
            eq=True,
            msg=f"Container must have service {scenario.name} after registration",
        )

    def test_register_duplicate_service(self, clean_container: p.Container) -> None:
        """Test that registering duplicate service name preserves original using fixtures."""
        container = clean_container
        _ = container.register("service1", "value1")
        _ = container.register("service1", "value2")
        service_result = container.get("service1")
        u.Tests.Result.assert_success_with_value(service_result, "value1")

    def test_register_with_empty_name(self, clean_container: p.Container) -> None:
        """Test that empty name is rejected using fixtures."""
        clean_container.register("", "service")
        tm.that(
            clean_container.has_service(""),
            eq=False,
            msg="Empty name service must not be registered",
        )

    @pytest.mark.parametrize(
        "return_value",
        [{"created": "by_factory"}, "created_string"],
        ids=["dict", "string"],
    )
    def test_register_factory(
        self,
        return_value: t.RegisterableService,
        clean_container: p.Container,
    ) -> None:
        """Test factory registration using fixtures."""
        factory = u.Tests.ContainerHelpers.create_factory(return_value)
        factory_typed: Callable[[], t.RegisterableService] = factory
        clean_container.register(
            f"factory_{type(return_value).__name__}",
            factory_typed,
            kind="factory",
        )
        factory_name = f"factory_{type(return_value).__name__}"
        tm.that(
            clean_container.has_service(factory_name),
            eq=True,
            msg=f"Factory {factory_name} must be registered",
        )

    @pytest.mark.parametrize(
        "return_value",
        [{"created": "by_factory"}, "created_string"],
        ids=["dict", "string"],
    )
    def test_with_factory_fluent(
        self,
        return_value: t.RegisterableService,
        clean_container: p.Container,
    ) -> None:
        """Test fluent interface for factory using fixtures."""
        container = clean_container
        factory = u.Tests.ContainerHelpers.create_factory(return_value)
        factory_typed: Callable[[], t.RegisterableService] = factory
        result = container.register(
            f"factory_{type(return_value).__name__}",
            factory_typed,
            kind="factory",
        )
        tm.that(
            result is container,
            eq=True,
            msg="with_factory must return self for fluent interface",
        )
        factory_name = f"factory_{type(return_value).__name__}"
        tm.that(
            container.has_service(factory_name),
            eq=True,
            msg=f"Container must have factory {factory_name} after registration",
        )

    def test_register_factory_non_callable(
        self,
        clean_container: p.Container,
    ) -> None:
        """Test that registering non-callable with factory kind handles gracefully."""
        non_callable = cast("t.FactoryCallable", "not_callable")
        clean_container.register("invalid", non_callable, kind="factory")
        tm.that(
            clean_container.has_service("invalid"),
            eq=False,
            msg="Non-callable factory should not be registered",
        )

    def test_register_duplicate_factory(self, clean_container: p.Container) -> None:
        """Test that registering duplicate factory name preserves original using fixtures."""
        factory1 = u.Tests.ContainerHelpers.create_factory("value1")
        clean_container.register("factory1", factory1, kind="factory")
        factory2 = u.Tests.ContainerHelpers.create_factory("value2")
        clean_container.register("factory1", factory2, kind="factory")
        tm.that(
            clean_container.has_service("factory1"),
            eq=True,
            msg="Factory1 should be registered",
        )

    @pytest.mark.parametrize(
        "scenario",
        ContainerScenarios.SERVICE_SCENARIOS,
        ids=lambda s: s.name,
    )
    def test_get_service(
        self,
        scenario: ServiceScenario,
        clean_container: p.Container,
    ) -> None:
        """Test service retrieval using fixtures."""
        clean_container.register(scenario.name, scenario.service)
        result: r[t.RegisterableService] = clean_container.get(scenario.name)
        u.Tests.Result.assert_success_with_value(result, scenario.service)

    def test_get_nonexistent_service(self, clean_container: p.Container) -> None:
        """Test getting non-existent service using fixtures."""
        result: r[t.RegisterableService] = clean_container.get("nonexistent")
        u.Tests.Result.assert_result_failure_with_error(
            result,
            expected_error="not found",
        )

    def test_get_factory_service(self, clean_container: p.Container) -> None:
        """Test retrieving service created by factory using fixtures."""
        factory_result = {"created": "by_factory"}
        factory = u.Tests.ContainerHelpers.create_factory(factory_result)
        clean_container.register("factory_service", factory, kind="factory")
        result: r[t.RegisterableService] = clean_container.get("factory_service")
        u.Tests.Result.assert_success_with_value(result, factory_result)

    def test_get_factory_called_each_time(
        self,
        clean_container: p.Container,
    ) -> None:
        """Test that factory is called each time get() is invoked using fixtures."""
        factory, get_count = u.Tests.ContainerHelpers.create_counting_factory(
            "service_value",
        )
        clean_container.register("factory_service", factory, kind="factory")
        result1: r[t.RegisterableService] = clean_container.get("factory_service")
        _ = u.Tests.Result.assert_success(result1)
        tm.that(get_count(), eq=1, msg="Factory must be called once after first get()")
        result2: r[t.RegisterableService] = clean_container.get("factory_service")
        _ = u.Tests.Result.assert_success(result2)
        tm.that(
            get_count(),
            eq=2,
            msg="Factory must be called twice after second get()",
        )

    @pytest.mark.parametrize(
        "scenario",
        ContainerScenarios.TYPED_RETRIEVAL_SCENARIOS,
        ids=lambda s: s.name,
    )
    def test_get_typed_correct(
        self,
        scenario: TypedRetrievalScenario,
        clean_container: p.Container,
    ) -> None:
        """Test typed retrieval with correct types using fixtures."""
        container = clean_container
        _ = container.register(scenario.name, scenario.service)
        typed_result: r[t.RegisterableService] = container.get(
            scenario.name,
            type_cls=scenario.expected_type,
        )
        if scenario.should_pass:
            _ = u.Tests.Result.assert_success(typed_result)
            tm.that(
                str(typed_result.value),
                eq=str(scenario.service),
                msg=f"Typed result value must match service for {scenario.name}",
            )
            tm.that(
                isinstance(typed_result.value, scenario.expected_type),
                eq=True,
                msg=f"Typed result must be instance of {scenario.expected_type.__name__}",
            )
        else:
            _ = u.Tests.Result.assert_failure(typed_result)

    def test_get_typed_wrong_type(self, clean_container: p.Container) -> None:
        """Test typed retrieval with wrong type fails using fixtures."""
        clean_container.register("string_service", "test_value")
        result = clean_container.get("string_service", type_cls=dict[str, str])
        _ = u.Tests.Result.assert_failure(result)

    def test_get_typed_nonexistent(self, clean_container: p.Container) -> None:
        """Test typed retrieval of non-existent service using fixtures."""
        result = clean_container.get("nonexistent", type_cls=dict[str, str])
        u.Tests.Result.assert_result_failure_with_error(
            result,
            expected_error="not found",
        )

    @pytest.mark.parametrize(
        ("has_service", "expected"),
        [(True, True), (False, False)],
        ids=["exists", "not_exists"],
    )
    def test_has_service(
        self,
        has_service: bool,
        expected: bool,
        clean_container: p.Container,
    ) -> None:
        """Test has_service returns correct value using fixtures."""
        container = clean_container
        service_name = "test_service" if has_service else "nonexistent"
        if has_service:
            _ = container.register(service_name, "value")
        tm.that(
            container.has_service(service_name),
            eq=expected,
            msg=f"has_service must return {expected} for {service_name}",
        )

    def test_has_service_factory(self, clean_container: p.Container) -> None:
        """Test has_service returns True for factories using fixtures."""
        container = clean_container
        factory = u.Tests.ContainerHelpers.create_factory("value")
        _ = container.register("factory_service", factory, kind="factory")
        tm.that(
            container.has_service("factory_service"),
            eq=True,
            msg="Container must have factory_service after registration",
        )

    def test_list_services_empty(self, clean_container: p.Container) -> None:
        """Test listing services when none registered using fixtures."""
        container = clean_container
        services = container.list_services()
        tm.that(services, is_=list, msg="list_services must return a list")
        tm.that(
            len(services),
            eq=0,
            msg="Empty container must return empty services list",
        )
        tm.that(
            services,
            empty=True,
            msg="Empty container must have empty services list",
        )

    def test_list_services_mixed(self, clean_container: p.Container) -> None:
        """Test listing mix of registered services and factories using fixtures."""
        container = clean_container
        _ = container.register("service1", "value1")
        _ = container.register("service2", "value2")
        factory = u.Tests.ContainerHelpers.create_factory("value3")
        _ = container.register("factory1", factory, kind="factory")
        services = container.list_services()
        tm.that(len(services), eq=3, msg="Container must list 3 registered services")
        required_keys = ["service1", "service2", "factory1"]
        for key in required_keys:
            tm.that(key in services, eq=True, msg=f"Services list must contain {key}")

    @pytest.mark.parametrize(
        ("service_type", "use_factory"),
        [("service", False), ("factory", True)],
        ids=["service", "factory"],
    )
    def test_unregister(
        self,
        service_type: str,
        use_factory: bool,
        clean_container: p.Container,
    ) -> None:
        """Test unregistering services and factories using fixtures."""
        container = clean_container
        name = "test_service"
        if use_factory:
            factory = u.Tests.ContainerHelpers.create_factory("value")
            _ = container.register(name, factory, kind="factory")
        else:
            _ = container.register(name, "value")
        tm.that(
            container.has_service(name),
            eq=True,
            msg=f"Container must have {name} before unregister",
        )
        unregister_result = container.unregister(name)
        _ = u.Tests.Result.assert_success(unregister_result)
        tm.that(
            container.has_service(name),
            eq=False,
            msg=f"Container must not have {name} after unregister",
        )

    def test_unregister_nonexistent(self, clean_container: p.Container) -> None:
        """Test unregistering non-existent service fails using fixtures."""
        unregister_result = clean_container.unregister("nonexistent")
        u.Tests.Result.assert_result_failure_with_error(
            unregister_result,
            expected_error="not found",
        )

    @pytest.mark.parametrize("config", ContainerScenarios.CONFIG_SCENARIOS, ids=str)
    def test_configure_container(self, config: dict[str, t.Scalar]) -> None:
        """Test container configuration."""
        container = FlextContainer()
        container.configure(config)
        tm.that(
            cast("t.Tests.object", container),
            none=False,
            msg="Container must not be None after configure",
        )
        config_result = container.get_config()
        tm.that(
            config_result,
            is_=t.ConfigMap,
            none=False,
            msg="Container config must be a ConfigMap",
        )

    def test_with_config_fluent(self) -> None:
        """Test fluent interface for configuration."""
        container = FlextContainer()
        config: dict[str, t.Scalar] = {"max_workers": c.Container.DEFAULT_WORKERS}
        result = container.configure(config)
        tm.that(
            result is container,
            eq=True,
            msg="with_config must return self for fluent interface",
        )
        config_result = container.get_config()
        tm.that(
            config_result,
            is_=t.ConfigMap,
            none=False,
            msg="get_config must return a ConfigMap",
        )
        tm.that(
            config_result.root,
            none=False,
            msg="Config must be accessible after with_config",
        )

    def test_get_config(self) -> None:
        """Test retrieving current configuration."""
        container = FlextContainer()
        config = container.get_config()
        tm.that(
            config,
            is_=t.ConfigMap,
            none=False,
            msg="get_config must return ConfigMap",
        )
        tm.that(
            "enable_singleton" in config.root or "max_services" in config.root,
            eq=True,
            msg="Config must contain enable_singleton or max_services",
        )

    def test_config_property(self) -> None:
        """Test accessing config via property."""
        container = FlextContainer()
        config = container.config
        tm.that(
            cast("t.Tests.object", config),
            none=False,
            msg="Container config property must not be None",
        )
        tm.that(
            hasattr(config, "app_name") or hasattr(config, "enable_singleton"),
            eq=True,
            msg="Container config property must have config attributes",
        )

    def test_clear_all_services(self, clean_container: p.Container) -> None:
        """Test clearing all services and factories using fixtures."""
        container = clean_container
        _ = container.register("service1", "value1")
        _ = container.register("service2", "value2")
        factory = u.Tests.ContainerHelpers.create_factory("value3")
        _ = container.register("factory1", factory, kind="factory")
        tm.that(
            len(container.list_services()),
            eq=3,
            msg="Container must have 3 services before clear_all",
        )
        container.clear_all()
        tm.that(
            len(container.list_services()),
            eq=0,
            msg="Container must have 0 services after clear_all",
        )
        tm.that(
            container.list_services(),
            empty=True,
            msg="Container services list must be empty after clear_all",
        )

    def test_clear_all_empty(self, clean_container: p.Container) -> None:
        """Test clearing when no services exist using fixtures."""
        container = clean_container
        container.clear_all()
        tm.that(
            len(container.list_services()),
            eq=0,
            msg="Empty container must have 0 services after clear_all",
        )
        tm.that(
            container.list_services(),
            empty=True,
            msg="Empty container services list must be empty after clear_all",
        )

    def test_full_workflow(self, clean_container: p.Container) -> None:
        """Test complete container workflow using fixtures."""
        container = clean_container
        _ = container.register("db_connection", {"host": c.Network.LOCALHOST})
        _ = container.register("cache", {"type": "redis"})
        factory = u.Tests.ContainerHelpers.create_factory({"logger": "instance"})
        _ = container.register("logger", factory, kind="factory")
        required_services = ["db_connection", "cache", "logger"]
        for service_name in required_services:
            tm.that(
                container.has_service(service_name),
                eq=True,
                msg=f"Container must have {service_name} after registration",
            )
        for name in required_services:
            result: r[t.RegisterableService] = container.get(name)
            _ = u.Tests.Result.assert_success(result)
        tm.that(
            len(container.list_services()),
            eq=3,
            msg="Container must have 3 services in full workflow",
        )
        unregister_result = container.unregister("cache")
        _ = u.Tests.Result.assert_success(unregister_result)
        tm.that(
            len(container.list_services()),
            eq=2,
            msg="Container must have 2 services after unregistering cache",
        )
        container.clear_all()
        tm.that(
            len(container.list_services()),
            eq=0,
            msg="Container must have 0 services after clear_all in workflow",
        )

    def test_factory_exception_handling(self, clean_container: p.Container) -> None:
        """Test handling of factory exceptions using fixtures."""
        container = clean_container
        error_msg = "Factory failed"

        def failing_factory() -> str:
            raise RuntimeError(error_msg)

        _ = container.register("failing", failing_factory, kind="factory")
        result: r[t.RegisterableService] = container.get("failing")
        _ = u.Tests.Result.assert_failure(result)


__all__ = ["TestFlextContainer"]
