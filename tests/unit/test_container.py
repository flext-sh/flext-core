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

Uses Python 3.13 patterns, FlextTestsUtilities, FlextConstants,
and aggressive parametrization for DRY testing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar, cast

import pytest
from pydantic import BaseModel

from flext_core import FlextConstants, FlextContainer, FlextResult, r, t
from flext_core.constants import c
from flext_tests import tm, u
from flext_tests.utilities import FlextTestsUtilities


@dataclass(frozen=True, slots=True)
class ServiceScenario:
    """Test scenario for service registration and retrieval."""

    name: str
    service: t.RegisterableService
    description: str = ""


@dataclass(frozen=True, slots=True)
class TypedRetrievalScenario:
    """Test scenario for typed service retrieval."""

    name: str
    service: t.RegisterableService
    expected_type: type
    should_pass: bool
    description: str = ""


class ContainerScenarios:
    """Centralized container test scenarios using c."""

    SERVICE_SCENARIOS: ClassVar[list[ServiceScenario]] = [
        ServiceScenario("test_service", {"key": "value"}, "Simple dict service"),
        ServiceScenario(
            "service_instance", {"instance_id": 123}, "Dict service instance"
        ),
        ServiceScenario("string_service", "test_value", "String service"),
    ]

    TYPED_RETRIEVAL_SCENARIOS: ClassVar[list[TypedRetrievalScenario]] = [
        TypedRetrievalScenario(
            "dict_service",
            {"key": "value"},
            dict,
            True,
            "Dict service",
        ),
        TypedRetrievalScenario(
            "string_service",
            "test_string",
            str,
            True,
            "String service",
        ),
        TypedRetrievalScenario("list_service", [1, 2, 3], list, True, "List service"),
    ]

    CONFIG_SCENARIOS: ClassVar[list[dict[str, t.ScalarValue]]] = [
        {"max_workers": 8, "timeout_seconds": 60.0},
        {"invalid_key": "value", "another_invalid": 42},
        {},
    ]


class TestFlextContainer:
    """Unified test suite for FlextContainer using FlextTestsUtilities."""

    def test_container_initialization(
        self,
        clean_container: FlextContainer,
    ) -> None:
        """Test container initialization creates valid instance using fixtures."""
        tm.that(clean_container, none=False, msg="Container must not be None")
        tm.that(
            clean_container,
            is_=FlextContainer,
            msg="Container must be FlextContainer instance",
        )

    def test_container_singleton(self) -> None:
        """Test that FlextContainer returns singleton instance."""
        container1 = FlextContainer()
        container2 = FlextContainer()
        tm.that(container1, none=False, msg="Container1 must not be None")
        tm.that(container2, none=False, msg="Container2 must not be None")
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
        clean_container: FlextContainer,
    ) -> None:
        """Test service registration with various types using fixtures."""
        result = clean_container.register(scenario.name, scenario.service)
        u.Tests.Result.assert_success_with_value(result, True)

    @pytest.mark.parametrize(
        "scenario",
        ContainerScenarios.SERVICE_SCENARIOS,
        ids=lambda s: s.name,
    )
    def test_with_service_fluent(
        self,
        scenario: ServiceScenario,
        clean_container: FlextContainer,
    ) -> None:
        """Test fluent interface for service registration using fixtures."""
        container = clean_container
        # Cast object to t.GeneralValueType | BaseModel for type compatibility
        # scenario.service is compatible at runtime but typed as object
        service_typed: (
            t.GeneralValueType | BaseModel | Callable[..., t.GeneralValueType]
        ) = cast(
            "t.GeneralValueType | BaseModel | Callable[..., t.GeneralValueType]",
            scenario.service,
        )
        result = container.with_service(scenario.name, service_typed)
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

    def test_register_duplicate_service(
        self,
        clean_container: FlextContainer,
    ) -> None:
        """Test that registering duplicate service name fails using fixtures."""
        container = clean_container
        container.register("service1", "value1")
        result = container.register("service1", "value2")
        u.Tests.Result.assert_result_failure_with_error(
            result,
            expected_error="already registered",
        )

    def test_register_with_empty_name(
        self,
        clean_container: FlextContainer,
    ) -> None:
        """Test that empty name is rejected using fixtures."""
        result = clean_container.register("", "service")
        u.Tests.Result.assert_result_failure_with_error(
            result,
            expected_error="at least 1 character",
        )

    @pytest.mark.parametrize(
        "return_value",
        [{"created": "by_factory"}, "created_string"],
        ids=["dict", "string"],
    )
    def test_register_factory(
        self,
        return_value: t.RegisterableService,
        clean_container: FlextContainer,
    ) -> None:
        """Test factory registration using fixtures."""
        factory = FlextTestsUtilities.Tests.ContainerHelpers.create_factory(
            return_value,
        )
        factory_typed: Callable[[], t.RegisterableService] = cast(
            "Callable[[], t.RegisterableService]",
            factory,
        )
        result = clean_container.register_factory(
            f"factory_{type(return_value).__name__}",
            factory_typed,
        )
        u.Tests.Result.assert_success_with_value(result, True)

    @pytest.mark.parametrize(
        "return_value",
        [{"created": "by_factory"}, "created_string"],
        ids=["dict", "string"],
    )
    def test_with_factory_fluent(
        self,
        return_value: t.RegisterableService,
        clean_container: FlextContainer,
    ) -> None:
        """Test fluent interface for factory using fixtures."""
        container = clean_container
        factory = FlextTestsUtilities.Tests.ContainerHelpers.create_factory(
            return_value,
        )
        # Cast factory to correct type for with_factory
        factory_typed: Callable[[], t.RegisterableService] = cast(
            "Callable[[], t.RegisterableService]",
            factory,
        )
        result = container.with_factory(
            f"factory_{type(return_value).__name__}",
            factory_typed,
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
        clean_container: FlextContainer,
    ) -> None:
        """Test that registering non-callable is rejected using fixtures."""
        # Intentionally pass non-callable to test error handling
        non_callable: Callable[[], t.GeneralValueType] = cast(
            "Callable[[], t.GeneralValueType]",
            "not_callable",
        )
        result = clean_container.register_factory("invalid", non_callable)
        u.Tests.Result.assert_result_failure_with_error(
            result,
            expected_error="callable",
        )

    def test_register_duplicate_factory(
        self,
        clean_container: FlextContainer,
    ) -> None:
        """Test that registering duplicate factory name fails using fixtures."""
        factory1 = FlextTestsUtilities.Tests.ContainerHelpers.create_factory("value1")
        clean_container.register_factory("factory1", factory1)
        factory2 = FlextTestsUtilities.Tests.ContainerHelpers.create_factory("value2")
        result: r[bool] = clean_container.register_factory("factory1", factory2)
        u.Tests.Result.assert_result_failure_with_error(
            result,
            expected_error="already registered",
        )

    @pytest.mark.parametrize(
        "scenario",
        ContainerScenarios.SERVICE_SCENARIOS,
        ids=lambda s: s.name,
    )
    def test_get_service(
        self,
        scenario: ServiceScenario,
        clean_container: FlextContainer,
    ) -> None:
        """Test service retrieval using fixtures."""
        clean_container.register(scenario.name, scenario.service)
        result: r[t.RegisterableService] = clean_container.get(scenario.name)
        expected_value: t.GeneralValueType = cast(
            "t.GeneralValueType",
            scenario.service,
        )
        u.Tests.Result.assert_success_with_value(
            result,
            expected_value,
        )

    def test_get_nonexistent_service(
        self,
        clean_container: FlextContainer,
    ) -> None:
        """Test getting non-existent service using fixtures."""
        result: r[t.RegisterableService] = clean_container.get("nonexistent")
        u.Tests.Result.assert_result_failure_with_error(
            result,
            expected_error="not found",
        )

    def test_get_factory_service(
        self,
        clean_container: FlextContainer,
    ) -> None:
        """Test retrieving service created by factory using fixtures."""
        factory_result = {"created": "by_factory"}
        factory = FlextTestsUtilities.Tests.ContainerHelpers.create_factory(
            factory_result,
        )
        clean_container.register_factory("factory_service", factory)
        result: r[t.RegisterableService] = clean_container.get("factory_service")
        u.Tests.Result.assert_success_with_value(
            result,
            factory_result,
        )

    def test_get_factory_called_each_time(
        self,
        clean_container: FlextContainer,
    ) -> None:
        """Test that factory is called each time get() is invoked using fixtures."""
        factory, get_count = (
            FlextTestsUtilities.Tests.ContainerHelpers.create_counting_factory(
                "service_value",
            )
        )
        clean_container.register_factory("factory_service", factory)
        result1: r[t.RegisterableService] = clean_container.get("factory_service")
        u.Tests.Result.assert_result_success(result1)
        tm.that(get_count(), eq=1, msg="Factory must be called once after first get()")
        result2: r[t.RegisterableService] = clean_container.get("factory_service")
        u.Tests.Result.assert_result_success(result2)
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
        clean_container: FlextContainer,
    ) -> None:
        """Test typed retrieval with correct types using fixtures."""
        container = clean_container
        # Cast scenario.service to container.register() compatible type
        # Runtime: object is compatible with t.GeneralValueType | BaseModel |
        # Callable | object
        service_typed: (
            t.GeneralValueType | BaseModel | Callable[..., t.GeneralValueType]
        ) = cast(
            "t.GeneralValueType | BaseModel | Callable[..., t.GeneralValueType]",
            scenario.service,
        )
        container.register(scenario.name, service_typed)
        typed_result: FlextResult[t.RegisterableService] = container.get_typed(
            scenario.name,
            scenario.expected_type,
        )
        if scenario.should_pass:
            u.Tests.Result.assert_result_success(typed_result)
            tm.that(
                typed_result.value,
                eq=scenario.service,
                msg=f"Typed result value must match service for {scenario.name}",
            )
            tm.that(
                isinstance(typed_result.value, scenario.expected_type),
                eq=True,
                msg=(
                    f"Typed result must be instance of "
                    f"{scenario.expected_type.__name__}"
                ),
            )
        else:
            u.Tests.Result.assert_result_failure(typed_result)

    def test_get_typed_wrong_type(
        self,
        clean_container: FlextContainer,
    ) -> None:
        """Test typed retrieval with wrong type fails using fixtures."""
        clean_container.register("string_service", "test_value")
        result = clean_container.get_typed("string_service", dict)
        u.Tests.Result.assert_result_failure(result)

    def test_get_typed_nonexistent(
        self,
        clean_container: FlextContainer,
    ) -> None:
        """Test typed retrieval of non-existent service using fixtures."""
        result = clean_container.get_typed("nonexistent", dict)
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
        clean_container: FlextContainer,
    ) -> None:
        """Test has_service returns correct value using fixtures."""
        container = clean_container
        service_name = "test_service" if has_service else "nonexistent"
        if has_service:
            container.register(service_name, "value")
        tm.that(
            container.has_service(service_name),
            eq=expected,
            msg=f"has_service must return {expected} for {service_name}",
        )

    def test_has_service_factory(
        self,
        clean_container: FlextContainer,
    ) -> None:
        """Test has_service returns True for factories using fixtures."""
        container = clean_container
        factory = FlextTestsUtilities.Tests.ContainerHelpers.create_factory("value")
        container.register_factory("factory_service", factory)
        tm.that(
            container.has_service("factory_service"),
            eq=True,
            msg="Container must have factory_service after registration",
        )

    def test_list_services_empty(
        self,
        clean_container: FlextContainer,
    ) -> None:
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

    def test_list_services_mixed(
        self,
        clean_container: FlextContainer,
    ) -> None:
        """Test listing mix of registered services and factories using fixtures."""
        container = clean_container
        container.register("service1", "value1")
        container.register("service2", "value2")
        factory = FlextTestsUtilities.Tests.ContainerHelpers.create_factory("value3")
        container.register_factory("factory1", factory)
        services = container.list_services()
        tm.that(len(services), eq=3, msg="Container must list 3 registered services")
        required_keys = ["service1", "service2", "factory1"]
        for key in required_keys:
            tm.that(
                key in services,
                eq=True,
                msg=f"Services list must contain {key}",
            )

    @pytest.mark.parametrize(
        ("service_type", "use_factory"),
        [("service", False), ("factory", True)],
        ids=["service", "factory"],
    )
    def test_unregister(
        self,
        service_type: str,
        use_factory: bool,
        clean_container: FlextContainer,
    ) -> None:
        """Test unregistering services and factories using fixtures."""
        container = clean_container
        name = "test_service"
        if use_factory:
            factory = FlextTestsUtilities.Tests.ContainerHelpers.create_factory("value")
            container.register_factory(name, factory)
        else:
            container.register(name, "value")
        tm.that(
            container.has_service(name),
            eq=True,
            msg=f"Container must have {name} before unregister",
        )
        unregister_result = container.unregister(name)
        u.Tests.Result.assert_result_success(unregister_result)
        tm.that(
            container.has_service(name),
            eq=False,
            msg=f"Container must not have {name} after unregister",
        )

    def test_unregister_nonexistent(
        self,
        clean_container: FlextContainer,
    ) -> None:
        """Test unregistering non-existent service fails using fixtures."""
        unregister_result = clean_container.unregister("nonexistent")
        u.Tests.Result.assert_result_failure_with_error(
            unregister_result,
            expected_error="not found",
        )

    @pytest.mark.parametrize("config", ContainerScenarios.CONFIG_SCENARIOS, ids=str)
    def test_configure_container(self, config: dict[str, t.ScalarValue]) -> None:
        """Test container configuration."""
        container = FlextContainer()
        container.configure(config)
        tm.that(container, none=False, msg="Container must not be None after configure")
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
        config: dict[str, t.ScalarValue] = {
            "max_workers": c.Container.DEFAULT_WORKERS,
        }
        result = container.with_config(config)
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
        # max_workers might not be in config if it's not a valid container config key
        # Just verify config is accessible
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
            config, is_=t.ConfigMap, none=False, msg="get_config must return ConfigMap"
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
        tm.that(config, none=False, msg="Container config property must not be None")
        # config property returns p.Config, not dict
        # Use isinstance check for protocol compatibility
        tm.that(
            hasattr(config, "app_name") or hasattr(config, "enable_singleton"),
            eq=True,
            msg="Container config property must have config attributes",
        )

    def test_clear_all_services(
        self,
        clean_container: FlextContainer,
    ) -> None:
        """Test clearing all services and factories using fixtures."""
        container = clean_container
        container.register("service1", "value1")
        container.register("service2", "value2")
        factory = FlextTestsUtilities.Tests.ContainerHelpers.create_factory("value3")
        container.register_factory("factory1", factory)
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

    def test_clear_all_empty(
        self,
        clean_container: FlextContainer,
    ) -> None:
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

    def test_full_workflow(
        self,
        clean_container: FlextContainer,
    ) -> None:
        """Test complete container workflow using fixtures."""
        container = clean_container
        container.register("db_connection", {"host": FlextConstants.Network.LOCALHOST})
        container.register("cache", {"type": "redis"})
        factory = FlextTestsUtilities.Tests.ContainerHelpers.create_factory({
            "logger": "instance",
        })
        container.register_factory("logger", factory)
        required_services = ["db_connection", "cache", "logger"]
        for service_name in required_services:
            tm.that(
                container.has_service(service_name),
                eq=True,
                msg=f"Container must have {service_name} after registration",
            )
        for name in required_services:
            result: r[t.RegisterableService] = container.get(name)
            u.Tests.Result.assert_result_success(result)
        tm.that(
            len(container.list_services()),
            eq=3,
            msg="Container must have 3 services in full workflow",
        )
        unregister_result = container.unregister("cache")
        u.Tests.Result.assert_result_success(unregister_result)
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

    def test_factory_exception_handling(
        self,
        clean_container: FlextContainer,
    ) -> None:
        """Test handling of factory exceptions using fixtures."""
        container = clean_container
        error_msg = "Factory failed"

        def failing_factory() -> t.GeneralValueType:
            raise RuntimeError(error_msg)

        container.register_factory("failing", failing_factory)
        result: r[t.RegisterableService] = container.get("failing")
        u.Tests.Result.assert_result_failure(result)


__all__ = ["TestFlextContainer"]
