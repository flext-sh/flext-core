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

from flext_core import FlextContainer, FlextResult, r, t
from flext_core.constants import c
from flext_tests import u


@dataclass(frozen=True, slots=True)
class ServiceScenario:
    """Test scenario for service registration and retrieval."""

    name: str
    service: object
    description: str = ""


@dataclass(frozen=True, slots=True)
class TypedRetrievalScenario:
    """Test scenario for typed service retrieval."""

    name: str
    service: object
    expected_type: type
    should_pass: bool
    description: str = ""


class ContainerScenarios:
    """Centralized container test scenarios using c."""

    SERVICE_SCENARIOS: ClassVar[list[ServiceScenario]] = [
        ServiceScenario("test_service", {"key": "value"}, "Simple dict service"),
        ServiceScenario("service_instance", object(), "Generic object instance"),
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

    CONFIG_SCENARIOS: ClassVar[list[dict[str, object]]] = [
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
        assert clean_container is not None
        assert isinstance(clean_container, FlextContainer)

    def test_container_singleton(self) -> None:
        """Test that FlextContainer returns singleton instance."""
        container1 = FlextContainer()
        container2 = FlextContainer()
        assert container1 is container2

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
        # Cast object to GeneralValueType | BaseModel for type compatibility
        # scenario.service is compatible at runtime but typed as object
        service_typed: (
            t.GeneralValueType | BaseModel | Callable[..., t.GeneralValueType]
        ) = cast(
            "t.GeneralValueType | BaseModel | Callable[..., t.GeneralValueType]",
            scenario.service,
        )
        result = container.with_service(scenario.name, service_typed)
        assert result is container
        assert container.has_service(scenario.name)

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
        return_value: object,
        clean_container: FlextContainer,
    ) -> None:
        """Test factory registration using fixtures."""
        factory = FlextTestsUtilities.Tests.ContainerHelpers.create_factory(
            return_value,
        )
        result = clean_container.register_factory(
            f"factory_{type(return_value).__name__}",
            factory,
        )
        u.Tests.Result.assert_success_with_value(result, True)

    @pytest.mark.parametrize(
        "return_value",
        [{"created": "by_factory"}, "created_string"],
        ids=["dict", "string"],
    )
    def test_with_factory_fluent(
        self,
        return_value: object,
        clean_container: FlextContainer,
    ) -> None:
        """Test fluent interface for factory using fixtures."""
        container = clean_container
        factory = FlextTestsUtilities.Tests.ContainerHelpers.create_factory(
            return_value,
        )
        result = container.with_factory(
            f"factory_{type(return_value).__name__}",
            factory,
        )
        assert result is container

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
        result: r[t.GeneralValueType] = clean_container.get(scenario.name)
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
        result: r[t.GeneralValueType] = clean_container.get("nonexistent")
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
        result: r[t.GeneralValueType] = clean_container.get("factory_service")
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
        result1: r[t.GeneralValueType] = clean_container.get("factory_service")
        FlextTestsUtilities.Tests.TestUtilities.assert_result_success(result1)
        assert get_count() == 1
        result2: r[t.GeneralValueType] = clean_container.get("factory_service")
        FlextTestsUtilities.Tests.TestUtilities.assert_result_success(result2)
        assert get_count() == 2

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
        # Runtime: object is compatible with GeneralValueType | BaseModel |
        # Callable | object
        service_typed: (
            t.GeneralValueType | BaseModel | Callable[..., t.GeneralValueType] | object
        ) = scenario.service
        container.register(scenario.name, service_typed)
        typed_result: FlextResult[object] = container.get_typed(
            scenario.name,
            scenario.expected_type,
        )
        if scenario.should_pass:
            FlextTestsUtilities.Tests.TestUtilities.assert_result_success(typed_result)
            assert typed_result.value == scenario.service
        else:
            FlextTestsUtilities.Tests.TestUtilities.assert_result_failure(typed_result)

    def test_get_typed_wrong_type(
        self,
        clean_container: FlextContainer,
    ) -> None:
        """Test typed retrieval with wrong type fails using fixtures."""
        clean_container.register("string_service", "test_value")
        result = clean_container.get_typed("string_service", dict)
        FlextTestsUtilities.Tests.TestUtilities.assert_result_failure(result)

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
        if has_service:
            container.register("test_service", "value")
        assert (
            container.has_service("test_service" if has_service else "nonexistent")
            == expected
        )

    def test_has_service_factory(
        self,
        clean_container: FlextContainer,
    ) -> None:
        """Test has_service returns True for factories using fixtures."""
        container = clean_container
        factory = FlextTestsUtilities.Tests.ContainerHelpers.create_factory("value")
        container.register_factory("factory_service", factory)
        assert container.has_service("factory_service") is True

    def test_list_services_empty(
        self,
        clean_container: FlextContainer,
    ) -> None:
        """Test listing services when none registered using fixtures."""
        container = clean_container
        services = container.list_services()
        assert len(services) == 0

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
        assert len(services) == 3
        assert all(key in services for key in ["service1", "service2", "factory1"])

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
        assert container.has_service(name)
        unregister_result = container.unregister(name)
        FlextTestsUtilities.Tests.TestUtilities.assert_result_success(unregister_result)
        assert not container.has_service(name)

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
    def test_configure_container(self, config: dict[str, object]) -> None:
        """Test container configuration."""
        container = FlextContainer()
        # Cast dict[str, object] to Mapping[str, GeneralValueType]
        # for type compatibility
        config_typed: t.Types.ConfigurationMapping = cast(
            "t.Types.ConfigurationMapping",
            config,
        )
        container.configure(config_typed)
        assert container is not None

    def test_with_config_fluent(self) -> None:
        """Test fluent interface for configuration."""
        container = FlextContainer()
        config: dict[str, object] = {
            "max_workers": c.Container.DEFAULT_WORKERS,
        }
        # Cast dict[str, object] to ConfigurationMapping for type compatibility
        config_typed: t.Types.ConfigurationMapping = cast(
            "t.Types.ConfigurationMapping",
            config,
        )
        result = container.with_config(config_typed)
        assert result is container

    def test_get_config(self) -> None:
        """Test retrieving current configuration."""
        container = FlextContainer()
        config = container.get_config()
        assert isinstance(config, dict)
        assert "enable_singleton" in config or "max_services" in config

    def test_config_property(self) -> None:
        """Test accessing config via property."""
        container = FlextContainer()
        config = container.config
        assert config is not None

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
        assert len(container.list_services()) == 3
        container.clear_all()
        assert len(container.list_services()) == 0

    def test_clear_all_empty(
        self,
        clean_container: FlextContainer,
    ) -> None:
        """Test clearing when no services exist using fixtures."""
        container = clean_container
        container.clear_all()
        assert len(container.list_services()) == 0

    def test_full_workflow(
        self,
        clean_container: FlextContainer,
    ) -> None:
        """Test complete container workflow using fixtures."""
        container = clean_container
        container.register("db_connection", {"host": "localhost"})
        container.register("cache", {"type": "redis"})
        factory = FlextTestsUtilities.Tests.ContainerHelpers.create_factory({
            "logger": "instance",
        })
        container.register_factory("logger", factory)
        assert all(
            container.has_service(k) for k in ["db_connection", "cache", "logger"]
        )
        for name in ["db_connection", "cache", "logger"]:
            result: r[t.GeneralValueType] = container.get(name)
            FlextTestsUtilities.Tests.TestUtilities.assert_result_success(result)
        assert len(container.list_services()) == 3
        unregister_result = container.unregister("cache")
        FlextTestsUtilities.Tests.TestUtilities.assert_result_success(unregister_result)
        assert len(container.list_services()) == 2
        container.clear_all()
        assert len(container.list_services()) == 0

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
        result: r[t.GeneralValueType] = container.get("failing")
        FlextTestsUtilities.Tests.TestUtilities.assert_result_failure(result)


__all__ = ["TestFlextContainer"]
