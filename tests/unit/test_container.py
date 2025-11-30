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
from typing import ClassVar

import pytest

from flext_core import FlextContainer, FlextResult
from flext_core.constants import FlextConstants
from flext_tests.utilities import FlextTestsUtilities


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
    """Centralized container test scenarios using FlextConstants."""

    SERVICE_SCENARIOS: ClassVar[list[ServiceScenario]] = [
        ServiceScenario("test_service", {"key": "value"}, "Simple dict service"),
        ServiceScenario("service_instance", object(), "Generic object instance"),
        ServiceScenario("string_service", "test_value", "String service"),
    ]

    TYPED_RETRIEVAL_SCENARIOS: ClassVar[list[TypedRetrievalScenario]] = [
        TypedRetrievalScenario(
            "dict_service", {"key": "value"}, dict, True, "Dict service",
        ),
        TypedRetrievalScenario(
            "string_service", "test_string", str, True, "String service",
        ),
        TypedRetrievalScenario("list_service", [1, 2, 3], list, True, "List service"),
    ]

    CONFIG_SCENARIOS: ClassVar[list[dict[str, object]]] = [
        {"max_workers": 8, "timeout_seconds": 60.0},
        {"invalid_key": "value", "another_invalid": 42},
        {},
    ]


class ContainerTestHelpers:
    """Generalized helpers for container testing."""

    @staticmethod
    def create_clean_container() -> FlextContainer:
        """Create and clear container for testing."""
        container = FlextContainer()
        container.clear_all()
        return container

    @staticmethod
    def create_factory(return_value: object) -> Callable[[], object]:
        """Create a simple factory function."""
        return lambda: return_value

    @staticmethod
    def create_counting_factory() -> tuple[Callable[[], dict[str, str]], list[int]]:
        """Create factory that counts calls."""
        call_count = [0]

        def factory() -> dict[str, str]:
            call_count[0] += 1
            return {"call_count": str(call_count[0])}

        return factory, call_count


class TestFlextContainer:
    """Unified test suite for FlextContainer using FlextTestsUtilities."""

    def test_container_initialization(self) -> None:
        """Test container initialization creates valid instance."""
        container = FlextContainer()
        assert container is not None
        assert isinstance(container, FlextContainer)

    def test_container_singleton(self) -> None:
        """Test that FlextContainer returns singleton instance."""
        container1 = FlextContainer()
        container2 = FlextContainer()
        assert container1 is container2

    @pytest.mark.parametrize(
        "scenario", ContainerScenarios.SERVICE_SCENARIOS, ids=lambda s: s.name,
    )
    def test_register_service(self, scenario: ServiceScenario) -> None:
        """Test service registration with various types."""
        container = ContainerTestHelpers.create_clean_container()
        result = container.register(scenario.name, scenario.service)
        assert result.is_success
        assert result.value is True

    @pytest.mark.parametrize(
        "scenario", ContainerScenarios.SERVICE_SCENARIOS, ids=lambda s: s.name,
    )
    def test_with_service_fluent(self, scenario: ServiceScenario) -> None:
        """Test fluent interface for service registration."""
        container = ContainerTestHelpers.create_clean_container()
        result = container.with_service(scenario.name, scenario.service)
        assert result is container
        assert container.has_service(scenario.name)

    def test_register_duplicate_service(self) -> None:
        """Test that registering duplicate service name fails."""
        container = ContainerTestHelpers.create_clean_container()
        container.register("service1", "value1")
        result = container.register("service1", "value2")
        assert result.is_failure
        assert result.error is not None
        assert "already registered" in result.error

    def test_register_with_empty_name(self) -> None:
        """Test that empty name is rejected."""
        container = ContainerTestHelpers.create_clean_container()
        result = container.register("", "service")
        assert result.is_failure
        assert result.error is not None
        assert "at least 1 character" in result.error

    @pytest.mark.parametrize(
        "return_value",
        [{"created": "by_factory"}, "created_string"],
        ids=["dict", "string"],
    )
    def test_register_factory(self, return_value: object) -> None:
        """Test factory registration."""
        container = ContainerTestHelpers.create_clean_container()
        factory = ContainerTestHelpers.create_factory(return_value)
        result = container.register_factory(
            f"factory_{type(return_value).__name__}", factory,
        )
        assert result.is_success
        assert result.value is True

    @pytest.mark.parametrize(
        "return_value",
        [{"created": "by_factory"}, "created_string"],
        ids=["dict", "string"],
    )
    def test_with_factory_fluent(self, return_value: object) -> None:
        """Test fluent interface for factory."""
        container = ContainerTestHelpers.create_clean_container()
        factory = ContainerTestHelpers.create_factory(return_value)
        result = container.with_factory(
            f"factory_{type(return_value).__name__}", factory,
        )
        assert result is container

    def test_register_factory_non_callable(self) -> None:
        """Test that registering non-callable is rejected."""
        container = ContainerTestHelpers.create_clean_container()
        result = container.register_factory("invalid", "not_callable")
        assert result.is_failure
        assert result.error is not None
        assert "callable" in result.error

    def test_register_duplicate_factory(self) -> None:
        """Test that registering duplicate factory name fails."""
        container = ContainerTestHelpers.create_clean_container()
        factory1 = ContainerTestHelpers.create_factory("value1")
        container.register_factory("factory1", factory1)
        factory2 = ContainerTestHelpers.create_factory("value2")
        result = container.register_factory("factory1", factory2)
        assert result.is_failure
        assert result.error is not None
        assert "already registered" in result.error

    @pytest.mark.parametrize(
        "scenario", ContainerScenarios.SERVICE_SCENARIOS, ids=lambda s: s.name,
    )
    def test_get_service(self, scenario: ServiceScenario) -> None:
        """Test service retrieval."""
        container = ContainerTestHelpers.create_clean_container()
        container.register(scenario.name, scenario.service)
        result = container.get(scenario.name)
        FlextTestsUtilities.ResultHelpers.assert_success_with_value(
            result, scenario.service,
        )

    def test_get_nonexistent_service(self) -> None:
        """Test getting non-existent service."""
        container = ContainerTestHelpers.create_clean_container()
        result = container.get("nonexistent")
        FlextTestsUtilities.ResultHelpers.assert_failure_with_error(result, "not found")

    def test_get_factory_service(self) -> None:
        """Test retrieving service created by factory."""
        container = ContainerTestHelpers.create_clean_container()
        factory_result = {"created": "by_factory"}
        factory = ContainerTestHelpers.create_factory(factory_result)
        container.register_factory("factory_service", factory)
        result = container.get("factory_service")
        FlextTestsUtilities.ResultHelpers.assert_success_with_value(
            result, factory_result,
        )

    def test_get_factory_called_each_time(self) -> None:
        """Test that factory is called each time get() is invoked."""
        container = ContainerTestHelpers.create_clean_container()
        factory, call_count = ContainerTestHelpers.create_counting_factory()
        container.register_factory("factory_service", factory)
        result1 = container.get("factory_service")
        assert result1.is_success
        assert call_count[0] == 1
        result2 = container.get("factory_service")
        assert result2.is_success
        assert call_count[0] == 2

    @pytest.mark.parametrize(
        "scenario", ContainerScenarios.TYPED_RETRIEVAL_SCENARIOS, ids=lambda s: s.name,
    )
    def test_get_typed_correct(self, scenario: TypedRetrievalScenario) -> None:
        """Test typed retrieval with correct types."""
        container = ContainerTestHelpers.create_clean_container()
        container.register(scenario.name, scenario.service)
        typed_result: FlextResult[object] = container.get_typed(
            scenario.name, scenario.expected_type,
        )
        if scenario.should_pass:
            assert typed_result.is_success
            assert typed_result.value == scenario.service
        else:
            assert typed_result.is_failure

    def test_get_typed_wrong_type(self) -> None:
        """Test typed retrieval with wrong type fails."""
        container = ContainerTestHelpers.create_clean_container()
        container.register("string_service", "test_value")
        result = container.get_typed("string_service", dict)
        assert result.is_failure
        assert result.error is not None

    def test_get_typed_nonexistent(self) -> None:
        """Test typed retrieval of non-existent service."""
        container = ContainerTestHelpers.create_clean_container()
        result = container.get_typed("nonexistent", dict)
        assert result.is_failure
        assert result.error is not None
        assert "not found" in result.error

    @pytest.mark.parametrize(
        ("has_service", "expected"),
        [(True, True), (False, False)],
        ids=["exists", "not_exists"],
    )
    def test_has_service(self, has_service: bool, expected: bool) -> None:
        """Test has_service returns correct value."""
        container = ContainerTestHelpers.create_clean_container()
        if has_service:
            container.register("test_service", "value")
        assert (
            container.has_service("test_service" if has_service else "nonexistent")
            == expected
        )

    def test_has_service_factory(self) -> None:
        """Test has_service returns True for factories."""
        container = ContainerTestHelpers.create_clean_container()
        factory = ContainerTestHelpers.create_factory("value")
        container.register_factory("factory_service", factory)
        assert container.has_service("factory_service") is True

    def test_list_services_empty(self) -> None:
        """Test listing services when none registered."""
        container = ContainerTestHelpers.create_clean_container()
        services = container.list_services()
        assert len(services) == 0

    def test_list_services_mixed(self) -> None:
        """Test listing mix of registered services and factories."""
        container = ContainerTestHelpers.create_clean_container()
        container.register("service1", "value1")
        container.register("service2", "value2")
        factory = ContainerTestHelpers.create_factory("value3")
        container.register_factory("factory1", factory)
        services = container.list_services()
        assert len(services) == 3
        assert all(key in services for key in ["service1", "service2", "factory1"])

    @pytest.mark.parametrize(
        ("service_type", "use_factory"),
        [("service", False), ("factory", True)],
        ids=["service", "factory"],
    )
    def test_unregister(self, service_type: str, use_factory: bool) -> None:
        """Test unregistering services and factories."""
        container = ContainerTestHelpers.create_clean_container()
        name = "test_service"
        if use_factory:
            factory = ContainerTestHelpers.create_factory("value")
            container.register_factory(name, factory)
        else:
            container.register(name, "value")
        assert container.has_service(name)
        unregister_result = container.unregister(name)
        assert unregister_result.is_success
        assert not container.has_service(name)

    def test_unregister_nonexistent(self) -> None:
        """Test unregistering non-existent service fails."""
        container = ContainerTestHelpers.create_clean_container()
        unregister_result = container.unregister("nonexistent")
        assert unregister_result.is_failure
        assert unregister_result.error is not None
        assert "not found" in unregister_result.error

    @pytest.mark.parametrize("config", ContainerScenarios.CONFIG_SCENARIOS, ids=str)
    def test_configure_container(self, config: dict[str, object]) -> None:
        """Test container configuration."""
        container = FlextContainer()
        container.configure(config)
        assert container is not None

    def test_with_config_fluent(self) -> None:
        """Test fluent interface for configuration."""
        container = FlextContainer()
        config: dict[str, object] = {
            "max_workers": FlextConstants.Container.DEFAULT_WORKERS,
        }
        result = container.with_config(config)
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

    def test_clear_all_services(self) -> None:
        """Test clearing all services and factories."""
        container = ContainerTestHelpers.create_clean_container()
        container.register("service1", "value1")
        container.register("service2", "value2")
        factory = ContainerTestHelpers.create_factory("value3")
        container.register_factory("factory1", factory)
        assert len(container.list_services()) == 3
        container.clear_all()
        assert len(container.list_services()) == 0

    def test_clear_all_empty(self) -> None:
        """Test clearing when no services exist."""
        container = ContainerTestHelpers.create_clean_container()
        container.clear_all()
        assert len(container.list_services()) == 0

    def test_full_workflow(self) -> None:
        """Test complete container workflow."""
        container = ContainerTestHelpers.create_clean_container()
        container.register("db_connection", {"host": "localhost"})
        container.register("cache", {"type": "redis"})
        factory = ContainerTestHelpers.create_factory({"logger": "instance"})
        container.register_factory("logger", factory)
        assert all(
            container.has_service(k) for k in ["db_connection", "cache", "logger"]
        )
        for name in ["db_connection", "cache", "logger"]:
            result = container.get(name)
            assert result.is_success
        assert len(container.list_services()) == 3
        unregister_result = container.unregister("cache")
        assert unregister_result.is_success
        assert len(container.list_services()) == 2
        container.clear_all()
        assert len(container.list_services()) == 0

    def test_factory_exception_handling(self) -> None:
        """Test handling of factory exceptions."""
        container = ContainerTestHelpers.create_clean_container()
        error_msg = "Factory failed"

        def failing_factory() -> object:
            raise RuntimeError(error_msg)

        container.register_factory("failing", failing_factory)
        result = container.get("failing")
        assert result.is_failure


__all__ = ["TestFlextContainer"]
