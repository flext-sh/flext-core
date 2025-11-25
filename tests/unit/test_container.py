"""Comprehensive tests for FlextContainer - Dependency Injection Container.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import pytest

from flext_core import FlextContainer, FlextResult


class OperationType(StrEnum):
    """Operation types for container testing."""

    SERVICE = "service"
    FACTORY = "factory"


@dataclass(frozen=True, slots=True)
class ServiceScenario:
    """Test scenario for service registration and retrieval."""

    name: str
    service: object
    op_type: OperationType = OperationType.SERVICE
    description: str = ""


@dataclass(frozen=True, slots=True)
class TypedRetrievalScenario:
    """Test scenario for typed service retrieval."""

    name: str
    service: object
    expected_type: type
    should_pass: bool
    description: str = ""


@dataclass(frozen=True, slots=True)
class ConfigurationScenario:
    """Test scenario for container configuration."""

    config: dict[str, object]
    should_succeed: bool
    description: str = ""


class TestFlextContainer:
    """Unified test suite for FlextContainer dependency injection.

    Uses parametrization, factories, and modern Python 3.13 patterns
    to consolidate comprehensive testing without duplication.
    """

    # ========== Initialization Tests ==========

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

    # ========== Service Registration Tests ==========

    @pytest.mark.parametrize(
        "scenario",
        [
            ServiceScenario(
                name="test_service",
                service={"key": "value"},
                op_type=OperationType.SERVICE,
                description="Simple dict service",
            ),
            ServiceScenario(
                name="service_instance",
                service=object(),
                op_type=OperationType.SERVICE,
                description="Generic object instance",
            ),
            ServiceScenario(
                name="string_service",
                service="test_value",
                op_type=OperationType.SERVICE,
                description="String service",
            ),
        ],
    )
    def test_register_service(self, scenario: ServiceScenario) -> None:
        """Test service registration with various types."""
        container = FlextContainer()
        container.clear_all()

        result = container.register(scenario.name, scenario.service)
        assert result.is_success
        assert result.value is True

    @pytest.mark.parametrize(
        "scenario",
        [
            ServiceScenario(
                name="test_service",
                service={"key": "value"},
                op_type=OperationType.SERVICE,
                description="Simple dict service",
            ),
            ServiceScenario(
                name="service_instance",
                service=object(),
                op_type=OperationType.SERVICE,
                description="Generic object instance",
            ),
            ServiceScenario(
                name="string_service",
                service="test_value",
                op_type=OperationType.SERVICE,
                description="String service",
            ),
        ],
    )
    def test_with_service_fluent(self, scenario: ServiceScenario) -> None:
        """Test fluent interface for service registration."""
        container = FlextContainer()
        container.clear_all()

        result = container.with_service(scenario.name, scenario.service)
        assert result is container  # Fluent interface returns self
        assert container.has_service(scenario.name)

    def test_register_duplicate_service(self) -> None:
        """Test that registering duplicate service name fails."""
        container = FlextContainer()
        container.clear_all()

        result1 = container.register("service1", "value1")
        assert result1.is_success

        result2 = container.register("service1", "value2")
        assert result2.is_failure
        assert result2.error is not None
        assert "already registered" in result2.error

    def test_register_with_empty_name(self) -> None:
        """Test that empty name is rejected by Pydantic validation."""
        container = FlextContainer()
        container.clear_all()

        # Pydantic validates that name must have at least 1 character
        result = container.register("", "service")
        assert result.is_failure
        assert result.error is not None
        assert "at least 1 character" in result.error

    # ========== Factory Registration Tests ==========

    def test_register_factory_dict(self) -> None:
        """Test factory registration returning dict."""
        container = FlextContainer()
        container.clear_all()

        def factory_dict() -> object:
            return {"created": "by_factory"}

        result = container.register_factory("factory_dict", factory_dict)
        assert result.is_success
        assert result.value is True

    def test_register_factory_string(self) -> None:
        """Test factory registration returning string."""
        container = FlextContainer()
        container.clear_all()

        def factory_string() -> object:
            return "created_string"

        result = container.register_factory("factory_string", factory_string)
        assert result.is_success
        assert result.value is True

    def test_with_factory_fluent_dict(self) -> None:
        """Test fluent interface for factory returning dict."""
        container = FlextContainer()
        container.clear_all()

        def factory_dict() -> object:
            return {"created": "by_factory"}

        result = container.with_factory("factory_dict", factory_dict)
        assert result is container  # Fluent interface returns self
        assert container.has_service("factory_dict")

    def test_with_factory_fluent_string(self) -> None:
        """Test fluent interface for factory returning string."""
        container = FlextContainer()
        container.clear_all()

        def factory_string() -> object:
            return "created_string"

        result = container.with_factory("factory_string", factory_string)
        assert result is container  # Fluent interface returns self
        assert container.has_service("factory_string")

    def test_register_factory_non_callable(self) -> None:
        """Test that registering non-callable is rejected by Pydantic validation."""
        container = FlextContainer()
        container.clear_all()

        # Pydantic validates that factory input must be callable
        invalid_input: object = "not_callable"
        result = container.register_factory("invalid", invalid_input)  # type: ignore[arg-type]
        # Registration fails (Pydantic validation)
        assert result.is_failure
        assert result.error is not None
        assert "callable" in result.error

    def test_register_duplicate_factory(self) -> None:
        """Test that registering duplicate factory name fails."""
        container = FlextContainer()
        container.clear_all()

        def factory1() -> object:
            return "value1"

        result1 = container.register_factory("factory1", factory1)
        assert result1.is_success

        def factory2() -> object:
            return "value2"

        result2 = container.register_factory("factory1", factory2)
        assert result2.is_failure
        assert result2.error is not None
        assert "already registered" in result2.error

    # ========== Service Retrieval Tests ==========

    @pytest.mark.parametrize(
        "scenario",
        [
            ServiceScenario(
                name="test_service",
                service={"key": "value"},
                op_type=OperationType.SERVICE,
                description="Simple dict service",
            ),
            ServiceScenario(
                name="service_instance",
                service=object(),
                op_type=OperationType.SERVICE,
                description="Generic object instance",
            ),
            ServiceScenario(
                name="string_service",
                service="test_value",
                op_type=OperationType.SERVICE,
                description="String service",
            ),
        ],
    )
    def test_get_service(self, scenario: ServiceScenario) -> None:
        """Test service retrieval."""
        container = FlextContainer()
        container.clear_all()

        container.register(scenario.name, scenario.service)
        result = container.get(scenario.name)

        assert result.is_success
        assert result.value == scenario.service

    def test_get_nonexistent_service(self) -> None:
        """Test getting non-existent service."""
        container = FlextContainer()
        container.clear_all()

        result = container.get("nonexistent")
        assert result.is_failure
        assert result.error is not None
        assert "not found" in result.error

    def test_get_factory_service(self) -> None:
        """Test retrieving service created by factory."""
        container = FlextContainer()
        container.clear_all()

        factory_result = {"created": "by_factory"}

        def factory() -> object:
            return factory_result

        container.register_factory("factory_service", factory)
        result = container.get("factory_service")

        assert result.is_success
        assert result.value == factory_result

    def test_get_factory_called_each_time(self) -> None:
        """Test that factory is called each time get() is invoked (no caching)."""
        container = FlextContainer()
        container.clear_all()

        call_count = 0

        def counting_factory() -> dict[str, str]:
            nonlocal call_count
            call_count += 1
            return {"call_count": str(call_count)}

        container.register_factory("factory_service", counting_factory)

        result1 = container.get("factory_service")
        assert result1.is_success
        value1: object = result1.value
        assert isinstance(value1, dict)
        assert call_count == 1

        result2 = container.get("factory_service")
        assert result2.is_success
        value2: object = result2.value
        assert isinstance(value2, dict)
        # Current implementation calls factory each time (no caching)
        assert call_count == 2  # Factory called twice

    # ========== Typed Service Retrieval Tests ==========

    @pytest.mark.parametrize(
        "scenario",
        [
            TypedRetrievalScenario(
                name="dict_service",
                service={"key": "value"},
                expected_type=dict,
                should_pass=True,
                description="Dict service with correct type",
            ),
            TypedRetrievalScenario(
                name="string_service",
                service="test_string",
                expected_type=str,
                should_pass=True,
                description="String service with correct type",
            ),
            TypedRetrievalScenario(
                name="list_service",
                service=[1, 2, 3],
                expected_type=list,
                should_pass=True,
                description="List service with correct type",
            ),
        ],
    )
    def test_get_typed_correct(self, scenario: TypedRetrievalScenario) -> None:
        """Test typed retrieval with correct types."""
        container = FlextContainer()
        container.clear_all()

        container.register(scenario.name, scenario.service)
        result: FlextResult[object] = container.get_typed(
            scenario.name, scenario.expected_type
        )

        assert result.is_success
        assert result.value == scenario.service

    def test_get_typed_wrong_type(self) -> None:
        """Test typed retrieval with wrong type fails."""
        container = FlextContainer()
        container.clear_all()

        container.register("string_service", "test_value")

        result: FlextResult[object] = container.get_typed("string_service", dict)
        assert result.is_failure
        assert result.error is not None
        assert "type mismatch" in result.error or "not of type" in result.error

    def test_get_typed_nonexistent(self) -> None:
        """Test typed retrieval of non-existent service."""
        container = FlextContainer()
        container.clear_all()

        result: FlextResult[object] = container.get_typed("nonexistent", dict)
        assert result.is_failure
        assert result.error is not None
        assert "not found" in result.error

    # ========== Service Existence Tests ==========

    def test_has_service_true(self) -> None:
        """Test has_service returns True for registered services."""
        container = FlextContainer()
        container.clear_all()

        container.register("test_service", "value")
        assert container.has_service("test_service") is True

    def test_has_service_false(self) -> None:
        """Test has_service returns False for non-existent services."""
        container = FlextContainer()
        container.clear_all()

        assert container.has_service("nonexistent") is False

    def test_has_service_factory(self) -> None:
        """Test has_service returns True for factories."""
        container = FlextContainer()
        container.clear_all()

        def factory() -> object:
            return "value"

        container.register_factory("factory_service", factory)
        assert container.has_service("factory_service") is True

    # ========== Service Listing Tests ==========

    def test_list_services_empty(self) -> None:
        """Test listing services when none registered."""
        container = FlextContainer()
        container.clear_all()

        services = container.list_services()
        assert len(services) == 0

    def test_list_services_mixed(self) -> None:
        """Test listing mix of registered services and factories."""
        container = FlextContainer()
        container.clear_all()

        container.register("service1", "value1")
        container.register("service2", "value2")

        def factory() -> object:
            return "value3"

        container.register_factory("factory1", factory)

        services = container.list_services()
        assert len(services) == 3
        assert "service1" in services
        assert "service2" in services
        assert "factory1" in services

    # ========== Service Unregistration Tests ==========

    def test_unregister_service(self) -> None:
        """Test unregistering a service."""
        container = FlextContainer()
        container.clear_all()

        container.register("test_service", "value")
        assert container.has_service("test_service")

        result = container.unregister("test_service")
        assert result.is_success
        assert not container.has_service("test_service")

    def test_unregister_factory(self) -> None:
        """Test unregistering a factory."""
        container = FlextContainer()
        container.clear_all()

        def factory() -> object:
            return "value"

        container.register_factory("factory_service", factory)
        assert container.has_service("factory_service")

        result = container.unregister("factory_service")
        assert result.is_success
        assert not container.has_service("factory_service")

    def test_unregister_nonexistent(self) -> None:
        """Test unregistering non-existent service fails."""
        container = FlextContainer()
        container.clear_all()

        result = container.unregister("nonexistent")
        assert result.is_failure
        assert result.error is not None
        assert "not found" in result.error

    # ========== Configuration Tests ==========

    @pytest.mark.parametrize(
        "scenario",
        [
            ConfigurationScenario(
                config={"max_workers": 8, "timeout_seconds": 60.0},
                should_succeed=True,
                description="Valid configuration",
            ),
            ConfigurationScenario(
                config={"invalid_key": "value", "another_invalid": 42},
                should_succeed=True,
                description="Invalid keys (ignored gracefully)",
            ),
            ConfigurationScenario(
                config={},
                should_succeed=True,
                description="Empty configuration",
            ),
        ],
    )
    def test_configure_container(self, scenario: ConfigurationScenario) -> None:
        """Test container configuration."""
        container = FlextContainer()

        # configure() returns None (void), just call it and verify no exception
        container.configure(scenario.config)
        # Test passes if no exception was raised

    def test_with_config_fluent(self) -> None:
        """Test fluent interface for configuration."""
        container = FlextContainer()

        config: dict[str, object] = {"max_workers": 8}
        result = container.with_config(config)
        assert result is container  # Fluent interface returns self

    def test_get_config(self) -> None:
        """Test retrieving current configuration."""
        container = FlextContainer()

        config = container.get_config()
        assert isinstance(config, dict)
        assert "enable_singleton" in config
        assert "max_services" in config

    def test_config_property(self) -> None:
        """Test accessing config via property."""
        container = FlextContainer()

        config = container.config
        assert config is not None
        # config property returns FlextConfig instance

    # ========== Clear Tests ==========

    def test_clear_all_services(self) -> None:
        """Test clearing all services and factories."""
        container = FlextContainer()
        container.clear_all()

        container.register("service1", "value1")
        container.register("service2", "value2")

        def factory() -> object:
            return "value3"

        container.register_factory("factory1", factory)

        assert len(container.list_services()) == 3

        container.clear_all()
        assert len(container.list_services()) == 0

    def test_clear_all_empty(self) -> None:
        """Test clearing when no services exist."""
        container = FlextContainer()
        container.clear_all()

        # Should not raise exception
        container.clear_all()
        assert len(container.list_services()) == 0

    # ========== Complex Scenarios ==========

    def test_full_workflow(self) -> None:
        """Test complete container workflow."""
        container = FlextContainer()
        container.clear_all()

        # Register multiple services
        container.register("db_connection", {"host": "localhost"})
        container.register("cache", {"type": "redis"})

        # Register factories
        def factory() -> object:
            return {"logger": "instance"}

        container.register_factory("logger", factory)

        # Verify all exist
        assert container.has_service("db_connection")
        assert container.has_service("cache")
        assert container.has_service("logger")

        # Retrieve and verify
        db_result = container.get("db_connection")
        assert db_result.is_success

        cache_result = container.get("cache")
        assert cache_result.is_success

        logger_result = container.get("logger")
        assert logger_result.is_success

        # List all
        all_services = container.list_services()
        assert len(all_services) == 3

        # Unregister one
        result = container.unregister("cache")
        assert result.is_success
        assert len(container.list_services()) == 2

        # Clear all
        container.clear_all()
        assert len(container.list_services()) == 0

    def test_factory_exception_handling(self) -> None:
        """Test handling of factory exceptions."""
        container = FlextContainer()
        container.clear_all()

        def failing_factory() -> object:
            msg = "Factory failed"
            raise RuntimeError(msg)

        container.register_factory("failing", failing_factory)

        result = container.get("failing")
        assert result.is_failure
        assert result.error is not None


__all__ = ["TestFlextContainer"]
