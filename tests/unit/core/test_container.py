"""Core Dependency Injection Test Suite - Unit Testing Layer Container Foundation.

Comprehensive unit test suite for FlextContainer enterprise dependency injection system
that validates type-safe service management across the entire FLEXT ecosystem.

Module Role in Architecture:
    Testing Layer → Unit Tests → Dependency Injection Container Validation

    This module provides comprehensive unit testing that ensures:
    - Service registration and retrieval work reliably across 32 projects
    - Type-safe dependency injection without runtime failures
    - Container configuration and factory patterns function correctly
    - Service lifecycle management maintains proper state

Testing Strategy Coverage:
    ✅ Service Registration: Basic and factory-based service registration patterns
    ✅ Service Retrieval: Type-safe service resolution and error handling
    ✅ Container Configuration: Global container setup and management
    ✅ Factory Patterns: Dynamic service creation with dependency injection
    ✅ Error Handling: Comprehensive failure scenario validation
    ✅ Type Safety: Generic type parameter validation throughout

Enterprise Quality Standards:
    - Test Coverage: 95%+ coverage of container functionality
    - Performance: < 100ms per test, < 10s total suite execution
    - Isolation: Pure unit tests with mock services
    - Type Safety: Comprehensive validation of generic type parameters

Real-World Usage Validation:
    # Enterprise service container setup
    container = get_flext_container()

    # Service registration with type safety
    register_result = container.register("database_service", DatabaseService())

    # Type-safe service retrieval
    service_result = container.get("database_service")
    if service_result.success:
        database = service_result.data

Test Architecture Patterns:
    - Isolated Component Testing: Each container method tested independently
    - Mock Service Usage: SampleService classes for dependency simulation
    - Error Path Validation: Comprehensive failure scenario coverage
    - Factory Pattern Testing: Dynamic service creation validation

See Also:
    - src/flext_core/container.py: FlextContainer implementation
    - src/flext_core/result.py: FlextResult pattern used in container
    - examples/02_flext_container_dependency_injection.py: Usage examples
    - tests/integration/: Cross-module integration tests

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from flext_core.container import (
    FlextContainer,
    FlextServiceFactory,
    configure_flext_container,
    get_flext_container,
)
from flext_core.exceptions import FlextError

if TYPE_CHECKING:
    from flext_core.result import FlextResult


class SampleService:
    """Mock service class for testing dependency injection."""

    def __init__(
        self,
        name: str,
        config: dict[str, object] | None = None,
    ) -> None:
        """Initialize test service with name and optional config."""
        self.name = name
        self.config = config or {}
        self.initialized = True

    def process(self, data: str) -> str:
        """Mock processing method."""
        return f"Processed {data} by {self.name}"


@pytest.fixture
def sample_services() -> dict[str, object]:
    """Provide sample services for testing."""
    return {
        "database": SampleService("DatabaseService"),
        "logger": SampleService("LoggerService"),
        "cache": SampleService("CacheService", {"ttl": 3600}),
    }


@pytest.fixture
def service_factories() -> dict[str, FlextServiceFactory]:
    """Provide service factory functions for testing."""

    def create_database() -> SampleService:
        return SampleService("DatabaseService", {"host": "localhost"})

    def create_logger() -> SampleService:
        return SampleService("LoggerService", {"level": "INFO"})

    def create_cache() -> SampleService:
        return SampleService("CacheService", {"size": 1000})

    return {
        "database": create_database,
        "logger": create_logger,
        "cache": create_cache,
    }


@pytest.fixture
def failing_factory() -> FlextServiceFactory:
    """Provide a factory that raises an exception."""

    def create_failing_service() -> SampleService:
        msg = "Intentional test failure"
        raise FlextError(msg)

    return create_failing_service


@pytest.mark.unit
class TestFlextContainerBasicOperations:
    """Unit tests for basic FlextContainer operations."""

    def test_container_initialization(
        self,
        clean_container: FlextContainer,
    ) -> None:
        """Test FlextContainer initializes with empty state."""
        if len(clean_container.list_services()) != 0:
            raise AssertionError(
                f"Expected {0}, got {len(clean_container.list_services())}"
            )
        assert clean_container.get_service_count() == 0
        if clean_container.get_service_names() != []:
            raise AssertionError(
                f"Expected {[]}, got {clean_container.get_service_names()}"
            )

    def test_service_registration_success(
        self,
        clean_container: FlextContainer,
        sample_services: dict[str, SampleService],
    ) -> None:
        """Test successful service registration."""
        service = sample_services["database"]
        result = clean_container.register("database", service)

        assert result.success
        assert result.data is None
        assert clean_container.has("database")
        if "database" not in clean_container.list_services():
            raise AssertionError(
                f"Expected {'database'} in {clean_container.list_services()}"
            )

    @pytest.mark.parametrize(
        "service_name",
        [
            "database",
            "logger",
            "cache",
            "user_service",
            "api_client",
        ],
    )
    def test_service_registration_various_names(
        self,
        clean_container: FlextContainer,
        service_name: str,
    ) -> None:
        """Test service registration with various valid names."""
        service = SampleService(service_name)
        result = clean_container.register(service_name, service)

        assert result.success
        assert clean_container.has(service_name)

    @pytest.mark.parametrize(
        ("invalid_name", "expected_error"),
        [
            ("", "Service name cannot be empty"),
            ("   ", "Service name cannot be empty"),
            (None, "Service name cannot be empty"),
            (123, "Service name cannot be empty"),
            ([], "Service name cannot be empty"),
        ],
    )
    def test_service_registration_invalid_names(
        self,
        clean_container: FlextContainer,
        invalid_name: object,
        expected_error: str,
    ) -> None:
        """Test service registration with invalid names."""
        service = SampleService("test")
        # The container handles all invalid names through FlextResult
        result = clean_container.register(invalid_name, service)  # type: ignore[arg-type] # Intentional invalid name test

        assert result.is_failure
        assert result.error is not None
        if expected_error not in result.error:
            raise AssertionError(f"Expected expected_error in {result.error}")

    def test_service_retrieval_success(
        self,
        clean_container: FlextContainer,
        sample_services: dict[str, SampleService],
    ) -> None:
        """Test successful service retrieval."""
        service = sample_services["database"]
        clean_container.register("database", service)

        result = clean_container.get("database")

        assert result.success
        assert result.data is service
        assert isinstance(result.data, SampleService)
        if result.data.name != "DatabaseService":
            raise AssertionError(
                f"Expected {'DatabaseService'}, got {result.data.name}"
            )

    def test_service_retrieval_not_found(
        self,
        clean_container: FlextContainer,
    ) -> None:
        """Test service retrieval for non-existent service."""
        result = clean_container.get("nonexistent")

        assert result.is_failure
        assert result.error is not None
        assert result.error
        if "not found" not in (result.error or ""):
            raise AssertionError(f"Expected 'not found' in {result.error}")

    @pytest.mark.parametrize(
        "service_names",
        [
            ["database"],
            ["database", "logger"],
            ["database", "logger", "cache"],
            ["service1", "service2", "service3", "service4"],
        ],
    )
    def test_multiple_service_registration(
        self,
        clean_container: FlextContainer,
        service_names: list[str],
    ) -> None:
        """Test registering multiple services."""
        # Register all services
        for name in service_names:
            service = SampleService(name)
            result = clean_container.register(name, service)
            assert result.success

        # Verify all services are registered
        registered_services = clean_container.list_services()
        if len(registered_services) != len(service_names):
            raise AssertionError(
                f"Expected {len(service_names)}, got {len(registered_services)}"
            )

        for name in service_names:
            if name not in registered_services:
                raise AssertionError(f"Expected {name} in {registered_services}")
            assert clean_container.has(name)

    def test_service_overwriting(
        self,
        clean_container: FlextContainer,
    ) -> None:
        """Test service overwriting with new instance."""
        # Register initial service
        service1 = SampleService("Original")
        result1 = clean_container.register("service", service1)
        assert result1.success

        # Verify initial service
        get_result1 = clean_container.get("service")
        assert get_result1.success
        assert get_result1.data is service1

        # Register replacement service
        service2 = SampleService("Replacement")
        result2 = clean_container.register("service", service2)
        assert result2.success

        # Verify replacement service
        get_result2 = clean_container.get("service")
        assert get_result2.success
        assert get_result2.data is service2
        assert get_result2.data is not service1


@pytest.mark.unit
class TestFlextContainerSingletonPattern:
    """Unit tests for singleton factory pattern."""

    def test_singleton_factory_registration(
        self,
        clean_container: FlextContainer,
        service_factories: dict[str, FlextServiceFactory],
    ) -> None:
        """Test successful singleton factory registration."""
        factory = service_factories["database"]
        result = clean_container.register_factory("database", factory)

        assert result.success
        assert result.data is None
        assert clean_container.has("database")

    def test_singleton_factory_creation(
        self,
        clean_container: FlextContainer,
        service_factories: dict[str, FlextServiceFactory],
    ) -> None:
        """Test singleton factory creates instance on first access."""
        factory = service_factories["database"]
        clean_container.register_factory("database", factory)

        # First access creates instance
        result1 = clean_container.get("database")
        assert result1.success
        assert isinstance(result1.data, SampleService)
        if result1.data.name != "DatabaseService":
            raise AssertionError(
                f"Expected {'DatabaseService'}, got {result1.data.name}"
            )

    def test_singleton_factory_caching(
        self,
        clean_container: FlextContainer,
    ) -> None:
        """Test singleton factory caches instances."""
        call_count = 0

        def create_service() -> SampleService:
            nonlocal call_count
            call_count += 1
            return SampleService(f"Service_{call_count}")

        clean_container.register_factory("service", create_service)

        # Multiple accesses should return same instance
        result1 = clean_container.get("service")
        result2 = clean_container.get("service")
        result3 = clean_container.get("service")

        assert result1.success
        assert result2.success
        assert result3.success

        # Same instance returned
        assert result1.data is result2.data
        assert result2.data is result3.data

        # Factory called only once
        if call_count != 1:
            raise AssertionError(f"Expected {1}, got {call_count}")
        assert isinstance(result1.data, SampleService)
        if result1.data.name != "Service_1":
            raise AssertionError(f"Expected {'Service_1'}, got {result1.data.name}")

    def test_singleton_factory_failure_handling(
        self,
        clean_container: FlextContainer,
        failing_factory: FlextServiceFactory,
    ) -> None:
        """Test singleton factory failure handling."""
        result = clean_container.register_factory("failing", failing_factory)
        assert result.success

        # Factory failure should be handled gracefully
        get_result = clean_container.get("failing")
        assert get_result.is_failure
        assert get_result.error is not None
        if "Factory for 'failing' failed" not in get_result.error:
            raise AssertionError(
                f"Expected {"Factory for 'failing' failed"} in {get_result.error}"
            )
        assert "Intentional test failure" in get_result.error

    def test_non_callable_factory_rejection(
        self,
        clean_container: FlextContainer,
    ) -> None:
        """Test rejection of non-callable factory."""
        # This would be caught by type system, but test runtime behavior
        # We need to bypass type checking for this test case
        factory = "not_callable"
        result = clean_container.register_factory("service", factory)  # type: ignore[arg-type] # Intentional invalid factory test

        assert result.is_failure
        assert result.error is not None
        assert result.error
        if "must be callable" not in (result.error or ""):
            raise AssertionError(f"Expected 'must be callable' in {result.error}")

    @pytest.mark.parametrize("factory_count", [1, 3, 5, 10])
    def test_multiple_singleton_factories(
        self,
        clean_container: FlextContainer,
        factory_count: int,
    ) -> None:
        """Test multiple singleton factories work independently."""
        factories = {}

        # Register multiple factories
        for i in range(factory_count):
            service_name = f"service_{i}"

            def create_service(name: str = service_name) -> SampleService:
                return SampleService(name)

            factories[service_name] = create_service
            result = clean_container.register_factory(
                service_name,
                create_service,
            )
            assert result.success

        # Verify all services work independently
        for i in range(factory_count):
            service_name = f"service_{i}"
            get_result: FlextResult[object] = clean_container.get(service_name)
            assert get_result.success
            assert isinstance(get_result.data, SampleService)
            service_instance = get_result.data
            if service_instance.name != service_name:
                raise AssertionError(
                    f"Expected {service_name}, got {service_instance.name}"
                )


@pytest.mark.unit
class TestFlextContainerServiceManagement:
    """Unit tests for service lifecycle management."""

    def test_service_existence_check(
        self,
        clean_container: FlextContainer,
        sample_services: dict[str, SampleService],
    ) -> None:
        """Test service existence checking."""
        # Non-existent service
        assert not clean_container.has("nonexistent")

        # Register and check
        service = sample_services["database"]
        clean_container.register("database", service)
        assert clean_container.has("database")

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("valid_service", False),  # Not registered
            ("", False),  # Invalid name
            ("   ", False),  # Whitespace name
            (None, False),  # None name
            (123, False),  # Non-string name
        ],
    )
    def test_service_existence_edge_cases(
        self,
        clean_container: FlextContainer,
        name: object,
        *,
        expected: bool,
    ) -> None:
        """Test service existence checking with edge cases."""
        result = clean_container.has(str(name))
        assert result is expected

    def test_service_removal_success(
        self,
        clean_container: FlextContainer,
        sample_services: dict[str, SampleService],
    ) -> None:
        """Test successful service removal."""
        service = sample_services["database"]
        clean_container.register("database", service)

        # Verify service exists
        assert clean_container.has("database")

        # Remove service
        result = clean_container.unregister("database")
        assert result.success
        assert result.data is None

        # Verify service is gone
        assert not clean_container.has("database")
        if "database" in clean_container.list_services():
            raise AssertionError(
                f"Expected 'database' to be removed from services, but found it in {clean_container.list_services()}"
            )

    def test_service_removal_not_found(
        self,
        clean_container: FlextContainer,
    ) -> None:
        """Test removal of non-existent service."""
        result = clean_container.unregister("nonexistent")

        assert result.is_failure
        assert result.error is not None
        assert result.error
        if "not found" not in (result.error or ""):
            raise AssertionError(f"Expected 'not found' in {result.error}")

    def test_singleton_removal_clears_cache(
        self,
        clean_container: FlextContainer,
        service_factories: dict[str, FlextServiceFactory],
    ) -> None:
        """Test singleton removal clears both registry and cache."""
        factory = service_factories["database"]
        clean_container.register_factory("database", factory)

        # Create instance (populates cache)
        result1 = clean_container.get("database")
        assert result1.success

        # Remove service
        remove_result = clean_container.unregister("database")
        assert remove_result.success

        # Verify complete removal
        assert not clean_container.has("database")
        if clean_container.get_service_count() != 0:
            raise AssertionError(
                f"Expected {0}, got {clean_container.get_service_count()}"
            )

    def test_container_clear_operation(
        self,
        clean_container: FlextContainer,
        sample_services: dict[str, SampleService],
    ) -> None:
        """Test clearing all services from container."""
        # Register multiple services
        for name, service in sample_services.items():
            clean_container.register(name, service)

        if len(clean_container.list_services()) != len(sample_services):
            raise AssertionError(
                f"Expected {len(sample_services)}, got {len(clean_container.list_services())}"
            )

        # Clear container
        result = clean_container.clear()
        assert result.success
        assert result.data is None

        # Verify empty state
        if len(clean_container.list_services()) != 0:
            raise AssertionError(
                f"Expected {0}, got {len(clean_container.list_services())}"
            )
        assert clean_container.get_service_count() == 0
        if clean_container.get_service_names() != []:
            raise AssertionError(
                f"Expected {[]}, got {clean_container.get_service_names()}"
            )

    def test_service_listing_accuracy(
        self,
        clean_container: FlextContainer,
        sample_services: dict[str, SampleService],
    ) -> None:
        """Test service listing returns accurate results."""
        # Initially empty
        services = clean_container.get_service_names()
        if services != []:
            raise AssertionError(f"Expected {[]}, got {services}")

        # Add services incrementally
        expected_services = []
        for name, service in sample_services.items():
            clean_container.register(name, service)
            expected_services.append(name)

            current_services = clean_container.get_service_names()
            if len(current_services) != len(expected_services):
                raise AssertionError(
                    f"Expected {len(expected_services)}, got {len(current_services)}"
                )
            assert set(current_services) == set(expected_services)


@pytest.mark.unit
class TestFlextContainerGlobalManagement:
    """Unit tests for global container management."""

    def test_get_global_container_singleton(self) -> None:
        """Test global container returns same instance."""
        container1 = get_flext_container()
        container2 = get_flext_container()

        assert container1 is container2
        assert isinstance(container1, FlextContainer)

    def test_configure_global_container_with_instance(
        self,
        sample_services: dict[str, SampleService],
    ) -> None:
        """Test configuring global container with custom instance."""
        # Create pre-configured container
        custom_container = FlextContainer()
        service = sample_services["database"]
        custom_container.register("database", service)

        # Configure as global
        result = configure_flext_container(custom_container)

        assert result is custom_container
        assert get_flext_container() is custom_container
        assert get_flext_container().has("database")

    def test_configure_global_container_with_none(self) -> None:
        """Test configuring global container with None creates new instance.

        Resets to new container when None is passed.
        """
        # Ensure we have an existing container with data
        existing_container = get_flext_container()
        existing_container.register("test_service", "test_data")

        # Reset to new container
        new_container = configure_flext_container(None)

        assert new_container is not existing_container
        assert isinstance(new_container, FlextContainer)
        assert get_flext_container() is new_container
        assert not new_container.has("test_service")

    def test_global_container_persistence(
        self,
        sample_services: dict[str, SampleService],
    ) -> None:
        """Test global container maintains state across calls."""
        container = get_flext_container()
        service = sample_services["logger"]
        container.register("logger", service)

        # Verify persistence across multiple calls
        container2 = get_flext_container()
        assert container2.has("logger")

        result = container2.get("logger")
        assert result.success
        assert result.data is service


@pytest.mark.integration
class TestFlextContainerIntegration:
    """Integration tests for FlextContainer with complex scenarios."""

    def test_mixed_service_types_integration(
        self,
        clean_container: FlextContainer,
        sample_services: dict[str, SampleService],
        service_factories: dict[str, FlextServiceFactory],
    ) -> None:
        """Test integration of regular services and singleton factories.

        Tests mixing direct services with factory patterns.
        """
        # Register mix of direct services and factories
        database_service = sample_services["database"]
        logger_factory = service_factories["logger"]

        clean_container.register("database", database_service)
        clean_container.register_factory("logger", logger_factory)

        # Verify both work correctly
        db_result = clean_container.get("database")
        logger_result = clean_container.get("logger")

        assert db_result.success
        assert logger_result.success

        assert db_result.data is database_service
        assert isinstance(logger_result.data, SampleService)
        if logger_result.data.name != "LoggerService":
            raise AssertionError(
                f"Expected {'LoggerService'}, got {logger_result.data.name}"
            )

    def test_service_replacement_scenarios(
        self,
        clean_container: FlextContainer,
        sample_services: dict[str, SampleService],
        service_factories: dict[str, FlextServiceFactory],
    ) -> None:
        """Test various service replacement scenarios."""
        # Start with direct service
        original_service = sample_services["cache"]
        clean_container.register("cache", original_service)

        result1 = clean_container.get("cache")
        assert result1.data is original_service

        # Replace with factory
        cache_factory = service_factories["cache"]
        clean_container.register_factory("cache", cache_factory)

        result2 = clean_container.get("cache")
        assert result2.success
        assert result2.data is not original_service
        assert isinstance(result2.data, SampleService)

        # Replace factory with another direct service
        new_service = SampleService("NewCacheService")
        clean_container.register("cache", new_service)

        result3 = clean_container.get("cache")
        assert result3.data is new_service

    def test_container_state_consistency(
        self,
        clean_container: FlextContainer,
        sample_services: dict[str, SampleService],
    ) -> None:
        """Test container maintains consistent state during operations.

        Validates state consistency during incremental operations.
        """
        services_to_register = {
            "service1": sample_services["database"],
            "service2": sample_services["logger"],
            "service3": sample_services["cache"],
        }

        # Register services incrementally and verify state
        for i, (name, service) in enumerate(services_to_register.items(), 1):
            clean_container.register(name, service)

            # Verify container state is consistent
            service_list = clean_container.list_services()
            if len(service_list) != i:
                raise AssertionError(f"Expected {i}, got {len(service_list)}")
            if name not in service_list:
                raise AssertionError(f"Expected {name} in {service_list}")
            assert clean_container.has(name)

            # Verify all previously registered services still work
            for registered_name in service_list:
                result = clean_container.get(registered_name)
                assert result.success

    def test_factory_error_isolation(
        self,
        clean_container: FlextContainer,
        service_factories: dict[str, FlextServiceFactory],
        failing_factory: FlextServiceFactory,
    ) -> None:
        """Test that failing factories don't affect other services."""
        # Register working factory and failing factory
        working_factory = service_factories["database"]
        clean_container.register_factory("working", working_factory)
        clean_container.register_factory("failing", failing_factory)

        # Working service should still work
        working_result = clean_container.get("working")
        assert working_result.success
        assert isinstance(working_result.data, SampleService)

        # Failing service should fail gracefully
        failing_result = clean_container.get("failing")
        assert failing_result.is_failure

        # Container should remain functional
        assert clean_container.has("working")
        assert clean_container.has("failing")  # Registration still exists
        if len(clean_container.list_services()) != 2:
            raise AssertionError(
                f"Expected {2}, got {len(clean_container.list_services())}"
            )
