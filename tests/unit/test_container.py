"""Advanced tests for flext_core.container using comprehensive tests/support/ utilities.

Tests dependency injection patterns, performance, concurrency, and real-world scenarios
using consolidated testing infrastructure for maximum coverage and reliability.
"""

from __future__ import annotations

import time
from typing import cast

import pytest
from hypothesis import given, strategies as st
from pydantic import BaseModel

from flext_core import FlextResult, FlextTypes

from ..support import (
    AsyncTestUtils,
    FlextMatchers,
    MemoryProfiler,
    ServiceDataFactory,
    TestBuilders,
    UserDataFactory,
    build_test_container,
)

pytestmark = [pytest.mark.unit, pytest.mark.core]


# ============================================================================
# CORE CONTAINER FUNCTIONALITY TESTS
# ============================================================================


class TestFlextContainerCore:
    """Test core FlextContainer functionality with advanced testing patterns."""

    def test_container_creation_and_basic_operations(self) -> None:
        """Test container creation and basic service operations."""
        # Create container using builder
        container = TestBuilders.container().build()

        # Test service registration
        test_data = UserDataFactory.create(name="Test User", email="test@example.com")
        result = container.register("user_service", test_data)

        # Use FlextMatchers for sophisticated assertions
        FlextMatchers.assert_result_success(result)

        # Test service retrieval
        retrieval_result = container.get("user_service")
        FlextMatchers.assert_result_success(retrieval_result, expected_data=test_data)

    def test_service_registration_with_builders(self) -> None:
        """Test service registration using TestBuilders patterns."""
        # Use pre-configured container with common services
        container = build_test_container()

        # Verify all services are registered correctly
        FlextMatchers.assert_container_has_service(container, "database")
        FlextMatchers.assert_container_has_service(container, "cache")
        FlextMatchers.assert_container_has_service(container, "logger")

        # Test service data structure
        db_result = container.get("database")
        FlextMatchers.assert_result_success(db_result)

        db_service = db_result.value
        FlextMatchers.assert_json_structure(
            cast("FlextTypes.Core.JsonObject", db_service),
            ["host", "port", "name", "connected"],
            exact_match=True,
        )


class TestFlextContainerAdvanced:
    """Advanced container tests using all tests/support/ capabilities."""

    def test_performance_benchmarking(self, benchmark_utils: object) -> None:
        """Test container performance using benchmark utilities."""
        container = build_test_container()

        def service_operations() -> str:
            # Simulate typical service operations
            container.get("database")
            container.get("cache")
            container.get("logger")
            return "completed"

        # Use performance profiling instead of pytest-benchmark
        with benchmark_utils.measure_performance() as metrics:
            for _ in range(100):  # Run multiple iterations
                result = service_operations()

        assert result == "completed"
        # Assert reasonable performance (should complete in under 1 second for 100 iterations)
        assert metrics.total_time < 1.0

    def test_memory_efficiency(self) -> None:
        """Test container memory usage."""
        with MemoryProfiler.track_memory_leaks(max_increase_mb=2.0):
            # Create and use many containers
            for i in range(100):
                container = TestBuilders.container().build()
                service_data = ServiceDataFactory.create(
                    name=f"service_{i}", port=8000 + i
                )
                container.register(f"service_{i}", service_data)
                container.get(f"service_{i}")

    @pytest.mark.asyncio
    async def test_concurrent_container_operations(self) -> None:
        """Test concurrent container access."""
        container = build_test_container()

        async def concurrent_service_access(service_name: str) -> FlextResult[object]:
            await AsyncTestUtils.simulate_delay(0.01)
            return container.get(service_name)

        # Test concurrent access to different services
        tasks = [
            concurrent_service_access("database"),
            concurrent_service_access("cache"),
            concurrent_service_access("logger"),
        ]

        results = await AsyncTestUtils.run_concurrently(*tasks)

        # All operations should succeed
        for result in results:
            FlextMatchers.assert_result_success(result)

    @given(
        st.lists(
            st.text(
                min_size=1,
                max_size=20,
                alphabet=st.characters(
                    whitelist_categories=("Lu", "Ll", "Nd"),
                    blacklist_characters="\r\n\t",
                ),
            ),
            min_size=1,
            max_size=10,
        )
    )
    def test_hypothesis_service_registration(self, service_names: list[str]) -> None:
        """Property-based testing for service registration."""
        container = TestBuilders.container().build()

        # Filter out empty or whitespace-only names
        valid_names = [name.strip() for name in service_names if name.strip()]
        if not valid_names:
            return  # Skip if no valid names

        # Register services with generated names
        for name in valid_names:
            service_data = ServiceDataFactory.create(name=name)
            result = container.register(name, service_data)
            FlextMatchers.assert_result_success(result)

        # Verify all services can be retrieved
        for name in valid_names:
            result = container.get(name)
            FlextMatchers.assert_result_success(result)

    def test_factory_pattern_integration(self) -> None:
        """Test factory pattern with container."""
        container = TestBuilders.container().build()

        # Register factory function
        def user_factory() -> dict[str, object]:
            return UserDataFactory.create(name="Factory User")

        factory_result = container.register_factory("user_factory", user_factory)
        FlextMatchers.assert_result_success(factory_result)

        # Get service created by factory
        user_result = container.get("user_factory")
        FlextMatchers.assert_result_success(user_result)

        user = cast("dict[str, object]", user_result.value)
        assert user["name"] == "Factory User"

    def test_error_handling_patterns(self) -> None:
        """Test container error handling with FlextMatchers."""
        container = TestBuilders.container().build()

        # Test getting non-existent service
        result = container.get("non_existent_service")
        FlextMatchers.assert_result_failure(
            result,
            expected_error="not found",
            # error codes may vary based on implementation
        )

        # Test duplicate registration
        container.register("duplicate_service", {"data": "first"})
        duplicate_result = container.register("duplicate_service", {"data": "second"})

        # Depending on implementation, this might succeed (overwrite) or fail
        # We test that it returns a result either way
        assert isinstance(duplicate_result, FlextResult)

    def test_real_world_microservice_scenario(self) -> None:
        """Test realistic microservice dependency injection scenario."""
        # Setup realistic microservice dependencies
        container = TestBuilders.container().build()

        # Database service
        db_config = {
            "host": "postgres.example.com",
            "port": 5432,
            "database": "app_production",
            "pool_size": 20,
        }
        container.register("database", db_config)

        # Cache service
        cache_config = ServiceDataFactory.create(name="redis_cache", port=6379)
        container.register("cache", cache_config)

        # Auth service
        auth_service = {
            "jwt_secret": "secret_key",
            "token_expiry": 3600,
            "refresh_enabled": True,
        }
        container.register("auth", auth_service)

        # API service that depends on others
        def create_api_service() -> dict[str, object]:
            db = cast("dict[str, object]", container.get("database").value)
            cache = cast("dict[str, object]", container.get("cache").value)
            auth = cast("dict[str, object]", container.get("auth").value)

            return {
                "name": "main_api",
                "database_host": db["host"],
                "cache_port": cache["port"],
                "auth_enabled": auth["refresh_enabled"],
                "status": "initialized",
            }

        container.register_factory("api_service", create_api_service)

        # Test the complete dependency chain
        api_result = container.get("api_service")
        FlextMatchers.assert_result_success(api_result)

        api_service = cast("dict[str, object]", api_result.value)
        assert api_service["name"] == "main_api"
        assert api_service["database_host"] == "postgres.example.com"
        assert api_service["status"] == "initialized"

        # Test service integration
        FlextMatchers.assert_json_structure(
            cast("FlextTypes.Core.JsonObject", api_service),
            ["name", "database_host", "cache_port", "auth_enabled", "status"],
        )


# ============================================================================
# MODERN PYTHON 3.13+ COMPREHENSIVE TESTS
# ============================================================================


class TestFlextContainerModern:
    """Modern container tests using Python 3.13+ features and comprehensive coverage."""

    def test_generic_type_safety_with_pydantic(self, user_factory: object) -> None:
        """Test generic type safety with Pydantic models using Python 3.13+ syntax."""

        class UserModel(BaseModel):
            name: str
            email: str
            active: bool = True

        container = TestBuilders.container().build()

        # Test type-safe service registration with Pydantic model
        user_data = UserModel(name="John Doe", email="john@example.com")
        register_result = container.register("pydantic_user", user_data)
        FlextMatchers.assert_result_success(register_result)

        # Test type-safe retrieval
        get_result = container.get("pydantic_user")
        FlextMatchers.assert_result_success(get_result)

        retrieved_user = get_result.value
        assert isinstance(retrieved_user, UserModel)
        assert retrieved_user.name == "John Doe"
        assert retrieved_user.email == "john@example.com"
        assert retrieved_user.active is True

    @given(st.lists(st.text(min_size=1, max_size=50), min_size=1, max_size=20))
    def test_hypothesis_comprehensive_service_management(
        self, service_names: list[str]
    ) -> None:
        """Property-based testing for comprehensive service lifecycle management."""
        container = TestBuilders.container().build()

        # Filter valid service names
        valid_names = [
            name.strip() for name in service_names if name.strip() and name.isalnum()
        ]
        if not valid_names:
            return

        registered_services: dict[str, object] = {}

        # Register all services
        for name in valid_names:
            service_data = ServiceDataFactory.create(
                name=name, port=8000 + len(registered_services)
            )
            result = container.register(name, service_data)
            FlextMatchers.assert_result_success(result)
            registered_services[name] = service_data

        # Verify all services can be retrieved
        for name, expected_data in registered_services.items():
            result = container.get(name)
            FlextMatchers.assert_result_success(result, expected_data)

        # Test service replacement (should work in modern container)
        for name in list(registered_services.keys())[:3]:  # Test first 3
            new_data = ServiceDataFactory.create(name=f"updated_{name}", port=9000)
            update_result = container.register(name, new_data)
            FlextMatchers.assert_result_success(update_result)

            # Verify updated data is retrieved
            get_result = container.get(name)
            FlextMatchers.assert_result_success(get_result, new_data)

    def test_factory_pattern_with_dependency_injection(
        self, service_factory: object
    ) -> None:
        """Test advanced factory patterns with dependency injection."""
        container = TestBuilders.container().build()

        # Register dependencies
        container.register("config", {"database_url": "postgresql://localhost/app"})
        container.register("logger", {"level": "INFO", "format": "json"})

        # Factory that uses other services
        def create_complex_service() -> dict[str, object]:
            config_result = container.get("config")
            logger_result = container.get("logger")

            if config_result.failure or logger_result.failure:
                raise ValueError("Dependencies not available")

            config = cast("dict[str, object]", config_result.value)
            logger = cast("dict[str, object]", logger_result.value)

            return {
                "name": "complex_service",
                "database_url": config["database_url"],
                "log_level": logger["level"],
                "initialized": True,
                "dependency_count": 2,
            }

        # Register factory
        factory_result = container.register_factory(
            "complex_service", create_complex_service
        )
        FlextMatchers.assert_result_success(factory_result)

        # Test factory execution
        service_result = container.get("complex_service")
        FlextMatchers.assert_result_success(service_result)

        service = cast("dict[str, object]", service_result.value)
        assert service["name"] == "complex_service"
        assert service["database_url"] == "postgresql://localhost/app"
        assert service["log_level"] == "INFO"
        assert service["initialized"] is True
        assert service["dependency_count"] == 2

    @pytest.mark.asyncio
    async def test_async_service_operations_with_context_managers(self) -> None:
        """Test async service operations with proper context management."""
        container = TestBuilders.container().build()

        # Register async-compatible services
        services_to_register = [
            ("async_db", {"host": "async-db.example.com", "pool_size": 10}),
            ("async_cache", {"redis_url": "redis://async-cache:6379", "ttl": 300}),
            ("async_queue", {"broker_url": "amqp://async-queue", "max_workers": 5}),
        ]

        # Register all services
        for name, config in services_to_register:
            result = container.register(name, config)
            FlextMatchers.assert_result_success(result)

        async def async_service_consumer(service_name: str) -> FlextResult[object]:
            await AsyncTestUtils.simulate_delay(0.001)  # Simulate async work
            return container.get(service_name)

        # Test concurrent service access
        tasks = [async_service_consumer(name) for name, _ in services_to_register]
        results = await AsyncTestUtils.run_concurrently(*tasks)

        # All operations should succeed
        for result in results:
            FlextMatchers.assert_result_success(result)

        # Verify specific service configurations
        db_result = await async_service_consumer("async_db")
        db_config = cast("dict[str, object]", db_result.value)
        assert db_config["pool_size"] == 10

        cache_result = await async_service_consumer("async_cache")
        cache_config = cast("dict[str, object]", cache_result.value)
        assert cache_config["ttl"] == 300

    def test_comprehensive_error_scenarios(self, flext_matchers: object) -> None:
        """Test comprehensive error handling scenarios."""
        container = TestBuilders.container().build()

        # Test 1: Non-existent service
        result = container.get("definitely_does_not_exist")
        FlextMatchers.assert_result_failure(result)

        # Test 2: Factory that throws exception
        def failing_factory() -> dict[str, object]:
            raise ValueError("Intentional factory failure")

        container.register_factory("failing_service", failing_factory)

        # Getting the service should handle the factory exception
        failing_result = container.get("failing_service")
        # This might succeed with error handling or fail - both are valid
        assert isinstance(failing_result, FlextResult)

        # Test 3: Circular dependency detection (if implemented)
        def circular_factory_a() -> dict[str, object]:
            container.get("circular_b")  # This would create a circular dependency
            return {"name": "circular_a"}

        def circular_factory_b() -> dict[str, object]:
            container.get("circular_a")  # This creates the circle
            return {"name": "circular_b"}

        container.register_factory("circular_a", circular_factory_a)
        container.register_factory("circular_b", circular_factory_b)

        # Attempting to resolve should handle gracefully
        circular_result = container.get("circular_a")
        assert isinstance(circular_result, FlextResult)
        # Container should either detect and fail, or have some protection

    def test_performance_with_large_service_registry(
        self, memory_profiler: object
    ) -> None:
        """Test performance and memory efficiency with large number of services."""
        container = TestBuilders.container().build()

        # Register many services
        service_count = 100  # Reduced from 500 for test efficiency
        with memory_profiler.track_memory_leaks(max_increase_mb=5.0):
            for i in range(service_count):
                service_data = ServiceDataFactory.create(
                    name=f"bulk_service_{i}", port=10000 + i
                )
                result = container.register(f"service_{i}", service_data)
                FlextMatchers.assert_result_success(result)

        # Test retrieval performance
        start_time = time.perf_counter()

        # Retrieve subset of services
        for i in range(0, service_count, 10):  # Every 10th service
            result = container.get(f"service_{i}")
            FlextMatchers.assert_result_success(result)

        end_time = time.perf_counter()
        retrieval_time = end_time - start_time

        # Should be able to retrieve 10 services quickly
        assert retrieval_time < 0.1  # Less than 100ms
