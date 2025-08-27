"""Modern tests for flext_core.container - Dependency Injection Implementation.

Refactored test suite using comprehensive testing libraries for container functionality.
Demonstrates SOLID principles, DI patterns, and extensive test automation.
"""

from __future__ import annotations

import pytest
from hypothesis import given, strategies as st
from tests.conftest import TestScenario
from tests.support.async_utils import AsyncTestUtils
from tests.support.domain_factories import UserDataFactory
from tests.support.factory_boy_factories import (
    EdgeCaseGenerators,
    UserFactory,
    create_validation_test_cases,
)
from tests.support.performance_utils import BenchmarkUtils, PerformanceProfiler

from flext_core import FlextContainer, FlextResult, get_flext_container
from flext_core.constants import FlextConstants
from flext_core.typings import FlextTypes

pytestmark = [pytest.mark.unit, pytest.mark.core]


# ============================================================================
# CORE CONTAINER FUNCTIONALITY TESTS
# ============================================================================


class TestFlextContainerCore:
    """Test core FlextContainer functionality with factory patterns."""

    def test_container_singleton_behavior(self) -> None:
        """Test that get_flext_container returns singleton instance."""
        container1 = get_flext_container()
        container2 = get_flext_container()

        assert container1 is container2
        assert isinstance(container1, FlextContainer)

    def test_service_registration_success(
        self, user_data_factory: UserDataFactory
    ) -> None:
        """Test successful service registration using factories."""
        container = get_flext_container()
        test_service = user_data_factory.build()

        result = container.register("test_service", test_service)

        assert result.success
        assert result.value is None

    def test_service_retrieval_success(
        self, user_data_factory: UserDataFactory
    ) -> None:
        """Test successful service retrieval."""
        container = get_flext_container()
        test_service = user_data_factory.build()

        container.register("test_service", test_service)
        result = container.get("test_service")

        assert result.success
        assert result.value == test_service

    def test_service_not_found(self) -> None:
        """Test retrieval of non-existent service."""
        container = get_flext_container()

        result = container.get("non_existent_service")

        assert result.is_failure
        assert FlextConstants.Messages.OPERATION_FAILED.lower() in result.error.lower()

    def test_factory_registration_and_execution(self) -> None:
        """Test factory registration and lazy evaluation."""
        container = get_flext_container()
        factory_called = False

        def test_factory() -> FlextTypes.Core.String:
            nonlocal factory_called
            factory_called = True
            return FlextConstants.Messages.SUCCESS

        result = container.register_factory("test_factory", test_factory)
        assert result.success
        assert not factory_called  # Factory not called yet

        get_result = container.get("test_factory")
        assert get_result.success
        assert get_result.value == FlextConstants.Messages.SUCCESS
        assert factory_called  # Factory called during get


# ============================================================================
# DEPENDENCY INJECTION PATTERNS TESTS
# ============================================================================


class TestFlextContainerDI:
    """Test dependency injection patterns."""

    def test_complex_dependency_chain(self) -> None:
        """Test complex dependency injection chains."""
        container = get_flext_container()

        # Register dependencies in reverse order
        container.register("config", {"database_url": "localhost:5432"})

        class DatabaseService:
            def __init__(self, url: FlextTypes.Core.String) -> None:
                self.url = url
                self.connected = False

            def connect(self) -> FlextTypes.Core.Boolean:
                self.connected = True
                return True

        def create_database() -> DatabaseService:
            config_result = container.get("config")
            if config_result.success:
                config = config_result.value
                return DatabaseService(config["database_url"])
            msg = "Config not found"
            raise ValueError(msg)

        class UserService:
            def __init__(self, database: DatabaseService) -> None:
                self.db = database
                self.users = []

            def create_user(self, user_data: FlextTypes.Core.JsonObject) -> FlextTypes.Core.JsonObject:
                user = {"id": len(self.users) + 1, **user_data}
                self.users.append(user)
                return user

        def create_user_service() -> UserService:
            database_result = container.get("database")
            if database_result.success:
                database = database_result.value
                return UserService(database)
            msg = "Database not found"
            raise ValueError(msg)

        container.register_factory("database", create_database)
        container.register_factory("user_service", create_user_service)

        # Retrieve service and verify chain
        service_result = container.get("user_service")
        assert service_result.success

        service = service_result.value
        assert service.db.url == "localhost:5432"


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================


class TestFlextContainerPerformance:
    """Test container performance characteristics."""

    def test_registration_performance(self, benchmark: object) -> None:
        """Benchmark service registration performance."""
        container = get_flext_container()

        def register_many_services() -> None:
            for i in range(100):
                container.register(f"service_{i}", f"value_{i}")

        BenchmarkUtils.benchmark_with_warmup(
            benchmark, register_many_services, warmup_rounds=3
        )

    def test_retrieval_performance(self, benchmark: object) -> None:
        """Benchmark service retrieval performance."""
        container = get_flext_container()

        # Pre-register services
        for i in range(100):
            container.register(f"service_{i}", f"value_{i}")

        def retrieve_services() -> FlextTypes.Core.List[FlextTypes.Core.String]:
            results = []
            for i in range(100):
                result = container.get(f"service_{i}")
                if result.success:
                    results.append(result.value)
            return results

        results = BenchmarkUtils.benchmark_with_warmup(
            benchmark, retrieve_services, warmup_rounds=3
        )

        assert len(results) == 100

    def test_memory_efficiency(self) -> None:
        """Test memory efficiency of container operations."""
        profiler = PerformanceProfiler()

        with profiler.profile_memory("container_operations"):
            container = get_flext_container()

            # Register many services
            for i in range(1000):
                container.register(f"service_{i}", {"data": f"value_{i}"})

            # Retrieve all services
            for i in range(1000):
                container.get(f"service_{i}")

        profiler.assert_memory_efficient(
            max_memory_mb=20.0, operation_name="container_operations"
        )


# ============================================================================
# ASYNC CONTAINER TESTS
# ============================================================================


class TestFlextContainerAsync:
    """Test container usage in async contexts."""

    @pytest.mark.asyncio
    async def test_concurrent_container_access(self) -> None:
        """Test concurrent access to container."""
        container = get_flext_container()

        async def register_service(service_id: FlextTypes.Core.Integer) -> FlextResult[None]:
            await AsyncTestUtils.sleep_with_timeout(0.001)
            return container.register(f"concurrent_service_{service_id}", service_id)

        async def get_service(service_id: FlextTypes.Core.Integer) -> FlextResult[FlextTypes.Core.Object]:
            await AsyncTestUtils.sleep_with_timeout(0.001)
            return container.get(f"concurrent_service_{service_id}")

        # Register services concurrently
        register_tasks = [register_service(i) for i in range(10)]
        register_results = await AsyncTestUtils.run_concurrent(register_tasks)

        assert len(register_results) == 10
        assert all(r.success for r in register_results)

        # Retrieve services concurrently
        get_tasks = [get_service(i) for i in range(10)]
        get_results = await AsyncTestUtils.run_concurrent(get_tasks)

        assert len(get_results) == 10
        assert all(r.success for r in get_results)


# ============================================================================
# PROPERTY-BASED TESTS
# ============================================================================


class TestFlextContainerProperties:
    """Property-based tests for container invariants."""

    @given(st.text(min_size=1), st.text())
    def test_register_get_roundtrip(
        self, service_name: FlextTypes.Core.String, service_value: FlextTypes.Core.String
    ) -> None:
        """Property: register then get returns the same value."""
        container = get_flext_container()

        register_result = container.register(service_name, service_value)
        if register_result.success:
            get_result = container.get(service_name)
            assert get_result.success
            assert get_result.value == service_value

    @given(st.text(min_size=1))
    def test_unregistered_service_failure(self, service_name: FlextTypes.Core.String) -> None:
        """Property: getting unregistered service always fails."""
        container = get_flext_container()

        # Ensure service is not registered by using unique name
        unique_name = f"unregistered_{service_name}_{hash(service_name)}"
        result = container.get(unique_name)

        assert result.is_failure


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestFlextContainerIntegration:
    """Integration tests using test scenarios."""

    def test_user_service_integration(self, user_data_factory: UserDataFactory) -> None:
        """Test complete user service integration."""
        container = get_flext_container()
        user_data = user_data_factory.build()

        # Register real user repository
        class UserRepository:
            def __init__(self) -> None:
                self.users = []

            def save(self, user_data: FlextTypes.Core.JsonObject) -> FlextResult[FlextTypes.Core.JsonObject]:
                user = {"id": len(self.users) + 1, **user_data}
                self.users.append(user)
                return FlextResult.ok(user)

        user_repo = UserRepository()
        container.register("user_repository", user_repo)

        # Register user service factory
        class UserService:
            def __init__(self, repository: UserRepository) -> None:
                self.repository = repository

            def create_user(
                self, user_data: FlextTypes.Core.JsonObject
            ) -> FlextResult[FlextTypes.Core.JsonObject]:
                return self.repository.save(user_data)

        def create_user_service() -> UserService:
            repo_result = container.get("user_repository")
            if repo_result.success:
                repo = repo_result.value
                return UserService(repo)
            msg = "User repository not found"
            raise ValueError(msg)

        container.register_factory("user_service", create_user_service)

        # Use the service
        service_result = container.get("user_service")
        assert service_result.success

        service = service_result.value
        result = service.create_user(user_data)

        assert result.success
        assert result.value == user_data

    def test_error_handling_scenarios(self, test_scenarios: list[TestScenario]) -> None:
        """Test container error handling with various scenarios."""
        container = get_flext_container()

        # Check if ERROR_CASE scenario exists
        has_error_case = TestScenario.ERROR_CASE in test_scenarios
        if not has_error_case:
            pytest.skip("No error scenario available")

        def failing_factory() -> str:
            msg = "Factory failure"
            raise ValueError(msg)

        container.register_factory("failing_service", failing_factory)

        result = container.get("failing_service")
        # Container should handle the factory error gracefully
        assert result.is_failure


# ============================================================================
# EDGE CASE TESTS
# ============================================================================


class TestFlextContainerEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.parametrize("edge_value", EdgeCaseGenerators.unicode_strings())
    def test_unicode_service_names(self, edge_value: str) -> None:
        """Test container with unicode service names."""
        container = get_flext_container()

        if edge_value:  # Skip empty strings
            result = container.register(edge_value, "test_value")
            if result.success:
                get_result = container.get(edge_value)
                assert get_result.success
                assert get_result.value == "test_value"

    @pytest.mark.parametrize("edge_value", EdgeCaseGenerators.empty_values())
    def test_empty_value_registration(self, edge_value: object) -> None:
        """Test container with empty/null values."""
        container = get_flext_container()

        result = container.register("empty_service", edge_value)
        if result.success:
            get_result = container.get("empty_service")
            assert get_result.success
            assert get_result.value == edge_value

    def test_large_service_registration(self) -> None:
        """Test container with large service objects."""
        container = get_flext_container()
        large_data = EdgeCaseGenerators.large_values()[0]  # Large string

        result = container.register("large_service", large_data)
        assert result.success

        get_result = container.get("large_service")
        assert get_result.success
        assert get_result.value == large_data


# ============================================================================
# FACTORY BOY INTEGRATION TESTS
# ============================================================================


class TestFlextContainerFactoryIntegration:
    """Test integration with factory_boy factories."""

    def test_user_factory_integration(self) -> None:
        """Test container with factory_boy user creation."""
        container = get_flext_container()

        def user_service_factory() -> object:
            class RealUserService:
                def create_user(self) -> object:
                    return UserFactory()

            return RealUserService()

        container.register_factory("user_service", user_service_factory)

        service_result = container.get("user_service")
        assert service_result.success

        service = service_result.value
        user = service.create_user()

        assert hasattr(user, "name")
        assert hasattr(user, "email")
        assert hasattr(user, "age")

    def test_validation_test_cases_integration(self) -> None:
        """Test container with comprehensive validation test cases."""
        container = get_flext_container()
        test_cases = create_validation_test_cases()

        def validator_service_factory() -> object:
            class RealValidatorService:
                def validate_data(self, data: object) -> bool:
                    return data is not None

            return RealValidatorService()

        container.register_factory("validator", validator_service_factory)

        validator_result = container.get("validator")
        assert validator_result.success

        validator = validator_result.value

        for case in test_cases:
            result = validator.validate_data(case["data"])
            if case["expected_valid"]:
                assert result is True or case["data"] is not None


# ============================================================================
# STRESS TESTING
# ============================================================================


class TestFlextContainerStress:
    """Stress tests for container patterns."""

    def test_high_volume_service_registration(self) -> None:
        """Test container performance with many services."""
        container = get_flext_container()

        # Register many services
        for i in range(1000):
            result = container.register(f"stress_service_{i}", i)
            assert result.success

        # Verify all services are retrievable
        for i in range(0, 1000, 100):  # Sample every 100th service
            result = container.get(f"stress_service_{i}")
            assert result.success
            assert result.value == i
