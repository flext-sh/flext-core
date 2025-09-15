"""Comprehensive container coverage tests using flext_tests patterns.

Tests critical FlextContainer functionality with edge cases, error handling,
and performance scenarios to achieve higher coverage using standardized patterns.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import FlextContainer
from flext_tests import FlextTestsMatchers


class TestFlextContainerComprehensiveCoverage:
    """Comprehensive coverage tests for FlextContainer using flext_tests patterns."""

    def test_service_registration_edge_cases(
        self, clean_container: FlextContainer
    ) -> None:
        """Test service registration edge cases for coverage."""
        container = clean_container

        # Test registration with None service
        result = container.register("test_service", None)
        FlextTestsMatchers.assert_result_success(result)  # Container accepts None

        # Test registration with empty string name
        result = container.register("", "service")
        FlextTestsMatchers.assert_result_failure(result)

        # Test registration with whitespace-only name
        result = container.register("   ", "service")
        FlextTestsMatchers.assert_result_failure(result)

    def test_service_retrieval_edge_cases(
        self, clean_container: FlextContainer
    ) -> None:
        """Test service retrieval edge cases for coverage."""
        container = clean_container

        # Test getting non-existent service
        result = container.get("non_existent")
        FlextTestsMatchers.assert_result_failure(result)

        # Test getting with empty string name
        result = container.get("")
        FlextTestsMatchers.assert_result_failure(result)

        # Register and retrieve valid service
        container.register("valid_service", {"data": "test"})
        result = container.get("valid_service")
        FlextTestsMatchers.assert_result_success(result)
        assert result.value == {"data": "test"}

    def test_container_clear_functionality(
        self, clean_container: FlextContainer
    ) -> None:
        """Test container clear functionality."""
        container = clean_container

        # Register multiple services
        for i in range(3):
            container.register(f"service_{i}", {"index": i})

        assert container.get_service_count() == 3

        # Clear container
        container.clear()

        # Verify all services are gone
        assert container.get_service_count() == 0

    def test_service_unregistration(self, clean_container: FlextContainer) -> None:
        """Test service unregistration edge cases."""
        container = clean_container

        # Test unregistering non-existent service
        result = container.unregister("non_existent")
        FlextTestsMatchers.assert_result_failure(result)

        # Test unregistering existing service
        container.register("temp_service", {"temp": "data"})
        result = container.unregister("temp_service")
        FlextTestsMatchers.assert_result_success(result)

        # Verify service is gone
        get_result = container.get("temp_service")
        FlextTestsMatchers.assert_result_failure(get_result)

    def test_container_has_method(self, clean_container: FlextContainer) -> None:
        """Test container has() method comprehensively."""
        container = clean_container

        # Test non-existent service
        assert not container.has("non_existent")

        # Register service and test
        container.register("existing_service", {"data": "test"})
        assert container.has("existing_service")

        # Unregister and test again
        container.unregister("existing_service")
        assert not container.has("existing_service")

    def test_service_listing(self, clean_container: FlextContainer) -> None:
        """Test service listing functionality."""
        container = clean_container

        # Test empty container
        assert container.get_service_count() == 0
        assert container.get_service_names() == []

        # Register services
        services = {"service_a": "data_a", "service_b": "data_b"}
        for name, data in services.items():
            container.register(name, data)

        # Test service count and names
        assert container.get_service_count() == 2
        names = container.get_service_names()
        assert len(names) == 2
        assert all(name in names for name in services)

    def test_factory_registration(self, clean_container: FlextContainer) -> None:
        """Test factory registration functionality."""
        container = clean_container

        # Test factory registration
        def create_service() -> dict[str, str]:
            return {"type": "factory_created", "id": "123"}

        result = container.register_factory("factory_service", create_service)
        FlextTestsMatchers.assert_result_success(result)

        # Test factory execution
        service_result = container.get("factory_service")
        FlextTestsMatchers.assert_result_success(service_result)
        assert service_result.value["type"] == "factory_created"

    def test_container_error_handling(self, clean_container: FlextContainer) -> None:
        """Test comprehensive error handling scenarios."""
        container = clean_container

        # Test invalid service names
        invalid_names = ["", "   ", "\t", "\n"]
        for invalid_name in invalid_names:
            result = container.register(invalid_name, "data")
            FlextTestsMatchers.assert_result_failure(result)

            get_result = container.get(invalid_name)
            FlextTestsMatchers.assert_result_failure(get_result)

    def test_container_type_safety(self, clean_container: FlextContainer) -> None:
        """Test container type safety and value preservation."""
        container = clean_container

        # Test various data types
        test_data = {
            "string": "test_string",
            "integer": 42,
            "boolean": True,
            "none": None,
            "list": [1, 2, 3],
        }

        for key, value in test_data.items():
            # Register with type-specific key
            service_name = f"typed_{key}"
            result = container.register(service_name, value)
            FlextTestsMatchers.assert_result_success(result)

            # Retrieve and verify type preservation
            get_result = container.get(service_name)
            FlextTestsMatchers.assert_result_success(get_result)
            assert get_result.value == value
            assert isinstance(get_result.value, type(value))

    def test_global_container_access(self, clean_container: FlextContainer) -> None:
        """Test global container instance management."""
        # Test global container access
        global_container = FlextContainer.get_global()
        assert isinstance(global_container, FlextContainer)

        # Test that multiple calls return same instance
        global_container2 = FlextContainer.get_global()
        assert global_container is global_container2

        # Test clean_container is different from global
        assert clean_container is not global_container
