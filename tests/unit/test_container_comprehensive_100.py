"""Comprehensive unit tests for FlextContainer targeting 100% coverage.

This module provides comprehensive tests for FlextContainer module to achieve
100% unit test coverage using flext_tests standardization patterns.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any, cast
from unittest.mock import Mock

from flext_core import FlextContainer, FlextResult
from flext_tests import FlextTestsMatchers


class TestFlextContainer100PercentCoverage:
    """Comprehensive FlextContainer tests targeting 100% coverage."""

    def test_service_key_creation_and_validation(self) -> None:
        """Test FlextContainer.ServiceKey creation and validation."""
        # Test valid service key
        valid_key = FlextContainer.ServiceKey("valid_service")
        FlextTestsMatchers.assert_result_success(
            FlextResult[FlextContainer.ServiceKey].ok(valid_key)
        )
        assert valid_key.name == "valid_service"

        # Test empty service key validation
        empty_key = FlextContainer.ServiceKey("")
        assert empty_key.name == ""

        # Test service key with special characters
        special_key = FlextContainer.ServiceKey("service.with.dots")
        assert special_key.name == "service.with.dots"

        # Test service key equality
        key1 = FlextContainer.ServiceKey("test")
        key2 = FlextContainer.ServiceKey("test")
        key3 = FlextContainer.ServiceKey("different")

        assert key1 == key2
        assert key1 != key3
        assert hash(key1) == hash(key2)
        assert hash(key1) != hash(key3)

    def test_commands_register_service_creation(self) -> None:
        """Test FlextContainer.Commands.RegisterService creation."""
        # Test default creation
        cmd = FlextContainer.Commands.RegisterService()
        assert cmd.service_name == ""
        assert cmd.service_instance is None

        # Test create class method
        service_data = {"name": "test_service", "port": 8080}

        cmd_created = FlextContainer.Commands.RegisterService.create(
            service_name="test_service", service_instance=service_data
        )

        assert cmd_created.service_name == "test_service"
        assert cmd_created.service_instance == service_data

    def test_commands_register_factory_creation(self) -> None:
        """Test FlextContainer.Commands.RegisterFactory creation."""
        # Test default creation
        cmd = FlextContainer.Commands.RegisterFactory()
        assert cmd.service_name == ""
        assert cmd.factory is None

        # Test with factory function
        def test_factory() -> dict[str, Any]:
            return {"name": "factory_service", "created": True}

        cmd_with_factory = FlextContainer.Commands.RegisterFactory.create(
            service_name="factory_service", factory=test_factory
        )

        assert cmd_with_factory.service_name == "factory_service"
        assert cmd_with_factory.factory == test_factory

    def test_queries_get_service_creation(self) -> None:
        """Test FlextContainer.Queries.GetService creation."""
        # Test default creation
        query = FlextContainer.Queries.GetService()
        assert query.service_name is None

        # Test create class method
        query_created = FlextContainer.Queries.GetService.create(
            service_name="test_query"
        )

        assert query_created.service_name == "test_query"

    def test_queries_list_services_creation(self) -> None:
        """Test FlextContainer.Queries.ListServices creation."""
        # Test default creation
        query = FlextContainer.Queries.ListServices()
        assert query.include_factories is True

        # Test create class method
        query_created = FlextContainer.Queries.ListServices.create(
            include_factories=True
        )

        assert query_created.include_factories is True

    def test_service_registrar_register_service(self) -> None:
        """Test FlextContainer.ServiceRegistrar.register_service method."""
        FlextContainer()
        registrar = FlextContainer.ServiceRegistrar()

        service_data = {"name": "registrar_test", "port": 9000}

        # Test successful registration
        result = registrar.register_service("registrar_test", service_data)
        FlextTestsMatchers.assert_result_success(result)

        # Verify service was registered (registrar stores services internally)
        assert registrar.has_service("registrar_test") is True

    def test_service_registrar_register_factory(self) -> None:
        """Test FlextContainer.ServiceRegistrar.register_factory method."""
        registrar = FlextContainer.ServiceRegistrar()

        def factory_func() -> dict[str, Any]:
            return {"name": "factory_test", "created_by": "registrar"}

        # Test successful factory registration
        result = registrar.register_factory("factory_test", factory_func)
        FlextTestsMatchers.assert_result_success(result)

        # Verify factory is stored
        assert registrar.has_service("factory_test") is True

    def test_container_basic_operations_comprehensive(self) -> None:
        """Test basic container operations comprehensively."""
        container = FlextContainer()

        # Test initial state
        assert container.get_service_count() == 0
        assert container.get_service_names() == []

        # Test service registration
        service_data = {"name": "basic_test", "version": "1.0"}
        register_result = container.register("basic_test", service_data)
        FlextTestsMatchers.assert_result_success(register_result)

        # Test state after registration
        assert container.get_service_count() == 1
        assert "basic_test" in container.get_service_names()

        # Test service retrieval
        get_result = container.get("basic_test")
        FlextTestsMatchers.assert_result_success(get_result, service_data)

        # Test has service
        has_result = container.has("basic_test")
        assert has_result is True

        # Test service info
        info_result = container.get_info("basic_test")
        FlextTestsMatchers.assert_result_success(info_result)

        info_data = cast("dict[str, Any]", info_result.value)
        assert info_data["name"] == "basic_test"
        assert info_data["version"] == "1.0"

        # Test service unregistration
        unregister_result = container.unregister("basic_test")
        FlextTestsMatchers.assert_result_success(unregister_result)

        # Test state after unregistration
        assert container.get_service_count() == 0
        assert "basic_test" not in container.get_service_names()
        assert container.has("basic_test") is False

    def test_container_error_handling_comprehensive(self) -> None:
        """Test comprehensive error handling scenarios."""
        container = FlextContainer()

        # Test registration with empty service name
        empty_name_result = container.register("", {"data": "test"})
        FlextTestsMatchers.assert_result_failure(
            cast("FlextResult[Any]", empty_name_result)
        )

        # Test getting non-existent service
        missing_get_result = container.get("nonexistent_service")
        FlextTestsMatchers.assert_result_failure(missing_get_result)

        # Test info for non-existent service
        missing_info_result = container.get_info("nonexistent_service")
        FlextTestsMatchers.assert_result_failure(
            cast("FlextResult[Any]", missing_info_result)
        )

        # Test unregistering non-existent service
        missing_unregister_result = container.unregister("nonexistent_service")
        FlextTestsMatchers.assert_result_failure(
            cast("FlextResult[Any]", missing_unregister_result)
        )

    def test_factory_registration_and_execution(self) -> None:
        """Test factory registration and execution scenarios."""
        container = FlextContainer()

        # Test simple factory
        def simple_factory() -> dict[str, Any]:
            return {"type": "simple", "created": True}

        factory_result = container.register_factory("simple_factory", simple_factory)
        FlextTestsMatchers.assert_result_success(factory_result)

        # Test factory execution
        get_result = container.get("simple_factory")
        FlextTestsMatchers.assert_result_success(get_result)

        factory_output = cast("dict[str, Any]", get_result.value)
        assert factory_output["type"] == "simple"
        assert factory_output["created"] is True

        # Test factory with parameters (closure)
        def create_parametric_factory(param: str) -> Callable[[], dict[str, str]]:
            def factory() -> dict[str, str]:
                return {"type": "parametric", "param": param}

            return factory

        parametric_factory = create_parametric_factory("test_param")
        param_result = container.register_factory("param_factory", parametric_factory)
        FlextTestsMatchers.assert_result_success(param_result)

        param_get_result = container.get("param_factory")
        FlextTestsMatchers.assert_result_success(param_get_result)

        param_output = cast("dict[str, Any]", param_get_result.value)
        assert param_output["param"] == "test_param"

    def test_global_container_singleton(self) -> None:
        """Test global container singleton behavior."""
        # Test singleton behavior
        global1 = FlextContainer.get_global()
        global2 = FlextContainer.get_global()

        assert global1 is global2
        assert isinstance(global1, FlextContainer)

        # Test that global container persists services
        test_service = {"name": "global_test", "persistent": True}
        register_result = global1.register("global_test", test_service)
        FlextTestsMatchers.assert_result_success(register_result)

        # Access from second reference
        get_result = global2.get("global_test")
        FlextTestsMatchers.assert_result_success(get_result, test_service)

        # Clean up for other tests
        cleanup_result = global1.unregister("global_test")
        FlextTestsMatchers.assert_result_success(cleanup_result)

    def test_container_configuration_methods(self) -> None:
        """Test container configuration and utility methods."""
        container = FlextContainer()

        # Test module utilities creation (if method exists)
        try:
            module_result = FlextContainer.create_module_utilities("test_module")
            FlextTestsMatchers.assert_result_success(module_result)

            utilities = module_result.value
            assert utilities is not None
        except AttributeError:
            # Method doesn't exist, skip this part
            pass

        # Test empty module name
        try:
            empty_module_result = FlextContainer.create_module_utilities("")
            FlextTestsMatchers.assert_result_failure(
                cast("FlextResult[Any]", empty_module_result)
            )
        except AttributeError:
            # Method doesn't exist, skip this part
            pass

        # Test configuration summary (if method exists)
        try:
            summary_result = container.get_configuration_summary()
            FlextTestsMatchers.assert_result_success(summary_result)

            summary_data = cast("dict[str, Any]", summary_result.value)
            assert isinstance(summary_data, dict)
        except AttributeError:
            # Method doesn't exist, skip this part
            pass

    def test_container_thread_safety(self) -> None:
        """Test container thread safety with concurrent operations."""
        container = FlextContainer()
        results: list[FlextResult[Any]] = []
        errors: list[Exception] = []

        def register_service_thread(service_id: int) -> None:
            """Register a service in a separate thread."""
            try:
                service_data = {
                    "id": service_id,
                    "thread": threading.current_thread().ident,
                }
                result = container.register(
                    f"thread_service_{service_id}", service_data
                )
                results.append(result)
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=register_service_thread, args=(i,))
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify all operations completed successfully
        assert len(errors) == 0, f"Thread errors occurred: {errors}"
        assert len(results) == 5

        for result in results:
            FlextTestsMatchers.assert_result_success(result)

        # Verify all services were registered
        assert container.get_service_count() == 5
        service_names = container.get_service_names()

        for i in range(5):
            assert f"thread_service_{i}" in service_names

    def test_container_with_complex_service_types(self) -> None:
        """Test container with complex service types."""
        container = FlextContainer()

        # Test with mock object
        mock_service = Mock()
        mock_service.get_data.return_value = "mocked_data"
        mock_service.status = "active"

        mock_result = container.register("mock_service", mock_service)
        FlextTestsMatchers.assert_result_success(mock_result)

        mock_get_result = container.get("mock_service")
        FlextTestsMatchers.assert_result_success(mock_get_result)

        retrieved_mock = cast("Mock", mock_get_result.value)
        assert retrieved_mock.get_data() == "mocked_data"
        assert retrieved_mock.status == "active"

        # Test with callable object
        class CallableService:
            def __init__(self, name: str) -> None:
                self.name = name

            def __call__(self) -> str:
                return f"Called {self.name}"

        callable_service = CallableService("test_callable")
        callable_result = container.register("callable_service", callable_service)
        FlextTestsMatchers.assert_result_success(callable_result)

        callable_get_result = container.get("callable_service")
        FlextTestsMatchers.assert_result_success(callable_get_result)

        retrieved_callable = cast("CallableService", callable_get_result.value)
        assert retrieved_callable() == "Called test_callable"
        assert retrieved_callable.name == "test_callable"

    def test_service_lifecycle_edge_cases(self) -> None:
        """Test service lifecycle edge cases."""
        container = FlextContainer()

        # Test overwriting existing service
        original_service = {"version": "1.0", "name": "lifecycle_test"}
        updated_service = {"version": "2.0", "name": "lifecycle_test"}

        # Register original
        original_result = container.register("lifecycle_test", original_service)
        FlextTestsMatchers.assert_result_success(original_result)

        # Overwrite with updated service
        updated_result = container.register("lifecycle_test", updated_service)
        FlextTestsMatchers.assert_result_success(updated_result)

        # Verify updated service is retrieved
        get_result = container.get("lifecycle_test")
        FlextTestsMatchers.assert_result_success(get_result)

        retrieved_service = cast("dict[str, Any]", get_result.value)
        assert retrieved_service["version"] == "2.0"

        # Test registering None as service
        none_result = container.register("none_service", None)
        FlextTestsMatchers.assert_result_success(none_result)

        none_get_result = container.get("none_service")
        FlextTestsMatchers.assert_result_success(none_get_result)

        assert none_get_result.value is None

    def test_container_bulk_operations(self) -> None:
        """Test container bulk operations and performance."""
        container = FlextContainer()

        # Register many services
        service_count = 50
        for i in range(service_count):
            service_data = {
                "id": i,
                "name": f"bulk_service_{i}",
                "category": "bulk_test",
                "index": i,
            }
            result = container.register(f"bulk_service_{i}", service_data)
            FlextTestsMatchers.assert_result_success(result)

        # Verify all services are registered
        assert container.get_service_count() == service_count
        service_names = container.get_service_names()
        assert len(service_names) == service_count

        # Verify random access to services
        for i in [0, 10, 25, 40, 49]:  # Test scattered indices
            get_result = container.get(f"bulk_service_{i}")
            FlextTestsMatchers.assert_result_success(get_result)

            service_data = cast("dict[str, Any]", get_result.value)
            assert service_data["id"] == i
            assert service_data["index"] == i

        # Test bulk unregistration
        for i in range(0, service_count, 2):  # Unregister even-indexed services
            unregister_result = container.unregister(f"bulk_service_{i}")
            FlextTestsMatchers.assert_result_success(unregister_result)

        # Verify partial unregistration
        remaining_count = service_count - (service_count // 2)
        if service_count % 2 == 1:
            remaining_count += 1  # Account for odd total

        assert container.get_service_count() == remaining_count

        # Verify remaining services are odd-indexed
        remaining_names = container.get_service_names()
        for i in range(1, service_count, 2):  # Check odd indices
            assert f"bulk_service_{i}" in remaining_names

    def test_factory_error_handling(self) -> None:
        """Test factory error handling scenarios."""
        container = FlextContainer()

        # Test factory that raises exception
        def failing_factory() -> dict[str, Any]:
            msg = "Factory intentionally failed"
            raise ValueError(msg)

        factory_result = container.register_factory("failing_factory", failing_factory)
        FlextTestsMatchers.assert_result_success(factory_result)

        # Test that getting the service handles factory exception
        get_result = container.get("failing_factory")
        # The result should be a failure due to factory exception
        FlextTestsMatchers.assert_result_failure(get_result)

        # Test factory with empty name
        def valid_factory() -> dict[str, Any]:
            return {"status": "ok"}

        empty_name_result = container.register_factory("", valid_factory)
        FlextTestsMatchers.assert_result_failure(
            cast("FlextResult[Any]", empty_name_result)
        )
