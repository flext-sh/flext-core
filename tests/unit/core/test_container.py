"""Comprehensive tests for FlextContainer system.

Tests container functionality including edge cases, error conditions,
and advanced features for complete coverage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Callable

import pytest

from flext_core.container import (
    FlextContainer,
    FlextGlobalContainerManager,
    FlextServiceRegistrar,
    FlextServiceRetrivier,
    ServiceKey,
    configure_flext_container,
    get_flext_container,
    get_typed,
    register_typed,
)
from flext_core.exceptions import FlextError

if TYPE_CHECKING:
    from collections.abc import Callable

# Constants
EXPECTED_BULK_SIZE = 2


class SampleService:
    """Test service for container testing."""

    def __init__(self, name: str = "test") -> None:
        self.name = name

    def get_name(self) -> str:
        """Get service name."""
        return self.name


class DependentService:
    """Service with dependencies for auto-wiring tests."""

    def __init__(
        self,
        test_service: SampleService,
        optional_param: str = "default",
    ) -> None:
        self.test_service = test_service
        self.optional_param = optional_param


class ComplexService:
    """Service with complex constructor for testing."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.args = args
        self.kwargs = kwargs


@pytest.mark.unit
class TestFlextServiceRegistrar:
    """Test FlextServiceRegistrar component directly."""

    def test_registrar_initialization(self) -> None:
        """Test registrar initializes with empty state."""
        registrar = FlextServiceRegistrar()

        if len(registrar.get_services_dict()) != 0:
            raise AssertionError(
                f"Expected {0}, got {len(registrar.get_services_dict())}"
            )
        assert len(registrar.get_factories_dict()) == 0
        if registrar.get_service_count() != 0:
            raise AssertionError(f"Expected {0}, got {registrar.get_service_count()}")
        assert registrar.get_service_names() == []

    def test_service_name_validation_empty(self) -> None:
        """Test service name validation with empty names."""
        registrar = FlextServiceRegistrar()
        service = SampleService("test")

        # Test empty string
        result = registrar.register_service("", service)
        assert result.is_failure
        if "Service name cannot be empty" not in (result.error or ""):
            raise AssertionError(
                f"Expected 'Service name cannot be empty' in {result.error}"
            )

        # Test whitespace string
        result = registrar.register_service("   ", service)
        assert result.is_failure
        if "Service name cannot be empty" not in (result.error or ""):
            raise AssertionError(
                f"Expected 'Service name cannot be empty' in {result.error}"
            )

    def test_service_registration_replacement_warning(self) -> None:
        """Test service registration warns when replacing existing service."""
        registrar = FlextServiceRegistrar()
        service1 = SampleService("first")
        service2 = SampleService("second")

        # Register first service
        result1 = registrar.register_service("test", service1)
        assert result1.success

        # Register replacement service - should warn but succeed
        result2 = registrar.register_service("test", service2)
        assert result2.success

        # Verify replacement occurred
        assert registrar.get_services_dict()["test"] is service2

    def test_factory_registration_removes_existing_service(self) -> None:
        """Test factory registration removes existing service."""
        registrar = FlextServiceRegistrar()
        service = SampleService("test")

        # Register service first
        registrar.register_service("test", service)
        if "test" not in registrar.get_services_dict():
            raise AssertionError(
                f"Expected {'test'} in {registrar.get_services_dict()}"
            )

        # Register factory for same name
        def factory() -> SampleService:
            return SampleService("factory")

        result = registrar.register_factory("test", factory)
        assert result.success

        # Service should be removed, factory should be present
        if "test" in registrar.get_services_dict():
            raise AssertionError(
                f"Expected 'test' to be removed from services, but found it in {registrar.get_services_dict()}"
            )
        assert "test" in registrar.get_factories_dict()

    def test_factory_registration_non_callable(self) -> None:
        """Test factory registration rejects non-callable objects."""
        registrar = FlextServiceRegistrar()

        result = registrar.register_factory("test", "not_callable")
        assert result.is_failure
        if "Factory must be callable" not in (result.error or ""):
            raise AssertionError(
                f"Expected 'Factory must be callable' in {result.error}"
            )

    def test_unregister_service_success(self) -> None:
        """Test successful service unregistration."""
        registrar = FlextServiceRegistrar()
        service = SampleService("test")

        # Register and unregister service
        registrar.register_service("test", service)
        result = registrar.unregister_service("test")
        assert result.success
        if "test" in registrar.get_services_dict():
            raise AssertionError(
                f"Expected 'test' to be removed from services, but found it in {registrar.get_services_dict()}"
            )

    def test_unregister_factory_success(self) -> None:
        """Test successful factory unregistration."""
        registrar = FlextServiceRegistrar()

        def factory() -> SampleService:
            return SampleService("test")

        # Register and unregister factory
        registrar.register_factory("test", factory)
        result = registrar.unregister_service("test")
        assert result.success
        if "test" in registrar.get_factories_dict():
            raise AssertionError(
                f"Expected 'test' to be removed from factories, but found it in {registrar.get_factories_dict()}"
            )

    def test_unregister_nonexistent_service(self) -> None:
        """Test unregistering non-existent service."""
        registrar = FlextServiceRegistrar()

        result = registrar.unregister_service("nonexistent")
        assert result.is_failure
        if "Service 'nonexistent' not found" not in (result.error or ""):
            raise AssertionError(
                f"Expected \"Service 'nonexistent' not found\" in {result.error}"
            )

    def test_clear_all_services(self) -> None:
        """Test clearing all services and factories."""
        registrar = FlextServiceRegistrar()
        service = SampleService("test")

        def factory() -> SampleService:
            return SampleService("factory")

        # Register service and factory
        registrar.register_service("service", service)
        registrar.register_factory("factory", factory)

        # Clear all
        result = registrar.clear_all()
        assert result.success
        if len(registrar.get_services_dict()) != 0:
            raise AssertionError(
                f"Expected {0}, got {len(registrar.get_services_dict())}"
            )
        assert len(registrar.get_factories_dict()) == 0
        if registrar.get_service_count() != 0:
            raise AssertionError(f"Expected {0}, got {registrar.get_service_count()}")

    def test_has_service_check(self) -> None:
        """Test service existence checking."""
        registrar = FlextServiceRegistrar()
        service = SampleService("test")

        def factory() -> SampleService:
            return SampleService("factory")

        # Initially no services
        assert not registrar.has_service("test")

        # Register service
        registrar.register_service("service", service)
        assert registrar.has_service("service")

        # Register factory
        registrar.register_factory("factory", factory)
        assert registrar.has_service("factory")

    def test_get_service_names_combined(self) -> None:
        """Test getting all service names includes both services and factories."""
        registrar = FlextServiceRegistrar()
        service = SampleService("test")

        def factory() -> SampleService:
            return SampleService("factory")

        registrar.register_service("service", service)
        registrar.register_factory("factory", factory)

        names = registrar.get_service_names()
        if "service" not in names:
            raise AssertionError(f"Expected {'service'} in {names}")
        assert "factory" in names
        if len(names) != EXPECTED_BULK_SIZE:
            raise AssertionError(f"Expected {2}, got {len(names)}")


@pytest.mark.unit
class TestFlextServiceRetrivier:
    """Test FlextServiceRetrivier component directly."""

    def test_retriever_initialization(self) -> None:
        """Test retriever initializes with provided dictionaries."""
        services: dict[str, object] = {"test": SampleService("test")}
        factories: dict[str, Callable[[], object]] = {
            "factory": lambda: SampleService("factory")
        }

        retriever = FlextServiceRetrivier(services, factories)

        # Verify it has references to the dictionaries
        assert retriever._services is services
        assert retriever._factories is factories

    def test_get_service_from_cache(self) -> None:
        """Test retrieving service from cache."""
        service = SampleService("test")
        services: dict[str, object] = {"test": service}
        factories: dict[str, Callable[[], object]] = {}

        retriever = FlextServiceRetrivier(services, factories)

        result = retriever.get_service("test")
        assert result.success
        assert result.data is service

    def test_get_service_from_factory_with_caching(self) -> None:
        """Test retrieving service from factory and caching behavior."""
        services: dict[str, object] = {}
        call_count = 0

        def factory() -> SampleService:
            nonlocal call_count
            call_count += 1
            return SampleService(f"factory_{call_count}")

        factories: dict[str, Callable[[], object]] = {"test": factory}
        retriever = FlextServiceRetrivier(services, factories)

        # First call should execute factory
        result1 = retriever.get_service("test")
        assert result1.success
        if call_count != 1:
            raise AssertionError(f"Expected {1}, got {call_count}")
        assert isinstance(result1.data, SampleService)

        # Verify service was cached and factory removed
        if "test" not in services:
            raise AssertionError(f"Expected {'test'} in {services}")
        assert "test" not in factories

        # Second call should use cached service
        result2 = retriever.get_service("test")
        assert result2.success
        assert result2.data is result1.data
        if call_count != 1:
            raise AssertionError(f"Expected {1}, got {call_count}")

    def test_get_service_factory_failure(self) -> None:
        """Test factory execution failure handling."""
        services: dict[str, object] = {}

        def failing_factory() -> SampleService:
            msg = "Factory failed"
            raise FlextError(msg)

        factories: dict[str, Callable[[], object]] = {"test": failing_factory}
        retriever = FlextServiceRetrivier(services, factories)

        result = retriever.get_service("test")
        assert result.is_failure
        if "Factory for 'test' failed" not in (result.error or ""):
            raise AssertionError(
                f"Expected \"Factory for 'test' failed\" in {result.error}"
            )
        assert "Factory failed" in (result.error or "")

    def test_get_service_factory_various_exceptions(self) -> None:
        """Test factory failure with various exception types."""
        services: dict[str, object] = {}

        exception_types = [
            TypeError("Type error"),
            ValueError("Value error"),
            AttributeError("Attribute error"),
            RuntimeError("Runtime error"),
        ]

        for i, exception in enumerate(exception_types):

            def failing_factory(exc: Exception = exception) -> SampleService:
                raise exc

            factories: dict[str, Callable[[], object]] = {f"test_{i}": failing_factory}
            retriever = FlextServiceRetrivier(services, factories)

            result = retriever.get_service(f"test_{i}")
            assert result.is_failure
            if f"Factory for 'test_{i}' failed" not in (result.error or ""):
                raise AssertionError(
                    f"Expected \"Factory for 'test_{i}' failed\" in {result.error}"
                )

    def test_get_service_not_found(self) -> None:
        """Test getting non-existent service."""
        services: dict[str, object] = {}
        factories: dict[str, Callable[[], object]] = {}
        retriever = FlextServiceRetrivier(services, factories)

        result = retriever.get_service("nonexistent")
        assert result.is_failure
        if "Service 'nonexistent' not found" not in (result.error or ""):
            raise AssertionError(
                f"Expected \"Service 'nonexistent' not found\" in {result.error}"
            )

    def test_get_service_info_for_instance(self) -> None:
        """Test getting service info for instance."""
        service = SampleService("test")
        services: dict[str, object] = {"test": service}
        factories: dict[str, Callable[[], object]] = {}
        retriever = FlextServiceRetrivier(services, factories)

        result = retriever.get_service_info("test")
        assert result.success
        assert result.data is not None
        info = result.data
        assert isinstance(info, dict)
        if info["name"] != "test":
            raise AssertionError(f"Expected {'test'}, got {info['name']}")
        assert info["type"] == "instance"
        if info["class"] != "SampleService":
            raise AssertionError(f"Expected {'SampleService'}, got {info['class']}")
        assert isinstance(info["module"], str)
        if "test_container" not in info["module"]:
            raise AssertionError(f"Expected {'test_container'} in {info['module']}")

    def test_get_service_info_for_factory(self) -> None:
        """Test getting service info for factory."""

        def test_factory() -> SampleService:
            return SampleService("test")

        services: dict[str, object] = {}
        factories: dict[str, Callable[[], object]] = {"test": test_factory}
        retriever = FlextServiceRetrivier(services, factories)

        result = retriever.get_service_info("test")
        assert result.success
        assert result.data is not None
        info = result.data
        assert isinstance(info, dict)
        if info["name"] != "test":
            raise AssertionError(f"Expected {'test'}, got {info['name']}")
        assert info["type"] == "factory"
        if info["factory"] != "test_factory":
            raise AssertionError(f"Expected {'test_factory'}, got {info['factory']}")
        assert isinstance(info["module"], str)
        if "test_container" not in info["module"]:
            raise AssertionError(f"Expected {'test_container'} in {info['module']}")

    def test_get_service_info_not_found(self) -> None:
        """Test getting info for non-existent service."""
        services: dict[str, object] = {}
        factories: dict[str, Callable[[], object]] = {}
        retriever = FlextServiceRetrivier(services, factories)

        result = retriever.get_service_info("nonexistent")
        assert result.is_failure
        if "Service 'nonexistent' not found" not in (result.error or ""):
            raise AssertionError(
                f"Expected \"Service 'nonexistent' not found\" in {result.error}"
            )

    def test_list_services_mixed(self) -> None:
        """Test listing services with mixed types."""
        service = SampleService("test")
        services: dict[str, object] = {"instance": service}

        def factory() -> SampleService:
            return SampleService("factory")

        factories: dict[str, Callable[[], object]] = {"factory": factory}
        retriever = FlextServiceRetrivier(services, factories)

        service_list = retriever.list_services()
        if service_list["instance"] != "instance":
            raise AssertionError(
                f"Expected {'instance'}, got {service_list['instance']}"
            )
        assert service_list["factory"] == "factory"
        if len(service_list) != EXPECTED_BULK_SIZE:
            raise AssertionError(f"Expected {2}, got {len(service_list)}")


@pytest.mark.unit
class TestFlextContainerAdvancedFeatures:
    """Test advanced FlextContainer features not covered in basic tests."""

    def test_get_typed_success(self, clean_container: FlextContainer) -> None:
        """Test type-safe service retrieval success."""
        service = SampleService("test")
        clean_container.register("test", service)

        result = clean_container.get_typed("test", SampleService)
        assert result.success
        assert result.data is service
        assert isinstance(result.data, SampleService)

    def test_get_typed_type_mismatch(self, clean_container: FlextContainer) -> None:
        """Test type-safe service retrieval with type mismatch."""
        service = SampleService("test")
        clean_container.register("test", service)

        # Try to get as wrong type
        result = clean_container.get_typed("test", str)
        assert result.is_failure
        if "is SampleService, expected str" not in (result.error or ""):
            raise AssertionError(
                f"Expected 'is SampleService, expected str' in {result.error}"
            )

    def test_get_typed_service_not_found(self, clean_container: FlextContainer) -> None:
        """Test type-safe retrieval of non-existent service."""
        result = clean_container.get_typed("nonexistent", SampleService)
        assert result.is_failure
        if "not found" not in (result.error or ""):
            raise AssertionError(f"Expected 'not found' in {result.error}")

    def test_auto_wire_success(self, clean_container: FlextContainer) -> None:
        """Test successful auto-wiring of service with dependencies."""
        # Register dependency
        test_service = SampleService("dependency")
        clean_container.register("test_service", test_service)

        # Auto-wire dependent service
        result = clean_container.auto_wire(DependentService)
        assert result.success

        dependent = result.data
        assert isinstance(dependent, DependentService)
        assert dependent.test_service is test_service
        if dependent.optional_param != "default":
            raise AssertionError(
                f"Expected {'default'}, got {dependent.optional_param}"
            )

        # Verify service was registered
        assert clean_container.has("DependentService")

    def test_auto_wire_with_custom_name(self, clean_container: FlextContainer) -> None:
        """Test auto-wiring with custom service name."""
        test_service = SampleService("dependency")
        clean_container.register("test_service", test_service)

        result = clean_container.auto_wire(DependentService, "custom_name")
        assert result.success
        assert clean_container.has("custom_name")

    def test_auto_wire_missing_dependency(
        self,
        clean_container: FlextContainer,
    ) -> None:
        """Test auto-wiring fails when required dependency is missing."""
        result = clean_container.auto_wire(DependentService)
        assert result.is_failure
        if "Required dependency 'test_service' not found" not in (result.error or ""):
            raise AssertionError(
                f"Expected \"Required dependency 'test_service' not found\" in {result.error}"
            )

    def test_auto_wire_registration_failure(
        self,
        clean_container: FlextContainer,
    ) -> None:
        """Test auto-wiring when registration fails."""
        # Mock a registration failure by using invalid name
        test_service = SampleService("dependency")
        clean_container.register("test_service", test_service)

        # This would require mocking the register method to fail
        # For now, we test the constructor failure path
        class FailingService:
            def __init__(self, test_service: SampleService) -> None:
                msg = "Constructor failed"
                raise ValueError(msg)

        result = clean_container.auto_wire(FailingService)
        assert result.is_failure
        if "Auto-wiring failed" not in (result.error or ""):
            raise AssertionError(f"Expected 'Auto-wiring failed' in {result.error}")

    def test_get_or_create_existing_service(
        self,
        clean_container: FlextContainer,
    ) -> None:
        """Test get_or_create returns existing service."""
        service = SampleService("existing")
        clean_container.register("test", service)

        def factory() -> SampleService:
            return SampleService("new")

        result = clean_container.get_or_create("test", factory)
        assert result.success
        assert result.data is service  # Should return existing

    def test_get_or_create_new_service(self, clean_container: FlextContainer) -> None:
        """Test get_or_create creates new service when not found."""

        def factory() -> SampleService:
            return SampleService("created")

        result = clean_container.get_or_create("test", factory)
        assert result.success
        assert isinstance(result.data, SampleService)
        if result.data.name != "created":
            raise AssertionError(f"Expected {'created'}, got {result.data.name}")

        # Verify service was registered
        assert clean_container.has("test")

    def test_get_or_create_factory_failure(
        self,
        clean_container: FlextContainer,
    ) -> None:
        """Test get_or_create when factory fails."""

        def failing_factory() -> SampleService:
            msg = "Factory failed"
            raise ValueError(msg)

        result = clean_container.get_or_create("test", failing_factory)
        assert result.is_failure
        if "Factory failed for service 'test'" not in (result.error or ""):
            raise AssertionError(
                f"Expected \"Factory failed for service 'test'\" in {result.error}"
            )

    def test_batch_register_success(self, clean_container: FlextContainer) -> None:
        """Test successful batch registration."""
        services = {
            "service1": SampleService("first"),
            "service2": SampleService("second"),
            "service3": SampleService("third"),
        }

        result = clean_container.batch_register(cast("dict[str, object]", services))
        assert result.success
        if result.data != ["service1", "service2", "service3"]:
            raise AssertionError(
                f"Expected {['service1', 'service2', 'service3']}, got {result.data}"
            )

        # Verify all services were registered
        for name in services:
            assert clean_container.has(name)

    def test_batch_register_failure_rollback(
        self,
        clean_container: FlextContainer,
    ) -> None:
        """Test batch registration rollback on failure."""
        services = {
            "service1": SampleService("first"),
            "": SampleService("second"),  # This will fail due to empty name
            "service3": SampleService("third"),
        }

        result = clean_container.batch_register(cast("dict[str, object]", services))
        assert result.is_failure
        if "Batch registration failed" not in (result.error or ""):
            raise AssertionError(
                f"Expected 'Batch registration failed' in {result.error}"
            )

        # Verify rollback occurred - service1 should not be registered
        assert not clean_container.has("service1")
        assert not clean_container.has("service3")

    def test_container_repr(self, clean_container: FlextContainer) -> None:
        """Test container string representation."""
        # Empty container
        repr_str = repr(clean_container)
        if "FlextContainer(services: 0)" not in repr_str:
            raise AssertionError(
                f"Expected {'FlextContainer(services: 0)'} in {repr_str}"
            )

        # Container with services
        clean_container.register("test", SampleService("test"))
        repr_str = repr(clean_container)
        if "FlextContainer(services: 1)" not in repr_str:
            raise AssertionError(
                f"Expected {'FlextContainer(services: 1)'} in {repr_str}"
            )


@pytest.mark.unit
class TestServiceKey:
    """Test ServiceKey functionality."""

    def test_service_key_creation(self) -> None:
        """Test ServiceKey creation and basic functionality."""
        key = ServiceKey[SampleService]("test_service")
        if key.name != "test_service":
            raise AssertionError(f"Expected {'test_service'}, got {key.name}")
        assert str(key) == "test_service"

    def test_service_key_with_register_typed(
        self,
        clean_container: FlextContainer,
    ) -> None:
        """Test ServiceKey with register_typed function."""
        key = ServiceKey[SampleService]("test_service")
        service = SampleService("test")

        result = register_typed(key, service)
        clean_container = get_flext_container()
        assert result.success
        assert clean_container.has("test_service")

    def test_service_key_with_get_typed(self, clean_container: FlextContainer) -> None:
        """Test ServiceKey with get_typed function."""
        key = ServiceKey[SampleService]("test_service")
        service = SampleService("test")
        # Register in global container since get_typed uses global container
        global_container = get_flext_container()
        global_container.register("test_service", service)

        result = get_typed(key, SampleService)
        assert result.success
        assert result.data is service

    def test_get_typed_service_not_found(self, clean_container: FlextContainer) -> None:
        """Test get_typed with ServiceKey for non-existent service."""
        key = ServiceKey[SampleService]("nonexistent")

        result = get_typed(key, SampleService)
        assert result.is_failure
        if "not found" not in (result.error or ""):
            raise AssertionError(f"Expected 'not found' in {result.error}")


@pytest.mark.unit
class TestFlextGlobalContainerManager:
    """Test FlextGlobalContainerManager functionality."""

    def test_manager_initialization(self) -> None:
        """Test manager initializes with None container."""
        manager = FlextGlobalContainerManager()
        assert manager._container is None

    def test_get_container_lazy_creation(self) -> None:
        """Test container lazy creation on first access."""
        manager = FlextGlobalContainerManager()

        # First call creates container
        container1 = manager.get_container()
        assert isinstance(container1, FlextContainer)
        assert manager._container is container1

        # Second call returns same instance
        container2 = manager.get_container()
        assert container2 is container1

    def test_set_container(self) -> None:
        """Test setting custom container."""
        manager = FlextGlobalContainerManager()
        custom_container = FlextContainer()

        manager.set_container(custom_container)
        assert manager._container is custom_container
        assert manager.get_container() is custom_container


@pytest.mark.unit
class TestContainerEdgeCases:
    """Test edge cases and error conditions."""

    def test_container_with_none_service(self, clean_container: FlextContainer) -> None:
        """Test registering None as a service."""
        result = clean_container.register("none_service", None)
        assert result.success

        get_result = clean_container.get("none_service")
        assert get_result.success
        assert get_result.data is None

    def test_container_with_complex_objects(
        self,
        clean_container: FlextContainer,
    ) -> None:
        """Test registering complex objects."""
        complex_service = {
            "nested": {
                "data": [1, 2, 3],
                "config": {"enabled": True},
            },
            "callable": lambda x: x * 2,
        }

        result = clean_container.register("complex", complex_service)
        assert result.success

        get_result = clean_container.get("complex")
        assert get_result.success
        assert get_result.data is complex_service

    def test_factory_with_complex_signature(
        self,
        clean_container: FlextContainer,
    ) -> None:
        """Test factory with complex signature."""

        def complex_factory(*args: object, **kwargs: object) -> ComplexService:
            return ComplexService(*args, **kwargs)

        result = clean_container.register_factory("complex", complex_factory)
        assert result.success

        get_result = clean_container.get("complex")
        assert get_result.success
        assert isinstance(get_result.data, ComplexService)

    def test_service_info_edge_cases(self, clean_container: FlextContainer) -> None:
        """Test service info with edge cases."""

        # Service with complex class hierarchy
        class BaseService:
            pass

        class DerivedService(BaseService):
            pass

        service = DerivedService()
        clean_container.register("derived", service)

        result = clean_container.get_info("derived")
        assert result.success
        assert result.data is not None
        info = result.data
        assert isinstance(info, dict)
        if info["class"] != "DerivedService":
            raise AssertionError(f"Expected {'DerivedService'}, got {info['class']}")

    def test_container_validation_edge_cases(
        self,
        clean_container: FlextContainer,
    ) -> None:
        """Test container with various validation edge cases."""
        # Test with different string names that should be valid or invalid
        test_cases = [
            ("123", False),  # Numeric strings are valid service names
            ("[]", False),  # Special character strings are valid
            ("{}", False),  # Special character strings are valid
            ("", True),  # Empty strings should fail validation
            ("   ", True),  # Whitespace-only strings should fail validation
        ]

        for test_name, should_fail in test_cases:
            result = clean_container.register(test_name, SampleService("test"))
            if should_fail:
                assert result.is_failure
            else:
                assert result.success

    def test_performance_with_many_services(
        self,
        clean_container: FlextContainer,
    ) -> None:
        """Test container performance with many services."""
        # Register many services
        services = {}
        for i in range(100):
            service_name = f"service_{i}"
            service = SampleService(service_name)
            services[service_name] = service
            result = clean_container.register(service_name, service)
            assert result.success

        # Test bulk operations
        if clean_container.get_service_count() != 100:
            raise AssertionError(
                f"Expected {100}, got {clean_container.get_service_count()}"
            )
        service_names = clean_container.get_service_names()
        if len(service_names) != 100:
            raise AssertionError(f"Expected {100}, got {len(service_names)}")

        # Test retrieval performance
        for i in range(0, 100, 10):  # Test every 10th service
            service_name = f"service_{i}"
            get_result = clean_container.get(service_name)
            assert get_result.success
            assert get_result.data is services[service_name]


@pytest.mark.unit
class TestContainerValidationIntegration:
    """Test container integration with validation system."""

    def test_registrar_validation_integration(self) -> None:
        """Test registrar integrates with validation system correctly."""
        registrar = FlextServiceRegistrar()

        # Test the internal validation method directly
        result = registrar._validate_service_name("valid_name")
        assert result.success
        if result.data != "valid_name":
            raise AssertionError(f"Expected {'valid_name'}, got {result.data}")

        # Test invalid name
        result = registrar._validate_service_name("")
        assert result.is_failure

    def test_retriever_validation_integration(self) -> None:
        """Test retriever integrates with validation system correctly."""
        services: dict[str, object] = {}
        factories: dict[str, Callable[[], object]] = {}
        retriever = FlextServiceRetrivier(services, factories)

        # Test the internal validation method directly
        result = retriever._validate_service_name("valid_name")
        assert result.success
        if result.data != "valid_name":
            raise AssertionError(f"Expected {'valid_name'}, got {result.data}")

        # Test invalid name
        result = retriever._validate_service_name("")
        assert result.is_failure


@pytest.mark.integration
class TestContainerSystemIntegration:
    """Integration tests for complete container system."""

    def test_full_lifecycle_integration(self, clean_container: FlextContainer) -> None:
        """Test complete service lifecycle with full system integration."""
        # Register service
        service = SampleService("lifecycle")
        register_result = clean_container.register("lifecycle", service)
        assert register_result.success

        # Get service info
        info_result = clean_container.get_info("lifecycle")
        assert info_result.success
        assert info_result.data is not None
        info_data = info_result.data
        assert isinstance(info_data, dict)
        if info_data["type"] != "instance":
            raise AssertionError(f"Expected {'instance'}, got {info_data['type']}")

        # Retrieve service
        get_result = clean_container.get("lifecycle")
        assert get_result.success
        assert get_result.data is service

        # Type-safe retrieval
        typed_result = clean_container.get_typed("lifecycle", SampleService)
        assert typed_result.success
        assert typed_result.data is service

        # Unregister service
        unregister_result = clean_container.unregister("lifecycle")
        assert unregister_result.success

        # Verify removal
        assert not clean_container.has("lifecycle")

    def test_factory_to_service_conversion_integration(
        self,
        clean_container: FlextContainer,
    ) -> None:
        """Test complete factory to service conversion flow."""
        call_count = 0

        def counting_factory() -> SampleService:
            nonlocal call_count
            call_count += 1
            return SampleService(f"factory_{call_count}")

        # Register factory
        clean_container.register_factory("conversion", counting_factory)

        # Verify it's listed as factory
        services = clean_container.list_services()
        if services["conversion"] != "factory":
            raise AssertionError(f"Expected {'factory'}, got {services['conversion']}")

        # First access converts to service
        result1 = clean_container.get("conversion")
        assert result1.success
        if call_count != 1:
            raise AssertionError(f"Expected {1}, got {call_count}")

        # Verify it's now listed as instance
        services = clean_container.list_services()
        if services["conversion"] != "instance":
            raise AssertionError(f"Expected {'instance'}, got {services['conversion']}")

        # Second access uses cached service
        result2 = clean_container.get("conversion")
        assert result2.success
        assert result2.data is result1.data
        if call_count != 1:
            raise AssertionError(f"Expected {1}, got {call_count}")

        # Get info shows it's now an instance
        info_result = clean_container.get_info("conversion")
        assert info_result.success
        assert info_result.data is not None
        info_data = info_result.data
        assert isinstance(info_data, dict)
        if info_data["type"] != "instance":
            raise AssertionError(f"Expected {'instance'}, got {info_data['type']}")

    def test_global_container_full_integration(self) -> None:
        """Test global container management with full integration."""
        # Get initial global container
        container1 = get_flext_container()

        # Register service in global container
        service = SampleService("global")
        container1.register("global_service", service)

        # Verify service persists across global access
        container2 = get_flext_container()
        assert container2 is container1
        assert container2.has("global_service")

        # Configure new global container
        custom_container = FlextContainer()
        result_container = configure_flext_container(custom_container)
        assert result_container is custom_container

        # Verify new container is now global
        container3 = get_flext_container()
        assert container3 is custom_container
        assert not container3.has("global_service")  # New container is clean

        # Reset to auto-created container
        new_container = configure_flext_container(None)
        assert new_container is not custom_container
        assert get_flext_container() is new_container
