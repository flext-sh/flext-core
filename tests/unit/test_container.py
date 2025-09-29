"""Comprehensive tests for FlextContainer - Dependency Injection Container.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import FlextContainer


class TestFlextContainer:
    """Test suite for FlextContainer dependency injection."""

    def test_container_initialization(self) -> None:
        """Test container initialization."""
        container = FlextContainer()
        assert container is not None
        assert isinstance(container, FlextContainer)

    def test_container_register_service(self) -> None:
        """Test service registration."""
        container = FlextContainer()

        class TestService:
            def __init__(self) -> None:
                self.value = "test"

        service_instance = TestService()
        result = container.register("test_service", service_instance)
        assert result.is_success

    def test_container_create_service(self) -> None:
        """Test service creation with dependency injection."""
        container = FlextContainer()

        # Register a dependency
        class Dependency:
            def __init__(self) -> None:
                self.name = "dependency"

        class Service:
            def __init__(self, dependency: Dependency) -> None:
                self.dependency = dependency
                self.initialized = True

        dependency = Dependency()
        container.register("dependency", dependency)

        # Create service with dependency injection
        result = container.create_service(Service, "test_service")
        assert result.is_success
        service = result.value
        assert service is not None
        assert service.initialized
        assert service.dependency is dependency

    def test_container_auto_wire(self) -> None:
        """Test auto-wiring without registration."""
        container = FlextContainer()

        # Register a dependency
        class Dependency:
            def __init__(self) -> None:
                self.name = "dependency"

        class Service:
            def __init__(self, dependency: Dependency) -> None:
                self.dependency = dependency
                self.initialized = True

        dependency = Dependency()
        container.register("dependency", dependency)

        # Auto-wire service (creates instance but doesn't register)
        result = container.auto_wire(Service)
        assert result.is_success
        service = result.value
        assert service is not None
        assert service.initialized
        assert service.dependency is dependency

        # Service should not be registered
        assert not container.has("service")

    def test_container_auto_wire_missing_dependency(self) -> None:
        """Test auto-wiring with missing dependency."""
        container = FlextContainer()

        class Service:
            def __init__(self, _missing_dep: object) -> None:
                self.initialized = True

        # Auto-wire service with missing dependency
        result = container.auto_wire(Service)
        assert result.is_failure
        assert result.error is not None
        assert "Cannot resolve required dependency" in result.error

    def test_container_auto_wire_with_defaults(self) -> None:
        """Test auto-wiring with default parameters."""
        container = FlextContainer()

        class Service:
            def __init__(self, optional: str = "default") -> None:
                self.optional = optional
                self.initialized = True

        # Auto-wire service with default parameter
        result = container.auto_wire(Service)
        assert result.is_success
        service = result.value
        assert service is not None
        assert service.initialized
        assert service.optional == "default"

    def test_container_configure(self) -> None:
        """Test container configuration."""
        container = FlextContainer()

        # Configure container
        config: dict[str, object] = {
            "max_workers": 8,
            "timeout_seconds": 60.0,
            "environment": "testing",
        }
        result = container.configure(config)
        assert result.is_success

        # Check configuration was applied
        container_config = container.get_config()
        assert container_config["max_workers"] == 8
        assert container_config["timeout_seconds"] == 60.0
        assert container_config["environment"] == "testing"

    def test_container_configure_invalid_keys(self) -> None:
        """Test container configuration with invalid keys."""
        container = FlextContainer()

        # Configure with invalid keys (should be ignored)
        config: dict[str, object] = {"invalid_key": "value", "max_workers": 4}
        result = container.configure(config)
        assert result.is_success

        # Valid key should be applied
        container_config = container.get_config()
        assert container_config["max_workers"] == 4

    def test_container_batch_register(self) -> None:
        """Test batch service registration."""
        container = FlextContainer()

        services: dict[str, object] = {
            "service1": {"key": "value1"},
            "service2": {"key": "value2"},
            "service3": {"key": "value3"},
        }

        result = container.batch_register(services)
        assert result.is_success

        # All services should be registered
        for name in services:
            assert container.has(name)
            get_result = container.get(name)
            assert get_result.is_success

    def test_container_batch_register_with_duplicate(self) -> None:
        """Test batch registration with duplicate service."""
        container = FlextContainer()

        # Pre-register a service
        container.register("service1", {"key": "original"})

        services: dict[str, object] = {
            "service1": {"key": "duplicate"},  # This should fail
            "service2": {"key": "value2"},
        }

        result = container.batch_register(services)
        assert result.is_failure
        assert result.error is not None
        assert "already registered" in result.error

        # Original service should still be there
        get_result = container.get("service1")
        assert get_result.is_success
        assert isinstance(get_result.value, dict)
        assert get_result.value["key"] == "original"

    def test_container_batch_register_empty(self) -> None:
        """Test batch registration with empty dict."""
        container = FlextContainer()

        result = container.batch_register({})
        assert result.is_success

    def test_container_get_typed(self) -> None:
        """Test typed service retrieval."""
        container = FlextContainer()

        class TestService:
            def __init__(self) -> None:
                self.value = "test"

        service = TestService()
        container.register("test_service", service)

        # Get with correct type
        result = container.get_typed("test_service", TestService)
        assert result.is_success
        assert isinstance(result.value, TestService)
        assert result.value.value == "test"

    def test_container_get_typed_wrong_type(self) -> None:
        """Test typed service retrieval with wrong type."""
        container = FlextContainer()

        service = {"key": "value"}
        container.register("test_service", service)

        # Get with wrong type
        result = container.get_typed("test_service", dict)
        assert result.is_success  # Should succeed because dict is the correct type

        # Get with wrong type
        class WrongType:
            pass

        result = container.get_typed("test_service", WrongType)
        assert result.is_failure
        assert result.error is not None
        assert "type mismatch" in result.error

    def test_container_get_or_create(self) -> None:
        """Test get or create service."""
        container = FlextContainer()

        def factory() -> dict[str, str]:
            return {"created": "by_factory"}

        # Service doesn't exist, should create using factory
        result = container.get_or_create("test_service", factory)
        assert result.is_success
        assert isinstance(result.value, dict)
        assert result.value["created"] == "by_factory"

        # Service now exists, should return existing
        result2 = container.get_or_create("test_service", factory)
        assert result2.is_success
        assert result2.value is result.value  # Same instance

    def test_container_get_or_create_no_factory(self) -> None:
        """Test get or create without factory."""
        container = FlextContainer()

        # Service doesn't exist and no factory provided
        result = container.get_or_create("nonexistent")
        assert result.is_failure
        assert result.error is not None
        assert "not found and no factory provided" in result.error

    def test_container_list_services(self) -> None:
        """Test listing services."""
        container = FlextContainer()

        # Register different types of services
        container.register("instance", {"type": "instance"})
        container.register_factory("factory", lambda: {"type": "factory"})

        result = container.list_services()
        assert result.is_success

        services = result.value
        assert len(services) == 2

        # Check service info
        service_names = [s["name"] for s in services]
        assert "instance" in service_names
        assert "factory" in service_names

        service_types = [s["type"] for s in services]
        assert "instance" in service_types
        assert "factory" in service_types

    def test_container_get_service_names(self) -> None:
        """Test getting service names."""
        container = FlextContainer()

        container.register("service1", "value1")
        container.register("service2", "value2")

        result = container.get_service_names()
        assert result.is_success

        names = result.value
        assert set(names) == {"service1", "service2"}

    def test_container_get_info(self) -> None:
        """Test getting container info."""
        container = FlextContainer()

        container.register("test", "value")

        result = container.get_info()
        assert result.is_success

        info = result.value
        assert "service_count" in info
        assert "direct_services" in info
        assert "factories" in info
        assert "configuration" in info
        assert info["service_count"] == 1
        assert info["direct_services"] == 1

    def test_container_register_factory_invalid(self) -> None:
        """Test registering invalid factory."""
        container = FlextContainer()

        # Register non-callable as factory (string instead of function)
        result = container.register_factory("invalid", "this is not callable")
        assert result.is_failure
        assert result.error is not None
        assert "must be callable" in result.error

    def test_container_get_service_with_factory_caching(self) -> None:
        """Test that factory services are cached after first creation."""
        container = FlextContainer()

        call_count = 0

        def counting_factory() -> dict[str, str]:
            nonlocal call_count
            call_count += 1
            return {"call_count": str(call_count)}

        container.register_factory("cached_service", counting_factory)

        # First call should create the service
        result1 = container.get("cached_service")
        assert result1.is_success
        assert isinstance(result1.value, dict)
        assert result1.value["call_count"] == "1"
        assert call_count == 1

        # Second call should return cached instance
        result2 = container.get("cached_service")
        assert result2.is_success
        assert isinstance(result2.value, dict)
        assert result2.value["call_count"] == "1"  # Same instance, not recreated
        assert call_count == 1  # Factory not called again

    def test_container_clear(self) -> None:
        """Test clearing all services."""
        container = FlextContainer()

        container.register("service1", "value1")
        container.register_factory("service2", lambda: "value2")

        assert container.has("service1")
        assert container.has("service2")

        result = container.clear()
        assert result.is_success

        assert not container.has("service1")
        assert not container.has("service2")
        assert container.get_service_count() == 0

    def test_container_register_invalid_name(self) -> None:
        """Test registering with invalid service name."""
        container = FlextContainer()

        result = container.register("", "service")
        assert result.is_failure
        assert result.error is not None
        assert "empty" in result.error

        result = container.register("invalid/name", "service")
        assert result.is_failure
        assert result.error is not None
        assert "invalid characters" in result.error

    def test_container_unregister_nonexistent(self) -> None:
        """Test unregistering non-existent service."""
        container = FlextContainer()

        result = container.unregister("nonexistent")
        assert result.is_failure
        assert result.error is not None
        assert "not registered" in result.error

    def test_container_get_nonexistent_typed(self) -> None:
        """Test getting non-existent service with typing."""
        container = FlextContainer()

        result = container.get_typed("nonexistent", dict)
        assert result.is_failure
        assert result.error is not None
        assert "not found" in result.error

    def test_container_has_invalid_name(self) -> None:
        """Test has() with invalid service name."""
        container = FlextContainer()

        # Should return False for invalid names, not raise exception
        assert not container.has("")
        assert not container.has("invalid/name")

    def test_container_create_service_with_custom_name(self) -> None:
        """Test creating service with custom name."""
        container = FlextContainer()

        class Service:
            def __init__(self) -> None:
                self.initialized = True

        result = container.create_service(Service, "custom_name")
        assert result.is_success

        # Service should be registered with custom name
        assert container.has("custom_name")
        get_result = container.get("custom_name")
        assert get_result.is_success

    def test_container_create_service_missing_dependency(self) -> None:
        """Test creating service with missing dependency."""
        container = FlextContainer()

        class Service:
            def __init__(self, _missing: object) -> None:
                self.initialized = True

        result = container.create_service(Service)
        assert result.is_failure
        assert result.error is not None
        assert "Cannot resolve required dependency" in result.error

    def test_container_create_service_with_defaults(self) -> None:
        """Test creating service with default parameters."""
        container = FlextContainer()

        class Service:
            def __init__(self, optional: str = "default") -> None:
                self.optional = optional
                self.initialized = True

        result = container.create_service(Service)
        assert result.is_success
        service = result.value
        assert service.initialized
        assert service.optional == "default"

    def test_container_error_handling_in_batch_operations(self) -> None:
        """Test error handling in batch operations."""
        container = FlextContainer()

        # Test batch register with invalid service name first
        services: dict[str, object] = {
            "": {"key": "empty_name"},  # Invalid name first
            "valid": {"key": "value"},
            "valid2": {"key": "value2"},
        }

        result = container.batch_register(services)
        assert result.is_failure
        assert result.error is not None
        assert "empty" in result.error

        # Since batch operation fails fast, no services should be registered
        # when the first service has an invalid name
        assert not container.has("")
        assert not container.has("valid")
        assert not container.has("valid2")

    def test_container_exception_handling_in_service_creation(self) -> None:
        """Test exception handling during service creation."""
        container = FlextContainer()

        class FailingService:
            def __init__(self) -> None:
                error_msg = "Service creation failed"
                raise ValueError(error_msg)

        container.register("other_service", "some_value")

        result = container.create_service(FailingService)
        assert result.is_failure
        assert result.error is not None
        assert "Service creation failed" in result.error

    def test_container_get_service_with_exception_in_factory(self) -> None:
        """Test exception handling when factory throws exception."""
        container = FlextContainer()

        def failing_factory() -> dict[str, str]:
            error_msg = "Factory failed"
            raise RuntimeError(error_msg)

        container.register_factory("failing", failing_factory)

        result = container.get("failing")
        assert result.is_failure
        assert result.error is not None
        assert "Factory 'failing' failed" in result.error

    def test_container_global_singleton_functionality(self) -> None:
        """Test global singleton container functionality."""
        # Test that get_global() returns the same instance
        container1 = FlextContainer.get_global()
        container2 = FlextContainer.get_global()
        assert container1 is container2

        # Test global registration
        result = FlextContainer.register_global("global_service", "global_value")
        assert result.is_success

        # Should be accessible from global container
        global_container = FlextContainer.get_global()
        get_result = global_container.get("global_service")
        assert get_result.is_success
        assert get_result.value == "global_value"

    def test_container_configure_global(self) -> None:
        """Test global container configuration."""
        # Configure global container
        config: dict[str, object] = {"max_workers": 16, "timeout_seconds": 120.0}
        result = FlextContainer.configure_global(config)
        assert result.is_success

        # Check that global container has the configuration
        global_container = FlextContainer.get_global()
        container_config = global_container.get_config()
        assert container_config["max_workers"] == 16
        assert container_config["timeout_seconds"] == 120.0

    def test_container_get_global_typed(self) -> None:
        """Test global typed service retrieval."""
        # Register service in global container
        service = {"type": "test"}
        FlextContainer.register_global("typed_service", service)

        # Get with typing
        result = FlextContainer.get_global_typed("typed_service", dict)
        assert result.is_success
        assert isinstance(result.value, dict)
        assert result.value["type"] == "test"

    def test_container_module_utilities(self) -> None:
        """Test module utilities creation."""
        result = FlextContainer.create_module_utilities("test_module")
        assert result.is_success

        utilities = result.value
        assert utilities["module"] == "test_module"
        assert utilities["logger"] == "flext.test_module"
        assert "container" in utilities

    def test_container_repr(self) -> None:
        """Test container string representation."""
        container = FlextContainer()
        container.register("test", "value")

        repr_str = repr(container)
        assert "FlextContainer" in repr_str
        assert "services=1" in repr_str
        assert "factories=0" in repr_str
        assert "total_registered=1" in repr_str

    def test_container_register_factory_duplicate_name(self) -> None:
        """Test registering factory with duplicate name fails."""
        container = FlextContainer()

        def factory1() -> dict[str, str]:
            return {"factory": "1"}

        def factory2() -> dict[str, str]:
            return {"factory": "2"}

        # Register first factory
        result1 = container.register_factory("test_factory", factory1)
        assert result1.is_success

        # Try to register second factory with same name
        result2 = container.register_factory("test_factory", factory2)
        assert result2.is_failure
        assert result2.error is not None
        assert "already registered" in result2.error
