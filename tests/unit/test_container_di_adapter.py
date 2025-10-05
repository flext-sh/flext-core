"""Tests for FlextContainer DI Adapter Layer (v1.1.0).

This test module verifies the internal dependency-injector integration while
maintaining 100% backward compatibility with the existing FlextContainer API.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Any, cast

from flext_core import FlextContainer, FlextResult


class TestDIContainerInitialization:
    """Test internal DI container initialization and setup."""

    def test_di_container_exists(self) -> None:
        """Verify internal _di_container is initialized."""
        container = FlextContainer()

        # Internal DI container should exist
        assert hasattr(container, "_di_container")
        # DI container should be a dependency_injector DynamicContainer
        assert container._di_container.__class__.__name__ == "DynamicContainer"

    def test_tracking_dicts_exist(self) -> None:
        """Verify backward compatibility tracking dicts exist."""
        container = FlextContainer()

        # Compatibility dicts must exist
        assert hasattr(container, "_services")
        assert hasattr(container, "_factories")
        assert isinstance(container._services, dict)
        assert isinstance(container._factories, dict)

    def test_config_sync_on_init(self) -> None:
        """Verify FlextConfig is synced to DI container on initialization."""
        container = FlextContainer()

        # DI container should have config provider
        assert hasattr(container._di_container, "config")

        # Config should be a Configuration provider
        assert container._di_container.config.__class__.__name__ == "Configuration"


class TestServiceRegistrationSync:
    """Test service registration syncs to both tracking dict and DI container."""

    def test_register_service_dual_storage(self) -> None:
        """Service registration stores in both dict and DI container."""
        container = FlextContainer()
        test_service = {"value": "test"}

        # Register service
        result = container.register("test_service", test_service)
        assert result.is_success

        # Verify stored in tracking dict
        assert "test_service" in container._services
        assert container._services["test_service"] is test_service

        # Verify stored in DI container
        assert hasattr(container._di_container, "test_service")

        # Verify DI provider returns same instance (Singleton pattern)
        provider = getattr(container._di_container, "test_service")
        assert provider() is test_service

    def test_register_factory_dual_storage(self) -> None:
        """Factory registration stores in both dict and DI container."""
        container = FlextContainer()
        factory_calls: list[int] = []

        def test_factory() -> dict[str, int]:
            factory_calls.append(1)
            return {"instance": len(factory_calls)}

        # Register factory
        result = container.register_factory("test_factory", test_factory)
        assert result.is_success

        # Verify stored in tracking dict
        assert "test_factory" in container._factories
        assert container._factories["test_factory"] is test_factory

        # Verify stored in DI container
        assert hasattr(container._di_container, "test_factory")

        # Verify DI provider caches factory result (lazy singleton pattern)
        provider = getattr(container._di_container, "test_factory")
        instance1 = provider()
        instance2 = provider()
        assert instance1 is instance2  # Same instance (cached)
        assert len(factory_calls) == 1  # Factory called once, then cached

    def test_duplicate_registration_fails(self) -> None:
        """Duplicate service registration fails gracefully."""
        container = FlextContainer()

        # First registration succeeds
        result1 = container.register("service", {"value": 1})
        assert result1.is_success

        # Duplicate registration fails
        result2 = container.register("service", {"value": 2})
        assert result2.is_failure
        assert (
            result2.error is not None and "already registered" in result2.error.lower()
        )

        # Original service unchanged
        assert container._services["service"] == {"value": 1}


class TestServiceResolutionSync:
    """Test service resolution via DI container with FlextResult wrapping."""

    def test_get_service_via_di(self) -> None:
        """Service retrieval resolves via DI container."""
        container = FlextContainer()
        test_service = {"value": "test"}

        container.register("test_service", test_service)

        # Get service via FlextResult API
        result = container.get("test_service")
        assert result.is_success
        assert result.value is test_service

    def test_get_factory_result_via_di(self) -> None:
        """Factory result is cached after first call (lazy singleton pattern)."""
        container = FlextContainer()
        instance_count: list[int] = []

        def factory() -> dict[str, int]:
            instance_count.append(1)
            return {"instance": len(instance_count)}

        container.register_factory("factory", factory)

        # First retrieval - factory is called
        result1 = container.get("factory")
        assert result1.is_success
        instance1 = result1.value
        assert cast("dict[str, int]", instance1)["instance"] == 1

        # Second retrieval - cached result returned
        result2 = container.get("factory")
        assert result2.is_success
        instance2 = result2.value

        # Same instance (factory result cached)
        assert instance1 is instance2
        assert cast("dict[str, int]", instance2)["instance"] == 1  # Same cached result
        assert len(instance_count) == 1  # Factory only called once

    def test_get_nonexistent_service_fails(self) -> None:
        """Retrieving nonexistent service fails with FlextResult."""
        container = FlextContainer()

        result = container.get("nonexistent")
        assert result.is_failure
        assert result.error is not None and "not found" in result.error.lower()


class TestServiceUnregistrationSync:
    """Test service unregistration from both tracking dict and DI container."""

    def test_unregister_removes_from_both(self) -> None:
        """Unregistration removes from both dict and DI container."""
        container = FlextContainer()
        test_service = {"value": "test"}

        # Register and verify
        container.register("test_service", test_service)
        assert "test_service" in container._services
        assert hasattr(container._di_container, "test_service")

        # Unregister
        result = container.unregister("test_service")
        assert result.is_success

        # Verify removed from tracking dict
        assert "test_service" not in container._services

        # Verify removed from DI container (best-effort)
        # Note: DI container removal is best-effort, so we just check service
        # retrieval fails
        get_result = container.get("test_service")
        assert get_result.is_failure

    def test_unregister_factory_removes_from_both(self) -> None:
        """Factory unregistration removes from both storages."""
        container = FlextContainer()

        def factory() -> dict[str, str]:
            return {"value": "test"}

        # Register and verify
        container.register_factory("test_factory", factory)
        assert "test_factory" in container._factories
        assert hasattr(container._di_container, "test_factory")

        # Unregister
        result = container.unregister("test_factory")
        assert result.is_success

        # Verify removed from tracking dicts
        assert "test_factory" not in container._factories

        # Verify service no longer accessible
        get_result = container.get("test_factory")
        assert get_result.is_failure

    def test_unregister_nonexistent_fails(self) -> None:
        """Unregistering nonexistent service fails gracefully."""
        container = FlextContainer()

        result = container.unregister("nonexistent")
        assert result.is_failure
        assert result.error is not None and "not registered" in result.error.lower()


class TestFlextConfigSync:
    """Test FlextConfig synchronization with DI container."""

    def test_config_values_synced(self) -> None:
        """FlextConfig values are synced to DI Configuration provider."""
        container = FlextContainer()

        # Access the config provider
        config_provider = cast("Any", container._di_container.config)

        # Verify config values are synced
        # These should match FlextConfig defaults
        assert config_provider.environment() in {
            "production",
            "development",
            "staging",
            "testing",
        }
        assert isinstance(config_provider.debug(), bool)
        assert isinstance(config_provider.log_level(), str)

    def test_config_provider_type(self) -> None:
        """DI config provider is Configuration type."""
        container = FlextContainer()

        assert hasattr(container._di_container, "config")
        assert container._di_container.config.__class__.__name__ == "Configuration"


class TestFlextResultWrapping:
    """Test that all DI operations are wrapped in FlextResult."""

    def test_register_returns_flext_result(self) -> None:
        """register() returns FlextResult[None]."""
        container = FlextContainer()

        result = container.register("service", {"value": "test"})
        assert isinstance(result, FlextResult)
        assert result.is_success

    def test_register_factory_returns_flext_result(self) -> None:
        """register_factory() returns FlextResult[None]."""
        container = FlextContainer()

        result = container.register_factory("factory", lambda: {"value": "test"})
        assert isinstance(result, FlextResult)
        assert result.is_success

    def test_get_returns_flext_result(self) -> None:
        """get() returns FlextResult[object]."""
        container = FlextContainer()
        container.register("service", {"value": "test"})

        result = container.get("service")
        assert isinstance(result, FlextResult)
        assert result.is_success
        assert result.value == {"value": "test"}

    def test_unregister_returns_flext_result(self) -> None:
        """unregister() returns FlextResult[None]."""
        container = FlextContainer()
        container.register("service", {"value": "test"})

        result = container.unregister("service")
        assert isinstance(result, FlextResult)
        assert result.is_success

    def test_error_cases_return_flext_result(self) -> None:
        """Error cases return FlextResult with failure status."""
        container = FlextContainer()

        # Duplicate registration
        container.register("service", {"value": 1})
        result = container.register("service", {"value": 2})
        assert isinstance(result, FlextResult)
        assert result.is_failure

        # Nonexistent service
        result = container.get("nonexistent")
        assert isinstance(result, FlextResult)
        assert result.is_failure

        # Unregister nonexistent
        result = container.unregister("nonexistent")
        assert isinstance(result, FlextResult)
        assert result.is_failure


class TestExceptionTranslation:
    """Test that DI exceptions are translated to FlextResult failures."""

    def test_di_error_wrapped_in_result(self) -> None:
        """DI container errors are caught and wrapped in FlextResult."""
        container = FlextContainer()

        # Try to register non-callable as factory
        result = container.register_factory("bad_factory", "not_callable")
        assert result.is_failure
        assert result.error is not None and "must be callable" in result.error.lower()

    def test_resolution_error_wrapped(self) -> None:
        """Service resolution errors are wrapped in FlextResult."""
        container = FlextContainer()

        # Try to get nonexistent service
        result = container.get("nonexistent")
        assert result.is_failure
        assert isinstance(result.error, str)


class TestBackwardCompatibility:
    """Test that internal DI doesn't break existing behavior."""

    def test_has_method_still_works(self) -> None:
        """has() method works with DI adapter."""
        container = FlextContainer()

        assert not container.has("service")

        container.register("service", {"value": "test"})
        assert container.has("service")

        container.unregister("service")
        assert not container.has("service")

    def test_list_services_still_works(self) -> None:
        """list_services() returns correct list with DI adapter."""
        container = FlextContainer()

        # Get initial list
        result = container.list_services()
        assert result.is_success
        assert len(result.value) == 0

        # Add services
        container.register("service1", {"value": 1})
        container.register("service2", {"value": 2})

        result = container.list_services()
        assert result.is_success
        services = result.value

        # list_services returns list of service metadata dicts
        service_names = [cast("dict[str, str]", s)["name"] for s in services]
        assert "service1" in service_names
        assert "service2" in service_names
        assert len(services) == 2

    def test_clear_removes_all(self) -> None:
        """clear() removes all services from both storages."""
        container = FlextContainer()

        # Add multiple services
        container.register("service1", {"value": 1})
        container.register("service2", {"value": 2})
        container.register_factory("factory1", lambda: {"value": 3})

        # Clear all
        result = container.clear()
        assert result.is_success

        # Verify all removed
        list_result = container.list_services()
        assert list_result.is_success
        assert len(list_result.value) == 0
        assert len(container._services) == 0
        assert len(container._factories) == 0


class TestContainerReset:
    """Tests for container reset behavior via clear()."""

    def test_clear_resets_di_state(self) -> None:
        """Services are no longer resolvable after clear()."""

        container = FlextContainer()
        service_data = {"value": "to be cleared"}

        register_result = container.register("clearable_service", service_data)
        assert register_result.is_success

        # Sanity check service resolves prior to clear
        resolve_result = container.get("clearable_service")
        assert resolve_result.is_success and resolve_result.value is service_data

        clear_result = container.clear()
        assert clear_result.is_success

        post_clear_result = container.get("clearable_service")
        assert post_clear_result.is_failure
