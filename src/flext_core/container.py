"""Dependency injection container for service management.

Provides type-safe service registration and retrieval with factory support.
Implements dependency inversion principle for Clean Architecture.

Classes:
    FlextServiceRegistrar: Service registration component.
    FlextServiceLocator: Service retrieval component.
    FlextContainer: Main DI container combining both.

Functions:
    get_flext_container: Get global container instance.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Self, cast

from flext_core.constants import SERVICE_NAME_EMPTY
from flext_core.exceptions import FlextError
from flext_core.mixins import FlextLoggableMixin
from flext_core.result import FlextResult
from flext_core.validation import flext_validate_service_name

if TYPE_CHECKING:
    from collections.abc import Callable

    from flext_core.typings import T


class FlextServiceKey(str):
    """Typed service key supporting generic subscription at type-check time.

    Acts as a plain ``str`` at runtime while enabling usages like
    ``FlextServiceKey[T]("name")`` in type-checked code.
    """

    __slots__ = ()

    def __new__(cls, name: str) -> Self:
        """Create new service key."""
        return str.__new__(cls, str(name))

    @classmethod
    def __class_getitem__(cls, _item: object) -> type[FlextServiceKey]:
        """Support generic subscription without affecting runtime behavior."""
        return cls


# Backward-compat typed alias form used by tests
class _ServiceKeyAlias:
    """Runtime alias enabling ``ServiceKey[T](name)`` syntax in tests.

    Returns a ``FlextServiceKey`` when called. Supports subscription
    without doing anything at runtime.
    """

    @classmethod
    def __class_getitem__(cls, _item: object) -> _ServiceKeyAlias:
        return cls()

    def __call__(self, name: str) -> FlextServiceKey:
        return FlextServiceKey(name)


# =============================================================================
# SERVICE REGISTRATION - SRP: Registration Operations Only
# =============================================================================


class FlextServiceRegistrar(FlextLoggableMixin):
    """Service registration component.

    Handles service and factory registration with validation.
    Implements single responsibility principle.
    """

    def __init__(self) -> None:
        """Initialize service registrar with empty registry."""
        self._services: dict[str, object] = {}
        self._factories: dict[str, Callable[[], object]] = {}

    def _validate_service_name(self, name: str) -> FlextResult[str]:
        """Validate service name.

        Args:
            name: Service name to validate.

        Returns:
            Result with validated name or error.

        """
        if not flext_validate_service_name(name):
            return FlextResult.fail(SERVICE_NAME_EMPTY)
        return FlextResult.ok(name)

    def register_service(self, name: str, service: object) -> FlextResult[None]:
        """Register service instance.

        Args:
            name: Service identifier.
            service: Service instance.

        Returns:
            Result indicating success or failure.

        """
        # Fast path: simple validation without FlextResult overhead
        if not name or not isinstance(name, str) or not name.strip():
            return FlextResult.fail(SERVICE_NAME_EMPTY)

        validated_name = name.strip()

        # Only log warnings for actual problems (not every operation)
        if validated_name in self._services:
            self.logger.warning(
                "Service already registered, replacing",
                name=validated_name,
                existing_service_type=type(self._services[validated_name]).__name__,
                new_service_type=type(service).__name__,
            )

        self._services[validated_name] = service
        self.logger.debug(
            "Service registered",
            name=validated_name,
            service_type=type(service).__name__,
            service_id=id(service),
            total_services=len(self._services),
        )
        return FlextResult.ok(None)

    def register_factory(
        self,
        name: str,
        factory: object,
    ) -> FlextResult[None]:
        """Register service factory.

        Args:
            name: Service identifier.
            factory: Callable that creates service.

        Returns:
            Result indicating success or failure.

        """
        validation_result = self._validate_service_name(name)
        if validation_result.is_failure:
            return validation_result.map(lambda _: None)

        validated_name = validation_result.unwrap()

        if not callable(factory):
            return FlextResult.fail("Factory must be callable")

        if validated_name in self._services:
            del self._services[validated_name]

        factory_callable = cast("Callable[[], object]", factory)
        self._factories[validated_name] = factory_callable
        self.logger.debug(
            "Factory registered",
            name=validated_name,
            factory_type=type(factory).__name__,
            total_factories=len(self._factories),
        )
        return FlextResult.ok(None)

    def unregister_service(self, name: str) -> FlextResult[None]:
        """Unregister a service."""
        validation_result = self._validate_service_name(name)
        if validation_result.is_failure:
            return validation_result.map(lambda _: None)

        validated_name = validation_result.unwrap()

        if validated_name in self._services:
            del self._services[validated_name]
            return FlextResult.ok(None)

        if validated_name in self._factories:
            del self._factories[validated_name]
            return FlextResult.ok(None)

        return FlextResult.fail(f"Service '{validated_name}' not found")

    def clear_all(self) -> FlextResult[None]:
        """Clear all registered services and factories."""
        self._services.clear()
        self._factories.clear()
        return FlextResult.ok(None)

    def get_service_names(self) -> list[str]:
        """Get all registered service names."""
        return list(self._services.keys()) + list(self._factories.keys())

    def get_service_count(self) -> int:
        """Get total service count."""
        return len(self._services) + len(self._factories)

    def has_service(self, name: str) -> bool:
        """Check if service exists."""
        return name in self._services or name in self._factories

    def get_services_dict(self) -> dict[str, object]:
        """Get services dictionary (internal use)."""
        return self._services

    def get_factories_dict(self) -> dict[str, Callable[[], object]]:
        """Get factories dictionary (internal use)."""
        return self._factories


# =============================================================================
# SERVICE RETRIEVAL - SRP: Retrieval Operations Only
# =============================================================================


class FlextServiceRetriever(FlextLoggableMixin):
    """Service retrieval component implementing single responsibility principle."""

    def __init__(
        self,
        services: dict[str, object],
        factories: dict[str, Callable[[], object]],
    ) -> None:
        """Initialize service retriever with references."""
        self._services = services
        self._factories = factories

    def get_service(self, name: str) -> FlextResult[object]:
        """Retrieve a registered service - Performance optimized."""
        if not name or not isinstance(name, str) or not name.strip():
            return FlextResult.fail(SERVICE_NAME_EMPTY)

        validated_name = name.strip()

        # Check direct service registration first (most common case)
        if validated_name in self._services:
            return FlextResult.ok(self._services[validated_name])

        # Check factory registration
        if validated_name in self._factories:
            try:
                factory = self._factories[validated_name]
                service = factory()

                # Cache the factory result as a service for singleton behavior
                self._services[validated_name] = service
                # Remove from factories since it's now cached as a service
                del self._factories[validated_name]

                return FlextResult.ok(service)
            except (
                TypeError,
                ValueError,
                AttributeError,
                RuntimeError,
                FlextError,
            ) as e:
                return FlextResult.fail(f"Factory for '{validated_name}' failed: {e!s}")

        return FlextResult.fail(f"Service '{validated_name}' not found")

    def list_services(self) -> dict[str, str]:
        """List all services with their types."""
        services_info = {}

        for name in self._services:
            services_info[name] = "instance"

        for name in self._factories:
            services_info[name] = "factory"

        return services_info


# =============================================================================
# MAIN CONTAINER - SRP: Public API Orchestration
# =============================================================================


class FlextContainer(FlextLoggableMixin):
    """Enterprise dependency injection container with SOLID principles."""

    def __init__(self) -> None:
        """Initialize container with internal components."""
        # Call mixin setup first
        self.mixin_setup()
        self.logger.info("Initializing FlextContainer")

        # SRP: Delegate to focused internal components
        self._registrar = FlextServiceRegistrar()

        # DIP: Retriever depends on registrar's data abstractions
        services_dict = self._registrar.get_services_dict()
        factories_dict = self._registrar.get_factories_dict()
        self._retriever = FlextServiceRetriever(services_dict, factories_dict)

        self.logger.debug("FlextContainer initialized successfully")

    # Registration API - Delegate to registrar
    def register(self, name: str, service: object) -> FlextResult[None]:
        """Register a service instance."""
        return self._registrar.register_service(name, service)

    def register_factory(
        self,
        name: str,
        factory: Callable[[], object],
    ) -> FlextResult[None]:
        """Register a service factory."""
        return self._registrar.register_factory(name, factory)

    def unregister(self, name: str) -> FlextResult[None]:
        """Unregister service.

        Args:
            name: Service identifier.

        Returns:
            Result indicating success or failure.

        """
        return self._registrar.unregister_service(name)

    # Retrieval API - Delegate to retriever
    def get(self, name: str) -> FlextResult[object]:
        """Get a service by name."""
        return self._retriever.get_service(name)

    # Container management
    def clear(self) -> FlextResult[None]:
        """Clear all services."""
        return self._registrar.clear_all()

    def has(self, name: str) -> bool:
        """Check if service exists."""
        return self._registrar.has_service(name)

    def list_services(self) -> dict[str, str]:
        """List all services."""
        return self._retriever.list_services()

    def get_service_names(self) -> list[str]:
        """Get service names."""
        return self._registrar.get_service_names()

    def get_service_count(self) -> int:
        """Get service count."""
        return self._registrar.get_service_count()

    # Type-safe retrieval methods
    def get_typed(self, name: str, expected_type: type[T]) -> FlextResult[T]:
        """Get service with type checking - Performance optimized."""
        result = self.get(name)
        if result.is_failure:
            return FlextResult.fail(result.error or "Service not found")

        service = result.unwrap()

        # Simple isinstance check instead of complex FlextTypes.TypeGuards
        if not isinstance(service, expected_type):
            actual_type = type(service).__name__
            return FlextResult.fail(
                f"Service '{name}' is {actual_type}, expected {expected_type.__name__}",
            )

        # MyPy knows service is T after isinstance check
        return FlextResult.ok(service)

    def __repr__(self) -> str:
        """Return string representation of container."""
        count = self.get_service_count()
        return f"FlextContainer(services: {count})"


# =============================================================================
# GLOBAL CONTAINER MANAGEMENT
# =============================================================================


class FlextGlobalContainerManager(FlextLoggableMixin):
    """Thread-safe global container manager.

    Manages singleton instance of FlextContainer.
    """

    def __init__(self) -> None:
        super().__init__()
        self.mixin_setup()
        self._container: FlextContainer | None = None

    def get_container(self) -> FlextContainer:
        """Get or create global container.

        Returns:
            Global container instance.

        """
        if self._container is None:
            self.logger.info("Creating global FlextContainer instance")
            self._container = FlextContainer()
        return self._container

    def set_container(self, container: FlextContainer) -> None:
        """Set global container.

        Args:
            container: Container to set as global.

        """
        self.logger.info(
            "Configuring global FlextContainer",
            new_container=str(container),
        )
        self._container = container


# Global container manager instance
class _SimpleGlobalManager:
    """Simple global container manager."""

    def __init__(self) -> None:
        """Initialize with default container."""
        self._container = FlextContainer()

    def get_container(self) -> FlextContainer:
        """Get container."""
        return self._container

    def set_container(self, container: FlextContainer) -> None:
        """Set container."""
        self._container = container


_global_manager = _SimpleGlobalManager()


# =============================================================================
# FLEXT CONTAINER UTILS - Static class for container utility functions
# =============================================================================


class FlextContainerUtils:
    """Container utility functions."""

    @staticmethod
    def get_flext_container() -> FlextContainer:
        """Get global container instance.

        Returns:
            Global container instance.

        """
        return _global_manager.get_container()

    @staticmethod
    def configure_flext_container(container: FlextContainer | None) -> FlextContainer:
        """Configure global container.

        Args:
            container: Container to configure.

        Returns:
            Configured container.

        """
        if container is None:
            container = FlextContainer()
        _global_manager.set_container(container)
        return container


# Convenience functions for direct access
def get_flext_container() -> FlextContainer:
    """Get global container instance.

    Returns:
        Global container instance.

    """
    return FlextContainerUtils.get_flext_container()


def configure_flext_container(container: FlextContainer | None) -> FlextContainer:
    """Configure global container (convenience function).

    Args:
        container: Container to configure or None for new container.

    Returns:
        Configured container.

    """
    return FlextContainerUtils.configure_flext_container(container)


def get_typed(key: FlextServiceKey | str, expected_type: type[T]) -> FlextResult[T]:
    """Get typed service from global container.

    Args:
        key: Service identifier.
        expected_type: Expected service type.

    Returns:
        Type-safe result with service.

    """
    container = get_flext_container()
    # Convert FlextServiceKey to str for container.get_typed
    key_str = str(key)
    return container.get_typed(key_str, expected_type)


def register_typed(key: FlextServiceKey | str, service: T) -> FlextResult[None]:
    """Register service in global container.

    Args:
        key: Service identifier.
        service: Service instance.

    Returns:
        Result indicating success or failure.

    """
    container = get_flext_container()
    # Convert FlextServiceKey to str for container.register
    key_str = str(key)
    return container.register(key_str, service)


# =============================================================================
# LEGACY ALIASES - Backward compatibility
# =============================================================================

# Legacy alias for backward compatibility with existing tests (typo preserved)
FlextServiceRetrivier = FlextServiceRetriever
# Backward compatibility alias for ServiceKey
ServiceKey = _ServiceKeyAlias()


# Export API (after aliases for static analyzers)
__all__: list[str] = [
    "FlextContainer",
    "FlextContainerUtils",
    "FlextServiceKey",
    "ServiceKey",  # Backward compatibility alias
    "configure_flext_container",
    "get_flext_container",
    "get_typed",
    "register_typed",
]
