"""Dependency injection container."""

from __future__ import annotations

import inspect
from collections import UserString
from collections.abc import Callable
from typing import Generic, Self, TypeVar

from flext_core.constants import SERVICE_NAME_EMPTY
from flext_core.exceptions import FlextError
from flext_core.mixins import FlextLoggableMixin
from flext_core.result import FlextResult
from flext_core.typings import T
from flext_core.validation import flext_validate_service_name

TService = TypeVar("TService")


class FlextServiceKey(UserString, Generic[TService]):  # noqa: UP046
    """Typed service key for type-safe service resolution.

    A specialized string that acts as a plain string at runtime but provides type safety
    at type-check time. This enables type-safe service registration and retrieval in the
    FLEXT dependency injection container while maintaining runtime string support.

    The key supports generic subscription like FlextServiceKey[DatabaseService]("db")
    to provide compile-time type checking without runtime overhead.

    Attributes:
      data: The underlying string value of the service key.
      name: Alias for the string value (convenience property).

    Example:
      Type-safe service key usage:

      >>> key = FlextServiceKey[DatabaseService]("database")
      >>> print(key)  # Acts like a string
      'database'
      >>> print(key.name)
      'database'

    """

    __slots__ = ()

    def __new__(cls, name: str) -> Self:
        """Create a new service key."""
        instance = object.__new__(cls)
        instance.data = str(name)
        return instance

    # Convenience: test access key.name
    @property
    def name(self) -> str:
        """Return the service key name (string value)."""
        return str(self)

    @classmethod
    def __class_getitem__(cls, _item: object) -> type[FlextServiceKey]:  # type: ignore[type-arg]
        """Support generic subscription without affecting runtime behavior."""
        return cls


# =============================================================================
# SERVICE REGISTRATION - SRP: Registration Operations Only
# =============================================================================


class FlextServiceRegistrar(FlextLoggableMixin):
    """Service registration component implementing Single Responsibility Principle.

    This class handles all service and factory registration operations with validation
    and error handling. It's separated from service retrieval to follow SRP and
    provide clear separation of concerns in the dependency injection system.

    The registrar validates service names, prevents duplicate registrations, and
    maintains both direct service instances and factory functions for lazy instantiation.

    Attributes:
      _services: Internal registry mapping service names to instances.
      _factories: Internal registry mapping service names to factory functions.

    Example:
      Direct usage for service registration:

      >>> registrar = FlextServiceRegistrar()
      >>> result = registrar.register_service("logger", logger_instance)
      >>> print(result.is_success)
      True

      Factory registration for lazy instantiation:

      >>> factory_result = registrar.register_factory("db", lambda: DatabaseService())
      >>> print(factory_result.is_success)
      True

    """

    def __init__(self) -> None:
        """Initialize service registrar with empty registry."""
        super().__init__()
        self._services: dict[str, object] = {}
        self._factories: dict[str, Callable[[], object]] = {}

    @staticmethod
    def _validate_service_name(name: str) -> FlextResult[str]:
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

        # Verify factory signature to ensure it can be called without parameters
        try:
            sig = inspect.signature(factory)
            required_params = sum(
                1
                for p in sig.parameters.values()
                if p.default == p.empty
                and p.kind not in {p.VAR_POSITIONAL, p.VAR_KEYWORD}
            )
            if required_params > 0:
                return FlextResult.fail(
                    f"Factory must be callable without parameters, but requires {required_params} parameter(s)",
                )
        except (ValueError, TypeError, OSError) as e:
            return FlextResult.fail(f"Could not inspect factory signature: {e}")

        if validated_name in self._services:
            del self._services[validated_name]

        # Safe assignment after signature verification
        factory_callable = factory
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
        """Check if a service exists."""
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
        super().__init__()
        self._services = services
        self._factories = factories

    @staticmethod
    def _validate_service_name(name: str) -> FlextResult[str]:
        """Validate service name.

        Args:
            name: Service name to validate.

        Returns:
            Result with validated name or error.

        """
        if not flext_validate_service_name(name):
            return FlextResult.fail(SERVICE_NAME_EMPTY)
        return FlextResult.ok(name)

    def get_service(self, name: str) -> FlextResult[object]:
        """Retrieve a registered service - Performance optimized."""
        if not name or not isinstance(name, str) or not name.strip():
            return FlextResult.fail(SERVICE_NAME_EMPTY)

        validated_name = name.strip()

        # Check direct service registration first (the most common case)
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

    def get_service_info(self, name: str) -> FlextResult[dict[str, object]]:
        """Get detailed information about a registered service or factory.

        Args:
            name: Service name to get info for.

        Returns:
            FlextResult containing service info dictionary.

        """
        if not name or not isinstance(name, str) or not name.strip():
            return FlextResult.fail(SERVICE_NAME_EMPTY)

        validated_name = name.strip()

        # Check if a service is registered as instance
        if validated_name in self._services:
            service = self._services[validated_name]
            service_class = service.__class__
            return FlextResult.ok(
                {
                    "name": validated_name,
                    "type": "instance",
                    "class": service_class.__name__,
                    "module": service_class.__module__,
                },
            )

        # Check if a service is registered as factory
        if validated_name in self._factories:
            factory = self._factories[validated_name]
            factory_name = getattr(factory, "__name__", str(factory))
            factory_module = getattr(factory, "__module__", "unknown")
            return FlextResult.ok(
                {
                    "name": validated_name,
                    "type": "factory",
                    "factory": factory_name,
                    "module": factory_module,
                },
            )

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
        super().__init__()
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
        """Check if a service exists."""
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

    # -----------------------------------------------------------------
    # Convenience methods expected by tests and other projects
    # -----------------------------------------------------------------

    def get_info(self, name: str) -> FlextResult[dict[str, object]]:
        """Return basic info about a registered service or factory."""
        try:
            if name in self._registrar.get_services_dict():
                service = self._registrar.get_services_dict()[name]
                service_info: dict[str, object] = {
                    "name": name,
                    "kind": "instance",
                    "type": "instance",
                    "class": type(service).__name__,
                }
                return FlextResult.ok(service_info)
            if name in self._registrar.get_factories_dict():
                factory = self._registrar.get_factories_dict()[name]
                factory_info: dict[str, object] = {
                    "name": name,
                    "kind": "factory",
                    "type": "factory",
                    "class": type(factory).__name__,
                }
                return FlextResult.ok(factory_info)
            return FlextResult.fail(f"Service '{name}' not found")
        except (KeyError, AttributeError, TypeError) as e:
            return FlextResult.fail(f"Info retrieval failed: {e}")

    def get_or_create(
        self,
        name: str,
        factory: Callable[[], object],
    ) -> FlextResult[object]:
        """Get existing service or register-and-return a newly created one."""
        existing = self.get(name)
        if existing.is_success:
            return existing
        try:
            # Register factory and immediately resolve
            reg = self.register_factory(name, factory)
            if reg.is_failure:
                return FlextResult.fail(reg.error or "Factory registration failed")

            # Try to get the service - if factory fails, customize the error message
            service_result = self.get(name)
            if service_result.is_failure:
                error = service_result.error or ""
                if "Factory for" in error and "failed:" in error:
                    # Transform "Factory for 'test' failed: Factory failed" to
                    # "Factory failed for service 'test'"
                    return FlextResult.fail(f"Factory failed for service '{name}'")
                return service_result

            return service_result
        except (TypeError, ValueError, AttributeError, RuntimeError) as e:
            return FlextResult.fail(f"get_or_create failed: {e}")

    def auto_wire(
        self,
        service_class: type[T],
        service_name: str | None = None,
    ) -> FlextResult[T]:
        """Auto-wire service dependencies and register the service.

        Args:
            service_class: Class to instantiate with auto-wired dependencies.
            service_name: Optional service name (default to class name).

        Returns:
            FlextResult containing the instantiated and registered service.

        """
        try:
            # Use class name as default service name
            name = service_name or service_class.__name__

            # Get constructor signature
            sig = inspect.signature(service_class.__init__)
            args: list[object] = []
            kwargs: dict[str, object] = {}

            # Skip 'self' parameter
            params = list(sig.parameters.values())[1:]

            for param in params:
                if param.default is not inspect.Parameter.empty:
                    # Parameter has default, skip dependency resolution
                    continue

                # Try to resolve dependency from registered services
                dependency_result = self.get(param.name)
                if dependency_result.is_failure:
                    return FlextResult.fail(
                        f"Required dependency '{param.name}' not found for "
                        f"{service_class.__name__}",
                    )

                kwargs[param.name] = dependency_result.unwrap()

            # Instantiate service with resolved dependencies
            service_instance = service_class(*args, **kwargs)

            # Register the service
            register_result = self.register(name, service_instance)
            if register_result.is_failure:
                return FlextResult.fail(
                    f"Auto-wiring failed during registration: {register_result.error}",
                )

            return FlextResult.ok(service_instance)

        except (TypeError, ValueError, AttributeError, RuntimeError, OSError) as e:
            return FlextResult.fail(f"Auto-wiring failed: {e}")

    def batch_register(
        self,
        registrations: dict[str, object],
    ) -> FlextResult[list[str]]:
        """Register many services/factories atomically; roll back on failure."""
        # Snapshot current state for rollback
        services_snapshot = dict(self._registrar.get_services_dict())
        factories_snapshot = dict(self._registrar.get_factories_dict())
        registered_names = []
        try:
            for key, value in registrations.items():
                if callable(value):
                    result = self.register_factory(key, value)
                else:
                    result = self.register(key, value)
                if result.is_failure:
                    # Rollback
                    self._registrar.get_services_dict().clear()
                    self._registrar.get_services_dict().update(services_snapshot)
                    self._registrar.get_factories_dict().clear()
                    self._registrar.get_factories_dict().update(factories_snapshot)
                    return FlextResult.fail("Batch registration failed")
                registered_names.append(key)
            return FlextResult.ok(registered_names)
        except (TypeError, ValueError, AttributeError, RuntimeError, KeyError) as e:
            # Rollback on unexpected errors
            self._registrar.get_services_dict().clear()
            self._registrar.get_services_dict().update(services_snapshot)
            self._registrar.get_factories_dict().clear()
            self._registrar.get_factories_dict().update(factories_snapshot)
            return FlextResult.fail(f"Batch registration crashed: {e}")


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
        """Get or create a global container.

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


def get_typed[T](
    key: FlextServiceKey[T] | str,
    expected_type: type[T],
) -> FlextResult[T]:
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


def register_typed[T](key: FlextServiceKey[T] | str, service: T) -> FlextResult[None]:
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
# MODULE UTILITIES FACTORY - DRY helpers for per-module containers
# =============================================================================


def create_module_container_utilities(module_name: str) -> dict[str, object]:
    """Create standardized container helpers for a module namespace.

    Provides three utilities:
    - get_container: returns the global `FlextContainer`
    - configure_dependencies: no-op configurator returning `FlextResult[None]`
    (modules may monkey-patch/replace this later to register services)
    - get_service(name): fetches a service by name, with fallback to
    "{module_name}.{name}" to support namespaced registrations

    Args:
      module_name: Logical module namespace used for fallback lookups.

    Returns:
      Mapping with utility callables.

    """

    def _get_container() -> FlextContainer:
        return get_flext_container()

    def _configure_dependencies() -> FlextResult[None]:
        # Intentionally a no-op default. Modules can replace this function
        # to perform actual registrations when needed.
        return FlextResult.ok(None)

    def _get_service(name: str) -> FlextResult[object]:
        container = get_flext_container()
        direct = container.get(name)
        if direct.is_success:
            return direct
        # Fallback to namespaced key
        return container.get(f"{module_name}.{name}")

    return {
        "get_container": _get_container,
        "configure_dependencies": _configure_dependencies,
        "get_service": _get_service,
    }


# Export API (after aliases for static analyzers)
__all__: list[str] = [
    "FlextContainer",
    "FlextContainerUtils",
    "FlextServiceKey",
    "FlextServiceKey",  # Service key alias
    "configure_flext_container",
    "create_module_container_utilities",
    "get_flext_container",
    "get_typed",
    "register_typed",
]
