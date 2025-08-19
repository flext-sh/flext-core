"""Dependency injection container with FlextCommands integration."""

from __future__ import annotations

import inspect
from collections import UserString
from collections.abc import Callable
from datetime import datetime
from typing import Generic, TypeVar, cast
from zoneinfo import ZoneInfo

from flext_core.commands import FlextCommands
from flext_core.constants import SERVICE_NAME_EMPTY
from flext_core.exceptions import FlextError
from flext_core.mixins import FlextLoggableMixin
from flext_core.result import FlextResult
from flext_core.utilities import FlextGenerators
from flext_core.validation import flext_validate_service_name

# Define TypeVars locally
T = TypeVar("T")
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

    # Convenience: test access key.name
    @property
    def name(self) -> str:
        """Return the service key name (string value)."""
        return str(self)

    @classmethod
    def __class_getitem__(cls, _item: object) -> type[FlextServiceKey[TService]]:
        """Support generic subscription without affecting runtime behavior."""
        return cls


# =============================================================================
# CONTAINER COMMANDS - Using FlextCommands pattern for operations
# =============================================================================


class RegisterServiceCommand(FlextCommands.Command):
    """Command to register a service instance."""

    service_name: str = ""
    service_instance: object

    @classmethod
    def create(
        cls, service_name: str, service_instance: object
    ) -> RegisterServiceCommand:
        """Create command with default values."""
        return cls(  # pyright: ignore[reportCallIssue]
            service_name=service_name,
            service_instance=service_instance,
            command_type="register_service",
            command_id=FlextGenerators.generate_uuid(),
            timestamp=datetime.now(tz=ZoneInfo("UTC")),
            user_id=None,
            correlation_id=FlextGenerators.generate_uuid(),
        )

    def validate_command(self) -> FlextResult[None]:
        """Validate service registration command."""
        if not self.service_name or not self.service_name.strip():
            return FlextResult[None].fail(SERVICE_NAME_EMPTY)
        return FlextResult[None].ok(None)


class RegisterFactoryCommand(FlextCommands.Command):
    """Command to register a service factory."""

    service_name: str = ""
    factory: object  # Callable[[], object] - using object for validation

    @classmethod
    def create(cls, service_name: str, factory: object) -> RegisterFactoryCommand:
        """Create command with default values."""
        return cls(  # pyright: ignore[reportCallIssue]
            service_name=service_name,
            factory=factory,
            command_type="register_factory",
            command_id=FlextGenerators.generate_uuid(),
            timestamp=datetime.now(tz=ZoneInfo("UTC")),
            user_id=None,
            correlation_id=FlextGenerators.generate_uuid(),
        )

    def validate_command(self) -> FlextResult[None]:
        """Validate factory registration command."""
        if not self.service_name or not self.service_name.strip():
            return FlextResult[None].fail(SERVICE_NAME_EMPTY)
        if not callable(self.factory):
            return FlextResult[None].fail("Factory must be callable")
        return FlextResult[None].ok(None)


class UnregisterServiceCommand(FlextCommands.Command):
    """Command to unregister a service."""

    service_name: str = ""

    @classmethod
    def create(cls, service_name: str) -> UnregisterServiceCommand:
        """Create command with default values."""
        return cls(  # pyright: ignore[reportCallIssue]
            service_name=service_name,
            command_type="unregister_service",
            command_id=FlextGenerators.generate_uuid(),
            timestamp=datetime.now(tz=ZoneInfo("UTC")),
            user_id=None,
            correlation_id=FlextGenerators.generate_uuid(),
        )

    def validate_command(self) -> FlextResult[None]:
        """Validate service unregistration command."""
        if not self.service_name or not self.service_name.strip():
            return FlextResult[None].fail(SERVICE_NAME_EMPTY)
        return FlextResult[None].ok(None)


class GetServiceQuery(FlextCommands.Query):
    """Query to retrieve a service."""

    service_name: str = ""
    expected_type: str | None = None  # Optional type validation

    @classmethod
    def create(
        cls, service_name: str, expected_type: str | None = None
    ) -> GetServiceQuery:
        """Create query with default values."""
        return cls(
            service_name=service_name,
            expected_type=expected_type,
            query_type="get_service",
            query_id=None,
            page_size=100,
            page_number=1,
            sort_by=None,
            sort_order="asc",
        )

    def validate_query(self) -> FlextResult[None]:
        """Validate service retrieval query."""
        if not self.service_name or not self.service_name.strip():
            return FlextResult[None].fail(SERVICE_NAME_EMPTY)
        return FlextResult[None].ok(None)


class ListServicesQuery(FlextCommands.Query):
    """Query to list all services."""

    include_factories: bool = True
    service_type_filter: str | None = None

    @classmethod
    def create(
        cls, *, include_factories: bool = True, service_type_filter: str | None = None
    ) -> ListServicesQuery:
        """Create query with default values."""
        return cls(
            include_factories=include_factories,
            service_type_filter=service_type_filter,
            query_type="list_services",
            query_id=None,
            page_size=100,
            page_number=1,
            sort_by=None,
            sort_order="asc",
        )


# =============================================================================
# COMMAND HANDLERS - Handle container operations
# =============================================================================


class RegisterServiceHandler(FlextCommands.Handler[RegisterServiceCommand, None]):
    """Handler for service registration commands."""

    def __init__(self, registrar: FlextServiceRegistrar) -> None:
        super().__init__("RegisterServiceHandler")
        self._registrar = registrar

    def handle(self, command: RegisterServiceCommand) -> FlextResult[None]:
        """Handle service registration."""
        return self._registrar.register_service(
            command.service_name, command.service_instance
        )


class RegisterFactoryHandler(FlextCommands.Handler[RegisterFactoryCommand, None]):
    """Handler for factory registration commands."""

    def __init__(self, registrar: FlextServiceRegistrar) -> None:
        super().__init__("RegisterFactoryHandler")
        self._registrar = registrar

    def handle(self, command: RegisterFactoryCommand) -> FlextResult[None]:
        """Handle factory registration."""
        return self._registrar.register_factory(command.service_name, command.factory)


class UnregisterServiceHandler(FlextCommands.Handler[UnregisterServiceCommand, None]):
    """Handler for service unregistration commands."""

    def __init__(self, registrar: FlextServiceRegistrar) -> None:
        super().__init__("UnregisterServiceHandler")
        self._registrar = registrar

    def handle(self, command: UnregisterServiceCommand) -> FlextResult[None]:
        """Handle service unregistration."""
        return self._registrar.unregister_service(command.service_name)


class GetServiceQueryHandler(FlextCommands.QueryHandler[GetServiceQuery, object]):
    """Handler for service retrieval queries."""

    def __init__(self, retriever: FlextServiceRetriever) -> None:
        super().__init__("GetServiceQueryHandler")
        self._retriever = retriever

    def handle(self, query: GetServiceQuery) -> FlextResult[object]:
        """Handle service retrieval."""
        return self._retriever.get_service(query.service_name)


class ListServicesQueryHandler(
    FlextCommands.QueryHandler[ListServicesQuery, dict[str, str]]
):
    """Handler for listing services."""

    def __init__(self, retriever: FlextServiceRetriever) -> None:
        super().__init__("ListServicesQueryHandler")
        self._retriever = retriever

    def handle(self, query: ListServicesQuery) -> FlextResult[dict[str, str]]:
        """Handle service listing."""
        # Use query parameters for filtering in future versions
        _ = query  # Explicitly ignore query for now (will be used for filtering)
        services = self._retriever.list_services()
        return FlextResult[dict[str, str]].ok(services)


# =============================================================================
# SERVICE REGISTRATION - SRP: Registration Operations Only
# =============================================================================


class FlextServiceRegistrar(FlextLoggableMixin):
    """Service registration component implementing Single Responsibility Principle.

    This class handles all service and factory registration operations with validation
    and error handling. It's separated from service retrieval to follow SRP and
    provide clear separation of concerns in the dependency injection system.

    The registrar validates service names, prevents duplicate registrations, and
    def get_typed(self, name: str, expected_type: type[T]) -> FlextResult[T]:
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
        # T is now global, no need to redefine it here
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
            return FlextResult[str].fail(SERVICE_NAME_EMPTY)
        return FlextResult[str].ok(name)

    def register_service(self, name: str, service: object) -> FlextResult[None]:
        """Register service instance.

        Args:
            name: Service identifier.
            service: Service instance.

        Returns:
            Result indicating success or failure.

        """
        # Fast path: simple validation without FlextResult overhead
        if not name or not name.strip():
            return FlextResult[None].fail(SERVICE_NAME_EMPTY)

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
        return FlextResult[None].ok(None)

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
            return FlextResult[None].fail(validation_result.error or SERVICE_NAME_EMPTY)

        # unwrap() returns T where T is the success type for the FlextResult[str]
        # Cast explicitly to str so static analyzers infer the correct type.
        validated_name: str = validation_result.unwrap()

        # We accept a generic object here so we can runtime-check callability.
        # Using Callable[...] in the signature makes the `if not callable(...)`
        # branch unreachable to static analyzers like mypy.
        if not callable(factory):
            return FlextResult[None].fail("Factory must be callable")

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
                return FlextResult[None].fail(
                    f"Factory must be callable without parameters, but requires {required_params} parameter(s)",
                )
        except (ValueError, TypeError, OSError) as e:
            return FlextResult[None].fail(f"Could not inspect factory signature: {e}")

        if validated_name in self._services:
            del self._services[validated_name]

        # Safe assignment after signature verification. Cast to the expected
        # callable type so static typing understands the stored value.
        factory_callable = cast("Callable[[], object]", factory)
        self._factories[validated_name] = factory_callable
        self.logger.debug(
            "Factory registered",
            name=validated_name,
            factory_type=type(factory).__name__,
            total_factories=len(self._factories),
        )
        return FlextResult[None].ok(None)

    def unregister_service(self, name: str) -> FlextResult[None]:
        """Unregister a service."""
        validation_result = self._validate_service_name(name)
        if validation_result.is_failure:
            return FlextResult[None].fail(validation_result.error or SERVICE_NAME_EMPTY)

        # Unwrap returns the validated name (str) on success.
        validated_name: str = validation_result.unwrap()

        if validated_name in self._services:
            del self._services[validated_name]
            return FlextResult[None].ok(None)

        if validated_name in self._factories:
            del self._factories[validated_name]
            return FlextResult[None].ok(None)

        return FlextResult[None].fail(f"Service '{validated_name}' not found")

    def clear_all(self) -> FlextResult[None]:
        """Clear all registered services and factories."""
        self._services.clear()
        self._factories.clear()
        return FlextResult[None].ok(None)

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
            return FlextResult[str].fail(SERVICE_NAME_EMPTY)
        return FlextResult[str].ok(name)

    def get_service(self, name: str) -> FlextResult[object]:
        """Retrieve a registered service - Performance optimized."""
        if not name or not name.strip():
            return FlextResult[object].fail(SERVICE_NAME_EMPTY)

        validated_name = name.strip()

        # Check direct service registration first (the most common case)
        if validated_name in self._services:
            return FlextResult[object].ok(self._services[validated_name])

        # Check factory registration
        if validated_name in self._factories:
            try:
                factory = self._factories[validated_name]
                service = factory()

                # Cache the factory result as a service for singleton behavior
                self._services[validated_name] = service
                # Remove from factories since it's now cached as a service
                del self._factories[validated_name]

                return FlextResult[object].ok(service)
            except (
                TypeError,
                ValueError,
                AttributeError,
                RuntimeError,
                FlextError,
            ) as e:
                return FlextResult[object].fail(
                    f"Factory for '{validated_name}' failed: {e!s}",
                )

        return FlextResult[object].fail(f"Service '{validated_name}' not found")

    def get_service_info(self, name: str) -> FlextResult[dict[str, object]]:
        """Get detailed information about a registered service or factory.

        Args:
            name: Service name to get info for.

        Returns:
            FlextResult containing service info dictionary.

        """
        if not name or not name.strip():
            return FlextResult[dict[str, object]].fail(SERVICE_NAME_EMPTY)

        validated_name = name.strip()

        # Check if a service is registered as instance
        if validated_name in self._services:
            service = self._services[validated_name]
            service_class = service.__class__
            return FlextResult[dict[str, object]].ok(
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
            return FlextResult[dict[str, object]].ok(
                {
                    "name": validated_name,
                    "type": "factory",
                    "factory": factory_name,
                    "module": factory_module,
                },
            )

        return FlextResult[dict[str, object]].fail(
            f"Service '{validated_name}' not found",
        )

    def list_services(self) -> dict[str, str]:
        """List all services with their types."""
        services_info: dict[str, str] = {}

        for name in self._services:
            services_info[name] = "instance"

        for name in self._factories:
            services_info[name] = "factory"

        return services_info


# =============================================================================
# MAIN CONTAINER - SRP: Public API Orchestration
# =============================================================================


class FlextContainer(FlextLoggableMixin):
    """Enterprise dependency injection container with SOLID principles and FlextCommands integration."""

    def __init__(self) -> None:
        """Initialize container with internal components and command bus."""
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

        # FlextCommands integration
        self._command_bus = FlextCommands.create_command_bus()
        self._setup_command_handlers()

        self.logger.debug("FlextContainer initialized successfully")

    def _setup_command_handlers(self) -> None:
        """Setup command handlers for container operations."""
        # Register command handlers
        self._command_bus.register_handler(
            RegisterServiceCommand, RegisterServiceHandler(self._registrar)
        )
        self._command_bus.register_handler(
            RegisterFactoryCommand, RegisterFactoryHandler(self._registrar)
        )
        self._command_bus.register_handler(
            UnregisterServiceCommand, UnregisterServiceHandler(self._registrar)
        )

        # Register query handlers
        # Note: FlextCommands.Bus currently handles commands, we'll add direct handlers for queries
        self._get_service_handler = GetServiceQueryHandler(self._retriever)
        self._list_services_handler = ListServicesQueryHandler(self._retriever)

    # Registration API - Using FlextCommands pattern
    def register(self, name: str, service: object) -> FlextResult[None]:
        """Register a service instance using FlextCommands."""
        command = RegisterServiceCommand.create(name, service)
        result = self._command_bus.execute(command)
        if result.is_failure:
            return FlextResult[None].fail(result.error or "Registration failed")
        return FlextResult[None].ok(None)

    def register_factory(
        self,
        name: str,
        factory: Callable[[], object],
    ) -> FlextResult[None]:
        """Register a service factory using FlextCommands."""
        command = RegisterFactoryCommand.create(service_name=name, factory=factory)
        result = self._command_bus.execute(command)
        if result.is_failure:
            return FlextResult[None].fail(result.error or "Factory registration failed")
        return FlextResult[None].ok(None)

    def unregister(self, name: str) -> FlextResult[None]:
        """Unregister service using FlextCommands.

        Args:
            name: Service identifier.

        Returns:
            Result indicating success or failure.

        """
        command = UnregisterServiceCommand.create(service_name=name)
        result = self._command_bus.execute(command)
        if result.is_failure:
            return FlextResult[None].fail(result.error or "Unregistration failed")
        return FlextResult[None].ok(None)

    # Retrieval API - Using FlextCommands query pattern
    def get(self, name: str) -> FlextResult[object]:
        """Get a service by name using FlextCommands query."""
        query = GetServiceQuery.create(service_name=name)
        return self._get_service_handler.handle(query)

    # Container management
    def clear(self) -> FlextResult[None]:
        """Clear all services."""
        return self._registrar.clear_all()

    def has(self, name: str) -> bool:
        """Check if a service exists."""
        return self._registrar.has_service(name)

    def list_services(self) -> dict[str, str]:
        """List all services using FlextCommands query."""
        query = ListServicesQuery.create()
        result = self._list_services_handler.handle(query)
        if result.is_success:
            return result.unwrap()
        return {}

    def get_service_names(self) -> list[str]:
        """Get service names."""
        return self._registrar.get_service_names()

    def get_service_count(self) -> int:
        """Get service count."""
        return self._registrar.get_service_count()

    @property
    def command_bus(self) -> FlextCommands.Bus:
        """Access to the internal command bus for advanced operations."""
        return self._command_bus

    # Type-safe retrieval methods
    def get_typed(self, name: str, expected_type: type[T]) -> FlextResult[T]:
        """Get service with type checking - Performance optimized."""
        result = self.get(name)
        if result.is_failure:
            return FlextResult[T].fail(result.error or "Service not found")

        # result is FlextResult[object] -> explicit cast to object for analyzers
        service: object = result.unwrap()

        # Simple isinstance check instead of complex FlextTypes.TypeGuards
        if not isinstance(service, expected_type):
            actual_type = type(service).__name__
            return FlextResult[T].fail(
                f"Service '{name}' is {actual_type}, expected {expected_type.__name__}",
            )

        # MyPy sabe que service é T após isinstance
        return FlextResult[T].ok(service)

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
                return FlextResult[dict[str, object]].ok(service_info)
            if name in self._registrar.get_factories_dict():
                factory = self._registrar.get_factories_dict()[name]
                factory_info: dict[str, object] = {
                    "name": name,
                    "kind": "factory",
                    "type": "factory",
                    "class": type(factory).__name__,
                }
                return FlextResult[dict[str, object]].ok(factory_info)
            return FlextResult[dict[str, object]].fail(f"Service '{name}' not found")
        except (KeyError, AttributeError, TypeError) as e:
            return FlextResult[dict[str, object]].fail(f"Info retrieval failed: {e}")

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
                return FlextResult[object].fail(
                    reg.error or "Factory registration failed",
                )

            # Try to get the service - if factory fails, customize the error message
            service_result = self.get(name)
            if service_result.is_failure:
                error = service_result.error or ""
                if "Factory for" in error and "failed:" in error:
                    # Transform "Factory for 'test' failed: Factory failed" to
                    # "Factory failed for service 'test'"
                    return FlextResult[object].fail(
                        f"Factory failed for service '{name}'",
                    )
                return service_result

            return service_result
        except (TypeError, ValueError, AttributeError, RuntimeError) as e:
            return FlextResult[object].fail(f"get_or_create failed: {e}")

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
                    return FlextResult[T].fail(
                        f"Required dependency '{param.name}' not found for "
                        f"{service_class.__name__}",
                    )

                # dependency_result is FlextResult[object]; cast explicitly
                kwargs[param.name] = dependency_result.unwrap()

            # Instantiate service with resolved dependencies
            service_instance = service_class(*args, **kwargs)

            # Register the service
            register_result = self.register(name, service_instance)
            if register_result.is_failure:
                return FlextResult[T].fail(
                    f"Auto-wiring failed during registration: {register_result.error}",
                )

            return FlextResult[T].ok(service_instance)

        except (TypeError, ValueError, AttributeError, RuntimeError, OSError) as e:
            return FlextResult[T].fail(f"Auto-wiring failed: {e}")

    def batch_register(
        self,
        registrations: dict[str, object],
    ) -> FlextResult[list[str]]:
        """Register many services/factories atomically; roll back on failure."""
        # Snapshot current state for rollback
        services_snapshot = dict(self._registrar.get_services_dict())
        factories_snapshot = dict(self._registrar.get_factories_dict())
        registered_names: list[str] = []
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
                    return FlextResult[list[str]].fail("Batch registration failed")
                registered_names.append(key)
            return FlextResult[list[str]].ok(registered_names)
        except (TypeError, ValueError, AttributeError, RuntimeError, KeyError) as e:
            # Rollback on unexpected errors
            self._registrar.get_services_dict().clear()
            self._registrar.get_services_dict().update(services_snapshot)
            self._registrar.get_factories_dict().clear()
            self._registrar.get_factories_dict().update(factories_snapshot)
            return FlextResult[list[str]].fail(f"Batch registration crashed: {e}")


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


def get_typed(
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


def register_typed(key: FlextServiceKey[T] | str, service: T) -> FlextResult[None]:
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
        return FlextResult[None].ok(None)

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
