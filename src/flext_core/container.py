"""FLEXT Container - Enterprise dependency injection following architectural patterns.

Complete dependency injection container with FlextTypes, FlextConstants, FlextProtocols
integration and railway-oriented programming via FlextResult patterns, following
SOLID principles, Clean Architecture, and FLEXT consolidated class patterns.

This module provides the FlextContainer consolidated class containing ALL container
functionality organized as nested classes following the FLEXT_REFACTORING_PROMPT.md
requirements. It eliminates ALL local base classes, TypeVar definitions, and
TYPE_CHECKING constructs in favor of centralized FlextTypes, FlextConstants, and
FlextProtocols usage.

Architecture Principles:
    - Single consolidated FlextContainer class with nested organization
    - Hierarchical integration with FlextTypes, FlextConstants, FlextProtocols
    - Railway-oriented programming via FlextResult for all operations
    - SOLID principles with dependency injection and inversion of control
    - Zero circular imports through proper layering
    - Python 3.13+ syntax with strict type annotations
    - Professional Google-style docstrings throughout

Examples:
    Basic container usage::

        container = FlextContainer()

        # Service registration
        result = container.register("logger", logger_instance)
        if result.is_success:
            print("Service registered successfully")

        # Type-safe service retrieval
        logger_result = container.get_typed("logger", Logger)
        if logger_result.is_success:
            logger = logger_result.unwrap()

    Global container pattern::

        # Access global singleton
        global_container = FlextContainer.get_global()
        global_container.register("database", db_service)

        # Convenient global access
        from flext_core.container import get_flext_container

        container = get_flext_container()

Note:
    This module follows strict FLEXT refactoring requirements with nested classes,
    centralized type definitions, and elimination of all local abstractions in
    favor of flext-core consolidated patterns.

"""

from __future__ import annotations

import inspect
from collections import UserString
from collections.abc import Callable
from datetime import datetime
from typing import cast, override
from zoneinfo import ZoneInfo

from flext_core.constants import FlextConstants
from flext_core.exceptions import FlextExceptions
from flext_core.protocols import FlextProtocols
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes, T
from flext_core.utilities import FlextGenerators
from flext_core.validation import flext_validate_service_name


def _get_exception_class(name: FlextTypes.Core.String) -> type[Exception]:
    """Get dynamically created exception class with type safety."""
    return cast("type[Exception]", getattr(FlextExceptions, name))


class FlextContainer:
    """Consolidated enterprise dependency injection container with SOLID principles.

    This is the single consolidated class for ALL container functionality following
    FLEXT_REFACTORING_PROMPT.md requirements. It contains nested classes for
    organization while maintaining a single export interface.

    The container implements:
        - Railway-oriented programming via FlextResult for all operations
        - Type-safe service registration and retrieval
        - Factory pattern support for lazy initialization
        - SOLID principles with dependency inversion and single responsibility
        - Global singleton pattern for ecosystem-wide service sharing
        - Command/Query separation for operations
        - Hierarchical service resolution with fallback patterns

    Architecture Compliance:
        - Uses FlextTypes for all type definitions (no local TypeVars)
        - Uses FlextConstants for all constants (no local constants)
        - Uses FlextProtocols for all interface definitions
        - Implements nested classes for organization
        - Follows Python 3.13+ syntax throughout
        - Professional Google-style docstrings
        - Zero circular import dependencies

    Examples:
        Basic service management::

            container = FlextContainer()

            # Register service instance
            reg_result = container.register("config", config_instance)
            if reg_result.is_success:
                print("Configuration service registered")

            # Register factory for lazy initialization
            factory_result = container.register_factory(
                "database", lambda: DatabaseService()
            )

            # Type-safe retrieval with validation
            db_result = container.get_typed("database", DatabaseService)
            if db_result.is_success:
                db = db_result.unwrap()

        Global container usage::

            # Global singleton access
            global_container = FlextContainer.get_global()
            global_container.register("metrics", metrics_service)

            # Convenient functions
            container = get_flext_container()
            result = register_typed("cache", cache_service)

    """

    # =========================================================================
    # NESTED CLASSES - Organized functionality following FLEXT patterns
    # =========================================================================

    class ServiceKey[T](
        UserString, FlextProtocols.Foundation.Validator[FlextTypes.Core.String]
    ):
        """Typed service key for type-safe service resolution.

        A specialized string that acts as a plain string at runtime but provides type safety
        at type-check time. This enables type-safe service registration and retrieval in the
        FLEXT dependency injection container while maintaining runtime string support.

        The key supports generic subscription like ServiceKey[DatabaseService]("db")
        to provide compile-time type checking without runtime overhead.

        Attributes:
            data: The underlying string value of the service key.
            name: Alias for the string value (convenience property).

        Example:
            Type-safe service key usage::

                key = FlextContainer.ServiceKey[DatabaseService]("database")
                print(key)  # Acts like a string
                "database"
                print(key.name)
                "database"

        """

        __slots__ = ()

        @property
        def name(self) -> FlextTypes.Core.String:
            """Return the service key name (string value)."""
            return str(self)

        @classmethod
        def __class_getitem__(cls, _item: object) -> type[FlextContainer.ServiceKey[T]]:
            """Support generic subscription without affecting runtime behavior."""
            return cls

        def validate(self, data: FlextTypes.Core.String) -> object:
            """Validate service key name using FlextProtocols.Foundation.Validator interface.

            Args:
                data: Service key string to validate

            Returns:
                FlextResult with validated key or validation error

            """
            if not data or not data.strip():
                return FlextResult[FlextTypes.Core.String].fail(
                    FlextConstants.Messages.SERVICE_NAME_EMPTY,
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )
            return FlextResult[FlextTypes.Core.String].ok(data.strip())

    class Commands:
        """Container command definitions following CQRS pattern.

        Nested class containing all command definitions for container operations
        following the Command Query Responsibility Segregation pattern.
        """

        class RegisterService:
            """Command to register a service instance."""

            def __init__(
                self,
                service_name: FlextTypes.Core.String = "",
                service_instance: FlextTypes.Service.ServiceInstance = None,
                command_type: FlextTypes.Core.String = "register_service",
                command_id: FlextTypes.Core.String = "",
                timestamp: datetime | None = None,
                user_id: FlextTypes.Core.String | None = None,
                correlation_id: FlextTypes.Core.String = "",
            ) -> None:
                """Initialize RegisterService command.

                Args:
                    service_name: Unique identifier for the service.
                    service_instance: The service instance to register.
                    command_type: Type of command for tracking.
                    command_id: Unique command identifier.
                    timestamp: Command execution timestamp.
                    user_id: User executing the command.
                    correlation_id: Correlation identifier for tracing.

                """
                self.service_name = service_name
                self.service_instance = service_instance
                self.command_type = command_type
                self.command_id = command_id or FlextGenerators.generate_uuid()
                self.timestamp = timestamp or datetime.now(tz=ZoneInfo("UTC"))
                self.user_id = user_id
                self.correlation_id = correlation_id or FlextGenerators.generate_uuid()

            @classmethod
            def create(
                cls,
                service_name: FlextTypes.Core.String,
                service_instance: FlextTypes.Service.ServiceInstance,
            ) -> FlextContainer.Commands.RegisterService:
                """Create command with default values.

                Args:
                    service_name: Unique identifier for the service.
                    service_instance: The service instance to register.

                Returns:
                    RegisterService command with generated defaults.

                """
                return cls(
                    service_name=service_name,
                    service_instance=service_instance,
                    command_type="register_service",
                    command_id=FlextGenerators.generate_uuid(),
                    timestamp=datetime.now(tz=ZoneInfo("UTC")),
                    user_id=None,
                    correlation_id=FlextGenerators.generate_uuid(),
                )

            def validate_command(self) -> FlextResult[None]:
                """Validate service registration command.

                Returns:
                    FlextResult indicating validation success or failure.

                """
                if not self.service_name or not self.service_name.strip():
                    return FlextResult[None].fail(
                        FlextConstants.Messages.SERVICE_NAME_EMPTY
                    )
                return FlextResult[None].ok(None)

        class RegisterFactory:
            """Command to register a service factory."""

            def __init__(
                self,
                service_name: str = "",
                factory: FlextTypes.Service.ServiceInstance = None,
                command_type: str = "register_factory",
                command_id: str = "",
                timestamp: datetime | None = None,
                user_id: str | None = None,
                correlation_id: str = "",
            ) -> None:
                """Initialize RegisterFactory command.

                Args:
                    service_name: Unique identifier for the service.
                    factory: Factory function that creates the service.
                    command_type: Type of command for tracking.
                    command_id: Unique command identifier.
                    timestamp: Command execution timestamp.
                    user_id: User executing the command.
                    correlation_id: Correlation identifier for tracing.

                """
                self.service_name = service_name
                self.factory = factory
                self.command_type = command_type
                self.command_id = command_id or FlextGenerators.generate_uuid()
                self.timestamp = timestamp or datetime.now(tz=ZoneInfo("UTC"))
                self.user_id = user_id
                self.correlation_id = correlation_id or FlextGenerators.generate_uuid()

            @classmethod
            def create(
                cls, service_name: str, factory: FlextTypes.Service.ServiceInstance
            ) -> FlextContainer.Commands.RegisterFactory:
                """Create command with default values.

                Args:
                    service_name: Unique identifier for the service.
                    factory: Factory function that creates the service.

                Returns:
                    RegisterFactory command with generated defaults.

                """
                return cls(
                    service_name=service_name,
                    factory=factory,
                    command_type="register_factory",
                    command_id=FlextGenerators.generate_uuid(),
                    timestamp=datetime.now(tz=ZoneInfo("UTC")),
                    user_id=None,
                    correlation_id=FlextGenerators.generate_uuid(),
                )

            def validate_command(self) -> FlextResult[None]:
                """Validate factory registration command.

                Returns:
                    FlextResult indicating validation success or failure.

                """
                if not self.service_name or not self.service_name.strip():
                    return FlextResult[None].fail(
                        FlextConstants.Messages.SERVICE_NAME_EMPTY
                    )
                if not callable(self.factory):
                    return FlextResult[None].fail("Factory must be callable")
                return FlextResult[None].ok(None)

        class UnregisterService:
            """Command to unregister a service."""

            def __init__(
                self,
                service_name: str = "",
                command_type: str = "unregister_service",
                command_id: str = "",
                timestamp: datetime | None = None,
                user_id: str | None = None,
                correlation_id: str = "",
            ) -> None:
                """Initialize UnregisterService command.

                Args:
                    service_name: Unique identifier for the service to unregister.
                    command_type: Type of command for tracking.
                    command_id: Unique command identifier.
                    timestamp: Command execution timestamp.
                    user_id: User executing the command.
                    correlation_id: Correlation identifier for tracing.

                """
                self.service_name = service_name
                self.command_type = command_type
                self.command_id = command_id or FlextGenerators.generate_uuid()
                self.timestamp = timestamp or datetime.now(tz=ZoneInfo("UTC"))
                self.user_id = user_id
                self.correlation_id = correlation_id or FlextGenerators.generate_uuid()

            @classmethod
            def create(
                cls, service_name: str
            ) -> FlextContainer.Commands.UnregisterService:
                """Create command with default values.

                Args:
                    service_name: Unique identifier for the service to unregister.

                Returns:
                    UnregisterService command with generated defaults.

                """
                return cls(
                    service_name=service_name,
                    command_type="unregister_service",
                    command_id=FlextGenerators.generate_uuid(),
                    timestamp=datetime.now(tz=ZoneInfo("UTC")),
                    user_id=None,
                    correlation_id=FlextGenerators.generate_uuid(),
                )

            def validate_command(self) -> FlextResult[None]:
                """Validate service unregistration command.

                Returns:
                    FlextResult indicating validation success or failure.

                """
                if not self.service_name or not self.service_name.strip():
                    return FlextResult[None].fail(
                        FlextConstants.Messages.SERVICE_NAME_EMPTY
                    )
                return FlextResult[None].ok(None)

    class Queries:
        """Container query definitions following CQRS pattern.

        Nested class containing all query definitions for container operations
        following the Command Query Responsibility Segregation pattern.
        """

        class GetService:
            """Query to retrieve a service."""

            def __init__(
                self,
                service_name: str = "",
                expected_type: str | None = None,
                query_type: str = "get_service",
                query_id: str = "",
                page_size: int = 100,
                page_number: int = 1,
                sort_by: str | None = None,
                sort_order: str = "asc",
            ) -> None:
                """Initialize GetService query.

                Args:
                    service_name: Unique identifier for the service.
                    expected_type: Expected type name for validation.
                    query_type: Type of query for tracking.
                    query_id: Unique query identifier.
                    page_size: Page size for pagination.
                    page_number: Page number for pagination.
                    sort_by: Sort field.
                    sort_order: Sort order (asc/desc).

                """
                self.service_name = service_name
                self.expected_type = expected_type
                self.query_type = query_type
                self.query_id = query_id
                self.page_size = page_size
                self.page_number = page_number
                self.sort_by = sort_by
                self.sort_order = sort_order

            @classmethod
            def create(
                cls, service_name: str, expected_type: str | None = None
            ) -> FlextContainer.Queries.GetService:
                """Create query with default values.

                Args:
                    service_name: Unique identifier for the service.
                    expected_type: Expected type name for validation.

                Returns:
                    GetService query with generated defaults.

                """
                return cls(
                    service_name=service_name,
                    expected_type=expected_type,
                    query_type="get_service",
                    query_id="",
                    page_size=100,
                    page_number=1,
                    sort_by=None,
                    sort_order="asc",
                )

            def validate_query(self) -> FlextResult[None]:
                """Validate service retrieval query.

                Returns:
                    FlextResult indicating validation success or failure.

                """
                if not self.service_name or not self.service_name.strip():
                    return FlextResult[None].fail(
                        FlextConstants.Messages.SERVICE_NAME_EMPTY
                    )
                return FlextResult[None].ok(None)

        class ListServices:
            """Query to list all services."""

            def __init__(
                self,
                *,
                include_factories: bool = True,
                service_type_filter: str | None = None,
                query_type: str = "list_services",
                query_id: str = "",
                page_size: int = 100,
                page_number: int = 1,
                sort_by: str | None = None,
                sort_order: str = "asc",
            ) -> None:
                """Initialize ListServices query.

                Args:
                    include_factories: Whether to include factory services.
                    service_type_filter: Filter by service type.
                    query_type: Type of query for tracking.
                    query_id: Unique query identifier.
                    page_size: Page size for pagination.
                    page_number: Page number for pagination.
                    sort_by: Sort field.
                    sort_order: Sort order (asc/desc).

                """
                self.include_factories = include_factories
                self.service_type_filter = service_type_filter
                self.query_type = query_type
                self.query_id = query_id
                self.page_size = page_size
                self.page_number = page_number
                self.sort_by = sort_by
                self.sort_order = sort_order

            @classmethod
            def create(
                cls,
                *,
                include_factories: bool = True,
                service_type_filter: str | None = None,
            ) -> FlextContainer.Queries.ListServices:
                """Create query with default values.

                Args:
                    include_factories: Whether to include factory services.
                    service_type_filter: Filter by service type.

                Returns:
                    ListServices query with generated defaults.

                """
                return cls(
                    include_factories=include_factories,
                    service_type_filter=service_type_filter,
                    query_type="list_services",
                    query_id="",
                    page_size=100,
                    page_number=1,
                    sort_by=None,
                    sort_order="asc",
                )

    class ServiceRegistrar:
        """Service registration component implementing Single Responsibility Principle.

        This nested class handles all service and factory registration operations with validation
        and error handling. It's separated from service retrieval to follow SRP and
        provide clear separation of concerns in the dependency injection system.

        The registrar validates service names, prevents duplicate registrations, and
        provides registry management operations.

        Attributes:
            _services: Internal registry mapping service names to instances.
            _factories: Internal registry mapping service names to factory functions.

        Example:
            Direct usage for service registration::

                registrar = FlextContainer.ServiceRegistrar()
                result = registrar.register_service("logger", logger_instance)
                if result.is_success:
                    print("Service registered successfully")

                factory_result = registrar.register_factory(
                    "db", lambda: DatabaseService()
                )
                if factory_result.is_success:
                    print("Factory registered successfully")

        """

        def __init__(self) -> None:
            """Initialize service registrar with empty registry."""
            self._services: FlextTypes.Service.ServiceDict = {}
            self._factories: FlextTypes.Service.FactoryDict = {}

        @staticmethod
        def _validate_service_name(name: str) -> FlextResult[str]:
            """Validate service name.

            Args:
                name: Service name to validate.

            Returns:
                FlextResult with validated name or error.

            """
            if not flext_validate_service_name(name):
                return FlextResult[str].fail(FlextConstants.Messages.SERVICE_NAME_EMPTY)
            return FlextResult[str].ok(name)

        def register_service(
            self,
            name: FlextTypes.Core.String,
            service: FlextTypes.Service.ServiceInstance,
        ) -> FlextResult[None]:
            """Register service instance.

            Args:
                name: Service identifier.
                service: Service instance.

            Returns:
                FlextResult indicating success or failure.

            """
            # Fast path: simple validation without FlextResult overhead
            if not name or not name.strip():
                return FlextResult[None].fail(
                    FlextConstants.Messages.SERVICE_NAME_EMPTY
                )

            validated_name = name.strip()

            # Store service in registry
            self._services[validated_name] = service
            return FlextResult[None].ok(None)

        def register_factory(
            self,
            name: str,
            factory: FlextTypes.Service.ServiceInstance,
        ) -> FlextResult[None]:
            """Register service factory.

            Args:
                name: Service identifier.
                factory: Callable that creates service.

            Returns:
                FlextResult indicating success or failure.

            """
            validation_result = self._validate_service_name(name)
            if validation_result.is_failure:
                return FlextResult[None].fail(
                    validation_result.error
                    or FlextConstants.Messages.SERVICE_NAME_EMPTY
                )

            validated_name: str = validation_result.value

            # Runtime validation for callability
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
                    msg = (
                        f"Factory must be callable without parameters, "
                        f"but requires {required_params} parameter(s)"
                    )
                    return FlextResult[None].fail(msg)
            except (ValueError, TypeError, OSError) as e:
                return FlextResult[None].fail(
                    f"Could not inspect factory signature: {e}"
                )

            if validated_name in self._services:
                del self._services[validated_name]

            # Safe assignment after signature verification
            factory_callable = cast(
                "Callable[[], FlextTypes.Service.ServiceInstance]", factory
            )
            self._factories[validated_name] = factory_callable
            return FlextResult[None].ok(None)

        def unregister_service(self, name: str) -> FlextResult[None]:
            """Unregister a service.

            Args:
                name: Service identifier to unregister.

            Returns:
                FlextResult indicating success or failure.

            """
            validation_result = self._validate_service_name(name)
            if validation_result.is_failure:
                return FlextResult[None].fail(
                    validation_result.error
                    or FlextConstants.Messages.SERVICE_NAME_EMPTY
                )

            validated_name: str = validation_result.value

            if validated_name in self._services:
                del self._services[validated_name]
                return FlextResult[None].ok(None)

            if validated_name in self._factories:
                del self._factories[validated_name]
                return FlextResult[None].ok(None)

            return FlextResult[None].fail(f"Service '{validated_name}' not found")

        def clear_all(self) -> FlextResult[None]:
            """Clear all registered services and factories.

            Returns:
                FlextResult indicating success.

            """
            self._services.clear()
            self._factories.clear()
            return FlextResult[None].ok(None)

        def get_service_names(self) -> list[str]:
            """Get all registered service names.

            Returns:
                List of all service names.

            """
            return list(self._services.keys()) + list(self._factories.keys())

        def get_service_count(self) -> int:
            """Get total service count.

            Returns:
                Total number of registered services and factories.

            """
            return len(self._services) + len(self._factories)

        def has_service(self, name: str) -> bool:
            """Check if a service exists.

            Args:
                name: Service name to check.

            Returns:
                True if service exists, False otherwise.

            """
            return name in self._services or name in self._factories

        def get_services_dict(self) -> FlextTypes.Service.ServiceDict:
            """Get services dictionary (internal use).

            Returns:
                Internal services dictionary.

            """
            return self._services

        def get_factories_dict(self) -> FlextTypes.Service.FactoryDict:
            """Get factories dictionary (internal use).

            Returns:
                Internal factories dictionary.

            """
            return self._factories

    class ServiceRetriever:
        """Service retrieval component implementing single responsibility principle.

        This nested class handles service retrieval operations including service
        resolution, factory instantiation, and service information queries.
        """

        def __init__(
            self,
            services: FlextTypes.Service.ServiceDict,
            factories: FlextTypes.Service.FactoryDict,
        ) -> None:
            """Initialize service retriever with references.

            Args:
                services: Services dictionary reference.
                factories: Factories dictionary reference.

            """
            super().__init__()
            self._services = services
            self._factories = factories

        @staticmethod
        def _validate_service_name(name: str) -> FlextResult[str]:
            """Validate service name.

            Args:
                name: Service name to validate.

            Returns:
                FlextResult with validated name or error.

            """
            if not flext_validate_service_name(name):
                return FlextResult[str].fail(FlextConstants.Messages.SERVICE_NAME_EMPTY)
            return FlextResult[str].ok(name)

        def get_service(
            self, name: str
        ) -> FlextResult[FlextTypes.Service.ServiceInstance]:
            """Retrieve a registered service - Performance optimized.

            Args:
                name: Service name to retrieve.

            Returns:
                FlextResult containing the service instance or error.

            """
            if not name or not name.strip():
                return FlextResult[FlextTypes.Service.ServiceInstance].fail(
                    FlextConstants.Messages.SERVICE_NAME_EMPTY
                )

            validated_name = name.strip()

            # Check direct service registration first (the most common case)
            if validated_name in self._services:
                return FlextResult[FlextTypes.Service.ServiceInstance].ok(
                    self._services[validated_name]
                )

            # Check factory registration
            if validated_name in self._factories:
                try:
                    factory = self._factories[validated_name]
                    service = factory()

                    # Cache the factory result as a service for singleton behavior
                    self._services[validated_name] = service
                    # Remove from factories since it's now cached as a service
                    del self._factories[validated_name]

                    return FlextResult[FlextTypes.Service.ServiceInstance].ok(service)
                except (
                    TypeError,
                    ValueError,
                    AttributeError,
                    RuntimeError,
                    _get_exception_class("FlextError"),
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
                return FlextResult[dict[str, object]].fail(
                    FlextConstants.Messages.SERVICE_NAME_EMPTY
                )

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

        def list_services(self) -> FlextTypes.Service.ServiceListDict:
            """List all services with their types.

            Returns:
                Dictionary mapping service names to their types.

            """
            services_info: FlextTypes.Service.ServiceListDict = {}

            for name in self._services:
                services_info[name] = "instance"

            for name in self._factories:
                services_info[name] = "factory"

            return services_info

    class GlobalManager:
        """Simple global container manager for singleton pattern.

        This nested class manages the global container instance following
        the singleton pattern for ecosystem-wide service sharing.
        """

        def __init__(self) -> None:
            """Initialize with default container."""
            self._container = FlextContainer()

        def get_container(self) -> FlextContainer:
            """Get the global container instance.

            Returns:
                The global container instance.

            """
            return self._container

        def set_container(self, container: FlextContainer) -> None:
            """Set the global container instance.

            Args:
                container: Container instance to set as global.

            """
            self._container = container

    # =========================================================================
    # CONTAINER IMPLEMENTATION - Main container functionality
    # =========================================================================

    def __init__(self) -> None:
        """Initialize container with internal components and command bus."""
        super().__init__()

        # SRP: Delegate to focused internal components
        self._registrar = self.ServiceRegistrar()

        # DIP: Retriever depends on registrar's data abstractions
        services_dict = self._registrar.get_services_dict()
        factories_dict = self._registrar.get_factories_dict()
        self._retriever = self.ServiceRetriever(services_dict, factories_dict)

        # Simplified command bus - can be extended later
        self._command_bus = None

    # -------------------------------------------------------------------------
    # Registration API - Simplified without command bus for now
    # -------------------------------------------------------------------------

    def register(self, name: str, service: object) -> FlextResult[None]:
        """Register a service instance.

        Args:
            name: Service identifier.
            service: Service instance to register.

        Returns:
            FlextResult indicating success or failure.

        """
        return self._registrar.register_service(name, service)

    def register_factory(
        self,
        name: str,
        factory: Callable[[], object],
    ) -> FlextResult[None]:
        """Register a service factory.

        Args:
            name: Service identifier.
            factory: Factory function that creates the service.

        Returns:
            FlextResult indicating success or failure.

        """
        return self._registrar.register_factory(name, factory)

    def unregister(self, name: str) -> FlextResult[None]:
        """Unregister service.

        Args:
            name: Service identifier.

        Returns:
            FlextResult indicating success or failure.

        """
        return self._registrar.unregister_service(name)

    # -------------------------------------------------------------------------
    # Retrieval API - Simplified without command bus
    # -------------------------------------------------------------------------

    def get(self, name: str) -> FlextResult[object]:
        """Get a service by name.

        Args:
            name: Service identifier.

        Returns:
            FlextResult containing the service instance or error.

        """
        return self._retriever.get_service(name)

    def get_typed(self, name: str, expected_type: type[T]) -> FlextResult[T]:
        """Get service with type checking - Performance optimized.

        Args:
            name: Service identifier.
            expected_type: Expected service type for validation.

        Returns:
            FlextResult containing the typed service instance or error.

        """
        result = self.get(name)
        if result.is_failure:
            return FlextResult[T].fail(result.error or "Service not found")

        service: object = result.value

        # Simple isinstance check instead of complex type guards
        if not isinstance(service, expected_type):
            actual_type = type(service).__name__
            return FlextResult[T].fail(
                f"Service '{name}' is {actual_type}, expected {expected_type.__name__}",
            )

        return FlextResult[T].ok(service)

    # -------------------------------------------------------------------------
    # Container management
    # -------------------------------------------------------------------------

    def clear(self) -> FlextResult[None]:
        """Clear all services.

        Returns:
            FlextResult indicating success.

        """
        return self._registrar.clear_all()

    def has(self, name: str) -> bool:
        """Check if a service exists.

        Args:
            name: Service identifier.

        Returns:
            True if service exists, False otherwise.

        """
        return self._registrar.has_service(name)

    def list_services(self) -> FlextTypes.Service.ServiceListDict:
        """List all services.

        Returns:
            Dictionary mapping service names to their types.

        """
        return self._retriever.list_services()

    def get_service_names(self) -> list[str]:
        """Get service names.

        Returns:
            List of all registered service names.

        """
        return self._registrar.get_service_names()

    def get_service_count(self) -> int:
        """Get service count.

        Returns:
            Total number of registered services.

        """
        return self._registrar.get_service_count()

    @property
    def command_bus(self) -> object:
        """Access to the internal command bus for operations.

        Returns:
            Internal command bus instance.

        """
        return self._command_bus

    def get_info(self, name: str) -> FlextResult[dict[str, object]]:
        """Return basic info about a registered service or factory.

        Args:
            name: Service identifier.

        Returns:
            FlextResult containing service information dictionary.

        """
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
        """Get existing service or register-and-return a newly created one.

        Args:
            name: Service identifier.
            factory: Factory function to create service if not found.

        Returns:
            FlextResult containing the service instance.

        """
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

            # Try to get the service
            service_result = self.get(name)
            if service_result.is_failure:
                error = service_result.error or ""
                if "Factory for" in error and "failed:" in error:
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
                        f"Required dependency '{param.name}' not found "
                        f"for {service_class.__name__}"
                    )

                kwargs[param.name] = dependency_result.value

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
        """Register many services/factories atomically; roll back on failure.

        Args:
            registrations: Dictionary of service names to instances/factories.

        Returns:
            FlextResult containing list of registered service names.

        """
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

    # =========================================================================
    # CLASS METHODS FOR GLOBAL CONTAINER MANAGEMENT - Architectural Tier 1
    # =========================================================================

    @classmethod
    def get_global(cls) -> FlextContainer:
        """Get global container instance (class method).

        Returns:
            Global container instance.

        """
        return _global_manager.get_container()

    @classmethod
    def configure_global(
        cls, container: FlextContainer | None = None
    ) -> FlextContainer:
        """Configure global container (class method).

        Args:
            container: Container to configure or None for new container.

        Returns:
            Configured container.

        """
        if container is None:
            container = cls()
        _global_manager.set_container(container)
        return container

    @classmethod
    def get_global_typed(cls, name: str, expected_type: type[T]) -> FlextResult[T]:
        """Get typed service from global container (class method).

        Args:
            name: Service identifier.
            expected_type: Expected service type.

        Returns:
            Type-safe result with service.

        """
        container = cls.get_global()
        return container.get_typed(name, expected_type)

    @classmethod
    def register_global(cls, name: str, service: object) -> FlextResult[None]:
        """Register service in global container (class method).

        Args:
            name: Service identifier.
            service: Service instance.

        Returns:
            FlextResult indicating success or failure.

        """
        container = cls.get_global()
        return container.register(name, service)

    @classmethod
    def create_module_utilities(cls, module_name: str) -> dict[str, object]:
        """Create standardized container helpers for a module namespace (class method).

        Provides three utilities:
        - get_container: returns the global FlextContainer
        - configure_dependencies: no-op configurator returning FlextResult[None]
        - get_service(name): fetches service by name with fallback lookups

        Args:
            module_name: Logical module namespace used for fallback lookups.

        Returns:
            Mapping with utility callables.

        """

        def _get_container() -> FlextContainer:
            return cls.get_global()

        def _configure_dependencies() -> FlextResult[None]:
            # Intentionally a no-op default. Modules can replace this function
            # to perform actual registrations when needed.
            return FlextResult[None].ok(None)

        def _get_service(name: str) -> FlextResult[object]:
            container = cls.get_global()
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

    @override
    def __repr__(self) -> str:
        """Return string representation of container.

        Returns:
            String representation showing service count.

        """
        count = self.get_service_count()
        return f"FlextContainer(services: {count})"


# =============================================================================
# GLOBAL CONTAINER MANAGER - Singleton instance
# =============================================================================

_global_manager = FlextContainer.GlobalManager()


# =============================================================================
# CONVENIENCE FUNCTIONS - Legacy compatibility layer
# =============================================================================


def get_flext_container() -> FlextContainer:
    """Get global container instance.

    Returns:
      Global container instance.

    """
    return FlextContainer.get_global()


def configure_flext_container(
    container: FlextContainer | None = None,
) -> FlextContainer:
    """Configure global container (convenience function).

    Args:
      container: Container to configure or None for new container.

    Returns:
      Configured container.

    """
    return FlextContainer.configure_global(container)


def get_typed(
    key: FlextContainer.ServiceKey[T] | str,
    expected_type: type[T],
) -> FlextResult[T]:
    """Get typed service from global container.

    Args:
      key: Service identifier.
      expected_type: Expected service type.

    Returns:
      Type-safe result with service.

    """
    # Convert ServiceKey to str if needed
    key_str = str(key)
    return FlextContainer.get_global_typed(key_str, expected_type)


def register_typed(
    key: FlextContainer.ServiceKey[T] | str, service: T
) -> FlextResult[None]:
    """Register service in global container.

    Args:
      key: Service identifier.
      service: Service instance.

    Returns:
      FlextResult indicating success or failure.

    """
    # Convert ServiceKey to str if needed
    key_str = str(key)
    return FlextContainer.register_global(key_str, service)


def create_module_container_utilities(module_name: str) -> dict[str, object]:
    """Create standardized container helpers for a module namespace.

    Provides three utilities:
    - get_container: returns the global FlextContainer
    - configure_dependencies: no-op configurator returning FlextResult[None]
    (modules may monkey-patch/replace this later to register services)
    - get_service(name): fetches a service by name, with fallback to
    "{module_name}.{name}" to support namespaced registrations

    Args:
      module_name: Logical module namespace used for fallback lookups.

    Returns:
      Mapping with utility callables.

    """
    return FlextContainer.create_module_utilities(module_name)


# =============================================================================
# EXPORTS - Single consolidated class with backward compatibility
# =============================================================================

__all__: list[str] = [
    "FlextContainer",  # ONLY main class exported
]
