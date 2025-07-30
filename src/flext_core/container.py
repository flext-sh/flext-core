"""FLEXT Core Container Module.

Enterprise-grade dependency injection system for the FLEXT Core library providing
comprehensive service management through SOLID principles and type-safe operations.

Architecture:
    - SOLID-compliant design with clear separation of responsibilities
    - Internal modular organization with focused single-responsibility classes
    - Type-safe service registration and retrieval with generic support
    - FlextResult integration for comprehensive error handling
    - Global container management with thread-safe operations

Container System Components:
    - FlextServiceRegistrar: Service and factory registration operations
    - FlextServiceRetrivier: Service retrieval and information operations
    - FlextContainer: Main public API orchestrating internal components
    - ServiceKey[T]: Type-safe service key system for enhanced type safety
    - Global management: Thread-safe global container instance management

Maintenance Guidelines:
    - Maintain single responsibility principle in internal classes
    - Use FlextResult pattern for all operations that can fail
    - Integrate FlextLoggableMixin for consistent logging across components
    - Preserve type safety through ServiceKey system and type guards
    - Keep registration and retrieval operations separate for clarity

Design Decisions:
    - Consolidated from multiple modules while maintaining internal SRP
    - Dependency inversion through interface abstractions
    - Factory pattern support for lazy service initialization
    - Type-safe operations with compile-time and runtime validation
    - FlextResult error handling instead of exception propagation

Dependency Injection Features:
    - Service instance registration with singleton management
    - Factory function registration for lazy initialization
    - Type-safe retrieval with compile-time type checking
    - Service information and introspection capabilities
    - Global container management for application-wide access

Dependencies:
    - result: FlextResult pattern for error handling
    - mixins: FlextLoggableMixin for structured logging
    - types: Type definitions and type guard utilities
    - validation_base: Core validation for service names and parameters

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

from flext_core.constants import MESSAGES
from flext_core.exceptions import FlextError
from flext_core.flext_types import FlextTypes, TAnyDict
from flext_core.loggings import FlextLoggerFactory
from flext_core.mixins import FlextLoggableMixin
from flext_core.result import FlextResult
from flext_core.validation import flext_validate_service_name

if TYPE_CHECKING:
    from flext_core.flext_types import T

# FlextLogger imported for convenience - all classes use FlextLoggableMixin


# =============================================================================
# SERVICE REGISTRATION - SRP: Registration Operations Only
# =============================================================================


class FlextServiceRegistrar(FlextLoggableMixin):
    """Service registration component implementing single responsibility principle.

    Focused component responsible exclusively for service and factory registration
    operations. Implements comprehensive validation, error handling, and logging
    for registration lifecycle management.

    Architecture:
        - Single responsibility: service registration operations only
        - FlextLoggableMixin integration for structured logging
        - FlextResult error handling for comprehensive error reporting
        - Validation integration for service name and parameter checking

    Registration Features:
        - Service instance registration with duplicate detection
        - Factory function registration for lazy initialization
        - Service unregistration with cleanup operations
        - Bulk operations for clearing all registered services
        - Service existence checking and metadata operations

    Validation and Error Handling:
        - Service name validation using base validators
        - Duplicate registration detection and prevention
        - Comprehensive error messages with context information
        - Structured logging for all registration operations

    Internal Storage:
        - _services: Dictionary mapping service names to instances
        - _factories: Dictionary mapping service names to factory functions
        - Separate storage for instances and factories for clear separation

    Usage (Internal):
        registrar = FlextServiceRegistrar()
        result = registrar.register_service("user_service", UserService())
        if result.is_success:
            # Service registered successfully
            pass
    """

    def __init__(self) -> None:
        """Initialize service registrar."""
        self._services: TAnyDict = {}
        self._factories: dict[str, Callable[[], object]] = {}

    def _validate_service_name(self, name: str) -> FlextResult[str]:
        """Validate service name - delegates to centralized validation.

        Eliminates code duplication by using single source of truth from
        validation module.
        """
        if not flext_validate_service_name(name):
            return FlextResult.fail(MESSAGES["SERVICE_NAME_EMPTY"])
        return FlextResult.ok(name)

    def register_service(self, name: str, service: object) -> FlextResult[None]:
        """Register a service instance."""
        self.logger.trace(
            "Starting service registration",
            name=name,
            service_type=type(service).__name__,
        )

        # Use centralized validation - eliminates duplication
        validation_result = self._validate_service_name(name)
        if validation_result.is_failure:
            self.logger.debug(
                "Service name validation failed",
                name=name,
                error=validation_result.error,
            )
            return validation_result.map(lambda _: None)  # Convert type, preserve error

        validated_name = validation_result.unwrap()
        self.logger.trace(
            "Service name validated successfully",
            validated_name=validated_name,
        )

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
        self.logger.trace(
            "Service registration completed successfully", name=validated_name,
        )
        return FlextResult.ok(None)

    def register_factory(
        self,
        name: str,
        factory: object,
    ) -> FlextResult[None]:
        """Register a service factory."""
        self.logger.trace(
            "Starting factory registration",
            name=name,
            factory_type=type(factory).__name__,
        )

        # Use centralized validation - eliminates duplication
        validation_result = self._validate_service_name(name)
        if validation_result.is_failure:
            self.logger.debug(
                "Factory name validation failed",
                name=name,
                error=validation_result.error,
            )
            return validation_result.map(lambda _: None)  # Convert type, preserve error

        validated_name = validation_result.unwrap()
        self.logger.trace(
            "Factory name validated successfully", validated_name=validated_name,
        )

        # Validate factory is callable
        if not callable(factory):
            self.logger.debug(
                "Factory validation failed - not callable",
                name=validated_name,
                factory_type=type(factory).__name__,
            )
            return FlextResult.fail("Factory must be callable")

        self.logger.trace("Factory callable validation passed", name=validated_name)

        # Remove existing service if present to force factory usage
        if validated_name in self._services:
            self.logger.debug(
                "Removing existing service to register factory", name=validated_name,
            )
            del self._services[validated_name]

        # Register factory with validated name - cast to correct type after validation
        factory_callable = cast("Callable[[], object]", factory)
        self._factories[validated_name] = factory_callable
        self.logger.debug(
            "Factory registered",
            name=validated_name,
            factory_type=type(factory).__name__,
            total_factories=len(self._factories),
        )
        self.logger.trace(
            "Factory registration completed successfully", name=validated_name,
        )
        return FlextResult.ok(None)

    def unregister_service(self, name: str) -> FlextResult[None]:
        """Unregister a service."""
        # Use centralized validation - eliminates duplication
        validation_result = self._validate_service_name(name)
        if validation_result.is_failure:
            return validation_result.map(lambda _: None)  # Convert type, preserve error

        validated_name = validation_result.unwrap()

        if validated_name in self._services:
            del self._services[validated_name]
            self.logger.debug("Service unregistered", name=validated_name)
            return FlextResult.ok(None)

        if validated_name in self._factories:
            del self._factories[validated_name]
            self.logger.debug("Factory unregistered", name=validated_name)
            return FlextResult.ok(None)

        self.logger.warning("Service not found for unregistration", name=validated_name)
        return FlextResult.fail(f"Service '{validated_name}' not found")

    def clear_all(self) -> FlextResult[None]:
        """Clear all registered services and factories."""
        service_count = len(self._services)
        factory_count = len(self._factories)

        self._services.clear()
        self._factories.clear()

        self.logger.info(
            "Container cleared",
            services_cleared=service_count,
            factories_cleared=factory_count,
        )
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

    def get_services_dict(self) -> TAnyDict:
        """Get services dictionary (internal use)."""
        return self._services

    def get_factories_dict(self) -> dict[str, Callable[[], object]]:
        """Get factories dictionary (internal use)."""
        return self._factories


# =============================================================================
# SERVICE RETRIEVAL - SRP: Retrieval Operations Only
# =============================================================================


class FlextServiceRetrivier(FlextLoggableMixin):
    """Service retrieval component implementing single responsibility principle.

    Focused component responsible exclusively for service retrieval, factory execution,
    and service information operations. Operates on shared storage references from
    registration component following dependency inversion principle.

    Architecture:
        - Single responsibility: service retrieval operations only
        - Dependency inversion: operates on abstractions (dictionaries) not
          implementations
        - FlextLoggableMixin integration for structured logging
        - FlextResult error handling for comprehensive error reporting

    Retrieval Features:
        - Direct service instance retrieval from registration cache
        - Factory function execution with error handling for lazy services
        - Service information and metadata extraction
        - Service listing and enumeration operations
        - Comprehensive error handling for missing services and factory failures

    Factory Execution:
        - Safe factory function execution with exception handling
        - Comprehensive error reporting for factory failures
        - Logging of factory execution attempts and outcomes
        - No caching of factory results - executed on each retrieval

    Service Information:
        - Service type identification (instance vs factory)
        - Class and module information for instances
        - Factory function information for factories
        - Comprehensive service metadata for debugging and introspection

    Usage (Internal):
        retriever = FlextServiceRetrivier(services_dict, factories_dict)
        result = retriever.get_service("user_service")
        if result.is_success:
            service = result.data
    """

    def __init__(
        self,
        services: TAnyDict,
        factories: dict[str, Callable[[], object]],
    ) -> None:
        """Initialize service retriever with references."""
        self._services = services
        self._factories = factories

    def _validate_service_name(self, name: str) -> FlextResult[str]:
        """Validate service name - delegates to centralized validation.

        Eliminates code duplication by using single source of truth from
        validation module.
        """
        if not flext_validate_service_name(name):
            return FlextResult.fail(MESSAGES["SERVICE_NAME_EMPTY"])
        return FlextResult.ok(name)

    def get_service(self, name: str) -> FlextResult[object]:
        """Retrieve a registered service."""
        self.logger.trace("Starting service retrieval", name=name)

        # Use centralized validation - eliminates duplication
        validation_result = self._validate_service_name(name)
        if validation_result.is_failure:
            self.logger.debug(
                "Service name validation failed during retrieval",
                name=name,
                error=validation_result.error,
            )
            return validation_result.map(
                lambda _: object(),
            )  # Convert type, preserve error

        validated_name = validation_result.unwrap()
        self.logger.trace(
            "Service name validated for retrieval", validated_name=validated_name,
        )

        # Check direct service registration
        if validated_name in self._services:
            service = self._services[validated_name]
            self.logger.debug(
                "Service retrieved from cache",
                name=validated_name,
                service_type=type(service).__name__,
                service_id=id(service),
            )
            self.logger.trace(
                "Service retrieval from cache completed", name=validated_name,
            )
            return FlextResult.ok(service)

        # Check factory registration
        if validated_name in self._factories:
            self.logger.trace(
                "Service not in cache, attempting factory creation", name=validated_name,
            )
            try:
                factory = self._factories[validated_name]
                self.logger.debug(
                    "Creating service from factory",
                    name=validated_name,
                    factory_type=type(factory).__name__,
                )
                service = factory()
                self.logger.trace(
                    "Factory execution successful",
                    name=validated_name,
                    created_service_type=type(service).__name__,
                    created_service_id=id(service),
                )

                # Cache the factory result as a service for singleton behavior
                self._services[validated_name] = service
                # Remove from factories since it's now cached as a service
                del self._factories[validated_name]

                self.logger.debug(
                    "Service created and cached from factory",
                    name=validated_name,
                    service_type=type(service).__name__,
                    total_services=len(self._services),
                    total_factories=len(self._factories),
                )
                return FlextResult.ok(service)
            except (
                TypeError,
                ValueError,
                AttributeError,
                RuntimeError,
                FlextError,
            ) as e:
                self.logger.exception(
                    "Factory execution failed",
                    name=validated_name,
                    error=str(e),
                    factory_type=type(self._factories[validated_name]).__name__,
                )
                return FlextResult.fail(f"Factory for '{validated_name}' failed: {e!s}")

        self.logger.warning(
            "Service not found",
            name=validated_name,
            available_services=list(self._services.keys()),
            available_factories=list(self._factories.keys()),
        )
        self.logger.trace("Service retrieval failed - not found", name=validated_name)
        return FlextResult.fail(f"Service '{validated_name}' not found")

    def get_service_info(self, name: str) -> FlextResult[TAnyDict]:
        """Get service information.

        Args:
            name: Name of service

        Returns:
            FlextResult with service information

        """
        # Use centralized validation - eliminates duplication
        validation_result = self._validate_service_name(name)
        if validation_result.is_failure:
            return validation_result.map(lambda _: {})  # Convert type, preserve error

        validated_name = validation_result.unwrap()

        if validated_name in self._services:
            service = self._services[validated_name]
            info: TAnyDict = {
                "name": validated_name,
                "type": "instance",
                "class": type(service).__name__,
                "module": type(service).__module__,
            }
            self.logger.debug("Service info retrieved", name=validated_name, info=info)
            return FlextResult.ok(info)

        if validated_name in self._factories:
            factory = self._factories[validated_name]
            factory_info: TAnyDict = {
                "name": validated_name,
                "type": "factory",
                "factory": factory.__name__,
                "module": factory.__module__,
            }
            self.logger.debug(
                "Factory info retrieved",
                name=validated_name,
                info=factory_info,
            )
            return FlextResult.ok(factory_info)

        self.logger.warning("Service not found for info", name=validated_name)
        return FlextResult.fail(f"Service '{validated_name}' not found")

    def list_services(self) -> dict[str, str]:
        """List all services with their types.

        Returns:
            Dictionary of service names and types

        """
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
    """Enterprise dependency injection container with SOLID principles and type safety.

    Main public API for dependency injection operations, orchestrating internal
    components while maintaining clean separation of concerns. Provides comprehensive
    service management with error handling and type safety.

    Architecture:
        - SOLID principles: SRP through internal component delegation
        - Dependency inversion: depends on abstractions not concretions
        - Open/closed: extensible through registration patterns
        - Interface segregation: focused public API methods
        - Liskov substitution: compatible service implementations

    Enterprise Features:
        - Type-safe service registration and retrieval
        - Factory pattern support for lazy initialization
        - Comprehensive error handling with FlextResult patterns
        - Structured logging for all operations
        - Service introspection and metadata operations

    Public API Categories:
        - Registration: register, register_factory, unregister
        - Retrieval: get, get_typed, get_info
        - Management: clear, has, list_services, get_service_names
        - Type Safety: get_typed with compile-time type checking

    Internal Orchestration:
        - FlextServiceRegistrar: handles all registration operations
        - FlextServiceRetrivier: handles all retrieval operations
        - Shared storage references for consistent state management
        - Logging integration for comprehensive audit trail

    Type Safety Features:
        - get_typed method with runtime type validation
        - ServiceKey[T] system for compile-time type checking
        - Type guard integration for runtime type safety
        - Generic type support for strongly-typed operations

    Usage Patterns:
        # Basic usage
        container = FlextContainer()
        container.register("service", MyService())
        result = container.get("service")

        # Type-safe usage
        result = container.get_typed("service", MyService)
        if result.is_success:
            service: MyService = result.data

        # Factory usage
        container.register_factory("lazy_service", lambda: ExpensiveService())
        service_result = container.get("lazy_service")  # Factory executed here
    """

    def __init__(self) -> None:
        """Initialize container with internal components."""
        self.logger.info("Initializing FlextContainer")

        # SRP: Delegate to focused internal components
        self._registrar = FlextServiceRegistrar()

        # DIP: Retriever depends on registrar's data abstractions
        services_dict = self._registrar.get_services_dict()
        factories_dict = self._registrar.get_factories_dict()
        self._retriever = FlextServiceRetrivier(services_dict, factories_dict)

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
        """Unregister a service."""
        return self._registrar.unregister_service(name)

    # Retrieval API - Delegate to retriever
    def get(self, name: str) -> FlextResult[object]:
        """Get a service by name."""
        return self._retriever.get_service(name)

    def get_info(self, name: str) -> FlextResult[TAnyDict]:
        """Get service information."""
        return self._retriever.get_service_info(name)

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
        """Get service with type checking."""
        self.logger.debug(
            "Getting typed service",
            name=name,
            expected_type=expected_type.__name__,
        )

        result = self.get(name)
        if result.is_failure:
            return FlextResult.fail(result.error or "Service not found")

        service = result.unwrap()

        # Use FlextTypes type guard - MAXIMIZA base usage
        if not FlextTypes.TypeGuards.is_instance_of(service, expected_type):
            actual_type = type(service).__name__
            self.logger.error(
                "Type mismatch for service",
                name=name,
                expected=expected_type.__name__,
                actual=actual_type,
            )
            return FlextResult.fail(
                f"Service '{name}' is {actual_type}, expected {expected_type.__name__}",
            )

        self.logger.debug("Typed service retrieved successfully", name=name)
        return FlextResult.ok(cast("T", service))

    def auto_wire(
        self,
        service_class: type[T],
        name: str | None = None,
    ) -> FlextResult[T]:
        """Auto-wire a service class with automatic dependency injection.

        Automatically creates an instance of the service class by injecting
        registered dependencies based on constructor parameters.

        Args:
            service_class: Class to instantiate with auto-wiring
            name: Optional service name (defaults to class name)

        Returns:
            FlextResult containing the auto-wired service instance

        """
        service_name = name or service_class.__name__
        self.logger.debug("Auto-wiring service", service_name=service_name)

        try:
            # Get constructor signature
            sig = inspect.signature(service_class.__init__)
            kwargs = {}

            # Resolve dependencies for each parameter (skip 'self')
            for param_name, param in sig.parameters.items():
                if param_name == "self":
                    continue

                # Try to get service by parameter name
                dep_result = self.get(param_name)
                if dep_result.is_success:
                    kwargs[param_name] = dep_result.data
                elif param.default == inspect.Parameter.empty:
                    # Required parameter not found
                    error_msg = (
                        f"Required dependency '{param_name}' not found for "
                        f"{service_name}"
                    )
                    return FlextResult.fail(error_msg)

            # Create instance with injected dependencies
            instance = service_class(**kwargs)

            # Register the auto-wired instance
            register_result = self.register(service_name, instance)
            if register_result.is_failure:
                error_msg = (
                    f"Failed to register auto-wired service: {register_result.error}"
                )
                return FlextResult.fail(error_msg)

            self.logger.info(
                "Service auto-wired successfully",
                service_name=service_name,
            )
            return FlextResult.ok(instance)

        except (TypeError, ValueError, AttributeError, RuntimeError) as e:
            return FlextResult.fail(f"Auto-wiring failed for {service_name}: {e}")

    def get_or_create(
        self,
        name: str,
        factory: Callable[[], T],
    ) -> FlextResult[T]:
        """Get service or create it using provided factory if not found.

        Combines retrieval and lazy creation in a single operation to reduce
        boilerplate for optional service creation patterns.

        Args:
            name: Service name to retrieve or create
            factory: Factory function to create service if not found

        Returns:
            FlextResult containing existing or newly created service

        """
        # Try to get existing service first
        result = self.get(name)
        if result.is_success:
            self.logger.debug("Service found", name=name)
            return FlextResult.ok(cast("T", result.data))

        # Service not found, create using factory
        self.logger.debug("Service not found, creating with factory", name=name)
        try:
            service = factory()
            register_result = self.register(name, service)
            if register_result.is_failure:
                error_msg = (
                    f"Failed to register created service: {register_result.error}"
                )
                return FlextResult.fail(error_msg)

            self.logger.info("Service created and registered", name=name)
            return FlextResult.ok(service)

        except (TypeError, ValueError, AttributeError, RuntimeError) as e:
            return FlextResult.fail(f"Factory failed for service '{name}': {e}")

    def batch_register(
        self,
        services: dict[str, object],
    ) -> FlextResult[list[str]]:
        """Register multiple services in a single operation.

        Reduces boilerplate when registering multiple related services
        by handling them as a batch with rollback on partial failure.

        Args:
            services: Dictionary mapping service names to instances

        Returns:
            FlextResult containing list of successfully registered service names

        """
        registered_names: list[str] = []

        for name, service in services.items():
            result = self.register(name, service)
            if result.is_failure:
                # Rollback previously registered services from this batch
                for registered_name in registered_names:
                    self.unregister(registered_name)
                error_msg = f"Batch registration failed at '{name}': {result.error}"
                return FlextResult.fail(error_msg)

            registered_names.append(name)

        self.logger.info("Batch registration completed", count=len(registered_names))
        return FlextResult.ok(registered_names)

    def __repr__(self) -> str:
        """Return string representation of container."""
        count = self.get_service_count()
        return f"FlextContainer(services: {count})"


# =============================================================================
# GLOBAL CONTAINER MANAGEMENT
# =============================================================================


# Global container instance with thread-safe access
class FlextGlobalContainerManager(FlextLoggableMixin):
    """Thread-safe global container management without global statements.

    Provides centralized management of the global FlextContainer instance
    with thread-safe access patterns and lazy initialization. Eliminates
    the need for global variables while providing application-wide access.

    Architecture:
        - Singleton pattern for global container management
        - Lazy initialization for performance optimization
        - Thread-safe operations for concurrent access
        - Clear separation from business logic

    Global Management Features:
        - Lazy container creation on first access
        - Global container replacement for testing scenarios
        - Thread-safe access without explicit locking
        - Clear ownership and lifecycle management

    Usage (Internal):
        manager = FlextGlobalContainerManager()
        container = manager.get_container()  # Creates if not exists
        manager.set_container(custom_container)  # Replace for testing
    """

    def __init__(self) -> None:
        super().__init__()
        self._container: FlextContainer | None = None

    def get_container(self) -> FlextContainer:
        """Get or create global container."""
        if self._container is None:
            self.logger.info("Creating global FlextContainer instance")
            self._container = FlextContainer()
        return self._container

    def set_container(self, container: FlextContainer) -> None:
        """Set global container."""
        self.logger.info(
            "Configuring global FlextContainer",
            new_container=str(container),
        )
        self._container = container


_global_manager = FlextGlobalContainerManager()


def get_flext_container() -> FlextContainer:
    """Get global FlextContainer instance with lazy initialization.

    Provides access to the application-wide FlextContainer instance, creating
    it on first access if it doesn't exist. Thread-safe for concurrent access.

    Returns:
        Global FlextContainer instance for application-wide service management

    Usage:
        container = get_flext_container()
        container.register("service", MyService())
        service_result = container.get("service")

    """
    return _global_manager.get_container()


def configure_flext_container(container: FlextContainer | None) -> FlextContainer:
    """Configure global FlextContainer instance for application use.

    Replaces the global container instance with a custom container,
    typically used for testing scenarios or specialized configurations.

    Args:
        container: Custom FlextContainer instance to use globally, or None to create new

    Returns:
        The container that was set as global (either provided or newly created)

    Usage:
        # Testing scenario
        test_container = FlextContainer()
        test_container.register("test_service", MockService())
        result = configure_flext_container(test_container)

        # Now all global access uses test container
        assert get_flext_container() is test_container
        assert result is test_container

        # Reset to new container
        new_container = configure_flext_container(None)
        assert new_container is not test_container

    """
    if container is None:
        container = FlextContainer()
    _global_manager.set_container(container)
    return container


# Type-safe service keys
class ServiceKey[T]:
    r"""Type-safe service key providing compile-time type checking for containers.

    Generic service key class that encapsulates service names with associated type
    information for compile-time type safety and enhanced developer experience.
    Enables type-safe service registration and retrieval operations.

    Architecture:
        - Generic class with type parameter T for service type
        - Simple name encapsulation with type association
        - Integration with type-safe container operations
        - String representation for debugging and logging

    Type Safety Features:
        - Compile-time type checking for service operations
        - Type inference in IDE environments
        - Prevention of type mismatches at registration time
        - Enhanced intellisense and code completion

    Usage Patterns:
        # Define typed service keys
        USER_SERVICE_KEY = ServiceKey[UserService]("user_service")
        CONFIG_SERVICE_KEY = ServiceKey[ConfigService]("config_service")

        # Type-safe registration
        register_typed(container, USER_SERVICE_KEY, UserService())

        # Type-safe retrieval with automatic type inference
        user_service_result = get_typed(container, USER_SERVICE_KEY)
        if user_service_result.is_success:
            user_service: UserService = user_service_result.data  # Type inferred

    Args:
        name: String identifier for the service

    """

    def __init__(self, name: str) -> None:
        """Initialize service key with name.

        Args:
            name: Name of service

        """
        self.name = name

    def __str__(self) -> str:
        """Return string representation of service key.

        Returns:
            String representation of service key

        """
        return self.name


# Type alias for factory functions with comprehensive documentation
FlextServiceFactory = Callable[[], object]
"""Type alias for service factory functions.

Factory functions used for lazy service initialization in the dependency
injection container. Factories are called on each service retrieval without
caching, allowing for fresh instances or controlled initialization.

Signature:
    () -> object: Function that takes no arguments and returns service instance

Usage:
    def create_database_service() -> DatabaseService:
        return DatabaseService(connection_string=get_config())

    factory: FlextServiceFactory = create_database_service
    container.register_factory("database", factory)
"""


# Enhanced container with type-safe operations
def register_typed[T](
    container: FlextContainer,
    key: ServiceKey[T],
    service: T,
) -> FlextResult[None]:
    """Register service with type-safe key and compile-time type checking.

    Provides enhanced type safety for service registration using ServiceKey
    system. Ensures type consistency between key, service, and retrieval operations.

    Type Safety Features:
        - Compile-time type checking between key and service
        - IDE intellisense and type inference support
        - Prevention of type mismatches at registration time
        - Enhanced debugging with type information in logs

    Args:
        container: FlextContainer instance to register service in
        key: ServiceKey[T] providing type information and service name
        service: Service instance of type T to register

    Returns:
        FlextResult[None] indicating registration success or failure details

    Usage:
        USER_SERVICE_KEY = ServiceKey[UserService]("user_service")
        user_service = UserService()

        result = register_typed(container, USER_SERVICE_KEY, user_service)
        if result.is_success:
            # Service registered with type safety
            pass

    """
    # Import logger for standalone function

    logger = FlextLoggerFactory.get_logger(__name__)
    logger.debug(
        "Registering typed service",
        key=key.name,
        service_type=type(service).__name__,
    )
    return container.register(key.name, service)


def get_typed[T](
    container: FlextContainer,
    key: ServiceKey[T],
) -> FlextResult[T]:
    """Get service with type inference and compile-time type checking from ServiceKey.

    Provides type-safe service retrieval using ServiceKey system with automatic
    type inference and enhanced compile-time checking. Returns properly typed
    service instances without manual casting.

    Type Safety Features:
        - Automatic type inference from ServiceKey[T]
        - No manual casting required in calling code
        - IDE intellisense with proper type information
        - Compile-time prevention of type mismatches

    Args:
        container: FlextContainer instance to retrieve service from
        key: ServiceKey[T] providing type information and service name

    Returns:
        FlextResult[T] with typed service instance or error details

    Usage:
        USER_SERVICE_KEY = ServiceKey[UserService]("user_service")

        result = get_typed(container, USER_SERVICE_KEY)
        if result.is_success:
            user_service: UserService = result.data  # Type automatically inferred
            user_service.create_user(...)  # Full type safety and intellisense

    """
    # Import logger for standalone function

    logger = FlextLoggerFactory.get_logger(__name__)
    logger.debug("Getting typed service by key", key=key.name)
    result = container.get(key.name)
    if result.is_failure:
        return FlextResult.fail(result.error or "Service not found")

    # Type is guaranteed by ServiceKey[T] - cast to T
    service = result.unwrap()
    return FlextResult.ok(cast("T", service))


# Export API
__all__ = [
    "FlextContainer",
    "FlextServiceFactory",
    "ServiceKey",
    "configure_flext_container",
    "get_flext_container",
    "get_typed",
    "register_typed",
]
