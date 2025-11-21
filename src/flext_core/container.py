"""Dependency injection container for service management.

This module provides FlextContainer, a type-safe dependency injection container
for managing service lifecycles and resolving dependencies throughout the
FLEXT ecosystem.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import inspect
import threading
from collections.abc import Callable, Mapping
from contextlib import suppress
from typing import Self, TypeGuard, cast, override

from dependency_injector.containers import DynamicContainer
from dependency_injector.providers import Object, Singleton

from flext_core._utilities.validation import FlextUtilitiesValidation
from flext_core.config import FlextConfig
from flext_core.constants import FlextConstants
from flext_core.exceptions import FlextExceptions
from flext_core.models import FlextModels
from flext_core.protocols import FlextProtocols
from flext_core.result import FlextResult
from flext_core.runtime import FlextRuntime
from flext_core.typings import FactoryT, T


class FlextContainer(FlextProtocols.Configurable):
    """Type-safe dependency injection container for service management.

    Implements FlextProtocols.Configurable through protocol inheritance.
    All container instances satisfy the Configurable protocol contract by
    implementing the required methods: configure() and get_config().

    Provides centralized service management with singleton pattern support,
    type-safe registration and resolution, and automatic dependency injection
    via constructor inspection.

    Protocol Compliance:
        - configure(config: dict) -> FlextResult[bool] - Configure component settings
        - get_config() -> dict - Get current configuration
        - Automatic singleton pattern enforcement

    Features:
        - Type-safe service registration and resolution
        - Singleton and factory pattern support
        - Automatic dependency injection via constructor inspection
        - Batch operations with rollback on failure
        - Thread-safe singleton access via double-checked locking
        - Integration with FlextConfig for container configuration
        - FlextResult-based error handling for all operations
        - Advanced DI features via dependency-injector integration

    Nested Protocols:
        - Service Registry Pattern - Thread-safe service/factory tracking
        - Factory Resolution - Lazy singleton pattern with caching
        - Dependency Autowiring - Constructor inspection and injection
        - Configuration Management - FlextConfig synchronization

    Advanced Patterns:
        - safe_register_from_factory() - from_callable for safe factories
        - get_typed_with_recovery() - Lash pattern for error recovery with type preservation
        - validate_and_get() - flow_through for resolution pipelines

    BREAKING CHANGES (Phase 4 - v0.9.9):
        - register[T]() now uses generic type T instead of object
        - get_typed[T]() now returns FlextResult[T] instead of FlextResult[object]
        - _validate_service_type[T]() now returns FlextResult[T] instead of FlextResult[object]
        - get_typed_with_recovery[T]() now returns FlextResult[T] instead of FlextResult[object]
        - REQUIRED: Use get_typed[T](name, type_cls) for type-safe service retrieval

    Usage Example (Fluent Interface - v2.0.0+):
        >>> from flext_core import FlextContainer, FlextLogger
        >>>
        >>> # Fluent interface - method chaining (RECOMMENDED)
        >>> container = (
        ...     FlextContainer()
        ...     .with_config({"max_workers": 8, "timeout_seconds": 30})
        ...     .with_service("logger", FlextLogger(__name__))
        ...     .with_factory("database", create_database)
        ... )
        >>>
        >>> # Retrieve service (returns FlextResult)
        >>> logger_result = container.get("logger")
        >>> if logger_result.is_success:
        ...     logger = logger_result.unwrap()
        >>>
        >>> # Type-safe retrieval with generic type preservation
        >>> typed_logger_result: FlextResult[FlextLogger] = container.get_typed(
        ...     "logger", FlextLogger
        ... )
        >>> if typed_logger_result.is_success:
        ...     logger: FlextLogger = typed_logger_result.unwrap()

    Instance Compliance Verification:
        >>> from flext_core import FlextContainer, FlextProtocols
        >>> container = FlextContainer()
        >>> isinstance(container, FlextProtocols.Configurable)
        True  # Container instances satisfy Configurable protocol
    """

    _global_instance: FlextContainer | None = None
    _global_lock: threading.RLock = threading.RLock()

    def __new__(cls) -> Self:
        """Create or return the global singleton instance using double-checked locking."""
        if cls._global_instance is None:
            with cls._global_lock:
                if cls._global_instance is None:
                    # Create instance directly without recursion
                    instance = super().__new__(cls)
                    cls._global_instance = instance
        return cast("Self", cls._global_instance)

    def __init__(self) -> None:
        """Initialize container with optimized data structures and internal DI.

        Note: This method is called only once for the singleton instance.
        Subsequent calls to FlextContainer() return the same instance.
        """
        if hasattr(self, "_di_container"):
            return

        super().__init__()

        # Initialize dependency injection containers and providers first
        self.containers = FlextRuntime.dependency_containers()
        self.providers = FlextRuntime.dependency_providers()

        # Internal dependency-injector container (NEW v1.1.0)
        # Provides advanced DI features while maintaining backward compatibility
        self._di_container = self.containers.DynamicContainer()

        # Core service storage with type safety using Pydantic Models
        # ServiceRegistration tracks service metadata (registration_time, tags, type info)
        # FactoryRegistration tracks factory metadata (is_singleton, invocation_count, cached_instance)
        self._services: dict[str, FlextModels.ServiceRegistration] = {}
        self._factories: dict[str, FlextModels.FactoryRegistration] = {}

        # Container configuration using Pydantic Model
        # Replaces dict[str, object] with typed ContainerConfig
        self._global_config: FlextModels.ContainerConfig = (
            self._create_container_config()
        )
        self._user_overrides: dict[str, object] = {}

        # Sync FlextConfig to internal DI container
        self._sync_config_to_di()

    @property
    def config(self) -> FlextConfig:
        """Standard config access property.

        Provides unified access to FlextConfig. Subprojects can override
        this to return typed config with namespace access.

        Returns:
            FlextConfig: Current global configuration instance

        """
        return FlextConfig.get_global_instance()

    @property
    def _flext_config(self) -> FlextConfig:
        """Get current global FlextConfig singleton.

        DEPRECATED: Use self.config instead.

        Always returns the current global config instance, ensuring the
        container stays in sync even after FlextConfig.reset_global_instance()
        or upgrades to derived classes (FlextConfig, FlextBase.Config).

        This property pattern prevents stale config references in tests and
        ensures proper singleton behavior across the ecosystem.

        Returns:
            FlextConfig: Current global configuration instance

        """
        return self.config

    def _create_container_config(self) -> FlextModels.ContainerConfig:
        """Create container configuration from FlextConfig defaults using Pydantic Model.

        Returns:
            FlextModels.ContainerConfig: Typed container configuration with validation.

        """
        # Use ContainerConfig Model with defaults
        return FlextModels.ContainerConfig(
            enable_singleton=True,
            enable_factory_caching=True,
            max_services=1000,
            max_factories=500,
            validation_mode="strict",
            enable_auto_registration=False,
            enable_lifecycle_hooks=True,
            lazy_loading=True,
        )

    def _sync_config_to_di(self) -> None:
        """Sync FlextConfig to internal dependency-injector container.

        Uses FlextConfig's DI Configuration provider which is linked to
        Pydantic 2 BaseSettings, implementing the pattern from:
        https://python-dependency-injector.ets-labs.org/providers/configuration.html

        This enables:
        - Bidirectional sync between Pydantic and DI Configuration
        - Configuration values injectable through DI container
        - Pydantic validation preserved
        - FlextConfig as the single source of truth

        Note:
            Added in v1.1.0 as part of internal DI wrapper implementation.
            Enhanced to use FlextConfig's DI provider for proper Pydantic integration.
            If config provider is not ready during initialization, sync is deferred.

        """
        # Use FlextConfig's DI Configuration provider
        # This provider is linked to Pydantic settings and automatically
        # syncs configuration values
        # If config provider is not ready (e.g., during initialization),
        # skip sync - it will be synced on first use or manually via configure()
        # This is intentional to prevent deadlock during container initialization
        with suppress(Exception):
            config_provider = FlextConfig.get_di_config_provider()
            cast("DynamicContainer", self._di_container).config = config_provider

    # =========================================================================
    # CONFIGURABLE PROTOCOL IMPLEMENTATION - Protocol compliance for 1.0.0
    # =========================================================================

    @override
    def configure(self, config: dict[str, object] | FlextConfig) -> FlextResult[bool]:
        """Configure component with provided settings - Configurable protocol implementation.

        Args:
            config: Either a dict for backward compatibility or FlextConfig instance.
                   Using FlextConfig is preferred.

        Returns:
            FlextResult[bool]: Success with True if configured, failure with error details

        """
        return self.configure_container(config)

    @property
    def services(self) -> dict[str, FlextModels.ServiceRegistration]:
        """Get registered services with metadata (ServiceRegistration Models).

        Returns:
            dict[str, FlextModels.ServiceRegistration]: Dictionary mapping service names
            to ServiceRegistration Models containing service instances and metadata.

        """
        return self._services.copy()

    @property
    def factories(self) -> dict[str, FlextModels.FactoryRegistration]:
        """Get registered factories with metadata (FactoryRegistration Models).

        Returns:
            dict[str, FlextModels.FactoryRegistration]: Dictionary mapping factory names
            to FactoryRegistration Models containing factory callables and metadata.

        """
        return self._factories.copy()

    def get_config(self) -> FlextModels.ContainerConfig:
        """Get current configuration - Configurable protocol implementation.

        Returns:
            FlextModels.ContainerConfig: Typed container configuration Model.

        """
        self._refresh_global_config()
        return self._global_config

    # =========================================================================
    # CORE SERVICE MANAGEMENT - Primary operations with railway patterns
    # =========================================================================

    def with_service(
        self,
        name: str,
        service: object,
    ) -> Self:
        """Register a service with fluent interface - returns self for chaining.

        Use this method to register services for dependency injection throughout
        FLEXT applications. Returns self for method chaining (fluent interface).
        Raises exception on failure for fail-fast behavior.

        Args:
            name: Service name for later retrieval
            service: Service instance to register

        Returns:
            Self: Container instance for method chaining

        Raises:
            ValueError: If service name is invalid or service already registered

        Example:
            ```python
            from flext_core import FlextContainer, FlextLogger

            container = (
                FlextContainer()
                .with_service("logger", FlextLogger(__name__))
                .with_service("database", DatabaseService())
            )
            ```

        """
        result = FlextUtilitiesValidation.validate_identifier(name).flat_map(
            lambda validated_name: self._store_service(validated_name, service),
        )
        if result.is_failure:
            base_msg = f"Failed to register service '{name}'"
            error_msg = (
                f"{base_msg}: {result.error}"
                if result.error
                else f"{base_msg} (validation or storage failed)"
            )
            raise ValueError(error_msg)
        return self

    def _store_service(self, name: str, service: object) -> FlextResult[bool]:
        """Store service in registry using ServiceRegistration Model.

        Creates ServiceRegistration with metadata (registration_time, service_type, tags)
        and stores in internal DI container. Type T is preserved for static typing.

        Returns:
            FlextResult[bool]: Success with True if stored, failure with error details.

        Note:
            Updated in v1.1.0 to use ServiceRegistration Pydantic Model.

        """
        if name in self._services:
            return FlextResult[bool].fail(f"Service '{name}' already registered")

        provider_registered = False
        try:
            # Create ServiceRegistration Model with metadata
            service_registration = FlextModels.ServiceRegistration(
                name=name,
                service=service,
                service_type=type(service).__name__,
                tags=[],
                metadata=FlextModels.Metadata(attributes={}),
            )

            # Store ServiceRegistration Model in registry
            self._services[name] = service_registration

            # Store in internal DI container using Object provider
            # Object provider is for existing instances (singletons by nature)
            provider = Object(service)
            cast("DynamicContainer", self._di_container).set_provider(name, provider)
            provider_registered = True

            return FlextResult[bool].ok(True)
        except (TypeError, AttributeError, ValueError) as e:
            # Rollback on failure - only delete if we successfully registered it
            # TypeError/AttributeError: set_provider() or delattr() failures
            # ValueError: Validation or constraint violations
            if provider_registered and hasattr(self._di_container, name):
                delattr(self._di_container, name)
            self._services.pop(name, None)
            return FlextResult[bool].fail(f"Service storage failed: {e}")

    def with_factory(
        self,
        name: str,
        factory: Callable[[], FactoryT],
    ) -> Self:
        """Register service factory with fluent interface - returns self for chaining.

        Type T is preserved for static type checking and inferred from the
        factory callable's return type at registration time.
        Raises exception on failure for fail-fast behavior.

        Args:
            name: Service name for later retrieval
            factory: Factory callable that creates service instances

        Returns:
            Self: Container instance for method chaining

        Raises:
            ValueError: If service name is invalid or factory already registered

        Example:
            ```python
            container = (
                FlextContainer()
                .with_factory("logger", lambda: FlextLogger(__name__))
                .with_factory("database", create_database)
            )
            ```

        """
        result = FlextUtilitiesValidation.validate_identifier(name).flat_map(
            lambda validated_name: self._store_factory(validated_name, factory),
        )
        if result.is_failure:
            base_msg = f"Failed to register factory '{name}'"
            error_msg = (
                f"{base_msg}: {result.error}"
                if result.error
                else f"{base_msg} (validation or storage failed)"
            )
            raise ValueError(error_msg)
        return self

    def _store_factory(
        self,
        name: str,
        factory: Callable[[], FactoryT],
    ) -> FlextResult[bool]:
        """Store factory in registry using FactoryRegistration Model.

        Creates FactoryRegistration with metadata (is_singleton, invocation_count, cached_instance)
        and stores in internal DI container using Singleton provider.
        Type T is preserved for static type checking.

        Returns:
            FlextResult[bool]: Success with True if stored, failure with error details.

        Note:
            Updated in v1.1.0 to use FactoryRegistration Pydantic Model.

        """
        # Validate that factory is actually callable
        if not callable(factory):
            return FlextResult[bool].fail(f"Factory '{name}' must be callable")

        if name in self._factories:
            return FlextResult[bool].fail(f"Factory '{name}' already registered")

        provider_registered = False
        try:
            # Create FactoryRegistration Model with metadata
            factory_registration = FlextModels.FactoryRegistration(
                name=name,
                factory=factory,
                is_singleton=True,  # All factories are singletons by default
                cached_instance=None,  # Will be populated on first invocation
                metadata=FlextModels.Metadata(attributes={}),
                invocation_count=0,
            )

            # Store FactoryRegistration Model in registry
            self._factories[name] = factory_registration

            # Store in internal DI container using Singleton with factory
            # This creates lazy singleton: factory called once, result cached
            provider = Singleton(cast("Callable[[], object]", factory))
            cast("DynamicContainer", self._di_container).set_provider(name, provider)
            provider_registered = True

            return FlextResult[bool].ok(True)
        except (TypeError, AttributeError, ValueError) as e:
            # Rollback on failure - only delete if we successfully registered it
            # TypeError/AttributeError: set_provider() or delattr() failures
            # ValueError: Validation or constraint violations
            self._factories.pop(name, None)
            if provider_registered and hasattr(self._di_container, name):
                delattr(self._di_container, name)
            return FlextResult[bool].fail(f"Factory storage failed: {e}")

    def unregister(self, name: str) -> FlextResult[bool]:
        """Unregister service or factory with validation.

        Returns:
            FlextResult[bool]: Success with True if unregistered, failure with error details.

        """
        return FlextUtilitiesValidation.validate_identifier(name).flat_map(
            self._remove_service,
        )

    def _remove_service(self, name: str) -> FlextResult[bool]:
        """Remove service from tracking dicts AND internal DI container.

        Returns:
            FlextResult[bool]: Success with True if removed, failure with error details.

        Note:
            Updated in v1.1.0 to remove from DI container as well.

        """
        service_found = name in self._services
        factory_found = name in self._factories

        if not service_found and not factory_found:
            return FlextResult[bool].fail(f"Service '{name}' not registered")

        # Remove from tracking dicts (backward compatibility)
        self._services.pop(name, None)
        self._factories.pop(name, None)

        # Remove from internal DI container (access as attribute)
        # Check existence before deletion to avoid AttributeError
        if hasattr(self._di_container, name):
            delattr(self._di_container, name)

        return FlextResult[bool].ok(True)

    def get(self, name: str) -> FlextResult[object]:
        """Get service with factory resolution and caching.

        Returns:
            FlextResult[object]: Success with service instance or failure with error.

        """
        return FlextUtilitiesValidation.validate_identifier(name).flat_map(
            self._resolve_service,
        )

    def _resolve_service(self, name: str) -> FlextResult[object]:
        """Resolve service via internal DI container with FlextResult wrapping.

        Returns actual service instance from ServiceRegistration or FactoryRegistration.
        Updates factory metadata (cached_instance, invocation_count) on first call.

        Returns:
            FlextResult[object]: Success with service instance or failure with error.

        Note:
            Updated in v1.1.0 to use ServiceRegistration/FactoryRegistration Models.

        """
        try:
            # Resolve via internal DI container (access as attribute, not dict)
            # DynamicContainer allows attribute-style access: container.service_name()
            if hasattr(self._di_container, name):
                provider = getattr(self._di_container, name)
                service = provider()

                # Update factory metadata if this is a factory
                if name in self._factories:
                    factory_reg = self._factories[name]
                    # Update invocation count
                    factory_reg.invocation_count += 1
                    # Cache the instance on first invocation
                    if factory_reg.cached_instance is None:
                        factory_reg.cached_instance = service

                # Integration: Track successful service resolution
                FlextRuntime.Integration.track_service_resolution(name, resolved=True)

                return FlextResult[object].ok(service)

            # Integration: Track service not found
            FlextRuntime.Integration.track_service_resolution(
                name,
                resolved=False,
                error_message=f"Service '{name}' not found",
            )

            return FlextResult[object].fail(f"Service '{name}' not found")
        except (TypeError, ValueError, AttributeError, KeyError, RuntimeError) as e:
            # TypeError/ValueError/RuntimeError: Factory invocation or provider callable failures
            # AttributeError: Attribute access on provider object
            # KeyError: Dict access on internal structures
            # RuntimeError: Runtime failures from factory execution
            error_msg = (
                f"Factory '{name}' failed: {e}"
                if name in self._factories
                else f"Service resolution failed: {e}"
            )

            # Integration: Track resolution failure
            FlextRuntime.Integration.track_service_resolution(
                name,
                resolved=False,
                error_message=error_msg,
            )

            return FlextResult[object].fail(error_msg)

    def _invoke_factory_and_cache(self, name: str) -> FlextResult[object]:
        """Invoke factory from FactoryRegistration and update metadata.

        Returns:
            FlextResult[object]: Success with service instance or failure with error.

        """
        try:
            factory_reg = self._factories[name]
            # Invoke factory from FactoryRegistration
            service = factory_reg.factory()

            # Update metadata
            factory_reg.invocation_count += 1
            factory_reg.cached_instance = service

            return FlextResult[object].ok(service)
        except (KeyError, TypeError, ValueError, RuntimeError) as e:
            # KeyError: Factory not found in _factories dict
            # TypeError: Factory not callable or wrong number of arguments
            # ValueError: Factory validation or constraint violations
            # RuntimeError: Factory execution failures
            return FlextResult[object].fail(f"Factory '{name}' failed: {e}")

    def get_typed(self, name: str, expected_type: type[T]) -> FlextResult[T]:
        """Get service with type validation and generic return type preservation.

        Retrieves a service and validates it matches the expected type.
        Return type is properly preserved as FlextResult[T] for type safety.

        Args:
            name: Service name to retrieve
            expected_type: Expected service type for validation (used for runtime type checking)

        Returns:
            FlextResult[T]: Success with typed service or failure with error.

        """
        return self.get(name).flat_map(
            lambda service: self._validate_service_type(service, expected_type),
        )

    def _validate_service_type(
        self,
        service: object,
        expected_type: type[T],
    ) -> FlextResult[T]:
        """Validate service type and return with generic type preservation.

        Returns:
            FlextResult[T]: Success with validated typed service or failure with error.

        """
        if not isinstance(service, expected_type):
            return FlextResult[T].fail(
                f"Service type mismatch: expected {getattr(expected_type, '__name__', str(expected_type))}, got {getattr(type(service), '__name__', str(type(service)))}",
            )
        return FlextResult[T].ok(service)

    # =========================================================================
    # BATCH OPERATIONS - Efficient bulk service management
    # =========================================================================

    def batch_register(self, services: dict[str, object]) -> FlextResult[bool]:
        """Register multiple services atomically with rollback on failure.

        Uses railway pattern with automatic rollback and FlextUtilities validation.

        Returns:
            FlextResult[bool]: Success with True if all registered, failure with error details.

        """
        # Use railway pattern for atomic batch registration with rollback
        return self._validate_batch_services(services).flat_map(
            self._execute_batch_registration,
        )

    def _validate_batch_services(
        self,
        services: dict[str, object],
    ) -> FlextResult[dict[str, object]]:
        """Validate batch services using FlextUtilities."""
        # Allow empty dictionaries for batch_register flexibility

        # Use FlextUtilities for comprehensive validation
        validation_result = FlextUtilitiesValidation.validate_batch_services(services)
        if validation_result.is_failure:
            return FlextResult[dict[str, object]].fail(
                f"Batch validation failed: {validation_result.error}",
            )

        return FlextResult[dict[str, object]].ok(services)

    def _execute_batch_registration(
        self,
        services: dict[str, object],
    ) -> FlextResult[bool]:
        """Execute batch registration with automatic rollback using snapshot pattern."""
        # Create snapshot for rollback capability
        snapshot = self._create_registry_snapshot()

        try:
            # Process all registrations using railway pattern
            return self._process_batch_registrations(services).map(lambda _: True)
        except Exception as e:
            # Fallback rollback for unexpected exceptions
            return self._rollback_and_fail(snapshot, str(e))

    def _rollback_and_fail(
        self,
        snapshot: Mapping[str, object],
        error: str,
    ) -> FlextResult[bool]:
        """Rollback snapshot and return failure result."""
        typed_snapshot = cast(
            "dict[str, dict[str, FlextModels.ServiceRegistration] | dict[str, FlextModels.FactoryRegistration]]",
            snapshot,
        )
        self._restore_registry_snapshot(typed_snapshot)
        return FlextResult[bool].fail(f"Batch registration failed: {error}")

    def _create_registry_snapshot(
        self,
    ) -> dict[
        str,
        dict[str, FlextModels.ServiceRegistration]
        | dict[str, FlextModels.FactoryRegistration],
    ]:
        """Create snapshot of current registry state for rollback.

        Returns:
            dict containing services and factories with their specific Model types.

        """
        return {
            "services": self._services.copy(),
            "factories": self._factories.copy(),
        }

    def _process_batch_registrations(
        self,
        services: dict[str, object],
    ) -> FlextResult[bool]:
        """Process batch registrations with proper error handling.

        Returns:
            FlextResult[bool]: Success with True if all processed, failure with error details.

        """
        for name, service in services.items():
            # Validate service name
            validation_result = FlextUtilitiesValidation.validate_identifier(name)
            if validation_result.is_failure:
                base_msg = f"Invalid service name: {name}"
                error_msg = (
                    f"{base_msg}: {validation_result.error}"
                    if validation_result.error
                    else f"{base_msg} (validation failed)"
                )
                return FlextResult[bool].fail(error_msg)

            # Store the service
            storage_result = self._store_service(name, service)
            if storage_result.is_failure:
                base_msg = f"Failed to store service: {name}"
                error_msg = (
                    f"{base_msg}: {storage_result.error}"
                    if storage_result.error
                    else f"{base_msg} (storage operation failed)"
                )
                return FlextResult[bool].fail(error_msg)

        return FlextResult[bool].ok(True)

    @staticmethod
    def _is_service_registration_dict(
        value: dict[str, FlextModels.ServiceRegistration]
        | dict[str, FlextModels.FactoryRegistration]
        | None,
    ) -> TypeGuard[dict[str, FlextModels.ServiceRegistration]]:
        """Type guard to check if dict contains ServiceRegistration values."""
        if not isinstance(value, dict) or not value:
            return False
        first_value = next(iter(value.values()), None)
        return first_value is not None and isinstance(
            first_value,
            FlextModels.ServiceRegistration,
        )

    @staticmethod
    def _is_factory_registration_dict(
        value: dict[str, FlextModels.ServiceRegistration]
        | dict[str, FlextModels.FactoryRegistration]
        | None,
    ) -> TypeGuard[dict[str, FlextModels.FactoryRegistration]]:
        """Type guard to check if dict contains FactoryRegistration values."""
        if not isinstance(value, dict) or not value:
            return False
        first_value = next(iter(value.values()), None)
        return first_value is not None and isinstance(
            first_value,
            FlextModels.FactoryRegistration,
        )

    def _restore_registry_snapshot(
        self,
        snapshot: dict[
            str,
            dict[str, FlextModels.ServiceRegistration]
            | dict[str, FlextModels.FactoryRegistration],
        ],
    ) -> None:
        """Restore registry state from snapshot with Models."""
        services_snapshot = snapshot.get("services")
        factories_snapshot = snapshot.get("factories")

        # Type-safe restoration using type guards for union type narrowing
        if self._is_service_registration_dict(services_snapshot):
            self._services = services_snapshot
        if self._is_factory_registration_dict(factories_snapshot):
            self._factories = factories_snapshot

    # =========================================================================
    # ADVANCED FEATURES - Factory resolution and dependency injection
    # =========================================================================

    def get_or_create(
        self,
        name: str,
        factory: Callable[[], T] | None = None,
    ) -> FlextResult[object]:
        """Get existing service or create using provided factory.

        Returns:
            FlextResult[object]: Success with service instance or failure with error.

        """
        service_result: FlextResult[object] = self.get(name)

        if service_result.is_success:
            return service_result

        if factory is None:
            return FlextResult[object].fail(
                f"Service '{name}' not found and no factory provided",
            )

        return self._create_from_factory(name, factory)

    def _create_from_factory(
        self,
        name: str,
        factory: Callable[[], T],
    ) -> FlextResult[object]:
        """Create service from factory and register it.

        Returns:
            FlextResult[object]: Success with service instance or failure with error.

        """
        try:
            self.with_factory(name, factory)
        except ValueError as e:
            return FlextResult[object].fail(str(e))

        service_result = self.get(name)
        if service_result.is_success:
            # Type assertion since we know the service type matches
            service = service_result.unwrap()
            return FlextResult[object].ok(cast("T", service))
        # Fast fail: error must be str (FlextResult guarantees this)
        error_msg = service_result.error
        if error_msg is None:
            msg = "Service result is failure but error is None"
            return FlextResult[object].fail(msg)
        return FlextResult[object].fail(error_msg)

    def create_service(
        self,
        service_class: type[T],
        service_name: str | None = None,
    ) -> FlextResult[object]:
        """Create and register service with dependencies from container.

        Returns:
            FlextResult[object]: Success with instantiated service or failure with error.

        """
        try:
            # Determine service name
            if service_name is None:
                name = getattr(service_class, "__name__", "").lower()
                if not name:
                    return FlextResult[object].fail("Cannot determine service name")
            else:
                name = service_name

            # Try to resolve dependencies from container
            dependencies: dict[str, object] = {}
            signature = inspect.signature(service_class.__init__)

            for param_name, param in signature.parameters.items():
                if param_name == "self":
                    continue

                # Try to get dependency from container
                dep_result = self.get(param_name)
                if dep_result.is_success:
                    dependencies[param_name] = dep_result.value
                elif param.default != inspect.Parameter.empty:
                    # Use default value
                    dependencies[param_name] = param.default
                else:
                    return FlextResult[object].fail(
                        f"Cannot resolve required dependency '{param_name}'",
                    )

            # Create service instance
            service = service_class(**dependencies)

            # Register the service
            try:
                self.with_service(name, service)
            except ValueError as e:
                return FlextResult[object].fail(str(e))

            return FlextResult[object].ok(service)

        except (TypeError, ValueError, AttributeError, KeyError) as e:
            # TypeError: Service class instantiation with wrong arguments
            # ValueError: Validation or constraint violations
            # AttributeError: Missing attributes on service class or dependency
            # KeyError: Dependency resolution failures
            return FlextResult[object].fail(f"Service creation failed: {e}")

    def auto_wire(
        self,
        service_class: type[T],
    ) -> FlextResult[object]:
        """Auto-wire service dependencies without registering.

        Uses railway pattern with FlextUtilities for robust dependency resolution.

        Returns:
            FlextResult[object]: Success with instantiated service or failure with error.

        """
        return (
            self._analyze_constructor_signature(service_class)
            .flat_map(self._resolve_dependencies)
            .flat_map(lambda deps: self._instantiate_service(service_class, deps))
        )

    def _analyze_constructor_signature(
        self,
        service_class: type[T],
    ) -> FlextResult[dict[str, dict[str, object]]]:
        """Analyze constructor signature for dependency requirements."""
        try:
            signature = inspect.signature(service_class.__init__)
            dependencies: dict[str, dict[str, object]] = {}

            for param_name, param in signature.parameters.items():
                if param_name == "self":
                    continue

                # Use FlextUtilities for parameter analysis
                param_info = FlextUtilitiesValidation.analyze_constructor_parameter(
                    param_name,
                    param,
                )
                dependencies[param_name] = param_info

            return FlextResult[dict[str, dict[str, object]]].ok(dependencies)

        except Exception as e:
            return FlextResult[dict[str, dict[str, object]]].fail(
                f"Signature analysis failed: {e}",
            )

    def _resolve_dependencies(
        self,
        param_specs: dict[str, dict[str, object]],
    ) -> FlextResult[dict[str, object]]:
        """Resolve dependencies from container using railway pattern."""
        dependencies = {}
        resolution_errors = []

        for param_name, param_spec in param_specs.items():
            # Extract parameter info
            param_dict = param_spec
            has_default = param_dict.get("has_default", False)
            default_value = param_dict.get("default_value")

            # Try to resolve from container
            dep_result = self.get(param_name)

            if dep_result.is_success:
                dependencies[param_name] = dep_result.value
            elif has_default:
                dependencies[param_name] = default_value
            else:
                resolution_errors.append(param_name)

        if resolution_errors:
            return FlextResult[dict[str, object]].fail(
                f"Cannot resolve required dependencies: {', '.join(resolution_errors)}",
            )

        return FlextResult[dict[str, object]].ok(dependencies)

    def _instantiate_service(
        self,
        service_class: type[T],
        dependencies: dict[str, object],
    ) -> FlextResult[object]:
        """Instantiate service with resolved dependencies."""
        try:
            service = service_class(**dependencies)
            return FlextResult[object].ok(service)
        except Exception as e:
            return FlextResult[object].fail(f"Service instantiation failed: {e}")

    # =========================================================================
    # INSPECTION AND UTILITIES - Container introspection and management
    # =========================================================================

    def clear(self) -> FlextResult[bool]:
        """Clear all services and factories.

        Note: _flext_config is now a property that always returns the current
        global config, so no need to manually refresh it here.

        Returns:
            FlextResult[bool]: Success with True if cleared, failure with error details.

        """
        try:
            self._services.clear()
            self._factories.clear()
            # Reset DI container to clear all providers
            self._di_container = self.containers.DynamicContainer()
            return FlextResult[bool].ok(True)
        except (TypeError, AttributeError, ValueError, RuntimeError) as e:
            # TypeError: DynamicContainer instantiation failures
            # AttributeError: Clear() method failures or attribute access issues
            # ValueError: Validation or constraint violations
            # RuntimeError: Runtime failures during clearing
            return FlextResult[bool].fail(f"Failed to clear container: {e}")

    def has(self, name: str) -> bool:
        """Check if service is registered.

        Returns:
            bool: True if service is registered, False otherwise.

        """
        normalized = FlextUtilitiesValidation.validate_identifier(name)
        if normalized.is_failure:
            return False
        try:
            validated_name = normalized.unwrap()
            return validated_name in self._services or validated_name in self._factories
        except Exception:
            return False

    def list_services(
        self,
    ) -> FlextResult[
        list[FlextModels.ServiceRegistration | FlextModels.FactoryRegistration]
    ]:
        """List all registered services and factories with full metadata.

        Returns:
            FlextResult with list of ServiceRegistration and FactoryRegistration Models.

        """
        try:
            # Use list() for efficient copy (PERF402 compliant)
            services: list[
                FlextModels.ServiceRegistration | FlextModels.FactoryRegistration
            ] = [
                *list(self._services.values()),
                *list(self._factories.values()),
            ]

            return FlextResult[
                list[FlextModels.ServiceRegistration | FlextModels.FactoryRegistration]
            ].ok(services)
        except (TypeError, AttributeError, ValueError, RuntimeError) as e:
            # TypeError: Type casting or collection operation failures
            # AttributeError: Model attribute access failures
            # ValueError: Constraint violations
            # RuntimeError: Runtime failures during service listing
            return FlextResult[
                list[FlextModels.ServiceRegistration | FlextModels.FactoryRegistration]
            ].fail(
                f"Failed to list services: {e}",
            )

    def _build_service_info(
        self,
        name: str,
        service: object,
        service_type: str,
    ) -> dict[str, object]:
        """Build service information dictionary.

        Returns:
            dict[str, object]: Service information dictionary.

        """
        # Fast fail: service inspection must succeed or fail explicitly
        # No fallback to fake data - raise exception if inspection fails
        try:
            service_class = service.__class__
            service_class_name = service_class.__name__
            service_module = getattr(
                service_class,
                "__module__",
                None,
            )
            # Fast fail: module must exist for valid service
            if service_module is None:
                msg = f"Service {name} has no __module__ attribute"
                raise FlextExceptions.AttributeAccessError(
                    message=msg,
                    attribute_name="__module__",
                    attribute_context={
                        "service_name": name,
                        "service_type": service_type,
                    },
                )
            return {
                FlextConstants.Mixins.FIELD_NAME: name,
                FlextConstants.Mixins.FIELD_TYPE: service_type,
                FlextConstants.Mixins.FIELD_CLASS: service_class_name,
                FlextConstants.Mixins.FIELD_MODULE: service_module,
                "is_callable": callable(service),
                "id": id(service),
            }
        except (TypeError, AttributeError, RuntimeError) as e:
            # Fast fail: service inspection failures indicate invalid service
            msg = f"Failed to inspect service {name}: {type(e).__name__}: {e}"
            raise FlextExceptions.TypeError(
                message=msg,
                expected_type="service with inspectable attributes",
                actual_type=type(service).__name__,
            ) from e

    # =========================================================================
    # CONFIGURATION MANAGEMENT - FlextConfig integration
    # =========================================================================

    def configure_container(
        self,
        config: dict[str, object] | FlextConfig,
    ) -> FlextResult[bool]:
        """Configure container using ContainerConfig Model.

        Args:
            config: Either a dict for backward compatibility or FlextConfig instance.

        Returns:
            FlextResult[bool]: Success with True if configured, failure with error details.

        """
        try:
            # Update user overrides (for compatibility)
            if isinstance(config, FlextConfig):
                self._user_overrides.update({
                    "max_workers": config.max_workers,
                    "timeout_seconds": config.timeout_seconds,
                })
            else:
                self._user_overrides.update(config)

            # Refresh global config (recreates ContainerConfig Model)
            self._refresh_global_config()

            return FlextResult[bool].ok(True)
        except (TypeError, ValueError, AttributeError, KeyError, RuntimeError) as e:
            # TypeError: Type conversion failures
            # ValueError: Validation errors
            # AttributeError: Attribute access failures
            # KeyError: Dict key access failures
            # RuntimeError: Runtime failures
            return FlextResult[bool].fail(f"Container configuration failed: {e}")

    def with_config(self, config: dict[str, object] | FlextConfig) -> Self:
        """Configure container with fluent interface - returns self for chaining.

        Args:
            config: Either a dict or FlextConfig instance for configuration.

        Returns:
            Self: Container instance for method chaining

        Raises:
            ValueError: If configuration fails

        Example:
            ```python
            container = (
                FlextContainer()
                .with_config({"max_workers": 8, "timeout_seconds": 30})
                .with_service("logger", logger)
            )
            ```

        """
        result = self.configure_container(config)
        if result.is_failure:
            # Fast fail: error must be str (FlextResult guarantees this)
            error_msg = result.error
            if error_msg is None:
                msg = "Container configuration failed but error is None"
                raise ValueError(msg)
            raise ValueError(error_msg)
        return self

    def _refresh_global_config(self) -> None:
        """Refresh the effective global configuration as ContainerConfig Model."""
        # Recreate ContainerConfig Model (always uses defaults)
        # User overrides are maintained separately for backward compatibility
        self._global_config = self._create_container_config()

    # =========================================================================
    # GLOBAL CONTAINER MANAGEMENT - Singleton pattern with thread safety
    # =========================================================================

    @classmethod
    def ensure_global_instance(cls) -> FlextContainer:
        """Ensure global instance exists with thread safety.

        Returns:
            FlextContainer: The global container instance.

        """
        if cls._global_instance is None:
            with cls._global_lock:
                if cls._global_instance is None:
                    # Create instance using direct instantiation
                    cls._global_instance = cls()
        return cls._global_instance

    @classmethod
    def get_global(cls) -> FlextContainer:
        """Get the global container instance (compatibility method).

        Returns:
            FlextContainer: The global container instance.

        Note:
            This method provides backward compatibility.
            For new code, use FlextContainer() directly.

        """
        return cls.ensure_global_instance()

    @classmethod
    def create_module_utilities(
        cls,
        module_name: str,
    ) -> FlextResult[dict[str, object]]:
        """Create module-specific utilities with container integration.

        Returns:
            FlextResult[dict[str, object]]: Success with utilities dict[str, object] or failure with error.

        """
        if not module_name:
            return FlextResult[dict[str, object]].fail(
                "Module name must be non-empty string",
            )

        container = cls.ensure_global_instance()
        return FlextResult[dict[str, object]].ok({
            "container": container,
            "module": module_name,
            "logger": f"flext.{module_name}",
        })

    def safe_register_from_factory(
        self,
        name: str,
        factory: object,
    ) -> FlextResult[bool]:
        """Register service from factory using from_callable pattern.

        Demonstrates from_callable for safe factory registration.

        Args:
            name: Service name
            factory: Factory callable

        Returns:
            FlextResult[bool]: Success with True, failure with error details

        Example:
            >>> container = FlextContainer.get_global()
            >>> result = container.safe_register_from_factory(
            ...     "logger", lambda: create_logger()
            ... )

        """

        # Use from_callable to safely execute factory
        def _execute_factory() -> object:
            return factory() if callable(factory) else factory

        factory_result: FlextResult[object] = FlextResult[object].from_callable(
            _execute_factory,
        )

        if factory_result.is_failure:
            return FlextResult[bool].fail(
                f"Factory execution failed: {factory_result.error}",
            )

        service: object = factory_result.unwrap()
        try:
            self.with_service(name, service)
            return FlextResult[bool].ok(True)
        except ValueError as e:
            return FlextResult[bool].fail(str(e))

    def get_typed_with_recovery(
        self,
        name: str,
        expected_type: type[T],
        recovery_factory: object | None = None,
    ) -> FlextResult[T]:
        """Get typed service with recovery using lash pattern and type preservation.

        Demonstrates lash for error recovery with factory.
        Return type is properly preserved as FlextResult[T] for type safety.

        Args:
            name: Service name
            expected_type: Expected service type
            recovery_factory: Optional factory to create service if not found

        Returns:
            FlextResult[T]: Service or recovered value with proper type

        Example:
            >>> container = FlextContainer.get_global()
            >>> logger = container.get_typed_with_recovery(
            ...     "logger", LoggerService, lambda: LoggerService()
            ... )

        """

        def try_recovery(_error: str) -> FlextResult[T]:
            if recovery_factory is None:
                return FlextResult[T].fail(
                    f"Service '{name}' not found and no recovery factory provided",
                )

            def _execute_recovery_factory() -> object:
                return (
                    recovery_factory()
                    if callable(recovery_factory)
                    else recovery_factory
                )

            factory_result: FlextResult[object] = FlextResult[object].from_callable(
                _execute_recovery_factory,
            )

            if factory_result.is_failure:
                return FlextResult[T].fail(
                    f"Recovery factory failed: {factory_result.error}",
                )

            service: object = factory_result.unwrap()
            if not isinstance(service, expected_type):
                return FlextResult[T].fail(
                    f"Recovery factory returned wrong type: expected {getattr(expected_type, '__name__', str(expected_type))}, got {getattr(type(service), '__name__', str(type(service)))}",
                )

            # Register the recovered service for future use
            with suppress(ValueError):
                # Ignore registration failure, return recovered service anyway
                self.with_service(name, service)
            return FlextResult[T].ok(service)

        return self.get_typed(name, expected_type).lash(try_recovery)

    def validate_and_get(
        self,
        name: str,
        validators: list[Callable[[object], FlextResult[object]]] | None = None,
    ) -> FlextResult[object]:
        """Get service and validate using flow_through pattern.

        Demonstrates flow_through for service resolution and validation pipeline.

        Args:
            name: Service name
            validators: Optional list of validation functions

        Returns:
            FlextResult with validated service

        Example:
            >>> def validate_not_none(s):
            ...     return (
            ...         FlextResult[object].ok(s)
            ...         if s
            ...         else FlextResult[object].fail("None")
            ...     )
            >>> container = FlextContainer.get_global()
            >>> service = container.validate_and_get("logger", [validate_not_none])

        """

        def apply_validators(service: object) -> FlextResult[object]:
            if not validators:
                return FlextResult[object].ok(service)

            for validator in validators:
                if callable(validator):
                    # Call validator - returns FlextResult[object]
                    result = validator(service)

                    # Ensure result is a FlextResult
                    if not isinstance(result, FlextResult):
                        return FlextResult[object].fail(
                            f"Validator must return FlextResult, got {type(result)}",
                        )

                    validator_result = result

                    # Check if validation failed
                    if validator_result.is_failure:
                        return FlextResult[object].fail(
                            f"Validation failed: {validator_result.error}",
                        )
                    # Continue with validated service

            # All validators passed
            return FlextResult[object].ok(service)

        return self.get(name).flat_map(apply_validators)

    @override
    def __repr__(self) -> str:
        """String representation with service counts.

        Returns:
            str: String representation of the container.

        """
        total_registered = len(set(self._services.keys()) | set(self._factories.keys()))
        return (
            f"FlextContainer(services={len(self._services)}, "
            f"factories={len(self._factories)}, "
            f"total_registered={total_registered})"
        )

    # =========================================================================
    # Protocol Implementation: TypeValidator[T]
    # =========================================================================

    def validate_type(
        self,
        value: object,
        expected_type: type[object],
    ) -> FlextResult[object]:
        """Validate value matches expected type (TypeValidator protocol).

        Part of TypeValidator[T] protocol implementation.

        Args:
            value: Value to validate
            expected_type: Expected type for validation

        Returns:
            FlextResult[object]: Validated value or error

        """
        return self._validate_service_type(value, expected_type)

    def is_valid_type(self, value: object, expected_type: type[object]) -> bool:
        """Check if value is valid for expected type (TypeValidator protocol).

        Part of TypeValidator[T] protocol implementation.

        Args:
            value: Value to check
            expected_type: Expected type

        Returns:
            bool: True if valid, False otherwise

        """
        try:
            result = self._validate_service_type(value, expected_type)
            return result.is_success
        except (TypeError, AttributeError):
            # TypeError: Type validation failures
            # AttributeError: Attribute access failures during validation
            return False

    # =========================================================================
    # Protocol Implementation: ServiceRegistry[T]
    # =========================================================================

    def register_service(self, name: str, service: object) -> FlextResult[bool]:
        """Register service (ServiceRegistry protocol).

        Part of ServiceRegistry[T] protocol implementation.

        Args:
            name: Service identifier
            service: Service instance

        Returns:
            FlextResult[bool]: Success with True if registered, failure with error details

        """
        try:
            self.with_service(name, service)
            return FlextResult[bool].ok(True)
        except ValueError as e:
            return FlextResult[bool].fail(str(e))

    def get_service(self, name: str) -> FlextResult[object]:
        """Retrieve registered service (ServiceRegistry protocol).

        Part of ServiceRegistry[T] protocol implementation.

        Args:
            name: Service identifier

        Returns:
            FlextResult[object]: Service instance or error

        """
        return self.get(name)

    def has_service(self, name: str) -> bool:
        """Check if service is registered (ServiceRegistry protocol).

        Part of ServiceRegistry[T] protocol implementation.

        Args:
            name: Service identifier

        Returns:
            bool: True if registered

        """
        return self.has(name)

    # =========================================================================
    # Protocol Implementation: FactoryProvider[T]
    # =========================================================================

    def create_instance(self) -> FlextResult[object]:
        """Create new instance using factory (FactoryProvider protocol).

        Part of FactoryProvider[T] protocol implementation.
        Creates instance from registered factory.

        Returns:
            FlextResult[object]: Created instance or error

        """
        try:
            # Get any registered factory and invoke it
            if not self._factories:
                return FlextResult[object].fail(
                    "No factories registered",
                    error_code="FACTORY_NOT_FOUND",
                )

            # Use first factory found
            factory_name = next(iter(self._factories.keys()))
            factory_registration = self._factories[factory_name]

            # Extract factory callable from FactoryRegistration
            factory = factory_registration.factory

            if callable(factory):
                instance = factory()
                return FlextResult[object].ok(instance)

            return FlextResult[object].fail(
                f"Factory {factory_name} is not callable",
                error_code="FACTORY_NOT_CALLABLE",
            )
        except (TypeError, ValueError, KeyError) as e:
            # TypeError: Factory invocation failures or type errors
            # ValueError: Validation or constraint violations
            # KeyError: Dict access failures when finding factories
            return FlextResult[object].fail(
                f"Factory creation failed: {e}",
                error_code="FACTORY_ERROR",
                error_data={"exception": str(e)},
            )

    # =========================================================================
    # Protocol Implementation: SingletonProvider[T]
    # =========================================================================

    def get_instance(self) -> FlextResult[object]:
        """Get singleton instance (SingletonProvider protocol).

        Part of SingletonProvider[T] protocol implementation.
        Returns the global singleton instance.

        Returns:
            FlextResult[object]: Singleton instance or error

        """
        try:
            with self._global_lock:
                if self._global_instance is None:
                    self._global_instance = FlextContainer()
                return FlextResult[object].ok(self._global_instance)
        except (TypeError, RuntimeError, AttributeError) as e:
            # TypeError: FlextContainer instantiation failures
            # RuntimeError: Lock acquisition or threading failures
            # AttributeError: Instance attribute access failures
            return FlextResult[object].fail(
                f"Singleton retrieval failed: {e}",
                error_code="SINGLETON_ERROR",
                error_data={"exception": str(e)},
            )

    def reset_instance(self) -> FlextResult[bool]:
        """Reset singleton instance (SingletonProvider protocol).

        Part of SingletonProvider[T] protocol implementation.
        Resets the global singleton for testing purposes only.

        Returns:
            FlextResult[bool]: Success with True if reset, failure with error details

        """
        try:
            with self._global_lock:
                self._global_instance = None
                return FlextResult[bool].ok(True)
        except (RuntimeError, AttributeError) as e:
            # RuntimeError: Lock acquisition or threading failures
            # AttributeError: Instance attribute access failures
            return FlextResult[bool].fail(
                f"Singleton reset failed: {e}",
                error_code="SINGLETON_RESET_ERROR",
                error_data={"exception": str(e)},
            )


__all__ = [
    "FlextContainer",
]
