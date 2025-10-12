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
from collections.abc import Callable
from contextlib import suppress
from typing import Self, cast, override

from dependency_injector.containers import DynamicContainer
from dependency_injector.providers import Object, Singleton

from flext_core.config import FlextConfig
from flext_core.constants import FlextConstants
from flext_core.models import FlextModels
from flext_core.protocols import FlextProtocols
from flext_core.result import FlextResult
from flext_core.runtime import FlextRuntime
from flext_core.typings import FlextTypes, T


class FlextContainer(FlextProtocols.Infrastructure.Configurable):
    """Type-safe dependency injection container for service management.

    Provides centralized service management with singleton pattern support,
    type-safe registration and resolution, and automatic dependency injection
    via constructor inspection.

    Features:
    - Type-safe service registration and resolution
    - Singleton and factory pattern support
    - Automatic dependency injection via constructor inspection
    - Batch operations with rollback on failure
    - Thread-safe singleton access
    - Integration with FlextConfig for container configuration
    - FlextResult-based error handling for all operations

    Usage:
        >>> from flext_core import FlextContainer
        >>>
        >>> container = FlextContainer()
        >>> container.register("logger", FlextLogger(__name__))
        >>> logger_result = container.get("logger")
    """

    # =========================================================================
    # SINGLETON MANAGEMENT - Class-level global state
    # =========================================================================

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

        # Core service storage with type safety (MAINTAINED for compatibility)
        self._services: FlextTypes.Dict = {}
        self._factories: FlextTypes.Dict = {}

        # Use FlextConfig global singleton for container configuration
        self._flext_config: FlextConfig = FlextConfig.get_global_instance()
        self._global_config: FlextTypes.Dict = self._create_container_config()
        self._user_overrides: FlextTypes.Dict = {}

        # Sync FlextConfig to internal DI container
        self._sync_config_to_di()

    def _create_container_config(self) -> FlextTypes.Dict:
        """Create container configuration from FlextConfig defaults."""
        return {
            "max_workers": int(getattr(self._flext_config, "max_workers", 4)),
            "timeout_seconds": float(
                getattr(
                    self._flext_config,
                    "timeout_seconds",
                    FlextConstants.Container.TIMEOUT_SECONDS,
                ),
            ),
        }

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
    def configure(self, config: FlextTypes.Dict) -> FlextResult[None]:
        """Configure component with provided settings - Configurable protocol implementation.

        Returns:
            FlextResult[None]: Configuration result

        """
        return self.configure_container(config)

    @property
    def services(self) -> FlextTypes.Dict:
        """Get registered services dictionary (read-only access for compatibility).

        Returns:
            FlextTypes.Dict: Dictionary of registered service instances.

        Note:
            This property provides read-only access to the internal services registry
            for backward compatibility with existing code that expects direct access.

        """
        return self._services.copy()

    @property
    def factories(self) -> FlextTypes.Dict:
        """Get registered factories dictionary (read-only access for compatibility).

        Returns:
            FlextTypes.Dict: Dictionary of registered factory functions.

        Note:
            This property provides read-only access to the internal factories registry
            for backward compatibility with existing code that expects direct access.

        """
        return self._factories.copy()

    def get_config(self) -> FlextTypes.Dict:
        """Get current configuration - Configurable protocol implementation.

        Returns:
            FlextTypes.Dict: Current container configuration

        """
        self._refresh_global_config()
        return {
            "max_workers": self._global_config["max_workers"],
            "timeout_seconds": self._global_config["timeout_seconds"],
            "service_count": len(
                set(self._services.keys()) | set(self._factories.keys())
            ),
            "services": list(self._services.keys()),
            "factories": list(self._factories.keys()),
        }

    # =========================================================================
    # CORE SERVICE MANAGEMENT - Primary operations with railway patterns
    # =========================================================================

    def _validate_service_name(self, name: str) -> FlextResult[str]:
        """Validate and normalize service name using centralized validation.

        Returns:
            FlextResult[str]: Success with validated name or failure with error.

        """
        return FlextModels.Validation.validate_service_name(name)

    def register(
        self,
        name: str,
        service: object,
    ) -> FlextResult[None]:
        """Register a service in the FlextContainer.

        Use this method to register services for dependency injection throughout
        FLEXT applications. Returns FlextResult for explicit error handling.

        Args:
            name: Service name for later retrieval
            service: Service instance to register

        Returns:
            FlextResult[None]: Success if registered or failure with error details.

        Example:
            ```python
            from flext_core.api import FlextCore
            from flext_core.container import FlextContainer, FlextLogger

            container = FlextContainer()
            logger = FlextLogger(__name__)

            result: FlextResult[object] = container.register("logger", logger)
            if result.is_failure:
                print(f"Registration failed: {result.error}")
            ```

        """
        return self._validate_service_name(name).flat_map(
            lambda validated_name: self._store_service(validated_name, service),
        )

    def _store_service(self, name: str, service: object) -> FlextResult[None]:
        """Store service in registry AND internal DI container with conflict detection.

        Stores in both tracking dict (backward compatibility) and internal
        dependency-injector container (advanced DI features). Uses Object
        provider for existing service instances.

        Returns:
            FlextResult[None]: Success if stored or failure with error.

        Note:
            Updated in v1.1.0 to use internal DI container while maintaining API.

        """
        if name in self._services:
            return FlextResult[None].fail(f"Service '{name}' already registered")

        provider_registered = False
        try:
            # Store in tracking dict (backward compatibility)
            self._services[name] = service

            # Store in internal DI container using Object provider
            # Object provider is for existing instances (singletons by nature)
            provider = Object(service)
            cast("DynamicContainer", self._di_container).set_provider(name, provider)
            provider_registered = True

            return FlextResult[None].ok(None)
        except Exception as e:
            # Rollback on failure - only delete if we successfully registered it
            # We successfully called set_provider, so the attribute should exist
            # If it doesn't, something unexpected happened but rollback continues
            if provider_registered and hasattr(self._di_container, name):
                delattr(self._di_container, name)
            self._services.pop(name, None)
            return FlextResult[None].fail(f"Service storage failed: {e}")

    def register_factory(
        self,
        name: str,
        factory: Callable[[], object],
    ) -> FlextResult[None]:
        """Register service factory with validation.

        Returns:
            FlextResult[None]: Success if registered or failure with error.

        """
        return self._validate_service_name(name).flat_map(
            lambda validated_name: self._store_factory(validated_name, factory),
        )

    def _store_factory(
        self,
        name: str,
        factory: Callable[[], object],
    ) -> FlextResult[None]:
        """Store factory in registry AND internal DI container with callable validation.

        Stores in both tracking dict (backward compatibility) and internal
        dependency-injector container using Singleton provider with factory.
        This ensures factory is called once and result is cached (lazy singleton).

        Returns:
            FlextResult[None]: Success if stored or failure with error.

        Note:
            Updated in v1.1.0 to use DI Singleton(factory) for cached factory results.

        """
        # Validate that factory is actually callable
        if not callable(factory):
            return FlextResult[None].fail(f"Factory '{name}' must be callable")

        if name in self._factories:
            return FlextResult[None].fail(f"Factory '{name}' already registered")

        provider_registered = False
        try:
            # Store in tracking dict (backward compatibility)
            self._factories[name] = factory

            # Store in internal DI container using Singleton with factory
            # This creates lazy singleton: factory called once, result cached
            provider = Singleton(factory)
            cast("DynamicContainer", self._di_container).set_provider(name, provider)
            provider_registered = True

            return FlextResult[None].ok(None)
        except Exception as e:
            # Rollback on failure - only delete if we successfully registered it
            self._factories.pop(name, None)
            # We successfully called set_provider, so the attribute should exist
            # If it doesn't, something unexpected happened but rollback continues
            if provider_registered and hasattr(self._di_container, name):
                delattr(self._di_container, name)
            return FlextResult[None].fail(f"Factory storage failed: {e}")

    def unregister(self, name: str) -> FlextResult[None]:
        """Unregister service or factory with validation.

        Returns:
            FlextResult[None]: Success if unregistered or failure with error.

        """
        return self._validate_service_name(name).flat_map(self._remove_service)

    def _remove_service(self, name: str) -> FlextResult[None]:
        """Remove service from tracking dicts AND internal DI container.

        Returns:
            FlextResult[None]: Success if removed or failure with error.

        Note:
            Updated in v1.1.0 to remove from DI container as well.

        """
        service_found = name in self._services
        factory_found = name in self._factories

        if not service_found and not factory_found:
            return FlextResult[None].fail(f"Service '{name}' not registered")

        # Remove from tracking dicts (backward compatibility)
        self._services.pop(name, None)
        self._factories.pop(name, None)

        # Remove from internal DI container (access as attribute)
        # Check existence before deletion to avoid AttributeError
        if hasattr(self._di_container, name):
            delattr(self._di_container, name)

        return FlextResult[None].ok(None)

    def get(self, name: str) -> FlextResult[object]:
        """Get service with factory resolution and caching.

        Returns:
            FlextResult[object]: Success with service instance or failure with error.

        """
        return self._validate_service_name(name).flat_map(self._resolve_service)

    def _resolve_service(self, name: str) -> FlextResult[object]:
        """Resolve service via internal DI container with FlextResult wrapping.

        Resolves from dependency-injector container which handles both direct
        services (Singleton self.providers) and factories (Factory self.providers).
        Maintains FlextResult pattern for consistency with ecosystem.

        Returns:
            FlextResult[object]: Success with service instance or failure with error.

        Note:
            Updated in v1.1.0 to use DI container resolution while maintaining API.
            DynamicContainer stores self.providers like a dict, so we access by attribute.

        """
        try:
            # Resolve via internal DI container (access as attribute, not dict)
            # DynamicContainer allows attribute-style access: container.service_name()
            if hasattr(self._di_container, name):
                provider = getattr(self._di_container, name)
                service = provider()

                # Cache factory results in tracking dict for compatibility
                if name in self._factories and name not in self._services:
                    self._services[name] = service

                # Integration: Track successful service resolution
                FlextRuntime.Integration.track_service_resolution(name, resolved=True)

                return FlextResult[object].ok(service)

            # Integration: Track service not found
            FlextRuntime.Integration.track_service_resolution(
                name, resolved=False, error_message=f"Service '{name}' not found"
            )

            return FlextResult[object].fail(f"Service '{name}' not found")
        except Exception as e:
            # Preserve factory error messages for compatibility
            error_msg = (
                f"Factory '{name}' failed: {e}"
                if name in self._factories
                else f"Service resolution failed: {e}"
            )

            # Integration: Track resolution failure
            FlextRuntime.Integration.track_service_resolution(
                name, resolved=False, error_message=error_msg
            )

            return FlextResult[object].fail(error_msg)

    def _invoke_factory_and_cache(self, name: str) -> FlextResult[object]:
        """Invoke factory and cache result.

        Returns:
            FlextResult[object]: Success with service instance or failure with error.

        """
        try:
            factory_obj = self._factories[name]
            # Type-safe factory invocation
            factory = cast("Callable[[], object]", factory_obj)
            service = factory()

            # Cache the created service
            self._services[name] = service

            return FlextResult[object].ok(service)
        except Exception as e:
            return FlextResult[object].fail(f"Factory '{name}' failed: {e}")

    def get_typed[T](self, name: str, expected_type: type[T]) -> FlextResult[object]:
        """Get service with type validation.

        Returns:
            FlextResult[object]: Success with typed service or failure with error.

        """
        return self.get(name).flat_map(
            lambda service: self._validate_service_type(service, expected_type),
        )

    def _validate_service_type[T](
        self,
        service: object,
        expected_type: type[T],
    ) -> FlextResult[object]:
        """Validate service type and return typed result.

        Returns:
            FlextResult[object]: Success with validated service or failure with error.

        """
        if not isinstance(service, expected_type):
            return FlextResult[object].fail(
                f"Service type mismatch: expected {getattr(expected_type, '__name__', str(expected_type))}, got {getattr(type(service), '__name__', str(type(service)))}",
            )
        return FlextResult[object].ok(service)

    # =========================================================================
    # BATCH OPERATIONS - Efficient bulk service management
    # =========================================================================

    def batch_register(self, services: FlextTypes.Dict) -> FlextResult[None]:
        """Register multiple services atomically with rollback on failure.

        Returns:
            FlextResult[None]: Success if all registered or failure with error.

        """
        # Create snapshot for potential rollback
        snapshot = self._create_registry_snapshot()

        try:
            # Process all registrations with error handling
            result = self._process_batch_registrations(services)
            if result.is_failure:
                # Restore snapshot on failure
                self._restore_registry_snapshot(snapshot)
                return FlextResult[None].fail(
                    result.error or "Batch registration failed",
                )

            return FlextResult[None].ok(None)
        except Exception as e:
            # Restore snapshot on exception
            self._restore_registry_snapshot(snapshot)
            return FlextResult[None].fail(f"Batch registration error: {e}")

    def _create_registry_snapshot(self) -> FlextTypes.Dict:
        """Create snapshot of current registry state for rollback.

        Returns:
            FlextTypes.Dict: Snapshot containing services and factories.

        """
        return {
            "services": self._services.copy(),
            "factories": self._factories.copy(),
        }

    def _process_batch_registrations(
        self,
        services: FlextTypes.Dict,
    ) -> FlextResult[None]:
        """Process batch registrations with proper error handling.

        Returns:
            FlextResult[None]: Success if all processed or failure with error.

        """
        for name, service in services.items():
            # Validate service name
            validation_result = self._validate_service_name(name)
            if validation_result.is_failure:
                return FlextResult[None].fail(
                    validation_result.error or f"Invalid service name: {name}",
                )

            # Store the service
            storage_result = self._store_service(name, service)
            if storage_result.is_failure:
                return FlextResult[None].fail(
                    storage_result.error or f"Failed to store service: {name}",
                )

        return FlextResult[None].ok(None)

    def _restore_registry_snapshot(self, snapshot: FlextTypes.Dict) -> None:
        """Restore registry state from snapshot with type safety."""
        # Direct assignment - snapshot already has correct types
        services_snapshot = snapshot.get("services")
        factories_snapshot = snapshot.get("factories")

        if isinstance(services_snapshot, dict):
            self._services = services_snapshot
        if isinstance(factories_snapshot, dict):
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
        register_result = self.register_factory(name, factory)
        if register_result.is_failure:
            return FlextResult[object].fail(
                register_result.error or "Factory registration failed",
            )

        service_result = self.get(name)
        if service_result.is_success:
            # Type assertion since we know the service type matches
            service = service_result.unwrap()
            return FlextResult[object].ok(cast("T", service))
        return FlextResult[object].fail(
            service_result.error or "Service retrieval failed"
        )

    def create_service[T](
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
            dependencies: FlextTypes.Dict = {}
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
            register_result = self.register(name, service)
            if register_result.is_failure:
                return FlextResult[object].fail(
                    register_result.error or "Service registration failed",
                )

            return FlextResult[object].ok(service)

        except Exception as e:
            return FlextResult[object].fail(f"Service creation failed: {e}")

    def auto_wire[T](
        self,
        service_class: type[T],
    ) -> FlextResult[object]:
        """Auto-wire service dependencies without registering.

        Creates a service instance by resolving its constructor dependencies
        from the container, but does not register the created service.

        Returns:
            FlextResult[object]: Success with instantiated service or failure with error.

        """
        try:
            # Try to resolve dependencies from container
            dependencies: FlextTypes.Dict = {}
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
                        f"Cannot resolve required dependency '{param_name}' for auto-wiring",
                    )

            # Create service instance
            service = service_class(**dependencies)

            return FlextResult[object].ok(service)

        except Exception as e:
            return FlextResult[object].fail(f"Auto-wiring failed: {e}")

    # =========================================================================
    # INSPECTION AND UTILITIES - Container introspection and management
    # =========================================================================

    def clear(self) -> FlextResult[None]:
        """Clear all services and factories.

        Returns:
            FlextResult[None]: Success if cleared or failure with error.

        """
        try:
            self._services.clear()
            self._factories.clear()
            # Reset DI container to clear all providers
            self._di_container = self.containers.DynamicContainer()
            return FlextResult[None].ok(None)
        except Exception as e:
            return FlextResult[None].fail(f"Failed to clear container: {e}")

    def has(self, name: str) -> bool:
        """Check if service is registered.

        Returns:
            bool: True if service is registered, False otherwise.

        """
        normalized = FlextModels.Validation.validate_service_name(name)
        if normalized.is_failure:
            return False
        validated_name = normalized.value_or_none
        if validated_name is None:
            return False
        return validated_name in self._services or validated_name in self._factories

    def list_services(self) -> FlextResult[FlextTypes.List]:
        """List all registered services with metadata.

        Returns:
            FlextResult[FlextTypes.List]: Success with service list or failure with error.

        """
        # ISSUE: Duplicates get_service_names functionality - both methods iterate over same service collections
        try:
            services: FlextTypes.List = []
            for name in sorted(
                set(self._services.keys()) | set(self._factories.keys()),
            ):
                service_info: FlextTypes.Dict = {
                    FlextConstants.Mixins.FIELD_NAME: name,
                    FlextConstants.Mixins.FIELD_TYPE: "instance"
                    if name in self._services
                    else "factory",
                    FlextConstants.Mixins.FIELD_REGISTERED: True,
                }
                # Ensure type compatibility by explicitly casting to object before append
                services.append(cast("object", service_info))

            return FlextResult[FlextTypes.List].ok(services)
        except Exception as e:
            return FlextResult[FlextTypes.List].fail(
                f"Failed to list services: {e}",
            )

    def _build_service_info(
        self,
        name: str,
        service: object,
        service_type: str,
    ) -> FlextTypes.Dict:
        """Build service information dictionary.

        Returns:
            FlextTypes.Dict: Service information dictionary.

        """
        try:
            return {
                FlextConstants.Mixins.FIELD_NAME: name,
                FlextConstants.Mixins.FIELD_TYPE: service_type,
                FlextConstants.Mixins.FIELD_CLASS: service.__class__.__name__,
                FlextConstants.Mixins.FIELD_MODULE: getattr(
                    service.__class__,
                    "__module__",
                    FlextConstants.Mixins.IDENTIFIER_UNKNOWN,
                ),
                "is_callable": callable(service),
                "id": id(service),
            }
        except Exception:
            # Fallback for problematic services
            return {
                FlextConstants.Mixins.FIELD_NAME: name,
                FlextConstants.Mixins.FIELD_TYPE: service_type,
                FlextConstants.Mixins.FIELD_CLASS: FlextConstants.Mixins.IDENTIFIER_UNKNOWN,
                FlextConstants.Mixins.FIELD_MODULE: FlextConstants.Mixins.IDENTIFIER_UNKNOWN,
                "is_callable": False,
                "id": id(service),
                "error": "Failed to inspect service",
            }

    # =========================================================================
    # CONFIGURATION MANAGEMENT - FlextConfig integration
    # =========================================================================

    def configure_container(self, config: FlextTypes.Dict) -> FlextResult[None]:
        """Configure container with validated settings.

        Returns:
            FlextResult[None]: Success if configured or failure with error.

        """
        try:
            # Only allow specific configuration keys
            allowed_keys = {"max_workers", "timeout_seconds"}
            processed_config = {
                key: value for key, value in config.items() if key in allowed_keys
            }

            # Update user overrides
            self._user_overrides.update(processed_config)

            # Update global config
            self._refresh_global_config()

            return FlextResult[None].ok(None)
        except Exception as e:
            return FlextResult[None].fail(f"Container configuration failed: {e}")

    def _refresh_global_config(self) -> None:
        """Refresh the effective global configuration."""
        # Merge FlextConfig defaults with user overrides
        merged: FlextTypes.Dict = {}
        for source in (self._create_container_config(), self._user_overrides):
            merged.update({
                key: value for key, value in source.items() if value is not None
            })
        self._global_config = merged

    # =========================================================================
    # GLOBAL CONTAINER MANAGEMENT - Singleton pattern with thread safety
    # =========================================================================

    @classmethod
    def _ensure_global_instance(cls) -> FlextContainer:
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
        return cls._ensure_global_instance()

    @classmethod
    def create_module_utilities(
        cls,
        module_name: str,
    ) -> FlextResult[FlextTypes.Dict]:
        """Create module-specific utilities with container integration.

        Returns:
            FlextResult[FlextTypes.Dict]: Success with utilities dict or failure with error.

        """
        if not module_name:
            return FlextResult[FlextTypes.Dict].fail(
                "Module name must be non-empty string",
            )

        container = cls._ensure_global_instance()
        return FlextResult[FlextTypes.Dict].ok({
            "container": container,
            "module": module_name,
            "logger": f"flext.{module_name}",
        })

    def get_with_fallback(
        self,
        primary_name: str,
        *fallback_names: str,
    ) -> FlextResult[object]:
        """Get service trying multiple names using alt pattern.

        Demonstrates alt for service resolution fallback chains.

        Args:
            primary_name: Primary service name to try
            *fallback_names: Fallback service names to try in order

        Returns:
            FlextResult with service from first found name

        Example:
            >>> container = FlextContainer.get_global()
            >>> service = container.get_with_fallback("db_service", "database", "db")

        """
        result = self.get(primary_name)
        for fallback_name in fallback_names:
            result = result.alt(self.get(fallback_name))
        return result

    def safe_register_from_factory(
        self,
        name: str,
        factory: object,
    ) -> FlextResult[None]:
        """Register service from factory using from_callable pattern.

        Demonstrates from_callable for safe factory registration.

        Args:
            name: Service name
            factory: Factory callable

        Returns:
            FlextResult[None]: Success or error

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
            _execute_factory
        )

        if factory_result.is_failure:
            return FlextResult[None].fail(
                f"Factory execution failed: {factory_result.error}"
            )

        service: object = factory_result.unwrap()
        return self.register(name, service)

    def get_typed_with_recovery[T](
        self,
        name: str,
        expected_type: type[T],
        recovery_factory: object | None = None,
    ) -> FlextResult[object]:
        """Get typed service with recovery using lash pattern.

        Demonstrates lash for error recovery with factory.

        Args:
            name: Service name
            expected_type: Expected service type
            recovery_factory: Optional factory to create service if not found

        Returns:
            FlextResult[object]: Service or recovered value

        Example:
            >>> container = FlextContainer.get_global()
            >>> logger = container.get_typed_with_recovery(
            ...     "logger", LoggerService, lambda: LoggerService()
            ... )

        """

        def try_recovery(_error: str) -> FlextResult[object]:
            if recovery_factory is None:
                return FlextResult[object].fail(
                    f"Service '{name}' not found and no recovery factory provided"
                )

            def _execute_recovery_factory() -> object:
                return (
                    recovery_factory()
                    if callable(recovery_factory)
                    else recovery_factory
                )

            factory_result: FlextResult[object] = FlextResult[object].from_callable(
                _execute_recovery_factory
            )

            if factory_result.is_failure:
                return FlextResult[object].fail(
                    f"Recovery factory failed: {factory_result.error}"
                )

            service: object = factory_result.unwrap()
            if not isinstance(service, expected_type):
                return FlextResult[object].fail(
                    f"Recovery factory returned wrong type: expected {getattr(expected_type, '__name__', str(expected_type))}, got {getattr(type(service), '__name__', str(type(service)))}"
                )

            # Register the recovered service for future use
            self.register(name, service)
            return FlextResult[object].ok(service)

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

            validator_result: FlextResult[object] | None = None
            for validator in validators:
                if callable(validator):
                    # Call validator - returns FlextResult[object]
                    validator_result = validator(service)

                    # Check if validation failed
                    if validator_result.is_failure:
                        return FlextResult[object].fail(
                            f"Validation failed: {validator_result.error}"
                        )
                    # Continue with validated service

            if validator_result is None:
                return FlextResult[object].fail(
                    f"Validator must return FlextResult, got {None}"
                )

            result: FlextResult[object] = validator_result

            return result

        return self.get(name).flat_map(apply_validators)

    @override
    def __repr__(self) -> str:
        """String representation with service counts.

        Returns:
            str: String representation of the container.

        """
        total_registered = len(set(self._services.keys()) | set(self._factories.keys()))
        return (
            f"FlextCore.Container(services={len(self._services)}, "
            f"factories={len(self._factories)}, "
            f"total_registered={total_registered})"
        )


__all__ = [
    "FlextContainer",
]
