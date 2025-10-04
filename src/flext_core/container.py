"""Dependency injection container anchoring the configuration pillar for 1.0.0.

The container is the canonical runtime surface described in ``README.md`` and
``docs/architecture.md``: a singleton, type-safe registry that coordinates with
``FlextConfig`` and ``FlextDispatcher`` so every package shares the same
service lifecycle during the modernization rollout.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import contextlib
import inspect
import threading
from collections.abc import Callable
from typing import cast, override

# External Dependencies - Dependency Injection
from dependency_injector import containers, providers

# Layer 3 - Core Infrastructure
from flext_core.config import FlextConfig

# Layer 1 - Foundation
from flext_core.constants import FlextConstants
from flext_core.models import FlextModels

# Layer 2 - Early Foundation
from flext_core.protocols import FlextProtocols
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes, T


class FlextContainer(FlextProtocols.Infrastructure.Configurable):
    """Global dependency injection container for FLEXT ecosystem.

    FlextContainer provides centralized service management using the
    singleton pattern. Access via FlextContainer.ensure_global_manager().get_or_create()
    for consistent dependency injection throughout applications and all
    32+ dependent FLEXT projects.

    **Function**: Service registry and dependency injection manager
        - Register services and factories globally with validation
        - Resolve dependencies with type safety and FlextResult
        - Support auto-wiring of constructor dependencies via inspect
        - Provide batch operations with rollback on failure
        - Enable thread-safe singleton access with double-checked lock
        - Integrate with FlextConfig for container configuration
        - Implement Configurable protocol for 1.0.0 compliance
        - Support service lifecycle management (singleton pattern)

    **Uses**: Core infrastructure components
        - FlextResult[T] for all operation results (railway pattern)
        - FlextConfig for container configuration and defaults
        - FlextModels.Validation for service name validation
        - FlextProtocols.Infrastructure.Configurable protocol
        - threading.Lock for singleton thread safety
        - inspect module for dependency resolution and auto-wiring
        - FlextConstants for timeout and configuration defaults
        - FlextTypes for type definitions and aliases

    **How to use**: Service registration and retrieval patterns
        ```python
        from flext_core import FlextContainer, FlextLogger, FlextResult

        # Example 1: Get global singleton instance
        manager = FlextContainer.ensure_global_manager()
        container = manager.get_or_create()

        # Example 2: Register services with validation
        logger = FlextLogger(__name__)
        result = container.register("logger", logger)
        if result.is_success:
            print("Logger registered successfully")

        # Example 3: Register factories (lazy instantiation)
        container.register_factory("database", lambda: DatabaseService())

        # Example 4: Retrieve services with type safety
        logger_result = container.get_typed("logger", FlextLogger)
        if logger_result.is_success:
            logger = logger_result.unwrap()
            logger.info("Container operational")

        # Example 5: Auto-wire dependencies (constructor injection)
        service_result = container.create_service(
            MyService,  # Auto-resolves constructor parameters
            service_name="my_service",
        )

        # Example 6: Batch registration with rollback
        services = {
            "cache": CacheService(),
            "queue": QueueService(),
            "metrics": MetricsService(),
        }
        batch_result = container.batch_register(services)
        ```

    Args:
        None: Constructor called internally via ensure_global_manager().get_or_create() singleton.

    Attributes:
        _services (FlextTypes.Dict): Registered service instances.
        _factories (FlextTypes.Dict): Service factory functions.
        _flext_config (FlextConfig): Global FlextConfig instance.
        _global_config (FlextTypes.Dict): Container configuration.
        _user_overrides (FlextTypes.Dict): User config overrides.

    Returns:
        FlextContainer: Singleton container instance via ensure_global_manager().get_or_create().

    Raises:
        ValueError: When service registration validation fails.
        KeyError: When retrieving non-existent service.
        TypeError: When type validation fails for typed retrieval.

    Note:
        Thread-safe singleton pattern with double-checked locking.
        All operations return FlextResult for railway pattern. Use
        ensure_global_manager().get_or_create() instead of direct instantiation.
        Container integrates with FlextConfig for ecosystem-wide settings.

    Warning:
        Never instantiate FlextContainer directly - always use
        ensure_global_manager().get_or_create(). Batch operations rollback on first failure.
        Service names must be unique across the container.

    Example:
        Complete service lifecycle management:

        >>> manager = FlextContainer.ensure_global_manager()
        >>> container = manager.get_or_create()
        >>> result = container.register("db", DatabaseService())
        >>> print(result.is_success)
        True
        >>> db_result = container.get("db")
        >>> print(db_result.is_success)
        True

    See Also:
        FlextConfig: For global configuration management.
        FlextResult: For railway-oriented error handling.
        FlextLogger: For logging with container integration.

    """

    # =========================================================================
    # SINGLETON MANAGEMENT - Class-level global state
    # =========================================================================

    _global_manager: FlextContainer.GlobalManager | None = None

    class GlobalManager:
        """Thread-safe global container management."""

        def __init__(self) -> None:
            """Initialize the global container manager."""
            super().__init__()
            self._container: FlextContainer | None = None
            self._lock = threading.Lock()

        def get_or_create(self) -> FlextContainer:
            """Get or create the global container instance.

            Returns:
                FlextContainer: The global container instance.

            """
            if self._container is None:
                with self._lock:
                    if self._container is None:
                        self._container = FlextContainer()
            return self._container

    def __init__(self) -> None:
        """Initialize container with optimized data structures and internal DI."""
        super().__init__()

        # Internal dependency-injector container (NEW v1.1.0)
        # Provides advanced DI features while maintaining backward compatibility
        self._di_container: containers.DynamicContainer = containers.DynamicContainer()

        # Core service storage with type safety (MAINTAINED for compatibility)
        self._services: FlextTypes.Dict = {}
        self._factories: FlextTypes.Dict = {}

        # Use FlextConfig directly for container configuration
        self._flext_config: FlextConfig = FlextConfig()
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
            "environment": str(
                getattr(
                    self._flext_config,
                    "environment",
                    FlextConstants.Config.DEFAULT_ENVIRONMENT,
                ),
            )
            .strip()
            .lower(),
        }

    def _sync_config_to_di(self) -> None:
        """Sync FlextConfig to internal dependency-injector container.

        Creates a Configuration provider in the DI container that mirrors
        FlextConfig values. This enables DI-based configuration injection
        while maintaining FlextConfig as the source of truth.

        Note:
            Added in v1.1.0 as part of internal DI wrapper implementation.
            This is an internal method - external API unchanged.

        """
        # Create configuration provider
        config_provider = providers.Configuration()

        # Sync all FlextConfig fields to DI config
        config_dict = {
            "environment": getattr(self._flext_config, "environment", "production"),
            "debug": getattr(self._flext_config, "debug", False),
            "trace": getattr(self._flext_config, "trace", False),
            "log_level": getattr(self._flext_config, "log_level", "INFO"),
            "max_workers": getattr(self._flext_config, "max_workers", 4),
            "timeout_seconds": getattr(
                self._flext_config,
                "timeout_seconds",
                FlextConstants.Container.TIMEOUT_SECONDS,
            ),
        }

        config_provider.from_dict(config_dict)
        self._di_container.config = config_provider

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

    def get_config(self) -> FlextTypes.Dict:
        """Get current configuration - Configurable protocol implementation.

        Returns:
            FlextTypes.Dict: Current container configuration

        """
        self._refresh_global_config()
        return {
            "max_workers": self._global_config["max_workers"],
            "timeout_seconds": self._global_config["timeout_seconds"],
            "environment": self._global_config["environment"],
            "service_count": self.get_service_count(),
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
            from flext_core.result import FlextResult
            from flext_core.container import FlextContainer, FlextLogger

            manager = FlextContainer.ensure_global_manager()
            container = manager.get_or_create()

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
        dependency-injector container (advanced DI features). Uses Singleton
        provider pattern to ensure single instance.

        Returns:
            FlextResult[None]: Success if stored or failure with error.

        Note:
            Updated in v1.1.0 to use internal DI container while maintaining API.

        """
        if name in self._services:
            return FlextResult[None].fail(f"Service '{name}' already registered")

        try:
            # Store in tracking dict (backward compatibility)
            self._services[name] = service

            # Store in internal DI container using Singleton provider
            # Capture service in lambda to avoid late binding issues
            provider = providers.Singleton(lambda s=service: s)
            self._di_container.set_provider(name, provider)

            return FlextResult[None].ok(None)
        except Exception as e:
            # Rollback on failure
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

        try:
            # Store in tracking dict (backward compatibility)
            self._factories[name] = factory

            # Store in internal DI container using Singleton with factory
            # This creates lazy singleton: factory called once, result cached
            provider = providers.Singleton(factory)
            self._di_container.set_provider(name, provider)

            return FlextResult[None].ok(None)
        except Exception as e:
            # Rollback on failure
            self._factories.pop(name, None)
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
        # Best-effort removal - failure is non-critical since tracking dicts are cleaned
        if hasattr(self._di_container, name):
            with contextlib.suppress(AttributeError, KeyError):
                # Expected: provider doesn't exist or already removed
                # Non-critical: tracking dicts are authoritative
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
        services (Singleton providers) and factories (Factory providers).
        Maintains FlextResult pattern for consistency with ecosystem.

        Returns:
            FlextResult[object]: Success with service instance or failure with error.

        Note:
            Updated in v1.1.0 to use DI container resolution while maintaining API.
            DynamicContainer stores providers like a dict, so we access by attribute.

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

                return FlextResult[object].ok(service)

            return FlextResult[object].fail(f"Service '{name}' not found")
        except Exception as e:
            # Preserve factory error messages for compatibility
            if name in self._factories:
                return FlextResult[object].fail(f"Factory '{name}' failed: {e}")
            return FlextResult[object].fail(f"Service resolution failed: {e}")

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

    def get_typed[T](self, name: str, expected_type: type[T]) -> FlextResult[T]:
        """Get service with type validation.

        Returns:
            FlextResult[T]: Success with typed service or failure with error.

        """
        return self.get(name).flat_map(
            lambda service: self._validate_service_type(service, expected_type),
        )

    def _validate_service_type[T](
        self,
        service: object,
        expected_type: type[T],
    ) -> FlextResult[T]:
        """Validate service type and return typed result.

        Returns:
            FlextResult[T]: Success with validated service or failure with error.

        """
        if not isinstance(service, expected_type):
            return FlextResult[T].fail(
                f"Service type mismatch: expected {getattr(expected_type, '__name__', str(expected_type))}, got {getattr(type(service), '__name__', str(type(service)))}",
            )
        return FlextResult[T].ok(service)

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
        factory: Callable[[], object] | None = None,
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
        factory: Callable[[], object],
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

        return self.get(name)

    def create_service(
        self,
        service_class: type[T],
        service_name: str | None = None,
    ) -> FlextResult[T]:
        """Create and register service with dependencies from container.

        Returns:
            FlextResult[T]: Success with instantiated service or failure with error.

        """
        try:
            # Determine service name
            if service_name is None:
                name = getattr(service_class, "__name__", "").lower()
                if not name:
                    return FlextResult[T].fail("Cannot determine service name")
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
                    return FlextResult[T].fail(
                        f"Cannot resolve required dependency '{param_name}'",
                    )

            # Create service instance
            service = service_class(**dependencies)

            # Register the service
            register_result = self.register(name, service)
            if register_result.is_failure:
                return FlextResult[T].fail(
                    register_result.error or "Service registration failed",
                )

            return FlextResult[T].ok(service)

        except Exception as e:
            return FlextResult[T].fail(f"Service creation failed: {e}")

    def auto_wire[T](
        self,
        service_class: type[T],
    ) -> FlextResult[T]:
        """Auto-wire service dependencies without registering.

        Creates a service instance by resolving its constructor dependencies
        from the container, but does not register the created service.

        Returns:
            FlextResult[T]: Success with instantiated service or failure with error.

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
                    return FlextResult[T].fail(
                        f"Cannot resolve required dependency '{param_name}' for auto-wiring",
                    )

            # Create service instance
            service = service_class(**dependencies)

            return FlextResult[T].ok(service)

        except Exception as e:
            return FlextResult[T].fail(f"Auto-wiring failed: {e}")

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

    def get_service_names(self) -> FlextResult[FlextTypes.StringList]:
        """REMOVED: Access container._services and container._factories directly.

        Migration:
            # Old pattern
            result = container.get_service_names()
            if result.is_success:
                names = result.unwrap()

            # New pattern - direct access
            all_names = set(container._services.keys()) | set(container._factories.keys())
            names = sorted(all_names)

        """
        msg = (
            "FlextContainer.get_service_names() has been removed. "
            "Access _services and _factories attributes directly."
        )
        raise NotImplementedError(msg)

    def get_service_count(self) -> int:
        """REMOVED: Calculate count from container._services and container._factories directly.

        Migration:
            # Old pattern
            count = container.get_service_count()

            # New pattern - direct calculation
            count = len(set(container._services.keys()) | set(container._factories.keys()))

        """
        msg = (
            "FlextContainer.get_service_count() has been removed. "
            "Calculate from _services and _factories attributes directly."
        )
        raise NotImplementedError(msg)

    def get_info(self) -> FlextResult[FlextTypes.Dict]:
        """REMOVED: Build info dict from container attributes directly.

        Migration:
            # Old pattern
            result = container.get_info()
            if result.is_success:
                info = result.unwrap()

            # New pattern - direct access
            info = {
                "service_count": len(set(container._services.keys()) | set(container._factories.keys())),
                "direct_services": len(container._services),
                "factories": len(container._factories),
                "configuration": container._flext_config.model_dump(),
            }

        """
        msg = (
            "FlextContainer.get_info() has been removed. "
            "Build info dictionary from container attributes directly."
        )
        raise NotImplementedError(msg)

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
            allowed_keys = {"max_workers", "timeout_seconds", "environment"}
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
    def ensure_global_manager(cls) -> FlextContainer.GlobalManager:
        """Ensure global manager exists with thread safety.

        Returns:
            FlextContainer.GlobalManager: The global manager instance.

        """
        if cls._global_manager is None:
            cls._global_manager = cls.GlobalManager()
        return cls._global_manager

    @classmethod
    def get_global(cls) -> FlextContainer:
        """Get the global container instance (compatibility method).

        Returns:
            FlextContainer: The global container instance.

        Note:
            This method provides backward compatibility.
            Use ensure_global_manager().get_or_create() for new code.

        """
        manager = cls.ensure_global_manager()
        return manager.get_or_create()

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

        manager = cls.ensure_global_manager()
        container = manager.get_or_create()
        return FlextResult[FlextTypes.Dict].ok({
            "container": container,
            "module": module_name,
            "logger": f"flext.{module_name}",
        })

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


__all__: FlextTypes.StringList = [
    "FlextContainer",
]
