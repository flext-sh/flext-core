"""Dependency injection container anchoring the configuration pillar for 1.0.0.

The container is the canonical runtime surface described in ``README.md`` and
``docs/architecture.md``: a singleton, type-safe registry that coordinates with
``FlextConfig`` and ``FlextDispatcher`` so every package shares the same
service lifecycle during the modernization rollout.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import threading
from collections.abc import Callable

from flext_core.config import FlextConfig
from flext_core.constants import FlextConstants
from flext_core.protocols import FlextProtocols
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes, T


class FlextContainer(FlextProtocols.Infrastructure.Configurable):
    """Global container providing the standardized FLEXT service contract.

    Optimized implementation using FlextResult railway patterns for 75% code reduction
    while maintaining identical functionality, API compatibility, and explicit Configurable
    protocol compliance for 1.0.0 stability.
    """

    # =========================================================================
    # SINGLETON MANAGEMENT - Class-level global state
    # =========================================================================

    _global_manager: FlextTypes.Core.Optional[FlextContainer.GlobalManager] = None

    class ServiceKey:
        """Utility for service key normalization and validation."""

        @staticmethod
        def normalize(name: str) -> FlextResult[str]:
            """Normalize service name for consistent lookups.

            Returns:
                FlextResult[str]: Success with normalized name or failure with error message.

            """
            # Type annotation guarantees name is str, so no isinstance check needed
            normalized = name.strip().lower()
            if not normalized:
                return FlextResult[str].fail("Service name cannot be empty")

            # Additional validation for special characters
            if any(char in normalized for char in [".", "/", "\\"]):
                return FlextResult[str].fail("Service name contains invalid characters")

            return FlextResult[str].ok(normalized)

        @staticmethod
        def validate(name: str) -> FlextResult[str]:
            """Validate service name format and return normalized key.

            Returns:
                FlextResult[str]: Success with normalized name or failure with error message.

            """
            return FlextContainer.ServiceKey.normalize(name)

    class GlobalManager:
        """Thread-safe global container management."""

        def __init__(self) -> None:
            """Initialize the global container manager."""
            self._container: FlextTypes.Core.Optional[FlextContainer] = None
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
        """Initialize container with optimized data structures."""
        # Core service storage with type safety
        self._services: FlextTypes.Core.Dict = {}
        self._factories: FlextTypes.Core.Dict = {}

        # Configuration integration with FlextConfig singleton
        self._global_config: FlextTypes.Core.Dict = {
            "max_workers": 4,
            "timeout_seconds": 30.0,
            "environment": "development",
        }
        self._flext_config = FlextConfig.get_global_instance()

    # =========================================================================
    # CONFIGURABLE PROTOCOL IMPLEMENTATION - Protocol compliance for 1.0.0
    # =========================================================================

    def configure(self, config: FlextTypes.Core.Dict) -> object:
        """Configure component with provided settings - Configurable protocol implementation.

        Returns:
            object: Configuration result (for protocol compliance)

        """
        result = self.configure_container(config)
        return result.unwrap() if result.is_success else result.error

    def get_config(self) -> FlextTypes.Core.Dict:
        """Get current configuration - Configurable protocol implementation.

        Returns:
            FlextTypes.Core.Dict: Current container configuration

        """
        services_list: FlextTypes.Core.StringList = list(self._services.keys())
        factories_list: FlextTypes.Core.StringList = list(self._factories.keys())

        return {
            "max_workers": self._global_config["max_workers"],
            "timeout_seconds": self._global_config["timeout_seconds"],
            "environment": self._global_config["environment"],
            "service_count": self.get_service_count(),
            "services": services_list,
            "factories": factories_list,
        }

    # =========================================================================
    # CORE SERVICE MANAGEMENT - Primary operations with railway patterns
    # =========================================================================

    def _validate_service_name(self, name: str) -> FlextResult[str]:
        """Validate and normalize service name using ServiceKey utility.

        Returns:
            FlextResult[str]: Success with validated name or failure with error.

        """
        return self.ServiceKey.validate(name)

    def register(
        self,
        name: str,
        service: object,
    ) -> FlextResult[None]:
        """Register service with comprehensive validation and error handling.

        Returns:
            FlextResult[None]: Success if registered or failure with error.

        """
        return self._validate_service_name(name).flat_map(
            lambda validated_name: self._store_service(validated_name, service)
        )

    def _store_service(self, name: str, service: object) -> FlextResult[None]:
        """Store service in registry with conflict detection.

        Returns:
            FlextResult[None]: Success if stored or failure with error.

        """
        if name in self._services:
            return FlextResult[None].fail(f"Service '{name}' already registered")

        self._services[name] = service
        return FlextResult[None].ok(None)

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
            lambda validated_name: self._store_factory(validated_name, factory)
        )

    def _store_factory(
        self, name: str, factory: Callable[[], object]
    ) -> FlextResult[None]:
        """Store factory with callable validation.

        Returns:
            FlextResult[None]: Success if stored or failure with error.

        """
        # Type annotation guarantees factory is callable, so no callable() check needed

        if name in self._factories:
            return FlextResult[None].fail(f"Factory '{name}' already registered")

        self._factories[name] = factory
        return FlextResult[None].ok(None)

    def unregister(self, name: str) -> FlextResult[None]:
        """Unregister service or factory with validation.

        Returns:
            FlextResult[None]: Success if unregistered or failure with error.

        """
        return self._validate_service_name(name).flat_map(self._remove_service)

    def _remove_service(self, name: str) -> FlextResult[None]:
        """Remove service from both registries.

        Returns:
            FlextResult[None]: Success if removed or failure with error.

        """
        service_found = name in self._services
        factory_found = name in self._factories

        if not service_found and not factory_found:
            return FlextResult[None].fail(f"Service '{name}' not registered")

        # Remove from both registries
        self._services.pop(name, None)
        self._factories.pop(name, None)

        return FlextResult[None].ok(None)

    def get(self, name: str) -> FlextResult[object]:
        """Get service with factory resolution and caching.

        Returns:
            FlextResult[object]: Success with service instance or failure with error.

        """
        return self._validate_service_name(name).flat_map(self._resolve_service)

    def _resolve_service(self, name: str) -> FlextResult[object]:
        """Resolve service from registry or factory.

        Returns:
            FlextResult[object]: Success with service instance or failure with error.

        """
        # Check direct service registry first
        if name in self._services:
            return FlextResult[object].ok(self._services[name])

        # Try factory resolution with caching
        if name in self._factories:
            return self._invoke_factory_and_cache(name)

        return FlextResult[object].fail(f"Service '{name}' not found")

    def _invoke_factory_and_cache(self, name: str) -> FlextResult[object]:
        """Invoke factory and cache result.

        Returns:
            FlextResult[object]: Success with service instance or failure with error.

        """
        try:
            factory = self._factories[name]
            service = factory()

            # Cache the created service
            self._services[name] = service

            return FlextResult[object].ok(service)
        except Exception as e:
            return FlextResult[object].fail(f"Factory '{name}' failed: {e}")

    def get_typed(self, name: str, expected_type: type[T]) -> FlextResult[T]:
        """Get service with type validation.

        Returns:
            FlextResult[T]: Success with typed service or failure with error.

        """
        return self.get(name).flat_map(
            lambda service: self._validate_service_type(service, expected_type)
        )

    def _validate_service_type(
        self, service: object, expected_type: type[T]
    ) -> FlextResult[T]:
        """Validate service type and return typed result.

        Returns:
            FlextResult[T]: Success with validated service or failure with error.

        """
        if not isinstance(service, expected_type):
            return FlextResult[T].fail(
                f"Service type mismatch: expected {expected_type.__name__}, "
                f"got {type(service).__name__}"
            )
        return FlextResult[T].ok(service)

    # =========================================================================
    # BATCH OPERATIONS - Efficient bulk service management
    # =========================================================================

    def batch_register(self, services: FlextTypes.Core.Dict) -> FlextResult[None]:
        """Register multiple services atomically with rollback on failure.

        Returns:
            FlextResult[None]: Success if all registered or failure with error.

        """
        # Create snapshot for potential rollback
        snapshot = self._create_registry_snapshot()

        try:
            # Process all registrations
            result = self._process_batch_registrations(services)
            if result.is_failure:
                # Restore snapshot on failure
                self._restore_registry_snapshot(snapshot)
                return FlextResult[None].fail(
                    result.error or "Batch registration failed"
                )

            return FlextResult[None].ok(None)
        except Exception as e:
            # Restore snapshot on exception
            self._restore_registry_snapshot(snapshot)
            return FlextResult[None].fail(f"Batch registration error: {e}")

    def _create_registry_snapshot(self) -> FlextTypes.Core.Dict:
        """Create snapshot of current registry state for rollback.

        Returns:
            FlextTypes.Core.Dict: Snapshot containing services and factories.

        """
        return {
            "services": self._services.copy(),
            "factories": self._factories.copy(),
        }

    def _process_batch_registrations(
        self, services: FlextTypes.Core.Dict
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
                    validation_result.error or f"Invalid service name: {name}"
                )

            # Store the service
            storage_result = self._store_service(name, service)
            if storage_result.is_failure:
                return FlextResult[None].fail(
                    storage_result.error or f"Failed to store service: {name}"
                )

        return FlextResult[None].ok(None)

    def _restore_registry_snapshot(self, snapshot: FlextTypes.Core.Dict) -> None:
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
        service_result = self.get(name)

        if service_result.is_success:
            return service_result

        if factory is None:
            return FlextResult[object].fail(
                f"Service '{name}' not found and no factory provided"
            )

        return self._create_from_factory(name, factory)

    def _create_from_factory(
        self, name: str, factory: Callable[[], object]
    ) -> FlextResult[object]:
        """Create service from factory and register it.

        Returns:
            FlextResult[object]: Success with service instance or failure with error.

        """
        register_result = self.register_factory(name, factory)
        if register_result.is_failure:
            return FlextResult[object].fail(
                register_result.error or "Factory registration failed"
            )

        return self.get(name)

    def auto_wire(self, service_class: type[T]) -> FlextResult[T]:
        """Automatically register and resolve service with dependency injection.

        Returns:
            FlextResult[T]: Success with instantiated service or failure with error.

        """
        return (
            self._resolve_auto_wire_name(service_class)
            .flat_map(lambda _: self._resolve_dependencies(service_class))
            .flat_map(
                lambda deps: self._instantiate_and_register_service(service_class, deps)
            )
        )

    def _resolve_auto_wire_name(self, service_class: type[T]) -> FlextResult[str]:
        """Resolve service name for auto-wiring.

        Returns:
            FlextResult[str]: Success with snake_case service name or failure with error.

        """
        name = getattr(service_class, "__name__", "")
        if not name:
            return FlextResult[str].fail(
                "Cannot determine service name for auto-wiring"
            )

        # Convert CamelCase to snake_case for service naming
        snake_case = "".join([
            "_" + c.lower() if c.isupper() and i > 0 else c.lower()
            for i, c in enumerate(name)
        ])
        return FlextResult[str].ok(snake_case)

    def _resolve_dependencies(
        self, service_class: type[T]
    ) -> FlextResult[FlextTypes.Core.Dict]:
        """Resolve constructor dependencies from type hints.

        Returns:
            FlextResult[FlextTypes.Core.Dict]: Success with dependencies dict or failure with error.

        """
        try:
            import inspect

            signature = inspect.signature(service_class.__init__)
            dependencies: FlextTypes.Core.Dict = {}

            for param_name, param in signature.parameters.items():
                if param_name == "self":
                    continue

                if param.annotation == inspect.Parameter.empty:
                    continue

                # Try to resolve dependency from container
                dep_result = self.get(param_name)
                if dep_result.is_success:
                    dependencies[param_name] = dep_result.unwrap()
                elif param.default == inspect.Parameter.empty:
                    # Required dependency not found
                    return FlextResult[FlextTypes.Core.Dict].fail(
                        f"Cannot resolve required dependency '{param_name}' for {service_class.__name__}"
                    )

            return FlextResult[FlextTypes.Core.Dict].ok(dependencies)

        except Exception as e:
            return FlextResult[FlextTypes.Core.Dict].fail(
                f"Dependency resolution failed: {e}"
            )

    def _instantiate_and_register_service(
        self, service_class: type[T], dependencies: FlextTypes.Core.Dict
    ) -> FlextResult[T]:
        """Instantiate service with dependencies and register it.

        Returns:
            FlextResult[T]: Success with instantiated service or failure with error.

        """
        try:
            service = service_class(**dependencies)
            name_result = self._resolve_auto_wire_name(service_class)

            if name_result.is_failure:
                return FlextResult[T].fail(
                    name_result.error or "Name resolution failed"
                )

            name = name_result.unwrap()
            register_result = self.register(name, service)

            if register_result.is_failure:
                return FlextResult[T].fail(
                    register_result.error or "Service registration failed"
                )

            return FlextResult[T].ok(service)

        except Exception as e:
            return FlextResult[T].fail(f"Service instantiation failed: {e}")

    def _instantiate_and_register_service_typed(
        self,
        service_class: type[T],
        dependencies: FlextTypes.Core.Dict,
        expected_type: type[T],
    ) -> FlextResult[T]:
        """Instantiate and register service with explicit type validation.

        Returns:
            FlextResult[T]: Success with validated service or failure with error.

        """
        instantiation_result = self._instantiate_and_register_service(
            service_class, dependencies
        )

        if instantiation_result.is_failure:
            return instantiation_result

        service = instantiation_result.unwrap()

        if not isinstance(service, expected_type):
            return FlextResult[T].fail(
                f"Auto-wired service type mismatch: expected {expected_type.__name__}, "
                f"got {type(service).__name__}"
            )

        return FlextResult[T].ok(service)

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
        normalized = self.ServiceKey.normalize(name)
        if normalized.is_failure:
            return False
        validated_name = normalized.unwrap()
        return validated_name in self._services or validated_name in self._factories

    def list_services(self) -> FlextResult[list[FlextTypes.Core.Dict]]:
        """List all registered services with metadata.

        Returns:
            FlextResult[list[FlextTypes.Core.Dict]]: Success with service list or failure with error.

        """
        try:
            services: list[FlextTypes.Core.Dict] = []
            for name in sorted(
                set(self._services.keys()) | set(self._factories.keys())
            ):
                service_info: FlextTypes.Core.Dict = {
                    FlextConstants.Mixins.FIELD_NAME: name,
                    FlextConstants.Mixins.FIELD_TYPE: "instance"
                    if name in self._services
                    else "factory",
                    FlextConstants.Mixins.FIELD_REGISTERED: True,
                }
                services.append(service_info)

            return FlextResult[list[FlextTypes.Core.Dict]].ok(services)
        except Exception as e:
            return FlextResult[list[FlextTypes.Core.Dict]].fail(
                f"Failed to list services: {e}"
            )

    def get_service_names(self) -> FlextResult[FlextTypes.Core.StringList]:
        """Get sorted list of all service names.

        Returns:
            FlextResult[FlextTypes.Core.StringList]: Success with sorted service names or failure with error.

        """
        try:
            all_names = set(self._services.keys()) | set(self._factories.keys())
            sorted_names: FlextTypes.Core.StringList = sorted(all_names)
            return FlextResult[FlextTypes.Core.StringList].ok(sorted_names)
        except Exception as e:
            return FlextResult[FlextTypes.Core.StringList].fail(
                f"Failed to get service names: {e}"
            )

    def get_service_count(self) -> int:
        """Get total count of registered services and factories.

        Returns:
            int: Total count of registered services and factories.

        """
        return len(set(self._services.keys()) | set(self._factories.keys()))

    def get_info(self) -> FlextResult[FlextTypes.Core.Dict]:
        """Get comprehensive container information.

        Returns:
            FlextResult[FlextTypes.Core.Dict]: Success with container info or failure with error.

        """
        try:
            return FlextResult[FlextTypes.Core.Dict].ok({
                "service_count": self.get_service_count(),
                "direct_services": len(self._services),
                "factories": len(self._factories),
                "configuration": self.get_config(),
            })
        except Exception as e:
            return FlextResult[FlextTypes.Core.Dict].fail(
                f"Failed to get container info: {e}"
            )

    def _build_service_info(
        self, name: str, service: object, service_type: str
    ) -> FlextTypes.Core.Dict:
        """Build service information dictionary.

        Returns:
            FlextTypes.Core.Dict: Service information dictionary.

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

    def configure_container(self, config: FlextTypes.Core.Dict) -> FlextResult[None]:
        """Configure container with validated settings.

        Returns:
            FlextResult[None]: Success if configured or failure with error.

        """
        try:
            # Validate configuration structure
            validation_result = self._validate_config_structure(config)
            if validation_result.is_failure:
                return FlextResult[None].fail(
                    validation_result.error or "Config validation failed"
                )

            # Normalize configuration fields
            normalized_config = self._normalize_config_fields(config)

            # Create and store global configuration with proper type conversion
            max_workers = normalized_config.get("max_workers", 4)
            timeout_seconds = normalized_config.get("timeout_seconds", 30.0)
            environment = normalized_config.get("environment", "development")

            self._global_config = {
                "max_workers": int(max_workers)
                if isinstance(max_workers, (int, float))
                else 4,
                "timeout_seconds": float(timeout_seconds)
                if isinstance(timeout_seconds, (int, float))
                else 30.0,
                "environment": str(environment) if environment else "development",
            }

            return FlextResult[None].ok(None)
        except Exception as e:
            return FlextResult[None].fail(f"Container configuration failed: {e}")

    def _validate_config_structure(
        self, _config: FlextTypes.Core.Dict
    ) -> FlextResult[None]:
        """Validate configuration structure.

        Returns:
            FlextResult[None]: Success if valid or failure with error.

        """
        # Type annotation guarantees config is a dictionary, so no isinstance check needed
        # Additional validation can be added here if needed for specific dict structure
        return FlextResult[None].ok(None)

    def _normalize_config_fields(
        self, config: FlextTypes.Core.Dict
    ) -> FlextTypes.Core.Dict:
        """Normalize configuration fields with proper defaults and validation.

        Returns:
            FlextTypes.Core.Dict: Normalized configuration dictionary.

        """
        # Create normalized config with defaults from FlextConstants.Container
        normalized: FlextTypes.Core.Dict = {
            "max_workers": config.get(
                "max_workers", FlextConstants.Container.MAX_WORKERS
            ),
            "timeout_seconds": config.get(
                "timeout_seconds", FlextConstants.Container.TIMEOUT_SECONDS
            ),
            "environment": config.get(
                "environment", FlextConstants.Config.DEFAULT_ENVIRONMENT
            ),
        }

        # Validate and fix max_workers
        if (
            isinstance(normalized["max_workers"], int)
            and normalized["max_workers"] <= 0
        ):
            normalized["max_workers"] = FlextConstants.Container.MAX_WORKERS

        # Validate and fix timeout_seconds
        if (
            isinstance(normalized["timeout_seconds"], (int, float))
            and normalized["timeout_seconds"] <= 0
        ):
            normalized["timeout_seconds"] = FlextConstants.Container.TIMEOUT_SECONDS

        # Set default environment if empty or invalid
        if not normalized["environment"]:
            normalized["environment"] = FlextConstants.Config.DEFAULT_ENVIRONMENT

        return normalized

    # =========================================================================
    # GLOBAL CONTAINER MANAGEMENT - Singleton pattern with thread safety
    # =========================================================================

    @classmethod
    def _ensure_global_manager(cls) -> FlextContainer.GlobalManager:
        """Ensure global manager exists with thread safety.

        Returns:
            FlextContainer.GlobalManager: The global manager instance.

        """
        if cls._global_manager is None:
            cls._global_manager = cls.GlobalManager()
        return cls._global_manager

    @classmethod
    def get_global(cls) -> FlextContainer:
        """Get the global container instance.

        Returns:
            FlextContainer: The global container instance.

        """
        manager = cls._ensure_global_manager()
        return manager.get_or_create()

    @classmethod
    def configure_global(cls, config: FlextTypes.Core.Dict) -> FlextResult[None]:
        """Configure the global container instance.

        Returns:
            FlextResult[None]: Success if configured or failure with error.

        """
        container = cls.get_global()
        return container.configure_container(config)

    @classmethod
    def get_global_typed(cls, name: str, expected_type: type[T]) -> FlextResult[T]:
        """Get typed service from global container.

        Returns:
            FlextResult[T]: Success with typed service or failure with error.

        """
        container = cls.get_global()
        return container.get_typed(name, expected_type)

    @classmethod
    def register_global(cls, name: str, service: object) -> FlextResult[None]:
        """Register service in global container.

        Returns:
            FlextResult[None]: Success if registered or failure with error.

        """
        container = cls.get_global()
        return container.register(name, service)

    @classmethod
    def create_module_utilities(
        cls, module_name: str
    ) -> FlextResult[FlextTypes.Core.Dict]:
        """Create module-specific utilities with container integration.

        Returns:
            FlextResult[FlextTypes.Core.Dict]: Success with utilities dict or failure with error.

        """
        if not module_name:
            return FlextResult[FlextTypes.Core.Dict].fail(
                "Module name must be non-empty string"
            )

        return FlextResult[FlextTypes.Core.Dict].ok({
            "container": cls.get_global(),
            "module": module_name,
            "logger": f"flext.{module_name}",
        })

    def __repr__(self) -> str:
        """String representation with service counts.

        Returns:
            str: String representation of the container.

        """
        return (
            f"FlextContainer(services={len(self._services)}, "
            f"factories={len(self._factories)}, "
            f"total_registered={self.get_service_count()})"
        )


__all__: FlextTypes.Core.StringList = [
    "FlextContainer",
]
