"""FlextContainer - Dependency injection container for service management.

This module provides FlextContainer, a type-safe dependency injection container
for managing service lifecycles and resolving dependencies throughout the
FLEXT ecosystem.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Self, cast

from pydantic import BaseModel

from flext_core._models.container import FlextModelsContainer
from flext_core.config import FlextConfig
from flext_core.protocols import FlextProtocols
from flext_core.result import FlextResult
from flext_core.runtime import FlextRuntime
from flext_core.typings import FlextTypes, T


class FlextContainer(FlextProtocols.Configurable):
    """Type-safe dependency injection container for service management.

    Provides centralized service management with singleton pattern support,
    type-safe registration and resolution, and automatic dependency injection.
    """

    _global_instance: Self | None = None
    _global_lock: threading.RLock = threading.RLock()

    def __new__(cls) -> Self:
        """Create or return the global singleton instance."""
        # Double-checked locking pattern for thread-safe singleton
        if cls._global_instance is None:
            with cls._global_lock:
                # Check again inside lock (double-checked locking)
                if cls._global_instance is None:
                    instance = super().__new__(cls)
                    cls._global_instance = instance
        # After lock, _global_instance is guaranteed to be set
        # Type narrowing: mypy doesn't understand double-checked locking
        # but we know _global_instance is set after the lock
        if cls._global_instance is None:
            msg = "Failed to create global instance"
            raise RuntimeError(msg)
        return cls._global_instance

    def __init__(self) -> None:
        """Initialize container."""
        if hasattr(self, "_di_container"):
            return

        super().__init__()
        self.containers = FlextRuntime.dependency_containers()
        self.providers = FlextRuntime.dependency_providers()
        self._di_container = self.containers.DynamicContainer()
        self._services: dict[str, FlextModelsContainer.ServiceRegistration] = {}
        self._factories: dict[str, FlextModelsContainer.FactoryRegistration] = {}
        self._global_config: FlextModelsContainer.ContainerConfig = (
            FlextContainer._create_container_config()
        )
        self._user_overrides: dict[
            str,
            FlextTypes.ScalarValue
            | Sequence[FlextTypes.ScalarValue]
            | Mapping[str, FlextTypes.ScalarValue],
        ] = {}
        self._sync_config_to_di()

    @property
    def config(self) -> FlextConfig:
        """Get current global FlextConfig."""
        return FlextConfig.get_global_instance()

    @staticmethod
    def _create_container_config() -> FlextModelsContainer.ContainerConfig:
        """Create container configuration."""
        return FlextModelsContainer.ContainerConfig(
            enable_singleton=True,
            enable_factory_caching=True,
            max_services=1000,
            max_factories=500,
        )

    def _sync_config_to_di(self) -> None:
        """Sync FlextConfig to internal DI container."""
        # DynamicContainer doesn't have a config attribute
        # Configuration is managed through FlextConfig directly

    def configure(
        self,
        config: Mapping[str, FlextTypes.FlexibleValue],
    ) -> None:
        """Configure container settings."""
        self._user_overrides.update(config)
        self._sync_config_to_di()

    def get_config(
        self,
    ) -> dict[
        str,
        FlextTypes.ScalarValue
        | Sequence[FlextTypes.ScalarValue]
        | Mapping[str, FlextTypes.ScalarValue],
    ]:
        """Get current configuration."""
        return self._global_config.model_dump()

    def with_config(
        self,
        config: dict[
            str,
            FlextTypes.ScalarValue
            | Sequence[FlextTypes.ScalarValue]
            | Mapping[str, FlextTypes.ScalarValue],
        ],
    ) -> Self:
        """Fluent interface for configuration."""
        self.configure(config)
        return self

    def with_service(
        self,
        name: str,
        service: (
            FlextTypes.GeneralValueType
            | BaseModel
            | Callable[..., Any]
            | Any  # Accept any Python object for service registration
        ),
    ) -> Self:
        """Fluent interface for service registration.

        Accepts any service instance for registration in the container.
        Service type is tracked via ServiceRegistration for type safety.
        """
        self.register(name, service)
        return self

    def with_factory(
        self,
        name: str,
        factory: Callable[[], Any],  # Accept factories returning any type
    ) -> Self:
        """Fluent interface for factory registration."""
        self.register_factory(name, factory)
        return self

    def register(
        self,
        name: str,
        service: (
            FlextTypes.GeneralValueType
            | BaseModel
            | Callable[..., Any]
            | Any  # Accept any Python object for service registration
        ),
    ) -> FlextResult[bool]:
        """Register a service instance.

        Accepts any service instance for registration in the container.
        Service type is tracked via ServiceRegistration for type safety.
        """
        try:
            if name in self._services:
                return FlextResult[bool].fail(f"Service '{name}' already registered")
            registration = FlextModelsContainer.ServiceRegistration(
                name=name,
                service=service,
                service_type=type(service).__name__,
            )
            self._services[name] = registration
            return FlextResult[bool].ok(True)
        except Exception as e:
            return FlextResult[bool].fail(str(e))

    def register_factory(
        self,
        name: str,
        factory: Callable[[], Any],  # Accept factories returning any type
    ) -> FlextResult[bool]:
        """Register a service factory."""
        try:
            if name in self._factories:
                return FlextResult[bool].fail(f"Factory '{name}' already registered")
            # Cast factory to match FactoryRegistration type requirements
            # GeneralValueType is compatible with the union type expected by FactoryRegistration
            factory_typed = cast(
                "Callable[[], (FlextTypes.ScalarValue | Sequence[FlextTypes.ScalarValue] | Mapping[str, FlextTypes.ScalarValue])]",
                factory,
            )
            registration = FlextModelsContainer.FactoryRegistration(
                name=name,
                factory=factory_typed,
            )
            self._factories[name] = registration
            return FlextResult[bool].ok(True)
        except Exception as e:
            return FlextResult[bool].fail(str(e))

    def get[T](self, name: str) -> FlextResult[T]:
        """Get service by name."""
        # Try service first
        if name in self._services:
            service = self._services[name].service
            # Runtime type safety guaranteed by container registration
            return FlextResult[T].ok(cast("T", service))

        # Try factory
        if name in self._factories:
            try:
                instance = self._factories[name].factory()
                # Runtime type safety guaranteed by container registration
                return FlextResult[T].ok(cast("T", instance))
            except Exception as e:
                return FlextResult[T].fail(str(e))

        result: FlextResult[T] = FlextResult[T].fail(f"Service '{name}' not found")
        return result

    def get_typed(self, name: str, type_cls: type[T]) -> FlextResult[T]:
        """Get service with type safety."""
        result: FlextResult[T] = self.get(name)
        if result.is_failure:
            return FlextResult[T].fail(result.error or "Unknown error")
        if not isinstance(result.value, type_cls):
            type_name = getattr(type_cls, "__name__", str(type_cls))
            return FlextResult[T].fail(f"Service '{name}' is not of type {type_name}")
        return FlextResult[T].ok(result.value)

    def has_service(self, name: str) -> bool:
        """Check if service is registered."""
        return name in self._services or name in self._factories

    def list_services(self) -> list[str]:
        """List all registered services."""
        return list(self._services.keys()) + list(self._factories.keys())

    def unregister(self, name: str) -> FlextResult[bool]:
        """Unregister a service."""
        if name in self._services:
            del self._services[name]
            return FlextResult[bool].ok(True)
        if name in self._factories:
            del self._factories[name]
            return FlextResult[bool].ok(True)
        return FlextResult[bool].fail(f"Service '{name}' not found")

    def clear_all(self) -> None:
        """Clear all services and factories."""
        self._services.clear()
        self._factories.clear()


__all__ = ["FlextContainer"]
