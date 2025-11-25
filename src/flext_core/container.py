"""FlextContainer - Dependency injection container for service management.

This module provides FlextContainer, a type-safe dependency injection container
for managing service lifecycles and resolving dependencies throughout the
FLEXT ecosystem.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Self, cast

from flext_core.config import FlextConfig
from flext_core.models import FlextModels
from flext_core.protocols import FlextProtocols
from flext_core.result import FlextResult
from flext_core.runtime import FlextRuntime
from flext_core.typings import T


class FlextContainer(FlextProtocols.Configurable):
    """Type-safe dependency injection container for service management.

    Provides centralized service management with singleton pattern support,
    type-safe registration and resolution, and automatic dependency injection.
    """

    _global_instance: FlextContainer | None = None
    _global_lock: threading.RLock = threading.RLock()

    def __new__(cls) -> Self:
        """Create or return the global singleton instance."""
        if cls._global_instance is None:
            with cls._global_lock:
                if cls._global_instance is None:
                    instance = super().__new__(cls)
                    cls._global_instance = instance
        return cast("Self", cls._global_instance)

    def __init__(self) -> None:
        """Initialize container."""
        if hasattr(self, "_di_container"):
            return

        super().__init__()
        self.containers = FlextRuntime.dependency_containers()
        self.providers = FlextRuntime.dependency_providers()
        self._di_container = self.containers.DynamicContainer()
        self._services: dict[str, FlextModels.ServiceRegistration] = {}
        self._factories: dict[str, FlextModels.FactoryRegistration] = {}
        self._global_config: FlextModels.ContainerConfig = (
            self._create_container_config()
        )
        self._user_overrides: dict[str, object] = {}
        self._sync_config_to_di()

    @property
    def config(self) -> FlextConfig:
        """Get current global FlextConfig."""
        return FlextConfig.get_global_instance()

    def _create_container_config(self) -> FlextModels.ContainerConfig:
        """Create container configuration."""
        return FlextModels.ContainerConfig(
            enable_singleton=True,
            enable_factory_caching=True,
            max_services=1000,
            max_factories=500,
        )

    def _sync_config_to_di(self) -> None:
        """Sync FlextConfig to internal DI container."""
        # DynamicContainer doesn't have a config attribute
        # Configuration is managed through FlextConfig directly

    def configure(self, config: dict[str, object]) -> None:
        """Configure container settings."""
        self._user_overrides.update(config)
        self._sync_config_to_di()

    def get_config(self) -> dict[str, object]:
        """Get current configuration."""
        return self._global_config.model_dump()

    def with_config(self, config: dict[str, object]) -> Self:
        """Fluent interface for configuration."""
        self.configure(config)
        return self

    def with_service(self, name: str, service: object) -> Self:
        """Fluent interface for service registration."""
        self.register(name, service)
        return self

    def with_factory(self, name: str, factory: Callable[[], object]) -> Self:
        """Fluent interface for factory registration."""
        self.register_factory(name, factory)
        return self

    def register(self, name: str, service: object) -> FlextResult[bool]:
        """Register a service instance."""
        try:
            if name in self._services:
                return FlextResult[bool].fail(f"Service '{name}' already registered")
            registration = FlextModels.ServiceRegistration(
                name=name,
                service=service,
                service_type=type(service).__name__,
            )
            self._services[name] = registration
            return FlextResult[bool].ok(True)
        except Exception as e:
            return FlextResult[bool].fail(str(e))

    def register_factory(
        self, name: str, factory: Callable[[], object]
    ) -> FlextResult[bool]:
        """Register a service factory."""
        try:
            if name in self._factories:
                return FlextResult[bool].fail(f"Factory '{name}' already registered")
            registration = FlextModels.FactoryRegistration(
                name=name,
                factory=factory,
            )
            self._factories[name] = registration
            return FlextResult[bool].ok(True)
        except Exception as e:
            return FlextResult[bool].fail(str(e))

    def get(self, name: str) -> FlextResult[object]:
        """Get service by name."""
        # Try service first
        if name in self._services:
            return FlextResult[object].ok(self._services[name].service)

        # Try factory
        if name in self._factories:
            try:
                instance = self._factories[name].factory()
                return FlextResult[object].ok(instance)
            except Exception as e:
                return FlextResult[object].fail(str(e))

        return FlextResult[object].fail(f"Service '{name}' not found")

    def get_typed(self, name: str, type_cls: type[T]) -> FlextResult[T]:
        """Get service with type safety."""
        result = self.get(name)
        if result.is_failure:
            return FlextResult[T].fail(result.error or "Unknown error")
        if not isinstance(result.value, type_cls):
            type_name = getattr(type_cls, "__name__", str(type_cls))
            return FlextResult[T].fail(
                f"Service '{name}' is not of type {type_name}"
            )
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
