"""Dependency injection container for dispatcher-driven applications.

FlextContainer centralizes service discovery for CQRS handlers, domain services,
and infrastructure utilities. It keeps configuration isolated from the
dispatcher pipeline while providing predictable singleton semantics so
handlers, decorators, and utilities resolve collaborators without manual
wire-up.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from typing import Self, cast

from pydantic import BaseModel

from flext_core._models.container import FlextModelsContainer
from flext_core.config import FlextConfig
from flext_core.context import FlextContext
from flext_core.protocols import FlextProtocols
from flext_core.result import FlextResult
from flext_core.runtime import FlextRuntime
from flext_core.typings import FlextTypes, T


class FlextContainer(FlextProtocols.Configurable):
    """Singleton DI container aligned with the dispatcher-first CQRS flow.

    Services and factories are registered once and resolved by handlers,
    decorators, and utilities without leaking infrastructure concerns into the
    domain layer. Resolution returns ``FlextResult`` to keep failure handling
    explicit and composable.
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

    def __init__(
        self,
        *,
        _config: FlextConfig | None = None,
        _context: FlextContext | None = None,
        _services: dict[str, FlextModelsContainer.ServiceRegistration] | None = None,
        _factories: dict[str, FlextModelsContainer.FactoryRegistration] | None = None,
        _user_overrides: dict[str, FlextTypes.FlexibleValue] | None = None,
        _container_config: FlextModelsContainer.ContainerConfig | None = None,
    ) -> None:
        """Initialize container."""
        if hasattr(self, "_di_container"):
            return

        super().__init__()
        self.containers = FlextRuntime.dependency_containers()
        self.providers = FlextRuntime.dependency_providers()
        self._di_container = self.containers.DynamicContainer()
        self._services: dict[str, FlextModelsContainer.ServiceRegistration] = (
            _services or {}
        )
        self._factories: dict[str, FlextModelsContainer.FactoryRegistration] = (
            _factories or {}
        )
        self._global_config: FlextModelsContainer.ContainerConfig = (
            _container_config or self._create_container_config()
        )
        self._user_overrides: dict[str, FlextTypes.FlexibleValue] = (
            _user_overrides or {}
        )
        self._config = _config or FlextConfig.get_global_instance()
        self._context = _context or FlextContext()
        self._sync_config_to_di()

    @property
    def config(self) -> FlextProtocols.ConfigProtocol:
        """Get configuration bound to this container."""
        return self._config

    @property
    def context(self) -> FlextProtocols.ContextProtocol:
        """Get execution context bound to this container."""
        return self._context

    def _create_container_config(self) -> FlextModelsContainer.ContainerConfig:
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
    ) -> dict[str, object]:
        """Get current configuration."""
        return self._global_config.model_dump()

    def with_config(self, config: dict[str, object]) -> Self:
        """Fluent interface for configuration."""
        self.configure(config)
        return self

    def with_service(
        self,
        name: str,
        service: (
            FlextTypes.GeneralValueType
            | BaseModel
            | Callable[..., FlextTypes.GeneralValueType]
        ),
    ) -> Self:
        """Fluent interface for service registration.

        Accepts GeneralValueType (primitives, sequences, mappings), BaseModel instances,
        or callables for service registration.
        """
        self.register(name, service)
        return self

    def with_factory(
        self,
        name: str,
        factory: Callable[[], FlextTypes.GeneralValueType],
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
            | Callable[..., FlextTypes.GeneralValueType]
        ),
    ) -> FlextResult[bool]:
        """Register a service instance."""
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
        self, name: str, factory: Callable[[], object]
    ) -> FlextResult[bool]:
        """Register a service factory."""
        try:
            if name in self._factories:
                return FlextResult[bool].fail(f"Factory '{name}' already registered")
            registration = FlextModelsContainer.FactoryRegistration(
                name=name,
                factory=factory,
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

    def scoped(
        self,
        *,
        config: FlextConfig | None = None,
        context: FlextContext | None = None,
        subproject: str | None = None,
        services: Mapping[str, object] | None = None,
        factories: Mapping[str, Callable[[], object]] | None = None,
    ) -> FlextProtocols.ContainerProtocol:
        """Create an isolated container scope with optional overrides."""
        base_config = config or self.config.model_copy(deep=True)
        if subproject:
            base_config = base_config.model_copy(
                update={"app_name": f"{base_config.app_name}.{subproject}"},
                deep=True,
            )

        scoped_context = context if context is not None else self.context.clone()
        if subproject:
            scoped_context.set("subproject", subproject)

        cloned_services = {
            name: registration.model_copy(deep=True)
            for name, registration in self._services.items()
        }
        cloned_factories = {
            name: registration.model_copy(deep=True)
            for name, registration in self._factories.items()
        }

        for name, service in (services or {}).items():
            cloned_services[name] = FlextModelsContainer.ServiceRegistration(
                name=name,
                service=service,
                service_type=type(service).__name__,
            )
        for name, factory in (factories or {}).items():
            cloned_factories[name] = FlextModelsContainer.FactoryRegistration(
                name=name,
                factory=factory,
            )

        scoped_container = object.__new__(FlextContainer)
        scoped_container.__init__(
            _config=base_config,
            _context=scoped_context,
            _services=cloned_services,
            _factories=cloned_factories,
            _user_overrides=self._user_overrides.copy(),
            _container_config=self._global_config.model_copy(deep=True),
        )

        return scoped_container


__all__ = ["FlextContainer"]
