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
from collections.abc import Callable, Mapping, Sequence
from typing import Self, cast

from pydantic import BaseModel

from flext_core._models.container import FlextModelsContainer
from flext_core.config import FlextConfig
from flext_core.protocols import FlextProtocols
from flext_core.result import FlextResult
from flext_core.runtime import FlextRuntime
from flext_core.typings import FlextTypes, T


def _create_default_context() -> FlextProtocols.ContextProtocol:
    """Factory to create default context - breaks container↔context cycle.

    ARCHITECTURAL NOTE: This lazy import is INTENTIONAL to break mutual dependency.
    - FlextContainer.__init__ needs FlextContext for initialization
    - FlextContext.get_container() returns FlextContainer
    Using FlextProtocols for types + factory for instantiation resolves the cycle.
    """
    from flext_core.context import FlextContext  # noqa: PLC0415

    return cast("FlextProtocols.ContextProtocol", FlextContext())


class FlextContainer:
    """Singleton DI container aligned with the dispatcher-first CQRS flow.

    Services and factories are registered once and resolved by handlers,
    decorators, and utilities without leaking infrastructure concerns into the
    domain layer. Resolution returns ``FlextResult`` to keep failure handling
    explicit and composable.

    Implements FlextProtocols.Configurable and FlextProtocols.ContainerProtocol
    through structural typing (duck typing).
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
        _context: FlextProtocols.ContextProtocol | None = None,
        _services: dict[str, FlextModelsContainer.ServiceRegistration] | None = None,
        _factories: dict[str, FlextModelsContainer.FactoryRegistration] | None = None,
        _user_overrides: dict[str, FlextTypes.FlexibleValue] | None = None,
        _container_config: FlextModelsContainer.ContainerConfig | None = None,
    ) -> None:
        """Initialize container."""
        if hasattr(self, "_di_container"):
            return
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
        self._config = (
            _config if _config is not None else FlextConfig.get_global_instance()
        )
        self._context = _context or _create_default_context()
        self._sync_config_to_di()

    @property
    def config(self) -> FlextProtocols.ConfigProtocol:
        """Get configuration bound to this container."""
        return cast("FlextProtocols.ConfigProtocol", self._config)

    @property
    def context(self) -> FlextProtocols.ContextProtocol:
        """Get execution context bound to this container."""
        return self._context

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
        config: Mapping[str, FlextTypes.GeneralValueType],
    ) -> None:
        """Configure container settings."""
        # Convert GeneralValueType to FlexibleValue for _user_overrides
        # FlexibleValue is a subset of GeneralValueType, so we can safely convert
        converted: dict[str, FlextTypes.FlexibleValue] = {
            k: cast("FlextTypes.FlexibleValue", v) for k, v in config.items()
        }
        self._user_overrides.update(converted)
        self._sync_config_to_di()

    def get_config(
        self,
    ) -> FlextTypes.Types.ConfigurationMapping:
        """Get current configuration."""
        return cast(
            "FlextTypes.Types.ConfigurationMapping", self._global_config.model_dump()
        )

    def with_config(
        self,
        config: FlextTypes.Types.ConfigurationMapping,
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
        factory: Callable[[], FlextTypes.GeneralValueType],
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

    @classmethod
    def _create_scoped_instance(  # noqa: PLR0913 - Factory needs all initialization params
        cls,
        *,
        config: FlextConfig,
        context: FlextProtocols.ContextProtocol,
        services: dict[str, FlextModelsContainer.ServiceRegistration],
        factories: dict[str, FlextModelsContainer.FactoryRegistration],
        user_overrides: dict[str, FlextTypes.FlexibleValue],
        container_config: FlextModelsContainer.ContainerConfig,
    ) -> FlextContainer:
        """Create a scoped container instance bypassing singleton pattern.

        This is an internal factory method to safely create non-singleton containers.
        Note: SLF001 warnings are false positives - classmethod accessing own class
        instance's private attributes during construction is valid Python.
        """
        # Create raw instance without __new__ singleton logic
        # Use cast to help type checker understand this is a FlextContainer instance
        instance = cast("FlextContainer", object.__new__(cls))
        # Initialize directly - factory pattern requires private attribute access
        instance.containers = FlextRuntime.dependency_containers()
        instance.providers = FlextRuntime.dependency_providers()
        instance._di_container = instance.containers.DynamicContainer()  # noqa: SLF001
        instance._services = services  # noqa: SLF001
        instance._factories = factories  # noqa: SLF001
        instance._global_config = container_config  # noqa: SLF001
        instance._user_overrides = user_overrides  # noqa: SLF001
        instance._config = config  # noqa: SLF001
        instance._context = context  # noqa: SLF001
        instance._sync_config_to_di()  # noqa: SLF001
        return instance

    def scoped(
        self,
        *,
        config: FlextConfig | None = None,
        context: FlextProtocols.ContextProtocol | None = None,
        subproject: str | None = None,
        services: Mapping[
            str,
            FlextTypes.GeneralValueType
            | BaseModel
            | Callable[..., FlextTypes.GeneralValueType],
        ]
        | None = None,
        factories: Mapping[
            str,
            Callable[
                [],
                (
                    FlextTypes.ScalarValue
                    | Sequence[FlextTypes.ScalarValue]
                    | Mapping[str, FlextTypes.ScalarValue]
                ),
            ],
        ]
        | None = None,
    ) -> FlextProtocols.ContainerProtocol:
        """Create an isolated container scope with optional overrides."""
        base_config = (
            config if config is not None else self.config.model_copy(deep=True)
        )
        if subproject:
            base_config = base_config.model_copy(
                update={"app_name": f"{base_config.app_name}.{subproject}"},
                deep=True,
            )

        scoped_context = context if context is not None else self.context.clone()
        if subproject:
            # ContextProtocol.set returns None per protocol definition
            # But FlextContext.set returns FlextResult[bool] - call directly
            # Protocol allows None return, implementation can return FlextResult
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

        # Cast base_config to FlextConfig for factory method
        config_for_init = (
            base_config
            if isinstance(base_config, FlextConfig)
            else cast("FlextConfig", base_config)
        )

        # Use factory method to create scoped container (avoids mypy __init__ error)
        scoped_container = FlextContainer._create_scoped_instance(
            config=config_for_init,
            context=scoped_context,
            services=cloned_services,
            factories=cloned_factories,
            user_overrides=self._user_overrides.copy(),
            container_config=self._global_config.model_copy(deep=True),
        )

        return cast("FlextProtocols.ContainerProtocol", scoped_container)


__all__ = ["FlextContainer"]
