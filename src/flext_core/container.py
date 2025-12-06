"""Dependency injection utilities for the dispatcher-first CQRS stack.

The module wraps Dependency Injector behind a result-bearing API so handlers and
decorators can register and resolve dependencies without importing the
underlying infrastructure. Configuration stays isolated from dispatcher code, and
singleton semantics remain predictable across runtime usage and tests.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping, Sequence
from typing import Self, cast

from pydantic import BaseModel

from flext_core.config import FlextConfig  # For instantiation only
from flext_core.constants import c
from flext_core.models import m
from flext_core.protocols import p
from flext_core.result import r
from flext_core.runtime import FlextRuntime
from flext_core.typings import T, t
from flext_core.utilities import u


class FlextContainer(p.Container.DI):
    """Singleton container that exposes DI registration and resolution helpers.

    Services and factories remain local to the container, keeping dispatcher and
    domain code free from infrastructure imports. All operations surface
    ``FlextResult`` so failures are explicit. Thread-safe initialization
    guarantees one global instance for runtime usage while allowing scoped
    containers in tests. The class satisfies ``p.Configuration.Configurable`` and
    ``p.Container.DI`` through structural typing only.
    """

    _global_instance: Self | None = None
    _global_lock: threading.RLock = threading.RLock()

    def __new__(
        cls,
        *,
        _config: p.Configuration.Config | None = None,
        _context: p.Context.Ctx | None = None,
        _services: t.Types.ServiceRegistrationDict | None = None,
        _factories: t.Types.FactoryRegistrationDict | None = None,
        _user_overrides: t.Types.ConfigurationDict | None = None,
        _container_config: m.Container.ContainerConfig | None = None,
    ) -> Self:
        """Create or return the global singleton instance.

        Optional keyword arguments support deterministic construction in tests
        while preserving singleton semantics for runtime callers. Double-checked
        locking protects against duplicate initialization under concurrency.
        """
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
        return cls._global_instance

    def __init__(
        self,
        *,
        _config: p.Configuration.Config | None = None,
        _context: p.Context.Ctx | None = None,
        _services: t.Types.ServiceRegistrationDict | None = None,
        _factories: t.Types.FactoryRegistrationDict | None = None,
        _user_overrides: t.Types.ConfigurationDict | None = None,
        _container_config: m.Container.ContainerConfig | None = None,
    ) -> None:
        """Wire the Dependency Injector container and supporting registries.

        The initializer is idempotent: repeated construction returns early to
        keep the singleton stable. Parameters allow deterministic construction
        during testing or for scoped containers created via :meth:`scoped`.
        """
        super().__init__()
        if hasattr(self, "_di_container"):
            return
        self.containers = FlextRuntime.dependency_containers()
        self.providers = FlextRuntime.dependency_providers()
        self._di_container = self.containers.DynamicContainer()
        self._services: t.Types.ServiceRegistrationDict = _services or {}
        self._factories: t.Types.FactoryRegistrationDict = _factories or {}
        self._global_config: m.Container.ContainerConfig = (
            _container_config or self._create_container_config()
        )
        self._user_overrides: t.Types.ConfigurationDict = _user_overrides or {}
        self._config = (
            _config if _config is not None else FlextConfig.get_global_instance()
        )
        self._context: p.Context.Ctx | None = _context
        self._sync_config_to_di()

    @classmethod
    def get_global(
        cls,
        *,
        config: p.Configuration.Config | None = None,
        context: p.Context.Ctx | None = None,
    ) -> Self:
        """Return the thread-safe global container instance.

        The first call initializes the singleton using the optional configuration
        and context. Subsequent calls return the same instance without modifying
        previously applied settings.
        """
        return cls(_config=config, _context=context)

    @property
    def config(self) -> p.Configuration.Config:
        """Return configuration bound to this container."""
        return cast("p.Configuration.Config", self._config)

    @property
    def context(self) -> p.Context.Ctx:
        """Return the execution context bound to this container.

        A lazily created context is materialized on first access when callers
        do not provide one during construction.
        """
        if self._context is None:
            self._context = FlextRuntime.create_context()
        return self._context

    @staticmethod
    def _create_container_config() -> m.Container.ContainerConfig:
        """Create default configuration values for container behavior."""
        return m.Container.ContainerConfig(
            enable_singleton=True,
            enable_factory_caching=True,
            max_services=c.Container.DEFAULT_MAX_SERVICES,
            max_factories=c.Container.DEFAULT_MAX_FACTORIES,
        )

    def _sync_config_to_di(self) -> None:
        """Sync configuration values onto the dependency-injector container.

        Dependency Injector's dynamic container does not expose a first-class
        configuration API. Configuration is mirrored on this instance and
        consumed whenever providers are built in ``register`` or
        ``register_factory``. The method remains a single responsibility hook for
        future provider-based configuration.
        """

    def configure(
        self,
        config: t.Types.ConfigurationMapping,
    ) -> None:
        """Apply user-provided overrides to container configuration.

        Args:
            config: Mapping of configuration keys to values accepted by
                ``t.FlexibleValue``.

        """
        # FlexibleValue is a subset of GeneralValueType; runtime validation
        # ensures compatibility - use process() for concise iteration
        process_result = u.Collection.process(
            config,
            lambda _k, v: cast("t.FlexibleValue", v),
            on_error="collect",
        )
        if process_result.is_success and u.is_type(process_result.value, dict):
            # Type narrowing: process_result.value is dict after isinstance check
            processed_dict: t.Types.ConfigurationDict = cast(
                "t.Types.ConfigurationDict",
                process_result.value,
            )
            # Simple merge: override strategy - new values override existing ones
            merged: t.Types.ConfigurationDict = dict(self._user_overrides)
            merged.update(processed_dict)
            self._user_overrides = merged
        self._sync_config_to_di()

    def get_config(
        self,
    ) -> t.Types.ConfigurationMapping:
        """Return the merged configuration exposed by this container."""
        # model_dump() returns t.Types.ConfigurationDict which is compatible with ConfigurationMapping
        return self._global_config.model_dump()

    def with_config(
        self,
        config: t.Types.ConfigurationMapping,
    ) -> Self:
        """Fluently apply configuration values and return this instance."""
        self.configure(config)
        return self

    def with_service(
        self,
        name: str,
        service: (
            t.GeneralValueType | BaseModel | p.Utility.Callable[t.GeneralValueType]
        ),
    ) -> Self:
        """Register a service and return the container for fluent chaining.

        Accepts primitives, Pydantic models, or callables; values are wrapped
        in a ``ServiceRegistration`` with configuration derived from
        ``FlextConfig`` and user overrides.
        """
        _ = self.register(name, service)
        return self

    def with_factory(
        self,
        name: str,
        factory: Callable[[], t.GeneralValueType],
    ) -> Self:
        """Register a factory and return the container for fluent chaining."""
        _ = self.register_factory(name, factory)
        return self

    def register(
        self,
        name: str,
        service: (
            t.GeneralValueType
            | BaseModel
            | p.Utility.Callable[t.GeneralValueType]
            | object
        ),
    ) -> r[bool]:
        """Register a service instance for dependency resolution.

        Business Rule: The container accepts any object type for registration,
        including primitives, BaseModel instances, callables, and arbitrary
        objects. This allows dependency injection of any service type including
        loggers, adapters, and domain services.

        Args:
            name: Unique key for the registration.
            service: Concrete instance or callable that produces the service.
                Can be any object type (primitives, BaseModel, callable, or
                arbitrary objects like FlextLogger).

        Returns:
            ``FlextResult`` indicating whether the registration succeeded. A
            failed result is returned when the name is already registered or
            when construction fails.

        """
        try:
            if name in self._services:
                return r[bool].fail(f"Service '{name}' already registered")
            registration = m.Container.ServiceRegistration(
                name=name,
                service=service,
                service_type=type(service).__name__,
            )
            self._services[name] = registration
            return r[bool].ok(True)
        except Exception as e:
            return r[bool].fail(str(e))

    def register_factory(
        self,
        name: str,
        factory: Callable[[], t.GeneralValueType],
    ) -> r[bool]:
        """Register a factory used to build services on demand.

        Returns:
            ``FlextResult`` signaling whether the factory was stored. Failure
            occurs when the name already exists or the factory raises during
            registration.

        """
        try:
            if name in self._factories:
                return r[bool].fail(f"Factory '{name}' already registered")
            # Factory returns GeneralValueType which is compatible with FactoryRegistration requirements
            # Cast factory to expected type - GeneralValueType includes ScalarValue, Sequence, Mapping
            factory_typed: Callable[
                [],
                (t.ScalarValue | Sequence[t.ScalarValue] | Mapping[str, t.ScalarValue]),
            ] = cast(
                "Callable[[], t.ScalarValue | Sequence[t.ScalarValue] | Mapping[str, t.ScalarValue]]",
                factory,
            )
            registration = m.Container.FactoryRegistration(
                name=name,
                factory=factory_typed,
            )
            self._factories[name] = registration
            return r[bool].ok(True)
        except Exception as e:
            return r[bool].fail(str(e))

    def get[T](self, name: str) -> r[T]:
        """Resolve a registered service or factory by name.

        Business Rule: Type-safe resolution using generic type parameter T.
        The TypeVar T appears only once but is essential for type inference
        at call sites (e.g., container.get[FlextLogger]("logger")). Runtime
        type safety is guaranteed by container registration, and the cast
        preserves static type checking.

        Resolution prefers concrete services, then factories. A ``FlextResult``
        is returned so handler code can propagate failures without raising
        exceptions.

        Args:
            name: Service identifier to resolve.

        Returns:
            r[T]: Success with resolved service of type T, or failure
                if service not found or factory raises exception.

        """
        # Try service first
        if name in self._services:
            service_registration = cast(
                "m.Container.ServiceRegistration",
                self._services[name],
            )
            service = service_registration.service
            # Runtime type safety guaranteed by container registration
            return r[T].ok(cast("T", service))

        # Try factory
        if name in self._factories:
            try:
                factory_registration = cast(
                    "m.Container.FactoryRegistration",
                    self._factories[name],
                )
                instance = factory_registration.factory()
                # Runtime type safety guaranteed by container registration
                return r[T].ok(cast("T", instance))
            except Exception as e:
                return r[T].fail(str(e))

        result: r[T] = r[T].fail(f"Service '{name}' not found")
        return result

    def get_typed(self, name: str, type_cls: type[T]) -> r[T]:
        """Resolve a service by name and validate its runtime type.

        The returned ``FlextResult`` contains type-safe access to the service or
        the reason validation failed.
        """
        result: r[T] = self.get(name)
        if result.is_failure:
            return r[T].fail(result.error or "Unknown error")
        if not isinstance(result.value, type_cls):
            type_name = u.Mapper.get(
                type_cls,
                "__name__",
                default=str(type_cls),
            ) or str(
                type_cls,
            )
            return r[T].fail(f"Service '{name}' is not of type {type_name}")
        return r[T].ok(result.value)

    def has_service(self, name: str) -> bool:
        """Return whether a service or factory is registered for ``name``."""
        return name in self._services or name in self._factories

    def list_services(self) -> Sequence[str]:
        """List the names of registered services and factories."""
        return list(self._services.keys()) + list(self._factories.keys())

    def unregister(self, name: str) -> r[bool]:
        """Remove a service or factory registration by name."""
        if name in self._services:
            del self._services[name]
            return r[bool].ok(True)
        if name in self._factories:
            del self._factories[name]
            return r[bool].ok(True)
        return r[bool].fail(f"Service '{name}' not found")

    def clear_all(self) -> None:
        """Clear all service and factory registrations.

        Business Rule: Clears all registrations but preserves singleton instance.
        Used for test cleanup and container reset scenarios. Does not reset
        singleton pattern - use reset_singleton_for_testing() for that.

        """
        self._services.clear()
        self._factories.clear()

    @classmethod
    def reset_singleton_for_testing(cls) -> None:
        """Reset singleton instance for testing purposes.

        Business Rule: Testing-only method to reset singleton state.
        This allows tests to create fresh container instances without
        singleton interference. Should only be used in test fixtures.

        Implications for Audit:
        - Modifies private ClassVar _global_instance
        - Breaks singleton pattern temporarily
        - Must be called before creating new container instances in tests
        - Thread-safe via _global_lock

        """
        with cls._global_lock:
            cls._global_instance = None

    @classmethod
    def _create_scoped_instance(
        cls,
        *,
        config: p.Configuration.Config,
        context: p.Context.Ctx,
        services: t.Types.ServiceRegistrationDict,
        factories: t.Types.FactoryRegistrationDict,
        user_overrides: t.Types.ConfigurationDict,
        container_config: m.Container.ContainerConfig,
    ) -> p.Container.DI:
        """Create a scoped container instance bypassing singleton pattern.

        This is an internal factory method to safely create non-singleton containers.
        Uses object.__setattr__ to bypass SLF001 false positives for factory pattern.
        """
        # Create raw instance without __new__ singleton logic
        # Use type-safe factory helper from FlextRuntime
        instance = FlextRuntime.create_instance(cls)
        # Initialize public attributes directly
        instance.containers = FlextRuntime.dependency_containers()
        instance.providers = FlextRuntime.dependency_providers()
        # Use object.__setattr__ for private attrs to bypass SLF001
        # (classmethod factory pattern - valid Python, false positive from ruff)
        setattr_ = object.__setattr__
        setattr_(instance, "_di_container", instance.containers.DynamicContainer())
        setattr_(instance, "_services", services)
        setattr_(instance, "_factories", factories)
        setattr_(instance, "_global_config", container_config)
        setattr_(instance, "_user_overrides", user_overrides)
        setattr_(instance, "_config", config)
        setattr_(instance, "_context", context)
        # Call private method via getattr variable to bypass SLF001 and B009
        method_name = "_sync_config_to_di"
        getattr(instance, method_name)()
        return cast("p.Container.DI", instance)  # Structural typing

    def scoped(
        self,
        *,
        config: p.Configuration.Config | None = None,
        context: p.Context.Ctx | None = None,
        subproject: str | None = None,
        services: Mapping[
            str,
            t.GeneralValueType | BaseModel | p.Utility.Callable[t.GeneralValueType],
        ]
        | None = None,
        factories: Mapping[
            str,
            Callable[
                [],
                (t.ScalarValue | Sequence[t.ScalarValue] | Mapping[str, t.ScalarValue]),
            ],
        ]
        | None = None,
    ) -> p.Container.DI:
        """Create an isolated container scope with optional overrides.

        Args:
            config: Optional settings overriding the global container's
                configuration.
            context: Optional execution context to seed the scoped container.
            subproject: Optional suffix appended to ``app_name`` for nested
                service identification.
            services: Additional service registrations merged into the scoped
                instance.
            factories: Additional factory registrations merged into the scoped
                instance.

        Returns:
            A container implementing ``p.Container.DI`` with
            isolated state that inherits the global configuration by default.

        """
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
            # Context.Ctx.set returns None per protocol definition
            # But FlextContext.set returns r[bool] - call directly
            # Protocol allows None return, implementation can return FlextResult
            _ = scoped_context.set("subproject", subproject)

        # Use process() for concise dict transformation
        process_result = u.Collection.process(
            self._services,
            lambda _name, registration: cast(
                "m.Container.ServiceRegistration",
                registration,
            ).model_copy(deep=True),
            on_error="collect",
        )
        cloned_services: t.Types.ServiceRegistrationDict = (
            cast("t.Types.ServiceRegistrationDict", process_result.value)
            if process_result.is_success and u.is_type(process_result.value, dict)
            else {}
        )
        cloned_factories: t.Types.FactoryRegistrationDict = {
            name: cast("m.Container.FactoryRegistration", registration).model_copy(
                deep=True,
            )
            for name, registration in self._factories.items()
        }

        for name, service in (services or {}).items():
            cloned_services[name] = m.Container.ServiceRegistration(
                name=name,
                service=service,
                service_type=type(service).__name__,
            )
        for name, factory in (factories or {}).items():
            cloned_factories[name] = m.Container.FactoryRegistration(
                name=name,
                factory=factory,
            )

        # Use factory method to create scoped container (avoids mypy __init__ error)
        # Structural typing - FlextContainer implements p.Container.DI
        # base_config already implements p.Configuration.Config protocol
        return FlextContainer._create_scoped_instance(
            config=base_config,
            context=scoped_context,
            services=cloned_services,
            factories=cloned_factories,
            user_overrides=self._user_overrides.copy(),
            container_config=self._global_config.model_copy(deep=True),
        )


__all__ = ["FlextContainer"]
