"""Dependency injection utilities for the dispatcher-first CQRS stack.

The module wraps Dependency Injector behind a result-bearing API so handlers and
decorators can register and resolve dependencies without importing the
underlying infrastructure. Configuration stays isolated from dispatcher code, and
singleton semantics remain predictable across runtime usage and tests.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import inspect
import sys
import threading
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Self, overload, override

from dependency_injector import containers as di_containers, providers as di_providers
from pydantic import BaseModel, ValidationError

from flext_core import (
    FlextContext,
    FlextDispatcher,
    FlextLogger,
    FlextRuntime,
    FlextSettings,
    c,
    m,
    p,
    r,
    t,
    u,
)
from flext_core._decorators.discovery import FactoryDecoratorsDiscovery


class FlextContainer(p.Container):
    """Singleton container that exposes DI registration and resolution helpers.

    Services and factories remain local to the container, keeping dispatcher and
    domain code free from infrastructure imports. All operations surface
    ``r`` (r) so failures are explicit. Thread-safe initialization
    guarantees one global instance for runtime usage while allowing scoped
    containers in tests. The class satisfies ``p.Configurable`` and
    ``p.Container`` through structural typing only.
    """

    _global_instance: Self | None = None
    _global_lock: threading.RLock = threading.RLock()
    _context: p.Context | None = None
    _config: p.Settings | None = None
    _user_overrides: t.ConfigMap
    containers: ModuleType
    providers: ModuleType
    _di_bridge: di_containers.DeclarativeContainer
    _di_services: di_containers.DynamicContainer
    _di_resources: di_containers.DynamicContainer
    _di_container: di_containers.DynamicContainer
    _config_provider: di_providers.Configuration
    _base_config_provider: di_providers.Configuration
    _user_config_provider: di_providers.Configuration
    _services: dict[str, m.ServiceRegistration]
    _factories: dict[str, m.FactoryRegistration]
    _resources: dict[str, m.ResourceRegistration]
    _global_config: m.ContainerConfig

    def __new__(
        cls,
        *,
        _config: p.Settings | None = None,
        _context: p.Context | None = None,
        _services: Mapping[str, m.ServiceRegistration] | None = None,
        _factories: Mapping[str, m.FactoryRegistration] | None = None,
        _resources: Mapping[str, m.ResourceRegistration] | None = None,
        _user_overrides: Mapping[str, t.Scalar | t.ConfigMap | Sequence[t.Scalar]]
        | t.ConfigMap
        | None = None,
        _container_config: m.ContainerConfig | None = None,
    ) -> Self:
        """Create or return the global singleton instance.

        Optional keyword arguments support deterministic construction in tests
        while preserving singleton semantics for runtime callers. Double-checked
        locking protects against duplicate initialization under concurrency.
        """
        if cls._global_instance is None:
            with cls._global_lock:
                if cls._global_instance is None:
                    instance = super().__new__(cls)
                    cls._global_instance = instance
        return cls._global_instance

    def __init__(
        self,
        *,
        _config: p.Settings | None = None,
        _context: p.Context | None = None,
        _services: Mapping[str, m.ServiceRegistration] | None = None,
        _factories: Mapping[str, m.FactoryRegistration] | None = None,
        _resources: Mapping[str, m.ResourceRegistration] | None = None,
        _user_overrides: Mapping[str, t.Scalar | t.ConfigMap | Sequence[t.Scalar]]
        | t.ConfigMap
        | None = None,
        _container_config: m.ContainerConfig | None = None,
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
        self.initialize_di_components()
        self.initialize_registrations(
            services=_services,
            factories=_factories,
            resources=_resources,
            global_config=_container_config,
            user_overrides=_user_overrides,
            config=_config,
            context=_context,
        )
        self.sync_config_to_di()
        self.register_existing_providers()
        self.register_core_services()

    @property
    @override
    def config(self) -> p.Settings:
        """Return configuration bound to this container."""
        if self._config is None:
            error_msg = "Configuration must be initialized via initialize_registrations"
            raise RuntimeError(error_msg)
        return self._config

    @property
    @override
    def context(self) -> p.Context:
        """Return the execution context bound to this container.

        The context must be provided during container initialization via the
        `_context` parameter in `__init__` or `get_global()`. If no context
        was provided, this property will raise an error.

        Raises:
            RuntimeError: If context was not provided during initialization.

        Example:
            >>> container = FlextContainer.get_global(context=my_context)
            >>> ctx = container.context  # Returns the provided context

        """
        if not hasattr(self, "_context") or self._context is None:
            error_msg = "Context not initialized. Provide context during container creation: FlextContainer.get_global(context=...) or FlextContainer(_context=...)"
            raise RuntimeError(error_msg)
        return self._context

    @property
    def provide(self) -> Callable[[str], t.RegisterableService]:
        """Return the dependency-injector Provide helper scoped to the bridge.

        ``Provide`` is used alongside the ``@inject`` decorator to declare
        dependencies without importing ``dependency-injector`` in higher layers.
        It resolves registered providers (services, factories, resources,
        configuration) by name and injects the resulting value into the
        decorated callable. Example:

        .. code-block:: python

           from flext_core import FlextContainer, inject

           container = FlextContainer.get_global()
           _ = container.register(
               "token_factory", lambda: {"token": "abc123"}, kind="factory"
           )


           @inject
           def consume(token=container.provide["token_factory"]):
               return token["token"]


           assert consume() == "abc123"
        """
        provide_helper = (
            self._di_bridge.provide if hasattr(self._di_bridge, "provide") else None
        )
        if provide_helper is None or not callable(provide_helper):
            msg = "DI bridge Provide helper not initialized"
            raise RuntimeError(msg)
        provide_fn: Callable[[str], t.RegisterableService] = provide_helper

        def provide_callable(name: str) -> t.RegisterableService:
            provided = provide_fn(name)
            try:
                m.ServiceRegistration(name="provided", service=provided)
                return provided
            except ValidationError:
                msg = "DI bridge Provide helper returned unsupported type"
                raise TypeError(msg) from None

        return provide_callable

    @classmethod
    def _create_scoped_instance(
        cls,
        *,
        config: p.Settings,
        context: p.Context,
        services: Mapping[str, m.ServiceRegistration],
        factories: Mapping[str, m.FactoryRegistration],
        resources: Mapping[str, m.ResourceRegistration],
        user_overrides: t.ConfigMap,
        container_config: m.ContainerConfig,
    ) -> FlextContainer:
        """Create a scoped container instance bypassing singleton pattern.

        This is an internal factory method to safely create non-singleton containers.
        Uses direct attribute assignment (no frozen=True, compatible with FlextRuntime pattern).
        """
        instance = FlextRuntime.create_instance(cls)
        instance.containers = FlextRuntime.dependency_containers()
        instance.providers = FlextRuntime.dependency_providers()
        instance.initialize_di_components()
        instance.initialize_registrations(
            services=services,
            factories=factories,
            resources=resources,
            global_config=container_config,
            user_overrides=user_overrides,
            config=config,
            context=context,
        )
        instance.sync_config_to_di()
        instance.register_existing_providers()
        instance.register_core_services()
        return instance

    @classmethod
    def create(cls, *, auto_register_factories: bool = False) -> Self:
        """Factory method to create a new FlextContainer instance.

        This is the preferred way to instantiate FlextContainer. It provides
        a clean factory pattern that each class owns, respecting Clean
        Architecture principles where higher layers create their own instances.

        Business Rule: Creates a singleton FlextContainer instance with optional
        factory auto-registration. The singleton pattern ensures consistent
        container state across the application lifecycle. When
        ``auto_register_factories=True``, the method scans the calling module
        for functions decorated with ``@d.factory()`` and registers them
        automatically, enabling zero-config factory discovery for services.
        This factory method is the primary entry point for container creation
        in the FLEXT ecosystem.

        Audit Implication: Container creation is a critical infrastructure
        operation that establishes the dependency injection foundation for all
        downstream services. The singleton pattern ensures audit trail consistency
        by maintaining a single container instance. Factory auto-registration
        provides audit visibility into discovered services, enabling complete
        tracking of all registered dependencies and their lifecycle.

        Auto-registration of factories discovers all functions marked with
        @d.factory() decorator and registers them in the container automatically.
        This enables zero-config factory discovery for services.

        Note: FlextContainer uses a singleton pattern, so this method returns
        the global instance on repeated calls unless explicitly reset.

        Args:
            auto_register_factories: If True, scan calling module for @factory()
                decorated functions and auto-register them. Default: False.

        Returns:
            FlextContainer instance.

        Example:
            >>> container = FlextContainer.create(auto_register_factories=True)
            >>> result = container.get("service_name")

        """
        instance = cls()
        if auto_register_factories:
            frame = inspect.currentframe()
            if frame and frame.f_back:
                caller_globals = frame.f_back.f_globals
                module_name_raw = caller_globals.get("__name__", "__main__")
                module_name = str(module_name_raw) if module_name_raw else "__main__"
                caller_module = sys.modules.get(module_name)
                if caller_module:
                    factories = FactoryDecoratorsDiscovery.scan_module(caller_module)
                    for factory_name, factory_config in factories:
                        factory_func_raw = (
                            getattr(caller_module, factory_name)
                            if hasattr(caller_module, factory_name)
                            else None
                        )
                        if factory_func_raw is not None and u.is_factory(
                            factory_func_raw
                        ):
                            factory_func_ref: t.FactoryCallable = factory_func_raw

                            def factory_wrapper(
                                *,
                                _factory_func_ref: t.FactoryCallable = factory_func_ref,
                                _factory_name: str = factory_name,
                                _factory_config: m.FactoryDecoratorConfig = factory_config,
                            ) -> t.RegisterableService:
                                config_callable = getattr(_factory_config, "fn", None)
                                raw_result: t.RegisterableService
                                if callable(config_callable):
                                    config_raw = config_callable()
                                    if isinstance(
                                        config_raw,
                                        (
                                            str,
                                            int,
                                            float,
                                            bool,
                                            datetime,
                                            Path,
                                            BaseModel,
                                        ),
                                    ):
                                        raw_result = config_raw
                                    else:
                                        raw_result = str(config_raw)
                                elif config_callable is not None:
                                    return t.ConfigMap(root={})
                                else:
                                    raw_result = _factory_func_ref()
                                try:
                                    if not u.is_registerable_service(raw_result):
                                        msg = f"Factory '{_factory_name}' returned unsupported type: {raw_result.__class__.__name__}"
                                        raise TypeError(msg)
                                    m.ServiceRegistration(
                                        name=_factory_name, service=raw_result
                                    )
                                    return raw_result
                                except ValidationError:
                                    msg = f"Factory '{_factory_name}' returned unsupported type: {raw_result.__class__.__name__}"
                                    raise TypeError(msg) from None

                            _ = instance.register(
                                factory_config.name, factory_wrapper, kind="factory"
                            )
        return instance

    @classmethod
    def get_global(
        cls, *, config: p.Settings | None = None, context: p.Context | None = None
    ) -> Self:
        """Return the thread-safe global container instance.

        The first call initializes the singleton using the optional configuration
        and context. Subsequent calls return the same instance without modifying
        previously applied settings.
        """
        return cls(_config=config, _context=context)

    @classmethod
    def reset_for_testing(cls) -> None:
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

    @staticmethod
    def _create_container_config() -> m.ContainerConfig:
        """Create default configuration values for container behavior."""
        return m.ContainerConfig(
            enable_singleton=True,
            enable_factory_caching=False,
            max_services=c.Container.DEFAULT_MAX_SERVICES,
            max_factories=c.Container.DEFAULT_MAX_FACTORIES,
        )

    @override
    def clear_all(self) -> None:
        """Clear all service and factory registrations.

        Business Rule: Clears all registrations but preserves singleton instance.
        Used for test cleanup and container reset scenarios. Does not reset
        singleton pattern - use reset_for_testing() for that.

        """
        for name in self.list_services():
            if hasattr(self._di_services, name):
                delattr(self._di_services, name)
            if hasattr(self._di_resources, name):
                delattr(self._di_resources, name)
        self._services.clear()
        self._factories.clear()
        self._resources.clear()

    @override
    def configure(self, config: Mapping[str, t.Container] | None = None) -> Self:
        """Apply user-provided overrides to container configuration.

        Args:
            config: Mapping of configuration keys to values accepted by
                ``t.Scalar``.

        """
        if config is None:
            return self
        config_map: Mapping[str, t.Container] = config
        processed_dict = t.ConfigMap(root={})
        for key, value in config_map.items():
            processed_dict[str(key)] = FlextRuntime.normalize_to_container(value)
        merged = t.ConfigMap(root=dict(self._user_overrides.items()))
        merged.update(dict(processed_dict.items()))
        self._user_overrides = merged
        container_values = self._global_config.model_dump()
        applicable_overrides = {
            key: value for key, value in merged.items() if key in container_values
        }
        if applicable_overrides:
            self._global_config = self._global_config.model_copy(
                update=applicable_overrides, deep=True
            )
        self.sync_config_to_di()
        return self

    def create_module_logger(self, module_name: str | None = None) -> p.Logger:
        """Create a FlextLogger instance for the specified module.

        This method provides direct access to FlextLogger without going through
        the generic DI resolution. Use this for logging needs instead of
        container.get("logger").

        Args:
            module_name: Module name for the logger. Defaults to "flext_core".

        Returns:
            A FlextLogger instance configured for the specified module.

        """
        return FlextLogger.create_module_logger(module_name or "flext_core")

    @overload
    def get[T: t.RegisterableService](
        self, name: str, *, type_cls: type[T]
    ) -> r[T]: ...

    @overload
    def get(self, name: str, *, type_cls: None = None) -> r[t.RegisterableService]: ...

    @override
    def get[T: t.RegisterableService](
        self, name: str, *, type_cls: type[T] | None = None
    ) -> r[T] | r[t.RegisterableService]:
        """Resolve a registered service or factory by name.

        Returns the resolved service as RegisterableService or, when ``type_cls`` is
        provided, validates and returns the requested runtime type.

        Args:
            name: Service identifier to resolve.

        Returns:
            ``r[t.RegisterableService]`` indicating success or failure.

        Example:
            >>> container = FlextContainer()
            >>> container.register("logger", FlextLogger(__name__))
            >>> result = container.get("logger")
            >>> if result.is_success and isinstance(result.value, FlextLogger):
            ...     result.value.info("Resolved")

        """
        if name in self._services:
            service_registration = self._services[name]
            service = service_registration.service
            if type_cls is not None:
                service_for_check: t.RegisterableService = service
                if not u.is_instance_of(service_for_check, type_cls):
                    return r[T].fail(
                        f"Service '{name}' is not of type {(type_cls.__name__ if hasattr(type_cls, '__name__') else 'Unknown')}"
                    )
                typed_service: T = service_for_check
                return r[T].ok(typed_service)
            return r[t.RegisterableService].ok(service)
        if name in self._factories:
            try:
                factory_registration = self._factories[name]
                factory_callable: Callable[[], t.RegisterableService] = (
                    factory_registration.factory
                )
                resolved = factory_callable()
                if type_cls is not None:
                    resolved_for_check: t.RegisterableService = resolved
                    if not u.is_instance_of(resolved_for_check, type_cls):
                        return r[T].fail(f"Factory '{name}' returned wrong type")
                    typed_resolved: T = resolved_for_check
                    return r[T].ok(typed_resolved)
                return r[t.RegisterableService].ok(resolved)
            except (TypeError, ValueError, RuntimeError, KeyError, AttributeError) as e:
                return r[t.RegisterableService].fail(str(e))
        if name in self._resources:
            try:
                resource_registration = self._resources[name]
                resource_callable: Callable[[], t.RegisterableService] = (
                    resource_registration.factory
                )
                resolved = resource_callable()
                if not u.is_registerable_service(resolved):
                    return r[t.RegisterableService].fail(
                        f"Resource '{name}' returned unsupported runtime type"
                    )
                if type_cls is not None:
                    resource_for_check: t.RegisterableService = resolved
                    if not u.is_instance_of(resource_for_check, type_cls):
                        return r[T].fail(f"Resource '{name}' returned wrong type")
                    typed_resource: T = resource_for_check
                    return r[T].ok(typed_resource)
                return r[t.RegisterableService].ok(resolved)
            except (TypeError, ValueError, RuntimeError, KeyError, AttributeError) as e:
                return r[t.RegisterableService].fail(str(e))
        return r[t.RegisterableService].fail(f"Service '{name}' not found")

    @override
    def get_config(self) -> t.ConfigMap:
        """Return the merged configuration exposed by this container."""
        config_dict_raw = self._global_config.model_dump()
        return t.ConfigMap(
            root={
                str(key): FlextRuntime.normalize_to_container(value)
                for key, value in config_dict_raw.items()
            }
        )

    @override
    def has_service(self, name: str) -> bool:
        """Return whether a service or factory is registered for ``name``."""
        return (
            name in self._services or name in self._factories or name in self._resources
        )

    def initialize_di_components(self) -> None:
        """Initialize DI components (bridge, services, resources, container).

        Internal method to set up dependency injection infrastructure.
        Can be called from __init__ or _create_scoped_instance.
        Sets private attributes directly - this is internal initialization.
        """
        bridge_tuple = FlextRuntime.DependencyIntegration.create_layered_bridge()
        bridge = bridge_tuple[0]
        service_module = bridge_tuple[1]
        resource_module = bridge_tuple[2]
        di_container = di_containers.DynamicContainer()
        self._di_bridge = bridge
        self._di_services = service_module
        self._di_resources = resource_module
        self._di_container = di_container
        config_attr = "config"
        if not hasattr(bridge, config_attr):
            error_msg = "Bridge must have config provider"
            raise TypeError(error_msg)
        config_provider_obj = (
            getattr(bridge, config_attr) if hasattr(bridge, config_attr) else None
        )
        if config_provider_obj is None:
            error_msg = "Bridge config provider cannot be None"
            raise TypeError(error_msg)
        if not u.is_instance_of(config_provider_obj, di_providers.Configuration):
            error_msg = "Bridge must have config provider"
            raise TypeError(error_msg)
        config_provider = config_provider_obj
        base_config_provider = di_providers.Configuration()
        user_config_provider = di_providers.Configuration()
        self._base_config_provider = base_config_provider
        self._user_config_provider = user_config_provider
        override_method = getattr(config_provider, "override", None)
        if not callable(override_method):
            error_msg = "Bridge config provider must support override()"
            raise TypeError(error_msg)
        override_method(base_config_provider)
        override_method(user_config_provider)
        di_container.config = config_provider
        self._config_provider = config_provider

    def initialize_registrations(
        self,
        *,
        services: Mapping[str, m.ServiceRegistration] | None = None,
        factories: Mapping[str, m.FactoryRegistration] | None = None,
        resources: Mapping[str, m.ResourceRegistration] | None = None,
        global_config: m.ContainerConfig | None = None,
        user_overrides: Mapping[str, t.Scalar | t.ConfigMap | Sequence[t.Scalar]]
        | t.ConfigMap
        | None = None,
        config: p.Settings | None = None,
        context: p.Context | None = None,
    ) -> None:
        """Initialize service registrations and configuration.

        Internal method to set up registrations and config.
        Can be called from __init__ or _create_scoped_instance.
        Sets private attributes directly - this is internal initialization.
        """
        self._services = dict(services.items()) if services is not None else {}
        self._factories = dict(factories.items()) if factories is not None else {}
        self._resources = dict(resources.items()) if resources is not None else {}
        self._global_config = global_config or self._create_container_config()
        overrides_root: dict[str, t.NormalizedValue | BaseModel] = {}
        if user_overrides is not None:
            if isinstance(user_overrides, t.ConfigMap):
                overrides_root = dict(user_overrides.root)
            else:
                for ok, ov in user_overrides.items():
                    if isinstance(
                        ov, (str, int, float, bool, datetime, Path, BaseModel)
                    ):
                        overrides_root[ok] = ov
                    else:
                        overrides_root[ok] = list(ov)
        user_overrides_map = t.ConfigMap(root=overrides_root)
        self._user_overrides = user_overrides_map
        config_instance: p.Settings = (
            config if config is not None else FlextSettings.get_global()
        )
        self._config = config_instance
        self._context = context

    @override
    def list_services(self) -> Sequence[str]:
        """List the names of registered services and factories."""
        return (
            list(self._services.keys())
            + list(self._factories.keys())
            + list(self._resources.keys())
        )

    @override
    def register(
        self,
        name: str,
        impl: object,
        *,
        kind: str = "service",
    ) -> Self:
        """Register a service instance for dependency resolution.

        Business Rule: The container accepts service values for registration,
        including object, protocols (Config, Ctx, DI, Service, Log,
        Handler, Registry), and callables. This enables dependency injection of
        typed service instances and protocol implementations.

        Args:
            name: Unique key for the registration.
            impl: Concrete instance or callable used for registration.
                Must be object (primitives, BaseModel, callable,
                sequence, or mapping).

        Returns:
            ``r`` indicating whether the registration succeeded. A
            failed result is returned when the name is already registered or
            when construction fails.

        """
        if not name:
            return self
        if kind == "service" and (not u.is_registerable_service(impl)):
            return self
        try:
            if kind == "service":
                if not u.is_registerable_service(impl):
                    return self
                if hasattr(self._di_services, name):
                    return self
                service_impl: t.RegisterableService = impl
                registration = m.ServiceRegistration(
                    name=name,
                    service=service_impl,
                    service_type=service_impl.__class__.__name__,
                )
                self._services[name] = registration
                provider = FlextRuntime.DependencyIntegration.register_object(
                    self._di_services, name, service_impl
                )
                setattr(self._di_bridge, name, provider)
                setattr(self._di_container, name, provider)
                return self
            if kind == "factory":
                if not u.is_factory(impl):
                    return self
                if hasattr(self._di_services, name):
                    return self
                factory_fn: t.FactoryCallable = impl

                def normalized_factory() -> t.RegisterableService:
                    raw_result = factory_fn()
                    if not u.is_registerable_service(raw_result):
                        msg = f"Factory '{name}' returned value that does not satisfy RegisterableService protocol. Expected object, protocol, or callable."
                        raise ValueError(msg)
                    return raw_result

                factory_reg = m.FactoryRegistration(
                    name=name, factory=normalized_factory
                )
                self._factories[name] = factory_reg
                provider = FlextRuntime.DependencyIntegration.register_factory(
                    self._di_services,
                    name,
                    normalized_factory,
                    cache=self._global_config.enable_factory_caching,
                )
                setattr(self._di_bridge, name, provider)
                setattr(self._di_container, name, provider)
                return self
            if not u.is_resource(impl):
                return self
            if hasattr(self._di_resources, name):
                return self
            resource_reg = m.ResourceRegistration(name=name, factory=impl)
            self._resources[name] = resource_reg
            provider = FlextRuntime.DependencyIntegration.register_resource(
                self._di_resources, name, impl
            )
            setattr(self._di_bridge, name, provider)
            setattr(self._di_container, name, provider)
            return self
        except (TypeError, ValueError, RuntimeError, AttributeError) as e:
            _ = e
            return self

    def register_core_services(self) -> None:
        """Auto-register core services for easy DI access.

        Auto-registered services:
        - "config" → FlextSettings singleton
        - "logger" → FlextLogger factory (creates module logger)
        - "context" → FlextContext singleton
        - "container" → Self-reference for nested resolution

        Business Rule: Auto-registers FlextSettings, FlextLogger, and FlextContext
        with standard names ("config", "logger", "context") to enable easy
        dependency injection in downstream projects. Services are registered only
        if not already registered to avoid conflicts.

        This method ensures that core services are accessible via:
        - container.get("config") -> FlextSettings
        - container.get("logger") -> FlextLogger (factory)
        - container.get("context") -> FlextContext
        - container.get("container") -> FlextContainer (self-reference)

        Services are registered as:
        - "config": Singleton instance (container.config property)
        - "logger": Factory that creates module logger
        - "context": Singleton instance (container.context property)
        - "container": Self-reference for nested resolution

        Note: Uses has_service() which checks both dicts and DI container to avoid conflicts.
        """
        if (
            not self.has_service("config")
            and self._config is not None
            and u.is_registerable_service(self._config)
        ):
            _ = self.register("config", self._config)
        if not self.has_service("logger"):
            _ = self.register(
                "logger",
                lambda: FlextLogger.create_module_logger("flext_core"),
                kind="factory",
            )
        if (
            not self.has_service("context")
            and self._context is not None
            and u.is_registerable_service(self._context)
        ):
            _ = self.register("context", self._context)
        if not self.has_service("command_bus"):
            dispatcher = FlextDispatcher()
            service_candidate: object = dispatcher
            if not u.is_registerable_service(service_candidate):
                return
            dispatcher_name = "command_bus"
            registration = m.ServiceRegistration(
                name=dispatcher_name,
                service=service_candidate,
                service_type=type(service_candidate).__name__,
            )
            self._services[dispatcher_name] = registration
            if not hasattr(self._di_services, dispatcher_name):
                provider = FlextRuntime.DependencyIntegration.register_object(
                    self._di_services, dispatcher_name, service_candidate
                )
                setattr(self._di_bridge, dispatcher_name, provider)
                setattr(self._di_container, dispatcher_name, provider)

    def register_existing_providers(self) -> None:
        """Hydrate the dynamic container with current registrations."""
        for name, registration in self._services.items():
            if hasattr(self._di_services, name) or hasattr(self._di_container, name):
                continue
            provider = FlextRuntime.DependencyIntegration.register_object(
                self._di_services, name, registration.service
            )
            setattr(self._di_bridge, name, provider)
            setattr(self._di_container, name, provider)
        for name, factory_registration in self._factories.items():
            if hasattr(self._di_services, name) or hasattr(self._di_container, name):
                continue
            provider = FlextRuntime.DependencyIntegration.register_factory(
                self._di_services,
                name,
                factory_registration.factory,
                cache=self._global_config.enable_factory_caching,
            )
            setattr(self._di_bridge, name, provider)
            setattr(self._di_container, name, provider)
        for name, resource_registration in self._resources.items():
            if hasattr(self._di_resources, name) or hasattr(self._di_container, name):
                continue
            provider = FlextRuntime.DependencyIntegration.register_resource(
                self._di_resources, name, resource_registration.factory
            )
            setattr(self._di_bridge, name, provider)
            setattr(self._di_container, name, provider)

    @override
    def scoped(
        self,
        *,
        config: p.Settings | None = None,
        context: p.Context | None = None,
        subproject: str | None = None,
        services: Mapping[str, object] | None = None,
        factories: Mapping[str, Callable[..., object]] | None = None,
        resources: Mapping[str, Callable[..., object]] | None = None,
    ) -> FlextContainer:
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
            resources: Additional resource registrations merged into the scoped
                instance.

        Returns:
            A container implementing ``p.Container`` with
            isolated state that inherits the global configuration by default.

        """
        config_input = config
        if config is not None:
            base_config: p.Settings = config.model_copy(deep=True)
        else:
            base_config = self.config.model_copy(deep=True)
        if subproject and config_input is None:
            base_config = base_config.model_copy(
                update={"app_name": f"{base_config.app_name}.{subproject}"}
            )
        scoped_context: p.Context
        if context is None:
            ctx_instance = self.context
            clone_method = (
                ctx_instance.clone if hasattr(ctx_instance, "clone") else None
            )
            if callable(clone_method):
                candidate_context = clone_method()
                if u.is_context(candidate_context):
                    scoped_context = candidate_context
                else:
                    scoped_context = FlextContext()
            else:
                scoped_context = FlextContext()
        elif u.is_context(context):
            scoped_context = context
        else:
            scoped_context = self.context.clone()
        if subproject:
            _ = scoped_context.set("subproject", subproject)
        cloned_services: dict[str, m.ServiceRegistration] = {
            name: registration.model_copy(deep=False)
            for name, registration in self._services.items()
        }
        cloned_factories: dict[str, m.FactoryRegistration] = {
            name: registration.model_copy(deep=False)
            for name, registration in self._factories.items()
        }
        cloned_resources: dict[str, m.ResourceRegistration] = {
            name: registration.model_copy(deep=False)
            for name, registration in self._resources.items()
        }
        for name, service in (services or {}).items():
            if not u.is_registerable_service(service):
                continue
            cloned_services[name] = m.ServiceRegistration(
                name=name, service=service, service_type=service.__class__.__name__
            )
        for name, factory in (factories or {}).items():
            if not u.is_factory(factory):
                continue
            cloned_factories[name] = m.FactoryRegistration(name=name, factory=factory)
        for name, resource_factory in (resources or {}).items():
            if u.is_resource(resource_factory):
                cloned_resources[name] = m.ResourceRegistration(
                    name=name, factory=resource_factory
                )
        user_overrides_copy = t.ConfigMap(root=dict(self._user_overrides.items()))
        return FlextContainer._create_scoped_instance(
            config=base_config,
            context=scoped_context,
            services=cloned_services,
            factories=cloned_factories,
            resources=cloned_resources,
            user_overrides=user_overrides_copy,
            container_config=self._global_config.model_copy(deep=True),
        )

    def sync_config_to_di(self) -> None:
        """Synchronize FlextSettings to DI providers.Configuration.

        Dependency Injector's layered ``providers.Configuration`` instances are
        used to avoid manual merges while still honoring validated defaults from
        ``FlextSettings`` and runtime overrides. Base config and user overrides are
        applied as separate providers to keep precedence explicit.

        Also registers namespace configs as factories for easy DI access:
        - "config.ldif" → FlextLdifSettings instance
        - "config.ldap" → FlextLdapSettings instance
        - etc.
        """
        config_dict_raw = self._global_config.model_dump()
        config_map = t.ConfigMap(
            root={
                str(key): FlextRuntime.normalize_to_container(value)
                for key, value in config_dict_raw.items()
            }
        )
        _ = FlextRuntime.DependencyIntegration.bind_configuration(
            self._di_container, config_map
        )
        user_overrides_plain = dict(self._user_overrides.items())
        self._user_config_provider.from_dict(user_overrides_plain)
        namespace_registry_raw = getattr(
            self._config.__class__, "_namespace_registry", None
        )
        if not namespace_registry_raw or not u.is_mapping(namespace_registry_raw):
            return
        namespace_registry = namespace_registry_raw
        namespaces: list[str] = list(namespace_registry.keys())
        if not namespaces:
            return
        for namespace in namespaces:
            factory_name = f"config.{namespace}"
            config_class = FlextSettings.get_namespace_config(namespace)
            if config_class is None:
                continue
            config_class_non_null: type[BaseModel] = config_class

            def _create_namespace_config(
                ns: str = namespace, config_cls: type[BaseModel] = config_class_non_null
            ) -> BaseModel:
                """Factory for creating namespace config instance."""
                return FlextSettings.get_global().get_namespace(ns, config_cls)

            if not self.has_service(factory_name):
                self.register(factory_name, _create_namespace_config, kind="factory")

    def unregister(self, name: str) -> r[bool]:
        """Remove a service or factory registration by name."""
        removed = False
        if name in self._services:
            del self._services[name]
            removed = True
        if name in self._factories:
            del self._factories[name]
            removed = True
        if name in self._resources:
            del self._resources[name]
            removed = True
        if hasattr(self._di_services, name):
            delattr(self._di_services, name)
        if hasattr(self._di_resources, name):
            delattr(self._di_resources, name)
        if removed:
            return r[bool].ok(value=True)
        return r[bool].fail(f"Service '{name}' not found")

    @override
    def wire_modules(
        self,
        *,
        modules: Sequence[ModuleType] | None = None,
        packages: Sequence[str] | None = None,
        classes: Sequence[type] | None = None,
    ) -> None:
        """Wire modules/packages to the DI bridge for @inject/Provide usage."""
        FlextRuntime.DependencyIntegration.wire(
            self._di_container, modules=modules, packages=packages, classes=classes
        )

    def _get_default_config(self) -> p.Settings:
        """Get default configuration instance."""
        return FlextSettings.get_global()

    class Builder:
        """Builder utility for creating FlextContainer instances.

        Provides convenient access to FlextContainer.create() method for
        creating container instances with optional factory auto-registration.
        """

        @classmethod
        def create(cls, *, auto_register_factories: bool = False) -> FlextContainer:
            """Create a new FlextContainer instance.

            Args:
                auto_register_factories: If True, scan calling module for
                    @factory() decorated functions and auto-register them.
                    Default: False.

            Returns:
                FlextContainer instance.

            Example:
                >>> container = FlextContainer.Builder.create(
                ...     auto_register_factories=True
                ... )

            """
            return FlextContainer.create(
                auto_register_factories=auto_register_factories
            )


__all__ = ["FlextContainer"]
