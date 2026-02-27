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
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, Self, TypeGuard, cast, override

from dependency_injector import containers as di_containers, providers as di_providers
from pydantic import BaseModel, ValidationError

from flext_core._decorators import FactoryDecoratorsDiscovery
from flext_core.constants import c
from flext_core.context import FlextContext
from flext_core.dispatcher import FlextDispatcher
from flext_core.loggings import FlextLogger
from flext_core.models import m
from flext_core.protocols import p
from flext_core.result import r
from flext_core.runtime import FlextRuntime
from flext_core.settings import FlextSettings
from flext_core.typings import t


class FlextContainer(p.DI):
    """Singleton container that exposes DI registration and resolution helpers.

    Services and factories remain local to the container, keeping dispatcher and
    domain code free from infrastructure imports. All operations surface
    ``r`` (FlextResult) so failures are explicit. Thread-safe initialization
    guarantees one global instance for runtime usage while allowing scoped
    containers in tests. The class satisfies ``p.Configurable`` and
    ``p.DI`` through structural typing only.
    """

    _global_instance: Self | None = None
    _global_lock: threading.RLock = threading.RLock()
    _context: p.Context | None = None
    _config: p.Config | None = None
    _user_overrides: m.ConfigMap
    containers: ModuleType
    providers: ModuleType
    _di_bridge: di_containers.DeclarativeContainer
    _di_services: di_containers.DynamicContainer
    _di_resources: di_containers.DynamicContainer
    _di_container: di_containers.DynamicContainer
    _config_provider: di_providers.Configuration
    _base_config_provider: di_providers.Configuration
    _user_config_provider: di_providers.Configuration
    _services: MutableMapping[str, m.Container.ServiceRegistration]
    _factories: MutableMapping[str, m.Container.FactoryRegistration]
    _resources: MutableMapping[str, m.Container.ResourceRegistration]
    _global_config: m.Container.ContainerConfig

    @override
    def _protocol_name(self) -> str:
        """Return protocol name for BaseProtocol compliance."""
        return "FlextContainer"

    def __new__(
        cls,
        *,
        _config: p.Config | None = None,
        _context: p.Context | None = None,
        _services: Mapping[str, m.Container.ServiceRegistration] | None = None,
        _factories: Mapping[str, m.Container.FactoryRegistration] | None = None,
        _resources: Mapping[str, m.Container.ResourceRegistration] | None = None,
        _user_overrides: Mapping[
            str, t.ScalarValue | m.ConfigMap | Sequence[t.ScalarValue],
        ]
        | m.ConfigMap
        | None = None,
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
        _config: p.Config | None = None,
        _context: p.Context | None = None,
        _services: Mapping[str, m.Container.ServiceRegistration] | None = None,
        _factories: Mapping[str, m.Container.FactoryRegistration] | None = None,
        _resources: Mapping[str, m.Container.ResourceRegistration] | None = None,
        _user_overrides: Mapping[
            str, t.ScalarValue | m.ConfigMap | Sequence[t.ScalarValue],
        ]
        | m.ConfigMap
        | None = None,
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
        # Initialize DI components using public method
        self.initialize_di_components()
        # Initialize registrations using public method
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
        # Auto-register core services for easy DI access
        self.register_core_services()

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
            # Get the caller's frame to discover factories in calling module
            frame = inspect.currentframe()
            if frame and frame.f_back:
                caller_globals: Mapping[str, object] = frame.f_back.f_globals
                # Get module name from globals
                module_name_raw = caller_globals.get("__name__", "__main__")
                module_name = str(module_name_raw) if module_name_raw else "__main__"
                # Get module object from globals (usually available as __import__ or direct reference)

                caller_module = sys.modules.get(module_name)
                if caller_module:
                    # Scan module for factory-decorated functions
                    factories = FactoryDecoratorsDiscovery.scan_module(caller_module)
                    for factory_name, factory_config in factories:
                        # Get actual factory function from module
                        factory_func_raw = (
                            getattr(caller_module, factory_name)
                            if hasattr(caller_module, factory_name)
                            else None
                        )
                        # Type narrowing: Factory functions decorated with @factory()
                        # are expected to return RegisterableService, but getattr returns object
                        if callable(factory_func_raw):

                            def factory_wrapper(
                                *_args: Any,
                                _factory_func_ref: Callable[[], object] | None = (
                                    factory_func_raw
                                ),
                                _factory_name: str = factory_name,
                                **kwargs: Any,
                            ) -> t.RegisterableService:
                                fn_override = kwargs.get("fn")
                                if fn_override is not None:
                                    if callable(fn_override):
                                        raw_result = fn_override()
                                    else:
                                        return ""
                                else:
                                    if not callable(_factory_func_ref):
                                        msg = (
                                            f"Factory '{_factory_name}' is not callable"
                                        )
                                        raise TypeError(msg)
                                    raw_result = _factory_func_ref()
                                try:
                                    m.Container.ServiceRegistration(
                                        name=_factory_name,
                                        service=cast(
                                            "t.RegisterableService", raw_result,
                                        ),
                                    )
                                    return cast("t.RegisterableService", raw_result)
                                except ValidationError:
                                    msg = (
                                        f"Factory '{_factory_name}' returned unsupported type: "
                                        f"{raw_result.__class__.__name__}"
                                    )
                                    raise TypeError(msg) from None

                            # Register using the name from decorator config
                            _ = instance.register_factory(
                                factory_config.name,
                                factory_wrapper,
                            )

        return instance

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
           container.register_factory("token_factory", lambda: {"token": "abc123"})


           @inject
           def consume(token=container.provide["token_factory"]):
               return token["token"]


           assert consume() == "abc123"
        """
        # getattr accesses Provide from bridge object safely
        # Provide is always initialized by dependency-injector in _di_bridge
        provide_helper = (
            self._di_bridge.Provide
            if hasattr(self._di_bridge, "Provide")
            else None
        )
        # Type narrowing via isinstance check + RuntimeError if missing
        if provide_helper is None or not callable(provide_helper):
            msg = "DI bridge Provide helper not initialized"
            raise RuntimeError(msg)
        provide_fn: Callable[[str], object] = provide_helper

        # After narrowing, provide_helper is confirmed callable
        def provide_callable(name: str) -> t.RegisterableService:
            provided = provide_fn(name)
            try:
                m.Container.ServiceRegistration(
                    name="provided",
                    service=cast("t.RegisterableService", provided),
                )
                return cast("t.RegisterableService", provided)
            except ValidationError:
                msg = "DI bridge Provide helper returned unsupported type"
                raise TypeError(msg) from None

        return provide_callable

    @classmethod
    def get_global(
        cls,
        *,
        config: p.Config | None = None,
        context: p.Context | None = None,
    ) -> Self:
        """Return the thread-safe global container instance.

        The first call initializes the singleton using the optional configuration
        and context. Subsequent calls return the same instance without modifying
        previously applied settings.
        """
        return cls(_config=config, _context=context)

    @property
    @override
    def config(self) -> p.Config:
        """Return configuration bound to this container."""
        # Type narrowing: self._config is p.Config (set in initialize_registrations)
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
            error_msg = (
                "Context not initialized. Provide context during container creation: "
                "FlextContainer.get_global(context=...) or FlextContainer(_context=...)"
            )
            raise RuntimeError(error_msg)
        # Type narrowing: after check, _context is not None
        return self._context

    def initialize_di_components(self) -> None:
        """Initialize DI components (bridge, services, resources, container).

        Internal method to set up dependency injection infrastructure.
        Can be called from __init__ or _create_scoped_instance.
        Sets private attributes directly - this is internal initialization.
        """
        bridge_tuple = FlextRuntime.DependencyIntegration.create_layered_bridge()
        bridge = bridge_tuple[0]  # DeclarativeContainer with config attribute
        service_module = bridge_tuple[1]  # DynamicContainer
        resource_module = bridge_tuple[2]  # DynamicContainer
        di_container = di_containers.DynamicContainer()
        # Internal initialization - direct assignment to private attributes
        # These are set during object construction, not accessed from outside
        self._di_bridge = bridge
        self._di_services = service_module
        self._di_resources = resource_module
        self._di_container = di_container
        # bridge (DeclarativeContainer) has a config provider
        # The bridge is created by dependency-injector and has dynamic attributes
        # We access it without strict type checking for this internal setup
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
        if not self._is_instance_of(
            config_provider_obj,
            di_providers.Configuration,
        ):
            error_msg = "Bridge must have config provider"
            raise TypeError(error_msg)
        config_provider = config_provider_obj
        base_config_provider = di_providers.Configuration()
        user_config_provider = di_providers.Configuration()
        self._base_config_provider = base_config_provider
        self._user_config_provider = user_config_provider
        # Configure providers - override() returns OverridingContext for chaining
        # We call it for side effects (configuring the provider), not for the return value
        # override() may return None or OverridingContext - we don't need the return value
        _ = config_provider.override(base_config_provider)
        _ = config_provider.override(user_config_provider)
        di_container.config = config_provider
        self._config_provider = config_provider

    def initialize_registrations(
        self,
        *,
        services: Mapping[str, m.Container.ServiceRegistration] | None = None,
        factories: Mapping[str, m.Container.FactoryRegistration] | None = None,
        resources: Mapping[str, m.Container.ResourceRegistration] | None = None,
        global_config: m.Container.ContainerConfig | None = None,
        user_overrides: Mapping[
            str, t.ScalarValue | m.ConfigMap | Sequence[t.ScalarValue],
        ]
        | m.ConfigMap
        | None = None,
        config: p.Config | None = None,
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
        user_overrides_map = m.ConfigMap(
            root=dict(user_overrides.items()) if user_overrides is not None else {},
        )
        self._user_overrides = user_overrides_map
        # Type narrowing: config can be None, but property handles None case
        config_instance: p.Config = (
            config if config is not None else FlextSettings.get_global_instance()
        )
        self._config = config_instance
        # Type narrowing: context can be None, but property will raise error if accessed
        # Direct assignment is safe - _context is an instance attribute
        # _context is declared as p.Context | None = None (instance attribute)
        # If context is None, property will raise RuntimeError on access (no lazy creation)
        self._context = context

    def _get_default_config(self) -> p.Config:
        """Get default configuration instance."""
        return FlextSettings.get_global_instance()

    @staticmethod
    def _create_container_config() -> m.Container.ContainerConfig:
        """Create default configuration values for container behavior."""
        return m.Container.ContainerConfig(
            enable_singleton=True,
            # Factories should default to new instances on each resolution to
            # preserve expected DI semantics; caching can be opt-in via
            # ``configure``.
            enable_factory_caching=False,
            max_services=c.Container.DEFAULT_MAX_SERVICES,
            max_factories=c.Container.DEFAULT_MAX_FACTORIES,
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
        # L0: Base defaults from FlextSettings
        config_dict_raw = self._global_config.model_dump()
        config_map = t.ConfigMap(
            root={
                str(key): FlextRuntime.normalize_to_general_value(value)
                for key, value in config_dict_raw.items()
            },
        )
        _ = FlextRuntime.DependencyIntegration.bind_configuration(
            self._di_container,
            config_map,
        )

        # Apply user overrides
        # _user_overrides is always ConfigurationDict after initialize_registrations
        # (initialized as user_overrides or {}), so it's never None after __init__
        user_overrides_plain = dict(self._user_overrides.items())
        self._user_config_provider.from_dict(user_overrides_plain)

        # Register namespace configs as factories
        # Access namespace registry via FlextSettings (self._config), not ContainerConfig
        # Note: Namespace registry is accessed via FlextSettings class, not ContainerConfig
        # Use getattr to safely access registry if it exists
        namespace_registry = (
            self._config.__class__._namespace_registry
            if hasattr(self._config.__class__, "_namespace_registry")
            else {}
        )
        for namespace in namespace_registry:
            factory_name = f"config.{namespace}"

            # Get config class for this namespace from FlextSettings
            # Use getattr to safely access method if it exists
            get_namespace_config = (
                self._config.get_namespace_config
                if hasattr(self._config, "get_namespace_config")
                else None
            )
            if get_namespace_config is None:
                continue
            config_class = get_namespace_config(namespace)
            if config_class is None:
                continue

            def _create_namespace_config(
                ns: str = namespace,
                config_cls: type[BaseModel] = config_class,
            ) -> BaseModel:
                """Factory for creating namespace config instance."""
                get_namespace = (
                    self._config.get_namespace
                    if hasattr(self._config, "get_namespace")
                    else None
                )
                if get_namespace is None:
                    return config_cls()
                result = get_namespace(ns, config_cls)
                if FlextContainer._is_instance_of(result, BaseModel):
                    return result
                return config_cls()

            # Only register if not already registered
            if not self.has_service(factory_name):
                _ = self.register_factory(factory_name, _create_namespace_config)

    def register_existing_providers(self) -> None:
        """Hydrate the dynamic container with current registrations."""
        for name, registration in self._services.items():
            # Skip if already registered in DI container (e.g., from parent container)
            if hasattr(self._di_services, name) or hasattr(self._di_container, name):
                continue
            provider = FlextRuntime.DependencyIntegration.register_object(
                self._di_services,
                name,
                registration.service,
            )
            setattr(self._di_bridge, name, provider)
            setattr(self._di_container, name, provider)

        for name, factory_registration in self._factories.items():
            # Skip if already registered in DI container (e.g., from parent container)
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
            # Skip if already registered in DI container (e.g., from parent container)
            if hasattr(self._di_resources, name) or hasattr(self._di_container, name):
                continue
            provider = FlextRuntime.DependencyIntegration.register_resource(
                self._di_resources,
                name,
                resource_registration.factory,
            )
            setattr(self._di_bridge, name, provider)
            setattr(self._di_container, name, provider)

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
        # Register config if not already registered
        # Note: _di_container.config is the Configuration provider, not the service
        # We need to check if "config" is registered as a service, not just if the attribute exists
        # Type narrowing: FlextSettings extends BaseModel, so u.is_pydantic_model narrows type
        if not self.has_service("config") and self._config is not None:
            try:
                m.Container.ServiceRegistration(
                    name="config",
                    service=cast("t.RegisterableService", self._config),
                )
                _ = self.register("config", cast("t.RegisterableService", self._config))
            except ValidationError:
                pass  # Skip registration if validation fails

        # Register logger factory if not already registered
        # ServiceRegistration now uses SkipValidation - can store any service type
        if not self.has_service("logger"):
            # FlextLogger implements p.Log.StructlogLogger protocol structurally
            # ServiceRegistration.service field uses SkipValidation for protocols
            _ = self.register(
                "logger",
                FlextLogger.create_module_logger("flext_core"),
            )

        # Register context if not already registered
        # ServiceRegistration uses SkipValidation - can register any service type
        if not self.has_service("context") and self._context is not None:
            try:
                m.Container.ServiceRegistration(
                    name="context",
                    service=cast("t.RegisterableService", self._context),
                )
                _ = self.register(
                    "context", cast("t.RegisterableService", self._context),
                )
            except ValidationError:
                pass  # Skip registration if validation fails

        # Register dispatcher as command_bus if not already registered
        # FlextDispatcher now standalone (no FlextService inheritance)
        # Safe to instantiate without triggering container creation
        if not self.has_service("command_bus"):
            dispatcher = FlextDispatcher()
            _ = self.register("command_bus", cast("t.RegisterableService", dispatcher))

    @override
    def configure(
        self,
        config: Mapping[str, t.ScalarValue],
    ) -> None:
        """Apply user-provided overrides to container configuration.

        Args:
            config: Mapping of configuration keys to values accepted by
                ``t.ScalarValue``.

        """
        processed_dict = m.ConfigMap(root={})
        for key, value in config.items():
            processed_dict[str(key)] = FlextRuntime.normalize_to_general_value(value)

        merged = m.ConfigMap(root=dict(self._user_overrides.items()))
        merged.update(dict(processed_dict.items()))
        self._user_overrides = merged
        # Sync validated overrides onto the container config model so
        # provider registration respects updated settings (e.g.,
        # enable_factory_caching).
        container_values = self._global_config.model_dump()
        applicable_overrides = {
            key: value for key, value in merged.items() if key in container_values
        }
        if applicable_overrides:
            self._global_config = self._global_config.model_copy(
                update=applicable_overrides,
                deep=True,
            )
        self.sync_config_to_di()

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
            self._di_container,
            modules=modules,
            packages=packages,
            classes=classes,
        )

    @override
    def get_config(
        self,
    ) -> t.ConfigMap:
        """Return the merged configuration exposed by this container."""
        config_dict_raw = self._global_config.model_dump()
        return t.ConfigMap(
            root={
                str(key): FlextRuntime.normalize_to_general_value(value)
                for key, value in config_dict_raw.items()
            },
        )

    def with_config(
        self,
        config: Mapping[str, t.ScalarValue],
    ) -> Self:
        """Fluently apply configuration values and return this instance."""
        self.configure(config)
        return self

    @override
    def with_service(
        self,
        name: str,
        service: t.RegisterableService,
    ) -> Self:
        """Register a service and return the container for fluent chaining.

        Accepts service value (PayloadValue, protocols, callables);
        values are wrapped in a ``ServiceRegistration`` with configuration from
        ``FlextSettings`` and user overrides.
        """
        _ = self.register(name, service)
        return self

    @override
    def with_factory(
        self,
        name: str,
        factory: t.FactoryCallable,
    ) -> Self:
        """Register a factory and return the container for fluent chaining."""
        _ = self.register_factory(name, factory)
        return self

    def with_resource(
        self,
        name: str,
        factory: t.ResourceCallable,
    ) -> Self:
        """Register a lifecycle-managed resource for fluent chaining."""
        _ = self.register_resource(name, factory)
        return self

    @override
    def register(
        self,
        name: str,
        service: t.RegisterableService,
    ) -> r[bool]:
        """Register a service instance for dependency resolution.

        Business Rule: The container accepts service values for registration,
        including PayloadValue, protocols (Config, Ctx, DI, Service, Log,
        Handler, Registry), and callables. This enables dependency injection of
        typed service instances and protocol implementations.

        Args:
            name: Unique key for the registration.
            service: Concrete instance that produces the service.
                Must be PayloadValue (primitives, BaseModel, callable,
                sequence, or mapping).

        Returns:
            ``FlextResult`` indicating whether the registration succeeded. A
            failed result is returned when the name is already registered or
            when construction fails.

        """
        if not name:
            return r[bool].fail("Service name must have at least 1 character")
        if not self._is_registerable_service(service):
            return r[bool].fail(
                f"Service '{name}' does not satisfy RegisterableService protocol. "
                f"Expected PayloadValue, protocol implementation, or callable.",
            )
        try:
            if hasattr(self._di_services, name):
                return r[bool].fail(f"Service '{name}' already registered")
            registration = m.Container.ServiceRegistration(
                name=name,
                service=service,
                service_type=service.__class__.__name__,
            )
            self._services[name] = registration
            provider = FlextRuntime.DependencyIntegration.register_object(
                self._di_services,
                name,
                service,
            )
            setattr(self._di_bridge, name, provider)
            setattr(self._di_container, name, provider)
            return r[bool].ok(value=True)
        except Exception as e:
            return r[bool].fail(str(e))

    @staticmethod
    def _narrow_factory_result(value: t.RegisterableService) -> t.RegisterableService:
        return value

    @override
    def register_factory(
        self,
        name: str,
        factory: t.FactoryCallable,
    ) -> r[bool]:
        """Register a factory used to build services on demand.

        Accepts factories returning RegisterableService (including protocols
        like Log, Ctx, Config, etc.) not just PayloadValue.

        Returns:
            ``FlextResult`` signaling whether the factory was stored. Failure
            occurs when the name already exists or the factory raises during
            registration.

        """
        try:
            if hasattr(self._di_services, name):
                return r[bool].fail(f"Factory '{name}' already registered")

            def normalized_factory() -> t.RegisterableService:
                raw_result = factory()
                narrowed = FlextContainer._narrow_factory_result(raw_result)
                if not self._is_registerable_service(narrowed):
                    msg = (
                        f"Factory '{name}' returned value that does not satisfy "
                        f"RegisterableService protocol. Expected PayloadValue, protocol, or callable."
                    )
                    raise ValueError(msg)
                return narrowed

            registration = m.Container.FactoryRegistration(
                name=name,
                factory=normalized_factory,
            )
            self._factories[name] = registration
            provider = FlextRuntime.DependencyIntegration.register_factory(
                self._di_services,
                name,
                normalized_factory,
                cache=self._global_config.enable_factory_caching,
            )
            setattr(self._di_bridge, name, provider)
            setattr(self._di_container, name, provider)
            return r[bool].ok(value=True)
        except Exception as e:
            return r[bool].fail(str(e))

    def register_resource(
        self,
        name: str,
        factory: t.ResourceCallable,
    ) -> r[bool]:
        """Register a dependency-injector Resource provider."""
        try:
            if hasattr(self._di_resources, name):
                return r[bool].fail(f"Resource '{name}' already registered")
            # factory is already ResourceCallable (Callable[[], object])
            registration = m.Container.ResourceRegistration(
                name=name,
                factory=factory,
            )
            self._resources[name] = registration
            provider = FlextRuntime.DependencyIntegration.register_resource(
                self._di_resources,
                name,
                factory,
            )
            setattr(self._di_bridge, name, provider)
            setattr(self._di_container, name, provider)
            return r[bool].ok(value=True)
        except Exception as e:
            return r[bool].fail(str(e))

    @override
    def get(
        self,
        name: str,
    ) -> r[t.RegisterableService]:
        """Resolve a registered service or factory by name.

        Returns the resolved service as RegisterableService. For type-safe resolution
        with runtime type validation, use get_typed() which validates against a
        specific type class.

        Args:
            name: Service identifier to resolve.

        Returns:
            r[t.RegisterableService]: Success with resolved service, or failure if not found.
                Caller should use get_typed() or type identity / MRO checks for type narrowing.

        Example:
            >>> container = FlextContainer()
            >>> container.register("logger", FlextLogger(__name__))
            >>> result = container.get("logger")
            >>> if result.is_success and isinstance(result.value, FlextLogger):
            ...     result.value.info("Resolved")

        """
        # Try service first
        if name in self._services:
            service_registration = self._services[name]
            service = service_registration.service
            try:
                m.Container.ServiceRegistration(
                    name=name,
                    service=service,
                )
            except ValidationError:
                return r[t.RegisterableService].fail(
                    f"Service '{name}' has unsupported runtime type",
                )
            return r[t.RegisterableService].ok(service)

        # Try factory
        if name in self._factories:
            try:
                factory_registration = self._factories[name]
                resolved = factory_registration.factory()
                try:
                    m.Container.ServiceRegistration(
                        name=name,
                        service=resolved,
                    )
                except ValidationError:
                    return r[t.RegisterableService].fail(
                        f"Factory '{name}' returned unsupported runtime type",
                    )
                return r[t.RegisterableService].ok(resolved)
            except Exception as e:
                return r[t.RegisterableService].fail(str(e))

        # Try resource
        if name in self._resources:
            try:
                resource_registration = self._resources[name]
                resolved = resource_registration.factory()
                if not self._is_registerable_service(resolved):
                    return r[t.RegisterableService].fail(
                        f"Resource '{name}' returned unsupported runtime type",
                    )
                return r[t.RegisterableService].ok(resolved)
            except Exception as e:
                return r[t.RegisterableService].fail(str(e))

        return r[t.RegisterableService].fail(f"Service '{name}' not found")

    @staticmethod
    def _is_instance_of[T](value: object, type_cls: type[T]) -> TypeGuard[T]:
        """Type guard to narrow object to specific type T.

        Uses isinstance for type narrowing with MRO support.
        """
        return isinstance(value, type_cls) or type_cls in type(value).__mro__

    @staticmethod
    def _is_registerable_service(value: object) -> TypeGuard[t.RegisterableService]:
        # Use isinstance for proper type narrowing
        if isinstance(value, (str, int, float, bool, type(None))):
            return True
        if isinstance(value, (BaseModel, Path)):
            return True
        if callable(value):
            return True
        if isinstance(value, Mapping):
            return True
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray),
        ):
            return True
        if FlextContainer._is_context_protocol(value):
            return True
        if hasattr(value, "__dict__"):
            return True
        return bool(hasattr(value, "bind") and hasattr(value, "info"))

    @staticmethod
    def _is_context_protocol(value: object) -> TypeGuard[p.Context]:
        return bool(
            hasattr(value, "clone") and hasattr(value, "set") and hasattr(value, "get"),
        )

    @override
    def get_typed[T](self, name: str, type_cls: type[T]) -> r[T]:
        """Resolve a service by name and validate its runtime type.

        Provides runtime type validation on top of DI resolution. The resolved
        service is validated against type_cls using isinstance check.

        Args:
            name: Service identifier to resolve.
            type_cls: Expected type class for validation.

        Returns:
            r[T]: Success with type-validated service, or failure if not found
                or type mismatch.

        """
        # Try service first with runtime type validation
        if name in self._services:
            service = self._services[name].service
            if not self._is_instance_of(service, type_cls):
                return r[T].fail(
                    f"Service '{name}' is not of type {(type_cls.__name__ if hasattr(type_cls, '__name__') else 'Unknown')}",
                )
            # TypeGuard narrows service to type T
            return r[T].ok(service)

        # Try factory with runtime type validation
        if name in self._factories:
            try:
                resolved = self._factories[name].factory()
                if not self._is_instance_of(resolved, type_cls):
                    return r[T].fail(f"Factory '{name}' returned wrong type")
                # TypeGuard narrows resolved to type T
                return r[T].ok(resolved)
            except Exception as e:
                return r[T].fail(str(e))

        # Try resource with runtime type validation
        if name in self._resources:
            try:
                resolved = self._resources[name].factory()
                if not self._is_instance_of(resolved, type_cls):
                    return r[T].fail(f"Resource '{name}' returned wrong type")
                # TypeGuard narrows resolved to type T
                return r[T].ok(resolved)
            except Exception as e:
                return r[T].fail(str(e))

        return r[T].fail(f"Service '{name}' not found")

    def create_module_logger(self, module_name: str | None = None) -> FlextLogger:
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

    @override
    def has_service(self, name: str) -> bool:
        """Return whether a service or factory is registered for ``name``."""
        return (
            name in self._services or name in self._factories or name in self._resources
        )

    @override
    def list_services(self) -> Sequence[str]:
        """List the names of registered services and factories."""
        return (
            list(self._services.keys())
            + list(self._factories.keys())
            + list(self._resources.keys())
        )

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
    def clear_all(self) -> None:
        """Clear all service and factory registrations.

        Business Rule: Clears all registrations but preserves singleton instance.
        Used for test cleanup and container reset scenarios. Does not reset
        singleton pattern - use reset_singleton_for_testing() for that.

        """
        for name in self.list_services():
            if hasattr(self._di_services, name):
                delattr(self._di_services, name)
            if hasattr(self._di_resources, name):
                delattr(self._di_resources, name)
        self._services.clear()
        self._factories.clear()
        self._resources.clear()

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
        config: p.Config,
        context: p.Context,
        services: Mapping[str, m.Container.ServiceRegistration],
        factories: Mapping[str, m.Container.FactoryRegistration],
        resources: Mapping[str, m.Container.ResourceRegistration],
        user_overrides: m.ConfigMap,
        container_config: m.Container.ContainerConfig,
    ) -> FlextContainer:
        """Create a scoped container instance bypassing singleton pattern.

        This is an internal factory method to safely create non-singleton containers.
        Uses direct attribute assignment (no frozen=True, compatible with FlextRuntime pattern).
        """
        # Create raw instance without __new__ singleton logic
        # Use type-safe factory helper from FlextRuntime
        instance = FlextRuntime.create_instance(cls)
        # Initialize public attributes directly
        instance.containers = FlextRuntime.dependency_containers()
        instance.providers = FlextRuntime.dependency_providers()
        # Initialize DI components using public method
        instance.initialize_di_components()
        # Initialize registrations using public method
        instance.initialize_registrations(
            services=services,
            factories=factories,
            resources=resources,
            global_config=container_config,
            user_overrides=user_overrides,
            config=config,
            context=context,
        )
        # Call public methods (no bypass needed)
        instance.sync_config_to_di()
        # Register all providers on the bridge for @inject/Provide usage
        instance.register_existing_providers()
        # Auto-register core services for easy DI access
        instance.register_core_services()
        return instance

    @override
    def scoped(
        self,
        *,
        config: p.Config | None = None,
        context: p.Context | None = None,
        subproject: str | None = None,
        services: Mapping[str, t.RegisterableService] | None = None,
        factories: Mapping[str, t.FactoryCallable] | None = None,
        resources: Mapping[str, t.ResourceCallable] | None = None,
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
            A container implementing ``p.DI`` with
            isolated state that inherits the global configuration by default.

        """
        # Clone base config — config objects are always BaseModel (FlextSettings)
        config_input = config  # Keep original for subproject check
        if config is not None and isinstance(config, BaseModel):
            base_config: p.Config = cast("p.Config", config.model_copy(deep=True))
        elif isinstance(self.config, BaseModel):
            base_config = cast("p.Config", self.config.model_copy(deep=True))
        else:
            base_config = self._get_default_config()

        # Apply subproject suffix to app_name only when config is None
        if subproject and config_input is None and isinstance(base_config, BaseModel):
            base_config = cast(
                "p.Config",
                base_config.model_copy(
                    update={"app_name": f"{base_config.app_name}.{subproject}"},
                ),
            )

        scoped_context: p.Context
        if context is None:
            ctx_instance = self.context
            clone_method = (
                ctx_instance.clone
                if hasattr(ctx_instance, "clone")
                else None
            )
            if callable(clone_method):
                candidate_context = clone_method()
                if self._is_context_protocol(candidate_context):
                    scoped_context = candidate_context
                else:
                    scoped_context = FlextContext()
            else:
                scoped_context = FlextContext()
        elif self._is_context_protocol(context):
            scoped_context = context
        else:
            scoped_context = self.context.clone()

        if subproject:
            # Ctx.set returns None per protocol definition
            # But FlextContext.set returns r[bool] - call directly
            # Protocol allows None return, implementation can return FlextResult
            _ = scoped_context.set("subproject", subproject)

        # Clone services from parent container
        # Use deep=False to avoid issues with non-serializable objects (e.g., ContextVar in FlextContext)
        # The service instances themselves are shared, but the registration metadata is cloned
        cloned_services: MutableMapping[str, m.Container.ServiceRegistration] = {
            name: registration.model_copy(deep=False)
            for name, registration in self._services.items()
        }
        cloned_factories: MutableMapping[str, m.Container.FactoryRegistration] = {
            name: registration.model_copy(deep=False)
            for name, registration in self._factories.items()
        }
        cloned_resources: MutableMapping[str, m.Container.ResourceRegistration] = {
            name: registration.model_copy(deep=False)
            for name, registration in self._resources.items()
        }

        for name, service in (services or {}).items():
            # Type narrowing: service is compatible with ServiceRegistration.service type
            # ServiceRegistration.service accepts: t.ConfigMapValue | BaseModel | p.VariadicCallable[t.ConfigMapValue] | t.ConfigMapValue
            # The service parameter matches this union type
            cloned_services[name] = m.Container.ServiceRegistration(
                name=name,
                service=service,
                service_type=service.__class__.__name__,
            )
        for name, factory in (factories or {}).items():
            cloned_factories[name] = m.Container.FactoryRegistration(
                name=name,
                factory=factory,
            )
        for name, resource_factory in (resources or {}).items():
            # resources param is Mapping[str, object] - validate callable
            if callable(resource_factory):
                cloned_resources[name] = m.Container.ResourceRegistration(
                    name=name,
                    factory=resource_factory,
                )

        # Use factory method to create scoped container (avoids mypy __init__ error)
        # Structural typing - FlextContainer implements p.DI
        # cloned_services, cloned_factories, cloned_resources already have correct types
        # _user_overrides is always ConfigurationDict after initialize_registrations
        # (initialized as user_overrides or {}), so it's never None after __init__
        user_overrides_copy = m.ConfigMap(root=dict(self._user_overrides.items()))
        return FlextContainer._create_scoped_instance(
            config=base_config,
            context=scoped_context,
            services=cloned_services,
            factories=cloned_factories,
            resources=cloned_resources,
            user_overrides=user_overrides_copy,
            container_config=self._global_config.model_copy(deep=True),
        )

    class Builder:
        """Builder utility for creating FlextContainer instances.

        Provides convenient access to FlextContainer.create() method for
        creating container instances with optional factory auto-registration.
        """

        @classmethod
        def create(
            cls,
            *,
            auto_register_factories: bool = False,
        ) -> FlextContainer:
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
                auto_register_factories=auto_register_factories,
            )


__all__ = ["FlextContainer"]
