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
from types import ModuleType
from typing import Self, cast

from pydantic import BaseModel

from flext_core._decorators import FactoryDecoratorsDiscovery
from flext_core.config import FlextConfig
from flext_core.constants import c
from flext_core.loggings import FlextLogger
from flext_core.models import m
from flext_core.protocols import p
from flext_core.result import r
from flext_core.runtime import FlextRuntime
from flext_core.typings import T, t
from flext_core.utilities import u


class FlextContainer(FlextRuntime, p.Container.DI):
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
    # Instance attributes (initialized in __init__)
    _context: p.Context.Ctx | None = None

    def __new__(
        cls,
        *,
        _config: p.Configuration.Config | None = None,
        _context: p.Context.Ctx | None = None,
        _services: dict[str, m.Container.ServiceRegistration] | None = None,
        _factories: dict[str, m.Container.FactoryRegistration] | None = None,
        _resources: dict[str, m.Container.ResourceRegistration] | None = None,
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
        _services: dict[str, m.Container.ServiceRegistration] | None = None,
        _factories: dict[str, m.Container.FactoryRegistration] | None = None,
        _resources: dict[str, m.Container.ResourceRegistration] | None = None,
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
                caller_globals = frame.f_back.f_globals
                # Get module name from globals
                module_name = caller_globals.get("__name__", "__main__")
                # Get module object from globals (usually available as __import__ or direct reference)

                caller_module = sys.modules.get(module_name)
                if caller_module:
                    # Scan module for factory-decorated functions
                    factories = FactoryDecoratorsDiscovery.scan_module(caller_module)
                    for factory_name, factory_config in factories:
                        # Get actual factory function from module
                        factory_func = getattr(caller_module, factory_name, None)
                        if factory_func and callable(factory_func):
                            # Register using the name from decorator config
                            _ = instance.register_factory(
                                factory_config.name,
                                factory_func,
                            )

        return instance

    @property
    def provide(self) -> Callable[[str], object]:
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
        return self._di_bridge.Provide

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
        # Type narrowing: self._config is p.Configuration.Config
        return self._config

    @property
    def context(self) -> p.Context.Ctx:
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
        bridge, service_module, resource_module = (
            FlextRuntime.DependencyIntegration.create_layered_bridge()
        )
        di_container = self.containers.DynamicContainer()
        # Internal initialization - direct assignment to private attributes
        # These are set during object construction, not accessed from outside
        self._di_bridge = bridge
        self._di_services = service_module
        self._di_resources = resource_module
        self._di_container = di_container
        self._config_provider = bridge.config
        base_config_provider = self.providers.Configuration()
        user_config_provider = self.providers.Configuration()
        self._base_config_provider = base_config_provider
        self._user_config_provider = user_config_provider
        # Configure providers - override() returns OverridingContext for chaining
        # We call it for side effects (configuring the provider), not for the return value
        # override() may return None or OverridingContext - we don't need the return value
        self._config_provider.override(base_config_provider)
        self._config_provider.override(user_config_provider)
        di_container.config = self._config_provider

    def initialize_registrations(
        self,
        *,
        services: dict[str, m.Container.ServiceRegistration] | None = None,
        factories: dict[str, m.Container.FactoryRegistration] | None = None,
        resources: dict[str, m.Container.ResourceRegistration] | None = None,
        global_config: m.Container.ContainerConfig | None = None,
        user_overrides: t.Types.ConfigurationDict | None = None,
        config: p.Configuration.Config | None = None,
        context: p.Context.Ctx | None = None,
    ) -> None:
        """Initialize service registrations and configuration.

        Internal method to set up registrations and config.
        Can be called from __init__ or _create_scoped_instance.
        Sets private attributes directly - this is internal initialization.
        """
        self._services = services or {}
        self._factories = factories or {}
        self._resources = resources or {}
        self._global_config = global_config or self._create_container_config()
        self._user_overrides = user_overrides or {}
        # Type narrowing: config can be None, but property handles None case
        config_instance: p.Configuration.Config = (
            config if config is not None else FlextConfig.get_global_instance()
        )
        self._config = config_instance
        # Type narrowing: context can be None, but property will raise error if accessed
        # Direct assignment is safe - _context is an instance attribute
        # _context is declared as p.Context.Ctx | None = None (instance attribute)
        # If context is None, property will raise RuntimeError on access (no lazy creation)
        self._context = context

    def _get_default_config(self) -> p.Configuration.Config:
        """Get default configuration instance."""
        return FlextConfig.get_global_instance()

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
        """Synchronize FlextConfig to DI providers.Configuration.

        Dependency Injector's layered ``providers.Configuration`` instances are
        used to avoid manual merges while still honoring validated defaults from
        ``FlextConfig`` and runtime overrides. Base config and user overrides are
        applied as separate providers to keep precedence explicit.

        Also registers namespace configs as factories for easy DI access:
        - "config.ldif" → FlextLdifConfig instance
        - "config.ldap" → FlextLdapConfig instance
        - etc.
        """
        # L0: Base defaults from FlextConfig
        config_dict = self._global_config.model_dump()
        FlextRuntime.DependencyIntegration.bind_configuration(
            self._di_container,
            config_dict,
        )

        # Apply user overrides
        # Type narrowing: _user_overrides is always ConfigurationDict after initialize_registrations
        # (initialized as user_overrides or {}), so it's never None after __init__
        # Use type narrowing with explicit cast to help pyright
        user_overrides_dict: t.Types.ConfigurationDict = cast(
            "t.Types.ConfigurationDict",
            self._user_overrides if self._user_overrides is not None else {},
        )
        self._user_config_provider.from_dict(dict(user_overrides_dict))

        # Register namespace configs as factories
        # Access namespace registry via public method (get_namespace_config)
        # We need to iterate over registered namespaces - use getattr to access registry
        # Note: _namespace_registry is ClassVar, accessed via class, not instance
        namespace_registry = getattr(
            type(self._global_config),
            "_namespace_registry",
            {},
        )
        for namespace in namespace_registry:
            factory_name = f"config.{namespace}"

            # Get config class for this namespace
            config_class = self._global_config.get_namespace_config(namespace)
            if config_class is None:
                continue

            def _create_namespace_config(
                ns: str = namespace,
                config_cls: type[BaseModel] = config_class,
            ) -> BaseModel:
                """Factory for creating namespace config instance."""
                return self._global_config.get_namespace(ns, config_cls)

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
        - "config" → FlextConfig singleton
        - "logger" → FlextLogger factory (creates module logger)
        - "context" → FlextContext singleton
        - "container" → Self-reference for nested resolution

        Business Rule: Auto-registers FlextConfig, FlextLogger, and FlextContext
        with standard names ("config", "logger", "context") to enable easy
        dependency injection in downstream projects. Services are registered only
        if not already registered to avoid conflicts.

        This method ensures that core services are accessible via:
        - container.get("config") -> FlextConfig
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
        if not self.has_service("config"):
            _ = self.register("config", self._config)

        # Register logger factory if not already registered
        if not self.has_service("logger"):

            def _create_logger() -> FlextLogger:
                """Factory for creating module logger."""
                return FlextLogger.create_module_logger("flext_core")

            _ = self.register_factory("logger", _create_logger)

        # Register context if not already registered
        # Only register if context is initialized (may not be initialized during container creation)
        if not self.has_service("context") and self._context is not None:
            _ = self.register("context", self._context)

        # Register container self-reference if not already registered
        if not self.has_service("container"):
            _ = self.register("container", self)

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
        # Type narrowing: filter values to FlexibleValue compatible types
        process_result = u.Collection.process(
            config,
            lambda _k, v: (
                v
                if isinstance(v, (str, int, float, bool, type(None), dict, list))
                else str(v)
            ),
            on_error="collect",
        )
        if process_result.is_success and isinstance(process_result.value, dict):
            # Type narrowing: process_result.value is dict after isinstance check
            # process() returns dict[str, R] where R is FlexibleValue
            # and FlexibleValue is compatible with GeneralValueType (which ConfigurationDict uses)
            processed_dict: t.Types.ConfigurationDict = cast(
                "t.Types.ConfigurationDict",
                process_result.value,
            )
            # Simple merge: override strategy - new values override existing ones
            # Type narrowing: _user_overrides is always ConfigurationDict after initialize_registrations
            # (initialized as user_overrides or {}), so it's never None after __init__
            # Use cast to help pyright understand the type
            user_overrides_dict: t.Types.ConfigurationDict = cast(
                "t.Types.ConfigurationDict",
                self._user_overrides if self._user_overrides is not None else {},
            )
            merged: t.Types.ConfigurationDict = dict(user_overrides_dict)
            merged.update(processed_dict)
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

    def with_resource(
        self,
        name: str,
        factory: Callable[[], t.GeneralValueType],
    ) -> Self:
        """Register a lifecycle-managed resource for fluent chaining."""
        _ = self.register_resource(name, factory)
        return self

    def register[T](self, name: str, service: T) -> r[bool]:  # pyright: ignore[reportInvalidTypeVarUse] - T used for type inference at call sites
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
        if not name:
            return r[bool].fail("Service name must have at least 1 character")
        try:
            if hasattr(self._di_services, name):
                return r[bool].fail(f"Service '{name}' already registered")
            registration = m.Container.ServiceRegistration(
                name=name,
                service=service,
                service_type=type(service).__name__,
            )
            self._services[name] = registration
            provider = FlextRuntime.DependencyIntegration.register_object(
                self._di_services,
                name,
                service,
            )
            setattr(self._di_bridge, name, provider)
            setattr(self._di_container, name, provider)
            return r[bool].ok(True)
        except Exception as e:
            return r[bool].fail(str(e))

    def register_factory(
        self,
        name: str,
        factory: Callable[[], T],
    ) -> r[bool]:
        """Register a factory used to build services on demand.

        Returns:
            ``FlextResult`` signaling whether the factory was stored. Failure
            occurs when the name already exists or the factory raises during
            registration.

        """
        if not callable(factory):
            return r[bool].fail("Factory must be callable")
        try:
            if hasattr(self._di_services, name):
                return r[bool].fail(f"Factory '{name}' already registered")
            # Factory returns GeneralValueType which is compatible with FactoryRegistration requirements
            # Type narrowing: factory is already compatible with the expected type
            # GeneralValueType includes ScalarValue, Sequence, Mapping
            # Cast factory to expected type for FactoryRegistration
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
            provider = FlextRuntime.DependencyIntegration.register_factory(
                self._di_services,
                name,
                factory,
                cache=self._global_config.enable_factory_caching,
            )
            setattr(self._di_bridge, name, provider)
            setattr(self._di_container, name, provider)
            return r[bool].ok(True)
        except Exception as e:
            return r[bool].fail(str(e))

    def register_resource(
        self,
        name: str,
        factory: Callable[[], T],
    ) -> r[bool]:
        """Register a dependency-injector Resource provider."""
        if not callable(factory):
            return r[bool].fail("Resource factory must be callable")
        try:
            if hasattr(self._di_resources, name):
                return r[bool].fail(f"Resource '{name}' already registered")
            # Cast factory to expected type for ResourceRegistration
            factory_typed: Callable[[], t.GeneralValueType] = cast(
                "Callable[[], t.GeneralValueType]",
                factory,
            )
            registration = m.Container.ResourceRegistration(
                name=name,
                factory=factory_typed,
            )
            self._resources[name] = registration
            provider = FlextRuntime.DependencyIntegration.register_resource(
                self._di_resources,
                name,
                factory,
            )
            setattr(self._di_bridge, name, provider)
            setattr(self._di_container, name, provider)
            return r[bool].ok(True)
        except Exception as e:
            return r[bool].fail(str(e))

    def get[T](self, name: str) -> r[T]:  # pyright: ignore[reportInvalidTypeVarUse] - T used for return type inference
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

        Example:
            >>> from flext_core import FlextContainer, FlextLogger
            >>>
            >>> container = FlextContainer.create()
            >>> logger = FlextLogger(__name__)
            >>> container.register("logger", logger, singleton=True)
            >>>
            >>> logger_result = container.get[FlextLogger]("logger")
            >>> if logger_result.is_success:
            ...     logger_instance = logger_result.value
            ...     logger_instance.info("Service resolved successfully")

        """
        # Try service first
        if name in self._services:
            service_registration = self._services[name]
            service = service_registration.service
            # Runtime type safety guaranteed by container registration
            return r[T].ok(cast("T", service))

        # Try factory
        if name in self._factories:
            try:
                factory_registration = self._factories[name]
                instance = factory_registration.factory()
                # Runtime type safety guaranteed by container registration
                return r[T].ok(cast("T", instance))
            except Exception as e:
                return r[T].fail(str(e))

        # Try resource
        if name in self._resources:
            try:
                resource_registration = self._resources[name]
                resource_instance_raw = resource_registration.factory()
                # Runtime type safety guaranteed by container registration
                # resource_instance_raw is GeneralValueType, cast to T for type safety
                resource_instance: T = cast("T", resource_instance_raw)
                return r[T].ok(resource_instance)
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
        return (
            name in self._services or name in self._factories or name in self._resources
        )

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
            return r[bool].ok(True)
        return r[bool].fail(f"Service '{name}' not found")

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
        config: p.Configuration.Config,
        context: p.Context.Ctx,
        services: dict[str, m.Container.ServiceRegistration],
        factories: dict[str, m.Container.FactoryRegistration],
        resources: dict[str, m.Container.ResourceRegistration],
        user_overrides: t.Types.ConfigurationDict,
        container_config: m.Container.ContainerConfig,
    ) -> p.Container.DI:
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
        resources: Mapping[str, Callable[[], t.GeneralValueType]] | None = None,
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
            resources: Additional resource registrations merged into the scoped
                instance.

        Returns:
            A container implementing ``p.Container.DI`` with
            isolated state that inherits the global configuration by default.

        """
        # Clone base config if not provided
        base_config = (
            config if config is not None else self.config.model_copy(deep=True)
        )
        # Apply subproject suffix to app_name only when config is None
        # If config was explicitly provided, respect it (don't modify)
        # This allows explicit config to take precedence over subproject naming
        if subproject and config is None:
            # Only apply subproject when using global config (config is None)
            # Explicit config parameter means user wants that exact config
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

        # Clone services from parent container
        # Use deep=False to avoid issues with non-serializable objects (e.g., ContextVar in FlextContext)
        # The service instances themselves are shared, but the registration metadata is cloned
        cloned_services: dict[str, m.Container.ServiceRegistration] = {
            name: registration.model_copy(deep=False)
            for name, registration in self._services.items()
        }
        cloned_factories: t.Types.FactoryRegistrationDict = {
            name: registration.model_copy(deep=False)
            for name, registration in self._factories.items()
        }
        cloned_resources: dict[str, m.Container.ResourceRegistration] = {
            name: registration.model_copy(deep=False)
            for name, registration in self._resources.items()
        }

        for name, service in (services or {}).items():
            # Type narrowing: service is compatible with ServiceRegistration.service type
            # ServiceRegistration.service accepts: t.GeneralValueType | BaseModel | p.Utility.Callable[t.GeneralValueType] | object
            # The service parameter matches this union type
            cloned_services[name] = m.Container.ServiceRegistration(
                name=name,
                service=cast(
                    "t.GeneralValueType | BaseModel | p.Utility.Callable[t.GeneralValueType] | object",
                    service,
                ),
                service_type=type(service).__name__,
            )
        for name, factory in (factories or {}).items():
            cloned_factories[name] = m.Container.FactoryRegistration(
                name=name,
                factory=factory,
            )
        for name, resource_factory in (resources or {}).items():
            cloned_resources[name] = m.Container.ResourceRegistration(
                name=name,
                factory=resource_factory,
            )

        # Use factory method to create scoped container (avoids mypy __init__ error)
        # Structural typing - FlextContainer implements p.Container.DI
        # base_config already implements p.Configuration.Config protocol
        # cloned_services and cloned_factories contain ServiceRegistration/FactoryRegistration instances
        # Type annotation: cloned_services is dict[str, ServiceRegistration]
        # _create_scoped_instance expects dict[str, ServiceRegistration]
        # No cast needed - cloned_services is already the correct type
        services_typed: dict[str, m.Container.ServiceRegistration] = cloned_services
        # Type annotation: cloned_factories is FactoryRegistrationDict (dict[str, object])
        # but _create_scoped_instance expects dict[str, FactoryRegistration]
        # Cast is safe because cloned_factories contains FactoryRegistration instances
        factories_typed: dict[str, m.Container.FactoryRegistration] = cast(
            "dict[str, m.Container.FactoryRegistration]",
            cloned_factories,
        )
        # Type narrowing: _user_overrides is always ConfigurationDict after initialize_registrations
        # (initialized as user_overrides or {}), so it's never None after __init__
        # Use cast to help pyright understand the type
        user_overrides_dict: t.Types.ConfigurationDict = cast(
            "t.Types.ConfigurationDict",
            self._user_overrides if self._user_overrides is not None else {},
        )
        user_overrides_copy: t.Types.ConfigurationDict = user_overrides_dict.copy()
        return FlextContainer._create_scoped_instance(
            config=base_config,
            context=scoped_context,
            services=services_typed,
            factories=factories_typed,
            resources=cloned_resources,
            user_overrides=user_overrides_copy,
            container_config=self._global_config.model_copy(deep=True),
        )


__all__ = ["FlextContainer"]
