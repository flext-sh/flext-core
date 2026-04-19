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
from types import FrameType, ModuleType
from typing import Self, TypeIs, overload, override

from dependency_injector import containers as di_containers, providers as di_providers

from flext_core import FlextContext, FlextRuntime, FlextSettings, c, e, m, p, r, t, u


class FlextContainer(p.ContainerLifecycle):
    """Singleton container that exposes DI registration and resolution helpers.

    Services and factories remain local to the container, keeping dispatcher and
    domain code free from infrastructure imports. All operations surface
    ``r`` (r) so failures are explicit. Thread-safe initialization
    guarantees one global instance for runtime usage while allowing scoped
    containers in tests. The class satisfies ``p.ContainerLifecycle`` through
    structural typing only.
    """

    _global_instance: Self | None = None
    _global_lock: threading.RLock = threading.RLock()
    _context: p.Context
    _config: p.Settings
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
    _services: MutableMapping[str, m.ServiceRegistration]
    _factories: MutableMapping[str, m.FactoryRegistration]
    _resources: MutableMapping[str, m.ResourceRegistration]
    _internal_registrations: set[str]
    _global_config: m.ContainerConfig

    @staticmethod
    def _require_settings(
        settings: p.Settings | None,
        *,
        source: str,
    ) -> p.Settings:
        if isinstance(settings, p.Settings):
            return settings
        error_msg = f"{source} must implement p.Settings"
        raise TypeError(error_msg)

    @staticmethod
    def _require_context(
        context: p.Context | None,
        *,
        source: str,
    ) -> p.Context:
        if isinstance(context, p.Context):
            return context
        error_msg = f"{source} must implement p.Context"
        raise TypeError(error_msg)

    def __new__(
        cls,
        *,
        registration: m.ServiceRegistrationSpec | None = None,
    ) -> Self:
        """Create or return the global singleton instance.

        Optional keyword arguments support deterministic construction in tests
        while preserving singleton semantics for runtime callers. Double-checked
        locking protects against duplicate initialization under concurrency.
        """
        _ = registration
        if cls._global_instance is None:
            with cls._global_lock:
                if cls._global_instance is None:
                    instance = super().__new__(cls)
                    cls._global_instance = instance
        return cls._global_instance

    def __init__(
        self,
        *,
        registration: m.ServiceRegistrationSpec | None = None,
    ) -> None:
        """Wire the Dependency Injector container and supporting registries.

        The initializer is idempotent: repeated construction returns early to
        keep the singleton stable. Parameters allow deterministic construction
        during testing or for scoped containers created via :meth:`scoped`.
        """
        super().__init__()
        init_registration = registration or m.ServiceRegistrationSpec()
        if hasattr(self, "_di_container"):
            self._apply_explicit_bootstrap(init_registration)
            self.register_core_services()
            return
        self.containers = u.dependency_containers()
        self.providers = u.dependency_providers()
        self.initialize_di_components()
        self.initialize_registrations(
            services=init_registration.services,
            factories=init_registration.factories,
            resources=init_registration.resources,
            global_config=init_registration.container_config,
            user_overrides=init_registration.user_overrides,
            settings=init_registration.settings,
            context=init_registration.context,
        )
        self.sync_config_to_di()
        self.register_existing_providers()
        self.register_core_services()

    def _bind_provider(
        self,
        name: str,
        provider: p.ProviderLike[t.RegisterableService],
    ) -> None:
        """Register a resolved DI provider on both bridge and container namespaces."""
        setattr(self._di_bridge, name, provider)
        setattr(self._di_container, name, provider)

    @staticmethod
    def _core_names() -> set[str]:
        """Return reserved names for auto-registered core infrastructure."""
        return {
            str(c.Directory.CONFIG),
            str(c.ServiceName.LOGGER),
            str(c.ServiceName.COMMAND_BUS),
            str(c.FIELD_CONTEXT),
        }

    def _has_internal_registration(self, name: str) -> bool:
        """Return whether a name exists in explicit or internal DI state."""
        return (
            name in self._services
            or name in self._factories
            or name in self._resources
            or hasattr(self._di_services, name)
            or hasattr(self._di_resources, name)
        )

    def _clear_provider_registration(self, name: str) -> None:
        """Remove an existing provider binding from DI namespaces if present."""
        if hasattr(self._di_services, name):
            delattr(self._di_services, name)
        if hasattr(self._di_resources, name):
            delattr(self._di_resources, name)

    def _update_registered_object_service(
        self,
        name: str,
        service: t.RegisterableService,
    ) -> None:
        """Replace or insert an object-backed service across local and DI state."""
        self._services[name] = m.ServiceRegistration(
            name=name,
            service=service,
            service_type=service.__class__.__name__,
        )
        for di_ns in (self._di_services, self._di_resources):
            if hasattr(di_ns, name):
                delattr(di_ns, name)
        self._bind_provider(
            name,
            u.DependencyIntegration.register_object(self._di_services, name, service),
        )

    def _apply_explicit_bootstrap(
        self,
        registration: m.ServiceRegistrationSpec,
    ) -> None:
        """Apply explicit bootstrap overrides to an existing singleton instance."""
        if registration.settings is not None:
            settings = self._require_settings(
                registration.settings,
                source="bootstrap settings",
            )
            self._config = settings
            if u.registerable_service(settings):
                self._update_registered_object_service(c.Directory.CONFIG, settings)
                self.sync_config_to_di()
        if registration.context is not None:
            context = self._require_context(
                registration.context,
                source="bootstrap context",
            )
            self._context = context
            if u.registerable_service(context):
                self._update_registered_object_service(c.FIELD_CONTEXT, context)

    @property
    @override
    def settings(self) -> p.Settings:
        """Return configuration bound to this container.

        The configuration is always initialized to a valid FlextSettings instance
        during container initialization. It may be provided during container
        initialization via the registration specification in `__init__` or
        `shared()`, or the default FlextSettings.fetch_global() will be used.

        Valid-by-construction design: This property never raises an error because
        _config is guaranteed to be non-None after initialization.
        """
        return self._config

    @property
    @override
    def context(self) -> p.Context:
        """Return the execution context bound to this container.

        The context is always initialized to a valid FlextContext instance
        during container initialization. It may be provided during container
        initialization via the registration specification in `__init__` or
        `shared()`, or a default context will be created.

        Valid-by-construction design: This property never raises an error because
        _context is guaranteed to be non-None after initialization.

        Example:
            >>> container = FlextContainer.shared(context=my_context)
            >>> ctx = container.context  # Returns the provided context
            >>> # Or if no context provided:
            >>> container2 = FlextContainer.shared()
            >>> ctx2 = container2.context  # Returns default FlextContext()

        Returns:
            The execution context (either provided or default).

        """
        return self._context

    @property
    @override
    def provide(self) -> Callable[[str], t.RegisterableService]:
        """Return the dependency-injector Provide helper scoped to the bridge.

        ``Provide`` is used alongside the ``@inject`` decorator to declare
        dependencies without importing ``dependency-injector`` in higher layers.
        It resolves registered providers (services, factories, resources,
        configuration) by name and injects the resulting value into the
        decorated callable. Example:

        .. code-block:: python

           from flext_core import FlextContainer, inject

           container = FlextContainer.shared()
           _ = container.factory(
               "token_factory",
               lambda: {"token": "abc123"},
           )


           @inject
           def consume(token=container.provide["token_factory"]):
               return token["token"]


           assert consume() == "abc123"
        """
        provide_helper = self._di_bridge.provide
        if not callable(provide_helper):
            msg = c.ERR_CONTAINER_PROVIDE_HELPER_NOT_INITIALIZED
            raise TypeError(msg)
        provide_fn: Callable[[str], t.RegisterableService] = provide_helper

        def provide_callable(name: str) -> t.RegisterableService:
            provided = provide_fn(name)
            if not u.registerable_service(provided):
                raise TypeError(c.ERR_CONTAINER_PROVIDE_HELPER_UNSUPPORTED_TYPE)
            return provided

        return provide_callable

    @classmethod
    def _create_scoped_instance(
        cls,
        *,
        registration: m.ServiceRegistrationSpec,
    ) -> Self:
        """Create a scoped container instance bypassing singleton pattern.

        This is an internal factory method to safely create non-singleton containers.
        Uses direct attribute assignment (no frozen=True, compatible with u pattern).
        """
        scoped_registration = registration
        instance = u.create_instance(cls)
        instance.containers = u.dependency_containers()
        instance.providers = u.dependency_providers()
        instance.initialize_di_components()
        instance.initialize_registrations(
            services=scoped_registration.services,
            factories=scoped_registration.factories,
            resources=scoped_registration.resources,
            global_config=scoped_registration.container_config,
            user_overrides=scoped_registration.user_overrides,
            settings=scoped_registration.settings,
            context=scoped_registration.context,
        )
        instance.sync_config_to_di()
        instance.register_existing_providers()
        instance.register_core_services()
        return instance

    @classmethod
    def shared(
        cls,
        *,
        settings: p.Settings | None = None,
        context: p.Context | None = None,
        auto_register_factories: bool = False,
    ) -> Self:
        """Return the canonical shared container instance.

        The shared container is the single command-style entrypoint used by
        decorators, mixins, runtime bootstrap, and application services.
        Optional settings/context bootstrap is applied idempotently, and
        factory auto-registration can scan the caller module for ``@d.factory``
        definitions when requested.
        """
        instance = cls()
        if settings is not None or context is not None:
            instance._apply_explicit_bootstrap(
                m.ServiceRegistrationSpec(settings=settings, context=context)
            )
        if auto_register_factories:
            frame = inspect.currentframe()
            caller_module = FlextContainer._resolve_caller_module(frame)
            if caller_module is not None:
                FlextContainer._auto_register_module_factories(instance, caller_module)
        return instance

    @staticmethod
    def _resolve_caller_module(
        create_frame: FrameType | None,
    ) -> ModuleType | None:
        """Resolve the module that called create() via frame introspection."""
        if not create_frame or not create_frame.f_back:
            return None
        caller_globals = create_frame.f_back.f_globals
        module_name_raw = caller_globals.get("__name__", "__main__")
        module_name = str(module_name_raw) if module_name_raw else "__main__"
        return sys.modules.get(module_name)

    @staticmethod
    def _auto_register_module_factories(
        instance: p.Container,
        caller_module: ModuleType,
    ) -> None:
        """Scan module for @d.factory() functions and register them."""
        factories = u.scan_module(caller_module)
        module_symbols = vars(caller_module)
        for factory_name, factory_config in factories:
            factory_func_raw = module_symbols.get(factory_name)
            if factory_func_raw is None or not u.factory(factory_func_raw):
                continue
            factory_func_ref: t.FactoryCallable = factory_func_raw
            wrapper = FlextContainer._make_factory_wrapper(
                factory_func_ref,
                factory_name,
                factory_config,
            )
            _ = instance.factory(factory_config.name, wrapper)

    @staticmethod
    def _make_factory_wrapper(
        func_ref: t.FactoryCallable,
        name: str,
        settings: m.FactoryDecoratorConfig,
    ) -> t.FactoryCallable:
        """Build a closure that invokes factory with validation on return type."""

        def factory_wrapper(
            *,
            _factory_func_ref: t.FactoryCallable = func_ref,
            _factory_name: str = name,
            _factory_config: m.FactoryDecoratorConfig = settings,
        ) -> t.RegisterableService:
            _ = _factory_config
            raw_result: t.RegisterableService = _factory_func_ref()
            if not u.registerable_service(raw_result):
                msg = f"Factory '{_factory_name}' returned unsupported type: {raw_result.__class__.__name__}"
                raise TypeError(msg)
            return raw_result

        return factory_wrapper

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
            max_services=c.DEFAULT_SIZE,
            max_factories=c.DEFAULT_MAX_FACTORIES,
        )

    @override
    def clear(self) -> None:
        """Clear all service and factory registrations.

        Business Rule: Clears all registrations but preserves singleton instance.
        Used for test cleanup and container reset scenarios. Does not reset
        singleton pattern - use reset_for_testing() for that.

        """
        names_to_clear = {
            *self._services.keys(),
            *self._factories.keys(),
            *self._resources.keys(),
            *self._core_names(),
        }
        for name in names_to_clear:
            self._clear_provider_registration(str(name))
        self._services.clear()
        self._factories.clear()
        self._resources.clear()
        self._internal_registrations.clear()
        self._config = FlextSettings.fetch_global()
        self.register_core_services()

    @override
    def configure(self, settings: Mapping[str, t.Container] | None = None) -> Self:
        """Configure the container with flat validated overrides."""
        return self.apply(settings)

    @override
    def apply(self, settings: Mapping[str, t.Container] | None = None) -> Self:
        """Apply user-provided overrides to container configuration.

        Args:
            settings: Mapping of configuration keys to values accepted by
                ``t.Scalar``.

        """
        if settings is None:
            return self
        config_map: Mapping[str, t.Container] = settings
        merged = self._user_overrides.model_copy()
        merged.update({
            str(k): FlextRuntime.to_plain_container(u.normalize_to_container(v))
            for k, v in config_map.items()
        })
        self._user_overrides = merged
        if applicable := {
            k: v for k, v in merged.items() if k in m.ContainerConfig.model_fields
        }:
            self._global_config = self._global_config.model_copy(
                update=applicable, deep=True
            )
        self.sync_config_to_di()
        return self

    @override
    def logger(
        self,
        module_name: str | None = None,
        *,
        service_name: str | None = None,
        service_version: str | None = None,
        correlation_id: str | None = None,
    ) -> p.Logger:
        """Create a module logger for the specified runtime scope.

        This method provides direct access to the logger implementation without going through
        the generic DI resolution. Use this for logging needs instead of
        container.get("logger").

        Args:
            module_name: Module name for the logger. Defaults to "flext_core".

        Returns:
            A protocol-typed logger configured for the specified module.

        """
        return u.fetch_logger(module_name or c.DEFAULT_LOGGER_MODULE)

    @staticmethod
    def _is_service_of_type[T: t.RegisterableService](
        value: t.RegisterableService,
        cls: type[T],
    ) -> TypeIs[T]:
        """TypeIs guard that narrows an object to T for pyright."""
        return isinstance(value, cls)

    @staticmethod
    def _narrow_service[T: t.RegisterableService](
        service: t.RegisterableService,
        cls: type[T],
    ) -> r[T]:
        """Narrow a service object to a concrete type T via runtime checking."""
        if FlextContainer._is_service_of_type(service, cls):
            return r[T].ok(service)
        type_name = cls.__name__
        return r[T].from_result(
            e.fail_type_mismatch(
                type_name,
                type(service).__name__,
                service_name=type_name,
            )
        )

    @overload
    def resolve[T: t.RegisterableService](
        self,
        name: str,
        *,
        type_cls: type[T],
    ) -> r[T]: ...

    @overload
    def resolve(
        self,
        name: str,
        *,
        type_cls: None = None,
    ) -> r[t.RegisterableService]: ...

    @override
    def resolve[T: t.RegisterableService](
        self,
        name: str,
        *,
        type_cls: type[T] | None = None,
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
            >>> container.bind("logger", u.fetch_logger(__name__))
            >>> result = container.resolve("logger")
            >>> if result.success and isinstance(result.value, p.Logger):
            ...     result.value.info("Resolved")

        """
        if name in self._services:
            service_registration = self._services[name]
            service = service_registration.service
            if type_cls is not None:
                return self._narrow_service(service, type_cls)
            return r[t.RegisterableService].ok(service)
        if name in self._factories:
            try:
                factory_registration = self._factories[name]
                factory_callable: t.FactoryCallable = factory_registration.factory
                resolved = factory_callable()
                if type_cls is not None:
                    return self._narrow_service(resolved, type_cls)
                return r[t.RegisterableService].ok(resolved)
            except (
                TypeError,
                ValueError,
                RuntimeError,
                KeyError,
                AttributeError,
            ) as exc:
                return r[t.RegisterableService].from_result(
                    e.fail_operation("resolve factory service", exc)
                )
        if name in self._resources:
            try:
                resource_registration = self._resources[name]
                resource_callable: t.ResourceCallable = resource_registration.factory
                resolved = resource_callable()
                if not u.registerable_service(resolved):
                    return r[t.RegisterableService].from_result(
                        e.fail_type_mismatch(
                            "registerable service",
                            resolved.__class__.__name__,
                            service_name=name,
                        )
                    )
                if type_cls is not None:
                    return self._narrow_service(resolved, type_cls)
                return r[t.RegisterableService].ok(resolved)
            except (
                TypeError,
                ValueError,
                RuntimeError,
                KeyError,
                AttributeError,
            ) as exc:
                return r[t.RegisterableService].from_result(
                    e.fail_operation("resolve resource service", exc)
                )
        return r[t.RegisterableService].from_result(e.fail_not_found("service", name))

    @override
    def snapshot(self) -> m.ConfigMap:
        """Return the merged settings exposed by this container."""
        config_dict_raw = self._global_config.model_dump()
        return m.ConfigMap(
            root={
                str(key): u.normalize_to_container(value)
                for key, value in config_dict_raw.items()
            }
        )

    @override
    def has(self, name: str) -> bool:
        """Return whether a public service, factory, or resource is registered."""
        return (
            (name in self._services and name not in self._internal_registrations)
            or (name in self._factories and name not in self._internal_registrations)
            or (name in self._resources and name not in self._internal_registrations)
        )

    @override
    def initialize_di_components(self) -> None:
        """Initialize DI components (bridge, services, resources, container).

        Internal method to set up dependency injection infrastructure.
        Can be called from __init__ or _create_scoped_instance.
        Sets private attributes directly - this is internal initialization.
        """
        self._di_bridge, self._di_services, self._di_resources = (
            u.DependencyIntegration.create_layered_bridge()
        )
        self._di_container = di_containers.DynamicContainer()
        config_provider_raw = self._di_bridge.settings
        if not isinstance(config_provider_raw, di_providers.Configuration):
            error_msg = "Bridge must have settings provider"
            raise TypeError(error_msg)
        config_provider: di_providers.Configuration = config_provider_raw
        self._base_config_provider = di_providers.Configuration()
        self._user_config_provider = di_providers.Configuration()
        config_provider.override(self._base_config_provider)
        config_provider.override(self._user_config_provider)
        self._di_container.settings = config_provider
        self._config_provider = config_provider

    @override
    def initialize_registrations(
        self,
        *,
        services: Mapping[str, m.ServiceRegistration] | None = None,
        factories: Mapping[str, m.FactoryRegistration] | None = None,
        resources: Mapping[str, m.ResourceRegistration] | None = None,
        global_config: m.ContainerConfig | None = None,
        user_overrides: (
            t.UserOverridesMapping
            | m.ConfigMap
            | Mapping[str, m.ConfigMap | t.ScalarList | t.Scalar]
            | None
        ) = None,
        settings: p.Settings | None = None,
        context: p.Context | None = None,
    ) -> None:
        """Initialize service registrations and configuration.

        Internal method to set up registrations and settings.
        Can be called from __init__ or _create_scoped_instance.
        Sets private attributes directly - this is internal initialization.
        """
        self._services = dict(services) if services is not None else {}
        self._factories = dict(factories) if factories is not None else {}
        self._resources = dict(resources) if resources is not None else {}
        self._internal_registrations = {
            str(name)
            for name in (
                list(self._services.keys())
                + list(self._factories.keys())
                + list(self._resources.keys())
            )
            if str(name) in self._core_names()
        }
        self._global_config = global_config or self._create_container_config()
        if isinstance(user_overrides, m.ConfigMap):
            user_overrides_map = user_overrides
        elif user_overrides is not None:
            user_overrides_map = m.ConfigMap(
                root={
                    ok: list(ov)
                    if isinstance(ov, Sequence) and not isinstance(ov, str | bytes)
                    else ov
                    for ok, ov in user_overrides.items()
                }
            )
        else:
            user_overrides_map = m.ConfigMap(root={})
        self._user_overrides = user_overrides_map
        config_instance = self._require_settings(
            settings if settings is not None else FlextSettings.fetch_global(),
            source="container settings",
        )
        self._config = config_instance
        # Always guarantee context is initialized to valid-by-construction design.
        # If no context was provided, create a default FlextContext instance.
        # This eliminates untestable error states where context could be None.
        if context is not None:
            self._context = self._require_context(context, source="container context")
        else:
            default_context = FlextContext()
            if not isinstance(default_context, p.Context):
                error_msg = "default context must implement p.Context"
                raise TypeError(error_msg)
            self._context = default_context

    @override
    def names(self) -> t.StrSequence:
        """List explicitly registered services, factories, and resources.

        Auto-registered core infrastructure remains resolvable through the
        container, but it is intentionally hidden from this public listing so
        callers see only user-facing registrations.
        """
        registered_names = (
            list(self._services.keys())
            + list(self._factories.keys())
            + list(self._resources.keys())
        )
        return [
            name
            for name in registered_names
            if str(name) not in self._internal_registrations
        ]

    def _bind_service(self, name: str, impl: t.RegisterableService) -> Self:
        """Register a concrete service instance under ``name``."""
        if not name or self.has(name):
            return self
        self._clear_provider_registration(name)
        self._internal_registrations.discard(name)
        registration = m.ServiceRegistration(
            name=name,
            service=impl,
            service_type=impl.__class__.__name__,
        )
        self._services[name] = registration
        try:
            self._bind_provider(
                name,
                u.DependencyIntegration.register_object(self._di_services, name, impl),
            )
        except (TypeError, ValueError, RuntimeError, AttributeError):
            del self._services[name]
        return self

    def _bind_factory(self, name: str, impl: t.FactoryCallable) -> Self:
        """Register a factory callable under ``name``."""
        if not name or self.has(name):
            return self
        self._clear_provider_registration(name)
        self._internal_registrations.discard(name)

        def normalized_factory() -> t.RegisterableService:
            raw_result = impl()
            if not u.registerable_service(raw_result):
                raise ValueError(
                    c.ERR_CONTAINER_FACTORY_INVALID_REGISTERABLE.format(
                        name=name,
                    ),
                )
            return raw_result

        self._factories[name] = m.FactoryRegistration(
            name=name,
            factory=normalized_factory,
        )
        try:
            self._bind_provider(
                name,
                u.DependencyIntegration.register_factory(
                    self._di_services,
                    name,
                    normalized_factory,
                    cache=self._global_config.enable_factory_caching,
                ),
            )
        except (TypeError, ValueError, RuntimeError, AttributeError):
            del self._factories[name]
        return self

    def _bind_resource(self, name: str, impl: t.ResourceCallable) -> Self:
        """Register a resource factory under ``name``."""
        if not name or self.has(name):
            return self
        self._clear_provider_registration(name)
        self._internal_registrations.discard(name)
        self._resources[name] = m.ResourceRegistration(name=name, factory=impl)
        try:
            self._bind_provider(
                name,
                u.DependencyIntegration.register_resource(
                    self._di_resources, name, impl
                ),
            )
        except (TypeError, ValueError, RuntimeError, AttributeError):
            del self._resources[name]
        return self

    @override
    def bind(self, name: str, impl: t.RegisterableService) -> Self:
        """Bind a concrete service instance or value."""
        return self._bind_service(name, impl)

    @override
    def factory(self, name: str, impl: t.FactoryCallable) -> Self:
        """Bind a factory callable."""
        if not u.factory(impl):
            return self
        return self._bind_factory(name, impl)

    @override
    def resource(self, name: str, impl: t.ResourceCallable) -> Self:
        """Bind a resource factory."""
        if not u.resource(impl):
            return self
        return self._bind_resource(name, impl)

    @override
    def register_core_services(self) -> None:
        """Auto-register core services for easy DI access.

        Auto-registered services:
        - "settings" → FlextSettings singleton
        - "logger" → protocol-typed logger factory via `u.fetch_logger(...)`
        - "context" → FlextContext singleton
        - "container" → Self-reference for nested resolution

        Business Rule: Auto-registers FlextSettings, the public logging DSL, and FlextContext
        with standard names ("settings", "logger", "context") to enable easy
        dependency injection in downstream projects. Services are registered only
        if not already registered to avoid conflicts.

        This method ensures that core services are accessible via:
        - container.resolve("settings") -> FlextSettings
        - container.resolve("logger") -> `p.Logger` (factory)
        - container.resolve("context") -> FlextContext
        - container.resolve("container") -> FlextContainer (self-reference)

        Services are registered as:
        - "settings": Singleton instance (container.settings property)
        - "logger": Factory that creates module logger
        - "context": Singleton instance (container.context property)
        - "container": Self-reference for nested resolution

        Note: Core bootstrap uses internal DI-aware checks so public registration
        helpers can remain focused on user-defined names.
        """
        if not self._has_internal_registration(
            str(c.Directory.CONFIG)
        ) and u.registerable_service(self._config):
            _ = self.bind(c.Directory.CONFIG, self._config)
            self._internal_registrations.add(str(c.Directory.CONFIG))
        if not self._has_internal_registration(str(c.ServiceName.LOGGER)):
            _ = self.factory(
                c.ServiceName.LOGGER,
                lambda: u.fetch_logger(c.DEFAULT_LOGGER_MODULE),
            )
            self._internal_registrations.add(str(c.ServiceName.LOGGER))
        if not self._has_internal_registration(
            str(c.FIELD_CONTEXT)
        ) and u.registerable_service(self._context):
            _ = self.bind(c.FIELD_CONTEXT, self._context)
            self._internal_registrations.add(str(c.FIELD_CONTEXT))
        if not self._has_internal_registration(str(c.ServiceName.COMMAND_BUS)):
            dispatcher = u.build_dispatcher()
            if u.registerable_service(dispatcher):
                _ = self.bind(c.ServiceName.COMMAND_BUS, dispatcher)
                self._internal_registrations.add(str(c.ServiceName.COMMAND_BUS))

    @override
    def register_existing_providers(self) -> None:
        """Hydrate the dynamic container with current registrations."""
        cache = self._global_config.enable_factory_caching
        for name, reg in self._services.items():
            if not (
                hasattr(self._di_services, name) or hasattr(self._di_container, name)
            ):
                self._bind_provider(
                    name,
                    u.DependencyIntegration.register_object(
                        self._di_services, name, reg.service
                    ),
                )
        for name, reg in self._factories.items():
            if not (
                hasattr(self._di_services, name) or hasattr(self._di_container, name)
            ):
                self._bind_provider(
                    name,
                    u.DependencyIntegration.register_factory(
                        self._di_services, name, reg.factory, cache=cache
                    ),
                )
        for name, reg in self._resources.items():
            if not (
                hasattr(self._di_resources, name) or hasattr(self._di_container, name)
            ):
                self._bind_provider(
                    name,
                    u.DependencyIntegration.register_resource(
                        self._di_resources, name, reg.factory
                    ),
                )

    @override
    def scope(
        self,
        *,
        settings: p.Settings | None = None,
        context: p.Context | None = None,
        subproject: str | None = None,
        services: Mapping[str, t.RegisterableService] | None = None,
        factories: Mapping[str, t.FactoryCallable] | None = None,
        resources: Mapping[str, t.ResourceCallable] | None = None,
    ) -> Self:
        """Create an isolated container scope with optional overrides.

        Args:
            settings: Optional settings overriding the global container's
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
        settings_source = settings if settings is not None else self._config
        base_config: p.Settings = settings_source.model_copy(deep=True)
        if subproject and settings is None:
            base_config = base_config.model_copy(
                update={"app_name": f"{base_config.app_name}.{subproject}"},
            )
        scoped_context: p.Context
        if context is None:
            scoped_context = self._require_context(
                self.context,
                source="scoped base context",
            ).clone()
        else:
            scoped_context = self._require_context(
                context,
                source="scoped override context",
            )
        if subproject:
            _ = scoped_context.set("subproject", subproject)
        cloned_services: MutableMapping[str, m.ServiceRegistration] = {
            name: registration.model_copy(deep=False)
            for name, registration in self._services.items()
        }
        cloned_factories: MutableMapping[str, m.FactoryRegistration] = {
            name: registration.model_copy(deep=False)
            for name, registration in self._factories.items()
        }
        cloned_resources: MutableMapping[str, m.ResourceRegistration] = {
            name: registration.model_copy(deep=False)
            for name, registration in self._resources.items()
        }
        for name, service in (services or {}).items():
            if not u.registerable_service(service):
                continue
            cloned_services[name] = m.ServiceRegistration(
                name=name,
                service=service,
                service_type=service.__class__.__name__,
            )
        for name, factory in (factories or {}).items():
            if u.factory(factory):
                cloned_factories[name] = m.FactoryRegistration(
                    name=name, factory=factory
                )
        for name, resource_factory in (resources or {}).items():
            if u.resource(resource_factory):
                cloned_resources[name] = m.ResourceRegistration(
                    name=name, factory=resource_factory
                )
        return self.__class__._create_scoped_instance(
            registration=m.ServiceRegistrationSpec(
                settings=base_config,
                context=scoped_context,
                services=cloned_services,
                factories=cloned_factories,
                resources=cloned_resources,
                user_overrides=self._user_overrides.model_copy(),
                container_config=self._global_config.model_copy(deep=True),
            ),
        )

    @override
    def sync_config_to_di(self) -> None:
        """Synchronize FlextSettings to DI providers.Configuration.

        Dependency Injector's layered ``providers.Configuration`` instances are
        used to avoid manual merges while still honoring validated defaults from
        ``FlextSettings`` and runtime overrides. Base settings and user overrides are
        applied as separate providers to keep precedence explicit.

        Also registers namespace configs as factories for easy DI access:
        - "settings.ldif" → FlextLdifSettings instance
        - "settings.ldap" → FlextLdapSettings instance
        - etc.
        """
        config_dict_raw = self._global_config.model_dump()
        config_map = m.ConfigMap(
            root={
                str(key): u.normalize_to_container(value)
                for key, value in config_dict_raw.items()
            }
        )
        _ = u.DependencyIntegration.bind_configuration(
            self._di_container,
            config_map,
        )
        user_overrides_plain = dict(self._user_overrides)
        self._user_config_provider.from_dict(user_overrides_plain)
        namespaces = FlextSettings.registered_namespaces()
        if not namespaces:
            return
        for namespace in namespaces:
            factory_name = f"settings.{namespace}"
            settings_class = FlextSettings.resolve_namespace_settings(namespace)
            if settings_class is None:
                continue
            settings_class_non_null: t.SettingsClass = settings_class

            def namespace_factory(
                *,
                _namespace: str = namespace,
                _settings_class: t.SettingsClass = settings_class_non_null,
            ) -> t.ModelCarrier:
                namespace_settings = FlextSettings.fetch_global().fetch_namespace(
                    _namespace,
                    _settings_class,
                )
                if not u.pydantic_model(namespace_settings):
                    msg = f"Namespace settings '{_namespace}' must be a Pydantic model"
                    raise TypeError(msg)
                return namespace_settings

            if not self.has(factory_name):
                self.factory(
                    factory_name,
                    namespace_factory,
                )

    @override
    def drop(self, name: str) -> r[bool]:
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
        return r[bool].from_result(e.fail_not_found("service", name))

    @override
    def wire(
        self,
        *,
        modules: Sequence[ModuleType] | None = None,
        packages: t.StrSequence | None = None,
        classes: Sequence[type] | None = None,
    ) -> None:
        """Wire modules/packages to the DI bridge for @inject/Provide usage."""
        u.DependencyIntegration.wire(
            self._di_container,
            modules=modules,
            packages=packages,
            classes=classes,
        )

    @override
    def dispatcher(self) -> r[p.Dispatcher]:
        """Resolve the canonical dispatcher / command bus."""
        result = self.resolve(c.ServiceName.COMMAND_BUS)
        if result.failure:
            return r[p.Dispatcher].from_result(
                e.fail_not_found("dispatcher", c.ServiceName.COMMAND_BUS)
            )
        candidate = result.value
        if isinstance(candidate, p.Dispatcher):
            return r[p.Dispatcher].ok(candidate)
        return r[p.Dispatcher].from_result(
            e.fail_type_mismatch(
                "dispatcher",
                candidate.__class__.__name__,
                service_name=c.ServiceName.COMMAND_BUS,
            )
        )


__all__: list[str] = ["FlextContainer"]
