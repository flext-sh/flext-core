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
from pydantic import BaseModel, ValidationError

from flext_core import c, e, m, p, r, t, u
from flext_core.context import FlextContext
from flext_core.settings import FlextSettings


def _is_service_of_type[T: t.RegisterableService](
    value: object,
    cls: type[T],
) -> TypeIs[T]:
    """TypeIs guard that narrows an object to T for pyright."""
    return isinstance(value, cls)


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
    _services: MutableMapping[str, m.ServiceRegistration]
    _factories: MutableMapping[str, m.FactoryRegistration]
    _resources: MutableMapping[str, m.ResourceRegistration]
    _global_config: m.ContainerConfig

    @staticmethod
    def _merge_registration_specs(
        base: m.ServiceRegistrationSpec,
        override: m.ServiceRegistrationSpec,
    ) -> m.ServiceRegistrationSpec:
        """Merge two ServiceRegistrationSpec using Pydantic v2 model_copy.

        Produces a copy of *base* with every non-None field from *override*
        applied on top.  ``model_copy(update=...)`` is the canonical Pydantic v2
        API for this pattern — it bypasses the synthesized ``__init__`` entirely,
        copies validated field values directly, and preserves the exact field
        types declared on the model (no ``LaxStr`` / ``Any`` widening).
        """
        override_updates: MutableMapping[
            str,
            t.RegistrationKwarg | t.UserOverridesMapping | t.ConfigMap,
        ] = {}
        if override.settings is not None:
            override_updates["settings"] = override.settings
        if override.context is not None:
            override_updates["context"] = override.context
        if override.services is not None:
            override_updates["services"] = override.services
        if override.factories is not None:
            override_updates["factories"] = override.factories
        if override.resources is not None:
            override_updates["resources"] = override.resources
        if override.user_overrides is not None:
            override_updates["user_overrides"] = override.user_overrides
        if override.container_config is not None:
            override_updates["container_config"] = override.container_config
        return base.model_copy(update=override_updates)

    @staticmethod
    def _resolve_bootstrap_registration(
        registration: m.ServiceRegistrationSpec | None,
        **registration_kwargs: t.RegistrationKwarg,
    ) -> m.ServiceRegistrationSpec:
        base_registration = (
            registration if registration is not None else m.ServiceRegistrationSpec()
        )
        if not registration_kwargs:
            return base_registration
        override_registration = m.ServiceRegistrationSpec.model_validate({
            "settings": registration_kwargs.get("_config"),
            "context": registration_kwargs.get("_context"),
            "services": registration_kwargs.get("_services"),
            "factories": registration_kwargs.get("_factories"),
            "resources": registration_kwargs.get("_resources"),
            "user_overrides": registration_kwargs.get("_user_overrides"),
            "container_config": registration_kwargs.get("_container_config"),
        })
        return FlextContainer._merge_registration_specs(
            base_registration,
            override_registration,
        )

    @staticmethod
    def _resolve_scoped_registration(
        registration: m.ServiceRegistrationSpec,
        **registration_kwargs: t.RegistrationKwarg,
    ) -> m.ServiceRegistrationSpec:
        if not registration_kwargs:
            return registration
        override_registration = m.ServiceRegistrationSpec.model_validate(
            registration_kwargs,
        )
        return FlextContainer._merge_registration_specs(
            registration,
            override_registration,
        )

    def __new__(
        cls,
        *,
        registration: m.ServiceRegistrationSpec | None = None,
        **registration_kwargs: t.RegistrationKwarg,
    ) -> Self:
        """Create or return the global singleton instance.

        Optional keyword arguments support deterministic construction in tests
        while preserving singleton semantics for runtime callers. Double-checked
        locking protects against duplicate initialization under concurrency.
        """
        _ = cls._resolve_bootstrap_registration(registration, **registration_kwargs)
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
        **registration_kwargs: t.RegistrationKwarg,
    ) -> None:
        """Wire the Dependency Injector container and supporting registries.

        The initializer is idempotent: repeated construction returns early to
        keep the singleton stable. Parameters allow deterministic construction
        during testing or for scoped containers created via :meth:`scoped`.
        """
        super().__init__()
        if hasattr(self, "_di_container"):
            return
        init_registration = self._resolve_bootstrap_registration(
            registration,
            **registration_kwargs,
        )
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

    @property
    @override
    def settings(self) -> p.Settings:
        """Return configuration bound to this container."""
        if self._config is None:
            error_msg = c.ERR_CONTAINER_CONFIG_NOT_INITIALIZED
            raise RuntimeError(error_msg)
        return self._config

    @property
    @override
    def context(self) -> p.Context:
        """Return the execution context bound to this container.

        The context must be provided during container initialization via the
        registration specification in `__init__` or `fetch_global()`. If no
        context was provided, this property will raise an error.

        Raises:
            RuntimeError: If context was not provided during initialization.

        Example:
            >>> container = FlextContainer.fetch_global(context=my_context)
            >>> ctx = container.context  # Returns the provided context

        """
        if not hasattr(self, "_context") or self._context is None:
            error_msg = c.ERR_CONTAINER_CONTEXT_NOT_INITIALIZED
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

           container = FlextContainer.fetch_global()
           _ = container.register(
               "token_factory",
               lambda: {"token": "abc123"},
               kind=c.ContainerKind.FACTORY,
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
            msg = c.ERR_CONTAINER_PROVIDE_HELPER_NOT_INITIALIZED
            raise RuntimeError(msg)
        provide_fn: Callable[[str], t.RegisterableService] = provide_helper

        def provide_callable(name: str) -> t.RegisterableService:
            provided = provide_fn(name)
            try:
                m.ServiceRegistration(name="provided", service=provided)
                return provided
            except ValidationError:
                msg = c.ERR_CONTAINER_PROVIDE_HELPER_UNSUPPORTED_TYPE
                raise TypeError(msg) from None

        return provide_callable

    @classmethod
    def _create_scoped_instance(
        cls,
        *,
        registration: m.ServiceRegistrationSpec,
        **registration_kwargs: t.RegistrationKwarg,
    ) -> Self:
        """Create a scoped container instance bypassing singleton pattern.

        This is an internal factory method to safely create non-singleton containers.
        Uses direct attribute assignment (no frozen=True, compatible with u pattern).
        """
        scoped_registration = cls._resolve_scoped_registration(
            registration,
            **registration_kwargs,
        )
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
        automatically, enabling zero-settings factory discovery for services.
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
        This enables zero-settings factory discovery for services.

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
        for factory_name, factory_config in factories:
            factory_func_raw = getattr(caller_module, factory_name, None)
            if factory_func_raw is None or not u.factory(factory_func_raw):
                continue
            factory_func_ref: t.FactoryCallable = factory_func_raw
            wrapper = FlextContainer._make_factory_wrapper(
                factory_func_ref,
                factory_name,
                factory_config,
            )
            _ = instance.register(
                factory_config.name,
                wrapper,
                kind=c.ContainerKind.FACTORY,
            )

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
            config_callable = getattr(_factory_config, "fn", None)
            raw_result: t.RegisterableService
            match config_callable:
                case _ if callable(config_callable):
                    config_raw = config_callable()
                    if isinstance(config_raw, (*t.CONTAINER_TYPES, BaseModel)):
                        raw_result = config_raw
                    else:
                        raw_result = str(config_raw)
                case None:
                    raw_result = _factory_func_ref()
                case _:
                    return t.ConfigMap(root={})
            try:
                m.ServiceRegistration(name=_factory_name, service=raw_result)
                return raw_result
            except ValidationError:
                msg = f"Factory '{_factory_name}' returned unsupported type: {raw_result.__class__.__name__}"
                raise TypeError(msg) from None

        return factory_wrapper

    @classmethod
    def fetch_global(
        cls,
        *,
        settings: p.Settings | None = None,
        context: p.Context | None = None,
    ) -> Self:
        """Return the thread-safe global container instance.

        The first call initializes the singleton using the optional configuration
        and context. Subsequent calls return the same instance without modifying
        previously applied settings.
        """
        instance = cls()
        if settings is not None:
            instance._config = settings
        if context is not None:
            instance._context = context
        return instance

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
    def configure(self, settings: t.FlatContainerMapping | None = None) -> Self:
        """Apply user-provided overrides to container configuration.

        Args:
            settings: Mapping of configuration keys to values accepted by
                ``t.Scalar``.

        """
        if settings is None:
            return self
        config_map: t.FlatContainerMapping = settings
        processed_dict = t.ConfigMap(root={})
        for key, value in config_map.items():
            processed_dict[str(key)] = u.normalize_to_container(value)
        merged = t.ConfigMap(root=dict(self._user_overrides))
        merged.update(dict(processed_dict))
        self._user_overrides = merged
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
        return self

    @override
    def create_module_logger(
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
        _ = service_name
        _ = service_version
        _ = correlation_id
        return u.fetch_logger(module_name or c.DEFAULT_LOGGER_MODULE)

    @staticmethod
    def _narrow_service[T: t.RegisterableService](
        service: object,
        cls: type[T],
    ) -> r[T]:
        """Narrow a service object to a concrete type T via runtime checking."""
        if _is_service_of_type(service, cls):
            return r[T].ok(service)
        type_name = cls.__name__ if hasattr(cls, "__name__") else "Unknown"
        return e.fail_type_mismatch(
            type_name,
            type(service).__name__,
            service_name=type_name,
        )

    @overload
    def get[T: t.RegisterableService](
        self,
        name: str,
        *,
        type_cls: type[T],
    ) -> r[T]: ...

    @overload
    def get(self, name: str, *, type_cls: None = None) -> r[t.RegisterableService]: ...

    @override
    def get[T: t.RegisterableService](
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
            >>> container.register("logger", u.fetch_logger(__name__))
            >>> result = container.get("logger")
            >>> if result.success and isinstance(result.value, p.Logger):
            ...     result.value.info("Resolved")

        """
        if name in self._services:
            service_registration = self._services[name]
            service = service_registration.service
            if type_cls is not None:
                return self._narrow_service(service, type_cls)
            narrowed_service: t.RegisterableService = service
            return r[t.RegisterableService].ok(narrowed_service)
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
                return e.fail_operation("resolve factory service", exc)
        if name in self._resources:
            try:
                resource_registration = self._resources[name]
                resource_callable: t.ResourceCallable = resource_registration.factory
                resolved = resource_callable()
                if not u.registerable_service(resolved):
                    return e.fail_type_mismatch(
                        "registerable service",
                        resolved.__class__.__name__,
                        service_name=name,
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
                return e.fail_operation("resolve resource service", exc)
        return e.fail_not_found("service", name)

    @override
    def resolve_settings(self) -> t.ConfigMap:
        """Return the merged settings exposed by this container."""
        config_dict_raw = self._global_config.model_dump()
        return t.ConfigMap(
            root={
                str(key): u.normalize_to_container(value)
                for key, value in config_dict_raw.items()
            },
        )

    @override
    def has_service(self, name: str) -> bool:
        """Return whether a service or factory is registered for ``name``."""
        return (
            name in self._services or name in self._factories or name in self._resources
        )

    @override
    def initialize_di_components(self) -> None:
        """Initialize DI components (bridge, services, resources, container).

        Internal method to set up dependency injection infrastructure.
        Can be called from __init__ or _create_scoped_instance.
        Sets private attributes directly - this is internal initialization.
        """
        bridge_tuple = u.DependencyIntegration.create_layered_bridge()
        bridge = bridge_tuple[0]
        service_module = bridge_tuple[1]
        resource_module = bridge_tuple[2]
        di_container = di_containers.DynamicContainer()
        self._di_bridge = bridge
        self._di_services = service_module
        self._di_resources = resource_module
        self._di_container = di_container
        config_attr = c.Directory.CONFIG
        if not hasattr(bridge, config_attr):
            error_msg = c.ERR_CONTAINER_BRIDGE_MUST_HAVE_CONFIG_PROVIDER
            raise TypeError(error_msg)
        config_provider_obj = (
            getattr(bridge, config_attr) if hasattr(bridge, config_attr) else None
        )
        if config_provider_obj is None:
            error_msg = c.ERR_CONTAINER_BRIDGE_CONFIG_PROVIDER_CANNOT_BE_NONE
            raise TypeError(error_msg)
        if not u.instance_of(config_provider_obj, di_providers.Configuration):
            error_msg = c.ERR_CONTAINER_BRIDGE_MUST_HAVE_CONFIG_PROVIDER
            raise TypeError(error_msg)
        config_provider = config_provider_obj
        base_config_provider = di_providers.Configuration()
        user_config_provider = di_providers.Configuration()
        self._base_config_provider = base_config_provider
        self._user_config_provider = user_config_provider
        override_method = getattr(config_provider, "override", None)
        if not callable(override_method):
            error_msg = c.ERR_CONTAINER_BRIDGE_CONFIG_PROVIDER_MUST_SUPPORT_OVERRIDE
            raise TypeError(error_msg)
        override_method(base_config_provider)
        override_method(user_config_provider)
        di_container.settings = config_provider
        self._config_provider = config_provider

    @override
    def initialize_registrations(
        self,
        *,
        services: Mapping[str, m.ServiceRegistration] | None = None,
        factories: Mapping[str, m.FactoryRegistration] | None = None,
        resources: Mapping[str, m.ResourceRegistration] | None = None,
        global_config: m.ContainerConfig | None = None,
        user_overrides: t.UserOverridesMapping | t.ConfigMap | None = None,
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
        self._global_config = global_config or self._create_container_config()
        overrides_root: MutableMapping[str, t.ValueOrModel] = {}
        if user_overrides is not None:
            if isinstance(user_overrides, t.ConfigMap):
                overrides_root = dict(user_overrides.root)
            else:
                for ok, ov in user_overrides.items():
                    if isinstance(ov, (*t.CONTAINER_TYPES, BaseModel)):
                        overrides_root[ok] = ov
                    elif isinstance(ov, Sequence):
                        overrides_root[ok] = list(ov)
                    else:
                        overrides_root[ok] = ov
        user_overrides_map = t.ConfigMap(root=overrides_root)
        self._user_overrides = user_overrides_map
        config_instance: p.Settings = (
            settings if settings is not None else FlextSettings.fetch_global()
        )
        self._config = config_instance
        self._context = context

    @override
    def list_services(self) -> t.StrSequence:
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
        impl: t.RegisterableService,
        *,
        kind: str = c.ContainerKind.SERVICE,
    ) -> Self:
        """Register a service instance for dependency resolution.

        Business Rule: The container accepts service values for registration,
        including recursive containers, protocols (Config, Ctx, DI, Service, Log,
        Handler, Registry), and callables. This enables dependency injection of
        typed service instances and protocol implementations.

        Args:
            name: Unique key for the registration.
            impl: Concrete instance or callable used for registration.
                Must be a canonical registerable service (primitives, Pydantic model, callable,
                sequence, or mapping).

        Returns:
            ``r`` indicating whether the registration succeeded. A
            failed result is returned when the name is already registered or
            when construction fails.

        """
        if not name:
            return self
        try:
            if kind == c.ContainerKind.SERVICE:
                if hasattr(self._di_services, name):
                    return self
                service_impl: t.RegisterableService = impl
                registration = m.ServiceRegistration(
                    name=name,
                    service=service_impl,
                    service_type=service_impl.__class__.__name__,
                )
                self._services[name] = registration
                provider = u.DependencyIntegration.register_object(
                    self._di_services,
                    name,
                    service_impl,
                )
                setattr(self._di_bridge, name, provider)
                setattr(self._di_container, name, provider)
                return self
            if kind == c.ContainerKind.FACTORY:
                if not u.factory(impl):
                    return self
                if hasattr(self._di_services, name):
                    return self
                factory_fn: t.FactoryCallable = impl

                def normalized_factory() -> t.RegisterableService:
                    raw_result = factory_fn()
                    if not u.registerable_service(raw_result):
                        raise ValueError(
                            c.ERR_CONTAINER_FACTORY_INVALID_REGISTERABLE.format(
                                name=name,
                            ),
                        )
                    return raw_result

                factory_reg = m.FactoryRegistration(
                    name=name,
                    factory=normalized_factory,
                )
                self._factories[name] = factory_reg
                provider = u.DependencyIntegration.register_factory(
                    self._di_services,
                    name,
                    normalized_factory,
                    cache=self._global_config.enable_factory_caching,
                )
                setattr(self._di_bridge, name, provider)
                setattr(self._di_container, name, provider)
                return self
            if not u.resource(impl):
                return self
            if hasattr(self._di_resources, name):
                return self
            resource_reg = m.ResourceRegistration(name=name, factory=impl)
            self._resources[name] = resource_reg
            provider = u.DependencyIntegration.register_resource(
                self._di_resources,
                name,
                impl,
            )
            setattr(self._di_bridge, name, provider)
            setattr(self._di_container, name, provider)
            return self
        except (TypeError, ValueError, RuntimeError, AttributeError) as exc:
            _ = exc
            return self

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
        - container.get("settings") -> FlextSettings
        - container.get("logger") -> `p.Logger` (factory)
        - container.get("context") -> FlextContext
        - container.get("container") -> FlextContainer (self-reference)

        Services are registered as:
        - "settings": Singleton instance (container.settings property)
        - "logger": Factory that creates module logger
        - "context": Singleton instance (container.context property)
        - "container": Self-reference for nested resolution

        Note: Uses has_service() which checks both dicts and DI container to avoid conflicts.
        """
        if (
            not self.has_service(c.Directory.CONFIG)
            and self._config is not None
            and u.registerable_service(self._config)
        ):
            _ = self.register(c.Directory.CONFIG, self._config)
        if not self.has_service(c.ServiceName.LOGGER):
            _ = self.register(
                c.ServiceName.LOGGER,
                lambda: u.fetch_logger(c.DEFAULT_LOGGER_MODULE),
                kind=c.ContainerKind.FACTORY,
            )
        if (
            not self.has_service(c.FIELD_CONTEXT)
            and self._context is not None
            and u.registerable_service(self._context)
        ):
            _ = self.register(c.FIELD_CONTEXT, self._context)
        if not self.has_service(c.ServiceName.COMMAND_BUS):
            dispatcher = u.build_dispatcher()
            dispatcher_candidate = dispatcher
            if not u.registerable_service(dispatcher_candidate):
                return
            service_candidate: t.RegisterableService = dispatcher_candidate
            dispatcher_name = c.ServiceName.COMMAND_BUS
            registration = m.ServiceRegistration(
                name=dispatcher_name,
                service=service_candidate,
                service_type=type(service_candidate).__name__,
            )
            self._services[dispatcher_name] = registration
            if not hasattr(self._di_services, dispatcher_name):
                provider = u.DependencyIntegration.register_object(
                    self._di_services,
                    dispatcher_name,
                    service_candidate,
                )
                setattr(self._di_bridge, dispatcher_name, provider)
                setattr(self._di_container, dispatcher_name, provider)

    @override
    def register_existing_providers(self) -> None:
        """Hydrate the dynamic container with current registrations."""
        for name, registration in self._services.items():
            if hasattr(self._di_services, name) or hasattr(self._di_container, name):
                continue
            provider = u.DependencyIntegration.register_object(
                self._di_services,
                name,
                registration.service,
            )
            setattr(self._di_bridge, name, provider)
            setattr(self._di_container, name, provider)
        for name, factory_registration in self._factories.items():
            if hasattr(self._di_services, name) or hasattr(self._di_container, name):
                continue
            provider = u.DependencyIntegration.register_factory(
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
            provider = u.DependencyIntegration.register_resource(
                self._di_resources,
                name,
                resource_registration.factory,
            )
            setattr(self._di_bridge, name, provider)
            setattr(self._di_container, name, provider)

    @override
    def scoped(
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
        config_input = settings
        if settings is not None:
            base_config: p.Settings = settings.model_copy(deep=True)
        else:
            base_config = self.settings.model_copy(deep=True)
        if subproject and config_input is None:
            base_config = base_config.model_copy(
                update={"app_name": f"{base_config.app_name}.{subproject}"},
            )
        scoped_context: p.Context
        if context is None:
            ctx_instance = self.context
            candidate_context = ctx_instance.clone()
            if u.context(candidate_context):
                scoped_context = candidate_context
            else:
                scoped_context = FlextContext()
        else:
            scoped_context = context
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
        input_factories: Mapping[str, t.FactoryCallable] = factories or {}
        for name, factory in input_factories.items():
            if not u.factory(factory):
                continue
            cloned_factories[name] = m.FactoryRegistration(name=name, factory=factory)
        input_resources: Mapping[str, t.ResourceCallable] = resources or {}
        for name, resource_factory in input_resources.items():
            if u.resource(resource_factory):
                cloned_resources[name] = m.ResourceRegistration(
                    name=name,
                    factory=resource_factory,
                )
        user_overrides_copy = t.ConfigMap(root=dict(self._user_overrides))
        return self.__class__._create_scoped_instance(
            registration=m.ServiceRegistrationSpec.model_validate({
                "settings": base_config,
                "context": scoped_context,
                "services": cloned_services,
                "factories": cloned_factories,
                "resources": cloned_resources,
                "user_overrides": user_overrides_copy,
                "container_config": self._global_config.model_copy(deep=True),
            }),
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
        config_map = t.ConfigMap(
            root={
                str(key): u.normalize_to_container(value)
                for key, value in config_dict_raw.items()
            },
        )
        _ = u.DependencyIntegration.bind_configuration(
            self._di_container,
            config_map,
        )
        user_overrides_plain = dict(self._user_overrides)
        self._user_config_provider.from_dict(user_overrides_plain)
        namespace_registry_raw = getattr(
            self._config.__class__,
            "_namespace_registry",
            None,
        )
        if not namespace_registry_raw or not u.mapping(namespace_registry_raw):
            return
        namespace_registry = namespace_registry_raw
        namespaces: t.StrSequence = list(namespace_registry.keys())
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
                if u.pydantic_model(namespace_settings):
                    return namespace_settings
                error_msg = (
                    f"Namespace settings '{_namespace}' must be a Pydantic model"
                )
                raise TypeError(
                    error_msg,
                )

            if not self.has_service(factory_name):
                self.register(
                    factory_name,
                    namespace_factory,
                    kind=c.ContainerKind.FACTORY,
                )

    @override
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
        return e.fail_not_found("service", name)

    @override
    def wire_modules(
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

    def _get_default_config(self) -> p.Settings:
        """Get default settings instance."""
        return FlextSettings.fetch_global()


__all__: list[str] = ["FlextContainer"]
