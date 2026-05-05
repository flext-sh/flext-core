"""Dependency injection container for the dispatcher-first CQRS stack.

This module wraps dependency_injector behind a result-bearing API so handlers
and decorators can register/resolve dependencies without importing the
underlying infrastructure. Configuration stays isolated from dispatcher code.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import inspect
import sys
import threading
from collections.abc import (
    Callable,
    MutableMapping,
    Sequence,
)
from types import FrameType, ModuleType
from typing import Self, overload, override

from dependency_injector import containers as di_containers

from flext_core import (
    FlextContext,
    FlextRuntime,
    FlextSettings,
    c,
    e,
    m,
    p,
    r,
    t,
    u,
)


class FlextContainer(p.ContainerLifecycle):
    """Singleton DI container wrapping dependency_injector with result-bearing API.

    Services and factories remain local to the container, keeping dispatcher and
    domain code free from infrastructure imports. All operations surface
    ``r`` (Result) so failures are explicit. Thread-safe initialization
    guarantees one global instance for runtime usage while allowing scoped
    containers in tests.
    """

    _global_instance: Self | None = None
    _global_lock: threading.RLock = threading.RLock()
    _context: p.Context
    _config: p.Settings
    _user_overrides: m.ConfigMap
    _di_bridge: di_containers.DeclarativeContainer
    _di_services: di_containers.DynamicContainer
    _di_resources: di_containers.DynamicContainer
    _di_container: di_containers.DynamicContainer
    _services: MutableMapping[str, m.ServiceRegistration]
    _factories: MutableMapping[str, m.FactoryRegistration]
    _resources: MutableMapping[str, m.ResourceRegistration]
    _internal_registrations: set[str]
    _global_config: m.ContainerConfig

    def __new__(cls, *, registration: m.ServiceRegistrationSpec | None = None) -> Self:
        """Create or return the global singleton instance."""
        _ = registration
        if cls._global_instance is None:
            with cls._global_lock:
                if cls._global_instance is None:
                    instance = super().__new__(cls)
                    cls._global_instance = instance
        return cls._global_instance

    def __init__(
        self, *, registration: m.ServiceRegistrationSpec | None = None
    ) -> None:
        """Initialize the singleton container (idempotent)."""
        if hasattr(self, "_di_container"):
            init_registration = registration or m.ServiceRegistrationSpec()
            self._apply_explicit_bootstrap(init_registration)
            self.register_core_services()
            return
        self.initialize_di_components()
        self.initialize_registrations(registration=registration)
        self.sync_config_to_di()
        self.register_existing_providers()
        self.register_core_services()

    @property
    @override
    def settings(self) -> p.Settings:
        """Return configuration bound to this container."""
        return self._config

    @property
    @override
    def context(self) -> p.Context:
        """Return the execution context bound to this container."""
        return self._context

    @property
    @override
    def provide(self) -> Callable[[str], t.RegisterableService]:
        """Return the dependency-injector Provide helper scoped to the bridge."""
        return self._di_bridge.provide

    @classmethod
    def shared(
        cls,
        *,
        settings: p.Settings | None = None,
        context: p.Context | None = None,
        auto_register_factories: bool = False,
    ) -> Self:
        """Return the canonical shared container instance."""
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
    def _resolve_caller_module(frame: FrameType | None) -> ModuleType | None:
        """Resolve the module that called the factory via frame introspection."""
        if not frame or not frame.f_back:
            return None
        module_name = str(frame.f_back.f_globals.get("__name__", "__main__"))
        return sys.modules.get(module_name)

    @staticmethod
    def _auto_register_module_factories(
        instance: p.Container, caller_module: ModuleType
    ) -> None:
        """Scan module for @d.factory() functions and register them."""
        factories = u.scan_module(caller_module)
        module_symbols = vars(caller_module)
        for factory_name, factory_config in factories:
            factory_func = module_symbols.get(factory_name)
            if factory_func is None or not u.factory(factory_func):
                continue

            _ = instance.factory(factory_config.name, factory_func)

    @classmethod
    def reset_for_testing(cls) -> None:
        """Reset singleton instance for testing purposes."""
        with cls._global_lock:
            cls._global_instance = None

    @override
    def clear(self) -> None:
        """Clear all service and factory registrations."""
        names_to_clear = {
            *self._services.keys(),
            *self._factories.keys(),
            *self._resources.keys(),
            str(c.Directory.CONFIG),
            str(c.ServiceName.LOGGER),
            c.FIELD_CONTEXT,
            str(c.ServiceName.COMMAND_BUS),
        }
        for name in names_to_clear:
            for di_ns in (self._di_services, self._di_resources):
                if hasattr(di_ns, name):
                    delattr(di_ns, name)
        self._services.clear()
        self._factories.clear()
        self._resources.clear()
        self._internal_registrations.clear()
        self._config = FlextSettings.fetch_global()
        self.register_core_services()

    @override
    def apply(self, settings: t.UserOverridesMapping | None = None) -> Self:
        """Apply user-provided overrides to container configuration."""
        if settings is None:
            return self
        merged = self._user_overrides.model_copy()
        merged.update({
            k: FlextRuntime.normalize_to_container(v) for k, v in settings.items()
        })
        self._user_overrides = merged
        self._global_config = self._global_config.model_copy(
            update=dict(merged), deep=True
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
        """Create a module logger for the specified runtime scope."""
        _ = service_name, service_version, correlation_id
        return u.fetch_logger(module_name or c.DEFAULT_LOGGER_MODULE)

    def _resolve_callable[T: t.RegisterableService](
        self,
        callable_obj: t.FactoryCallable,
        kind: str,
        type_cls: type[T] | None,
    ) -> p.Result[T] | r[t.RegisterableService]:
        """Invoke a factory/resource callable and narrow to ``type_cls`` if given."""
        try:
            resolved = callable_obj()
        except c.EXC_BROAD_RUNTIME as exc:
            return r[t.RegisterableService].from_result(
                e.fail_operation(f"resolve {kind}", exc)
            )
        if type_cls is not None:
            if isinstance(resolved, type_cls):
                return r[T].ok(resolved)
            return r[T].from_result(
                e.fail_type_mismatch(type_cls.__name__, type(resolved).__name__)
            )
        return r[t.RegisterableService].ok(resolved)

    @overload
    def resolve[T: t.RegisterableService](
        self, name: str, *, type_cls: type[T]
    ) -> p.Result[T]: ...

    @overload
    def resolve(
        self, name: str, *, type_cls: None = None
    ) -> p.Result[t.RegisterableService]: ...

    @override
    def resolve[T: t.RegisterableService](
        self, name: str, *, type_cls: type[T] | None = None
    ) -> p.Result[T] | r[t.RegisterableService]:
        """Resolve a registered service or factory by name."""
        service_registration = self._services.get(name)
        callable_registration = next(
            (
                (kind, registration.factory)
                for kind, registrations in (
                    ("factory", self._factories),
                    ("resource", self._resources),
                )
                if (registration := registrations.get(name)) is not None
            ),
            None,
        )
        if service_registration is not None:
            service = service_registration.service
            if type_cls is None:
                result: p.Result[T] | r[t.RegisterableService] = r[
                    t.RegisterableService
                ].ok(service)
            elif isinstance(service, type_cls):
                result = r[T].ok(service)
            else:
                result = r[T].from_result(
                    e.fail_type_mismatch(type_cls.__name__, type(service).__name__)
                )
        elif callable_registration is not None:
            kind, callable_obj = callable_registration
            result = self._resolve_callable(callable_obj, kind, type_cls)
        else:
            result = r[t.RegisterableService].from_result(
                e.fail_not_found("service", name)
            )
        return result

    @override
    def snapshot(self) -> m.ConfigMap:
        """Return the merged settings exposed by this container."""
        config_dict = self._global_config.model_dump()
        return m.ConfigMap(
            root={k: u.normalize_to_container(v) for k, v in config_dict.items()}
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
        """Initialize DI components (bridge, services, resources, container)."""
        self._di_bridge, self._di_services, self._di_resources = (
            u.DependencyIntegration.create_layered_bridge()
        )
        self._di_container = di_containers.DynamicContainer()
        config_provider = self._di_bridge.settings
        if not hasattr(config_provider, "override"):
            error_msg = "Bridge settings provider missing"
            raise TypeError(error_msg)
        self._di_container.settings = config_provider

    @override
    def initialize_registrations(
        self, *, registration: m.ServiceRegistrationSpec | None = None
    ) -> None:
        """Initialize service registrations and configuration."""
        spec = registration or m.ServiceRegistrationSpec()
        self._services = dict(spec.services or {})
        self._factories = dict(spec.factories or {})
        self._resources = dict(spec.resources or {})
        self._internal_registrations = {
            name
            for name in self._services
            if name
            in {
                str(c.Directory.CONFIG),
                str(c.ServiceName.LOGGER),
                c.FIELD_CONTEXT,
                str(c.ServiceName.COMMAND_BUS),
            }
        }
        self._global_config = spec.container_config or m.ContainerConfig()
        user_overrides_input = spec.user_overrides
        if isinstance(user_overrides_input, m.ConfigMap):
            self._user_overrides = user_overrides_input
        elif user_overrides_input:
            self._user_overrides = m.ConfigMap(
                root={
                    k: list(v)
                    if isinstance(v, Sequence) and not isinstance(v, str | bytes)
                    else v
                    for k, v in user_overrides_input.items()
                }
            )
        else:
            self._user_overrides = m.ConfigMap(root={})
        self._config = spec.settings or FlextSettings.fetch_global()
        context = spec.context
        self._context = context if context is not None else FlextContext.create()

    @override
    def names(self) -> t.StrSequence:
        """List explicitly registered services, factories, and resources."""
        return [
            name
            for name in (
                list(self._services.keys())
                + list(self._factories.keys())
                + list(self._resources.keys())
            )
            if name not in self._internal_registrations
        ]

    @override
    def bind(self, name: str, impl: t.RegisterableService) -> Self:
        """Bind a concrete service instance or value."""
        if not name or self.has(name):
            return self
        self._internal_registrations.discard(name)
        try:
            self._update_registered_object_service(name, impl)
        except c.EXC_ATTR_RUNTIME_TYPE:
            del self._services[name]
        return self

    @override
    def factory(self, name: str, impl: t.FactoryCallable) -> Self:
        """Bind a factory callable."""
        if not name:
            return self
        if self.has(name):
            return self
        self._internal_registrations.discard(name)
        for di_ns in (self._di_services, self._di_resources):
            if hasattr(di_ns, name):
                delattr(di_ns, name)

        def normalized_factory() -> t.RegisterableService:
            raw = impl()
            try:
                _ = u.normalize_registerable_service(raw)
            except ValueError as exc:
                raise ValueError(
                    c.ERR_CONTAINER_FACTORY_INVALID_REGISTERABLE.format(name=name)
                ) from exc
            return raw

        self._factories[name] = m.FactoryRegistration(
            name=name, factory=normalized_factory
        )
        try:
            u.DependencyIntegration.register_factory(
                self._di_services,
                name,
                normalized_factory,
                cache=self._global_config.enable_factory_caching,
            )
            setattr(self._di_bridge, name, getattr(self._di_services, name))
        except c.EXC_ATTR_RUNTIME_TYPE:
            del self._factories[name]
        return self

    @override
    def resource(self, name: str, impl: t.ResourceCallable) -> Self:
        """Bind a lifecycle-managed resource factory."""
        if not name:
            return self
        if self.has(name):
            return self
        self._internal_registrations.discard(name)
        for di_ns in (self._di_services, self._di_resources):
            if hasattr(di_ns, name):
                delattr(di_ns, name)
        self._resources[name] = m.ResourceRegistration(name=name, factory=impl)
        try:
            u.DependencyIntegration.register_resource(self._di_resources, name, impl)
            setattr(self._di_bridge, name, getattr(self._di_resources, name))
        except c.EXC_ATTR_RUNTIME_TYPE:
            del self._resources[name]
        return self

    @override
    def register_core_services(self) -> None:
        """Auto-register core services for easy DI access."""
        if str(c.Directory.CONFIG) not in self._internal_registrations:
            self.bind(c.Directory.CONFIG, self._config)
            self._internal_registrations.add(str(c.Directory.CONFIG))
        if str(c.ServiceName.LOGGER) not in self._internal_registrations:
            self.factory(
                c.ServiceName.LOGGER, lambda: u.fetch_logger(c.DEFAULT_LOGGER_MODULE)
            )
            self._internal_registrations.add(str(c.ServiceName.LOGGER))
        if c.FIELD_CONTEXT not in self._internal_registrations:
            self.bind(c.FIELD_CONTEXT, self._context)
            self._internal_registrations.add(c.FIELD_CONTEXT)
        if str(c.ServiceName.COMMAND_BUS) not in self._internal_registrations:
            self.bind(c.ServiceName.COMMAND_BUS, u.build_dispatcher())
            self._internal_registrations.add(str(c.ServiceName.COMMAND_BUS))

    @override
    def register_existing_providers(self) -> None:
        """Hydrate the dynamic container with current registrations."""
        cache = self._global_config.enable_factory_caching
        for name, reg in self._services.items():
            if not (
                hasattr(self._di_services, name) or hasattr(self._di_container, name)
            ):
                u.DependencyIntegration.register_object(
                    self._di_services, name, reg.service
                )
        for name, reg in self._factories.items():
            if not (
                hasattr(self._di_services, name) or hasattr(self._di_container, name)
            ):
                u.DependencyIntegration.register_factory(
                    self._di_services, name, reg.factory, cache=cache
                )
        for name, reg in self._resources.items():
            if not (
                hasattr(self._di_resources, name) or hasattr(self._di_container, name)
            ):
                u.DependencyIntegration.register_resource(
                    self._di_resources, name, reg.factory
                )

    @override
    def scope(
        self,
        *,
        subproject: str | None = None,
        registration: m.ServiceRegistrationSpec | None = None,
    ) -> Self:
        """Create an isolated container scope with optional overrides."""
        scope_registration = registration or m.ServiceRegistrationSpec()
        settings_source = (
            scope_registration.settings
            if scope_registration.settings is not None
            else self._config
        )
        base_config: p.Settings = settings_source.clone()
        if subproject and scope_registration.settings is None:
            base_config = base_config.clone(
                app_name=f"{base_config.app_name}.{subproject}"
            )
        scoped_context = (
            self.context.clone()
            if scope_registration.context is None
            else scope_registration.context
        )
        if subproject:
            _ = scoped_context.set("subproject", subproject)
        cloned_services = {
            name: reg.model_copy(deep=False) for name, reg in self._services.items()
        }
        cloned_factories = {
            name: reg.model_copy(deep=False) for name, reg in self._factories.items()
        }
        cloned_resources = {
            name: reg.model_copy(deep=False) for name, reg in self._resources.items()
        }
        cloned_services.update(scope_registration.services or {})
        cloned_factories.update(scope_registration.factories or {})
        cloned_resources.update(scope_registration.resources or {})
        scoped = u.create_instance(self.__class__)
        scoped.initialize_di_components()
        scoped.initialize_registrations(
            registration=m.ServiceRegistrationSpec(
                settings=base_config,
                context=scoped_context,
                services=cloned_services,
                factories=cloned_factories,
                resources=cloned_resources,
                user_overrides=self._user_overrides.model_copy(),
                container_config=self._global_config.model_copy(deep=True),
            )
        )
        scoped.sync_config_to_di()
        scoped.register_existing_providers()
        scoped.register_core_services()
        return scoped

    @override
    def sync_config_to_di(self) -> None:
        """Synchronize FlextSettings to DI providers.Configuration."""
        config_dict = self._global_config.model_dump()
        config_map = m.ConfigMap(
            root={k: u.normalize_to_container(v) for k, v in config_dict.items()}
        )
        _ = u.DependencyIntegration.bind_configuration(self._di_container, config_map)
        namespaces = FlextSettings.registered_namespaces()
        for namespace in namespaces or []:
            factory_name = f"settings.{namespace}"
            settings_class = FlextSettings.resolve_namespace_settings(namespace)
            if settings_class is None:
                continue

            def namespace_factory(
                _namespace: str = namespace,
                _settings_class: t.SettingsClass = settings_class,
            ) -> p.Settings:
                return FlextSettings.fetch_global().fetch_namespace(
                    _namespace, _settings_class
                )

            if factory_name not in self._factories:
                self.factory(factory_name, namespace_factory)
                self._internal_registrations.add(factory_name)

    @override
    def drop(self, name: str) -> p.Result[bool]:
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
        for di_ns in (self._di_services, self._di_resources):
            if hasattr(di_ns, name):
                delattr(di_ns, name)
        if removed:
            return r[bool].ok(True)
        return r[bool].from_result(e.fail_not_found("service", name))

    @override
    def wire(
        self,
        *,
        modules: t.SequenceOf[ModuleType] | None = None,
        packages: t.StrSequence | None = None,
        classes: t.SequenceOf[type] | None = None,
    ) -> None:
        """Wire modules/packages to the DI bridge for @inject/Provide usage."""
        u.DependencyIntegration.wire(
            self._di_container, modules=modules, packages=packages, classes=classes
        )

    @override
    def dispatcher(self) -> p.Result[p.Dispatcher]:
        """Resolve the canonical dispatcher / command bus."""
        result = self.resolve(c.ServiceName.COMMAND_BUS)
        if result.failure:
            return r[p.Dispatcher].from_result(
                e.fail_not_found("dispatcher", c.ServiceName.COMMAND_BUS)
            )
        if isinstance(result.value, p.Dispatcher):
            return r[p.Dispatcher].ok(result.value)
        return r[p.Dispatcher].from_result(
            e.fail_type_mismatch("dispatcher", u.type_name(result.value))
        )

    def _apply_explicit_bootstrap(
        self, registration: m.ServiceRegistrationSpec
    ) -> None:
        """Apply explicit bootstrap overrides to an existing singleton instance."""
        if registration.settings is not None:
            self._config = registration.settings
            self._update_registered_object_service(
                c.Directory.CONFIG, registration.settings
            )
            self.sync_config_to_di()
        if registration.context is not None:
            self._context = registration.context
            self._update_registered_object_service(
                c.FIELD_CONTEXT, registration.context
            )

    def _update_registered_object_service(
        self, name: str, service: t.RegisterableService
    ) -> None:
        """Replace or insert an object-backed service across local and DI state."""
        self._services[name] = m.ServiceRegistration(
            name=name,
            service=service,
            service_type=u.type_name(service),
        )
        for di_ns in (self._di_services, self._di_resources):
            if hasattr(di_ns, name):
                delattr(di_ns, name)
        u.DependencyIntegration.register_object(self._di_services, name, service)
        setattr(self._di_bridge, name, getattr(self._di_services, name))


__all__: t.MutableSequenceOf[str] = ["FlextContainer"]
