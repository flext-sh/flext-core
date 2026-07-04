"""Dependency injection container for the dispatcher-first CQRS stack.

This module wraps dependency_injector behind a result-bearing API so handlers
and decorators can register/resolve dependencies without importing the
underlying infrastructure. Configuration stays isolated from dispatcher code.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Self, override

from flext_core import c, e, m, p, r, t, u

from .flextcontainer_part_03 import (
    FlextContainer as FlextContainerPart03,
)

if TYPE_CHECKING:
    from types import ModuleType


class FlextContainer(FlextContainerPart03, ABC):
    def register_existing_providers(self) -> None:
        """Hydrate the dynamic container with current registrations."""
        cache = self._global_config.enable_factory_caching
        for name, reg in self._services.items():
            if not (
                hasattr(self._di_services, name) or hasattr(self._di_container, name)
            ):
                u.DependencyIntegration.register_object(
                    self._di_services,
                    name,
                    reg.service,
                )
        for name, reg in self._factories.items():
            if not (
                hasattr(self._di_services, name) or hasattr(self._di_container, name)
            ):
                u.DependencyIntegration.register_factory(
                    self._di_services,
                    name,
                    reg.factory,
                    cache=cache,
                )
        for name, reg in self._resources.items():
            if not (
                hasattr(self._di_resources, name) or hasattr(self._di_container, name)
            ):
                u.DependencyIntegration.register_resource(
                    self._di_resources,
                    name,
                    reg.factory,
                )

    @override
    def register_core_services(self) -> None:
        """Auto-register core services for easy DI access."""
        if str(c.Directory.CONFIG) not in self._internal_registrations:
            self.bind(c.Directory.CONFIG, self._config)
            self._internal_registrations.add(str(c.Directory.CONFIG))
        if str(c.ServiceName.LOGGER) not in self._internal_registrations:
            self.factory(
                c.ServiceName.LOGGER,
                lambda: u.fetch_logger(c.DEFAULT_LOGGER_MODULE),
            )
            self._internal_registrations.add(str(c.ServiceName.LOGGER))
        if c.FIELD_CONTEXT not in self._internal_registrations:
            self.bind(c.FIELD_CONTEXT, self._context)
            self._internal_registrations.add(c.FIELD_CONTEXT)
        if str(c.ServiceName.COMMAND_BUS) not in self._internal_registrations:
            self.bind(c.ServiceName.COMMAND_BUS, u.build_dispatcher())
            self._internal_registrations.add(str(c.ServiceName.COMMAND_BUS))

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
        base_config_dump = base_config.model_dump()
        base_app_name = base_config_dump.get("app_name")
        if (
            subproject
            and scope_registration.settings is None
            and isinstance(base_app_name, str)
        ):
            base_config = base_config.clone(app_name=f"{base_app_name}.{subproject}")
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
            ),
        )
        scoped.sync_config_to_di()
        scoped.register_existing_providers()
        scoped.register_core_services()
        return scoped

    def sync_config_to_di(self) -> None:
        """Synchronize FlextSettings to DI providers.Configuration."""
        config_dict = self._global_config.model_dump()
        config_map = m.ConfigMap(
            root={k: u.normalize_to_container(v) for k, v in config_dict.items()},
        )
        _ = u.DependencyIntegration.bind_configuration(self._di_container, config_map)
        namespaces = self._settings_type.registered_namespaces()
        for namespace in namespaces or []:
            factory_name = f"settings.{namespace}"
            settings_class = self._settings_type.resolve_namespace_settings(namespace)
            if settings_class is None:
                continue

            def namespace_factory(
                _namespace: str = namespace,
                _settings_class: t.SettingsClass = settings_class,
            ) -> p.Settings:
                return self._settings_type.fetch_global().fetch_namespace(
                    _namespace,
                    _settings_class,
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
            self._di_container,
            modules=modules,
            packages=packages,
            classes=classes,
        )


__all__: list[str] = ["FlextContainer"]
