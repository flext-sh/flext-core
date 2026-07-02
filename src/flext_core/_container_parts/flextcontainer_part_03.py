"""Dependency injection container for the dispatcher-first CQRS stack.

This module wraps dependency_injector behind a result-bearing API so handlers
and decorators can register/resolve dependencies without importing the
underlying infrastructure. Configuration stays isolated from dispatcher code.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from abc import ABC
from collections.abc import (
    Sequence,
)
from typing import Self, override

from flext_core.constants import c
from flext_core.models import m
from flext_core.typings import t
from flext_core.utilities import u

from .flextcontainer_part_02 import (
    FlextContainer as FlextContainerPart02,
)


class FlextContainer(FlextContainerPart02, ABC):
    def initialize_registrations(
        self, *, registration: m.ServiceRegistrationSpec | None = None,
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
                },
            )
        else:
            self._user_overrides = m.ConfigMap(root={})
        self._config = (
            spec.settings.clone()
            if spec.settings is not None
            else self._settings_type.fetch_global()
        )
        context = spec.context
        self._context = context if context is not None else self._context_type.create()

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
                    c.ERR_CONTAINER_FACTORY_INVALID_REGISTERABLE.format(name=name),
                ) from exc
            return raw

        self._factories[name] = m.FactoryRegistration(
            name=name, factory=normalized_factory,
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

    def _update_registered_object_service(
        self, name: str, service: t.RegisterableService,
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


__all__: list[str] = ["FlextContainer"]
