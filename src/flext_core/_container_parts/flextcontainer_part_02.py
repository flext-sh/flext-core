"""Dependency injection container for the dispatcher-first CQRS stack.

This module wraps dependency_injector behind a result-bearing API so handlers
and decorators can register/resolve dependencies without importing the
underlying infrastructure. Configuration stays isolated from dispatcher code.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from abc import ABC
from typing import overload, override

from dependency_injector import containers as di_containers

from flext_core import c, e, m, p, r, t, u
from flext_core._loggings_parts.flextlogger_part_05 import FlextLogger

from .flextcontainer_part_01 import (
    FlextContainer as FlextContainerPart01,
)


class FlextContainer(FlextContainerPart01, ABC):
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
        logger: p.Logger = FlextLogger.fetch_logger(
            module_name or c.DEFAULT_LOGGER_MODULE,
        )
        return logger

    def _resolve_callable[T: t.RegisterableService](
        self,
        callable_obj: t.FactoryCallable,
        kind: str,
        type_cls: type[T] | None,
    ) -> p.Result[T] | p.Result[t.RegisterableService]:
        """Invoke a factory/resource callable and narrow to ``type_cls`` if given."""
        try:
            resolved = callable_obj()
        except c.EXC_BROAD_RUNTIME as exc:
            return r[t.RegisterableService].from_result(
                e.fail_operation(f"resolve {kind}", exc),
            )
        if type_cls is not None:
            if isinstance(resolved, type_cls):
                return r[T].ok(resolved)
            return r[T].from_result(
                e.fail_type_mismatch(type_cls.__name__, type(resolved).__name__),
            )
        return r[t.RegisterableService].ok(resolved)

    @overload
    def resolve[T: t.RegisterableService](
        self,
        name: str,
        *,
        type_cls: type[T],
    ) -> p.Result[T]: ...

    @overload
    def resolve(
        self,
        name: str,
        *,
        type_cls: None = None,
    ) -> p.Result[t.RegisterableService]: ...

    @override
    def resolve[T: t.RegisterableService](
        self,
        name: str,
        *,
        type_cls: type[T] | None = None,
    ) -> p.Result[T] | p.Result[t.RegisterableService]:
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
                result: p.Result[T] | p.Result[t.RegisterableService] = r[
                    t.RegisterableService
                ].ok(service)
            elif isinstance(service, type_cls):
                result = r[T].ok(service)
            else:
                result = r[T].from_result(
                    e.fail_type_mismatch(type_cls.__name__, type(service).__name__),
                )
        elif callable_registration is not None:
            kind, callable_obj = callable_registration
            result = self._resolve_callable(callable_obj, kind, type_cls)
        else:
            result = r[t.RegisterableService].from_result(
                e.fail_not_found("service", name),
            )
        return result

    @override
    def snapshot(self) -> m.ConfigMap:
        """Return the merged settings exposed by this container."""
        config_dict = self._global_config.model_dump()
        return m.ConfigMap(
            root={k: u.normalize_to_container(v) for k, v in config_dict.items()},
        )

    @override
    def has(self, name: str) -> bool:
        """Return whether a public service, factory, or resource is registered."""
        return (
            (name in self._services and name not in self._internal_registrations)
            or (name in self._factories and name not in self._internal_registrations)
            or (name in self._resources and name not in self._internal_registrations)
        )

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


__all__: list[str] = ["FlextContainer"]
