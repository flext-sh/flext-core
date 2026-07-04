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
from typing import TYPE_CHECKING, Self, override

from flext_core import c, e, m, p, r, t, u

from .flextcontainer_part_04 import (
    FlextContainer as FlextContainerPart04,
)

if TYPE_CHECKING:
    from types import FrameType, ModuleType


class FlextContainer(FlextContainerPart04):
    @override
    def dispatcher(self) -> p.Result[p.Dispatcher]:
        """Resolve the canonical dispatcher / command bus."""
        result = self.resolve(c.ServiceName.COMMAND_BUS)
        if result.failure:
            return r[p.Dispatcher].from_result(
                e.fail_not_found("dispatcher", c.ServiceName.COMMAND_BUS),
            )
        if isinstance(result.value, p.Dispatcher):
            return r[p.Dispatcher].ok(result.value)
        return r[p.Dispatcher].from_result(
            e.fail_type_mismatch("dispatcher", u.type_name(result.value)),
        )

    def __init__(
        self,
        *,
        registration: m.ServiceRegistrationSpec | None = None,
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
                m.ServiceRegistrationSpec(settings=settings, context=context),
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
        instance: p.Container,
        caller_module: ModuleType,
    ) -> None:
        """Scan module for @d.factory() functions and register them."""
        factories = u.scan_module(caller_module)
        module_symbols = vars(caller_module)
        for factory_name, factory_config in factories:
            factory_func = module_symbols.get(factory_name)
            if factory_func is None or not u.factory(factory_func):
                continue

            _ = instance.factory(factory_config.name, factory_func)

    def _apply_explicit_bootstrap(
        self,
        registration: m.ServiceRegistrationSpec,
    ) -> None:
        """Apply explicit bootstrap overrides to an existing singleton instance."""
        if registration.settings is not None:
            self._config = registration.settings
            self._update_registered_object_service(
                c.Directory.CONFIG,
                registration.settings,
            )
            self.sync_config_to_di()
        if registration.context is not None:
            self._context = registration.context
            self._update_registered_object_service(
                c.FIELD_CONTEXT,
                registration.context,
            )

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
        self._config = self._settings_type.fetch_global()
        self.register_core_services()

    @override
    def apply(self, settings: t.UserOverridesMapping | None = None) -> Self:
        """Apply user-provided overrides to container configuration."""
        if settings is None:
            return self
        merged = self._user_overrides.model_copy()
        merged.update({k: u.normalize_to_container(v) for k, v in settings.items()})
        self._user_overrides = merged
        self._global_config = self._global_config.model_copy(
            update=dict(merged),
            deep=True,
        )
        self.sync_config_to_di()
        return self


__all__: list[str] = ["FlextContainer"]
