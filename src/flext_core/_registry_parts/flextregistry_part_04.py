"""Handler registration and discovery utilities.

Provides handler registration with binding, tracking, and batch operations
for the FLEXT dispatcher system.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core import c, e, m, p, r, t

from .flextregistry_part_03 import (
    FlextRegistry as FlextRegistryPart03,
)

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
    )


class FlextRegistry(FlextRegistryPart03):
    def register_plugin(
        self,
        category: str,
        name: str,
        plugin: t.RegistrablePlugin,
        *,
        validate: Callable[[t.RegistrablePlugin], r[bool]] | None = None,
        scope: c.RegistrationScope = c.RegistrationScope.INSTANCE,
    ) -> p.Result[bool]:
        """Register a plugin with optional validation.

        Args:
            category: Plugin category (e.g., "protocols", "validators")
            name: Plugin name within the category
            plugin: Plugin instance to register
            validate: Optional validation callable returning r[bool]
            scope: Registration scope ("instance" or "class")

        Returns:
            r[bool]: Success if registered, failure with error details.

        """
        result: p.Result[bool] = r[bool].ok(True)
        if not name:
            params = m.RegistryPluginParams(
                category=category,
                name=name,
                scope=scope,
            )
            result = e.fail_validation(
                m.ValidationErrorParams(field="name", value=name),
                error=e.render_template(
                    c.ERR_REGISTRY_CATEGORY_NAME_CANNOT_BE_EMPTY,
                    category=category,
                    params=params,
                ),
            )
        if result.success and validate:
            try:
                validation_result = validate(plugin)
                if validation_result.failure:
                    result = e.fail_operation(
                        "validate plugin registration",
                        validation_result.error or c.ERR_VALIDATION_FAILED,
                    )
            except c.EXC_RUNTIME_TYPE as exc:
                result = e.fail_operation("validate plugin registration", exc)
        if result.success:
            key = f"{category}::{name}"
            if scope == c.RegistrationScope.INSTANCE:
                if key not in self._state.registered_keys:
                    normalized_plugin = self._normalize_registration_impl(plugin)
                    self.container.bind(key, normalized_plugin)
                    self._remember_registered_key(key)
            else:
                cls = type(self)
                if key not in cls._class_registered_keys:
                    cls._class_plugin_storage[key] = plugin
                    cls._class_registered_keys.add(key)
        return result

    def unregister_plugin(
        self,
        category: str,
        name: str,
        *,
        scope: c.RegistrationScope = c.RegistrationScope.INSTANCE,
    ) -> p.Result[bool]:
        """Unregister a plugin.

        Args:
            category: Plugin category
            name: Plugin name

        Returns:
            r[bool]: Success if unregistered, failure if not found.

        """
        key = f"{category}::{name}"
        if scope == c.RegistrationScope.INSTANCE:
            if key not in self._state.registered_keys:
                return e.fail_not_found(category, name)
            self._forget_registered_key(key)
            return r[bool].ok(True)
        cls = type(self)
        if key not in cls._class_registered_keys:
            return e.fail_not_found(category, name)
        del cls._class_plugin_storage[key]
        cls._class_registered_keys.discard(key)
        return r[bool].ok(True)


__all__: list[str] = ["FlextRegistry"]
