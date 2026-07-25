"""Handler registration and discovery utilities.

Provides handler registration with binding, tracking, and batch operations
for the FLEXT dispatcher system.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import override

from flext_core import c, e, m, p, r, t, u

from .flextregistry_part_01 import (
    FlextRegistry as FlextRegistryPart01,
)


class FlextRegistry(FlextRegistryPart01):
    @staticmethod
    def _narrow_value(
        value: (
            t.JsonValue
            | t.RegisterableService
            | t.RegistrablePlugin
            | m.BaseModel
            | None
        ),
    ) -> t.JsonPayload | None:
        """Safe conversion using centralized utilities."""
        narrowed: t.JsonPayload | None = None
        if value is None:
            narrowed = None
        elif isinstance(value, m.BaseModel):
            narrowed = value
        elif isinstance(
            value,
            (p.Logger, p.Settings, p.Context, p.Dispatcher),
        ) or callable(value):
            narrowed = str(value)
        else:
            normalized = u.normalize_to_metadata(value)
            narrowed = t.json_value_adapter().validate_python(normalized)
        return narrowed

    @staticmethod
    def _normalize_registration_impl(
        value: t.RegistrablePlugin,
    ) -> t.RegisterableService:
        """Normalize registry payloads to the container bind contract."""
        if callable(value):

            def normalized_callable(
                *args: p.AttributeProbe,
                **kwargs: p.AttributeProbe,
            ) -> t.JsonPayload | m.BaseModel | None:
                result = value(*args, **kwargs)
                return FlextRegistry._narrow_value(result)

            return normalized_callable
        return FlextRegistry._narrow_value(value)

    def _get_handler_mode(self, value: t.JsonPayload) -> c.HandlerType:
        """Safe conversion to HandlerType (falls back to COMMAND)."""
        text = str(value)
        if text in c.HandlerType.__members__:
            return c.HandlerType[text]
        try:
            return c.HandlerType(text)
        except ValueError:
            return c.HandlerType.COMMAND

    def _get_status(self, value: t.JsonPayload) -> c.Status:
        """Safe conversion to CommonStatus (falls back to ACTIVE)."""
        text = str(value)
        if text in c.Status.__members__:
            return c.Status[text]
        try:
            return c.Status(text)
        except ValueError:
            return c.Status.ACTIVE

    @override
    def execute(self) -> p.Result[bool]:
        """Validate registry is properly initialized.

        Returns:
            r[bool]: Success if dispatcher is configured, failure otherwise.

        """
        dispatcher = self._state.dispatcher
        if dispatcher is None or (not dispatcher):
            return e.fail_operation("execute registry", c.ERR_DISPATCHER_NOT_CONFIGURED)
        return r[bool].ok(True)

    def _remember_registered_key(self, key: str) -> None:
        """Persist one instance-scoped registry key via immutable model state."""
        self._state = self._state.model_copy(
            update={
                "registered_keys": self._state.registered_keys | frozenset({key}),
            },
        )

    def _forget_registered_key(self, key: str) -> None:
        """Remove one instance-scoped registry key via immutable model state."""
        self._state = self._state.model_copy(
            update={
                "registered_keys": frozenset(
                    existing_key
                    for existing_key in self._state.registered_keys
                    if existing_key != key
                ),
            },
        )

    def fetch_plugin(
        self,
        category: str,
        name: str,
        *,
        scope: c.RegistrationScope = c.RegistrationScope.INSTANCE,
    ) -> p.Result[t.JsonPayload | None]:
        """Get a registered plugin by category and name.

        Returns:
            Success with plugin (RegisterableService) or failure if not found.

        """
        key = f"{category}::{name}"
        if scope == c.RegistrationScope.INSTANCE:
            if key not in self._state.registered_keys:
                return e.fail_not_found(category, name)
            raw_result = self.container.resolve(key)
            if raw_result.failure:
                return e.fail_operation(
                    f"retrieve {category} '{name}'",
                    raw_result.error or c.CQRS_OPERATION_FAILED,
                )
            return r[t.JsonPayload | None].ok(self._narrow_value(raw_result.value))
        cls = type(self)
        if key not in cls._class_registered_keys:
            return e.fail_not_found(category, name)
        return r[t.JsonPayload | None].ok(
            self._narrow_value(cls._class_plugin_storage[key]),
        )

    def list_plugins(
        self,
        category: str,
        *,
        scope: c.RegistrationScope = c.RegistrationScope.INSTANCE,
    ) -> p.Result[t.StrSequence]:
        """List all plugins in a category.

        Args:
            category: Plugin category to list

        Returns:
            r[t.StrSequence]: Success with list of plugin names.

        """
        keys = self._state.registered_keys
        if scope == c.RegistrationScope.CLASS:
            keys = self._class_registered_keys
        plugins = [k.split("::")[1] for k in keys if k.startswith(f"{category}::")]
        return r[t.StrSequence].ok(plugins)

    def _add_successful_registration(
        self,
        key: str,
        registration: m.RegistrationDetails,
        summary: m.RegistrySummary,
    ) -> None:
        """Add successful registration to summary."""
        self._remember_registered_key(key)
        summary.registered.append(registration)

    def _finalize_summary(
        self,
        summary: m.RegistrySummary,
    ) -> p.Result[m.RegistrySummary]:
        """Finalize summary based on error state.

        Returns:
            r[m.RegistrySummary]: Success result with summary or failure result with errors.

        """
        if summary.errors:
            return e.fail_operation(
                "finalize registry summary",
                "; ".join(summary.errors),
            )
        return r[m.RegistrySummary].ok(summary)


__all__: list[str] = ["FlextRegistry"]
