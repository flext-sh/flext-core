"""Facade for FlextUtilitiesLoggingContext.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from types import MappingProxyType
from typing import ClassVar

from typing_extensions import TypeForm

from flext_core import c, e, m, p, r, t
from flext_core._utilities import (
    FlextUtilitiesBase,
    FlextUtilitiesLoggingConfig,
    FlextUtilitiesPydantic,
)


class FlextUtilitiesLoggingContext(FlextUtilitiesLoggingConfig):
    """Compose logging context through its internal part hierarchy."""

    _scope_keys: ClassVar[ContextVar[Mapping[str, frozenset[str]]]] = ContextVar(
        "flext_logging_scope_keys",
        default=MappingProxyType({}),
    )

    @classmethod
    @contextmanager
    def logging_scope(cls, scope: str, **context: t.JsonPayload) -> Generator[None]:
        """Restore previous correlation values and ownership even on failure.

        Yields:
            Control within the operation's native logging context.

        """
        scopes = cls._scope_keys.get()
        token = cls._scope_keys.set({
            **scopes,
            scope: scopes.get(scope, frozenset()) | context.keys(),
        })
        try:
            with cls.structlog().contextvars.bound_contextvars(**context):
                yield
        finally:
            cls._scope_keys.reset(token)

    @classmethod
    def bind_context(cls, scope: str, **context: t.JsonPayload) -> p.Result[bool]:
        """Bind native context and record this context's scoped key ownership.

        Returns:
            Binding success or the context failure.

        """
        try:
            cls.structlog().contextvars.bind_contextvars(**context)
            scopes = cls._scope_keys.get()
            cls._scope_keys.set({
                **scopes,
                scope: scopes.get(scope, frozenset()) | context.keys(),
            })
            return r[bool].ok(value=True)
        except c.CONTEXT_EXCEPTIONS as exc:
            return e.fail_operation(f"{c.LoggingOperation.BIND_CONTEXT} '{scope}'", exc)

    @classmethod
    def bind_global_context(cls, **context: t.JsonPayload) -> p.Result[bool]:
        """Bind values in the current native logging context.

        Returns:
            Binding success or the context failure.

        """
        try:
            cls.structlog().contextvars.bind_contextvars(**context)
            return r[bool].ok(value=True)
        except c.CONTEXT_EXCEPTIONS as exc:
            return e.fail_operation(c.LoggingOperation.BIND_GLOBAL, exc)

    @classmethod
    def clear_global_context(cls) -> p.Result[bool]:
        """Clear this context's native values and scoped key ownership.

        Returns:
            Clearing success or the context failure.

        """
        try:
            cls.structlog().contextvars.clear_contextvars()
            cls._scope_keys.set({})
            return r[bool].ok(value=True)
        except c.CONTEXT_EXCEPTIONS as exc:
            return e.fail_operation(c.LoggingOperation.CLEAR_GLOBAL, exc)

    @classmethod
    def clear_scope(cls, scope: str) -> p.Result[bool]:
        """Unbind the keys owned by this scope in the current context.

        Returns:
            Unbinding success or the context failure.

        """
        try:
            scopes = dict(cls._scope_keys.get())
            cls.structlog().contextvars.unbind_contextvars(*scopes.pop(scope, ()))
            cls._scope_keys.set(scopes)
            return r[bool].ok(value=True)
        except c.CONTEXT_EXCEPTIONS as exc:
            return e.fail_operation(f"{c.LoggingOperation.CLEAR_SCOPE} '{scope}'", exc)

    @classmethod
    def unbind_global_context(cls, *keys: str) -> p.Result[bool]:
        """Unbind native keys and retire their scoped ownership.

        Returns:
            Unbinding success or the context failure.

        """
        try:
            cls.structlog().contextvars.unbind_contextvars(*keys)
            cls._scope_keys.set({
                scope: owned.difference(keys)
                for scope, owned in cls._scope_keys.get().items()
            })
            return r[bool].ok(value=True)
        except c.CONTEXT_EXCEPTIONS as exc:
            return e.fail_operation(c.LoggingOperation.UNBIND_GLOBAL, exc)

    @staticmethod
    def to_container_value(
        value: p.LogValue | t.JsonValue,
    ) -> t.JsonValue:
        """Admit ordinary fields before canonical JSON normalization.

        Native exception triples belong to reserved ``exc_info`` and bypass
        this boundary in the bound logger's native emission path.

        Returns:
            The JSON-compatible logging value.

        """
        if isinstance(value, BaseException):
            return str(value)
        if isinstance(value, tuple):
            admitted = FlextUtilitiesBase.type_adapter(
                TypeForm(t.JsonPayload),
                config=m.ArbitraryTypesModel.model_config,
            ).validate_python(list(value))
            return FlextUtilitiesPydantic.normalize_to_json_value(admitted)
        return FlextUtilitiesPydantic.normalize_to_json_value(value)

    @staticmethod
    def to_container_context(
        context: Mapping[str, p.LogValue | t.JsonValue],
    ) -> t.JsonMapping:
        """Serialize owned fields once before entering the native processor chain.

        Returns:
            The JSON-compatible context mapping.

        """
        return {
            key: FlextUtilitiesLoggingContext.to_container_value(value)
            for key, value in context.items()
        }


__all__: list[str] = ["FlextUtilitiesLoggingContext"]
