"""Logging context binding and value normalization.

MRO part 01: native contextvars binding and scope-key ownership.

Extracted from FlextUtilitiesLogging as an MRO mixin to keep the facade under
the 200-line cap (AGENTS.md §3.1).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from contextvars import ContextVar
from types import MappingProxyType
from typing import ClassVar

from flext_core import FlextExceptions, FlextResult, c, p, t
from flext_core._utilities import FlextUtilitiesLoggingConfig


class FlextUtilitiesLoggingContextPart01(FlextUtilitiesLoggingConfig):
    """Native context-local bindings; scopes track key ownership, never values."""

    _scope_keys: ClassVar[ContextVar[Mapping[str, frozenset[str]]]] = ContextVar(
        "flext_logging_scope_keys",
        default=MappingProxyType({}),
    )

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
            return FlextResult[bool].ok(value=True)
        except c.CONTEXT_EXCEPTIONS as exc:
            return FlextExceptions.fail_operation(
                f"{c.LoggingOperation.BIND_CONTEXT} '{scope}'", exc
            )

    @classmethod
    def bind_global_context(cls, **context: t.JsonPayload) -> p.Result[bool]:
        """Bind values in the current native logging context.

        Returns:
            Binding success or the context failure.

        """
        try:
            cls.structlog().contextvars.bind_contextvars(**context)
            return FlextResult[bool].ok(value=True)
        except c.CONTEXT_EXCEPTIONS as exc:
            return FlextExceptions.fail_operation(c.LoggingOperation.BIND_GLOBAL, exc)

    @classmethod
    def clear_global_context(cls) -> p.Result[bool]:
        """Clear this context's native values and scoped key ownership.

        Returns:
            Clearing success or the context failure.

        """
        try:
            cls.structlog().contextvars.clear_contextvars()
            cls._scope_keys.set({})
            return FlextResult[bool].ok(value=True)
        except c.CONTEXT_EXCEPTIONS as exc:
            return FlextExceptions.fail_operation(c.LoggingOperation.CLEAR_GLOBAL, exc)

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
            return FlextResult[bool].ok(value=True)
        except c.CONTEXT_EXCEPTIONS as exc:
            return FlextExceptions.fail_operation(
                f"{c.LoggingOperation.CLEAR_SCOPE} '{scope}'", exc
            )

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
            return FlextResult[bool].ok(value=True)
        except c.CONTEXT_EXCEPTIONS as exc:
            return FlextExceptions.fail_operation(c.LoggingOperation.UNBIND_GLOBAL, exc)


__all__: list[str] = ["FlextUtilitiesLoggingContextPart01"]
