"""Context operations and proxies sharing structlog's context-variable storage.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timedelta
from typing import ClassVar

import structlog.contextvars

from flext_core import c, p, t
from flext_core._models import FlextModelsContainers
from flext_core._models._context import (
    FlextModelsContextScope,
    FlextModelsContextTokens,
)
from flext_core._utilities import FlextUtilitiesGenerators


class FlextUtilitiesContext:
    """Context I/O, state transitions and singleton context-variable proxies."""

    class StructlogProxyContextVar[ValueT: t.JsonPayload | datetime]:
        """ContextVar-like operations backed exclusively by structlog."""

        def __init__(self, key: str, default: ValueT | None = None) -> None:
            super().__init__()
            self._key = key
            self._default = default

        @staticmethod
        def reset(token: p.ContextProxyToken) -> None:
            """Restore the value captured before a set operation."""
            if token.previous_value is None:
                structlog.contextvars.unbind_contextvars(token.key)
            else:
                _ = structlog.contextvars.bind_contextvars(**{
                    token.key: token.previous_value
                })

        def get(self) -> t.JsonPayload | datetime | None:
            """Read the same backing value as structlog's context processors.

            Returns:
                The bound value or this proxy's default.

            """
            context: t.MappingKV[str, t.JsonPayload | datetime] = (
                structlog.contextvars.get_contextvars()
            )
            value = context.get(self._key)
            return self._default if value is None else value

        def set(self, value: ValueT | None) -> p.ContextProxyToken:
            """Set or unbind a value and return its validated reset token.

            Returns:
                The token restoring the previous value.

            """
            previous = self.get()
            if value is None:
                structlog.contextvars.unbind_contextvars(self._key)
            else:
                _ = structlog.contextvars.bind_contextvars(**{self._key: value})
            return FlextModelsContextTokens.StructlogProxyToken(
                key=self._key, previous_value=previous
            )

    CORRELATION_ID: ClassVar[p.ContextProxy[str]] = StructlogProxyContextVar[str](
        c.ContextKey.CORRELATION_ID
    )
    PARENT_CORRELATION_ID: ClassVar[p.ContextProxy[str]] = StructlogProxyContextVar[
        str
    ](c.ContextKey.PARENT_CORRELATION_ID)
    SERVICE_NAME: ClassVar[p.ContextProxy[str]] = StructlogProxyContextVar[str](
        c.ContextKey.SERVICE_NAME
    )
    SERVICE_VERSION: ClassVar[p.ContextProxy[str]] = StructlogProxyContextVar[str](
        c.ContextKey.SERVICE_VERSION
    )
    USER_ID: ClassVar[p.ContextProxy[str]] = StructlogProxyContextVar[str](
        c.ContextKey.USER_ID
    )
    REQUEST_ID: ClassVar[p.ContextProxy[str]] = StructlogProxyContextVar[str](
        c.ContextKey.REQUEST_ID
    )
    REQUEST_TIMESTAMP: ClassVar[p.ContextProxy[datetime]] = StructlogProxyContextVar[
        datetime
    ](c.ContextKey.REQUEST_TIMESTAMP)
    OPERATION_NAME: ClassVar[p.ContextProxy[str]] = StructlogProxyContextVar[str](
        c.ContextKey.OPERATION_NAME
    )
    OPERATION_START_TIME: ClassVar[p.ContextProxy[datetime]] = StructlogProxyContextVar[
        datetime
    ](c.ContextKey.OPERATION_START_TIME)
    OPERATION_METADATA: ClassVar[p.ContextProxy[t.JsonDict]] = StructlogProxyContextVar[
        t.JsonDict
    ](c.ContextKey.OPERATION_METADATA)

    @staticmethod
    def create_str_proxy(key: str, default: str | None = None) -> p.ContextProxy[str]:
        """Create a string proxy over the shared structlog storage.

        Returns:
            The string context proxy.

        """
        return FlextUtilitiesContext.StructlogProxyContextVar[str](key, default)

    @staticmethod
    def create_datetime_proxy(
        key: str, default: datetime | None = None
    ) -> p.ContextProxy[datetime]:
        """Create a timestamp proxy over the shared structlog storage.

        Returns:
            The timestamp context proxy.

        """
        return FlextUtilitiesContext.StructlogProxyContextVar[datetime](key, default)

    @staticmethod
    def create_dict_proxy(
        key: str, default: t.JsonDict | None = None
    ) -> p.ContextProxy[t.JsonDict]:
        """Create a mapping proxy over the shared structlog storage.

        Returns:
            The mapping context proxy.

        """
        return FlextUtilitiesContext.StructlogProxyContextVar[t.JsonDict](key, default)

    @staticmethod
    def resolve_correlation_id() -> str | None:
        """Read the active correlation identifier.

        Returns:
            The active identifier, when present.

        """
        value = FlextUtilitiesContext.CORRELATION_ID.get()
        return value if isinstance(value, str) else None

    @staticmethod
    @contextmanager
    def new_correlation(
        correlation_id: str | None = None, parent_id: str | None = None
    ) -> Generator[str]:
        """Scope a correlation identifier and restore both previous identifiers.

        Yields:
            The scoped correlation identifier.

        """
        correlation_id = (
            correlation_id
            if correlation_id is not None
            else FlextUtilitiesGenerators.generate("correlation")
        )
        current = FlextUtilitiesContext.CORRELATION_ID.get()
        token = FlextUtilitiesContext.CORRELATION_ID.set(correlation_id)
        parent_token = (
            FlextUtilitiesContext.PARENT_CORRELATION_ID.set(parent_id)
            if parent_id
            else (
                FlextUtilitiesContext.PARENT_CORRELATION_ID.set(current)
                if isinstance(current, str)
                else None
            )
        )
        try:
            yield correlation_id
        finally:
            FlextUtilitiesContext.CORRELATION_ID.reset(token)
            if parent_token is not None:
                FlextUtilitiesContext.PARENT_CORRELATION_ID.reset(parent_token)

    @staticmethod
    def apply_correlation_id(correlation_id: str | None) -> None:
        """Apply or clear the active correlation identifier."""
        _ = FlextUtilitiesContext.CORRELATION_ID.set(correlation_id)

    @staticmethod
    def ensure_correlation_id() -> str:
        """Return the active identifier or generate and apply one."""
        current = FlextUtilitiesContext.CORRELATION_ID.get()
        if isinstance(current, str) and current:
            return current
        new_id = FlextUtilitiesGenerators.generate("correlation")
        _ = FlextUtilitiesContext.CORRELATION_ID.set(new_id)
        return new_id

    @staticmethod
    @contextmanager
    def service_context(
        service_name: str, version: str | None = None
    ) -> Generator[None]:
        """Scope service identity without losing the enclosing service values."""
        token = FlextUtilitiesContext.SERVICE_NAME.set(service_name)
        version_token = (
            FlextUtilitiesContext.SERVICE_VERSION.set(version) if version else None
        )
        try:
            yield
        finally:
            FlextUtilitiesContext.SERVICE_NAME.reset(token)
            if version_token is not None:
                FlextUtilitiesContext.SERVICE_VERSION.reset(version_token)

    @staticmethod
    def resolve_operation_name() -> str | None:
        """Return the active operation name."""
        value = FlextUtilitiesContext.OPERATION_NAME.get()
        return str(value) if value is not None else None

    @staticmethod
    def apply_operation_name(operation_name: str) -> None:
        """Apply the active operation name."""
        _ = FlextUtilitiesContext.OPERATION_NAME.set(operation_name)

    @staticmethod
    @contextmanager
    def timed_operation(
        operation_name: str | None = None,
    ) -> Generator[FlextModelsContainers.ConfigMap]:
        """Scope operation timing and finalize the yielded metadata on exit.

        Yields:
            The mutable operation metadata.

        """
        start = FlextUtilitiesGenerators.generate_datetime_utc()
        perf = time.perf_counter()
        payload: t.JsonDict = {
            str(c.MetadataKey.START_TIME): start.isoformat(),
            str(c.ContextKey.OPERATION_NAME): operation_name or "",
        }
        metadata = FlextModelsContainers.ConfigMap.model_validate(payload)
        start_token = FlextUtilitiesContext.OPERATION_START_TIME.set(start)
        meta_token = FlextUtilitiesContext.OPERATION_METADATA.set(payload)
        op_token = (
            FlextUtilitiesContext.OPERATION_NAME.set(operation_name)
            if operation_name
            else None
        )
        try:
            yield metadata
        finally:
            duration = time.perf_counter() - perf
            metadata.root.update({
                c.MetadataKey.END_TIME: (
                    start + timedelta(seconds=duration)
                ).isoformat(),
                c.MetadataKey.DURATION_SECONDS: duration,
            })
            FlextUtilitiesContext.OPERATION_START_TIME.reset(start_token)
            FlextUtilitiesContext.OPERATION_METADATA.reset(meta_token)
            if op_token is not None:
                FlextUtilitiesContext.OPERATION_NAME.reset(op_token)

    @staticmethod
    def export_full_context() -> t.MappingKV[str, t.Scalar]:
        """Export active scalar context variables, formatting operation timestamps.

        Returns:
            The active scalar context snapshot.

        """
        exported: t.MutableMappingKV[str, t.Scalar] = {}
        for key, proxy in (
            (c.ContextKey.CORRELATION_ID, FlextUtilitiesContext.CORRELATION_ID),
            (
                c.ContextKey.PARENT_CORRELATION_ID,
                FlextUtilitiesContext.PARENT_CORRELATION_ID,
            ),
            (c.ContextKey.SERVICE_NAME, FlextUtilitiesContext.SERVICE_NAME),
            (c.ContextKey.SERVICE_VERSION, FlextUtilitiesContext.SERVICE_VERSION),
            (c.ContextKey.USER_ID, FlextUtilitiesContext.USER_ID),
            (c.ContextKey.REQUEST_ID, FlextUtilitiesContext.REQUEST_ID),
            (c.ContextKey.OPERATION_NAME, FlextUtilitiesContext.OPERATION_NAME),
        ):
            if (value := proxy.get()) is not None:
                exported[key] = str(value)
        if (value := FlextUtilitiesContext.OPERATION_START_TIME.get()) is not None:
            exported[c.ContextKey.OPERATION_START_TIME] = (
                value.isoformat() if isinstance(value, datetime) else str(value)
            )
        return exported

    @staticmethod
    def clear_context() -> None:
        """Unbind every singleton proxy from this process context."""
        for proxy in (
            FlextUtilitiesContext.CORRELATION_ID,
            FlextUtilitiesContext.PARENT_CORRELATION_ID,
            FlextUtilitiesContext.SERVICE_NAME,
            FlextUtilitiesContext.SERVICE_VERSION,
            FlextUtilitiesContext.USER_ID,
            FlextUtilitiesContext.REQUEST_ID,
            FlextUtilitiesContext.OPERATION_NAME,
        ):
            _ = proxy.set(None)
        _ = FlextUtilitiesContext.OPERATION_START_TIME.set(None)
        _ = FlextUtilitiesContext.OPERATION_METADATA.set(None)
        _ = FlextUtilitiesContext.REQUEST_TIMESTAMP.set(None)

    @staticmethod
    def create_context_runtime(
        metadata: p.ContextMetadata | None = None,
    ) -> p.ContextRuntimeState:
        """Create independent built-in scopes with an immutable metadata snapshot.

        Returns:
            The independent runtime state.

        """
        scope_vars: t.MutableMappingKV[
            str, ContextVar[p.MutableRootDict[t.JsonPayload] | None]
        ] = {
            scope: ContextVar(f"flext_{scope}_context", default=None)
            for scope in (
                c.ContextScope.GLOBAL,
                c.ContextScope.USER,
                c.ContextScope.SESSION,
            )
        }
        values = {"scope_vars": scope_vars}
        if metadata is not None:
            return FlextModelsContextScope.ContextRuntimeState.model_validate({
                **values,
                "metadata": metadata.model_copy(),
            })
        return FlextModelsContextScope.ContextRuntimeState.model_validate(values)

    @staticmethod
    def resolve_context_scope_var(
        state: p.ContextRuntimeState, scope: str
    ) -> tuple[
        p.ContextRuntimeState, ContextVar[p.MutableRootDict[t.JsonPayload] | None]
    ]:
        """Reuse a scope variable or add one without mutating the prior state.

        Returns:
            The resulting state and selected scope variable.

        """
        existing = state.scope_vars.get(scope)
        if existing is not None:
            return state, existing
        scope_var: ContextVar[p.MutableRootDict[t.JsonPayload] | None] = ContextVar(
            f"flext_{scope}_context", default=None
        )
        updated = dict(state.scope_vars)
        updated[scope] = scope_var
        return state.model_copy(update={"scope_vars": updated}), scope_var

    @staticmethod
    def update_context_operation(
        state: p.ContextRuntimeState, operation: str
    ) -> p.ContextRuntimeState:
        """Copy state with the existing counter and operation-metric updates.

        Returns:
            The state with updated operation statistics.

        """
        counters: t.MappingKV[str, int] = {
            "sets": state.statistics.sets,
            "gets": state.statistics.gets,
            "removes": state.statistics.removes,
            "clears": state.statistics.clears,
        }
        updates: t.MutableMappingKV[str, t.JsonPayload] = {}
        counter = f"{operation}s"
        if counter in counters:
            updates[counter] = counters[counter] + 1
        operations: t.JsonDict = {
            key: value
            for key, value in state.statistics.operations.items()
            if isinstance(value, c.PRIMITIVES_TYPES)
        }
        current = operations.get(operation)
        if isinstance(current, int):
            operations[operation] = current + 1
            updates["operations"] = operations
        if not updates:
            return state
        statistics = state.statistics.model_copy(update=updates)
        return state.model_copy(update={"statistics": statistics})


__all__: t.StrSequence = ("FlextUtilitiesContext",)
